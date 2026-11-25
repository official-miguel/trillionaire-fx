"""Risk engine: sizing, hard caps, drawdown breaker, cost gate, health monitor.

Nothing here can increase size after losses, widen a stop, average down or add to
a losing position. Hard-rule violations BLOCK a signal; they are never "scored".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Any

import numpy as np
from scipy import stats as sps

from tfx.config import RISK_HARD_CAP_PCT, SymbolConfig

ROLLOVER_UTC_WINDOWS = ((time(20, 30), time(22, 30)),)  # around 17:00 New York (both DST)


# ------------------------------------------------------------- FX conversion
class FxConverter:
    """Convert amounts between currencies using the latest closes of known pairs."""

    def __init__(self, rates: dict[str, float]) -> None:
        # rates: {"EURUSD": 1.09, "USDJPY": 155.2, ...} (base+quote -> price)
        self.rates = {k.upper(): float(v) for k, v in rates.items() if v and v > 0}

    def _direct(self, frm: str, to: str) -> float | None:
        if frm == to:
            return 1.0
        if frm + to in self.rates:
            return self.rates[frm + to]
        if to + frm in self.rates:
            return 1 / self.rates[to + frm]
        return None

    def rate(self, frm: str, to: str) -> float:
        """Direct rate, or one hop through USD or EUR. Unknown currencies raise KeyError."""
        frm, to = frm.upper(), to.upper()
        d = self._direct(frm, to)
        if d is not None:
            return d
        for via in ("USD", "EUR"):
            a, b = self._direct(frm, via), self._direct(via, to)
            if a is not None and b is not None:
                return a * b
        raise KeyError(f"no FX rate to convert {frm} -> {to}")

    def convert(self, amount: float, frm: str, to: str) -> float:
        return amount * self.rate(frm, to)


@dataclass(frozen=True)
class PositionSize:
    units: float
    lots: float
    amount_at_risk: float
    account_ccy: str
    stop_distance: float
    stop_distance_pips: float
    risk_pct: float


def position_size(balance: float, risk_pct: float, entry: float, stop: float,
                  sym: SymbolConfig, account_ccy: str, fx: FxConverter,
                  size_multiplier: float = 1.0) -> PositionSize:
    """Units so that hitting ``stop`` loses ``balance x risk% x size_multiplier``."""
    if risk_pct > RISK_HARD_CAP_PCT:
        raise ValueError(f"risk {risk_pct}% exceeds the {RISK_HARD_CAP_PCT}% hard cap")
    if not (math.isfinite(stop) and stop > 0) or stop == entry:
        raise ValueError("a valid stop different from the entry is required")
    dist = abs(entry - stop)
    risk_cash = balance * risk_pct / 100 * max(0.0, min(size_multiplier, 1.0))
    loss_per_unit = fx.convert(dist, sym.quote, account_ccy)
    units = risk_cash / loss_per_unit
    lot_size = 100 if sym.base == "XAU" else 100_000
    return PositionSize(units, units / lot_size, risk_cash, account_ccy, dist, dist / sym.pip,
                        risk_pct * size_multiplier)


# ---------------------------------------------------------------- hard caps
@dataclass
class OpenPosition:
    symbol: str
    strategy: str
    direction: int
    risk_pct: float          # % of balance at risk now (0 once the stop is at breakeven+)


@dataclass
class RiskContext:
    risk_pct: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    equity_peak: float = 0.0
    equity_now: float = 0.0
    breaker_reviewed: bool = False
    data_confirmed: bool = True
    pair_validated: bool = True
    strategy_health: str = "OK"
    kelly_fraction: float | None = None        # f* from validated stats (None = unknown)
    spread_now: float | None = None            # in pips
    typical_spread: float | None = None
    now: datetime | None = None


@dataclass(frozen=True)
class Limits:
    hard_cap_pct: float = RISK_HARD_CAP_PCT
    kelly_cap: float = 0.25
    max_open_risk_pct: float = 3.0
    max_per_currency: int = 2
    drawdown_breaker_pct: float = 15.0
    max_spread_multiple: float = 3.0

    @classmethod
    def from_config(cls, risk: dict[str, Any], strategies: dict[str, Any],
                    costs: dict[str, Any]) -> Limits:
        return cls(kelly_cap=float(risk.get("kelly_fraction_cap", 0.25)),
                   max_open_risk_pct=float(risk.get("max_open_risk_pct", 3.0)),
                   max_per_currency=int(strategies.get("filters", {}).get(
                       "max_trades_per_currency", 2)),
                   drawdown_breaker_pct=float(risk.get("drawdown_breaker_pct", 15.0)),
                   max_spread_multiple=float(costs.get("max_spread_multiple", 3.0)))


def drawdown_pct(peak: float, now: float) -> float:
    return 0.0 if peak <= 0 else max(0.0, (peak - now) / peak * 100)


def kelly_fraction(win_rate: float, payoff: float) -> float:
    """Full-Kelly fraction f* = W - (1 - W) / payoff (can be <= 0: no edge)."""
    if payoff <= 0 or not math.isfinite(payoff):
        return 0.0 if payoff <= 0 else win_rate
    return win_rate - (1 - win_rate) / payoff


def in_rollover_window(now: datetime) -> bool:
    t = now.astimezone(UTC).time()
    return any(a <= t <= b for a, b in ROLLOVER_UTC_WINDOWS)


def hard_rule_violations(symbol: str, stop: float | None, ctx: RiskContext,
                         limits: Limits, new_risk_pct: float | None = None) -> list[str]:
    """Every reason this signal must be BLOCKED. Empty list = allowed."""
    v: list[str] = []
    risk = ctx.risk_pct if new_risk_pct is None else new_risk_pct
    if stop is None or not math.isfinite(stop):
        v.append("No stop defined: an entry without a stop is never allowed.")
    if not ctx.data_confirmed:
        v.append("Data not confirmed by two sources for the latest bar.")
    if not ctx.pair_validated:
        v.append("This strategy/timeframe has not passed validation (locked).")
    if ctx.strategy_health in ("DISABLED", "PAUSED"):
        v.append(f"Strategy {ctx.strategy_health.lower()} by the health monitor.")
    if risk > limits.hard_cap_pct:
        v.append(f"Risk {risk}% is above the {limits.hard_cap_pct}% hard cap.")
    if ctx.kelly_fraction is not None:
        kcap = max(0.0, ctx.kelly_fraction) * limits.kelly_cap * 100
        if risk > kcap:
            v.append(f"Risk {risk}% is above the fractional-Kelly cap ({kcap:.2f}%).")
    dd = drawdown_pct(ctx.equity_peak, ctx.equity_now)
    if dd > limits.drawdown_breaker_pct and not ctx.breaker_reviewed:
        v.append(f"PAUSED - REVIEW: drawdown {dd:.1f}% > {limits.drawdown_breaker_pct}%. "
                 "Complete the written review before new signals.")
    open_risk = sum(p.risk_pct for p in ctx.open_positions)
    if open_risk + risk > limits.max_open_risk_pct + 1e-9:
        v.append(f"Open risk would be {open_risk + risk:.2f}% (cap "
                 f"{limits.max_open_risk_pct}%).")
    for ccy in (symbol[:3], symbol[3:]):
        n = sum(1 for p in ctx.open_positions if ccy in (p.symbol[:3], p.symbol[3:]))
        if n >= limits.max_per_currency:
            v.append(f"Already {n} open trades exposed to {ccy} "
                     f"(cap {limits.max_per_currency}).")
    if (ctx.spread_now is not None and ctx.typical_spread
            and ctx.spread_now > limits.max_spread_multiple * ctx.typical_spread):
        v.append(f"Spread {ctx.spread_now:.1f} pips is more than "
                 f"{limits.max_spread_multiple:g}x typical: wait for normal spreads.")
    if ctx.now is not None and in_rollover_window(ctx.now):
        v.append("Within 30 minutes of the daily rollover: wait until it has passed.")
    return v


# -------------------------------------------------------- volatility target
def vol_target_scale(daily_returns: np.ndarray, target_annual_pct: float = 10.0) -> float:
    """Scale (<= 1) so that realised portfolio volatility does not exceed the target.

    Never above 1: volatility targeting may only REDUCE size.
    """
    r = np.asarray(daily_returns, float)
    if len(r) < 20 or r.std() == 0:
        return 1.0
    realised = r.std(ddof=1) * math.sqrt(252) * 100
    return float(min(1.0, target_annual_pct / realised))


# ------------------------------------------------------------ health monitor
@dataclass(frozen=True)
class Health:
    status: str        # OK | REDUCE | PAUSED | DISABLED
    size_factor: float
    reasons: list[str]


def health_check(live_r: list[float], live_daily_returns: list[float],
                 backtest_r: list[float], backtest_max_dd: float) -> Health:
    reasons: list[str] = []
    status, factor = "OK", 1.0
    lr: Any = np.asarray(live_daily_returns[-252:], float)
    if len(lr) >= 120 and lr.std() > 0 and lr.mean() / lr.std() * math.sqrt(252) < 0:
        reasons.append("Rolling 12-month live Sharpe is below 0.")
        status, factor = "DISABLED", 0.0
    if len(live_daily_returns) > 1:
        eq: Any = np.cumprod(1 + np.asarray(live_daily_returns, float))
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        if backtest_max_dd < 0 and dd < 1.5 * backtest_max_dd:
            reasons.append(f"Live drawdown {dd:.1%} is worse than 1.5x the backtest maximum "
                           f"({backtest_max_dd:.1%}).")
            status, factor = "DISABLED", 0.0
    recent: Any = np.asarray(live_r[-30:], float)
    bt: Any = np.asarray(backtest_r, float)
    if status == "OK" and len(recent) >= 30 and len(bt) >= 30 and bt.std() > 0:
        z = (recent.mean() - bt.mean()) / (bt.std(ddof=1) / math.sqrt(len(recent)))
        ks_p = float(sps.ks_2samp(recent, bt).pvalue)
        if z < -3 or ks_p < 0.01:
            reasons.append(f"Severe drift vs backtest (z = {z:.1f}, KS p = {ks_p:.3f}).")
            status, factor = "PAUSED", 0.0
        elif z < -2 or ks_p < 0.05:
            reasons.append(f"Drift vs backtest (z = {z:.1f}, KS p = {ks_p:.3f}): size halved.")
            status, factor = "REDUCE", 0.5
    return Health(status, factor, reasons)
