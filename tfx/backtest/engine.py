"""Event-driven daily backtester with realistic costs and the full exit toolkit.

Rules that make the results honest
----------------------------------
* Signals come from bar t's CLOSE; orders fill at bar t+1's OPEN (never earlier).
* Costs per side: half the spread + slippage; commission per round turn; a nightly
  carry/swap drag. ``cost_multiplier`` scales all of them (1.5x / 2x stress tests).
* Inside a bar the stop is assumed to be hit BEFORE any profit target (conservative).
* Stops only ever move in the profitable direction. The engine asserts it.
* Pyramiding adds only when the stop already protects breakeven and the trade is in
  profit — never to a loser. The engine asserts it.
* Results are in R (multiples of initial risk) and as a daily R series for equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tfx.strategies.rules import StrategyParams, signals_for, trend_exit


class RuleViolationError(AssertionError):
    """A risk rule was broken inside the engine (stop widened, add to a loser...)."""


@dataclass(frozen=True)
class Costs:
    spread: float              # price units, full spread
    commission: float          # price units per unit traded, round turn
    slippage: float            # price units per side
    swap_per_night: float      # price units per unit per night held
    multiplier: float = 1.0

    @property
    def half_spread(self) -> float:
        return self.spread / 2 * self.multiplier

    @property
    def per_side_fill(self) -> float:
        return (self.spread / 2 + self.slippage) * self.multiplier

    @property
    def commission_side(self) -> float:
        return self.commission / 2 * self.multiplier

    @property
    def swap(self) -> float:
        return self.swap_per_night * self.multiplier

    def scaled(self, m: float) -> Costs:
        return Costs(self.spread, self.commission, self.slippage, self.swap_per_night, m)


@dataclass(frozen=True)
class Management:
    partial_at_r: float = 1.0
    partial_fraction: float = 0.5
    breakeven_after_partial: bool = True
    profit_lock_at_r: float = 3.0
    profit_lock_keep_r: float = 1.5
    pyramid_at_r: float = 2.0
    pyramid_risk_fraction: float = 0.25
    max_pyramid_adds: int = 1
    trail_channel: bool = True
    allow_grade_b: bool = True
    grade_b_size: float = 0.5

    @classmethod
    def from_config(cls, m: dict[str, Any], overrides: dict[str, Any] | None = None
                    ) -> Management:
        base = cls(partial_at_r=float(m.get("partial_at_r", 1.0)),
                   partial_fraction=float(m.get("partial_fraction", 0.5)),
                   pyramid_at_r=float(m.get("pyramid_at_r", 2.0)),
                   pyramid_risk_fraction=float(m.get("pyramid_risk_fraction", 0.25)),
                   max_pyramid_adds=min(2, int(m.get("max_pyramid_adds", 1))))
        if overrides:
            base = cls(**{**base.__dict__, **{k: v for k, v in overrides.items()
                                               if k in base.__dict__}})
        return base


@dataclass
class Trade:
    symbol: str
    strategy: str
    direction: int
    grade: str
    size: float                       # fraction of full risk (1.0 = full, 0.5 = half)
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry: float
    stop0: float
    risk: float                       # |entry - stop0| in price
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    r_gross: float = 0.0
    r_cost: float = 0.0
    r_net: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars: int = 0
    partial: bool = False
    adds: int = 0
    stop_path: list[float] = field(default_factory=list)
    daily_r: dict[pd.Timestamp, float] = field(default_factory=dict)

    @property
    def currencies(self) -> tuple[str, str]:
        return self.symbol[:3], self.symbol[3:]


def _check_stop_move(direction: int, old: float, new: float) -> float:
    if direction * (new - old) < -1e-12:
        raise RuleViolationError(f"stop widened from {old} to {new}")
    return new


def run_symbol(symbol: str, strategy: str, df: pd.DataFrame, f: pd.DataFrame,
               p: StrategyParams, mgmt: Management, costs: Costs,
               start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
               signals: pd.DataFrame | None = None) -> list[Trade]:
    """Backtest one strategy on one symbol. ``signals`` may be injected (tests)."""
    sig = signals if signals is not None else signals_for(strategy, f, p)
    idx = f.index
    o, h, lo, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    d_sig = sig["direction"].to_numpy(int)
    stops = sig["stop"].to_numpy(float)
    grades = sig["grade"].to_numpy(object)
    long_x, short_x = trend_exit(f, p)
    long_x, short_x = long_x.to_numpy(bool), short_x.to_numpy(bool)
    trail_lo = f.get(f"dcl_{p.exit_channel}", f["low"].rolling(p.exit_channel).min().shift(1))
    trail_hi = f.get(f"dch_{p.exit_channel}", f["high"].rolling(p.exit_channel).max().shift(1))
    trail_lo, trail_hi = trail_lo.to_numpy(float), trail_hi.to_numpy(float)
    ema5, adx = f["ema_5"].to_numpy(float), f["adx_14"].to_numpy(float)
    lo_i = 0 if start is None else int(idx.searchsorted(start))
    hi_i = len(idx) if end is None else int(idx.searchsorted(end, side="right"))

    trades: list[Trade] = []
    t: Trade | None = None
    units = 0.0           # position size in "R per 1.0 price unit" (1R move == size)
    exit_next = ""
    stop = np.nan
    mark = np.nan

    def close_out(i: int, price: float, reason: str) -> None:
        nonlocal t, units
        assert t is not None
        fill = price - t.direction * costs.per_side_fill
        pnl = units * t.direction * (fill - mark)
        cost = units * costs.commission_side
        t.daily_r[idx[i]] = t.daily_r.get(idx[i], 0.0) + pnl - cost
        t.r_cost += units * (costs.per_side_fill + costs.commission_side)
        t.exit_date, t.exit_price, t.exit_reason = idx[i], price, reason
        t.bars = i - idx.get_loc(t.entry_date) + 1
        t.r_net = sum(t.daily_r.values()) / t.size
        t.r_gross = t.r_net + t.r_cost / t.size
        trades.append(t)
        t, units = None, 0.0

    for i in range(max(lo_i, 1), hi_i):
        # ---- 1. pending exit decided at the previous close: exit at this open
        if t is not None and exit_next:
            close_out(i, o[i], exit_next)
            exit_next = ""
        # ---- 2. entry from the previous bar's signal, at this open
        j = i - 1
        if t is None and d_sig[j] != 0 and j >= lo_i:
            g = str(grades[j])
            size = 1.0 if g == "A" else (mgmt.grade_b_size if g == "B" and mgmt.allow_grade_b
                                         else 0.0)
            if strategy == "mean_reversion":
                size *= 0.5  # 0.25% vs the 0.5% default risk
            d = int(d_sig[j])
            entry = o[i] + d * costs.per_side_fill
            risk = d * (entry - stops[j])
            if size > 0 and np.isfinite(stops[j]) and risk > 0:
                t = Trade(symbol, strategy, d, g, size, idx[j], idx[i], entry, stops[j], risk)
                units = size / risk
                stop = stops[j]
                mark = entry
                t.stop_path.append(stop)
                t.daily_r[idx[i]] = -units * costs.commission_side
                t.r_cost += units * (costs.per_side_fill + costs.commission_side)
        if t is None:
            continue
        d, entry, risk = t.direction, t.entry, t.risk
        # ---- 3. stop first (conservative), including gaps through the stop
        hit = lo[i] <= stop if d == 1 else h[i] >= stop
        if hit:
            px = min(stop, o[i]) if d == 1 else max(stop, o[i])
            if idx[i] == t.entry_date:
                px = stop if (d == 1 and o[i] > stop) or (d == -1 and o[i] < stop) else px
            close_out(i, px, "stop")
            continue
        best = h[i] if d == 1 else lo[i]
        worst = lo[i] if d == 1 else h[i]
        t.mfe_r = max(t.mfe_r, d * (best - entry) / risk)
        t.mae_r = min(t.mae_r, d * (worst - entry) / risk)
        # ---- 4. partial exit at +X R, then stop to breakeven + spread
        lvl = entry + d * mgmt.partial_at_r * risk
        if not t.partial and d * (best - lvl) >= 0 and mgmt.partial_fraction > 0:
            part = units * mgmt.partial_fraction
            fill = lvl - d * costs.half_spread
            t.daily_r[idx[i]] = t.daily_r.get(idx[i], 0.0) + part * d * (fill - mark) \
                - part * costs.commission_side
            t.r_cost += part * (costs.half_spread + costs.commission_side)
            units -= part
            t.partial = True
            if mgmt.breakeven_after_partial:
                be = entry + d * costs.spread * costs.multiplier
                if d * (be - stop) > 0:
                    stop = _check_stop_move(d, stop, be)
                    t.stop_path.append(stop)
        # ---- 5. pyramiding: only in profit, only when the stop protects breakeven
        if (strategy == "trend" and t.adds < mgmt.max_pyramid_adds
                and t.mfe_r >= mgmt.pyramid_at_r * (t.adds + 1)):
            add_px = entry + d * mgmt.pyramid_at_r * (t.adds + 1) * risk
            in_profit = d * (c[i] - entry) > 0
            protected = d * (stop - entry) >= 0
            if in_profit and protected:
                add_risk = d * (add_px - stop)
                if add_risk <= 0:
                    raise RuleViolationError("pyramid add with no risk distance")
                add_units = mgmt.pyramid_risk_fraction * t.size / add_risk
                add_fill = add_px + d * costs.per_side_fill
                # bring the new units onto the common mark price
                t.daily_r[idx[i]] = t.daily_r.get(idx[i], 0.0) + add_units * d * (
                    mark - add_fill) - add_units * costs.commission_side
                t.r_cost += add_units * (costs.per_side_fill + costs.commission_side)
                units += add_units
                t.adds += 1
        # ---- 6. end of bar: mark to market, swap, trailing, exits
        t.daily_r[idx[i]] = t.daily_r.get(idx[i], 0.0) + units * d * (c[i] - mark) \
            - units * costs.swap
        t.r_cost += units * costs.swap
        mark = c[i]
        new_stop = stop
        if strategy in ("trend", "pullback") and mgmt.trail_channel:
            ch = trail_lo[i] if d == 1 else trail_hi[i]
            if np.isfinite(ch) and d * (ch - new_stop) > 0:
                new_stop = ch
        if t.mfe_r >= mgmt.profit_lock_at_r:
            lock = entry + d * mgmt.profit_lock_keep_r * risk
            if d * (lock - new_stop) > 0:
                new_stop = lock
        if new_stop != stop:
            stop = _check_stop_move(d, stop, new_stop)
            t.stop_path.append(stop)
        held = i - idx.get_loc(t.entry_date) + 1
        if strategy in ("trend", "pullback"):
            if (d == 1 and long_x[i]) or (d == -1 and short_x[i]):
                exit_next = "exit channel"
        else:
            if (d == 1 and c[i] >= ema5[i]) or (d == -1 and c[i] <= ema5[i]):
                exit_next = "target (5 EMA)"
            elif held >= p.mr_time_stop:
                exit_next = "time stop"
            elif adx[i] > 25:
                exit_next = "trend emerging (ADX > 25)"
        if i == hi_i - 1 and t is not None:
            close_out(i, c[i], "end of test")
    return trades


def daily_r_frame(trades: list[Trade], index: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series(0.0, index=index)
    for t in trades:
        for day, r in t.daily_r.items():
            if day in out.index:
                out[day] += r
    return out


def apply_portfolio_caps(trades: list[Trade], max_open_total_risk: float,
                         risk_pct: float, max_per_currency: int = 2,
                         max_open_by_strategy: dict[str, int] | None = None) -> list[Trade]:
    """Keep trades in entry order subject to the same caps the live risk engine uses."""
    kept: list[Trade] = []
    open_: list[Trade] = []
    caps = max_open_by_strategy or {}
    for t in sorted(trades, key=lambda x: (x.entry_date, x.symbol, x.strategy)):
        open_ = [o for o in open_ if o.exit_date is not None and o.exit_date > t.entry_date]
        risk_now = sum(o.size * risk_pct for o in open_)
        if risk_now + t.size * risk_pct > max_open_total_risk + 1e-12:
            continue
        cur = t.currencies
        if any(sum(1 for o in open_ if c in o.currencies) >= max_per_currency for c in cur):
            continue
        cap = caps.get(t.strategy)
        if cap is not None and sum(1 for o in open_ if o.strategy == t.strategy) >= cap:
            continue
        kept.append(t)
        open_.append(t)
    return kept
