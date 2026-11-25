"""Hypothetical replay: what the system's rules would have done over a past window.

Uses the SAME backtest engine, costs and portfolio caps as validation, on confirmed
data, with signals only from bars inside the window. Trades still open at the end
are marked to the last close and labelled "open". Results are HYPOTHETICAL:
strategies that failed validation are locked in the live app, and a single week is
far too short to mean anything statistically.

Small accounts: each trade is checked against the broker's minimum position
(0.01 lot). If the minimum lot would risk more than the chosen risk %, the trade is
flagged "below minimum lot" and the risk that would actually be required is shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from tfx.backtest.engine import Management, Trade, apply_portfolio_caps, run_symbol
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.risk.engine import FxConverter
from tfx.strategies.features import build_features
from tfx.strategies.rules import STRATEGIES, StrategyParams
from tfx.validation.proof import EXCLUDED, costs_for, load_report

MIN_LOT = 0.01
DISCLAIMER = ("HYPOTHETICAL replay of rules that FAILED validation (locked in the live app: it "
              "would have shown WAIT). Short windows are statistically meaningless, and long "
              "windows include the years used to choose the parameters. Past performance "
              "does not guarantee future results.")
ZONE_STRATEGIES = ("ict", "ict_flex", "supply_demand")
#: the live order: ICT first; supply & demand only when no ICT trade is open on the pair
STACK = "ict_then_sd"
ALL_REPLAYABLE = (*STRATEGIES, *ZONE_STRATEGIES, STACK)


def _zone_trades(cfg: AppConfig, strat: str, sym: str, df: pd.DataFrame,
                 params: dict[str, Any], start: pd.Timestamp) -> list[Trade]:
    from tfx.strategies import ict, supply_demand

    conf = cfg.section("strategies").get(strat, {})
    if strat == "supply_demand":
        return supply_demand.simulate(sym, df, supply_demand.SDParams.from_config(conf, params),
                                      costs_for(cfg, sym), start)
    return ict.simulate(sym, df, ict.ICTParams.from_config(conf, params), costs_for(cfg, sym),
                        start)


def _stack(first: list[Trade], second: list[Trade]) -> list[Trade]:
    """ICT trades always; a supply & demand trade only if no ICT trade overlaps it."""
    out = list(first)
    for t in second:
        clash = any(not ((t.exit_date is not None and t.exit_date < u.entry_date)
                         or (u.exit_date is not None and u.exit_date < t.entry_date))
                    for u in first)
        if not clash:
            out.append(t)
    return sorted(out, key=lambda x: x.entry_date)


@dataclass
class ReplayTrade:
    symbol: str
    strategy: str
    side: str
    signal_date: str
    entry_date: str
    entry: float
    stop: float
    exit_date: str
    exit_price: float
    status: str               # closed | open (marked to market)
    exit_reason: str
    r: float
    risk_amount: float
    pnl: float
    lots: float
    min_lot_risk: float       # amount at risk if forced up to 0.01 lot
    min_lot_risk_pct: float
    tradable: bool


@dataclass
class ReplayResult:
    start: str
    end: str
    starting_balance: float
    ending_balance: float
    pnl: float
    return_pct: float
    risk_pct: float
    trades: list[ReplayTrade] = field(default_factory=list)
    untradable: int = 0
    disclaimer: str = DISCLAIMER
    notes: list[str] = field(default_factory=list)


def _rates(cache: CandleCache, cfg: AppConfig) -> FxConverter:
    rates = {}
    for s in cfg.symbols:
        df = cache.load(s)
        if not df.empty:
            rates[s] = float(df["close"].iloc[-1])
    return FxConverter(rates)


def replay(cfg: AppConfig, cache: CandleCache, start: date, end: date, balance: float,
           risk_pct: float, account_ccy: str = "USD",
           strategies: tuple[str, ...] = STRATEGIES) -> ReplayResult:
    if risk_pct > 1.0:
        raise ValueError("risk per trade is capped at 1%")
    rep = load_report() or {}
    chosen = {p["strategy"]: p.get("selected_params", {}) for p in rep.get("pairs", [])
              if p.get("timeframe") == "D1"}
    fx = _rates(cache, cfg)
    s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
    trades: list[Trade] = []
    for sym in cfg.symbols:
        if sym in EXCLUDED:
            continue
        df = cache.load(sym)
        if df.empty:
            continue
        f = build_features(df)
        for strat in strategies:
            params = chosen.get(strat, {})
            if strat in ZONE_STRATEGIES or strat == STACK:
                s_from = s0 - pd.Timedelta(days=4)
                if strat == STACK:
                    zt = _stack(_zone_trades(cfg, "ict", sym, df, chosen.get("ict", {}), s_from),
                                _zone_trades(cfg, "supply_demand", sym, df,
                                             chosen.get("supply_demand", {}), s_from))
                else:
                    zt = _zone_trades(cfg, strat, sym, df, params, s_from)
                trades += [t for t in zt if s0 <= t.signal_date <= e0]
                continue
            sp = {k: v for k, v in params.items() if k not in ("partial_at_r",)}
            mp = {k: v for k, v in params.items() if k == "partial_at_r"}
            p = StrategyParams.from_config(cfg.section("strategies"), sp)
            m = Management.from_config(cfg.section("strategies").get("management", {}), mp)
            # run from just before the window so a Monday entry from Friday's signal works
            for t in run_symbol(sym, strat, df, f, p, m, costs_for(cfg, sym),
                                start=s0 - pd.Timedelta(days=4)):
                if s0 <= t.signal_date <= e0:
                    trades.append(t)
    risk_frac = risk_pct / 100
    caps = {"mean_reversion": int(cfg.section("strategies").get("mean_reversion", {})
                                  .get("max_open", 2))}
    kept = apply_portfolio_caps(trades, cfg.section("risk").get("max_open_risk_pct", 3.0) / 100,
                                risk_frac, int(cfg.section("strategies").get("filters", {})
                                               .get("max_trades_per_currency", 2)), caps)
    bal = balance
    out: list[ReplayTrade] = []
    for t in sorted(kept, key=lambda x: (x.exit_date or x.entry_date, x.symbol)):
        sym_cfg = cfg.symbol(t.symbol)
        risk_amt = bal * risk_frac * t.size
        loss_per_unit = fx.convert(t.risk, sym_cfg.quote, account_ccy)
        units = risk_amt / loss_per_unit if loss_per_unit > 0 else 0.0
        lot_size = 100 if sym_cfg.base == "XAU" else 100_000
        lots = units / lot_size
        min_risk = MIN_LOT * lot_size * loss_per_unit
        pnl = t.r_net * risk_amt
        bal += pnl
        is_open = t.exit_reason == "end of test"
        out.append(ReplayTrade(
            t.symbol, t.strategy, "LONG" if t.direction == 1 else "SHORT",
            str(t.signal_date.date()), str(t.entry_date.date()), round(t.entry, 5),
            round(t.stop0, 5), str(t.exit_date.date()) if t.exit_date is not None else "",
            round(float(t.exit_price or 0), 5), "open (marked to market)" if is_open
            else "closed", t.exit_reason, round(t.r_net, 2), round(risk_amt, 2),
            round(pnl, 2), round(lots, 4), round(min_risk, 2),
            round(min_risk / bal * 100 if bal > 0 else 0, 2), lots >= MIN_LOT))
    res = ReplayResult(str(start), str(end), balance, round(bal, 2), round(bal - balance, 2),
                       round((bal / balance - 1) * 100, 2), risk_pct, out,
                       sum(1 for x in out if not x.tradable))
    if res.untradable:
        res.notes.append(f"{res.untradable} trade(s) are below the broker minimum of {MIN_LOT} "
                         f"lot at {risk_pct}% risk on a {balance:g} {account_ccy} account. "
                         "Taking them at 0.01 lot would risk the amount shown in 'min-lot risk'.")
    if not out:
        res.notes.append("The rules produced no trades in this window.")
    return res


def max_drawdown_pct(r: ReplayResult) -> float:
    """Largest peak-to-trough fall of the balance, in % (closed-trade equity)."""
    bal, peak, worst = r.starting_balance, r.starting_balance, 0.0
    for t in r.trades:
        bal += t.pnl
        peak = max(peak, bal)
        worst = min(worst, bal / peak - 1 if peak > 0 else 0.0)
    return round(worst * 100, 2)


def _plain(x: Any) -> Any:
    """Convert numpy scalars (bool/float/int) to plain Python for JSON."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        return x.item()
    return x


def as_dict(r: ReplayResult) -> dict[str, Any]:
    from dataclasses import asdict

    out: dict[str, Any] = _plain(asdict(r))
    return out


MOMENTUM_NOTE = ("Momentum sleeves have no per-trade stop: shown as the sleeve's return over the "
                 "window at its own volatility target, applied to the starting balance.")


def momentum_window(cfg: AppConfig, cache: CandleCache, strategy: str, start: date,
                    end: date, balance: float) -> dict[str, Any]:
    """Return of a momentum sleeve over [start, end] (validation-selected parameters)."""
    from tfx.validation.proof import ProofRunner
    from tfx.validation.registry import ValidationRegistry

    rep = load_report() or {}
    params: dict[str, Any] = next((p.get("selected_params", {}) for p in rep.get("pairs", [])
                   if p.get("strategy") == strategy), {})
    if not params:
        grid = cfg.section("parameter_grid").get(strategy, {})
        params = {k: v[len(v) // 2] for k, v in grid.items()}
    runner = ProofRunner(cfg, cache, ValidationRegistry(":memory:"))
    returns, trades = runner.run_momentum(strategy, params)
    win = returns[(returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))]
    growth = float((1 + win).prod()) if len(win) else 1.0
    n_trades = sum(1 for t in trades if pd.Timestamp(start) <= t.entry_date <= pd.Timestamp(end))
    return {"strategy": strategy, "kind": "sleeve", "starting_balance": balance,
            "ending_balance": round(balance * growth, 2),
            "pnl": round(balance * (growth - 1), 2), "return_pct": round((growth - 1) * 100, 2),
            "trades": n_trades, "untradable": None, "params": params, "note": MOMENTUM_NOTE}


def compare(cfg: AppConfig, cache: CandleCache, start: date, end: date, balance: float,
            risk_pct: float, account_ccy: str = "USD", include_momentum: bool = True
            ) -> dict[str, Any]:
    """Each strategy on its own, plus all rule-based strategies together."""
    rows: list[dict[str, Any]] = []
    for strat in (*ALL_REPLAYABLE, "all"):
        chosen = STRATEGIES if strat == "all" else (strat,)
        r = replay(cfg, cache, start, end, balance, risk_pct, account_ccy, chosen)
        closed = [t for t in r.trades if t.status == "closed"]
        wins = [t for t in closed if t.r > 0]
        rows.append({"strategy": strat, "kind": "per-trade", "starting_balance": balance,
                     "ending_balance": r.ending_balance, "pnl": r.pnl,
                     "return_pct": r.return_pct, "trades": len(r.trades),
                     "closed": len(closed), "open": len(r.trades) - len(closed),
                     "win_rate_closed": round(len(wins) / len(closed), 3) if closed else None,
                     "untradable": r.untradable,
                     "max_drawdown_pct": max_drawdown_pct(r),
                     "note": "all = trend + pullback + mean reversion" if strat == "all" else
                     ("ICT first, supply & demand when ICT has nothing (live order)"
                      if strat == STACK else "")})
    if include_momentum:
        for strat in ("xs_momentum", "ts_momentum"):
            rows.append(momentum_window(cfg, cache, strat, start, end, balance))
    return {"start": str(start), "end": str(end), "starting_balance": balance,
            "risk_pct": risk_pct, "rows": _plain(rows), "disclaimer": DISCLAIMER}
