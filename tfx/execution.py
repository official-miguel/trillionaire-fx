"""Execution guidance: order type/price/expiry, execution quality, slippage learning.

Guidance only. TrillionaireFX never sends orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from tfx.risk.engine import in_rollover_window

OVERLAP_UTC_HOURS = range(12, 17)        # London / New York overlap
THIN_UTC_HOURS = {20, 21, 22, 23}        # around rollover / Asia open


@dataclass(frozen=True)
class OrderPlan:
    order_type: str        # STOP-ENTRY | LIMIT | MARKET-AT-OPEN
    price: float
    expiry: str
    text: str


def order_plan(strategy: str, direction: int, close: float, atr: float, level: float | None,
               ema20: float | None, session: date, digits: int = 5) -> OrderPlan:
    side = "BUY" if direction == 1 else "SELL"
    nxt = session + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    if strategy == "trend":
        ref = max(close, level or close) if direction == 1 else min(close, level or close)
        px = round(ref + direction * 0.1 * atr, digits)
        return OrderPlan("STOP-ENTRY", px, f"end of {nxt:%a %d %b} (UTC)",
                         f"{side} STOP at {px} (just beyond the breakout level); cancel if not "
                         f"filled by the end of {nxt:%a %d %b}.")
    if strategy == "pullback":
        px = round(ema20 if ema20 is not None else close, digits)
        exp = nxt + timedelta(days=2)
        while exp.weekday() >= 5:
            exp += timedelta(days=1)
        return OrderPlan("LIMIT", px, f"end of {exp:%a %d %b} (UTC)",
                         f"{side} LIMIT at {px} (the 20 EMA); cancel after {exp:%a %d %b}.")
    return OrderPlan("MARKET-AT-OPEN", round(close, digits), f"{nxt:%a %d %b} open",
                     f"{side} at the next session open (mean reversion is time-sensitive); "
                     "skip if the open gaps beyond the stop.")


def execution_quality(spread_by_hour: pd.Series, typical_spread_points: float | None,
                      now: datetime) -> dict[str, object]:
    """0-100 score for entering now, plus the best hours, from broker spread history."""
    hour = now.hour
    if spread_by_hour.empty:
        base: float = 70.0 if hour in OVERLAP_UTC_HOURS else (
            30.0 if hour in THIN_UTC_HOURS else 55.0)
        best: list[int] = list(OVERLAP_UTC_HOURS)
        note = "No broker spread history yet: score from session times only."
    else:
        s = spread_by_hour.astype(float)
        cur = float(s.get(hour, s.median()))
        rel = float(s.min() / cur) if cur > 0 else 0.0
        base = 100 * rel
        if hour in OVERLAP_UTC_HOURS:
            base = min(100, base + 10)
        best = [int(h) for h in s.sort_values().index[:4]]
        note = f"Median broker spread now {cur:.0f} points (best hour {s.idxmin()}:00 UTC)."
    if in_rollover_window(now):
        base, note = min(base, 10), "Rollover window: spreads widen. Wait."
    return {"score": round(float(np.clip(base, 0, 100))), "best_utc_hours": best,
            "prefer_overlap": "London/New York overlap 12:00-16:00 UTC when not time-critical",
            "note": note}


@dataclass(frozen=True)
class SlippageReport:
    by_symbol: dict[str, float]           # average slippage in pips (positive = worse)
    assumed_pips: float
    over_assumption: dict[str, float]     # ratio real/assumed for symbols above 1.25x
    revalidate: bool


def slippage_report(fills: list[tuple[str, float]], assumed_pips: float,
                    min_fills: int = 5) -> SlippageReport:
    """``fills``: (symbol, slippage_pips). Real costs > 1.25x assumption => re-validate."""
    df = pd.DataFrame(fills, columns=["symbol", "pips"])
    by = df.groupby("symbol")["pips"].agg(["mean", "count"]) if len(df) else pd.DataFrame()
    avg = {str(k): float(v) for k, v in by["mean"].items()} if len(by) else {}
    over = {}
    for sym, row in (by.iterrows() if len(by) else []):
        if row["count"] >= min_fills and assumed_pips > 0 and row["mean"] > 1.25 * assumed_pips:
            over[str(sym)] = float(row["mean"] / assumed_pips)
    return SlippageReport(avg, assumed_pips, over, bool(over))


def cost_in_r(spread_pips: float, commission_pips: float, slippage_pips: float,
              stop_pips: float) -> float:
    """Round-trip cost of one trade expressed in R (1R = the stop distance)."""
    if stop_pips <= 0:
        return float("inf")
    return (spread_pips + commission_pips + 2 * slippage_pips) / stop_pips
