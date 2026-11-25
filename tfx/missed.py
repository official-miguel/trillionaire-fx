"""Missed-trade control: what happened to setups that were skipped or reduced.

* Missed Opportunities report: every non-taken valid setup with its hypothetical
  result (same engine, same costs) and the reason, split by reason.
* Second-chance entries, pullback re-entry watches, re-entry after a stop-out.
* Opportunity-capture rate: valid setups traded / valid setups available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import Costs, Management, run_symbol
from tfx.strategies.rules import StrategyParams

VALID = ("BUY", "SELL")


def hypothetical_r(symbol: str, strategy: str, direction: int, stop: float,
                   signal_date: pd.Timestamp, df: pd.DataFrame, f: pd.DataFrame,
                   p: StrategyParams, m: Management, costs: Costs) -> float | None:
    """Result in R had the setup been taken (None while the trade would still be open)."""
    if signal_date not in f.index:
        return None
    sig = pd.DataFrame({"direction": 0, "stop": np.nan, "grade": "", "confidence": 0.0,
                        "reason": "", "watch_level": np.nan}, index=f.index)
    i = f.index.get_loc(signal_date)
    sig.iloc[i, 0], sig.iloc[i, 1], sig.iloc[i, 2] = direction, stop, "A"
    trades = run_symbol(symbol, strategy, df, f, p, m, costs, start=signal_date, signals=sig)
    if not trades or trades[0].exit_reason == "end of test":
        return None
    return float(trades[0].r_net)


@dataclass(frozen=True)
class MissedReport:
    rows: list[dict[str, Any]]
    by_reason: dict[str, dict[str, float]]
    capture_rate: float | None
    valid_setups: int
    taken: int
    net_effect_r: float


def missed_report(signals: list[dict[str, Any]], taken_signal_ids: set[int],
                  hypo: dict[int, float | None]) -> MissedReport:
    """``signals``: dicts with id, status, outcome/skip reason, size_mult, symbol, date."""
    rows = []
    valid = [s for s in signals if s["status"] in VALID]
    for s in signals:
        if s["status"] not in (*VALID, "BLOCKED", "WATCH"):
            continue
        taken = s["id"] in taken_signal_ids
        if taken and s.get("size_mult", 1) >= 1:
            continue
        reason = ("reduced size" if taken else s.get("reason") or "user did not act")
        r = hypo.get(s["id"])
        rows.append({**s, "reason": reason, "hypothetical_r": r, "taken": taken})
    df = pd.DataFrame(rows)
    by: dict[str, dict[str, float]] = {}
    if len(df):
        for reason, g in df.groupby("reason"):
            vals = g["hypothetical_r"].dropna()
            by[str(reason)] = {"count": float(len(g)), "avg_r": float(vals.mean()) if len(
                vals) else 0.0, "sum_r": float(vals.sum())}
    taken_n = sum(1 for s in valid if s["id"] in taken_signal_ids)
    cap = taken_n / len(valid) if valid else None
    net = float(df["hypothetical_r"].dropna().sum()) if len(df) else 0.0
    return MissedReport(rows, by, cap, len(valid), taken_n, net)


def second_chance(direction: int, breakout_level: float, close: float, atr: float,
                  trend_intact: bool, original_stop: float) -> dict[str, Any] | None:
    """Late entry within 1 ATR of a missed breakout, at half the probe size (x0.5)."""
    distance = direction * (close - breakout_level)
    if not trend_intact or distance < 0 or distance > atr:
        return None
    return {"size_multiplier": 0.25 * 0.5, "stop": original_stop,
            "text": "Second-chance entry: the breakout is less than 1 ATR away and the trend "
                    "is intact. Reduced size (probe x 0.5), original stop."}


def pullback_watch(direction: int, ema20: float, breakout_level: float,
                   expiry_sessions: int = 5) -> dict[str, Any]:
    lvl = max(ema20, breakout_level) if direction == 1 else min(ema20, breakout_level)
    return {"order_type": "LIMIT", "price": lvl, "expiry_sessions": expiry_sessions,
            "text": f"Pullback re-entry watch: limit at {lvl:.5g} for {expiry_sessions} "
                    "sessions."}


def reentry_allowed(last_stop_out: pd.Timestamp | None, today: pd.Timestamp,
                    new_breakout: bool, within_days: int = 20) -> tuple[bool, float, str]:
    """Re-entry after a stop-out: normal size only, never the same day, no revenge sizing."""
    if last_stop_out is None:
        return (new_breakout, 1.0, "")
    if today.normalize() <= last_stop_out.normalize():
        return (False, 0.0, "No re-entry on the day of a stop-out.")
    if new_breakout and (today - last_stop_out).days <= within_days:
        return (True, 1.0, "Trend resumed with a new breakout: normal size (never larger).")
    return (new_breakout, 1.0, "")
