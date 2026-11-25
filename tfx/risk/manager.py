"""Live trade manager: today's R-multiple and the next required action, in plain English.

Advice comes from fixed rules tied to measured conditions. It never suggests
widening a stop, averaging down, adding to a loser or increasing size after losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class ManagedTrade:
    symbol: str
    strategy: str
    direction: int
    entry: float
    initial_stop: float
    stop: float
    opened: pd.Timestamp
    partial_done: bool = False
    adds: int = 0

    @property
    def risk(self) -> float:
        return abs(self.entry - self.initial_stop)

    def r_at(self, price: float) -> float:
        return self.direction * (price - self.entry) / self.risk if self.risk else 0.0


@dataclass(frozen=True)
class Action:
    code: str
    text: str
    new_stop: float | None = None


def next_actions(t: ManagedTrade, bar: pd.Series, bars_held: int, spread: float,
                 partial_at_r: float = 1.0, pyramid_at_r: float = 2.0, max_adds: int = 1,
                 profit_lock_at_r: float = 3.0, keep_r: float = 1.5,
                 mr_time_stop: int = 5, exit_channel: int = 20) -> list[Action]:
    """Rules applied to the latest CLOSED bar (``bar`` holds price and feature columns)."""
    d = t.direction
    close, high, low = float(bar["close"]), float(bar["high"]), float(bar["low"])
    r_now = t.r_at(close)
    best_r = t.r_at(high if d == 1 else low)
    out: list[Action] = []
    if (d == 1 and low <= t.stop) or (d == -1 and high >= t.stop):
        return [Action("STOP_HIT", f"Stop at {t.stop:.5g} was touched: the trade is closed. "
                                   "Log the exit; do not re-enter today.")]
    if t.strategy in ("trend", "pullback"):
        lo, hi = bar.get(f"dcl_{exit_channel}"), bar.get(f"dch_{exit_channel}")
        if (d == 1 and lo is not None and close < lo) or (d == -1 and hi is not None
                                                          and close > hi):
            out.append(Action("EXIT_RULE", "Exit rule triggered: close at the next open. "
                                           "Do not wait for a bounce."))
    else:
        e5 = bar.get("ema_5")
        if e5 is not None and ((d == 1 and close >= e5) or (d == -1 and close <= e5)):
            out.append(Action("TARGET", "Mean-reversion target (5 EMA) reached: close at the "
                                        "next open."))
        elif bars_held >= mr_time_stop:
            out.append(Action("TIME_STOP", "Mean-reversion time stop reached: close at the "
                                           "next open."))
        elif float(bar.get("adx_14", 0) or 0) > 25:
            out.append(Action("TREND_EMERGING", "ADX rose above 25: the range is breaking. "
                                                "Close at the next open."))
    if not t.partial_done and best_r >= partial_at_r:
        be = t.entry + d * spread
        out.append(Action("TAKE_PARTIAL", f"+{partial_at_r:g}R reached: take half, move the "
                                          f"stop to breakeven ({be:.5g}).", be))
    new_stop = t.stop
    if t.strategy in ("trend", "pullback"):
        ch = bar.get(f"dcl_{exit_channel}") if d == 1 else bar.get(f"dch_{exit_channel}")
        if ch is not None and not (isinstance(ch, float) and math.isnan(ch)) and \
                d * (float(ch) - new_stop) > 0:
            new_stop = float(ch)
    if best_r >= profit_lock_at_r:
        lock = t.entry + d * keep_r * t.risk
        if d * (lock - new_stop) > 0:
            new_stop = lock
    if d * (new_stop - t.stop) > 1e-12:
        out.append(Action("TRAIL", f"Tighten the stop to {new_stop:.5g} (it only ever moves "
                                   "in your favour).", new_stop))
    stop_protects = d * (max(t.stop, new_stop) if d == 1 else min(t.stop, new_stop)
                         ) >= d * t.entry
    if (t.strategy == "trend" and t.adds < max_adds and best_r >= pyramid_at_r * (t.adds + 1)
            and r_now > 0 and stop_protects):
        out.append(Action("PYRAMID_OK", f"+{pyramid_at_r:g}R in a confirmed trend with the "
                                        "stop at or beyond breakeven: one add-on of 25% of "
                                        "the original risk is allowed (optional)."))
    if not out:
        out.append(Action("HOLD", f"Hold. Trade is at {r_now:+.2f}R; the stop stays at "
                                  f"{t.stop:.5g}. No action needed today."))
    return out
