"""Shared limit-order simulator for zone strategies (supply & demand, missed-trade study).

Identical execution rules to ``tfx.strategies.ict.simulate`` (kept separate so the
validated ICT rule hash does not change):

* the LIMIT order is live from the bar after ``known_bar`` up to ``expiry_bar``;
* if the stop is touched before the entry the order is cancelled, unless the bar
  opened beyond the entry (then it filled on the way to the stop: a loss);
* stop first on any bar where stop and target could both be hit; no target on the
  fill bar; time stop after ``time_stop`` bars; real costs on every fill and night.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tfx.backtest.engine import Costs, Trade


@dataclass(frozen=True)
class LimitOrder:
    direction: int
    known_bar: int      # the bar whose close made the setup known
    entry: float
    stop: float
    target: float
    expiry_bar: int


def fill_and_manage(symbol: str, strategy: str, df: pd.DataFrame, od: LimitOrder,
                    costs: Costs, time_stop: int, target_label: str) -> Trade | None:
    idx = df.index
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    d = od.direction
    fill_bar = None
    for t in range(od.known_bar + 1, min(od.expiry_bar, len(df) - 1) + 1):
        if (d == 1 and lo[t] <= od.stop) or (d == -1 and h[t] >= od.stop):
            if (d == 1 and o[t] > od.entry) or (d == -1 and o[t] < od.entry):
                fill_bar = t
            break
        if (d == 1 and lo[t] <= od.entry) or (d == -1 and h[t] >= od.entry):
            fill_bar = t
            break
    if fill_bar is None:
        return None
    fill_px = min(o[fill_bar], od.entry) if d == 1 else max(o[fill_bar], od.entry)
    entry = fill_px + d * costs.half_spread
    risk = d * (entry - od.stop)
    if risk <= 0:
        return None
    t0 = Trade(symbol, strategy, d, "A", 1.0, idx[od.known_bar], idx[fill_bar], entry,
               od.stop, risk)
    t0.stop_path.append(od.stop)
    units = 1.0 / risk
    mark = entry
    t0.daily_r[idx[fill_bar]] = -units * costs.commission_side
    t0.r_cost += units * (costs.half_spread + costs.commission_side)
    exit_px, reason, last = None, "", fill_bar
    for t in range(fill_bar, len(df)):
        last = t
        if (d == 1 and lo[t] <= od.stop) or (d == -1 and h[t] >= od.stop):
            gap_px = min(od.stop, o[t]) if d == 1 else max(od.stop, o[t])
            px = gap_px if t > fill_bar else od.stop
            exit_px, reason = px - d * costs.per_side_fill, "stop"
            break
        best = h[t] if d == 1 else lo[t]
        worst = lo[t] if d == 1 else h[t]
        t0.mfe_r = max(t0.mfe_r, d * (best - entry) / risk)
        t0.mae_r = min(t0.mae_r, d * (worst - entry) / risk)
        if t > fill_bar and d * (best - od.target) >= 0:
            exit_px, reason = od.target - d * costs.half_spread, f"target ({target_label})"
            break
        if t - fill_bar >= time_stop:
            exit_px, reason = c[t] - d * costs.per_side_fill, "time stop"
            break
        t0.daily_r[idx[t]] = t0.daily_r.get(idx[t], 0.0) + units * d * (c[t] - mark) \
            - units * costs.swap
        t0.r_cost += units * costs.swap
        mark = c[t]
    if exit_px is None:
        exit_px, reason = c[last], "end of test"
    t0.daily_r[idx[last]] = t0.daily_r.get(idx[last], 0.0) + units * d * (exit_px - mark) \
        - units * costs.commission_side
    t0.r_cost += units * (costs.per_side_fill + costs.commission_side)
    t0.exit_date, t0.exit_price, t0.exit_reason = idx[last], exit_px, reason
    t0.bars = last - fill_bar + 1
    t0.r_net = float(sum(t0.daily_r.values()))
    t0.r_gross = t0.r_net + t0.r_cost
    return t0


def simulate_orders(symbol: str, strategy: str, df: pd.DataFrame, orders: list[LimitOrder],
                    costs: Costs, time_stop: int, target_label: str,
                    start: pd.Timestamp | None = None) -> list[Trade]:
    """One position at a time per symbol, orders in the order they became known."""
    s0 = 0 if start is None else int(df.index.searchsorted(start))
    out: list[Trade] = []
    busy_until = -1
    for od in sorted(orders, key=lambda x: x.known_bar):
        if od.known_bar < s0 or od.known_bar <= busy_until:
            continue
        tr = fill_and_manage(symbol, strategy, df, od, costs, time_stop, target_label)
        if tr is None:
            continue
        out.append(tr)
        busy_until = int(np.searchsorted(df.index, tr.exit_date))
    return out
