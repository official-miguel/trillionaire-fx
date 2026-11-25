"""Supply & demand zones as fixed, deterministic rules (closed bars only).

Declared 2026-11-24, BEFORE any supply-and-demand result was seen (config.yaml
``strategies.supply_demand``):

1. Base: 1-3 consecutive candles whose bodies are each <= ``base_body_atr`` x ATR(14)
   (price paused: orders being placed).
2. Leg-out: the next candle has a body >= ``departure_atr`` x ATR in one direction
   and closes beyond the whole base (strong departure = imbalance).
   Up leg-out -> DEMAND zone; down leg-out -> SUPPLY zone.
3. Zone: demand = [lowest low of base + leg-out, highest body top of the base];
   supply = [lowest body bottom of the base, highest high of base + leg-out].
   The zone is KNOWN at the leg-out close; never earlier.
4. With the trend only: demand while EMA(50) > EMA(200), supply while below.
5. Fresh zones only: the FIRST return to the zone is traded; a zone is dead after it
   is touched once, or after ``max_age`` bars.
6. Entry: LIMIT at the proximal edge. Stop: distal edge + ``stop_buffer_atr`` x ATR.
   Zones whose stop is tighter than 0.2 ATR are skipped (costs would dominate).
7. Target: ``target`` = 2R, or the nearest fresh opposing zone at >= 1R (else 2R).
   Time stop ``time_stop`` bars after the fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import Costs, Trade
from tfx.strategies import indicators as ind
from tfx.strategies.limit_sim import LimitOrder, simulate_orders


@dataclass(frozen=True)
class SDParams:
    base_body_atr: float = 0.5
    max_base: int = 3
    departure_atr: float = 1.5
    max_age: int = 60
    stop_buffer_atr: float = 0.1
    target: str = "2R"           # "2R" | "opposing"
    time_stop: int = 15
    trend_filter: bool = True

    @classmethod
    def from_config(cls, strat: dict[str, Any], overrides: dict[str, Any] | None = None
                    ) -> SDParams:
        keys = cls.__dataclass_fields__
        base = {k: strat[k] for k in keys if k in strat}
        base.update({k: v for k, v in (overrides or {}).items() if k in keys})
        return cls(**base)


@dataclass
class Zone:
    direction: int              # +1 demand, -1 supply
    created_bar: int            # leg-out bar (zone known at its close)
    base_start: int
    proximal: float
    distal: float
    low: float
    high: float
    strength: float             # leg-out body in ATR
    touched_bar: int | None = None
    broken_bar: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "demand" if self.direction == 1 else "supply"


def find_zones(df: pd.DataFrame, p: SDParams, atr: np.ndarray | None = None) -> list[Zone]:
    """All zones, each built only from bars up to its leg-out bar, with first touch."""
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    a = atr if atr is not None else ind.atr(df, 14).to_numpy(float)
    body = np.abs(c - o)
    zones: list[Zone] = []
    for t in range(1, n):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        leg = c[t] - o[t]
        if abs(leg) < p.departure_atr * a[t]:
            continue
        nb = 0       # longest run of small-bodied candles directly before t (<= max_base)
        while nb < p.max_base and t - nb - 1 >= 0 and body[t - nb - 1] <= p.base_body_atr * a[t]:
            nb += 1
        if nb == 0:
            continue
        b0, b1 = t - nb, t - 1
        top = float(np.maximum(o[b0:b1 + 1], c[b0:b1 + 1]).max())
        bot = float(np.minimum(o[b0:b1 + 1], c[b0:b1 + 1]).min())
        if leg > 0 and c[t] > h[b0:b1 + 1].max():
            distal = float(lo[b0:t + 1].min())
            z = Zone(1, t, b0, top, distal, distal, top, round(abs(leg) / a[t], 2))
        elif leg < 0 and c[t] < lo[b0:b1 + 1].min():
            distal = float(h[b0:t + 1].max())
            z = Zone(-1, t, b0, bot, distal, bot, distal, round(abs(leg) / a[t], 2))
        else:
            continue
        if z.high <= z.low:
            continue
        for k in range(t + 1, min(n, t + p.max_age + 1)):     # first touch / break
            if (z.direction == 1 and lo[k] <= z.proximal) or (
                    z.direction == -1 and h[k] >= z.proximal):
                z.touched_bar = k
                if (z.direction == 1 and c[k] < z.distal) or (
                        z.direction == -1 and c[k] > z.distal):
                    z.broken_bar = k
                break
        zones.append(z)
    return zones


def orders(df: pd.DataFrame, p: SDParams, atr: np.ndarray | None = None) -> list[LimitOrder]:
    a = atr if atr is not None else ind.atr(df, 14).to_numpy(float)
    e50 = ind.ema(df["close"], 50).to_numpy(float)
    e200 = ind.ema(df["close"], 200).to_numpy(float)
    zones = find_zones(df, p, a)
    out = []
    for z in zones:
        t = z.created_bar
        if p.trend_filter:
            if not (np.isfinite(e200[t]) and t >= 200):
                continue
            if (z.direction == 1 and e50[t] <= e200[t]) or (
                    z.direction == -1 and e50[t] >= e200[t]):
                continue
        stop = z.distal - z.direction * p.stop_buffer_atr * a[t]
        risk = z.direction * (z.proximal - stop)
        if risk < 0.2 * a[t]:
            continue
        tgt = z.proximal + z.direction * 2 * risk
        if p.target == "opposing":
            opp = [y.proximal for y in zones
                   if y.direction == -z.direction and y.created_bar <= t
                   and (y.touched_bar is None or y.touched_bar > t)
                   and t - y.created_bar <= p.max_age
                   and z.direction * (y.proximal - z.proximal) >= risk]
            if opp:
                tgt = min(opp) if z.direction == 1 else max(opp)
        out.append(LimitOrder(z.direction, t, z.proximal, stop, tgt, t + p.max_age))
    return out


def simulate(symbol: str, df: pd.DataFrame, p: SDParams, costs: Costs,
             start: pd.Timestamp | None = None) -> list[Trade]:
    label = "2R" if p.target == "2R" else "opposing zone"
    return simulate_orders(symbol, "supply_demand", df, orders(df, p), costs, p.time_stop,
                           label, start)


def active_order(df: pd.DataFrame, p: SDParams) -> tuple[Zone, LimitOrder] | None:
    """The newest fresh zone whose LIMIT order is still pending at the last closed bar."""
    a = ind.atr(df, 14).to_numpy(float)
    last = len(df) - 1
    zones = {z.created_bar: z for z in find_zones(df, p, a)}
    for od in sorted(orders(df, p, a), key=lambda x: -x.known_bar):
        z = zones[od.known_bar]
        if od.expiry_bar < last:
            break
        if z.touched_bar is None:
            return z, od
    return None


def chart_zones(df: pd.DataFrame, p: SDParams, max_items: int = 6) -> list[dict[str, Any]]:
    """Display-only fresh zones (untouched, not expired) for the chart."""
    if len(df) < 30:
        return []
    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index
    times = [int(x) for x in idx.as_unit("s").asi8]
    last = len(df) - 1
    out = []
    for z in reversed(find_zones(df, p)):
        if z.touched_bar is not None or last - z.created_bar > p.max_age:
            continue
        out.append({"dir": z.kind, "low": z.low, "high": z.high, "time": times[z.base_start],
                    "label": f"{z.kind.title()} zone (leg-out {z.strength:.1f} ATR)",
                    "icon": "▤"})
        if len(out) >= max_items:
            break
    return out
