"""ICT / smart-money concepts as fixed, deterministic rules (closed bars only).

Primitives
----------
* Swing high/low: bar i is a swing if its high (low) is the extreme of bars
  i-n..i+n. It is only KNOWN at bar i+n (confirmation) - never earlier.
* Liquidity: unswept confirmed swing highs (buy-side, "LQ") and lows (sell-side).
* Sweep: a bar trades beyond the last confirmed swing low (high) and CLOSES back
  inside it (stop run).
* Market structure shift (MSS / BOS): a close beyond the last confirmed opposite swing.
* Fair value gap (FVG): bullish when low[k] > high[k-2]; bearish when high[k] < low[k-2].
  Consequent encroachment (CE) = the gap midpoint.
* Order block (OB): the last opposite-colour candle before the displacement leg.
* Premium / discount: above / below 50% of the dealing range.
* Kill zones (intraday display only): London 07:00-10:00 UTC, New York 12:00-15:00 UTC.

Strategy (see config.yaml `strategies.ict`)
-------------------------------------------
sweep -> MSS within `mss_window` bars -> FVG between them, CE in discount (premium)
-> LIMIT entry at CE valid `fvg_expiry` bars -> stop beyond the sweep extreme + 0.1 ATR
-> target 2R or nearest opposing liquidity (>= 1R) -> time stop.
Simulation is conservative: on any bar where the stop and the target could both be
hit, the stop is assumed first; no target is taken on the fill bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import Costs, Trade
from tfx.strategies import indicators as ind

KILL_ZONES_UTC = {"London": (7, 10), "New York": (12, 15)}


@dataclass(frozen=True)
class ICTParams:
    swing_n: int = 3
    mss_window: int = 10
    fvg_expiry: int = 5
    target: str = "2R"          # "2R" | "liquidity"
    time_stop: int = 10
    stop_buffer_atr: float = 0.1
    # Real-market tolerances (0 = strict textbook). Declared before any ICT result.
    sweep_close_tol_atr: float = 0.0   # close may stay this far beyond the swept level
    pool_tol_atr: float = 0.0          # swing levels this close form one liquidity pool
    gap_tol_atr: float = 0.0           # "near FVG": wicks may overlap by up to this much
    discount_max: float = 0.5          # entry must be in the lowest X of the range (long)

    @classmethod
    def from_config(cls, strat: dict[str, Any], overrides: dict[str, Any] | None = None
                    ) -> ICTParams:
        tol = strat.get("tolerances", {})
        base = cls(mss_window=int(strat.get("mss_window", 10)),
                   time_stop=int(strat.get("time_stop", 10)),
                   sweep_close_tol_atr=float(tol.get("sweep_close_tol_atr", 0.0)),
                   pool_tol_atr=float(tol.get("pool_tol_atr", 0.0)),
                   gap_tol_atr=float(tol.get("gap_tol_atr", 0.0)),
                   discount_max=float(tol.get("discount_max", 0.5)))
        if overrides:
            base = cls(**{**base.__dict__, **{k: v for k, v in overrides.items()
                                               if k in base.__dict__}})
        return base


@dataclass
class Setup:
    direction: int
    sweep_bar: int
    sweep_price: float          # extreme of the sweep candle
    swept_level: float          # the liquidity level that was taken
    mss_bar: int
    mss_level: float            # swing level broken by the MSS
    fvg_bar: int
    fvg_low: float
    fvg_high: float
    ob_low: float | None
    ob_high: float | None
    entry: float                # CE (FVG midpoint)
    stop: float
    target: float
    range_low: float
    range_high: float
    expiry_bar: int
    quality: float              # 0-100 zone quality (displacement + gap size)
    tags: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ swings
def swings(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: is bar i a swing high / swing low (needs n bars on both sides)."""
    h, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    w = 2 * n + 1
    hi_max = pd.Series(h).rolling(w, center=True, min_periods=w).max().to_numpy()
    lo_min = pd.Series(lo).rolling(w, center=True, min_periods=w).min().to_numpy()
    return (h == hi_max) & np.isfinite(hi_max), (lo == lo_min) & np.isfinite(lo_min)


def find_setups(df: pd.DataFrame, p: ICTParams, atr: np.ndarray | None = None
                ) -> list[Setup]:
    """All ICT setups, each built ONLY from bars up to its MSS bar (no look-ahead)."""
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    a = atr if atr is not None else ind.atr(df, 14).to_numpy(float)
    sh, sl = swings(df, p.swing_n)
    conf_hi: list[tuple[int, float]] = []   # confirmed swing highs (index, price)
    conf_lo: list[tuple[int, float]] = []
    pend: dict[int, tuple[int, float, float]] = {}   # dir -> (sweep bar, extreme, level)
    out: list[Setup] = []
    for t in range(n):
        i = t - p.swing_n           # swing confirmed at t
        if i >= 0:
            if sh[i]:
                conf_hi.append((i, h[i]))
            if sl[i]:
                conf_lo.append((i, lo[i]))
        if not np.isfinite(a[t]) or not conf_hi or not conf_lo:
            continue
        last_hi, last_lo = conf_hi[-1][1], conf_lo[-1][1]
        # 1. sweeps of liquidity (close back inside)
        # liquidity pool: nearby swing levels (equal lows/highs) count as one level
        tol_p = p.pool_tol_atr * a[t]
        pool_lo = min(v for _, v in conf_lo[-3:] if abs(v - last_lo) <= tol_p)
        pool_hi = max(v for _, v in conf_hi[-3:] if abs(v - last_hi) <= tol_p)
        tol_c = p.sweep_close_tol_atr * a[t]
        if lo[t] < pool_lo and c[t] > pool_lo - tol_c:
            prev = pend.get(1)
            if prev is None or lo[t] < prev[1] or t - prev[0] > p.mss_window:
                pend[1] = (t, lo[t], pool_lo)
        if h[t] > pool_hi and c[t] < pool_hi + tol_c:
            prev = pend.get(-1)
            if prev is None or h[t] > prev[1] or t - prev[0] > p.mss_window:
                pend[-1] = (t, h[t], pool_hi)
        # 2. market structure shift after a sweep
        for d in (1, -1):
            if d not in pend:
                continue
            s, ext, level = pend[d]
            if t - s > p.mss_window:
                del pend[d]
                continue
            if t == s:
                continue
            brk = last_hi if d == 1 else last_lo
            if not ((d == 1 and c[t] > brk) or (d == -1 and c[t] < brk)):
                continue
            del pend[d]
            # 3. most recent FVG between the sweep and the MSS bar
            gap = None
            for k in range(t, s + 1, -1):
                if k - 2 < 0:
                    break
                tol_g = p.gap_tol_atr * a[t]
                if d == 1 and lo[k] > h[k - 2] - tol_g:
                    gap = (k, min(h[k - 2], lo[k]), max(h[k - 2], lo[k]))
                    break
                if d == -1 and h[k] < lo[k - 2] + tol_g:
                    gap = (k, min(h[k], lo[k - 2]), max(h[k], lo[k - 2]))
                    break
            if gap is None:
                continue
            k, g_lo, g_hi = gap
            ce = (g_lo + g_hi) / 2
            if d == 1:   # dealing range: sweep extreme to the MSS leg's extreme
                rng_lo, rng_hi = ext, float(h[s:t + 1].max())
            else:
                rng_lo, rng_hi = float(lo[s:t + 1].min()), ext
            span = rng_hi - rng_lo
            pos = (ce - rng_lo) / span if d == 1 else (rng_hi - ce) / span
            if span <= 0 or pos > p.discount_max + 1e-12:
                continue   # 4. must be in discount (long) / premium (short)
            stop = ext - d * p.stop_buffer_atr * a[t]
            risk = d * (ce - stop)
            if risk <= 0:
                continue
            tgt = ce + d * 2 * risk
            if p.target == "liquidity":
                pool = [v for _, v in (conf_hi if d == 1 else conf_lo)
                        if d * (v - ce) >= risk]
                if pool:
                    tgt = min(pool) if d == 1 else max(pool)
            # order block: last opposite candle before the displacement (s..k)
            ob = None
            for j in range(k - 1, s - 1, -1):
                if (d == 1 and c[j] < o[j]) or (d == -1 and c[j] > o[j]):
                    ob = (lo[j], h[j])
                    break
            displacement = abs(c[t] - o[s]) / a[t] if a[t] > 0 else 0.0
            quality = float(np.clip(25 * min(displacement, 3) / 3 * 2
                                    + 50 * min((g_hi - g_lo) / a[t], 1.0), 0, 100))
            out.append(Setup(d, s, ext, level, t, brk, k, g_lo, g_hi,
                             None if ob is None else ob[0], None if ob is None else ob[1],
                             ce, stop, tgt, rng_lo, rng_hi, t + p.fvg_expiry,
                             round(quality, 1)))
    return out


# --------------------------------------------------------------- simulator
def simulate(symbol: str, df: pd.DataFrame, p: ICTParams, costs: Costs,
             start: pd.Timestamp | None = None) -> list[Trade]:
    """Limit-order execution of ICT setups, one position at a time per symbol."""
    idx = df.index
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    s0 = 0 if start is None else int(idx.searchsorted(start))
    trades: list[Trade] = []
    busy_until = -1
    for st in find_setups(df, p):
        if st.mss_bar < s0 or st.mss_bar <= busy_until:
            continue
        d = st.direction
        fill_bar = None
        for t in range(st.mss_bar + 1, min(st.expiry_bar, len(df) - 1) + 1):
            if (d == 1 and lo[t] <= st.stop) or (d == -1 and h[t] >= st.stop):
                if (d == 1 and o[t] > st.entry) or (d == -1 and o[t] < st.entry):
                    fill_bar = t          # filled on the way to the stop: a loss
                break
            if (d == 1 and lo[t] <= st.entry) or (d == -1 and h[t] >= st.entry):
                fill_bar = t
                break
        if fill_bar is None:
            continue
        fill_px = min(o[fill_bar], st.entry) if d == 1 else max(o[fill_bar], st.entry)
        entry = fill_px + d * costs.half_spread
        risk = d * (entry - st.stop)
        if risk <= 0:
            continue
        t0 = Trade(symbol, "ict", d, "A", 1.0, idx[st.mss_bar], idx[fill_bar], entry, st.stop,
                   risk)
        t0.stop_path.append(st.stop)
        units = 1.0 / risk
        mark = entry
        t0.daily_r[idx[fill_bar]] = -units * costs.commission_side
        t0.r_cost += units * (costs.half_spread + costs.commission_side)
        exit_px, reason, last = None, "", fill_bar
        for t in range(fill_bar, len(df)):
            last = t
            hit_stop = (d == 1 and lo[t] <= st.stop) or (d == -1 and h[t] >= st.stop)
            if hit_stop:
                px = min(st.stop, o[t]) if t > fill_bar and d == 1 else (
                    max(st.stop, o[t]) if t > fill_bar else st.stop)
                exit_px, reason = px - d * costs.per_side_fill, "stop"
                break
            best = h[t] if d == 1 else lo[t]
            worst = lo[t] if d == 1 else h[t]
            t0.mfe_r = max(t0.mfe_r, d * (best - entry) / risk)
            t0.mae_r = min(t0.mae_r, d * (worst - entry) / risk)
            if t > fill_bar and d * (best - st.target) >= 0:
                exit_px, reason = st.target - d * costs.half_spread, "target (liquidity)" \
                    if p.target == "liquidity" else "target (2R)"
                break
            if t - fill_bar >= p.time_stop:
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
        t0.r_net = sum(t0.daily_r.values())
        t0.r_gross = t0.r_net + t0.r_cost
        trades.append(t0)
        busy_until = last
    return trades


# ------------------------------------------------------- display annotations
def chart_annotations(df: pd.DataFrame, p: ICTParams, max_items: int = 6) -> dict[str, Any]:
    """Display-only ICT objects for the chart: LQ lines, open FVGs, sweeps, MSS, OB."""
    if len(df) < 4 * p.swing_n + 10:
        return {"liquidity": [], "fvgs": [], "events": [], "zones": []}
    h, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    t = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).as_unit(
        "s").asi8 // 1
    times = [int(x) for x in t]
    sh, sl = swings(df, p.swing_n)
    last = len(df) - 1
    liq = []
    for i in np.flatnonzero(sh)[::-1]:
        if i + p.swing_n > last:
            continue
        if h[i + 1:].max(initial=-np.inf) < h[i]:          # still unswept
            liq.append({"side": "buy-side", "price": float(h[i]), "time": times[i],
                        "label": "LQ (buy-side)", "icon": "≡"})
        if len(liq) >= max_items // 2:
            break
    for i in np.flatnonzero(sl)[::-1]:
        if i + p.swing_n > last:
            continue
        if lo[i + 1:].min(initial=np.inf) > lo[i]:
            liq.append({"side": "sell-side", "price": float(lo[i]), "time": times[i],
                        "label": "LQ (sell-side)", "icon": "≡"})
        if sum(1 for x in liq if x["side"] == "sell-side") >= max_items // 2:
            break
    fvgs = []
    for k in range(last, max(2, last - 150), -1):
        if lo[k] > h[k - 2] and lo[k + 1:].min(initial=np.inf) > h[k - 2]:
            fvgs.append({"dir": "bullish", "low": float(h[k - 2]), "high": float(lo[k]),
                         "time": times[k - 1], "label": "FVG (bullish)", "icon": "▱"})
        elif h[k] < lo[k - 2] and h[k + 1:].max(initial=-np.inf) < lo[k - 2]:
            fvgs.append({"dir": "bearish", "low": float(h[k]), "high": float(lo[k - 2]),
                         "time": times[k - 1], "label": "FVG (bearish)", "icon": "▱"})
        if len(fvgs) >= max_items:
            break
    events, zones = [], []
    for st in find_setups(df, p)[-3:]:
        side = "sell-side" if st.direction == 1 else "buy-side"
        events.append({"time": times[st.sweep_bar], "price": st.sweep_price,
                       "label": f"Sweep of {side} {st.swept_level:.5g}", "icon": "⤓"
                       if st.direction == 1 else "⤒"})
        events.append({"time": times[st.mss_bar], "price": st.mss_level,
                       "label": f"MSS / BOS at {st.mss_level:.5g}", "icon": "⇧"
                       if st.direction == 1 else "⇩"})
        if st.ob_low is not None and st.ob_high is not None:
            zones.append({"low": st.ob_low, "high": st.ob_high, "time": times[st.sweep_bar],
                          "label": f"{'Demand' if st.direction == 1 else 'Supply'} OB "
                                   f"(quality {st.quality:.0f})", "icon": "▭",
                          "dir": "demand" if st.direction == 1 else "supply"})
    return {"liquidity": liq, "fvgs": fvgs, "events": events, "zones": zones,
            "kill_zones_utc": KILL_ZONES_UTC}
