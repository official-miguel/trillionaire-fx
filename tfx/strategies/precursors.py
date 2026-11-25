"""Move precursors: what a chart looked like at the close BEFORE a move started.

Declared 2026-11-24 BEFORE the move study was run (config.yaml ``move_study``).
Every rule is known at the close of bar t (no look-ahead) and returns a direction
(+1 long, -1 short, 0 nothing):

* squeeze_breakout - volatility was compressed (ATR5/ATR50 in its lowest 20% over a
  year) on bar t-1, and bar t closes beyond the 10-day high/low.
* nr7_breakout     - bar t-1 had the narrowest range of the last 7 bars, and bar t
  closes beyond its high/low.
* donchian20       - close beyond the prior 20-day high/low.
* sweep_reversal   - bar t trades beyond the last confirmed swing low/high (3 bars each
  side) and closes back inside it (stop run).
* momentum_candle  - body of bar t >= 1.5 ATR(14).
* trend_discount   - EMA50 above (below) EMA200 and the close in the lowest (highest)
  25% of the 20-day range: buying dips in an up-trend.
* sd_first_touch   - first return into a fresh supply/demand zone, with the trend.

Each rule also has a ``+weekly`` version that only keeps signals in the direction of
the weekly trend. 14 rules in total; every one is counted as a trial.

A precursor trade is a market order at the next open, stop ``stop_atr`` x ATR(14)
away, target ``target_r`` x the risk, time stop ``time_stop`` bars, one position at a
time per symbol, real costs (shared limit-order simulator, stop first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import Costs, Trade
from tfx.strategies import indicators as ind
from tfx.strategies.features import weekly_bias
from tfx.strategies.ict import swings
from tfx.strategies.limit_sim import LimitOrder, simulate_orders

BASE_RULES = ("squeeze_breakout", "nr7_breakout", "donchian20", "sweep_reversal",
              "momentum_candle", "trend_discount", "sd_first_touch")
ALL_RULES = (*BASE_RULES, *(f"{r}+weekly" for r in BASE_RULES))
RULE_TEXT = {
    "squeeze_breakout": "breakout after a volatility squeeze",
    "nr7_breakout": "breakout of the narrowest bar of 7 (NR7)",
    "donchian20": "close beyond the 20-day high/low",
    "sweep_reversal": "liquidity sweep that closes back inside",
    "momentum_candle": "strong momentum candle (body >= 1.5 ATR)",
    "trend_discount": "dip into the lower 25% of the range in an up-trend (or mirror)",
    "sd_first_touch": "first return into a fresh supply/demand zone with the trend",
}


@dataclass(frozen=True)
class BracketParams:
    stop_atr: float = 1.0
    target_r: float = 3.0
    time_stop: int = 10

    @classmethod
    def from_config(cls, strat: dict[str, Any], overrides: dict[str, Any] | None = None
                    ) -> BracketParams:
        keys = cls.__dataclass_fields__
        base = {k: strat[k] for k in keys if k in strat}
        base.update({k: v for k, v in (overrides or {}).items() if k in keys})
        return cls(**base)


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Direction arrays (+1/-1/0) per rule, each value known at the close of its bar."""
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    a14 = ind.atr(df, 14).to_numpy(float)
    a5, a50 = ind.atr(df, 5), ind.atr(df, 50)
    sq_pct = ind.rolling_percentile(a5 / a50, 252).to_numpy(float)
    dch10 = ind.donchian_high(df, 10).to_numpy(float)
    dcl10 = ind.donchian_low(df, 10).to_numpy(float)
    dch20 = ind.donchian_high(df, 20).to_numpy(float)
    dcl20 = ind.donchian_low(df, 20).to_numpy(float)
    e50 = ind.ema(df["close"], 50).to_numpy(float)
    e200 = ind.ema(df["close"], 200).to_numpy(float)
    up_trend = (e50 > e200) & (np.arange(n) >= 200)
    dn_trend = (e50 < e200) & (np.arange(n) >= 200)
    rng = h - lo
    out: dict[str, np.ndarray] = {r: np.zeros(n, dtype=int) for r in BASE_RULES}

    prev_sq = np.concatenate([[False], (sq_pct <= 20)[:-1]])
    out["squeeze_breakout"] = np.where(prev_sq & (c > dch10), 1,
                                       np.where(prev_sq & (c < dcl10), -1, 0))
    nr7 = pd.Series(rng).rolling(7).min().to_numpy() == rng
    prev_nr7 = np.concatenate([[False], nr7[:-1]])
    prev_h = np.concatenate([[np.nan], h[:-1]])
    prev_l = np.concatenate([[np.nan], lo[:-1]])
    out["nr7_breakout"] = np.where(prev_nr7 & (c > prev_h), 1,
                                   np.where(prev_nr7 & (c < prev_l), -1, 0))
    out["donchian20"] = np.where(c > dch20, 1, np.where(c < dcl20, -1, 0))
    body = c - o
    out["momentum_candle"] = np.where(body >= 1.5 * a14, 1, np.where(body <= -1.5 * a14, -1, 0))
    hi20 = pd.Series(h).rolling(20).max().to_numpy()
    lo20 = pd.Series(lo).rolling(20).min().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (c - lo20) / (hi20 - lo20)
    out["trend_discount"] = np.where(up_trend & (pos <= 0.25), 1,
                                     np.where(dn_trend & (pos >= 0.75), -1, 0))
    sh, sl = swings(df, 3)
    last_hi, last_lo = np.nan, np.nan
    sweep: np.ndarray = np.zeros(n, dtype=int)
    for t in range(n):
        i = t - 3
        if i >= 0:
            if sh[i]:
                last_hi = h[i]
            if sl[i]:
                last_lo = lo[i]
        if np.isfinite(last_lo) and lo[t] < last_lo < c[t]:
            sweep[t] = 1
        elif np.isfinite(last_hi) and h[t] > last_hi > c[t]:
            sweep[t] = -1
    out["sweep_reversal"] = sweep
    from tfx.strategies.supply_demand import SDParams, find_zones
    sd: np.ndarray = np.zeros(n, dtype=int)
    for z in find_zones(df, SDParams()):
        tb = z.touched_bar
        if tb is None or z.broken_bar == tb:
            continue
        if (z.direction == 1 and up_trend[tb]) or (z.direction == -1 and dn_trend[tb]):
            sd[tb] = z.direction
    out["sd_first_touch"] = sd
    for k in out:
        out[k] = np.where(np.isfinite(a14), out[k], 0).astype(int)
    wb = weekly_bias(df).to_numpy(int)
    for r in BASE_RULES:
        out[f"{r}+weekly"] = np.where(out[r] == wb, out[r], 0)
    return out


def bracket_orders(df: pd.DataFrame, direction: np.ndarray, p: BracketParams
                   ) -> list[LimitOrder]:
    """Market entry at the next open (a limit at that open always fills on that bar)."""
    o = df["open"].to_numpy(float)
    a14 = ind.atr(df, 14).to_numpy(float)
    out = []
    for t in np.flatnonzero(direction != 0):
        if t + 1 >= len(df) or not np.isfinite(a14[t]):
            continue
        d = int(direction[t])
        # a hair through the open: the limit then ALWAYS fills at the open, even on a bar
        # that also hits the stop (counted as a loss, never silently cancelled)
        entry = o[t + 1] - d * 1e-9
        risk = p.stop_atr * a14[t]
        out.append(LimitOrder(d, int(t), entry, entry - d * risk, entry + d * p.target_r * risk,
                              int(t) + 1))
    return out


def simulate(symbol: str, df: pd.DataFrame, rules: tuple[str, ...], p: BracketParams,
             costs: Costs, start: pd.Timestamp | None = None,
             sig: dict[str, np.ndarray] | None = None) -> list[Trade]:
    """Trade the union of ``rules`` (first rule wins on a conflicting bar)."""
    s = sig if sig is not None else signals(df)
    combined: np.ndarray = np.zeros(len(df), dtype=int)
    for r in rules:
        combined = np.where(combined == 0, s[r], combined)
    return simulate_orders(symbol, "move_hunter", df, bracket_orders(df, combined, p), costs,
                           p.time_stop, f"{p.target_r:g}R", start)
