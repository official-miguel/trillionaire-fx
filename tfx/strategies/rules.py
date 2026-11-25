"""Entry rules of the three core strategies, evaluated on CLOSED bars.

Each function returns a frame aligned with the feature frame:
``direction`` (+1 long, -1 short, 0 none), ``stop`` (initial stop price), ``grade``
("A", "B", "WATCH" or ""), ``confidence`` (0-1) and ``reason`` (plain English).
Orders are filled at the NEXT bar's open by the backtester; nothing here may read
bar t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

STRATEGIES = ("trend", "pullback", "mean_reversion")


@dataclass(frozen=True)
class StrategyParams:
    lookbacks: tuple[int, ...] = (20, 55, 100)
    adx_min: float = 20.0
    adx_soft_min: float = 15.0
    ema_proximity_pct: float = 1.0
    initial_stop_atr: float = 2.0
    exit_channel: int = 20
    watch_atr: float = 0.5
    mr_adx_max: float = 20.0
    mr_atr_pct_max: float = 70.0
    mr_rsi_long: float = 10.0
    mr_rsi_short: float = 90.0
    mr_stop_atr: float = 1.5
    mr_time_stop: int = 5
    shock_atr_pct: float = 90.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, strategies: dict[str, Any], overrides: dict[str, Any] | None = None
                    ) -> StrategyParams:
        t = strategies.get("trend", {})
        m = strategies.get("mean_reversion", {})
        g = strategies.get("grades", {})
        fl = strategies.get("filters", {})
        p = cls(
            lookbacks=tuple(t.get("donchian_lookbacks", (20, 55, 100))),
            adx_min=float(t.get("adx_min", 20)), adx_soft_min=float(g.get("b_adx_soft_min", 15)),
            ema_proximity_pct=float(g.get("b_ema_proximity_pct", 1.0)),
            initial_stop_atr=float(t.get("initial_stop_atr", 2.0)),
            exit_channel=int(t.get("exit_channel", 20)),
            watch_atr=float(g.get("watchlist_atr_distance", 0.5)),
            mr_adx_max=float(m.get("adx_max", 20)), mr_atr_pct_max=float(m.get("atr_pct_max", 70)),
            mr_rsi_long=float(m.get("rsi_long_below", 10)),
            mr_rsi_short=float(m.get("rsi_short_above", 90)),
            mr_stop_atr=float(m.get("stop_atr", 1.5)), mr_time_stop=int(m.get("time_stop_days", 5)),
            shock_atr_pct=float(fl.get("shock_atr_pct", 90)),
        )
        if overrides:
            p = cls(**{**p.__dict__, **{k: v for k, v in overrides.items() if k in p.__dict__}})
        return p


def _empty(f: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"direction": 0, "stop": np.nan, "grade": "", "confidence": 0.0,
                         "reason": "", "watch_level": np.nan}, index=f.index)


def trend_signals(f: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = _empty(f)
    c = f["close"]
    long_hits = sum((c > f[f"dch_{n}"]).astype(int) for n in p.lookbacks)
    short_hits = sum((c < f[f"dcl_{n}"]).astype(int) for n in p.lookbacks)
    k = len(p.lookbacks)
    above = (c > f["ema_100"]) & (c > f["ema_200"])
    below = (c < f["ema_100"]) & (c < f["ema_200"])
    near_200 = (c - f["ema_200"]).abs() / c * 100 <= p.ema_proximity_pct
    adx = f["adx_14"]
    shock = f["atr_pct"] > p.shock_atr_pct
    for d, hits, side in ((1, long_hits, above), (-1, short_hits, below)):
        a = (hits > 0) & side & (adx > p.adx_min) & ~shock
        b = (hits > 0) & ~a & ~shock & (
            (side & (adx > p.adx_soft_min) & (adx <= p.adx_min))
            | ((side | near_200) & (adx > p.adx_min) & near_200))
        sel = a | b
        out.loc[sel, "direction"] = d
        out.loc[a, "grade"] = "A"
        out.loc[b, "grade"] = "B"
        out.loc[sel, "confidence"] = hits[sel] / k
        out.loc[sel, "stop"] = c[sel] - d * p.initial_stop_atr * f["atr_20"][sel]
    lvl_l, lvl_s = f[f"dch_{p.lookbacks[0]}"], f[f"dcl_{p.lookbacks[0]}"]
    watch_l = (out["direction"] == 0) & above & ((lvl_l - c) <= p.watch_atr * f["atr_20"]) & (
        lvl_l > c)
    watch_s = (out["direction"] == 0) & below & ((c - lvl_s) <= p.watch_atr * f["atr_20"]) & (
        lvl_s < c)
    out.loc[watch_l, ["grade", "watch_level"]] = ["WATCH", np.nan]
    out.loc[watch_l, "watch_level"] = lvl_l[watch_l]
    out.loc[watch_s, "grade"] = "WATCH"
    out.loc[watch_s, "watch_level"] = lvl_s[watch_s]
    out.loc[out["direction"] == 1, "reason"] = (
        "Close broke above the prior Donchian high while above the 100/200 EMAs")
    out.loc[out["direction"] == -1, "reason"] = (
        "Close broke below the prior Donchian low while below the 100/200 EMAs")
    return out


def trend_exit(f: pd.DataFrame, p: StrategyParams) -> tuple[pd.Series, pd.Series]:
    """(long_exit, short_exit): close beyond the opposite exit channel."""
    n = p.exit_channel
    lo = f[f"dcl_{n}"] if f"dcl_{n}" in f else f["low"].rolling(n).min().shift(1)
    hi = f[f"dch_{n}"] if f"dch_{n}" in f else f["high"].rolling(n).max().shift(1)
    return f["close"] < lo, f["close"] > hi


def pullback_signals(f: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = _empty(f)
    c, o = f["close"], f["open"]
    trend_up = (c > f["ema_200"]) & (f["adx_14"] > p.adx_min)
    trend_dn = (c < f["ema_200"]) & (f["adx_14"] > p.adx_min)
    shock = f["atr_pct"] > p.shock_atr_pct
    long_ = trend_up & (f["low"] <= f["ema_20"]) & (c > f["ema_20"]) & (c > o) & ~shock
    short = trend_dn & (f["high"] >= f["ema_20"]) & (c < f["ema_20"]) & (c < o) & ~shock
    swing_lo = f["low"].rolling(5).min()
    swing_hi = f["high"].rolling(5).max()
    buf = 0.1 * f["atr_20"]
    out.loc[long_, ["direction", "grade", "confidence"]] = [1, "A", 1.0]
    out.loc[long_, "stop"] = (swing_lo - buf)[long_]
    out.loc[short, ["direction", "grade", "confidence"]] = [-1, "A", 1.0]
    out.loc[short, "stop"] = (swing_hi + buf)[short]
    out.loc[long_, "reason"] = "Uptrend pulled back to the 20 EMA and closed back up"
    out.loc[short, "reason"] = "Downtrend rallied to the 20 EMA and closed back down"
    return out


def mean_reversion_signals(f: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = _empty(f)
    c = f["close"]
    ranging = (f["adx_14"] < p.mr_adx_max) & (f["atr_pct"] < p.mr_atr_pct_max)
    long_ = ranging & (f["rsi_2"] < p.mr_rsi_long) & (c > f["ema_200"])
    short = ranging & (f["rsi_2"] > p.mr_rsi_short) & (c < f["ema_200"])
    out.loc[long_, ["direction", "grade", "confidence"]] = [1, "A", 1.0]
    out.loc[short, ["direction", "grade", "confidence"]] = [-1, "A", 1.0]
    out.loc[long_, "stop"] = (c - p.mr_stop_atr * f["atr_14"])[long_]
    out.loc[short, "stop"] = (c + p.mr_stop_atr * f["atr_14"])[short]
    out.loc[long_, "reason"] = "Quiet range: 2-day RSI deeply oversold above the 200 EMA"
    out.loc[short, "reason"] = "Quiet range: 2-day RSI deeply overbought below the 200 EMA"
    return out


def signals_for(strategy: str, f: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    if strategy == "trend":
        return trend_signals(f, p)
    if strategy == "pullback":
        return pullback_signals(f, p)
    if strategy == "mean_reversion":
        return mean_reversion_signals(f, p)
    raise ValueError(f"unknown strategy {strategy!r}")
