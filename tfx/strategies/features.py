"""Feature frame shared by strategies, scoring, the backtester and the chart overlays."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tfx.strategies import indicators as ind


def build_features(df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """All indicators at bar t computed from bars <= t."""
    p = params or {}
    f = df[["open", "high", "low", "close"]].copy()
    channels = {*p.get("donchian_lookbacks", [20, 55, 100]), *p.get("exit_channels", [10, 20, 30]),
                200}
    for n in sorted(channels):
        f[f"dch_{n}"] = ind.donchian_high(df, n)
        f[f"dcl_{n}"] = ind.donchian_low(df, n)
    for n in (5, 20, 100, 200):
        f[f"ema_{n}"] = ind.ema(df["close"], n)
    f["atr_14"] = ind.atr(df, 14)
    f["atr_20"] = ind.atr(df, 20)
    f["adx_14"] = ind.adx(df, 14)
    f["rsi_2"] = ind.rsi(df["close"], 2)
    f["atr_pct"] = ind.rolling_percentile(f["atr_20"] / df["close"], 252)
    f["autocorr"] = ind.autocorrelation(df["close"], 60)
    f["regime"] = regime(f)
    return f


def regime(f: pd.DataFrame, shock_pct: float = 90.0, trend_adx: float = 20.0) -> pd.Series:
    """Auditable rule set: SHOCK (ATR percentile > 90), TRENDING (ADX > 20), else CHOPPY."""
    out = pd.Series("CHOPPY", index=f.index, dtype="object")
    out[f["adx_14"] > trend_adx] = "TRENDING"
    out[f["atr_pct"] > shock_pct] = "SHOCK"
    out[f["adx_14"].isna() | f["atr_pct"].isna()] = "UNKNOWN"
    return out


def weekly_bias(df: pd.DataFrame) -> pd.Series:
    """+1 / -1 / 0 weekly trend (close vs 40-week EMA AND 26-week Donchian midline),
    aligned so a daily bar only sees weeks that have already closed."""
    w = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last"}).dropna()
    e = ind.ema(w["close"], 40)
    mid = (w["high"].rolling(26).max() + w["low"].rolling(26).min()) / 2
    bias = ((w["close"] > e) & (w["close"] > mid)).astype(int) - (
        (w["close"] < e) & (w["close"] < mid)).astype(int)
    bias = bias.where(e.notna(), 0)
    # the week labelled Friday F is only known after F closes: use it from the next bar on
    daily = bias.reindex(df.index.union(bias.index)).ffill().reindex(df.index)
    return daily.shift(1).fillna(0).astype(int)
