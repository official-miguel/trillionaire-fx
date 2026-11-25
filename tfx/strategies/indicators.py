"""Technical indicators. Every value at bar t uses bars <= t only (no look-ahead).

Wilder smoothing (ATR, ADX, RSI) uses ``ewm(alpha=1/n, adjust=False)``.
Donchian channels are the PRIOR n-bar extremes (shifted by one bar), so "close
beyond the prior N-day high" never compares a bar with itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def donchian_high(df: pd.DataFrame, n: int) -> pd.Series:
    """Highest high of the n bars BEFORE bar t."""
    return df["high"].rolling(n, min_periods=n).max().shift(1)


def donchian_low(df: pd.DataFrame, n: int) -> pd.Series:
    return df["low"].rolling(n, min_periods=n).min().shift(1)


def rolling_percentile(s: pd.Series, window: int = 252) -> pd.Series:
    """Percentile (0-100) of the current value within the trailing window (inclusive)."""
    def pct(x: np.ndarray) -> float:
        return float((x[:-1] <= x[-1]).mean() * 100) if len(x) > 1 else np.nan

    return s.rolling(window, min_periods=max(20, window // 4)).apply(pct, raw=True)


def autocorrelation(close: pd.Series, window: int = 60, lag: int = 1) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(window, min_periods=window).corr(r.shift(lag))


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    vals = ha_open.to_numpy(copy=True)
    hc = ha_close.to_numpy()
    for i in range(1, len(vals)):
        vals[i] = (vals[i - 1] + hc[i - 1]) / 2
    ha_open = pd.Series(vals, index=df.index)
    return pd.DataFrame({"open": ha_open, "close": ha_close,
                         "high": pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1),
                         "low": pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)})
