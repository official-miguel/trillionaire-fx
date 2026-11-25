"""Turn Yahoo's FX data into genuine UTC-midnight daily bars.

The problem (measured 2026-11-23)
---------------------------------
Since about 2011, Yahoo's *daily* FX rows are not real OHLC bars. A row dated D has
open ≈ close ≈ the price at about 00:00 UTC on D (a start-of-day snapshot), while its
high/low cover day D. The row for the day in progress carries the previous row's
close as a placeholder. Before about 2010 the rows were ordinary OHLC bars.

The fix (user decision: "Repaired Yahoo + MT5")
-----------------------------------------------
* **Recent history (~730 days):** aggregate Yahoo's *hourly* bars, which are proper
  OHLC, into UTC days. These are available as soon as each UTC day ends, so live
  signals are not delayed. Checked against MT5: 98.1% of days within 0.2 ATR
  (EURUSD 2025-2026).
* **Older history, "snapshot" era:** repaired bar D = (open = close_D, high_D, low_D,
  close = close_{D+1}), since close_{D+1} is the price at the end of day D.
* **Older history, pre-2010 era:** used as delivered (already OHLC).
* **Changeover:** bars within ``TRANSITION_PAD`` rows of a regime switch are dropped
  rather than guessed.

Every raw Yahoo response is still written unchanged to the immutable raw log.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

SNAPSHOT_RATIO = 0.2       # rolling median |close-open|/range below this => snapshot era
ROLL_WINDOW = 41
TRANSITION_PAD = 10        # rows dropped either side of an era switch
MIN_HOURS_PER_DAY = 12     # fewer hourly bars => partial day, not used
PRICE = ["open", "high", "low", "close"]


def is_yahoo_fx(ticker: str | None) -> bool:
    return bool(ticker) and str(ticker).upper().endswith("=X")


def _norm_daily(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)
    keep = [c for c in (*PRICE, "volume") if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        # Yahoo stamps daily rows at local (London) midnight. Keep that calendar date:
        # converting to UTC would move every summer row to the previous day.
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def snapshot_regime(daily: pd.DataFrame) -> pd.Series:
    """True where rows behave like start-of-day snapshots (close ≈ open)."""
    rng = (daily["high"] - daily["low"]).replace(0, np.nan)
    ratio = (daily["close"] - daily["open"]).abs() / rng
    med = ratio.rolling(ROLL_WINDOW, center=True, min_periods=15).median()
    return (med < SNAPSHOT_RATIO).fillna(False)


def repair_daily(raw_daily: pd.DataFrame) -> pd.DataFrame:
    """Repair snapshot-era rows, keep OHLC-era rows, drop rows near era switches."""
    d = _norm_daily(raw_daily)
    d = d[d.index.day_of_week < 5]
    if d.empty:
        return d
    snap = snapshot_regime(d)
    out = d.copy()
    nxt_close = d["close"].shift(-1)
    out.loc[snap, "open"] = d.loc[snap, "close"]
    out.loc[snap, "close"] = nxt_close[snap]
    out.loc[snap, "high"] = np.maximum(d.loc[snap, "high"],
                                       out.loc[snap, ["open", "close"]].max(axis=1))
    out.loc[snap, "low"] = np.minimum(d.loc[snap, "low"],
                                      out.loc[snap, ["open", "close"]].min(axis=1))
    # drop the transition zone around every era switch
    switch: Any = np.flatnonzero(snap.to_numpy()[1:] != snap.to_numpy()[:-1]) + 1
    drop: Any = np.zeros(len(out), dtype=bool)
    for s in switch:
        drop[max(0, s - TRANSITION_PAD):s + TRANSITION_PAD] = True
    out = out[~drop]
    # The last snapshot row has no "next close" yet, and the row before it may have borrowed
    # the in-progress row's placeholder close: neither is a complete, trustworthy bar.
    out = out.dropna(subset=PRICE)
    if len(snap) and bool(snap.iloc[-1]) and len(out):
        out = out.iloc[:-1]
    return out


def hourly_to_utc_days(hourly: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Aggregate UTC hourly bars into complete UTC days (Mon-Fri, >= 12 hours)."""
    if hourly.empty:
        return hourly.iloc[0:0][PRICE]
    h = hourly.rename(columns=str.lower)
    idx = pd.DatetimeIndex(h.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    h = h.set_axis(idx)
    h = h[h.index + pd.Timedelta(hours=1) <= pd.Timestamp(now).tz_convert("UTC")]
    key = pd.DatetimeIndex(h.index).normalize().tz_localize(None)
    g = h.groupby(key)
    days = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last(),
                         "hours": g["close"].size()})
    if "volume" in h.columns:
        days["volume"] = g["volume"].sum(min_count=1)
    end = days.index + pd.Timedelta(days=1)
    complete = end <= pd.Timestamp(now).tz_convert("UTC").tz_localize(None)
    days = days[complete & (days.index.day_of_week < 5) & (days["hours"] >= MIN_HOURS_PER_DAY)]
    return days.drop(columns=["hours"])


def build_fx_daily(raw_daily: pd.DataFrame | None, raw_hourly: pd.DataFrame | None,
                   now: datetime) -> pd.DataFrame:
    """Repaired older daily bars + hourly-built recent days (hourly wins on overlap)."""
    recent = hourly_to_utc_days(raw_hourly, now) if raw_hourly is not None else None
    older = repair_daily(raw_daily) if raw_daily is not None and not raw_daily.empty else None
    if recent is None or recent.empty:
        if older is None:
            return pd.DataFrame(columns=PRICE)
        return older
    if older is None or older.empty:
        return recent
    first_recent = recent.index[0]
    combined = pd.concat([older[older.index < first_recent], recent])
    combined.index.name = "date"
    return combined.sort_index()


def splice_agreement(raw_daily: pd.DataFrame, raw_hourly: pd.DataFrame, now: datetime,
                     limit_atr: float = 0.2) -> tuple[int, float | None]:
    """How well repaired daily bars match hourly-built days where both exist."""
    from tfx.data.crosscheck import disagreement_series

    rep = repair_daily(raw_daily)
    hd = hourly_to_utc_days(raw_hourly, now)
    dis = disagreement_series(hd, rep)
    if dis.empty:
        return 0, None
    return len(dis), float((dis <= limit_atr).mean() * 100)
