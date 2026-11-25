"""Normalisation and quality checks for daily OHLC candles.

Canonical frame: ``DatetimeIndex`` (tz-naive, midnight, name ``date``), sorted,
unique, with float columns ``open, high, low, close, volume`` (volume may be NaN).
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tfx.data.session import SessionCalendar

OHLC = ["open", "high", "low", "close"]
COLUMNS = [*OHLC, "volume"]


class DataQualityError(ValueError):
    """Raised when a provider response cannot be turned into usable candles."""


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    invalid_dropped: int = 0
    non_session_dropped: int = 0
    incomplete_dropped: int = 0
    ohlc_repaired: int = 0
    spikes: list[date] = field(default_factory=list)
    gaps: list[date] = field(default_factory=list)  # bar dates preceded by missing sessions

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duplicates_dropped": self.duplicates_dropped,
            "invalid_dropped": self.invalid_dropped,
            "non_session_dropped": self.non_session_dropped,
            "incomplete_dropped": self.incomplete_dropped,
            "ohlc_repaired": self.ohlc_repaired,
            "spikes": [d.isoformat() for d in self.spikes],
            "gaps": [d.isoformat() for d in self.gaps],
        }


def empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="date").as_unit("ns")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}, index=idx)


def clean_candles(
    raw: pd.DataFrame,
    calendar: SessionCalendar,
    last_complete: date,
    spike_flag_pct: float = 8.0,
) -> tuple[pd.DataFrame, QualityReport]:
    """Return a canonical, validated candle frame and a report of what changed.

    ``last_complete`` is the last completed session; later bars are in progress
    and are removed so no consumer can ever see an unfinished candle.
    """
    report = QualityReport(rows_in=len(raw))
    if raw is None or raw.empty:
        raise DataQualityError("provider returned no rows")

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df = df[COLUMNS]

    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().as_unit("ns")
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")

    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    report.duplicates_dropped = int(dup.sum())
    df = df[~dup]

    prices = df[OHLC]
    bad = prices.isna().any(axis=1) | (prices <= 0).any(axis=1) | ~np.isfinite(prices).all(axis=1)
    report.invalid_dropped = int(bad.sum())
    df = df[~bad]

    session_mask = np.array([calendar.is_session_day(ts.date()) for ts in df.index], dtype=bool)
    report.non_session_dropped = int((~session_mask).sum())
    df = df[session_mask]

    incomplete = df.index > pd.Timestamp(last_complete)
    report.incomplete_dropped = int(incomplete.sum())
    df = df[~incomplete]

    # Repair inconsistent extremes (common in free FX feeds): high must bound
    # open/close from above, low from below.
    hi = df[["open", "high", "close"]].max(axis=1)
    lo = df[["open", "low", "close"]].min(axis=1)
    repaired = (hi != df["high"]) | (lo != df["low"])
    report.ohlc_repaired = int(repaired.sum())
    df = df.assign(high=hi, low=lo)

    vol = df["volume"]
    if vol.notna().any() and (vol.fillna(0) == 0).all():
        df = df.assign(volume=np.nan)  # FX feeds report 0 volume: treat as unavailable

    report.gaps, report.spikes = profile_series(df, calendar, spike_flag_pct)

    if df.empty:
        raise DataQualityError("no usable rows after cleaning")
    report.rows_out = len(df)
    return df, report


def profile_series(df: pd.DataFrame, calendar: SessionCalendar,
                   spike_flag_pct: float = 8.0) -> tuple[list[date], list[date]]:
    """(gaps, spikes): bar dates preceded by missing sessions, and outsized daily moves."""
    if len(df) < 2:
        return [], []
    moves = df["close"].pct_change().abs() * 100
    spikes = [ts.date() for ts in moves[moves > spike_flag_pct].index]
    days = [ts.date() for ts in df.index]
    gaps = [cur for prev, cur in itertools.pairwise(days)
            if calendar.sessions_between(prev, cur) > 1]
    return gaps, spikes


def frame_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a canonical candle frame (data-version id)."""
    h = hashlib.sha256()
    for ts, row in zip(df.index, df[COLUMNS].itertuples(index=False), strict=True):
        vals = ",".join("nan" if pd.isna(v) else f"{float(v):.10g}" for v in row)
        h.update(f"{ts.date().isoformat()},{vals}\n".encode())
    return h.hexdigest()
