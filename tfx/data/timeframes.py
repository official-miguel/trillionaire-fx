"""Timeframes, closed-bar rules and look-ahead-safe multi-timeframe alignment.

Conventions (all anchored on the 17:00 New York FX close)
-------------------------------------------------------
* "FX time" = New York wall clock + 7 hours, so every FX day starts at 00:00 FX
  time (17:00 New York) and daylight-saving changes are handled automatically.
* H1 / H4 bars are labelled by their UTC **open** time. H4 buckets start at
  00/04/08/12/16/20 FX time (17/21/01/05/09/13 New York).
* D1 bars are labelled by their New York **closing date** (see ``session.py``).
* W1 bars are labelled by the last session date of the week (normally Friday);
  MN1 bars by the last session date of the month.

The one rule that matters
-------------------------
A bar may be used only after :func:`available_at` — the moment it has closed
(plus the provider-settlement delay for D1/W1/MN1). :func:`align_higher_timeframe`
enforces this, and :func:`assert_no_lookahead` lets tests and the backtester
prove it.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from tfx.core_types import Timeframe as Timeframe  # re-export
from tfx.data.session import NEW_YORK, SessionCalendar

_HOURS = {Timeframe.H1: 1, Timeframe.H4: 4}


class LookAheadError(AssertionError):
    """A bar was used before it closed."""


# --------------------------------------------------------------------- FX time
def to_fx_time(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC (tz-aware) -> naive FX time (New York + 7h)."""
    return utc_index.tz_convert(NEW_YORK).tz_localize(None) + pd.Timedelta(hours=7)


def from_fx_time(fx_naive: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ny = fx_naive - pd.Timedelta(hours=7)
    return ny.tz_localize(NEW_YORK, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


# ----------------------------------------------------------------- availability
def daily_close_utc(labels: pd.DatetimeIndex, calendar: SessionCalendar) -> pd.DatetimeIndex:
    """UTC moment a D1/W1/MN1 bar (by label) has ended and settled, per the calendar anchor."""
    closes = [calendar.usable_from_utc(ts.date()) for ts in labels]
    return pd.DatetimeIndex(pd.to_datetime(closes, utc=True))


def available_at(tf: Timeframe, labels: pd.DatetimeIndex,
                 calendar: SessionCalendar | None = None) -> pd.DatetimeIndex:
    """When each bar (by label) has closed and may be used, in UTC."""
    if tf.is_intraday:
        idx = labels if labels.tz is not None else labels.tz_localize("UTC")
        return idx.tz_convert("UTC") + pd.Timedelta(hours=_HOURS[tf])
    return daily_close_utc(labels, calendar or SessionCalendar())


# ------------------------------------------------------------------- resampling
def _ohlc(groups: Any) -> pd.DataFrame:
    out = pd.DataFrame({
        "open": groups["open"].first(), "high": groups["high"].max(),
        "low": groups["low"].min(), "close": groups["close"].last(),
    })
    if "volume" in groups.obj.columns:
        out["volume"] = groups["volume"].sum(min_count=1)
    return out


def resample_daily(daily: pd.DataFrame, tf: Timeframe, calendar: SessionCalendar,
                   last_complete: date) -> pd.DataFrame:
    """D1 -> W1 or MN1. Only fully completed weeks/months are returned."""
    if tf not in (Timeframe.W1, Timeframe.MN1):
        raise ValueError(f"resample_daily cannot build {tf}")
    if daily.empty:
        return daily.copy()
    idx = pd.DatetimeIndex(daily.index)
    key = idx.to_period("W-FRI") if tf is Timeframe.W1 else idx.to_period("M")
    groups = daily.groupby(key)
    out = _ohlc(groups)
    labels = groups.apply(lambda g: g.index[-1], include_groups=False)
    out.index = pd.DatetimeIndex(labels.to_numpy(), name="date")
    # complete only if the period's final possible session has closed
    ends = [_last_session_of(p, calendar) for p in groups.groups]
    keep = [end <= last_complete for end in ends]
    return out[keep]


def _last_session_of(period: pd.Period, calendar: SessionCalendar) -> date:
    d: date = period.end_time.date()
    while not calendar.is_session_day(d):
        d = d - timedelta(days=1)
    return d


def resample_hourly(h1: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
    """H1 (UTC open-time labels) -> H4 on FX-time boundaries; only closed H4 bars."""
    if h1.empty:
        return h1.copy()
    idx = pd.DatetimeIndex(h1.index)
    idx = idx if idx.tz is not None else idx.tz_localize("UTC")
    fx = to_fx_time(idx)
    bucket_fx = fx.floor("4h")
    groups = h1.groupby(bucket_fx)
    out = _ohlc(groups)
    out.index = from_fx_time(pd.DatetimeIndex(out.index))
    out.index.name = "time"
    closed = available_at(Timeframe.H4, pd.DatetimeIndex(out.index)) <= now_utc
    return out[closed]


# ------------------------------------------------------------------- alignment
def align_higher_timeframe(
    htf: pd.DataFrame,
    htf_tf: Timeframe,
    decision_times: pd.DatetimeIndex,
    calendar: SessionCalendar | None = None,
) -> pd.DataFrame:
    """For each decision time, the latest HTF bar that had already CLOSED.

    Returns a frame indexed by ``decision_times`` with the HTF columns plus
    ``htf_label`` and ``htf_available_at``; rows with no closed HTF bar are NaN.
    """
    if decision_times.tz is None:
        raise ValueError("decision_times must be timezone-aware (UTC)")
    avail = available_at(htf_tf, pd.DatetimeIndex(htf.index), calendar)
    right = htf.copy()
    right["htf_label"] = htf.index
    right["htf_available_at"] = avail
    right = right.sort_values("htf_available_at").reset_index(drop=True)
    left = pd.DataFrame({"t": decision_times.tz_convert("UTC")})
    merged = pd.merge_asof(left, right, left_on="t", right_on="htf_available_at",
                           direction="backward", allow_exact_matches=True)
    merged.index = decision_times
    return merged.drop(columns=["t"])


def assert_no_lookahead(decision_times: pd.DatetimeIndex,
                        used_available_at: pd.Series | pd.DatetimeIndex) -> None:
    """Raise :class:`LookAheadError` if any decision used a bar not yet closed."""
    used = pd.DatetimeIndex(used_available_at)
    mask = ~pd.isna(used)
    bad = used[mask] > decision_times[mask]
    if bool(bad.any()):
        first = decision_times[mask][bad][0]
        raise LookAheadError(f"decision at {first} used a bar that closed later")


def decision_times_for(tf: Timeframe, labels: pd.DatetimeIndex,
                       calendar: SessionCalendar | None = None) -> pd.DatetimeIndex:
    """A strategy on ``tf`` decides when each of its own bars closes."""
    return available_at(tf, labels, calendar)
