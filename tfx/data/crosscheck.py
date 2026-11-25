"""Cross-source verification of daily bars.

Two free feeds rarely agree to the pip: their closing times differ slightly, and
for gold one of them is a futures proxy with a basis. So we do not compare price
*levels*. We compare each bar's close-to-close *change* in both sources and scale
the difference by the primary series' ATR(20):

    disagreement = |dClose_primary - dClose_reference| / ATR20_primary

A bad tick, a missing day spliced over, or a mislabelled bar shows up as a large
disagreement. A constant basis or a small timing offset does not.

Policy (set in ``config.yaml`` ``data.cross_check``):
* latest bar disagreement > ``max_disagreement_atr`` (default 0.2) -> DISPUTED -> no signal
* latest bar missing from the reference -> UNVERIFIED -> no signal when
  ``require_second_source`` is true
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import pandas as pd


class CheckStatus(StrEnum):
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    UNVERIFIED = "UNVERIFIED"
    SKIPPED = "SKIPPED"  # module disabled in config


@dataclass(frozen=True)
class CrossCheck:
    status: CheckStatus
    reference_source: str | None
    latest_disagreement_atr: float | None
    disputed_dates: list[date] = field(default_factory=list)
    message: str = ""


def atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple-average true range (only past and current bars; no look-ahead)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def disagreement_series(primary: pd.DataFrame, reference: pd.DataFrame,
                        atr_period: int = 20) -> pd.Series:
    """Per-date disagreement in ATR units, on dates present in both sources."""
    if primary.empty or reference.empty:
        return pd.Series(dtype="float64")
    dp = primary["close"].diff()
    dr = reference["close"].diff()
    scale = atr(primary, atr_period)
    common = dp.index.intersection(dr.index)
    out = ((dp.loc[common] - dr.loc[common]).abs() / scale.loc[common]).dropna()
    # A diff is only comparable when both sources also share the previous bar.
    prev_ok = []
    for ts in out.index:
        p_prev = _prev(primary.index, ts)
        prev_ok.append(p_prev is not None and p_prev == _prev(reference.index, ts))
    return out[prev_ok]


def _prev(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.get_loc(ts)
    assert isinstance(pos, int)
    return index[pos - 1] if pos > 0 else None


def cross_check(
    primary: pd.DataFrame,
    reference: pd.DataFrame,
    reference_source: str | None,
    latest: date,
    max_disagreement_atr: float = 0.2,
    window_bars: int = 20,
) -> CrossCheck:
    if reference.empty or reference_source is None:
        return CrossCheck(CheckStatus.UNVERIFIED, None, None, [],
                          "No second data source available to verify the latest bar.")
    dis = disagreement_series(primary, reference)
    recent = dis[dis.index > (primary.index[-window_bars - 1] if len(primary) > window_bars
                              else pd.Timestamp.min)]
    disputed = [ts.date() for ts in recent[recent > max_disagreement_atr].index]
    ts_latest = pd.Timestamp(latest)
    if ts_latest not in dis.index:
        return CrossCheck(CheckStatus.UNVERIFIED, reference_source, None, disputed,
                          f"{reference_source} has no comparable bar for {latest.isoformat()}.")
    value = float(dis.loc[ts_latest])
    if value > max_disagreement_atr:
        return CrossCheck(
            CheckStatus.DISPUTED, reference_source, value, disputed,
            f"The {latest.isoformat()} bar differs from {reference_source} by {value:.2f} ATR "
            f"(limit {max_disagreement_atr:.2f}).",
        )
    note = (f" {len(disputed)} earlier bar(s) in the last {window_bars} disagreed."
            if disputed else "")
    return CrossCheck(CheckStatus.VERIFIED, reference_source, value, disputed,
                      f"Latest bar confirmed by {reference_source} ({value:.2f} ATR).{note}")
