from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.data.quality import DataQualityError, clean_candles, frame_hash
from tfx.data.session import SessionCalendar

CAL = SessionCalendar(150)
LAST = date(2026, 11, 20)


def _frame(rows: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.to_datetime(dates))


def test_clean_happy_path_is_canonical() -> None:
    raw = synthetic_candles("2026-10-01", "2026-11-20")
    df, rep = clean_candles(raw, CAL, LAST)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index.name == "date"
    assert rep.rows_out == len(df) == len(raw)
    assert df["volume"].isna().all()  # all-zero FX volume -> unavailable


def test_uppercase_columns_tz_index_and_strings() -> None:
    raw = synthetic_candles("2026-11-02", "2026-11-06").rename(columns=str.title)
    raw.index = raw.index.tz_localize("Europe/London")
    raw["Close"] = raw["Close"].astype(str)
    df, _ = clean_candles(raw, CAL, LAST)
    assert df.index.tz is None
    assert df["close"].dtype == np.float64


def test_duplicates_keep_last_and_sort() -> None:
    raw = _frame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 9]},
                 ["2026-11-03", "2026-11-02", "2026-11-03"])
    df, rep = clean_candles(raw, CAL, LAST)
    assert rep.duplicates_dropped == 1
    assert list(df.index.strftime("%d")) == ["02", "03"]
    assert df.loc["2026-11-03", "close"] == 9


def test_invalid_rows_dropped() -> None:
    raw = _frame({"open": [1.0, np.nan, 1.0, 1.0], "high": [1.1, 1.1, 1.1, np.inf],
                  "low": [0.9, 0.9, -1.0, 0.9], "close": [1.0, 1.0, 1.0, 1.0]},
                 ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"])
    df, rep = clean_candles(raw, CAL, LAST)
    assert rep.invalid_dropped == 3
    assert len(df) == 1


def test_weekend_and_holiday_bars_dropped() -> None:
    raw = _frame({c: [1.0] * 4 for c in ["open", "high", "low", "close"]},
                 ["2025-12-24", "2025-12-25", "2025-12-27", "2025-12-29"])
    df, rep = clean_candles(raw, CAL, LAST)
    assert rep.non_session_dropped == 2
    assert [d.day for d in df.index] == [24, 29]


def test_in_progress_bar_removed_no_lookahead() -> None:
    raw = synthetic_candles("2026-11-16", "2026-11-24")
    df, rep = clean_candles(raw, CAL, LAST)
    assert df.index[-1].date() == LAST
    assert rep.incomplete_dropped == 2


def test_ohlc_extremes_repaired() -> None:
    raw = _frame({"open": [1.0], "high": [0.95], "low": [1.05], "close": [1.02]}, ["2026-11-02"])
    df, rep = clean_candles(raw, CAL, LAST)
    assert rep.ohlc_repaired == 1
    assert df.iloc[0]["high"] == 1.02 and df.iloc[0]["low"] == 1.0


def test_spike_flagged_not_dropped() -> None:
    raw = _frame({c: [1.0, 1.2, 1.2] for c in ["open", "high", "low", "close"]},
                 ["2026-11-02", "2026-11-03", "2026-11-04"])
    df, rep = clean_candles(raw, CAL, LAST, spike_flag_pct=8.0)
    assert rep.spikes == [date(2026, 11, 3)] and len(df) == 3
    assert rep.as_dict()["spikes"] == ["2026-11-03"]


@pytest.mark.parametrize(
    "raw",
    [
        pd.DataFrame(),
        _frame({"open": [1.0], "high": [1.0], "low": [1.0]}, ["2026-11-02"]),
        _frame({c: [np.nan] for c in ["open", "high", "low", "close"]}, ["2026-11-02"]),
        _frame({c: [1.0] for c in ["open", "high", "low", "close"]}, ["2026-11-30"]),
    ],
)
def test_unusable_inputs_raise(raw: pd.DataFrame) -> None:
    with pytest.raises(DataQualityError):
        clean_candles(raw, CAL, LAST)


def test_frame_hash_deterministic_and_sensitive() -> None:
    a, _ = clean_candles(synthetic_candles("2026-10-01", "2026-11-20"), CAL, LAST)
    b = a.copy()
    assert frame_hash(a) == frame_hash(b)
    b.iloc[3, 3] += 1e-6
    assert frame_hash(a) != frame_hash(b)
