from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.data.cache import CandleCache
from tfx.data.quality import clean_candles, frame_hash
from tfx.data.session import SessionCalendar

NOW = datetime(2026, 11, 21, 12, tzinfo=UTC)


def _clean(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    df, _ = clean_candles(synthetic_candles(start, end, seed), SessionCalendar(150),
                          date(2026, 11, 20))
    return df


def test_roundtrip_and_meta(cache: CandleCache) -> None:
    df = _clean("2026-01-01", "2026-11-20")
    info = cache.write("EURUSD", df, "yfinance", "full", NOW)
    back = cache.load("EURUSD")
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    meta = cache.meta("EURUSD")
    assert meta == info
    assert meta is not None and meta.rows == len(df) and meta.sha256 == frame_hash(df)
    assert meta.last_fetch_utc == NOW


def test_load_range_and_unknown_symbol(cache: CandleCache) -> None:
    cache.write("EURUSD", _clean("2026-01-01", "2026-11-20"), "yfinance", "full", NOW)
    part = cache.load("EURUSD", start=date(2026, 11, 2), end=date(2026, 11, 6))
    assert len(part) == 5
    assert cache.load("GBPUSD").empty
    assert cache.meta("GBPUSD") is None


def test_upsert_overwrites_overlap(cache: CandleCache) -> None:
    df = _clean("2026-01-01", "2026-11-20")
    cache.write("EURUSD", df.iloc[:-5], "yfinance", "full", NOW)
    revised = df.iloc[-10:].copy()
    revised["close"] = revised["close"] * 1.001
    cache.write("EURUSD", revised, "yfinance", "incremental", NOW)
    back = cache.load("EURUSD")
    assert len(back) == len(df)
    assert back["close"].iloc[-1] == pytest.approx(revised["close"].iloc[-1])


def test_source_switch_replaces_series(cache: CandleCache) -> None:
    cache.write("XAUUSD", _clean("2026-01-01", "2026-11-20"), "yfinance", "full", NOW)
    short = _clean("2026-10-01", "2026-11-20", seed=9)
    cache.write("XAUUSD", short, "stooq", "full", NOW)
    back = cache.load("XAUUSD")
    assert len(back) == len(short)  # no spliced history from the old source
    meta = cache.meta("XAUUSD")
    assert meta is not None and meta.source == "stooq"


def test_version_log_records_success_and_failure(cache: CandleCache) -> None:
    cache.write("EURUSD", _clean("2026-01-01", "2026-11-20"), "yfinance", "full", NOW)
    cache.log_failure("EURUSD", "stooq", NOW, "HTTP 503")
    versions = cache.versions("EURUSD")
    assert [v.status for v in versions] == ["error", "ok"]
    assert versions[1].sha256 and versions[0].detail == "HTTP 503"
    assert len(cache.versions()) == 2


def test_refuses_empty_write(cache: CandleCache) -> None:
    with pytest.raises(ValueError):
        cache.write("EURUSD", _clean("2026-11-02", "2026-11-06").iloc[0:0], "x", "full", NOW)


def test_file_backed_cache_persists(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "tfx.sqlite"
    c1 = CandleCache(path)
    c1.write("EURUSD", _clean("2026-10-01", "2026-11-20"), "stooq", "full", NOW)
    c1.close()
    assert path.exists()
    for target in (path, f"sqlite:///{path.as_posix()}"):
        c = CandleCache(target)
        assert not c.load("EURUSD").empty
        c.close()
