from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.conftest import FakeProvider, MutableClock, ny, synthetic_candles
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.crosscheck import CheckStatus, atr, cross_check, disagreement_series
from tfx.data.quality import clean_candles
from tfx.data.service import NO_SIGNAL_HEADLINE, DataStatus, MarketDataService
from tfx.data.session import SessionCalendar

FULL = synthetic_candles("2003-01-01", "2026-11-20")
FRIDAY = date(2026, 11, 20)
CLOCK = ny(2026, 11, 20, 21)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_candles(df, SessionCalendar(150), FRIDAY)[0]


def _bump_last_close(df: pd.DataFrame, atr_mult: float) -> pd.DataFrame:
    out = df.copy()
    a = atr(_clean(df)).iloc[-1]
    out.iloc[-1, out.columns.get_loc("close")] += atr_mult * a
    out.iloc[-1, out.columns.get_loc("high")] += max(atr_mult, 0) * a
    return out


# ------------------------------------------------------------- pure function
def test_atr_matches_hand_computation() -> None:
    df = pd.DataFrame({"high": [10.0, 11.0, 12.0], "low": [9.0, 9.5, 10.0],
                       "close": [9.5, 10.5, 11.0]})
    # TR: [1.0, max(1.5, 1.5, 0) = 1.5, max(2.0, 1.5, 0.5) = 2.0]
    assert atr(df, 2).tolist()[1:] == [1.25, 1.75]
    assert pd.isna(atr(df, 2).iloc[0])


def test_identical_sources_verify() -> None:
    p = _clean(FULL)
    res = cross_check(p, p.iloc[-120:], "stooq", FRIDAY)
    assert res.status is CheckStatus.VERIFIED and res.latest_disagreement_atr == 0.0


def test_constant_basis_is_not_a_dispute() -> None:
    p = _clean(FULL)
    ref = p.iloc[-120:].copy()
    ref[["open", "high", "low", "close"]] += 0.05  # futures-style basis
    assert cross_check(p, ref, "stooq", FRIDAY).status is CheckStatus.VERIFIED


@pytest.mark.parametrize(("mult", "status"), [(0.15, CheckStatus.VERIFIED),
                                              (0.25, CheckStatus.DISPUTED),
                                              (-0.5, CheckStatus.DISPUTED)])
def test_disagreement_threshold(mult: float, status: CheckStatus) -> None:
    p = _clean(FULL)
    ref = _clean(_bump_last_close(FULL, mult)).iloc[-120:]
    res = cross_check(p, ref, "stooq", FRIDAY, max_disagreement_atr=0.2)
    assert res.status is status
    assert res.latest_disagreement_atr == pytest.approx(abs(mult), rel=1e-6)


def test_missing_latest_bar_is_unverified() -> None:
    p = _clean(FULL)
    res = cross_check(p, p.iloc[-120:-1], "stooq", FRIDAY)
    assert res.status is CheckStatus.UNVERIFIED and "no comparable bar" in res.message
    assert cross_check(p, p.iloc[0:0], None, FRIDAY).status is CheckStatus.UNVERIFIED


def test_reference_gap_before_latest_is_not_compared() -> None:
    p = _clean(FULL)
    ref = p.iloc[-120:].drop(p.index[-2])  # reference missing Thursday
    assert disagreement_series(p, ref).index[-1] != p.index[-1]
    assert cross_check(p, ref, "stooq", FRIDAY).status is CheckStatus.UNVERIFIED


def test_earlier_disputes_reported_but_latest_verified() -> None:
    p = _clean(FULL)
    ref = p.iloc[-120:].copy()
    ref.iloc[-5, ref.columns.get_loc("close")] *= 1.02
    res = cross_check(p, ref, "stooq", FRIDAY)
    assert res.status is CheckStatus.VERIFIED and len(res.disputed_dates) >= 1
    assert "earlier bar" in res.message
    assert disagreement_series(p.iloc[0:0], ref).empty


# -------------------------------------------------------------- integration
def _svc(cfg: AppConfig, cache: CandleCache, **providers: FakeProvider) -> MarketDataService:
    return MarketDataService(cfg, cache, providers, clock=MutableClock(CLOCK))


def test_disputed_bar_blocks_signal(cfg: AppConfig, cache: CandleCache) -> None:
    md = _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL),
              mt5=FakeProvider("mt5", _bump_last_close(FULL, 0.5))).get("EURUSD")
    assert md.status is DataStatus.DISPUTED and not md.signal_allowed
    assert md.message.startswith(NO_SIGNAL_HEADLINE) and "disagree" in md.message


def test_no_second_source_blocks_when_required(cfg: AppConfig, cache: CandleCache) -> None:
    md = _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL),
              mt5=FakeProvider("mt5", fail=True)).get("EURUSD")
    assert md.status is DataStatus.UNVERIFIED and not md.signal_allowed


def test_second_source_optional_setting(cfg: AppConfig, cache: CandleCache) -> None:
    cfg.data.cross_check.require_second_source = False
    md = _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL),
              mt5=FakeProvider("mt5", fail=True)).get("EURUSD")
    assert md.status is DataStatus.FRESH and md.check is not None
    assert md.check.status is CheckStatus.UNVERIFIED


def test_cross_check_can_be_disabled_and_system_runs(cfg: AppConfig, cache: CandleCache) -> None:
    cfg.data.cross_check.enabled = False
    st = FakeProvider("mt5", _bump_last_close(FULL, 5.0))
    md = _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL), mt5=st).get("EURUSD")
    assert md.status is DataStatus.FRESH and md.check is not None
    assert md.check.status is CheckStatus.SKIPPED and st.calls == []


def test_reference_never_same_source_as_primary(cfg: AppConfig, cache: CandleCache) -> None:
    svc = _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL),
               mt5=FakeProvider("mt5", FULL))
    svc.get("EURUSD")
    _, ref_source = cache.load_reference("EURUSD")
    assert ref_source == "mt5"
    # Primary switches to MT5: the MT5 reference can no longer verify it.
    cache.write("EURUSD", _clean(FULL), "mt5", "replace", CLOCK)
    assert svc.assess("EURUSD").status is DataStatus.UNVERIFIED
    assert svc.refresh_reference("EURUSD").source == "yfinance"
    assert svc.assess("EURUSD").status is DataStatus.FRESH


def test_reference_up_to_date_and_failures_logged(cfg: AppConfig, cache: CandleCache) -> None:
    st = FakeProvider("mt5", FULL)
    svc = _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL), mt5=st)
    svc.get("EURUSD")
    assert svc.refresh_reference("EURUSD").mode == "up-to-date" and len(st.calls) == 1
    cache2 = CandleCache(":memory:")
    svc2 = _svc(cfg, cache2, yfinance=FakeProvider("yfinance", FULL),
                mt5=FakeProvider("mt5", fail="HTTP 429"))
    svc2.get("EURUSD")
    assert any(v.detail == "reference: HTTP 429" for v in cache2.versions("EURUSD"))
    assert not svc2.refresh_reference("GBPUSD").ok  # nothing cached yet
    cache2.close()


# --------------------------------------------------------- immutable raw log
def test_raw_log_stores_responses_exactly(cfg: AppConfig, cache: CandleCache) -> None:
    _svc(cfg, cache, yfinance=FakeProvider("yfinance", FULL),
         mt5=FakeProvider("mt5", FULL)).get("EURUSD")
    raws = cache.raw_fetches("EURUSD")
    assert [r.source for r in raws] == ["yfinance", "mt5"]
    decoded = CandleCache.decode_raw(raws[0])
    assert len(decoded) == len(FULL) == raws[0].rows  # unfiltered, as received
    assert len(cache.raw_fetches()) == 2


def test_raw_log_is_immutable(cache: CandleCache) -> None:
    cache.log_raw("EURUSD", "stooq", datetime(2026, 11, 20, tzinfo=UTC), FULL.iloc[:5])
    with cache.engine.begin() as conn, pytest.raises((IntegrityError, OperationalError),
                                                     match="immutable"):
        conn.execute(text("UPDATE raw_fetches SET rows = 0"))
    with cache.engine.begin() as conn, pytest.raises((IntegrityError, OperationalError),
                                                     match="immutable"):
        conn.execute(text("DELETE FROM raw_fetches"))
    assert len(cache.raw_fetches("EURUSD")) == 1


def test_reference_write_rejects_empty(cache: CandleCache) -> None:
    with pytest.raises(ValueError):
        cache.write_reference("EURUSD", _clean(FULL).iloc[0:0], "stooq")


def test_quality_reports_gaps() -> None:
    df = _clean(FULL).drop([pd.Timestamp("2026-11-17"), pd.Timestamp("2026-11-18")])
    _, rep = clean_candles(df, SessionCalendar(150), FRIDAY)
    assert date(2026, 11, 19) in rep.gaps and "2026-11-19" in rep.as_dict()["gaps"]  # type: ignore[operator]
