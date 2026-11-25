from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tests.conftest import FakeProvider, MutableClock, ny, synthetic_candles
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.service import NO_SIGNAL_HEADLINE, DataStatus, MarketDataService

FULL = synthetic_candles("2003-01-01", "2026-11-24")  # includes 2 in-progress days
FRIDAY_EVENING = ny(2026, 11, 20, 21, 0)               # Fri 20 Nov bar complete


def _svc(cfg: AppConfig, cache: CandleCache, clock: MutableClock,
         **providers: FakeProvider) -> MarketDataService:
    """Unless a test says otherwise, MT5 serves the same data as a second source."""
    providers.setdefault("mt5", FakeProvider("mt5", FULL))
    return MarketDataService(cfg, cache, providers, clock=clock)


def test_primary_provider_full_download_is_fresh(cfg: AppConfig, cache: CandleCache) -> None:
    yf = FakeProvider("yfinance", FULL)
    svc = _svc(cfg, cache, MutableClock(FRIDAY_EVENING), yfinance=yf)
    md = svc.get("eurusd")
    assert md.status is DataStatus.FRESH and md.signal_allowed
    assert md.last_bar == date(2026, 11, 20) == md.expected_session
    assert md.source == "yfinance" and md.data_version and md.headline == "DATA OK"
    assert yf.calls == [("EURUSD=X", cfg.data.history_start)]
    # No-look-ahead: bars after the last completed session never reach consumers.
    assert md.candles.index.max() == pd.Timestamp("2026-11-20")


def test_fallback_when_primary_fails(cfg: AppConfig, cache: CandleCache) -> None:
    svc = _svc(cfg, cache, MutableClock(FRIDAY_EVENING),
               yfinance=FakeProvider("yfinance", fail=True), mt5=FakeProvider("mt5", FULL))
    res = svc.refresh("EURUSD")
    assert res.ok and res.source == "mt5" and res.mode == "full"
    assert res.errors == ["yfinance: yfinance down"]
    assert [v.status for v in cache.versions("EURUSD")] == ["ok", "error"]


def test_gold_futures_proxy_confirmed_by_mt5_spot(cfg: AppConfig, cache: CandleCache) -> None:
    from tfx.data.providers import Alignment

    yf, mt5 = FakeProvider("yfinance", FULL), FakeProvider("mt5", FULL)
    md = _svc(cfg, cache, MutableClock(FRIDAY_EVENING), yfinance=yf, mt5=mt5).get("XAUUSD")
    assert md.source == "yfinance" and md.is_proxy and "futures proxy" in md.message
    assert md.check is not None and md.check.reference_source == "mt5"
    # GC=F bars end at the New York close, so the MT5 reference uses D1 NY-close bars
    assert mt5.alignments == [Alignment.NY_CLOSE]
    assert md.signal_allowed


def test_gold_from_yahoo_flagged_as_proxy(cfg: AppConfig, cache: CandleCache) -> None:
    md = _svc(cfg, cache, MutableClock(FRIDAY_EVENING), yfinance=FakeProvider("yfinance", FULL),
              mt5=FakeProvider("mt5", fail=True)).get("XAUUSD")
    assert md.is_proxy and "proxy" in md.message
    assert md.status is DataStatus.UNVERIFIED  # no second source left to confirm it


def test_all_fail_no_cache_is_unavailable(cfg: AppConfig, cache: CandleCache) -> None:
    md = _svc(cfg, cache, MutableClock(FRIDAY_EVENING),
              yfinance=FakeProvider("yfinance", fail=True),
              mt5=FakeProvider("mt5", fail=True)).get("EURUSD")
    assert md.status is DataStatus.UNAVAILABLE and not md.signal_allowed
    assert md.message.startswith(NO_SIGNAL_HEADLINE) and md.headline == NO_SIGNAL_HEADLINE
    assert md.candles.empty


def test_offline_with_fresh_cache_still_signals(cfg: AppConfig, cache: CandleCache) -> None:
    clock = MutableClock(FRIDAY_EVENING)
    _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", FULL)).refresh("EURUSD")
    clock.now = ny(2026, 11, 22, 10)  # Sunday, offline; Friday is still the latest session
    md = _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", fail=True)).get("EURUSD")
    assert md.status is DataStatus.FRESH


def test_offline_with_old_cache_is_stale_never_current(cfg: AppConfig, cache: CandleCache) -> None:
    clock = MutableClock(FRIDAY_EVENING)
    _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", FULL)).refresh("EURUSD")
    clock.now = ny(2026, 11, 24, 20, 0)  # Tue evening: Mon + Tue sessions missing
    md = _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", fail=True),
              mt5=FakeProvider("mt5", fail="HTTP 503")).get("EURUSD")
    assert md.status is DataStatus.STALE and not md.signal_allowed
    assert md.lag_sessions == 2
    assert md.message.startswith(NO_SIGNAL_HEADLINE)
    assert not md.candles.empty  # history still chartable, flagged as not current


def test_up_to_date_cache_skips_network(cfg: AppConfig, cache: CandleCache) -> None:
    yf = FakeProvider("yfinance", FULL)
    svc = _svc(cfg, cache, MutableClock(FRIDAY_EVENING), yfinance=yf)
    svc.refresh("EURUSD")
    res = svc.refresh("EURUSD")
    assert res.mode == "up-to-date" and len(yf.calls) == 1


def test_incremental_update_appends(cfg: AppConfig, cache: CandleCache) -> None:
    clock = MutableClock(FRIDAY_EVENING)
    yf = FakeProvider("yfinance", FULL)
    svc = _svc(cfg, cache, clock, yfinance=yf)
    svc.refresh("EURUSD")
    clock.now = ny(2026, 11, 23, 20)  # Monday bar now complete
    res = svc.refresh("EURUSD")
    assert res.mode == "incremental"
    assert yf.calls[-1][1] == date(2026, 11, 20) - pd.Timedelta(days=14)
    md = svc.assess("EURUSD")
    assert md.last_bar == date(2026, 11, 23)
    assert md.status is DataStatus.UNVERIFIED  # reference not yet refreshed for Monday
    svc.refresh_reference("EURUSD")
    assert svc.assess("EURUSD").status is DataStatus.FRESH


def test_revised_history_triggers_full_redownload(cfg: AppConfig, cache: CandleCache) -> None:
    clock = MutableClock(FRIDAY_EVENING)
    _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", FULL)).refresh("EURUSD")
    revised = FULL.copy()
    revised[["open", "high", "low", "close"]] *= 1.02  # 2% revision > 0.5% tolerance
    yf2 = FakeProvider("yfinance", revised)
    clock.now = ny(2026, 11, 23, 20)
    res = _svc(cfg, cache, clock, yfinance=yf2).refresh("EURUSD")
    assert res.mode == "replace" and len(yf2.calls) == 2
    assert yf2.calls[-1][1] == cfg.data.history_start
    first = cache.load("EURUSD").index[0]
    assert first == pd.Timestamp("2003-01-02")  # 1 Jan is a non-session day
    assert cache.load("EURUSD")["close"].iloc[0] == pytest.approx(revised.loc[first, "close"])


def test_source_switch_replaces_whole_series(cfg: AppConfig, cache: CandleCache) -> None:
    clock = MutableClock(FRIDAY_EVENING)
    _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", FULL)).refresh("EURUSD")
    other = synthetic_candles("2010-01-01", "2026-11-24", seed=99)
    clock.now = ny(2026, 11, 23, 20)
    res = _svc(cfg, cache, clock, yfinance=FakeProvider("yfinance", fail=True),
               mt5=FakeProvider("mt5", other)).refresh("EURUSD")
    assert res.source == "mt5" and res.mode == "replace"
    assert cache.load("EURUSD").index[0] == pd.Timestamp("2010-01-04")  # first session after 1 Jan


def test_garbage_response_is_a_failure_not_a_crash(cfg: AppConfig, cache: CandleCache) -> None:
    junk = pd.DataFrame({"foo": [1, 2]}, index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
    md = _svc(cfg, cache, MutableClock(FRIDAY_EVENING),
              yfinance=FakeProvider("yfinance", junk),
              mt5=FakeProvider("mt5", fail=True)).get("EURUSD")
    assert md.status is DataStatus.UNAVAILABLE
    assert cache.versions("EURUSD")[0].status == "error"


def test_refresh_crash_is_contained(cfg: AppConfig, cache: CandleCache) -> None:
    class Exploding(FakeProvider):
        def fetch_daily(self, ticker: str, start: date, alignment: object = None) -> pd.DataFrame:
            raise RuntimeError("boom")

    svc = _svc(cfg, cache, MutableClock(FRIDAY_EVENING), yfinance=Exploding("yfinance"),
               mt5=Exploding("mt5"))
    md = svc.get("EURUSD")
    assert md.status is DataStatus.UNAVAILABLE
    assert any("RuntimeError: boom" in (v.detail or "") for v in cache.versions("EURUSD"))
    # an unexpected crash in one provider no longer stops the fallback
    svc2 = _svc(cfg, CandleCache(":memory:"), MutableClock(FRIDAY_EVENING),
                yfinance=Exploding("yfinance"), mt5=FakeProvider("mt5", FULL))
    assert svc2.refresh("EURUSD").source == "mt5"


def test_unavailable_provider_and_unknown_symbol(cfg: AppConfig, cache: CandleCache) -> None:
    class Off(FakeProvider):
        def available(self) -> bool:
            return False

    svc = _svc(cfg, cache, MutableClock(FRIDAY_EVENING), yfinance=Off("yfinance", FULL),
               mt5=Off("mt5", FULL))
    res = svc.refresh("EURUSD")
    assert not res.ok and "no provider" in res.errors[0]
    with pytest.raises(KeyError):
        svc.get("BTCUSD")


def test_deterministic_data_version(cfg: AppConfig) -> None:
    hashes = set()
    for _ in range(2):
        c = CandleCache(":memory:")
        md = _svc(cfg, c, MutableClock(FRIDAY_EVENING),
                  yfinance=FakeProvider("yfinance", FULL)).get("EURUSD")
        hashes.add(md.data_version)
        c.close()
    assert len(hashes) == 1
