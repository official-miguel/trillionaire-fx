from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

import tfx.__main__ as cli
from tests.conftest import MutableClock
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.intraday import IntradayService
from tfx.data.providers import Alignment, ProviderError
from tfx.data.quality import clean_candles
from tfx.data.research import (
    ResearchHistory,
    expected_trading_hours,
    hourly_to_days,
    missing_runs,
)
from tfx.data.service import MarketDataService
from tfx.data.session import SessionCalendar

NOW = datetime(2026, 11, 20, 23, 0, tzinfo=UTC)
HOL = {(12, 25), (1, 1)}


def fx_hours(start: str, end: str, seed: int = 11) -> pd.DataFrame:
    idx = expected_trading_hours(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                                 HOL)
    rng = np.random.default_rng(seed)
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(idx))))
    o = np.concatenate([[1.1], c[:-1]])
    w = np.abs(rng.normal(0, 0.0004, len(idx)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w,
                         "close": c, "volume": 50.0, "spread_points": 9.0}, index=idx)


H = fx_hours("2024-11-01", "2026-11-20 21:00")   # ~2 years of research history


class FakeResearchMt5:
    name = "mt5"

    def __init__(self, frame: pd.DataFrame | None, up: bool = True) -> None:
        self.frame, self.up = frame, up
        self.calls: list[tuple[date, date]] = []
        self.last_error = None if up else "terminal not ready"

    def available(self) -> bool:
        return self.up

    def fetch_hourly_range(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((start, end))
        assert self.frame is not None
        part = self.frame[(self.frame.index >= pd.Timestamp(start, tz="UTC"))
                          & (self.frame.index < pd.Timestamp(end, tz="UTC"))]
        if part.empty:
            raise ProviderError("no hourly bars returned")
        return part

    def fetch_hourly(self, ticker: str, start: date) -> pd.DataFrame:
        return self.fetch_hourly_range(ticker, start, date(2027, 1, 1))


class FakeYahooHourly:
    name = "yfinance"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def available(self) -> bool:
        return True

    def fetch_hourly(self, ticker: str, start: date) -> pd.DataFrame:
        return self.frame[self.frame.index >= pd.Timestamp(start, tz="UTC")]


@pytest.fixture
def small_cfg(cfg: AppConfig) -> AppConfig:
    r = cfg.data.research
    r.start, r.min_years = date(2024, 11, 1), 1.5
    r.min_hourly_compared, r.min_daily_compared = 500, 200
    return cfg


def _setup(cfg: AppConfig, cache: CandleCache, research: pd.DataFrame,
           yahoo_h1: pd.DataFrame | None = None, mt5_up: bool = True
           ) -> tuple[ResearchHistory, FakeResearchMt5, IntradayService]:
    clock = MutableClock(NOW)
    yahoo_h1 = H[H.index >= pd.Timestamp("2025-06-01", tz="UTC")] if yahoo_h1 is None \
        else yahoo_h1
    mt5 = FakeResearchMt5(research, up=mt5_up)
    intraday = IntradayService(cfg, cache, {"yfinance": FakeYahooHourly(yahoo_h1)}, clock)
    intraday.refresh("EURUSD")
    daily, _ = clean_candles(hourly_to_days(H, Alignment.UTC_MIDNIGHT),
                             SessionCalendar(150), date(2026, 11, 20))
    cache.write("EURUSD", daily, "yfinance", "full", NOW)  # Yahoo daily (London days)
    return ResearchHistory(cfg, cache, mt5, intraday, clock), mt5, intraday


# ------------------------------------------------------------------ helpers
def test_expected_hours_follow_fx_week_and_holidays() -> None:
    week = expected_trading_hours(pd.Timestamp("2026-11-15 00:00", tz="UTC"),
                                  pd.Timestamp("2026-11-22 00:00", tz="UTC"), HOL)
    assert len(week) == 120  # Sun 17:00 to Fri 17:00 New York
    a, b = pd.Timestamp("2026-12-21", tz="UTC"), pd.Timestamp("2026-12-28", tz="UTC")
    with_hol, without = expected_trading_hours(a, b, HOL), expected_trading_hours(a, b, set())
    assert len(without) - len(with_hol) == 24  # the 25 Dec session is removed


def test_missing_runs() -> None:
    exp = pd.date_range("2026-11-16", periods=10, freq="h", tz="UTC")
    have = exp.delete([2, 3, 4, 8])
    assert missing_runs(exp, have) == [3, 1]
    assert missing_runs(exp, exp) == []


def test_hourly_to_days_alignments() -> None:
    london = hourly_to_days(H, Alignment.LONDON_MIDNIGHT)
    ny = hourly_to_days(H, Alignment.NY_CLOSE)
    utc = hourly_to_days(H, Alignment.UTC_MIDNIGHT)
    assert utc.loc[pd.Timestamp("2026-07-15"), "close"] == H[H.index < pd.Timestamp(
        "2026-07-16", tz="UTC")]["close"].iloc[-1]
    day = pd.Timestamp("2026-11-18")
    members = H[(H.index >= pd.Timestamp("2026-11-18", tz="UTC"))
                & (H.index < pd.Timestamp("2026-11-19", tz="UTC"))]  # London = UTC in Nov
    assert london.loc[day, "close"] == members["close"].iloc[-1]
    ny_members = H[(H.index >= pd.Timestamp("2026-11-17 22:00", tz="UTC"))
                   & (H.index < pd.Timestamp("2026-11-18 22:00", tz="UTC"))]
    assert ny.loc[day, "high"] == ny_members["high"].max()
    assert (london.index.day_of_week < 5).all()


# ---------------------------------------------------------------- verdicts
def test_clean_history_passes(small_cfg: AppConfig, cache: CandleCache) -> None:
    rh, _, _ = _setup(small_cfg, cache, H)
    assert rh.download("EURUSD").startswith("ok")
    q = rh.assess("EURUSD")
    assert q.verdict == "PASS" and q.usable, q.reasons
    assert q.coverage_pct is not None and q.coverage_pct > 99
    assert q.hourly_disagree_pct == 0.0 and q.daily_disagree_pct == 0.0
    assert q.hourly_compared >= 500 and q.daily_compared >= 200
    assert rh.usable_for_validation("EURUSD")
    assert len(cache.raw_fetches("EURUSD@H1-research")) >= 2  # raw log per year


def test_short_history_fails_with_reason(small_cfg: AppConfig, cache: CandleCache) -> None:
    small_cfg.data.research.min_years = 8
    rh, _, _ = _setup(small_cfg, cache, H)
    rh.download("EURUSD")
    q = rh.assess("EURUSD")
    assert q.verdict == "FAIL" and any("years of H1 history" in r for r in q.reasons)
    assert not rh.usable_for_validation("EURUSD")


def test_disagreement_with_yahoo_fails(small_cfg: AppConfig, cache: CandleCache) -> None:
    bad = H.copy()
    rng = np.random.default_rng(5)
    bad["close"] = bad["close"] * (1 + rng.normal(0, 0.002, len(bad)))  # noisy feed
    rh, _, _ = _setup(small_cfg, cache, bad)
    rh.download("EURUSD")
    q = rh.assess("EURUSD")
    assert q.verdict == "FAIL"
    assert q.hourly_disagree_pct is not None and q.hourly_disagree_pct > 5
    assert any("disagreement" in r for r in q.reasons)


def test_low_coverage_and_gaps_fail(small_cfg: AppConfig, cache: CandleCache) -> None:
    holey = H.drop(H.index[1000:2500])  # a long hole
    rh, _, _ = _setup(small_cfg, cache, holey)
    rh.download("EURUSD")
    q = rh.assess("EURUSD")
    assert q.verdict == "FAIL" and q.gaps >= 1 and q.longest_gap_hours >= 1000
    assert any("coverage" in r for r in q.reasons)


def test_no_data_and_mt5_offline(small_cfg: AppConfig, cache: CandleCache) -> None:
    rh, mt5, _ = _setup(small_cfg, cache, H, mt5_up=False)
    assert rh.download("EURUSD").startswith("MT5 not connected")
    q = rh.assess("EURUSD")
    assert q.verdict == "NO DATA" and not rh.usable_for_validation("EURUSD")
    assert mt5.calls == []


def test_download_skips_cached_past_years(small_cfg: AppConfig, cache: CandleCache) -> None:
    rh, mt5, _ = _setup(small_cfg, cache, H)
    rh.download("EURUSD")
    first = list(mt5.calls)
    assert [s.year for s, _ in first] == [2024, 2025, 2026]
    rh.download("EURUSD")
    second = mt5.calls[len(first):]
    # complete older years come from the cache; the last two are re-checked for revisions
    assert [s.year for s, _ in second] == [2025, 2026]


def test_latest_verdict_wins(small_cfg: AppConfig, cache: CandleCache) -> None:
    rh, _, _ = _setup(small_cfg, cache, H)
    rh.download("EURUSD")
    assert rh.assess("EURUSD").usable
    rh.clock = MutableClock(datetime(2026, 11, 21, tzinfo=UTC))
    small_cfg.data.research.min_years = 8
    rh.assess("EURUSD")
    assert not rh.usable_for_validation("EURUSD")


def test_research_history_never_confirms_live_bars(small_cfg: AppConfig,
                                                   cache: CandleCache) -> None:
    """Research data in the cache must not make a live intraday bar 'confirmed'."""
    rh, _, intraday = _setup(small_cfg, cache, H)
    rh.download("EURUSD")
    live = intraday.bars("EURUSD")
    assert not live.signal_allowed  # no live MT5 H1 stored -> unconfirmed
    assert intraday.load_h1("EURUSD", "mt5").empty


def test_cli_research_without_mt5(cfg: AppConfig, cache: CandleCache,
                                  monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    svc = MarketDataService(cfg, cache, {"mt5": FakeResearchMt5(None, up=False)},
                            clock=MutableClock(NOW))
    monkeypatch.setattr(cli, "_service", lambda: svc)
    monkeypatch.setattr(cli, "load_env", lambda: [])
    assert cli.main(["research"]) == 0
    out = capsys.readouterr().out
    assert "NO DATA" in out and "RESEARCH ONLY" in out and "Login to Trade Account" in out
