from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import MutableClock
from tfx.config import AppConfig
from tfx.core_types import Timeframe
from tfx.data.cache import CandleCache
from tfx.data.intraday import IntradayService, clean_hourly
from tfx.data.providers import ProviderError, YFinanceProvider
from tfx.data.quality import DataQualityError
from tfx.data.timeframes import available_at

NOW = datetime(2026, 11, 20, 15, 30, tzinfo=UTC)


def hourly(start: str, end: str, seed: int = 3) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    ny = idx.tz_convert("America/New_York")
    open_ = ~(((ny.day_of_week == 4) & (ny.hour >= 17)) | (ny.day_of_week == 5)
              | ((ny.day_of_week == 6) & (ny.hour < 17)))
    idx = idx[open_]
    rng = np.random.default_rng(seed)
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(idx))))
    o = np.concatenate([[1.1], c[:-1]])
    w = np.abs(rng.normal(0, 0.0005, len(idx)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w,
                         "close": c, "volume": 100.0, "spread_points": 8.0}, index=idx)


class FakeHourly:
    def __init__(self, name: str, frame: pd.DataFrame | None = None, fail: bool = False,
                 up: bool = True) -> None:
        self.name, self.frame, self.fail, self.up = name, frame, fail, up
        self.starts: list[date] = []
        self.last_error = "terminal not ready" if not up else None

    def available(self) -> bool:
        return self.up

    def fetch_hourly(self, ticker: str, start: date) -> pd.DataFrame:
        self.starts.append(start)
        if self.fail:
            raise ProviderError(f"{self.name} down")
        assert self.frame is not None
        return self.frame[self.frame.index >= pd.Timestamp(start, tz="UTC")]


H = hourly("2026-10-26", "2026-11-21")


def _svc(cfg: AppConfig, cache: CandleCache, **providers: Any) -> IntradayService:
    return IntradayService(cfg, cache, providers, MutableClock(NOW))


def test_clean_hourly_keeps_only_closed_valid_bars() -> None:
    raw = H.copy()
    raw.iloc[5, 0] = np.nan
    raw.iloc[6, 3] = -1
    raw = pd.concat([raw, raw.iloc[[10]]])
    df = clean_hourly(raw, NOW)
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (df.index + pd.Timedelta(hours=1) <= pd.Timestamp(NOW)).all()  # no open bar
    assert df.index[-1] == pd.Timestamp("2026-11-20 14:00", tz="UTC")
    assert len(df) == len(H[H.index + pd.Timedelta(hours=1) <= pd.Timestamp(NOW)]) - 2
    naive = H.iloc[:5].copy()
    naive.index = naive.index.tz_localize(None)
    assert clean_hourly(naive.drop(columns=["spread_points"]), NOW).index.tz is not None


@pytest.mark.parametrize("bad", [pd.DataFrame(), "html", H[["open", "close"]],
                                 H[H.index >= pd.Timestamp(NOW)]])
def test_clean_hourly_rejects_bad_input(bad: Any) -> None:
    with pytest.raises(DataQualityError):
        clean_hourly(bad, NOW)


def test_confirmed_when_sources_agree(cfg: AppConfig, cache: CandleCache) -> None:
    svc = _svc(cfg, cache, yfinance=FakeHourly("yfinance", H), mt5=FakeHourly("mt5", H))
    out = svc.refresh("EURUSD")
    assert out["yfinance"].startswith("ok") and out["mt5"].startswith("ok")
    d = svc.bars("EURUSD", Timeframe.H1)
    assert d.signal_allowed and d.latest_bar == pd.Timestamp("2026-11-20 14:00", tz="UTC")
    assert d.confirmed_pct is not None and d.confirmed_pct > 90
    h4 = svc.bars("EURUSD", Timeframe.H4)
    assert h4.signal_allowed
    assert (available_at(Timeframe.H4, pd.DatetimeIndex(h4.bars.index))
            <= pd.Timestamp(NOW)).all()
    assert len(cache.raw_fetches("EURUSD@H1")) == 2  # raw responses logged immutably


def test_disagreement_blocks_latest_bar(cfg: AppConfig, cache: CandleCache) -> None:
    ref = H.copy()
    last_closed = pd.Timestamp("2026-11-20 14:00", tz="UTC")
    ref.loc[last_closed, "close"] += 0.01  # ~ many ATRs of an hourly bar
    svc = _svc(cfg, cache, yfinance=FakeHourly("yfinance", H), mt5=FakeHourly("mt5", ref))
    svc.refresh("EURUSD")
    d = svc.bars("EURUSD")
    assert not d.signal_allowed and "disagree" in d.message


def test_no_mt5_means_unconfirmed_never_fallback(cfg: AppConfig, cache: CandleCache) -> None:
    svc = _svc(cfg, cache, yfinance=FakeHourly("yfinance", H), mt5=FakeHourly("mt5", up=False))
    out = svc.refresh("EURUSD")
    assert out["mt5"].startswith("unavailable")
    d = svc.bars("EURUSD")
    assert not d.signal_allowed and "MetaTrader 5" in d.message
    assert not d.bars["confirmed"].any()


def test_reference_missing_latest_bar(cfg: AppConfig, cache: CandleCache) -> None:
    ref = H[H.index < pd.Timestamp("2026-11-20 14:00", tz="UTC")]
    svc = _svc(cfg, cache, yfinance=FakeHourly("yfinance", H), mt5=FakeHourly("mt5", ref))
    svc.refresh("EURUSD")
    d = svc.bars("EURUSD")
    assert not d.signal_allowed and "no comparable" in d.message


def test_only_history_both_sources_cover_is_confirmed(cfg: AppConfig,
                                                      cache: CandleCache) -> None:
    long_mt5 = hourly("2026-09-01", "2026-11-21")
    short_yahoo = long_mt5[long_mt5.index >= pd.Timestamp("2026-11-02", tz="UTC")]
    svc = _svc(cfg, cache, yfinance=FakeHourly("yfinance", short_yahoo),
               mt5=FakeHourly("mt5", long_mt5))
    svc.refresh("EURUSD")
    d = svc.bars("EURUSD")
    assert d.confirmed_from is not None
    assert d.confirmed_from >= pd.Timestamp("2026-11-02", tz="UTC")


def test_incremental_refresh_and_failures(cfg: AppConfig, cache: CandleCache) -> None:
    yf = FakeHourly("yfinance", H)
    svc = _svc(cfg, cache, yfinance=yf, mt5=FakeHourly("mt5", fail=True))
    first = svc.refresh("EURUSD")
    assert first["mt5"].startswith("failed")
    svc.refresh("EURUSD")
    assert yf.starts[-1] == date(2026, 11, 18)  # last stored bar minus 2 days
    assert any(v.status == "error" for v in cache.versions("EURUSD@H1"))
    assert _svc(cfg, cache).refresh("EURUSD") == {"yfinance": "not configured",
                                                  "mt5": "not configured"}
    with pytest.raises(ValueError):
        svc.bars("EURUSD", Timeframe.D1)
    assert not _svc(cfg, CandleCache(":memory:")).bars("GBPUSD").signal_allowed


def test_spread_by_hour_from_mt5(cfg: AppConfig, cache: CandleCache) -> None:
    ref = H.copy()
    ref["spread_points"] = np.where(pd.DatetimeIndex(ref.index).hour == 21, 40.0, 8.0)
    svc = _svc(cfg, cache, yfinance=FakeHourly("yfinance", H), mt5=FakeHourly("mt5", ref))
    assert svc.spread_by_hour("EURUSD").empty
    svc.refresh("EURUSD")
    by_hour = svc.spread_by_hour("EURUSD")
    assert by_hour.loc[21] == 40.0 and by_hour.loc[12] == 8.0  # rollover hour is wide


def test_yfinance_hourly_normalises_and_caps_history() -> None:
    seen: list[date] = []

    def fn(t: str, s: date) -> pd.DataFrame:
        seen.append(s)
        f = H.iloc[:3].rename(columns=str.title)
        f.index = f.index.tz_convert("Europe/London")
        return f

    out = YFinanceProvider(hourly_fn=fn).fetch_hourly("EURUSD=X", date(2005, 1, 1))
    assert str(out.index.tz) == "UTC" and "spread_points" not in out.columns
    assert (datetime.now(UTC).date() - seen[0]).days <= 729
    with pytest.raises(ProviderError):
        YFinanceProvider(hourly_fn=lambda t, s: pd.DataFrame()).fetch_hourly("X", date.today())
    with pytest.raises(ProviderError):
        YFinanceProvider(hourly_fn=lambda t, s: (_ for _ in ()).throw(ValueError("x"))
                         ).fetch_hourly("X", date.today())
