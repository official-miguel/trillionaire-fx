"""Repaired Yahoo FX daily bars, hourly-built days and the UTC-midnight day boundary."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import MutableClock
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.providers import ProviderError, YFinanceProvider
from tfx.data.service import SERIES_CONSTRUCTION, DataStatus, MarketDataService
from tfx.data.session import SessionCalendar
from tfx.data.yahoo_fx import (
    build_fx_daily,
    hourly_to_utc_days,
    is_yahoo_fx,
    repair_daily,
    snapshot_regime,
    splice_agreement,
)

NOW = datetime(2026, 11, 21, 1, 0, tzinfo=UTC)   # Saturday 01:00 UTC: Friday complete


def true_hourly(start: str, end: str, seed: int = 4) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    ny = idx.tz_convert("America/New_York")
    open_ = ~(((ny.day_of_week == 4) & (ny.hour >= 17)) | (ny.day_of_week == 5)
              | ((ny.day_of_week == 6) & (ny.hour < 17)))
    idx = idx[open_]
    rng = np.random.default_rng(seed)
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.0007, len(idx))))
    o = np.concatenate([[1.1], c[:-1]])
    w = np.abs(rng.normal(0, 0.0003, len(idx)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w,
                         "close": c, "volume": 0.0}, index=idx)


def true_days(h: pd.DataFrame) -> pd.DataFrame:
    return hourly_to_utc_days(h, datetime(2030, 1, 1, tzinfo=UTC))


def yahoo_snapshot_rows(days: pd.DataFrame) -> pd.DataFrame:
    """Simulate Yahoo's post-2011 daily rows: open ≈ close ≈ the start-of-day price."""
    start_price = days["open"]
    return pd.DataFrame({"Open": start_price, "High": days["high"], "Low": days["low"],
                         "Close": start_price + 1e-6, "Volume": 0}, index=days.index)


H = true_hourly("2025-06-02", "2026-11-21")
DAYS = true_days(H)
SNAP = yahoo_snapshot_rows(DAYS)


# ------------------------------------------------------------------- repair
def test_snapshot_regime_detection() -> None:
    real = DAYS.rename(columns=str.title)
    assert not snapshot_regime(DAYS).any()
    assert snapshot_regime(SNAP.rename(columns=str.lower)).iloc[50:-50].all()
    assert is_yahoo_fx("EURUSD=X") and not is_yahoo_fx("GC=F") and not is_yahoo_fx(None)
    assert len(real) == len(DAYS)


def test_repair_reconstructs_true_bars() -> None:
    rep = repair_daily(SNAP)
    common = rep.index.intersection(DAYS.index)[:-1]
    assert len(common) > 300
    np.testing.assert_allclose(rep.loc[common, "open"], DAYS.loc[common, "open"], atol=1e-5)
    # Mon-Thu: repaired close(D) = Yahoo close(D+1) = start of D+1 = true close of D
    mon_thu = common[common.day_of_week != 4]
    np.testing.assert_allclose(rep.loc[mon_thu, "close"], DAYS.loc[mon_thu, "close"],
                               atol=1e-5)
    np.testing.assert_allclose(rep.loc[mon_thu, "high"], DAYS.loc[mon_thu, "high"], atol=1e-5)
    # Known limitation: Yahoo has no weekend rows, so a repaired FRIDAY closes at the
    # Monday 00:00 UTC price (it includes Sunday-evening trading).
    fridays = common[common.day_of_week == 4]
    monday_start = DAYS["open"].shift(-1).loc[fridays]
    np.testing.assert_allclose(rep.loc[fridays, "close"], monday_start, atol=1e-5)


def test_repair_never_uses_the_placeholder_row() -> None:
    rows = SNAP.copy()
    last = rows.index[-1]
    rows.loc[last, "Close"] = rows["Close"].iloc[-2]  # in-progress placeholder = prev close
    rep = repair_daily(rows)
    assert rep.index[-1] < rows.index[-2]  # neither of the last two rows is used


def test_london_stamped_summer_rows_keep_their_date() -> None:
    rows = SNAP.loc["2026-06-01":"2026-08-31"].copy()
    rows.index = pd.DatetimeIndex(rows.index).tz_localize("Europe/London")  # BST
    rep = repair_daily(pd.concat([SNAP.loc[:"2026-05-29"].tz_localize("Europe/London"), rows]))
    mondays = rep.loc["2026-06-01":"2026-08-20"]
    assert (mondays.index.day_of_week == 0).sum() >= 10  # summer Mondays survive


def test_pre_2010_ohlc_rows_kept_and_transition_dropped() -> None:
    old = DAYS.iloc[:200].rename(columns=str.title)            # genuine OHLC era
    new = yahoo_snapshot_rows(DAYS.iloc[200:])                  # snapshot era
    rows = pd.concat([old, new])
    rep = repair_daily(rows)
    np.testing.assert_allclose(rep.loc[DAYS.index[5], "close"], DAYS["close"].iloc[5])
    switch = DAYS.index[200]
    gap = rep.index[(rep.index > DAYS.index[180]) & (rep.index < DAYS.index[220])]
    assert switch not in rep.index and len(gap) < 40  # changeover rows dropped


# ---------------------------------------------------------- hourly -> days
def test_hourly_days_only_complete_weekdays() -> None:
    now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)   # Friday mid-day
    d = hourly_to_utc_days(H, now)
    assert d.index[-1] == pd.Timestamp("2026-11-19")   # Friday not finished
    assert (d.index.day_of_week < 5).all()
    one = H[(H.index >= pd.Timestamp("2026-11-18", tz="UTC"))
            & (H.index < pd.Timestamp("2026-11-19", tz="UTC"))]
    assert d.loc["2026-11-18", "high"] == one["high"].max()
    assert d.loc["2026-11-18", "close"] == one["close"].iloc[-1]
    assert hourly_to_utc_days(H.iloc[0:0], now).empty


def test_build_prefers_hourly_and_splices_repaired_history() -> None:
    old_rows = SNAP[SNAP.index < pd.Timestamp("2026-01-01")]
    recent_h = H[H.index >= pd.Timestamp("2025-12-01", tz="UTC")]
    built = build_fx_daily(pd.concat([old_rows, SNAP[SNAP.index >= "2026-01-01"]]),
                           recent_h, NOW)
    assert built.index.is_monotonic_increasing and built.index.is_unique
    assert built.index[-1] == pd.Timestamp("2026-11-20")         # fresh: Friday
    assert built.loc["2026-03-03", "close"] == DAYS.loc["2026-03-03", "close"]
    n, pct = splice_agreement(SNAP, recent_h, NOW)
    # Fridays (and the Mondays after them) differ by the Sunday-evening move, so
    # agreement is measured and reported, not assumed: ~80-90% with this synthetic data.
    assert n > 100 and pct is not None and 70 < pct < 100
    assert build_fx_daily(None, None, NOW).empty
    assert build_fx_daily(SNAP, None, NOW).equals(repair_daily(SNAP))
    assert not build_fx_daily(None, recent_h, NOW).empty


def test_provider_builds_fx_bars_and_attaches_raw_parts() -> None:
    prov = YFinanceProvider(history_fn=lambda t, s: SNAP,
                            hourly_fn=lambda t, s: H[H.index >= pd.Timestamp(s, tz="UTC")])
    df = prov.fetch_daily("EURUSD=X", date(2024, 1, 1))
    assert set(df.attrs["raw_parts"]) == {"daily", "hourly"}
    assert df.attrs["raw_parts"]["daily"] is SNAP

    def hourly_down(t: str, s: date) -> pd.DataFrame:
        raise ValueError("rate limited")

    with pytest.raises(ProviderError):
        YFinanceProvider(history_fn=lambda t, s: SNAP, hourly_fn=hourly_down).fetch_daily(
            "EURUSD=X", date(2024, 1, 1))
    with pytest.raises(ProviderError):
        YFinanceProvider(history_fn=lambda t, s: "html", hourly_fn=lambda t, s: H).fetch_daily(
            "EURUSD=X", date(2024, 1, 1))


# ------------------------------------------------------------- UTC calendar
@pytest.mark.parametrize(("now", "expected"), [
    (datetime(2026, 11, 20, 23, 59, tzinfo=UTC), date(2026, 11, 19)),
    (datetime(2026, 11, 21, 0, 29, tzinfo=UTC), date(2026, 11, 19)),   # within delay
    (datetime(2026, 11, 21, 0, 30, tzinfo=UTC), date(2026, 11, 20)),
    (datetime(2026, 11, 23, 10, 0, tzinfo=UTC), date(2026, 11, 20)),   # Monday -> Friday
    (datetime(2026, 12, 26, 1, 0, tzinfo=UTC), date(2026, 12, 24)),    # Christmas skipped
])
def test_utc_midnight_calendar(now: datetime, expected: date) -> None:
    cal = SessionCalendar(30, anchor="utc_midnight")
    assert cal.last_completed_session(now) == expected
    assert cal.usable_from_utc(date(2026, 11, 20)) == datetime(2026, 11, 21, 0, 30, tzinfo=UTC)
    assert cal.next_session_open(date(2026, 11, 20)) == datetime(2026, 11, 21, tzinfo=UTC)


def test_config_uses_utc_midnight(cfg: AppConfig) -> None:
    assert cfg.data.day_boundary == "utc_midnight" and cfg.data.bar_complete_delay_minutes == 30


# ------------------------------------------------------ one-off rebuild
class FakeDaily:
    def __init__(self, name: str, frame: pd.DataFrame) -> None:
        self.name, self.frame, self.calls = name, frame, 0
        self.alignments: list[Any] = []

    def available(self) -> bool:
        return True

    def fetch_daily(self, ticker: str, start: date, alignment: Any = None) -> pd.DataFrame:
        self.calls += 1
        self.alignments.append(alignment)
        return self.frame[self.frame.index >= pd.Timestamp(start)]


def test_old_construction_forces_full_rebuild(cfg: AppConfig, cache: CandleCache) -> None:
    days = DAYS.copy()
    svc = MarketDataService(cfg, cache, {"yfinance": FakeDaily("yfinance", days),
                                         "mt5": FakeDaily("mt5", days)},
                            clock=MutableClock(NOW))
    cache.write("EURUSD", days.iloc[:-1] * 1.0, "yfinance", "full", NOW)  # old-style cache
    assert cache.flag("construction:EURUSD") is None
    res = svc.refresh("EURUSD")
    assert res.mode == "replace"
    assert cache.flag("construction:EURUSD") == SERIES_CONSTRUCTION
    assert svc.refresh("EURUSD").mode == "up-to-date"
    md = svc.get("EURUSD")
    assert md.status is DataStatus.FRESH
    assert cache.flag("ref_alignment:EURUSD") == "utc_midnight"


def test_raw_parts_logged_separately(cfg: AppConfig, cache: CandleCache) -> None:
    prov = YFinanceProvider(history_fn=lambda t, s: SNAP,
                            hourly_fn=lambda t, s: H[H.index >= pd.Timestamp(s, tz="UTC")])
    svc = MarketDataService(cfg, cache, {"yfinance": prov}, clock=MutableClock(NOW))
    svc.refresh("EURUSD")
    assert len(cache.raw_fetches("EURUSD@daily")) == 1
    assert len(cache.raw_fetches("EURUSD@hourly")) == 1
