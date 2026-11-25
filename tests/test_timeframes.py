from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import synthetic_candles
from tfx.config import AppConfig, StrategyTimeframes
from tfx.core_types import Timeframe
from tfx.data.session import SessionCalendar
from tfx.data.timeframes import (
    LookAheadError,
    align_higher_timeframe,
    assert_no_lookahead,
    available_at,
    decision_times_for,
    from_fx_time,
    resample_daily,
    resample_hourly,
    to_fx_time,
)
from tfx.validation.registry import ValidationRegistry

CAL = SessionCalendar(150)
NOW = datetime(2026, 11, 23, 12, tzinfo=UTC)


def _hourly(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    ny = idx.tz_convert("America/New_York")
    open_ = ~(((ny.day_of_week == 4) & (ny.hour >= 17)) | (ny.day_of_week == 5)
              | ((ny.day_of_week == 6) & (ny.hour < 17)))
    idx = idx[open_]
    c = 1.1 + np.cumsum(np.full(len(idx), 1e-4))
    return pd.DataFrame({"open": c - 5e-5, "high": c + 2e-4, "low": c - 2e-4, "close": c},
                        index=idx)


# ----------------------------------------------------------------- enum
def test_timeframe_order() -> None:
    assert Timeframe.W1.is_higher_than(Timeframe.D1)
    assert not Timeframe.H1.is_higher_than(Timeframe.H4)
    assert Timeframe.H4.is_intraday and not Timeframe.D1.is_intraday


def test_fx_time_roundtrip_and_dst() -> None:
    utc = pd.DatetimeIndex(["2026-11-19 22:00", "2026-07-15 21:00"], tz="UTC")  # 17:00 NY
    fx = to_fx_time(utc)
    assert list(fx.hour) == [0, 0]  # FX day starts at the New York close, winter and summer
    assert list(from_fx_time(fx)) == list(utc)


# ---------------------------------------------------------- availability
def test_available_at_rules() -> None:
    h1 = pd.DatetimeIndex(["2026-11-20 10:00"], tz="UTC")
    assert available_at(Timeframe.H1, h1)[0] == pd.Timestamp("2026-11-20 11:00", tz="UTC")
    assert available_at(Timeframe.H4, h1)[0] == pd.Timestamp("2026-11-20 14:00", tz="UTC")
    d1 = pd.DatetimeIndex(["2026-11-20"])
    # 17:00 New York + 150 min settlement = 00:30 UTC next day (EST)
    assert available_at(Timeframe.D1, d1, CAL)[0] == pd.Timestamp("2026-11-21 00:30", tz="UTC")
    naive = pd.DatetimeIndex(["2026-11-20 10:00"])
    assert available_at(Timeframe.H1, naive)[0].tz is not None


# ------------------------------------------------------------ resampling
def test_weekly_and_monthly_only_complete_periods() -> None:
    daily = synthetic_candles("2026-09-01", "2026-11-23")
    w = resample_daily(daily, Timeframe.W1, CAL, last_complete=date(2026, 11, 20))
    assert w.index[-1] == pd.Timestamp("2026-11-20")              # last full week (Friday)
    assert pd.Timestamp("2026-11-23") not in w.index              # Monday's week not complete
    week = daily.loc["2026-11-16":"2026-11-20"]
    assert w.loc["2026-11-20", "open"] == week["open"].iloc[0]
    assert w.loc["2026-11-20", "high"] == week["high"].max()
    assert w.loc["2026-11-20", "close"] == week["close"].iloc[-1]
    m = resample_daily(daily, Timeframe.MN1, CAL, last_complete=date(2026, 11, 20))
    assert m.index[-1] == pd.Timestamp("2026-10-30")              # November not finished
    assert resample_daily(daily.iloc[0:0], Timeframe.W1, CAL, date(2026, 11, 20)).empty
    with pytest.raises(ValueError):
        resample_daily(daily, Timeframe.H4, CAL, date(2026, 11, 20))


def test_week_ending_on_holiday_completes_on_last_session() -> None:
    daily = synthetic_candles("2026-12-14", "2026-12-24")  # Fri 25 Dec is a holiday
    w = resample_daily(daily, Timeframe.W1, CAL, last_complete=date(2026, 12, 24))
    assert w.index[-1] == pd.Timestamp("2026-12-24")


def test_h4_buckets_follow_ny_close_and_drop_open_bar() -> None:
    h1 = _hourly("2026-11-16", "2026-11-21")
    now = pd.Timestamp("2026-11-20 15:30", tz="UTC")
    h4 = resample_hourly(h1[h1.index < now], now)
    fx_hours = set(to_fx_time(pd.DatetimeIndex(h4.index)).hour)
    assert fx_hours <= {0, 4, 8, 12, 16, 20}
    assert (available_at(Timeframe.H4, pd.DatetimeIndex(h4.index)) <= now).all()
    first = h4.index[0]
    members = h1[(h1.index >= first) & (h1.index < first + pd.Timedelta(hours=4))]
    assert h4.iloc[0]["high"] == members["high"].max()
    assert resample_hourly(h1.iloc[0:0], now).empty


# ------------------------------------------------------------- alignment
def test_htf_bias_only_from_closed_bars() -> None:
    daily = synthetic_candles("2026-10-01", "2026-11-20")
    h4_labels = pd.date_range("2026-11-02", "2026-11-21", freq="4h", tz="UTC")
    decisions = decision_times_for(Timeframe.H4, h4_labels)
    aligned = align_higher_timeframe(daily, Timeframe.D1, decisions, CAL)
    assert_no_lookahead(decisions, aligned["htf_available_at"])
    # On Wed 18 Nov 20:00 UTC (15:00 NY) the 18 Nov daily bar has NOT closed yet.
    t = pd.Timestamp("2026-11-18 20:00", tz="UTC")
    assert aligned.loc[t, "htf_label"] == pd.Timestamp("2026-11-17")


def test_injected_lookahead_is_caught() -> None:
    """A naive join by calendar date (a classic bug) must fail the look-ahead check."""
    h4_labels = pd.date_range("2026-11-02", "2026-11-21", freq="4h", tz="UTC")
    decisions = decision_times_for(Timeframe.H4, h4_labels)
    # BUG: use the daily bar dated the same calendar day as the decision
    same_day = pd.DatetimeIndex(decisions.tz_convert("UTC").tz_localize(None).normalize())
    used = available_at(Timeframe.D1, same_day, CAL)
    with pytest.raises(LookAheadError):
        assert_no_lookahead(decisions, used)


def test_alignment_requires_aware_times_and_handles_no_history() -> None:
    daily = synthetic_candles("2026-11-02", "2026-11-20")
    with pytest.raises(ValueError):
        align_higher_timeframe(daily, Timeframe.D1, pd.DatetimeIndex(["2026-11-03"]), CAL)
    early = pd.DatetimeIndex(["2026-10-01 00:00"], tz="UTC")
    out = align_higher_timeframe(daily, Timeframe.D1, early, CAL)
    assert pd.isna(out["close"].iloc[0])  # nothing had closed yet -> no bias, never a peek


def test_weekly_bias_for_daily_strategy() -> None:
    daily = synthetic_candles("2026-08-03", "2026-11-20")
    weekly = resample_daily(daily, Timeframe.W1, CAL, date(2026, 11, 20))
    decisions = decision_times_for(Timeframe.D1, pd.DatetimeIndex(daily.index), CAL)
    aligned = align_higher_timeframe(weekly, Timeframe.W1, decisions, CAL)
    assert_no_lookahead(decisions, aligned["htf_available_at"])
    thu = decisions[list(daily.index).index(pd.Timestamp("2026-11-19"))]
    assert aligned.loc[thu, "htf_label"] == pd.Timestamp("2026-11-13")  # previous week
    fri = decisions[list(daily.index).index(pd.Timestamp("2026-11-20"))]
    assert aligned.loc[fri, "htf_label"] == pd.Timestamp("2026-11-20")  # closes together


# ---------------------------------------------------------------- config
def test_strategy_timeframes_from_config(cfg: AppConfig) -> None:
    st = cfg.strategy_timeframes()
    assert st["trend"].candidates == [Timeframe.D1, Timeframe.H4]
    assert st["pullback"].bias is Timeframe.D1
    assert st["mean_reversion"].candidates == [Timeframe.D1, Timeframe.H4, Timeframe.H1]
    assert st["carry"].rebalance is Timeframe.MN1 and st["xs_momentum"].candidates == [
        Timeframe.W1]
    assert all(s.execution is (Timeframe.D1 if name == "supply_demand" else Timeframe.H1)
               for name, s in st.items())   # S&D fills its limit on daily bars


@pytest.mark.parametrize("bad", [
    {"candidates": ["D1"], "default": "H4"},
    {"candidates": ["D1", "D1"], "default": "D1"},
    {"candidates": ["D1"], "default": "D1", "bias": "H4"},
    {"candidates": ["H4"], "default": "H4", "execution": "D1"},
    {"candidates": ["W1"], "default": "W1", "rebalance": "D1"},
    {"candidates": [], "default": "D1"},
])
def test_invalid_timeframe_declarations_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StrategyTimeframes.model_validate(bad)


# -------------------------------------------------------------- registry
@pytest.fixture
def reg() -> ValidationRegistry:
    r = ValidationRegistry(":memory:")
    yield r  # type: ignore[misc]
    r.close()


def test_unvalidated_timeframe_never_signals(reg: ValidationRegistry) -> None:
    d = reg.gate("trend", Timeframe.H4, NOW)
    assert not d.allowed and "not validated" in d.reason


def test_fail_and_expired_pass_are_locked(reg: ValidationRegistry) -> None:
    with pytest.raises(ValueError):
        reg.record_result("trend", Timeframe.D1, "FAIL", "r1", NOW)  # FAIL needs reasons
    with pytest.raises(ValueError):
        reg.record_result("trend", Timeframe.D1, "MAYBE", "r1", NOW)  # type: ignore[arg-type]
    reg.record_result("trend", Timeframe.D1, "FAIL", "r1", NOW, ["PBO 0.31 > 0.05"])
    d = reg.gate("trend", Timeframe.D1, NOW)
    assert not d.allowed and "PBO 0.31" in d.reason
    reg.record_result("trend", Timeframe.D1, "PASS", "r2", NOW - timedelta(days=60),
                      metrics={"sharpe": 0.6})
    assert not reg.gate("trend", Timeframe.D1, NOW).allowed  # newest validation is the FAIL
    reg.record_result("pullback", Timeframe.D1, "PASS", "r3", NOW - timedelta(days=41))
    d = reg.gate("pullback", Timeframe.D1, NOW)
    assert not d.allowed and "expired" in d.reason
    reg.record_result("pullback", Timeframe.D1, "PASS", "r4", NOW - timedelta(days=3))
    assert reg.gate("pullback", Timeframe.D1, NOW).allowed


def test_timeframe_menu_marks_unvalidated(reg: ValidationRegistry) -> None:
    reg.record_result("trend", Timeframe.D1, "PASS", "r1", NOW)
    menu = reg.timeframe_menu("trend", [Timeframe.D1, Timeframe.H4], NOW)
    assert [(m.timeframe, m.validated, m.label) for m in menu] == [
        (Timeframe.D1, True, "D1"), (Timeframe.H4, False, "H4 (not validated)")]


def test_trials_counted_across_timeframes(reg: ValidationRegistry) -> None:
    for tf in (Timeframe.D1, Timeframe.H4):
        for stop in (1.5, 2.0, 2.5):
            reg.record_trial("run1", "trend", tf, {"stop_atr": stop}, NOW)
    reg.record_trial("run2", "trend", Timeframe.D1, {"stop_atr": 2.0}, NOW)  # repeat
    reg.record_trial("run1", "mean_reversion", Timeframe.H1, {"rsi": 10}, NOW)
    assert reg.trial_count("trend") == 6
    assert reg.trial_count() == 7


def test_registry_file_backed(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "v" / "reg.sqlite"
    r = ValidationRegistry(p)
    r.record_result("trend", Timeframe.D1, "PASS", "r", NOW)
    r.close()
    r2 = ValidationRegistry(f"sqlite:///{p.as_posix()}")
    assert r2.gate("trend", Timeframe.D1, NOW).allowed
    r2.close()
