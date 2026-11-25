from __future__ import annotations

from datetime import date, datetime

import pytest

from tests.conftest import ny
from tfx.data.session import SessionCalendar

CAL = SessionCalendar(bar_complete_delay_minutes=150)  # bars complete at 19:30 NY


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (ny(2026, 11, 18, 12, 0), date(2026, 11, 17)),   # Wed midday -> Tue bar
        (ny(2026, 11, 18, 19, 29), date(2026, 11, 17)),  # just before completion
        (ny(2026, 11, 18, 19, 30), date(2026, 11, 18)),  # completion moment
        (ny(2026, 11, 18, 23, 59), date(2026, 11, 18)),
        (ny(2026, 11, 21, 10, 0), date(2026, 11, 20)),   # Saturday -> Friday
        (ny(2026, 11, 22, 18, 0), date(2026, 11, 20)),   # Sunday evening -> Friday
        (ny(2026, 11, 23, 9, 0), date(2026, 11, 20)),    # Monday morning -> Friday
        (ny(2026, 11, 24, 0, 30), date(2026, 11, 23)),   # after midnight, Mon bar done
        (ny(2026, 12, 26, 12, 0), date(2026, 12, 24)),   # Christmas is not a session
        (ny(2027, 1, 1, 21, 0), date(2026, 12, 31)),     # New Year's Day skipped
        (ny(2026, 3, 9, 20, 0), date(2026, 3, 9)),       # day after US DST change
        (ny(2026, 11, 2, 20, 0), date(2026, 11, 2)),     # day after DST ends
    ],
)
def test_last_completed_session(now: datetime, expected: date) -> None:
    assert CAL.last_completed_session(now) == expected


def test_zero_delay_uses_exact_ny_close() -> None:
    cal = SessionCalendar(bar_complete_delay_minutes=0)
    assert cal.last_completed_session(ny(2026, 11, 18, 16, 59)) == date(2026, 11, 17)
    assert cal.last_completed_session(ny(2026, 11, 18, 17, 0)) == date(2026, 11, 18)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        CAL.last_completed_session(datetime(2026, 1, 5, 12))


def test_sessions_between_skips_weekends_and_holidays() -> None:
    assert CAL.sessions_between(date(2026, 11, 20), date(2026, 11, 23)) == 1  # Fri -> Mon
    assert CAL.sessions_between(date(2026, 11, 23), date(2026, 11, 23)) == 0
    assert CAL.sessions_between(date(2026, 11, 24), date(2026, 11, 23)) == 0
    assert CAL.sessions_between(date(2026, 12, 23), date(2026, 12, 28)) == 2  # 24th, 28th
    assert CAL.sessions_between(date(2026, 11, 16), date(2026, 11, 20)) == 4


def test_is_session_day() -> None:
    assert CAL.is_session_day(date(2026, 11, 20))
    assert not CAL.is_session_day(date(2026, 11, 21))
    assert not CAL.is_session_day(date(2026, 12, 25))


def test_next_session_open_is_ny_17() -> None:
    opened = CAL.next_session_open(date(2026, 11, 20))
    assert (opened.hour, opened.minute) == (17, 0)
    assert str(opened.tzinfo) == "America/New_York"
