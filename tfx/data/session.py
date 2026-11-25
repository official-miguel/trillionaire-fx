"""FX trading-session calendar.

Two daily-bar conventions are supported (``anchor``):

* ``"utc_midnight"`` (used for live data since 2026-11-23): a bar labelled D covers
  00:00-24:00 UTC on day D and is complete at 00:00 UTC on D+1 (plus a settlement
  delay). This matches the repaired / hourly-built Yahoo FX bars and the MT5 check.
* ``"ny_close"``: a bar labelled D ends at 17:00 New York on D.

Signals are evaluated only on *completed* bars; any bar dated after the last
completed session is in progress and must never be used (no look-ahead).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
NY_CLOSE = time(17, 0)

Anchor = Literal["utc_midnight", "ny_close"]


def _parse_mmdd(items: Iterable[str]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for item in items:
        m, _, d = item.partition("-")
        out.add((int(m), int(d)))
    return out


class SessionCalendar:
    """Answers "which daily bar should exist by now?" for the chosen day boundary."""

    def __init__(
        self,
        bar_complete_delay_minutes: int = 150,
        non_session_days: Iterable[str] = ("12-25", "01-01"),
        anchor: Anchor = "ny_close",
    ) -> None:
        self.delay = timedelta(minutes=bar_complete_delay_minutes)
        self._holidays = _parse_mmdd(non_session_days)
        self.anchor: Anchor = anchor

    @property
    def holidays(self) -> set[tuple[int, int]]:
        return set(self._holidays)

    def is_session_day(self, d: date) -> bool:
        return d.weekday() < 5 and (d.month, d.day) not in self._holidays

    def bar_end_utc(self, label: date) -> datetime:
        """When the bar labelled ``label`` ends (before the settlement delay), in UTC."""
        if self.anchor == "utc_midnight":
            return datetime.combine(label + timedelta(days=1), time(0, 0), tzinfo=UTC)
        return datetime.combine(label, NY_CLOSE, tzinfo=NEW_YORK).astimezone(UTC)

    def usable_from_utc(self, label: date) -> datetime:
        """When the bar labelled ``label`` may be used: its end plus the settlement delay."""
        return self.bar_end_utc(label) + self.delay

    def last_completed_session(self, now: datetime) -> date:
        """Date label of the most recent daily bar that is complete at ``now``.

        ``now`` must be timezone-aware.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if self.anchor == "utc_midnight":
            shifted = now.astimezone(UTC) - self.delay
            d = shifted.date() - timedelta(days=1)  # the UTC day that ended at 00:00
        else:
            ny = now.astimezone(NEW_YORK)
            shifted_ny = ny - self.delay  # "17:00 + delay" behaves like 17:00
            d = shifted_ny.date()
            if shifted_ny.time() < NY_CLOSE:
                d -= timedelta(days=1)
        while not self.is_session_day(d):
            d -= timedelta(days=1)
        return d

    def sessions_between(self, start_exclusive: date, end_inclusive: date) -> int:
        """Number of session days in (start, end]. 0 if start >= end."""
        if start_exclusive >= end_inclusive:
            return 0
        n = 0
        d = start_exclusive + timedelta(days=1)
        while d <= end_inclusive:
            if self.is_session_day(d):
                n += 1
            d += timedelta(days=1)
        return n

    def next_session_open(self, session: date) -> datetime:
        """When the session following ``session`` starts trading."""
        if self.anchor == "utc_midnight":
            return self.bar_end_utc(session)
        return datetime.combine(session, NY_CLOSE, tzinfo=NEW_YORK)
