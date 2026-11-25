"""Small shared types with no dependencies (safe to import from config)."""

from __future__ import annotations

from enum import StrEnum


class Timeframe(StrEnum):
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    @property
    def rank(self) -> int:
        return list(Timeframe).index(self)

    def is_higher_than(self, other: Timeframe) -> bool:
        return self.rank > other.rank

    @property
    def is_intraday(self) -> bool:
        return self in (Timeframe.H1, Timeframe.H4)
