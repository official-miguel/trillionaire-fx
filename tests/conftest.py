from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfx.config import AppConfig, load_config
from tfx.data.cache import CandleCache
from tfx.data.providers import ProviderError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> AppConfig:
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def cache() -> Iterator[CandleCache]:
    c = CandleCache(":memory:")
    yield c
    c.close()


def synthetic_candles(start: str, end: str, seed: int = 7, base: float = 1.1) -> pd.DataFrame:
    """Deterministic random-walk OHLC on business days."""
    idx = pd.bdate_range(start, end, name="date")
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx))))
    open_ = np.concatenate([[base], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, len(idx))) * close
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread, "close": close, "volume": 0.0},
        index=idx,
    )


class FakeProvider:
    """Serves slices of a fixed frame; can be told to fail."""

    def __init__(self, name: str, frame: pd.DataFrame | None = None,
                 fail: bool | str = False) -> None:
        self.name = name
        self.frame = frame
        self.fail = fail
        self.calls: list[tuple[str, date]] = []
        self.alignments: list[object] = []

    def available(self) -> bool:
        return True

    def fetch_daily(self, ticker: str, start: date, alignment: object = None) -> pd.DataFrame:
        self.calls.append((ticker, start))
        self.alignments.append(alignment)
        if self.fail:
            raise ProviderError(self.fail if isinstance(self.fail, str) else f"{self.name} down")
        assert self.frame is not None
        return self.frame[self.frame.index >= pd.Timestamp(start)].copy()


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def ny(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    from tfx.data.session import NEW_YORK

    return datetime(y, m, d, hh, mm, tzinfo=NEW_YORK).astimezone(UTC)


ClockFactory = Callable[[datetime], MutableClock]
