"""Chart bars for every timeframe (display only): MT5 live candles with Yahoo fallback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tfx.data.chart_bars import ALL_TIMEFRAMES, INTRADAY, ChartData, _norm_yahoo
from tfx.data.mt5_provider import Mt5Provider

DT = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
      ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]


class LiveMt5:
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15 = 1, 5, 15
    TIMEFRAME_M30, TIMEFRAME_H1, TIMEFRAME_H4 = 30, 16385, 16388

    def __init__(self, up: bool = True) -> None:
        self.up = up
        self.calls: list[tuple[str, int, int]] = []

    def initialize(self, **kw: Any) -> bool:
        return self.up

    def last_error(self) -> tuple[int, str]:
        return (-6, "not signed in")

    def account_info(self) -> Any:
        return SimpleNamespace(server="X-Demo", company="X", currency="USD", trade_mode=0)

    def shutdown(self) -> None:
        pass

    def symbols_get(self) -> list[Any]:
        return [SimpleNamespace(name="EURUSD.m")]

    def symbol_select(self, s: str, e: bool) -> bool:
        return True

    def copy_rates_from_pos(self, symbol: str, tf: int, start: int, count: int) -> np.ndarray:
        self.calls.append((symbol, tf, count))
        server = pd.date_range("2026-11-20 08:00", periods=5, freq="15min")  # server time
        arr = np.zeros(5, dtype=DT)
        arr["time"] = server.as_unit("s").asi8
        arr["open"] = arr["low"] = [1.10, 1.101, 1.102, 1.103, 1.104]
        arr["high"] = arr["close"] = [1.101, 1.102, 1.103, 1.104, 1.105]
        return arr[-count:]  # like MT5: the most recent `count` bars

    def symbol_info_tick(self, symbol: str) -> Any:
        return SimpleNamespace(bid=1.1050, ask=1.1051)


def _prov(fake: LiveMt5) -> Mt5Provider:
    return Mt5Provider(module_loader=lambda: fake)


def _yahoo(rows: int = 30) -> Any:
    idx = pd.date_range("2026-11-19", periods=rows, freq="h", tz="Europe/London")
    c = np.linspace(1.1, 1.12, rows)
    df = pd.DataFrame({"Open": c, "High": c + 0.001, "Low": c - 0.001, "Close": c}, index=idx)
    return lambda ticker, tf: df


def test_timeframe_lists() -> None:
    assert ALL_TIMEFRAMES == ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
    assert "D1" not in INTRADAY


def test_mt5_bars_converted_to_utc_and_cached() -> None:
    fake = LiveMt5()
    now = [0.0]
    cd = ChartData(_prov(fake), _yahoo(), ttl_seconds=5, clock=lambda: now[0])
    bars = cd.intraday("EURUSD", "EURUSD", "EURUSD=X", "M15")
    assert bars.source == "mt5" and len(bars.frame) == 5
    # server 08:00 (New York + 7h) = 01:00 New York = 06:00 UTC in November
    assert bars.frame.index[0] == pd.Timestamp("2026-11-20 06:00", tz="UTC")
    cd.intraday("EURUSD", "EURUSD", "EURUSD=X", "M15")
    assert len(fake.calls) == 1            # served from cache within the TTL
    now[0] = 6.0
    cd.intraday("EURUSD", "EURUSD", "EURUSD=X", "M15")
    assert len(fake.calls) == 2
    assert fake.calls[0][1] == LiveMt5.TIMEFRAME_M15
    price = cd.live_price("EURUSD")
    assert price is not None and (price["bid"], price["ask"]) == (1.105, 1.1051)


def test_yahoo_fallback_when_mt5_closed() -> None:
    cd = ChartData(_prov(LiveMt5(up=False)), _yahoo())
    bars = cd.intraday("EURUSD", "EURUSD", "EURUSD=X", "H1")
    assert bars.source == "yfinance" and "fallback" in bars.note
    assert str(bars.frame.index.tz) == "UTC"
    assert cd.live_price("EURUSD") is None


def test_h4_from_yahoo_hourly_and_no_data_message() -> None:
    h4 = _norm_yahoo(_yahoo(48)("x", "H4"), "H4")
    assert len(h4) >= 8 and (h4["high"] >= h4["low"]).all()
    nothing = ChartData(None, lambda t, tf: pd.DataFrame())
    bars = nothing.intraday("EURUSD", "EURUSD", "EURUSD=X", "M5")
    assert bars.source == "none" and "MetaTrader 5" in bars.note and bars.frame.empty
    with pytest.raises(ValueError):
        nothing.intraday("EURUSD", "EURUSD", "EURUSD=X", "D1")


def test_api_serves_all_timeframes_view_only(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from fastapi.testclient import TestClient

    from tests.conftest import FakeProvider, MutableClock, synthetic_candles
    from tfx.alerts.notify import Notifier
    from tfx.api.app import create_app
    from tfx.data.cache import CandleCache
    from tfx.data.service import MarketDataService
    from tfx.pipeline import Pipeline
    from tfx.store import Store
    from tfx.validation.registry import ValidationRegistry

    now = datetime(2026, 11, 21, 1, tzinfo=UTC)
    full = synthetic_candles("2003-01-01", "2026-11-20", seed=12)
    cache = CandleCache(":memory:")
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", full),
                                         "mt5": FakeProvider("mt5", full)},
                            clock=MutableClock(now))
    svc.get("EURUSD", refresh=True)
    store = Store(":memory:")
    pipe = Pipeline(svc, store, ValidationRegistry(":memory:"))
    chart = ChartData(_prov(LiveMt5()), _yahoo())
    c = TestClient(create_app(pipe, Notifier(store, desktop=lambda t, b: True), None,
                              clock=lambda: now, chart=chart))
    meta = c.get("/api/meta").json()
    assert meta["view_timeframes"] == list(ALL_TIMEFRAMES)
    m15 = c.get("/api/candles/EURUSD?tf=M15").json()
    assert m15["live"] and m15["source"] == "mt5" and len(m15["candles"]) == 5
    assert "not validated, no signals" in m15["note"]
    assert "setup" not in m15 and "signal_levels" not in m15   # never signals off D1
    for tf in ("W1", "MN1"):
        r = c.get(f"/api/candles/EURUSD?tf={tf}").json()
        assert r["candles"] and not r["live"] and "view only" in r["note"]
    assert c.get("/api/candles/EURUSD?tf=D1").json()["validated_view"]
    assert c.get("/api/candles/EURUSD?tf=M2").status_code == 422
    before = len(chart.mt5._loader().calls)
    live = c.get("/api/live/EURUSD?tf=M15").json()
    assert live["price"]["bid"] == 1.105 and len(live["bars"]) == 3
    c.get("/api/live/EURUSD?tf=M15")
    calls = chart.mt5._loader().calls[before:]
    assert len(calls) == 2 and all(n == 3 for _, _, n in calls)  # fresh, uncached, 3 bars
    assert c.get("/api/live/EURUSD?tf=D1").json()["bars"] == []
    store.close()
