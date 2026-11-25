from __future__ import annotations

import pathlib
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tfx.config import AppConfig, Mt5Config
from tfx.data.mt5_provider import Mt5Provider, server_to_utc
from tfx.data.providers import Alignment, ProviderError, build_providers
from tfx.data.session import NEW_YORK

RATE_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
              ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"),
              ("real_volume", "<u8")]


def _rates(server_index: pd.DatetimeIndex, close: np.ndarray) -> np.ndarray:
    arr = np.zeros(len(server_index), dtype=RATE_DTYPE)
    arr["time"] = server_index.as_unit("s").asi8  # MT5 gives epoch seconds of server time
    arr["open"] = close
    arr["high"] = close + 0.001
    arr["low"] = close - 0.001
    arr["close"] = close
    arr["tick_volume"] = 10
    arr["spread"] = 7
    return arr


class FakeMt5:
    TIMEFRAME_D1 = 16408
    TIMEFRAME_H1 = 16385

    def __init__(self, *, init_ok: bool = True, signed_in: bool = True,
                 names: tuple[str, ...] = ("EURUSD.m", "GBPUSDm", "GOLD", "USDJPY"),
                 h1: np.ndarray | None = None, d1: np.ndarray | None = None) -> None:
        self.init_ok, self.signed_in, self.names = init_ok, signed_in, names
        self.h1, self.d1 = h1, d1
        self.calls: list[tuple[Any, ...]] = []
        self.shut = False

    def initialize(self, **kw: Any) -> bool:
        self.calls.append(("initialize", kw))
        return self.init_ok

    def last_error(self) -> tuple[int, str]:
        return (-6, "Terminal: Authorization failed")

    def account_info(self) -> Any:
        if not self.signed_in:
            return None
        return SimpleNamespace(server="Broker-Demo", company="Example Broker Ltd",
                               currency="USD", trade_mode=0, login=12345678)

    def shutdown(self) -> None:
        self.shut = True

    def symbols_get(self) -> list[Any]:
        return [SimpleNamespace(name=n) for n in self.names]

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def copy_rates_range(self, symbol: str, tf: int, frm: Any, to: Any) -> np.ndarray | None:
        self.calls.append(("copy_rates_range", symbol, tf))
        return self.h1 if tf == self.TIMEFRAME_H1 else self.d1

    def __getattr__(self, name: str) -> Any:
        # Any trading call is a hard failure: the provider must stay read-only.
        raise AssertionError(f"Mt5Provider touched forbidden MT5 API: {name}")


def _prov(fake: FakeMt5, server_time: str = "ny_close") -> Mt5Provider:
    return Mt5Provider(server_time=server_time, module_loader=lambda: fake)


# ----------------------------------------------------------- time handling
def test_server_time_conversions() -> None:
    idx = pd.DatetimeIndex(["2026-11-20 00:00", "2026-07-16 00:00"])
    ny = server_to_utc(idx, "ny_close")
    # 00:00 server = 17:00 New York the previous day (EST in Nov, EDT in Jul)
    assert list(ny) == [pd.Timestamp("2026-11-19 22:00", tz="UTC"),
                        pd.Timestamp("2026-07-15 21:00", tz="UTC")]
    assert server_to_utc(idx, "utc")[0] == pd.Timestamp("2026-11-20", tz="UTC")
    assert server_to_utc(idx, "+2")[0] == pd.Timestamp("2026-11-19 22:00", tz="UTC")
    with pytest.raises(ProviderError):
        server_to_utc(idx, "broker")


# ------------------------------------------------------------- connection
def test_not_installed_not_running_not_signed_in() -> None:
    def missing() -> Any:
        raise ImportError("no module")

    p = Mt5Provider(module_loader=missing)
    assert not p.available() and "not installed" in (p.last_error or "")
    p = _prov(FakeMt5(init_ok=False))
    assert not p.available() and "Authorization failed" in (p.last_error or "")
    fake = FakeMt5(signed_in=False)
    p = _prov(fake)
    assert not p.available() and "not signed in" in (p.last_error or "") and fake.shut


def test_failed_connection_not_retried_for_five_minutes() -> None:
    fake = FakeMt5(init_ok=False)
    now = [1000.0]
    p = Mt5Provider(module_loader=lambda: fake, clock=lambda: now[0])
    for _ in range(12):  # one check per market must not block 12 times
        assert not p.available()
    assert sum(1 for c in fake.calls if c[0] == "initialize") == 1
    now[0] += 301
    fake.init_ok = True
    assert p.available()
    assert sum(1 for c in fake.calls if c[0] == "initialize") == 2


def test_terminal_path_passed_and_close() -> None:
    fake = FakeMt5()
    p = Mt5Provider(terminal_path="C:/MT5/terminal64.exe", module_loader=lambda: fake)
    assert p.available()
    assert fake.calls[0] == ("initialize", {"timeout": 10_000, "path": "C:/MT5/terminal64.exe"})
    p.close()
    assert fake.shut and p._mt5 is None


def test_account_summary_has_no_login_or_secret() -> None:
    summary = _prov(FakeMt5()).account_summary()
    assert summary == {"server": "Broker-Demo", "company": "Example Broker Ltd",
                       "currency": "USD", "is_demo": True}


def test_symbol_resolution_handles_broker_suffixes() -> None:
    p = _prov(FakeMt5())
    assert p.resolve("EURUSD") == "EURUSD.m"
    assert p.resolve("GBPUSD") == "GBPUSDm"
    assert p.resolve("USDJPY") == "USDJPY"
    assert p.resolve("XAUUSD") == "GOLD"
    with pytest.raises(ProviderError, match="no symbol"):
        p.resolve("NZDUSD")


# ---------------------------------------------------------------- candles
def _hourly_utc(start: str, end: str) -> pd.DataFrame:
    utc = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    ny = utc.tz_convert(NEW_YORK)
    # FX week: Sunday 17:00 NY to Friday 17:00 NY
    open_ = ~(((ny.day_of_week == 4) & (ny.hour >= 17)) | (ny.day_of_week == 5)
              | ((ny.day_of_week == 6) & (ny.hour < 17)))
    utc = utc[open_]
    close = 1.10 + np.arange(len(utc)) * 1e-5
    return pd.DataFrame({"close": close}, index=utc)


def test_hourly_bars_regrouped_into_london_days() -> None:
    h = _hourly_utc("2026-11-02", "2026-11-21")
    server = (h.index.tz_convert(NEW_YORK).tz_localize(None) + pd.Timedelta(hours=7))
    fake = FakeMt5(h1=_rates(pd.DatetimeIndex(server), h["close"].to_numpy()))
    df = _prov(fake).fetch_daily("EURUSD", date(2026, 11, 3), Alignment.LONDON_MIDNIGHT)
    london = h.index.tz_convert("Europe/London").normalize().tz_localize(None)
    expected = h["close"].groupby(london).last()
    common = df.index.intersection(expected.index)
    assert len(common) >= 13
    np.testing.assert_allclose(df.loc[common, "close"], expected.loc[common])
    assert df.index[0] > expected.index[0]  # partial first day dropped
    assert (df["spread_points"] == 7).all()
    assert ("copy_rates_range", "EURUSD.m", FakeMt5.TIMEFRAME_H1) in fake.calls


def test_d1_bars_labelled_by_server_date() -> None:
    server = pd.DatetimeIndex(["2026-11-19", "2026-11-20"])
    fake = FakeMt5(d1=_rates(server, np.array([1.1, 1.2])))
    df = _prov(fake).fetch_daily("EURUSD", date(2026, 11, 1))
    assert list(df.index) == [pd.Timestamp("2026-11-19"), pd.Timestamp("2026-11-20")]
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "spread_points"]
    with pytest.raises(ProviderError, match="ny_close"):
        _prov(FakeMt5(d1=fake.d1), server_time="utc").fetch_daily("EURUSD", date(2026, 11, 1))


def test_empty_rates_is_provider_error() -> None:
    with pytest.raises(ProviderError, match="no bars"):
        _prov(FakeMt5(d1=np.zeros(0, dtype=RATE_DTYPE))).fetch_daily("EURUSD", date(2026, 1, 1))


# ----------------------------------------------------------------- config
def test_mt5_config_and_registry(cfg: AppConfig) -> None:
    assert cfg.data.provider_order[:2] == ["yfinance", "mt5"]
    assert cfg.symbol("EURUSD").ticker_for("mt5") == "EURUSD"
    assert Mt5Config(server_time="+2").server_time == "+2"
    with pytest.raises(ValidationError):
        Mt5Config(server_time="broker time")
    assert "mt5" in build_providers()


def test_no_order_functions_anywhere_in_the_package() -> None:
    """Hard constraint: TrillionaireFX never places, modifies or closes trades."""
    forbidden = ("order_send", "order_check", "positions_get", "TRADE_ACTION")
    pkg = pathlib.Path(__file__).resolve().parents[2] / "tfx"
    offenders = [str(p) for p in pkg.rglob("*.py")
                 for word in forbidden if word in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_hourly_bars_converted_from_server_time_to_utc() -> None:
    server = pd.DatetimeIndex(["2026-11-20 00:00", "2026-11-20 01:00", "2026-07-16 00:00"])
    fake = FakeMt5(h1=_rates(server, np.array([1.1, 1.2, 1.3])))
    df = _prov(fake).fetch_hourly("EURUSD", date(2026, 11, 1))
    assert list(df.index) == [pd.Timestamp("2026-11-19 22:00", tz="UTC"),
                              pd.Timestamp("2026-11-19 23:00", tz="UTC"),
                              pd.Timestamp("2026-07-15 21:00", tz="UTC")]
    assert (df["spread_points"] == 7).all()
    with pytest.raises(ProviderError, match="no hourly bars"):
        _prov(FakeMt5(h1=np.zeros(0, dtype=RATE_DTYPE))).fetch_hourly("EURUSD", date(2026, 1, 1))
