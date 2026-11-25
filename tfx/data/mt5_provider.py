"""Read-only MetaTrader 5 data provider (your broker's own prices).

Uses MetaQuotes' free ``MetaTrader5`` Python package, which talks to the MT5
terminal running on this computer. TrillionaireFX never passes credentials to it
and never calls any order function: the terminal must already be open and
signed in to an account (a demo account is enough).

Broker server time
------------------
MT5 bar times are the broker's *server* clock written as if it were UTC. Most
retail FX brokers run the server at New York time + 7 hours, so the server day
starts at the 17:00 New York close. ``server_time`` selects the convention:

* ``"ny_close"`` (default) – server = New York + 7h (follows US daylight saving)
* ``"utc"``              – server clock is UTC
* ``"+2"`` / ``"-5"``    – fixed offset in hours from UTC

If the convention is wrong, the ``tfx sources`` report shows large disagreement.

Alignment
---------
* NY_CLOSE: D1 bars, labelled by server date (= the New York closing date).
* LONDON_MIDNIGHT: H1 bars re-aggregated into London-midnight days, to compare
  like for like with Yahoo's daily bars.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from tfx.data.providers import Alignment, ProviderError
from tfx.data.session import NEW_YORK

LONDON = ZoneInfo("Europe/London")

# Common broker aliases for symbols whose name differs from the market code.
ALIASES: dict[str, list[str]] = {"XAUUSD": ["GOLD", "XAUUSD"]}


def _load_mt5() -> Any:  # pragma: no cover - requires Windows + MetaTrader5 package
    import MetaTrader5

    return MetaTrader5


def server_to_utc(server_naive: pd.DatetimeIndex, server_time: str) -> pd.DatetimeIndex:
    """Convert MT5 server timestamps (naive) to tz-aware UTC."""
    if server_time == "ny_close":
        # server = NY wall clock + 7h  =>  NY wall clock = server - 7h
        ny_wall = server_naive - pd.Timedelta(hours=7)
        return ny_wall.tz_localize(NEW_YORK, ambiguous="NaT", nonexistent="shift_forward"
                                   ).tz_convert(UTC)
    if server_time == "utc":
        return server_naive.tz_localize(UTC)
    try:
        hours = float(server_time)
    except ValueError as exc:
        raise ProviderError(f"mt5: unknown server_time {server_time!r}") from exc
    return (server_naive - pd.Timedelta(hours=hours)).tz_localize(UTC)


class Mt5Provider:
    name = "mt5"

    INIT_TIMEOUT_MS = 10_000
    RETRY_AFTER_S = 300.0  # after a failed connection, don't retry (and block) for 5 minutes

    def __init__(self, server_time: str = "ny_close", terminal_path: str | None = None,
                 module_loader: Any = None, clock: Any = time.monotonic) -> None:
        self.server_time = server_time
        self.terminal_path = terminal_path or os.environ.get("MT5_TERMINAL_PATH") or None
        self._loader = module_loader or _load_mt5
        self._mt5: Any = None
        self._symbols: list[str] | None = None
        self.last_error: str | None = None
        self._clock = clock
        self._failed_at: float | None = None

    # -------------------------------------------------------------- session
    def _connect(self) -> Any:
        if self._mt5 is not None:
            return self._mt5
        if self._failed_at is not None and self._clock() - self._failed_at < self.RETRY_AFTER_S:
            raise ProviderError(self.last_error or "mt5: not connected (retrying later)")
        try:
            return self._try_connect()
        except ProviderError as exc:
            self._failed_at, self.last_error = self._clock(), str(exc)
            raise

    def _try_connect(self) -> Any:
        try:
            mt5 = self._loader()
        except ImportError as exc:
            raise ProviderError("mt5: MetaTrader5 package not installed (Windows only)") from exc
        kwargs: dict[str, Any] = {"timeout": self.INIT_TIMEOUT_MS}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            raise ProviderError(f"mt5: terminal not ready ({err}). Open MetaTrader 5 and sign "
                                "in to your demo account.")
        if mt5.account_info() is None:
            mt5.shutdown()
            raise ProviderError("mt5: terminal is not signed in to an account")
        self._mt5, self._failed_at = mt5, None
        return mt5

    def available(self) -> bool:
        try:
            self._connect()
        except ProviderError as exc:
            self.last_error = str(exc)
            return False
        return True

    def close(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    def account_summary(self) -> dict[str, Any]:
        """Non-secret account facts for display (no login number, no password)."""
        info = self._connect().account_info()
        return {"server": info.server, "company": info.company, "currency": info.currency,
                "is_demo": int(getattr(info, "trade_mode", -1)) == 0}

    # -------------------------------------------------------------- symbols
    def resolve(self, ticker: str) -> str:
        """Map a market code (EURUSD) to this broker's symbol name (EURUSD, EURUSD.m, GOLD...)."""
        mt5 = self._connect()
        if self._symbols is None:
            self._symbols = [s.name for s in (mt5.symbols_get() or [])]
        names = self._symbols
        for want in ALIASES.get(ticker, [ticker]):
            if want in names:
                return want
            # broker suffixes/prefixes such as EURUSD.m, EURUSDm, EURUSD.raw, m.EURUSD
            matches = sorted((n for n in names
                              if want in n.upper() and len(n) - len(want) <= 4), key=len)
            if matches:
                return matches[0]
        raise ProviderError(f"mt5: broker has no symbol matching {ticker}")

    # -------------------------------------------------------------- candles
    def fetch_daily(self, ticker: str, start: date,
                    alignment: Alignment | None = None) -> pd.DataFrame:
        mt5 = self._connect()
        symbol = self.resolve(ticker)
        if not mt5.symbol_select(symbol, True):
            raise ProviderError(f"mt5: cannot select {symbol} in Market Watch")
        align = alignment or Alignment.NY_CLOSE
        # Server time can be up to ~a day ahead of UTC: pad both ends.
        frm = datetime.combine(start - timedelta(days=2), datetime.min.time(), tzinfo=UTC)
        to = datetime.now(UTC) + timedelta(days=2)
        tf = mt5.TIMEFRAME_D1 if align is Alignment.NY_CLOSE else mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_range(symbol, tf, frm, to)
        if rates is None or len(rates) == 0:
            raise ProviderError(f"mt5 {symbol}: no bars returned ({mt5.last_error()})")
        df = pd.DataFrame(rates)
        server = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"))
        out = pd.DataFrame({
            "open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(), "close": df["close"].to_numpy(),
            "volume": df["tick_volume"].astype("float64").to_numpy(),
            "spread_points": df["spread"].astype("float64").to_numpy(),
        }, index=server)
        if align is Alignment.NY_CLOSE:
            if self.server_time != "ny_close":
                raise ProviderError("mt5: NY-close daily bars need server_time 'ny_close'")
            out.index = server.normalize()
            out.index.name = "date"
            return out
        if align is Alignment.UTC_MIDNIGHT:
            return self._utc_days(out)
        return self._london_days(out)

    def _utc_days(self, hourly: pd.DataFrame) -> pd.DataFrame:
        """Server-time H1 bars -> UTC-midnight days (Mon-Fri, >= 12 hourly bars)."""
        utc = server_to_utc(pd.DatetimeIndex(hourly.index), self.server_time)
        keep = ~utc.isna()
        h = hourly[keep].set_axis(utc[keep])
        key = pd.DatetimeIndex(h.index).normalize().tz_localize(None)
        g = h.groupby(key)
        daily = pd.DataFrame({
            "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
            "close": g["close"].last(), "volume": g["volume"].sum(),
            "spread_points": g["spread_points"].median(), "hours": g["close"].size(),
        })
        if len(daily) > 1:
            daily = daily.iloc[1:]  # first day is partial (fetch window starts mid-day)
        daily = daily[(daily.index.day_of_week < 5) & (daily["hours"] >= 12)]
        daily.index.name = "date"
        return daily.drop(columns=["hours"])

    def fetch_hourly(self, ticker: str, start: date) -> pd.DataFrame:
        """H1 bars from ``start`` to now, indexed by UTC open time (caller drops unclosed bars)."""
        return self.fetch_hourly_range(ticker, start, datetime.now(UTC).date() + timedelta(days=2))

    def fetch_hourly_range(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """H1 bars whose server time falls in [start-1d, end), indexed by UTC open time.

        History depth is limited by the broker and by the terminal's
        Tools > Options > Charts > "Max bars in chart" setting.
        """
        mt5 = self._connect()
        symbol = self.resolve(ticker)
        if not mt5.symbol_select(symbol, True):
            raise ProviderError(f"mt5: cannot select {symbol} in Market Watch")
        frm = datetime.combine(start - timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        to = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, frm, to)
        if rates is None or len(rates) == 0:
            raise ProviderError(f"mt5 {symbol}: no hourly bars returned ({mt5.last_error()})")
        df = pd.DataFrame(rates)
        server = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"))
        utc = server_to_utc(server, self.server_time)
        out = pd.DataFrame({
            "open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(), "close": df["close"].to_numpy(),
            "volume": df["tick_volume"].astype("float64").to_numpy(),
            "spread_points": df["spread"].astype("float64").to_numpy(),
        }, index=utc)
        out = out[~pd.isna(out.index)]
        out.index.name = "time"
        return out

    def _london_days(self, hourly: pd.DataFrame) -> pd.DataFrame:
        utc = server_to_utc(pd.DatetimeIndex(hourly.index), self.server_time)
        keep = ~utc.isna()
        hourly = hourly[keep]
        london_day = utc[keep].tz_convert(LONDON).normalize().tz_localize(None)
        g = hourly.groupby(london_day)
        daily = pd.DataFrame({
            "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
            "close": g["close"].last(), "volume": g["volume"].sum(),
            "spread_points": g["spread_points"].median(),
        })
        if len(daily) > 1:
            daily = daily.iloc[1:]  # first day is partial (fetch window starts mid-day)
        daily.index.name = "date"
        return daily
