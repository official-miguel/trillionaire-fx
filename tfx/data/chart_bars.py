"""Chart bars for any timeframe, for DISPLAY ONLY.

* D1 comes from the confirmed daily series (the only timeframe that can produce
  signals, and only for validated strategies). W1 and MN1 are built from it.
* Intraday bars (M1 ... H4) come straight from the MT5 terminal (your broker's own
  prices). If MT5 is not connected, Yahoo is used as a fallback (limited history).
* Intraday and monthly bars are never used for signals: the API labels them
  "view only - not validated, no signals".
* Results are cached for a few seconds so live polling does not hammer MT5.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from tfx.data.mt5_provider import server_to_utc
from tfx.logging_setup import fields, get_logger

log = get_logger(__name__)

INTRADAY = ("M1", "M5", "M15", "M30", "H1", "H4")
ALL_TIMEFRAMES = (*INTRADAY, "D1", "W1", "MN1")
MT5_CONST = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
             "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4"}
# Yahoo interval and the longest period it serves for it
YAHOO = {"M1": ("1m", "7d"), "M5": ("5m", "60d"), "M15": ("15m", "60d"),
         "M30": ("30m", "60d"), "H1": ("60m", "730d"), "H4": ("60m", "730d")}
BAR_COUNT = 3000


@dataclass
class ChartBars:
    frame: pd.DataFrame          # UTC-indexed OHLC (open time)
    source: str                  # mt5 | yfinance | none
    note: str


def _from_mt5(mt5_provider: Any, ticker: str, tf: str, count: int) -> pd.DataFrame:
    mt5 = mt5_provider._connect()
    symbol = mt5_provider.resolve(ticker)
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, MT5_CONST[tf]), 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no {tf} bars for {symbol}")
    df = pd.DataFrame(rates)
    utc = server_to_utc(pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s")),
                        mt5_provider.server_time)
    out = pd.DataFrame({"open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
                        "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float)},
                       index=utc)
    return out[~pd.isna(out.index)].sort_index()


def _yahoo_intraday(ticker: str, tf: str) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf

    from tfx.data.providers import _yf_session_get

    interval, period = YAHOO[tf]
    df = yf.Ticker(ticker, session=_yf_session_get()).history(
        period=period, interval=interval, auto_adjust=False, actions=False)
    return df


def _norm_yahoo(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if tf == "H4":  # Yahoo has no 4h FX bars: build them from 1h on New York-close buckets
        from tfx.data.timeframes import resample_hourly

        df = resample_hourly(df, pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=4))
    return df.dropna().sort_index()


class ChartData:
    def __init__(self, mt5_provider: Any, yahoo_fn: Callable[[str, str], pd.DataFrame]
                 | None = None, ttl_seconds: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.mt5 = mt5_provider
        self._yahoo = yahoo_fn or _yahoo_intraday
        self.ttl = ttl_seconds
        self._clock = clock
        self._cache: dict[tuple[str, str], tuple[float, ChartBars]] = {}

    def intraday(self, symbol: str, mt5_ticker: str | None, yahoo_ticker: str | None,
                 tf: str) -> ChartBars:
        if tf not in INTRADAY:
            raise ValueError(f"{tf} is not an intraday timeframe")
        key = (symbol, tf)
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < self.ttl:
            return hit[1]
        bars = self._load(mt5_ticker, yahoo_ticker, tf)
        self._cache[key] = (self._clock(), bars)
        return bars

    def _load(self, mt5_ticker: str | None, yahoo_ticker: str | None, tf: str) -> ChartBars:
        errors = []
        if self.mt5 is not None and mt5_ticker and self.mt5.available():
            try:
                return ChartBars(_from_mt5(self.mt5, mt5_ticker, tf, BAR_COUNT), "mt5",
                                 "Broker prices from MetaTrader 5.")
            except Exception as exc:
                errors.append(f"MT5: {exc}")
        if yahoo_ticker:
            try:
                df = _norm_yahoo(self._yahoo(yahoo_ticker, tf), tf)
                if not df.empty:
                    return ChartBars(df, "yfinance",
                                     "MT5 not connected: Yahoo fallback (limited history).")
            except Exception as exc:
                errors.append(f"Yahoo: {type(exc).__name__}: {exc}")
        log.warning("no chart bars", extra=fields(tf=tf, errors=errors))
        return ChartBars(pd.DataFrame(columns=["open", "high", "low", "close"]), "none",
                         "No intraday data: open MetaTrader 5 and sign in. " + " ".join(errors))

    def latest(self, symbol: str, mt5_ticker: str | None, yahoo_ticker: str | None, tf: str,
               n: int = 3) -> ChartBars:
        """Last ``n`` bars, read FRESH from MT5 (no cache) for once-a-second live updates.

        Falls back to the cached series (Yahoo) when MT5 is not connected.
        """
        if self.mt5 is not None and mt5_ticker and self.mt5.available():
            try:
                return ChartBars(_from_mt5(self.mt5, mt5_ticker, tf, n), "mt5", "live")
            except Exception as exc:
                log.warning("live bars failed", extra=fields(tf=tf, error=str(exc)))
        bars = self.intraday(symbol, mt5_ticker, yahoo_ticker, tf)
        return ChartBars(bars.frame.iloc[-n:], bars.source, bars.note)

    def live_price(self, mt5_ticker: str | None) -> dict[str, Any] | None:
        """Current bid/ask from MT5 for display (never used for signals)."""
        if self.mt5 is None or not mt5_ticker or not self.mt5.available():
            return None
        try:
            mt5 = self.mt5._connect()
            tick = mt5.symbol_info_tick(self.mt5.resolve(mt5_ticker))
            if tick is None:
                return None
            return {"bid": float(tick.bid), "ask": float(tick.ask),
                    "time": datetime.now(UTC).isoformat()}
        except Exception:
            return None
