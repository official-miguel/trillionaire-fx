"""Swappable free data providers for daily FX candles.

Each provider returns a *raw* frame indexed by date with open/high/low/close
(and optionally volume). Cleaning, validation and in-progress-bar removal happen
in :mod:`tfx.data.quality`, so every provider is held to the same rules.

Network access is injected (``http_get`` / ``history_fn``) so tests never touch
the internet.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tfx.data.session import NEW_YORK
from tfx.data.yahoo_fx import build_fx_daily, is_yahoo_fx

LONDON = ZoneInfo("Europe/London")


class Alignment(StrEnum):
    """Where a provider's daily bar starts and how it is labelled."""

    NY_CLOSE = "ny_close"                # 17:00 NY -> 17:00 NY, labelled by closing date
    LONDON_MIDNIGHT = "london_midnight"  # 00:00 London -> 00:00 London, labelled by that date
    UTC_MIDNIGHT = "utc_midnight"        # 00:00 UTC -> 00:00 UTC, labelled by that date


# Native daily-bar alignment of each provider. A cross-check reference is
# requested on the PRIMARY's alignment whenever the reference provider allows it.
def native_alignment(provider: str, ticker: str | None) -> Alignment | None:
    """How ``provider`` labels daily bars for ``ticker``.

    Yahoo FX pairs (``EURUSD=X``) are rebuilt into UTC-midnight days (``yahoo_fx.py``);
    Yahoo futures (``GC=F``) use the exchange trading day, which ends at the 17:00
    New York close.
    """
    if provider == "yfinance" and ticker and ticker.upper().endswith("=F"):
        return Alignment.NY_CLOSE
    if provider == "yfinance" and is_yahoo_fx(ticker):
        return Alignment.UTC_MIDNIGHT
    return PROVIDER_ALIGNMENT.get(provider)


def looks_like_html(text: str) -> bool:
    head = (text or "").lstrip()[:200].lower()
    return head.startswith(("<!doctype", "<html", "<?xml", "<head", "<body")) or "<script" in head


PROVIDER_ALIGNMENT: dict[str, Alignment] = {
    "yfinance": Alignment.UTC_MIDNIGHT,
    "oanda": Alignment.NY_CLOSE,
    "mt5": Alignment.NY_CLOSE,
}

USER_AGENT = "TrillionaireFX/0.1 (local research tool)"


class ProviderError(RuntimeError):
    """A provider could not deliver data (network, format, auth, rate limit...)."""


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...

    def fetch_daily(self, ticker: str, start: date,
                    alignment: Alignment | None = None) -> pd.DataFrame:
        """``alignment`` is a request; providers that cannot honour it ignore it."""
        ...


HttpGet = Callable[[str, Mapping[str, Any], Mapping[str, str], float], "HttpResponse"]


class HttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


def _requests_get(
    url: str, params: Mapping[str, Any], headers: Mapping[str, str], timeout: float
) -> HttpResponse:
    return requests.get(url, params=dict(params), headers=dict(headers), timeout=timeout)


# --------------------------------------------------------------------------- yfinance
HistoryFn = Callable[[str, date], pd.DataFrame]


def ascii_support_dir() -> Path:
    """A writable per-user directory whose path is pure ASCII.

    libcurl (used by yfinance via curl_cffi) cannot open CA bundles or caches at
    non-ASCII paths on Windows, e.g. a project folder containing an emoji.
    """
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "TrillionaireFX",
        Path.home() / ".trillionairefx",
        Path(tempfile.gettempdir()) / "trillionairefx",
    ]
    for c in candidates:
        if str(c) not in {"TrillionaireFX", "."} and str(c).isascii():
            c.mkdir(parents=True, exist_ok=True)
            return c
    raise ProviderError("no ASCII-only directory available for the TLS certificate bundle")


def ascii_ca_bundle() -> str:
    """Path to a CA bundle libcurl can open (copied out of a non-ASCII venv path if needed)."""
    import certifi

    src = str(certifi.where())
    if src.isascii():
        return src
    dst = ascii_support_dir() / "cacert.pem"
    if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
        shutil.copyfile(src, dst)
    return str(dst)


_yf_session: Any = None


def _yf_history(ticker: str, start: date) -> pd.DataFrame:  # pragma: no cover - network
    global _yf_session
    import yfinance as yf  # local import: heavy, and optional at test time
    from curl_cffi import requests as curl_requests

    if _yf_session is None:
        yf.set_tz_cache_location(str(ascii_support_dir() / "yf-cache"))
        _yf_session = curl_requests.Session(impersonate="chrome", verify=ascii_ca_bundle())
    df = yf.Ticker(ticker, session=_yf_session).history(
        start=start.isoformat(), interval="1d", auto_adjust=False, actions=False, raise_errors=True
    )
    return df


def _yf_session_get() -> Any:  # pragma: no cover - network
    """Shared curl_cffi session for yfinance (CA bundle at an ASCII path)."""
    global _yf_session
    import yfinance as yf
    from curl_cffi import requests as curl_requests

    if _yf_session is None:
        yf.set_tz_cache_location(str(ascii_support_dir() / "yf-cache"))
        _yf_session = curl_requests.Session(impersonate="chrome", verify=ascii_ca_bundle())
    return _yf_session


def _yf_hourly(ticker: str, start: date) -> pd.DataFrame:  # pragma: no cover - network
    global _yf_session
    import yfinance as yf
    from curl_cffi import requests as curl_requests

    if _yf_session is None:
        yf.set_tz_cache_location(str(ascii_support_dir() / "yf-cache"))
        _yf_session = curl_requests.Session(impersonate="chrome", verify=ascii_ca_bundle())
    df = yf.Ticker(ticker, session=_yf_session).history(
        start=start.isoformat(), interval="1h", auto_adjust=False, actions=False, raise_errors=True
    )
    return df


YAHOO_HOURLY_MAX_DAYS = 729  # Yahoo serves 1h bars for roughly the last 730 days only


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, history_fn: HistoryFn | None = None,
                 hourly_fn: HistoryFn | None = None) -> None:
        self._history = history_fn or _yf_history
        self._hourly = hourly_fn or _yf_hourly

    def _fx_daily(self, ticker: str, start: date) -> pd.DataFrame:
        """UTC-midnight FX bars: hourly-built recent days + repaired older daily rows.

        Yahoo's raw daily FX rows are not real OHLC since ~2011 (see ``yahoo_fx.py``).
        Both raw responses are attached in ``attrs["raw_parts"]`` for the raw log.
        """
        try:
            raw_daily = self._history(ticker, start)
        except Exception as exc:
            raise ProviderError(f"yfinance {ticker}: {exc}") from exc
        if not isinstance(raw_daily, pd.DataFrame):
            raise ProviderError(f"yfinance {ticker}: unexpected response type")
        raw_hourly = self.fetch_hourly(ticker, start)  # raises ProviderError on failure
        built = build_fx_daily(raw_daily, raw_hourly, datetime.now(UTC))
        if built.empty:
            raise ProviderError(f"yfinance {ticker}: no complete daily bars could be built")
        built.attrs["raw_parts"] = {"daily": raw_daily, "hourly": raw_hourly}
        return built

    def fetch_hourly(self, ticker: str, start: date) -> pd.DataFrame:
        """H1 bars indexed by UTC open time; Yahoo keeps only ~730 days of them."""
        floor = datetime.now(UTC).date() - timedelta(days=YAHOO_HOURLY_MAX_DAYS)
        try:
            df = self._hourly(ticker, max(start, floor))
        except Exception as exc:
            raise ProviderError(f"yfinance {ticker} 1h: {exc}") from exc
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ProviderError(f"yfinance {ticker} 1h: empty or invalid response")
        df = df.rename(columns=str.lower)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        out = df[keep].copy()
        idx = pd.DatetimeIndex(out.index)
        out.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC"))
        out.index.name = "time"
        return out

    def available(self) -> bool:
        return True

    def fetch_daily(self, ticker: str, start: date,
                    alignment: Alignment | None = None) -> pd.DataFrame:
        if is_yahoo_fx(ticker):
            return self._fx_daily(ticker, start)
        try:
            df = self._history(ticker, start)
        except Exception as exc:
            raise ProviderError(f"yfinance {ticker}: {exc}") from exc
        if not isinstance(df, pd.DataFrame):
            raise ProviderError(f"yfinance {ticker}: unexpected response type {type(df).__name__}")
        if df.empty:
            raise ProviderError(f"yfinance {ticker}: empty response")
        df = df.rename(columns=str.lower)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        out = df[keep].copy()
        idx = pd.DatetimeIndex(out.index)
        if idx.tz is not None:
            # Yahoo labels FX bars in exchange-local time; the calendar date is the label.
            idx = idx.tz_localize(None)
        out.index = idx.normalize()
        return out


# --------------------------------------------------------------------------- Stooq
class StooqProvider:
    name = "stooq"
    URL = "https://stooq.com/q/d/l/"

    def __init__(self, http_get: HttpGet | None = None, timeout: float = 20.0) -> None:
        self._get = http_get or _requests_get
        self._timeout = timeout

    def available(self) -> bool:
        return True

    def fetch_daily(self, ticker: str, start: date,
                    alignment: Alignment | None = None) -> pd.DataFrame:
        params = {"s": ticker.lower(), "i": "d", "d1": start.strftime("%Y%m%d"),
                  "d2": (datetime.now(UTC).date() + timedelta(days=1)).strftime("%Y%m%d")}
        try:
            resp = self._get(self.URL, params, {"User-Agent": USER_AGENT}, self._timeout)
        except Exception as exc:
            raise ProviderError(f"stooq {ticker}: {exc}") from exc
        if resp.status_code != 200:
            raise ProviderError(f"stooq {ticker}: HTTP {resp.status_code}")
        return parse_stooq_csv(resp.text, ticker)


def parse_stooq_csv(text: str, ticker: str = "?") -> pd.DataFrame:
    body = (text or "").strip()
    if looks_like_html(body):
        raise ProviderError(f"stooq {ticker}: HTML page instead of CSV (bot check or error page)")
    if not body.lower().startswith("date"):
        snippet = body[:80].replace("\n", " ")
        raise ProviderError(f"stooq {ticker}: unexpected response: {snippet!r}")
    try:
        df = pd.read_csv(io.StringIO(body))
    except Exception as exc:
        raise ProviderError(f"stooq {ticker}: unparseable CSV: {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or df.empty:
        raise ProviderError(f"stooq {ticker}: no rows")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"), errors="coerce"))
    df = df[df.index.notna()]
    if df.empty:
        raise ProviderError(f"stooq {ticker}: no parseable dates")
    return df


# --------------------------------------------------------------------------- OANDA
class OandaProvider:
    """OANDA *practice* REST API — read-only candles. Never used for orders.

    By default candles are aligned to 17:00 America/New_York so their labels match
    :class:`~tfx.data.session.SessionCalendar`. As a cross-check reference for Yahoo
    they can instead be aligned to London midnight, so both feeds describe the
    same 24 hours.
    """

    name = "oanda"
    URL = "https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    MAX_COUNT = 5000

    def __init__(
        self, token: str | None = None, http_get: HttpGet | None = None, timeout: float = 20.0
    ) -> None:
        self._token = token if token is not None else os.environ.get("OANDA_API_TOKEN", "")
        self._get = http_get or _requests_get
        self._timeout = timeout

    def available(self) -> bool:
        return bool(self._token)

    def fetch_daily(self, ticker: str, start: date,
                    alignment: Alignment | None = None) -> pd.DataFrame:
        align = alignment or Alignment.NY_CLOSE
        hour, tz = {Alignment.NY_CLOSE: (17, "America/New_York"),
                    Alignment.LONDON_MIDNIGHT: (0, "Europe/London"),
                    Alignment.UTC_MIDNIGHT: (0, "UTC")}[align]
        if not self.available():
            raise ProviderError("oanda: OANDA_API_TOKEN not set")
        headers = {"Authorization": f"Bearer {self._token}", "User-Agent": USER_AGENT}
        rows: list[dict[str, Any]] = []
        cursor = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        for _ in range(20):  # 20 x 5000 days is far more than needed; bounds the loop
            params = {
                "granularity": "D", "price": "M", "count": self.MAX_COUNT,
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dailyAlignment": hour, "alignmentTimezone": tz,
            }
            try:
                resp = self._get(self.URL.format(instrument=ticker), params, headers, self._timeout)
            except Exception as exc:
                raise ProviderError(f"oanda {ticker}: {exc}") from exc
            if resp.status_code != 200:
                raise ProviderError(f"oanda {ticker}: HTTP {resp.status_code}")
            if looks_like_html(getattr(resp, "text", "") or ""):
                raise ProviderError(f"oanda {ticker}: HTML page instead of JSON")
            try:
                payload = resp.json()
            except ValueError as exc:
                raise ProviderError(f"oanda {ticker}: malformed JSON") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("candles", []), list):
                raise ProviderError(f"oanda {ticker}: unexpected payload shape")
            candles = payload.get("candles", [])
            batch = parse_oanda_candles(candles, align)
            rows.extend(batch)
            if len(candles) < self.MAX_COUNT:
                break
            last_open = pd.Timestamp(candles[-1]["time"]).to_pydatetime()
            cursor = last_open + timedelta(seconds=1)
        if not rows:
            raise ProviderError(f"oanda {ticker}: no complete candles")
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return df


def parse_oanda_candles(candles: list[dict[str, Any]],
                        alignment: Alignment = Alignment.NY_CLOSE) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candles:
        if not c.get("complete", False):
            continue  # in-progress candle: never used
        if alignment is Alignment.NY_CLOSE:
            # A candle opening at 17:00 NY on day D closes at 17:00 NY on day D+1;
            # its label is the closing date.
            opened = pd.Timestamp(c["time"]).tz_convert(NEW_YORK)
            label = (opened + pd.Timedelta(days=1)).date()
        elif alignment is Alignment.UTC_MIDNIGHT:
            label = pd.Timestamp(c["time"]).tz_convert("UTC").date()
        else:
            # London-midnight candles are labelled by the London date they open on.
            label = pd.Timestamp(c["time"]).tz_convert(LONDON).date()
        mid = c["mid"]
        out.append({
            "date": pd.Timestamp(label),
            "open": float(mid["o"]), "high": float(mid["h"]),
            "low": float(mid["l"]), "close": float(mid["c"]),
            "volume": float(c.get("volume", float("nan"))),
        })
    return out


def build_providers(timeout: float = 20.0, mt5_server_time: str = "ny_close",
                    mt5_terminal_path: str | None = None) -> dict[str, Provider]:
    from tfx.data.mt5_provider import Mt5Provider

    return {
        "yfinance": YFinanceProvider(),
        "mt5": Mt5Provider(mt5_server_time, mt5_terminal_path),
        "stooq": StooqProvider(timeout=timeout),
        "oanda": OandaProvider(timeout=timeout),
    }
