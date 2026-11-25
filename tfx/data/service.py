"""Market-data service: provider fallback, caching, cross-checking, staleness guard.

The single rule that matters downstream: ``MarketData.signal_allowed`` is True
only when the series ends on the latest *completed* NY-close session AND (when
cross-checking is enabled) a second source confirms that bar. Anything else is
reported as ``DATA UNAVAILABLE - NO SIGNAL`` and must never be shown as a current
signal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

import pandas as pd

from tfx.config import AppConfig, ProviderName
from tfx.data.cache import CandleCache, MetaInfo
from tfx.data.crosscheck import CheckStatus, CrossCheck, cross_check
from tfx.data.providers import Alignment, Provider, ProviderError, native_alignment
from tfx.data.quality import DataQualityError, QualityReport, clean_candles, empty_frame
from tfx.data.session import SessionCalendar
from tfx.logging_setup import fields, get_logger

log = get_logger(__name__)

# Bump when the way a primary series is built changes; cached series built differently
# are then replaced on the next refresh (2: UTC-midnight days, repaired Yahoo FX bars;
# 3: fix summer rows shifted a day by a UTC conversion).
SERIES_CONSTRUCTION = "3-utc-midnight-london-dated-rows"

NO_SIGNAL_HEADLINE = "DATA UNAVAILABLE - NO SIGNAL"

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class DataStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    DISPUTED = "DISPUTED"        # a second source disagrees on the latest bar
    UNVERIFIED = "UNVERIFIED"    # no second source could confirm the latest bar
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class MarketData:
    symbol: str
    candles: pd.DataFrame
    status: DataStatus
    expected_session: date
    last_bar: date | None
    lag_sessions: int | None
    source: str | None
    data_version: str | None
    message: str
    is_proxy: bool = False
    check: CrossCheck | None = None

    @property
    def signal_allowed(self) -> bool:
        return self.status is DataStatus.FRESH

    @property
    def headline(self) -> str:
        return "DATA OK" if self.signal_allowed else NO_SIGNAL_HEADLINE


@dataclass
class RefreshResult:
    symbol: str
    ok: bool
    source: str | None = None
    mode: str | None = None
    rows_written: int = 0
    errors: list[str] = field(default_factory=list)
    quality: QualityReport | None = None


class MarketDataService:
    def __init__(
        self,
        config: AppConfig,
        cache: CandleCache,
        providers: Mapping[str, Provider],
        clock: Clock = utc_now,
    ) -> None:
        self.cfg = config
        self.cache = cache
        self.providers = providers
        self.clock = clock
        self._force_replace = False
        self.calendar = SessionCalendar(
            config.data.bar_complete_delay_minutes, config.data.non_session_days,
            anchor=config.data.day_boundary,
        )

    # ------------------------------------------------------------------ refresh
    def _usable(self, symbol: str, exclude: str | None = None
                ) -> list[tuple[ProviderName, Provider, str]]:
        sym_cfg = self.cfg.symbol(symbol)
        out: list[tuple[ProviderName, Provider, str]] = []
        for pname in self.cfg.providers_for(symbol):
            provider = self.providers.get(pname)
            ticker = sym_cfg.ticker_for(pname)
            if pname == exclude or provider is None or ticker is None or not provider.available():
                continue
            out.append((pname, provider, ticker))
        return out

    def _raw(self, symbol: str, provider: Provider, pname: str, ticker: str,
             start: date, alignment: Alignment | None = None) -> pd.DataFrame:
        """Fetch and append the response, exactly as received, to the immutable raw log."""
        raw = provider.fetch_daily(ticker, start, alignment=alignment)
        parts = getattr(raw, "attrs", {}).get("raw_parts") if raw is not None else None
        if parts:  # provider built bars from several raw responses: log each one
            for name, part in parts.items():
                if isinstance(part, pd.DataFrame) and not part.empty:
                    self.cache.log_raw(f"{symbol}@{name}", pname, self.clock(), part)
        elif raw is not None and not raw.empty:
            self.cache.log_raw(symbol, pname, self.clock(), raw)
        return raw

    def refresh(self, symbol: str) -> RefreshResult:
        symbol = symbol.upper()
        now = self.clock()
        last_complete = self.calendar.last_completed_session(now)
        result = RefreshResult(symbol=symbol, ok=False)
        meta = self.cache.meta(symbol)
        if meta is not None and self.cache.flag(f"construction:{symbol}") != SERIES_CONSTRUCTION:
            log.info("series built by an older method: full rebuild",
                     extra=fields(symbol=symbol))
            meta = None
            self._force_replace = True
        else:
            self._force_replace = False

        if meta is not None and meta.last_date >= last_complete:
            result.ok, result.source, result.mode = True, meta.source, "up-to-date"
            return result

        for pname, provider, ticker in self._usable(symbol):
            try:
                mode, df, report = self._fetch(symbol, provider, pname, ticker, meta,
                                               last_complete)
            except Exception as exc:  # any provider failure => try the next provider
                if not isinstance(exc, (ProviderError, DataQualityError)):
                    exc = ProviderError(f"{type(exc).__name__}: {exc}")
                result.errors.append(f"{pname}: {exc}")
                self.cache.log_failure(symbol, pname, now, str(exc))
                log.warning("provider failed", extra=fields(symbol=symbol, provider=pname,
                                                            error=str(exc)))
                continue
            if self._force_replace:
                mode = "replace"
            info = self.cache.write(symbol, df, pname, mode, now)
            self.cache.set_flag(f"construction:{symbol}", SERIES_CONSTRUCTION)
            result.ok, result.source, result.mode = True, pname, mode
            result.rows_written, result.quality = len(df), report
            log.info("data refreshed", extra=fields(symbol=symbol, provider=pname, mode=mode,
                                                    rows=len(df), last=str(info.last_date),
                                                    sha256=info.sha256[:12],
                                                    quality=report.as_dict()))
            return result

        if not result.errors:
            result.errors.append("no provider configured/available for this symbol")
        log.error("all providers failed", extra=fields(symbol=symbol, errors=result.errors))
        return result

    def _fetch(
        self,
        symbol: str,
        provider: Provider,
        pname: ProviderName,
        ticker: str,
        meta: MetaInfo | None,
        last_complete: date,
    ) -> tuple[str, pd.DataFrame, QualityReport]:
        dcfg = self.cfg.data
        if meta is not None and meta.source == pname:
            start = meta.last_date - timedelta(days=dcfg.incremental_overlap_days)
            raw = self._raw(symbol, provider, pname, ticker, start)
            df, report = clean_candles(raw, self.calendar, last_complete, dcfg.spike_flag_pct)
            if self._overlap_consistent(symbol, df):
                return "incremental", df, report
            log.warning("overlap mismatch: re-downloading full history",
                        extra=fields(symbol=symbol, provider=pname))
        raw = self._raw(symbol, provider, pname, ticker, dcfg.history_start)
        df, report = clean_candles(raw, self.calendar, last_complete, dcfg.spike_flag_pct)
        mode = "replace" if meta is not None else "full"
        return mode, df, report

    def _overlap_consistent(self, symbol: str, new: pd.DataFrame) -> bool:
        old = self.cache.load(symbol, start=new.index[0].date())
        common = old.index.intersection(new.index)
        if len(common) == 0:
            return False  # no overlap at all -> cannot verify continuity
        rel = (new.loc[common, "close"] / old.loc[common, "close"] - 1).abs() * 100
        return bool((rel <= self.cfg.data.overlap_tolerance_pct).all())

    def refresh_reference(self, symbol: str) -> RefreshResult:
        """Fetch a recent window from a source other than the primary, for cross-checking."""
        symbol = symbol.upper()
        result = RefreshResult(symbol=symbol, ok=False)
        cc = self.cfg.data.cross_check
        meta = self.cache.meta(symbol)
        if not cc.enabled or meta is None:
            return result
        now = self.clock()
        last_complete = self.calendar.last_completed_session(now)
        ref, ref_source = self.cache.load_reference(symbol)
        primary_ticker = self.cfg.symbol(symbol).ticker_for(meta.source)  # type: ignore[arg-type]
        align = native_alignment(meta.source, primary_ticker)
        same_alignment = self.cache.flag(f"ref_alignment:{symbol}") == str(align)
        if (same_alignment and ref_source is not None and ref_source != meta.source
                and not ref.empty and ref.index[-1].date() >= last_complete):
            result.ok, result.source, result.mode = True, ref_source, "up-to-date"
            return result
        start = last_complete - timedelta(days=cc.reference_history_days)
        for pname, provider, ticker in self._usable(symbol, exclude=meta.source):
            try:
                raw = self._raw(symbol, provider, pname, ticker, start, align)
                df, report = clean_candles(raw, self.calendar, last_complete,
                                           self.cfg.data.spike_flag_pct)
            except Exception as exc:  # any provider failure => try the next provider
                result.errors.append(f"{pname}: {exc}")
                self.cache.log_failure(symbol, pname, now, f"reference: {exc}")
                continue
            self.cache.write_reference(symbol, df, pname)
            self.cache.set_flag(f"ref_alignment:{symbol}", str(align))
            result.ok, result.source, result.mode = True, pname, "reference"
            result.rows_written, result.quality = len(df), report
            return result
        log.warning("no reference source", extra=fields(symbol=symbol, errors=result.errors))
        return result

    # --------------------------------------------------------------------- read
    def get(self, symbol: str, refresh: bool = True) -> MarketData:
        symbol = symbol.upper()
        sym_cfg = self.cfg.symbol(symbol)
        if refresh:
            try:
                self.refresh(symbol)
                self.refresh_reference(symbol)
            except Exception as exc:  # never crash the caller on data problems
                log.exception("refresh crashed", extra=fields(symbol=symbol, error=str(exc)))
        return self.assess(symbol, is_proxy=sym_cfg.yahoo_is_proxy)

    def assess(self, symbol: str, is_proxy: bool = False) -> MarketData:
        now = self.clock()
        expected = self.calendar.last_completed_session(now)
        meta = self.cache.meta(symbol)
        candles = self.cache.load(symbol, end=expected) if meta else empty_frame()
        if meta is None or candles.empty:
            return MarketData(symbol, candles, DataStatus.UNAVAILABLE, expected, None, None, None,
                              None, f"{NO_SIGNAL_HEADLINE}: no cached data and no provider "
                              "responded.")
        last_bar = candles.index[-1].date()
        lag = self.calendar.sessions_between(last_bar, expected)
        proxy = is_proxy and meta.source == "yfinance"
        base = (symbol, candles)
        if lag > self.cfg.data.max_signal_lag_sessions:
            msg = (f"{NO_SIGNAL_HEADLINE}: latest bar is {last_bar.isoformat()} but the "
                   f"{expected.isoformat()} session has closed ({lag} session(s) behind). "
                   "The chart shows historical data only.")
            return MarketData(*base, DataStatus.STALE, expected, last_bar, lag, meta.source,
                              meta.sha256, msg, proxy)

        check = self._check(symbol, candles, meta.source, last_bar)
        if check.status is CheckStatus.DISPUTED:
            msg = f"{NO_SIGNAL_HEADLINE}: data sources disagree. {check.message}"
            status = DataStatus.DISPUTED
        elif (check.status is CheckStatus.UNVERIFIED
              and self.cfg.data.cross_check.require_second_source):
            msg = (f"{NO_SIGNAL_HEADLINE}: latest bar not confirmed by a second source. "
                   f"{check.message}")
            status = DataStatus.UNVERIFIED
        else:
            msg = (f"Data current to the {last_bar.isoformat()} New York close "
                   f"({meta.source}). {check.message}")
            status = DataStatus.FRESH
        if proxy:
            msg += " Primary source is a futures proxy for spot gold."
        return MarketData(*base, status, expected, last_bar, lag, meta.source, meta.sha256,
                          msg.strip(), proxy, check)

    def _check(self, symbol: str, candles: pd.DataFrame, primary_source: str,
               last_bar: date) -> CrossCheck:
        cc = self.cfg.data.cross_check
        if not cc.enabled:
            return CrossCheck(CheckStatus.SKIPPED, None, None, [], "Cross-check disabled.")
        ref, ref_source = self.cache.load_reference(symbol)
        if ref_source == primary_source:  # never "verify" a source against itself
            ref, ref_source = ref.iloc[0:0], None
        return cross_check(candles, ref, ref_source, last_bar, cc.max_disagreement_atr,
                           cc.window_bars)
