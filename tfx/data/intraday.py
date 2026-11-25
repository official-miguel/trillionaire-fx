"""Intraday (H1, and H4 built from H1) data with two-source confirmation.

* Both Yahoo (primary, ~730 days of 1h history) and the MT5 terminal (reference,
  broker history plus the broker's per-bar spread) are stored side by side.
* A bar is CONFIRMED only when both sources have it and their bar-to-bar change
  agrees within ``max_disagreement_atr`` x ATR(20) of that timeframe — the same
  rule as daily data. Bars that only one source has are UNCONFIRMED, never used
  for signals or for the strategy proof phase.
* Only closed bars are kept: an hourly bar is stored once its hour has ended.
* The broker's spread history (``spread_points``) feeds the "spread by hour"
  cost model and the execution-quality score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sqlalchemy import DateTime, Float, String, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Mapped, Session, mapped_column

from tfx.config import AppConfig
from tfx.core_types import Timeframe
from tfx.data.cache import Base, CandleCache
from tfx.data.crosscheck import disagreement_series
from tfx.data.providers import ProviderError
from tfx.data.quality import DataQualityError
from tfx.data.timeframes import resample_hourly
from tfx.logging_setup import fields, get_logger

log = get_logger(__name__)

INTRADAY_SOURCES = ("yfinance", "mt5")  # primary first, reference second
PRICE = ["open", "high", "low", "close"]


class HourlyProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def fetch_hourly(self, ticker: str, start: date) -> pd.DataFrame: ...


class HourlyRow(Base):
    __tablename__ = "intraday_h1"
    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)  # UTC open time
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_points: Mapped[float | None] = mapped_column(Float, nullable=True)


def clean_hourly(raw: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Validate H1 bars: numeric, positive, unique, sorted, CLOSED, consistent extremes."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataQualityError("no hourly rows")
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if any(c not in df.columns for c in PRICE):
        raise DataQualityError("hourly frame missing OHLC columns")
    for c in ("volume", "spread_points"):
        if c not in df.columns:
            df[c] = np.nan
    df = df[[*PRICE, "volume", "spread_points"]].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).as_unit("ns")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = df[PRICE].isna().any(axis=1) | (df[PRICE] <= 0).any(axis=1)
    df = df[~bad]
    closed = df.index + pd.Timedelta(hours=1) <= pd.Timestamp(now).tz_convert("UTC")
    df = df[closed]
    df = df.assign(high=df[["open", "high", "close"]].max(axis=1),
                   low=df[["open", "low", "close"]].min(axis=1))
    if df.empty:
        raise DataQualityError("no closed, valid hourly bars")
    df.index.name = "time"
    return df


def hourly_records(symbol: str, source: str, df: pd.DataFrame,
                   cols: list[str]) -> list[dict[str, Any]]:
    """Fast row dicts for bulk upsert (NaN -> None, UTC-aware index -> naive UTC)."""
    frame = df[cols].astype("float64").astype(object)
    frame = frame.where(frame.notna(), None)
    times = pd.DatetimeIndex(df.index).tz_convert("UTC").tz_localize(None).to_pydatetime()
    recs: list[dict[str, Any]] = frame.to_dict("records")
    for rec, ts in zip(recs, times, strict=True):
        rec["symbol"], rec["source"], rec["time"] = symbol, source, ts
    return recs


@dataclass(frozen=True)
class IntradayData:
    symbol: str
    timeframe: Timeframe
    bars: pd.DataFrame          # primary bars + `confirmed` (bool) + `disagreement_atr`
    latest_bar: pd.Timestamp | None
    latest_confirmed: bool
    confirmed_from: pd.Timestamp | None
    confirmed_pct: float | None
    message: str

    @property
    def signal_allowed(self) -> bool:
        return self.latest_confirmed


class IntradayService:
    def __init__(self, config: AppConfig, cache: CandleCache,
                 providers: Mapping[str, Any], clock: Callable[[], datetime]) -> None:
        self.cfg = config
        self.cache = cache
        self.providers = providers
        self.clock = clock
        HourlyRow.__table__.create(cache.engine, checkfirst=True)  # type: ignore[attr-defined]

    # ------------------------------------------------------------- storage
    def _last_time(self, symbol: str, source: str) -> datetime | None:
        stmt = select(func.max(HourlyRow.time)).where(HourlyRow.symbol == symbol,
                                                      HourlyRow.source == source)
        with Session(self.cache.engine) as s:
            value = s.scalar(stmt)
        return value.replace(tzinfo=UTC) if value is not None else None

    def _write(self, symbol: str, source: str, df: pd.DataFrame) -> None:
        recs = hourly_records(symbol, source, df, [*PRICE, "volume", "spread_points"])
        stmt = sqlite_insert(HourlyRow)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "source", "time"],
            set_={c: stmt.excluded[c] for c in [*PRICE, "volume", "spread_points"]})
        with Session(self.cache.engine) as s, s.begin():
            s.execute(stmt, recs)

    def load_h1(self, symbol: str, source: str) -> pd.DataFrame:
        stmt = (select(HourlyRow).where(HourlyRow.symbol == symbol, HourlyRow.source == source)
                .order_by(HourlyRow.time))
        with Session(self.cache.engine) as s:
            rows = s.scalars(stmt).all()
        idx = pd.DatetimeIndex([r.time for r in rows], name="time").tz_localize("UTC")
        return pd.DataFrame({
            "open": [r.open for r in rows], "high": [r.high for r in rows],
            "low": [r.low for r in rows], "close": [r.close for r in rows],
            "volume": [np.nan if r.volume is None else r.volume for r in rows],
            "spread_points": [np.nan if r.spread_points is None else r.spread_points
                              for r in rows],
        }, index=idx.as_unit("ns")).astype("float64")

    # ------------------------------------------------------------- refresh
    def refresh(self, symbol: str) -> dict[str, str]:
        """Update H1 bars from every intraday source. Returns {source: outcome}."""
        icfg = self.cfg.data.intraday
        sym = self.cfg.symbol(symbol)
        now = self.clock()
        outcome: dict[str, str] = {}
        for source in INTRADAY_SOURCES:
            prov = self.providers.get(source)
            ticker = sym.ticker_for(source)  # type: ignore[arg-type]
            if prov is None or ticker is None or not hasattr(prov, "fetch_hourly"):
                outcome[source] = "not configured"
                continue
            if not prov.available():
                outcome[source] = f"unavailable: {getattr(prov, 'last_error', '') or ''}".strip()
                continue
            last = self._last_time(symbol, source)
            start = ((last - timedelta(days=2)).date() if last is not None
                     else (now - timedelta(days=icfg.history_days)).date())
            try:
                raw = prov.fetch_hourly(ticker, start)
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    self.cache.log_raw(f"{symbol}@H1", source, now, raw)
                df = clean_hourly(raw, now)
            except Exception as exc:  # never crash; record and move on
                msg = f"{type(exc).__name__}: {exc}" if not isinstance(
                    exc, (ProviderError, DataQualityError)) else str(exc)
                self.cache.log_failure(f"{symbol}@H1", source, now, msg)
                outcome[source] = f"failed: {msg}"
                continue
            self._write(symbol, source, df)
            outcome[source] = f"ok ({len(df)} bars)"
        log.info("intraday refresh", extra=fields(symbol=symbol, outcome=outcome))
        return outcome

    # ---------------------------------------------------------------- read
    def bars(self, symbol: str, tf: Timeframe = Timeframe.H1) -> IntradayData:
        if tf not in (Timeframe.H1, Timeframe.H4):
            raise ValueError("intraday bars are H1 or H4")
        now = pd.Timestamp(self.clock()).tz_convert("UTC")
        primary, reference = (self.load_h1(symbol, s) for s in INTRADAY_SOURCES)
        if tf is Timeframe.H4:
            primary = resample_hourly(primary, now) if not primary.empty else primary
            reference = resample_hourly(reference, now) if not reference.empty else reference
        if primary.empty:
            return IntradayData(symbol, tf, primary, None, False, None, None,
                                "DATA UNAVAILABLE - NO SIGNAL: no intraday bars from "
                                f"{INTRADAY_SOURCES[0]}.")
        limit = self.cfg.data.cross_check.max_disagreement_atr
        dis = disagreement_series(primary, reference) if not reference.empty else \
            pd.Series(dtype="float64")
        out = primary.copy()
        out["disagreement_atr"] = dis.reindex(out.index)
        out["confirmed"] = out["disagreement_atr"] <= limit
        latest = out.index[-1]
        ok = bool(out["confirmed"].iloc[-1])
        confirmed_idx = out.index[out["confirmed"]]
        pct = float(out["confirmed"].mean() * 100)
        if ok:
            msg = f"Latest {tf} bar {latest:%Y-%m-%d %H:%M} UTC confirmed by MT5."
        elif reference.empty:
            msg = ("DATA UNAVAILABLE - NO SIGNAL: no MT5 intraday bars to confirm against. "
                   "Open MetaTrader 5 and sign in.")
        elif pd.isna(out["disagreement_atr"].iloc[-1]):
            msg = (f"DATA UNAVAILABLE - NO SIGNAL: MT5 has no comparable {tf} bar for "
                   f"{latest:%Y-%m-%d %H:%M} UTC.")
        else:
            msg = (f"DATA UNAVAILABLE - NO SIGNAL: sources disagree on the latest {tf} bar "
                   f"by {out['disagreement_atr'].iloc[-1]:.2f} ATR (limit {limit:.2f}).")
        return IntradayData(symbol, tf, out, latest, ok,
                            confirmed_idx[0] if len(confirmed_idx) else None, pct, msg)

    def spread_by_hour(self, symbol: str) -> pd.Series:
        """Median broker spread (points) by UTC hour of day, from MT5 history."""
        ref = self.load_h1(symbol, "mt5")
        if ref.empty or ref["spread_points"].isna().all():
            return pd.Series(dtype="float64", name="spread_points")
        by_hour = ref["spread_points"].groupby(pd.DatetimeIndex(ref.index).hour).median()
        by_hour.index.name = "utc_hour"
        return by_hour.rename("spread_points")
