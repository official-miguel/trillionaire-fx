"""RESEARCH-ONLY H1 history from the broker's MT5 terminal (for H1/H4 backtests).

Why a separate store
--------------------
Live signals need every bar confirmed by Yahoo AND MT5 (see ``intraday.py``).
Yahoo keeps only ~2 years of hourly data, which is too short to validate H1/H4
strategies with a 3-year untouched holdout. This module downloads the broker's
deeper MT5 H1 history into its own table (``research_h1``) that NO live code
reads, and checks how trustworthy it is:

1. **Hourly check** over the overlap with Yahoo H1 (about 2 years): the share of
   bars whose bar-to-bar change differs by more than the cross-check ATR limit.
2. **Daily check** over the full span: MT5 H1 rebuilt into the same daily bars as
   the stored Yahoo daily series (UTC-midnight days for FX pairs, New York-close
   days for gold futures), compared with the same rule back to 2003-2005.
3. **Coverage, gaps and outliers** against the FX trading-hours calendar.

All thresholds are pre-declared in ``config.yaml`` (``data.research``). A market
whose research data FAILS keeps its H1/H4 strategy pairs "not validated".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import DateTime, Float, String, Text, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Mapped, Session, mapped_column

from tfx.config import AppConfig
from tfx.data.cache import Base, CandleCache
from tfx.data.crosscheck import atr, disagreement_series
from tfx.data.intraday import IntradayService, clean_hourly, hourly_records
from tfx.data.providers import Alignment, ProviderError, native_alignment
from tfx.data.quality import DataQualityError
from tfx.data.timeframes import to_fx_time
from tfx.logging_setup import fields, get_logger

log = get_logger(__name__)

RESEARCH_SOURCE = "mt5"
COLS = ["open", "high", "low", "close", "volume", "spread_points"]


class ResearchRow(Base):
    __tablename__ = "research_h1"
    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)  # UTC open time
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_points: Mapped[float | None] = mapped_column(Float, nullable=True)


class ResearchVerdictRow(Base):
    __tablename__ = "research_verdicts"
    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    assessed_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    verdict: Mapped[str] = mapped_column(String(4))
    reasons: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)


# ------------------------------------------------------------------ helpers
def expected_trading_hours(start: pd.Timestamp, end: pd.Timestamp,
                           holidays: set[tuple[int, int]]) -> pd.DatetimeIndex:
    """UTC hour starts inside the FX week (Sun 17:00 to Fri 17:00 New York), minus holidays."""
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    fx = to_fx_time(hours)
    holiday = np.array([(d.month, d.day) in holidays for d in fx.date], dtype=bool)
    return hours[(fx.day_of_week < 5) & ~holiday]


def missing_runs(expected: pd.DatetimeIndex, have: pd.DatetimeIndex) -> list[int]:
    """Lengths (in hours) of consecutive runs of expected hours with no bar."""
    missing = expected.difference(have)
    if missing.empty:
        return []
    pos = expected.get_indexer(missing)
    breaks = np.where(np.diff(pos) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(pos) - 1]])
    return [int(e - s + 1) for s, e in zip(starts, ends, strict=True)]


def hourly_to_days(h1: pd.DataFrame, alignment: Alignment) -> pd.DataFrame:
    """Group UTC-indexed H1 bars into UTC-midnight, London-midnight or New York-close days."""
    idx = pd.DatetimeIndex(h1.index)
    if alignment is Alignment.UTC_MIDNIGHT:
        key = idx.tz_convert("UTC").normalize().tz_localize(None)
    elif alignment is Alignment.LONDON_MIDNIGHT:
        key = idx.tz_convert("Europe/London").normalize().tz_localize(None)
    else:
        key = to_fx_time(idx).normalize()  # FX date == New York closing date
    g = h1.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    out = out[out.index.day_of_week < 5]
    out.index.name = "date"
    return out


@dataclass(frozen=True)
class ResearchQuality:
    symbol: str
    verdict: str                       # PASS | FAIL | NO DATA
    reasons: list[str]
    first_bar: pd.Timestamp | None = None
    last_bar: pd.Timestamp | None = None
    bars: int = 0
    years: float = 0.0
    coverage_pct: float | None = None
    gaps: int = 0                      # missing runs of >= gap_hours
    longest_gap_hours: int = 0
    outliers: int = 0
    hourly_compared: int = 0
    hourly_disagree_pct: float | None = None
    daily_compared: int = 0
    daily_disagree_pct: float | None = None
    daily_check_from: date | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.verdict == "PASS"


class ResearchHistory:
    def __init__(self, config: AppConfig, cache: CandleCache, mt5: Any,
                 intraday: IntradayService, clock: Callable[[], datetime]) -> None:
        self.cfg = config
        self.cache = cache
        self.mt5 = mt5
        self.intraday = intraday
        self.clock = clock
        for table in (ResearchRow.__table__, ResearchVerdictRow.__table__):
            table.create(cache.engine, checkfirst=True)  # type: ignore[attr-defined]

    # ----------------------------------------------------------- storage
    def load(self, symbol: str) -> pd.DataFrame:
        stmt = (select(ResearchRow)
                .where(ResearchRow.symbol == symbol, ResearchRow.source == RESEARCH_SOURCE)
                .order_by(ResearchRow.time))
        with Session(self.cache.engine) as s:
            rows = s.scalars(stmt).all()
        idx = pd.DatetimeIndex([r.time for r in rows], name="time").tz_localize("UTC")
        return pd.DataFrame({c: [getattr(r, c) for r in rows] for c in COLS},
                            index=idx.as_unit("ns")).astype("float64")

    def _write(self, symbol: str, df: pd.DataFrame) -> None:
        recs = hourly_records(symbol, RESEARCH_SOURCE, df, COLS)
        stmt = sqlite_insert(ResearchRow)
        stmt = stmt.on_conflict_do_update(index_elements=["symbol", "source", "time"],
                                          set_={c: stmt.excluded[c] for c in COLS})
        with Session(self.cache.engine) as s, s.begin():
            s.execute(stmt, recs)

    def _first_time(self, symbol: str) -> datetime | None:
        stmt = select(func.min(ResearchRow.time)).where(ResearchRow.symbol == symbol)
        with Session(self.cache.engine) as s:
            v = s.scalar(stmt)
        return v.replace(tzinfo=UTC) if v is not None else None

    # ---------------------------------------------------------- download
    def download(self, symbol: str) -> str:
        """Fetch MT5 H1 history year by year from ``research.start`` (skips stored years)."""
        rcfg = self.cfg.data.research
        sym = self.cfg.symbol(symbol)
        ticker = sym.ticker_for("mt5")
        if ticker is None:
            return "no MT5 symbol configured"
        if not self.mt5.available():
            return f"MT5 not connected: {getattr(self.mt5, 'last_error', '') or ''}".strip()
        now = self.clock()
        have = self.load(symbol)
        stored_years = set(pd.DatetimeIndex(have.index).year) if not have.empty else set()
        total, empty_years = 0, 0
        for year in range(rcfg.start.year, now.year + 1):
            if year in stored_years and year < now.year - 1:
                continue  # complete past year already cached
            start = max(date(year, 1, 1), rcfg.start)
            end = min(date(year + 1, 1, 1), now.date() + timedelta(days=1))
            try:
                raw = self.mt5.fetch_hourly_range(ticker, start, end)
                df = clean_hourly(raw, now)
            except (ProviderError, DataQualityError) as exc:
                empty_years += 1
                log.info("research year empty", extra=fields(symbol=symbol, year=year,
                                                             error=str(exc)))
                continue
            df = df[(df.index >= pd.Timestamp(start, tz="UTC"))
                    & (df.index < pd.Timestamp(end, tz="UTC"))]
            if df.empty:
                empty_years += 1
                continue
            self.cache.log_raw(f"{symbol}@H1-research", RESEARCH_SOURCE, now, raw)
            self._write(symbol, df)
            total += len(df)
        return f"ok ({total} bars written, {empty_years} year(s) without data)"

    # ------------------------------------------------------------ assess
    def assess(self, symbol: str) -> ResearchQuality:
        rcfg = self.cfg.data.research
        limit = self.cfg.data.cross_check.max_disagreement_atr
        h1 = self.load(symbol)
        if h1.empty:
            return ResearchQuality(symbol, "NO DATA",
                                   ["no MT5 research history downloaded yet"])
        first, last = h1.index[0], h1.index[-1]
        years = (last - first).days / 365.25
        holidays = {(int(m), int(d)) for m, d in
                    (x.split("-") for x in self.cfg.data.non_session_days)}
        expected = expected_trading_hours(first, last, holidays)
        have = pd.DatetimeIndex(h1.index)
        coverage = float(len(expected.intersection(have)) / len(expected) * 100)
        runs = [r for r in missing_runs(expected, have) if r >= rcfg.gap_hours]
        moves = (h1["close"].diff().abs() / atr(h1, 20)).dropna()
        outliers = int((moves > rcfg.outlier_atr).sum())

        # 1. hourly check vs Yahoo H1 over the overlap
        yahoo_h1 = self.intraday.load_h1(symbol, "yfinance")
        dis_h = disagreement_series(yahoo_h1, h1) if not yahoo_h1.empty else pd.Series(
            dtype="float64")
        h_pct = float((dis_h > limit).mean() * 100) if len(dis_h) else None

        # 2. daily check vs the stored Yahoo daily series over the full span
        meta = self.cache.meta(symbol)
        daily = self.cache.load(symbol)
        d_pct, d_n, d_from = None, 0, None
        if meta is not None and not daily.empty:
            align = native_alignment(meta.source, self.cfg.symbol(symbol).ticker_for(
                meta.source)) or Alignment.UTC_MIDNIGHT  # type: ignore[arg-type]
            rebuilt = hourly_to_days(h1, align)
            dis_d = disagreement_series(daily, rebuilt)
            d_n = len(dis_d)
            if d_n:
                d_pct = float((dis_d > limit).mean() * 100)
                d_from = dis_d.index[0].date()

        reasons: list[str] = []
        if years < rcfg.min_years:
            reasons.append(f"only {years:.1f} years of H1 history (need {rcfg.min_years}: "
                           "training data plus the 3-year holdout)")
        if coverage < rcfg.min_coverage_pct:
            reasons.append(f"coverage {coverage:.1f}% < {rcfg.min_coverage_pct}%")
        if h_pct is None or len(dis_h) < rcfg.min_hourly_compared:
            reasons.append(f"too few hourly bars to compare with Yahoo ({len(dis_h)} < "
                           f"{rcfg.min_hourly_compared}); refresh intraday data first")
        elif h_pct > rcfg.max_disagree_pct:
            reasons.append(f"hourly disagreement with Yahoo {h_pct:.1f}% > "
                           f"{rcfg.max_disagree_pct}%")
        if d_pct is None or d_n < rcfg.min_daily_compared:
            reasons.append(f"too few daily bars to compare ({d_n} < {rcfg.min_daily_compared})")
        elif d_pct > rcfg.max_disagree_pct:
            reasons.append(f"daily disagreement with Yahoo {d_pct:.1f}% > "
                           f"{rcfg.max_disagree_pct}%")
        outlier_rate = outliers / max(len(h1), 1) * 100
        if outlier_rate > rcfg.max_outlier_pct:
            reasons.append(f"outlier rate {outlier_rate:.2f}% > {rcfg.max_outlier_pct}%")

        q = ResearchQuality(
            symbol=symbol, verdict="FAIL" if reasons else "PASS", reasons=reasons,
            first_bar=first, last_bar=last, bars=len(h1), years=years,
            coverage_pct=coverage, gaps=len(runs), longest_gap_hours=max(runs, default=0),
            outliers=outliers, hourly_compared=len(dis_h), hourly_disagree_pct=h_pct,
            daily_compared=d_n, daily_disagree_pct=d_pct, daily_check_from=d_from,
        )
        self._record(q)
        return q

    def _record(self, q: ResearchQuality) -> None:
        import json

        with Session(self.cache.engine) as s, s.begin():
            s.add(ResearchVerdictRow(
                symbol=q.symbol, assessed_at=self.clock().astimezone(UTC).replace(tzinfo=None),
                verdict=q.verdict, reasons=json.dumps(q.reasons),
                summary=json.dumps({"bars": q.bars, "years": round(q.years, 2),
                                    "coverage_pct": q.coverage_pct,
                                    "hourly_disagree_pct": q.hourly_disagree_pct,
                                    "daily_disagree_pct": q.daily_disagree_pct})))

    def usable_for_validation(self, symbol: str) -> bool:
        """True only if the latest assessment PASSED. The backtester must check this."""
        stmt = (select(ResearchVerdictRow.verdict).where(ResearchVerdictRow.symbol == symbol)
                .order_by(ResearchVerdictRow.assessed_at.desc()).limit(1))
        with Session(self.cache.engine) as s:
            return s.scalar(stmt) == "PASS"
