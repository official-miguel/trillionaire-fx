"""SQLite candle cache with a data-version log (SQLAlchemy 2.x).

* ``candles`` — one row per (symbol, date). A symbol's series always comes from a
  single source; switching source replaces the whole series so prices from
  different feeds are never spliced together.
* ``series_meta`` — current source, coverage and content hash per symbol.
* ``data_versions`` — append-only log of every fetch attempt (ok or failed), so any
  backtest or signal can be traced to the exact data it used.
* ``raw_fetches`` — IMMUTABLE log of every provider response exactly as received
  (zlib-compressed CSV + SHA-256). SQLite triggers reject UPDATE and DELETE.
* ``ref_candles`` — a recent window from a *second* source, used only to
  cross-check the primary series (see :mod:`tfx.data.crosscheck`).
"""

from __future__ import annotations

import hashlib
import io
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    delete,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from tfx.data.quality import COLUMNS, empty_frame, frame_hash
from tfx.db import sqlite_engine


class Base(DeclarativeBase):
    pass


class CandleRow(Base):
    __tablename__ = "candles"
    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(16))


class SeriesMeta(Base):
    __tablename__ = "series_meta"
    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    source: Mapped[str] = mapped_column(String(16))
    last_fetch_utc: Mapped[datetime] = mapped_column(DateTime)
    first_date: Mapped[date] = mapped_column(Date)
    last_date: Mapped[date] = mapped_column(Date)
    rows: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))


class DataVersion(Base):
    __tablename__ = "data_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(12), index=True)
    source: Mapped[str] = mapped_column(String(16))
    fetched_at_utc: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(8))  # ok | error
    mode: Mapped[str] = mapped_column(String(12))  # full | incremental | replace
    rows: Mapped[int] = mapped_column(Integer, default=0)
    first_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class RefCandleRow(Base):
    __tablename__ = "ref_candles"
    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(16))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)


class RawFetch(Base):
    __tablename__ = "raw_fetches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(12), index=True)
    source: Mapped[str] = mapped_column(String(16))
    fetched_at_utc: Mapped[datetime] = mapped_column(DateTime)
    rows: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[bytes] = mapped_column(LargeBinary)  # zlib(CSV as received)


class AppFlag(Base):
    """Small key/value store, e.g. how each cached series was constructed."""

    __tablename__ = "app_flags"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


_IMMUTABLE_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS raw_fetches_no_update BEFORE UPDATE ON raw_fetches
       BEGIN SELECT RAISE(ABORT, 'raw_fetches is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS raw_fetches_no_delete BEFORE DELETE ON raw_fetches
       BEGIN SELECT RAISE(ABORT, 'raw_fetches is immutable'); END""",
]


@dataclass(frozen=True)
class MetaInfo:
    symbol: str
    source: str
    last_fetch_utc: datetime
    first_date: date
    last_date: date
    rows: int
    sha256: str


def _naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _rows_to_frame(rows: Sequence[CandleRow] | Sequence[RefCandleRow]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [np.nan if r.volume is None else r.volume for r in rows],
        },
        index=pd.DatetimeIndex([pd.Timestamp(r.date) for r in rows], name="date").as_unit("ns"),
    ).astype("float64")


def _records(symbol: str, df: pd.DataFrame, source: str) -> list[dict[str, object]]:
    return [
        {
            "symbol": symbol, "date": ts.date(), "source": source,
            **{c: (None if pd.isna(getattr(row, c)) else float(getattr(row, c)))
               for c in COLUMNS},
        }
        for ts, row in zip(df.index, df[COLUMNS].itertuples(index=False), strict=True)
    ]


class CandleCache:
    def __init__(self, url_or_path: str | Path = ":memory:") -> None:
        if isinstance(url_or_path, Path) or not str(url_or_path).startswith("sqlite"):
            p = str(url_or_path)
            if p != ":memory:":
                Path(p).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{p}"
        else:
            url = str(url_or_path)
        self.engine: Engine = sqlite_engine(url)
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            for ddl in _IMMUTABLE_TRIGGERS:
                conn.execute(text(ddl))

    def close(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------ reads
    def load(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        stmt = select(CandleRow).where(CandleRow.symbol == symbol)
        if start is not None:
            stmt = stmt.where(CandleRow.date >= start)
        if end is not None:
            stmt = stmt.where(CandleRow.date <= end)
        stmt = stmt.order_by(CandleRow.date)
        with Session(self.engine) as s:
            rows = s.scalars(stmt).all()
        if not rows:
            return empty_frame()
        return _rows_to_frame(rows)

    def load_reference(self, symbol: str) -> tuple[pd.DataFrame, str | None]:
        stmt = select(RefCandleRow).where(RefCandleRow.symbol == symbol).order_by(RefCandleRow.date)
        with Session(self.engine) as s:
            rows = s.scalars(stmt).all()
        if not rows:
            return empty_frame(), None
        return _rows_to_frame(rows), rows[0].source

    def raw_fetches(self, symbol: str | None = None) -> list[RawFetch]:
        stmt = select(RawFetch).order_by(RawFetch.id)
        if symbol:
            stmt = stmt.where(RawFetch.symbol == symbol)
        with Session(self.engine) as s:
            out = list(s.scalars(stmt).all())
            s.expunge_all()
        return out

    @staticmethod
    def decode_raw(row: RawFetch) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(zlib.decompress(row.payload).decode()), index_col=0)

    def meta(self, symbol: str) -> MetaInfo | None:
        with Session(self.engine) as s:
            m = s.get(SeriesMeta, symbol)
            if m is None:
                return None
            return MetaInfo(m.symbol, m.source, m.last_fetch_utc.replace(tzinfo=UTC),
                            m.first_date, m.last_date, m.rows, m.sha256)

    def versions(self, symbol: str | None = None, limit: int = 50) -> list[DataVersion]:
        stmt = select(DataVersion).order_by(DataVersion.id.desc()).limit(limit)
        if symbol:
            stmt = stmt.where(DataVersion.symbol == symbol)
        with Session(self.engine) as s:
            out = list(s.scalars(stmt).all())
            s.expunge_all()
        return out

    # ----------------------------------------------------------------- writes
    def write(self, symbol: str, df: pd.DataFrame, source: str, mode: str,
              fetched_at: datetime) -> MetaInfo:
        """Upsert (``incremental``/``full``) or wholesale replace (``replace``) a series."""
        if df.empty:
            raise ValueError("refusing to write an empty frame")
        records = _records(symbol, df, source)
        with Session(self.engine) as s, s.begin():
            existing = s.get(SeriesMeta, symbol)
            if mode == "replace" or (existing is not None and existing.source != source):
                s.execute(delete(CandleRow).where(CandleRow.symbol == symbol))
            stmt = sqlite_insert(CandleRow)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "date"],
                set_={c: stmt.excluded[c] for c in [*COLUMNS, "source"]},
            )
            s.execute(stmt, records)  # executemany: one compiled statement
        full = self.load(symbol)
        sha = frame_hash(full)
        info = MetaInfo(symbol, source, fetched_at.astimezone(UTC), full.index[0].date(),
                        full.index[-1].date(), len(full), sha)
        with Session(self.engine) as s, s.begin():
            s.merge(SeriesMeta(symbol=symbol, source=source,
                               last_fetch_utc=_naive_utc(fetched_at),
                               first_date=info.first_date, last_date=info.last_date,
                               rows=info.rows, sha256=sha))
            s.add(DataVersion(symbol=symbol, source=source, fetched_at_utc=_naive_utc(fetched_at),
                              status="ok", mode=mode, rows=len(df),
                              first_date=df.index[0].date(), last_date=df.index[-1].date(),
                              sha256=sha))
        return info

    def flag(self, key: str) -> str | None:
        with Session(self.engine) as s:
            row = s.get(AppFlag, key)
            return row.value if row is not None else None

    def set_flag(self, key: str, value: str) -> None:
        with Session(self.engine) as s, s.begin():
            s.merge(AppFlag(key=key, value=value))

    def write_reference(self, symbol: str, df: pd.DataFrame, source: str) -> None:
        """Replace the cross-check window for ``symbol`` with ``df`` from ``source``."""
        if df.empty:
            raise ValueError("refusing to write an empty reference frame")
        with Session(self.engine) as s, s.begin():
            s.execute(delete(RefCandleRow).where(RefCandleRow.symbol == symbol))
            s.execute(sqlite_insert(RefCandleRow), _records(symbol, df, source))

    def log_raw(self, symbol: str, source: str, fetched_at: datetime, raw: pd.DataFrame) -> str:
        """Append a provider response, exactly as received, to the immutable raw log."""
        csv = raw.to_csv().encode()
        sha = hashlib.sha256(csv).hexdigest()
        with Session(self.engine) as s, s.begin():
            s.add(RawFetch(symbol=symbol, source=source, fetched_at_utc=_naive_utc(fetched_at),
                           rows=len(raw), sha256=sha, payload=zlib.compress(csv, 6)))
        return sha

    def log_failure(self, symbol: str, source: str, fetched_at: datetime, detail: str) -> None:
        with Session(self.engine) as s, s.begin():
            s.add(DataVersion(symbol=symbol, source=source, fetched_at_utc=_naive_utc(fetched_at),
                              status="error", mode="-", rows=0, detail=detail[:2000]))
