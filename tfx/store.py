"""Application database: signals, trades, journal, discipline, drawings, settings.

Discipline records (breaches, stop changes, reflections) are append-only: SQLite
triggers reject UPDATE/DELETE so no rule can be bypassed without leaving a trace.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from tfx.db import sqlite_engine


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SignalRow(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    session_date: Mapped[str] = mapped_column(String(10), index=True)
    symbol: Mapped[str] = mapped_column(String(12), index=True)
    strategy: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(4), default="D1")
    status: Mapped[str] = mapped_column(String(12))   # BUY/SELL/WATCH/BLOCKED/WAIT
    direction: Mapped[int] = mapped_column(Integer, default=0)
    grade: Mapped[str] = mapped_column(String(8), default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    size_mult: Mapped[float] = mapped_column(Float, default=0.0)
    entry_ref: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_type: Mapped[str] = mapped_column(String(16), default="")
    order_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expiry: Mapped[str] = mapped_column(String(32), default="")
    reasons: Mapped[str] = mapped_column(Text, default="[]")     # block / skip reasons
    factors: Mapped[str] = mapped_column(Text, default="{}")
    outcome: Mapped[str] = mapped_column(String(16), default="")  # TAKEN / SKIPPED / ...
    rule_version: Mapped[str] = mapped_column(String(16), default="")
    rule_hash: Mapped[str] = mapped_column(String(64), default="")
    trace: Mapped[str] = mapped_column(Text, default="{}")   # "Why this signal" evidence


class TradeRow(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str] = mapped_column(String(8), default="paper")    # paper | live
    symbol: Mapped[str] = mapped_column(String(12))
    strategy: Mapped[str] = mapped_column(String(32), default="manual")
    direction: Mapped[int] = mapped_column(Integer)
    planned_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    planned_stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    planned_units: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry: Mapped[float] = mapped_column(Float)
    units: Mapped[float] = mapped_column(Float)
    initial_stop: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    risk_pct: Mapped[float] = mapped_column(Float)
    amount_at_risk: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_result: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_r: Mapped[float] = mapped_column(Float, default=0.0)
    mae_r: Mapped[float] = mapped_column(Float, default=0.0)
    partial_done: Mapped[bool] = mapped_column(Boolean, default=False)
    adds: Mapped[int] = mapped_column(Integer, default=0)
    off_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    checklist: Mapped[str] = mapped_column(Text, default="{}")
    emotions: Mapped[str] = mapped_column(Text, default="{}")
    notes: Mapped[str] = mapped_column(Text, default="")
    screenshot: Mapped[str] = mapped_column(Text, default="")   # data URL of the chart
    followed_rules: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    lesson: Mapped[str] = mapped_column(Text, default="")


class StopChangeRow(Base):
    __tablename__ = "stop_changes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(Integer, index=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    old: Mapped[float] = mapped_column(Float)
    new: Mapped[float] = mapped_column(Float)
    accepted: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text, default="")


class BreachRow(Base):
    __tablename__ = "breaches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    kind: Mapped[str] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(Text)
    trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class JournalRow(Base):
    __tablename__ = "journal"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    kind: Mapped[str] = mapped_column(String(24))   # note | cooldown_review | reflection | ...
    text: Mapped[str] = mapped_column(Text)
    trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class EmotionRow(Base):
    __tablename__ = "emotions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    confidence: Mapped[int] = mapped_column(Integer)
    stress: Mapped[int] = mapped_column(Integer)
    urgency: Mapped[int] = mapped_column(Integer)


class SettingRow(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class DrawingRow(Base):
    __tablename__ = "drawings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(12), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    data: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class FillRow(Base):
    __tablename__ = "fills"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String(12))
    side: Mapped[str] = mapped_column(String(6))       # entry | exit
    planned: Mapped[float] = mapped_column(Float)
    actual: Mapped[float] = mapped_column(Float)
    slippage_pips: Mapped[float] = mapped_column(Float)   # positive = worse than planned
    utc_hour: Mapped[int] = mapped_column(Integer)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AlertRow(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    channel: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    ok: Mapped[bool] = mapped_column(Boolean)


_APPEND_ONLY = ("breaches", "stop_changes", "journal", "emotions", "fills", "alerts")


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        p = str(path)
        if p != ":memory:" and not p.startswith("sqlite"):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        url = p if p.startswith("sqlite") else f"sqlite:///{p}"
        self.engine = sqlite_engine(url)
        Base.metadata.create_all(self.engine)
        self._migrate()
        with self.engine.begin() as c:
            for t in _APPEND_ONLY:
                for op in ("UPDATE", "DELETE"):
                    c.execute(text(f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE "
                                   f"{op} ON {t} BEGIN SELECT RAISE(ABORT, "
                                   f"'{t} is append-only'); END"))

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created (SQLite ALTER)."""
        with self.engine.begin() as c:
            for table in Base.metadata.sorted_tables:
                have = {r[1] for r in c.execute(text(f"PRAGMA table_info({table.name})"))}
                for col in table.columns:
                    if col.name in have:
                        continue
                    arg = getattr(col.default, "arg", None)
                    dflt = "" if arg is None or callable(arg) else " DEFAULT '" + str(arg) + "'"
                    ctype = col.type.compile(self.engine.dialect)
                    c.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {col.name} "
                                   f"{ctype}{dflt}"))

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def close(self) -> None:
        self.engine.dispose()

    # ---------------------------------------------------------------- generic
    def add(self, row: Any) -> Any:
        with self.session() as s, s.begin():
            s.add(row)
        return row

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.session() as s:
            row = s.get(SettingRow, key)
            return json.loads(row.value) if row is not None else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.session() as s, s.begin():
            s.merge(SettingRow(key=key, value=json.dumps(value, default=str)))

    def trades(self, status: str | None = None, mode: str | None = None) -> list[TradeRow]:
        stmt = select(TradeRow).order_by(TradeRow.opened_at)
        if status == "open":
            stmt = stmt.where(TradeRow.closed_at.is_(None))
        elif status == "closed":
            stmt = stmt.where(TradeRow.closed_at.is_not(None))
        if mode:
            stmt = stmt.where(TradeRow.mode == mode)
        with self.session() as s:
            return list(s.scalars(stmt).all())

    def trade(self, trade_id: int) -> TradeRow | None:
        with self.session() as s:
            return s.get(TradeRow, trade_id)

    def save(self, row: Any) -> Any:
        with self.session() as s, s.begin():
            merged = s.merge(row)
        return merged

    def signals(self, symbol: str | None = None, limit: int = 500) -> list[SignalRow]:
        stmt = select(SignalRow).order_by(SignalRow.id.desc()).limit(limit)
        if symbol:
            stmt = stmt.where(SignalRow.symbol == symbol)
        with self.session() as s:
            return list(s.scalars(stmt).all())

    def signal(self, signal_id: int) -> SignalRow | None:
        with self.session() as s:
            return s.get(SignalRow, signal_id)

    def breaches(self) -> list[BreachRow]:
        with self.session() as s:
            return list(s.scalars(select(BreachRow).order_by(BreachRow.id)).all())

    def breach(self, kind: str, detail: str, trade_id: int | None = None,
               at: datetime | None = None) -> BreachRow:
        row: BreachRow = self.add(BreachRow(kind=kind, detail=detail, trade_id=trade_id,
                                            at=at or utcnow()))
        return row

    def journal(self, kind: str | None = None) -> list[JournalRow]:
        stmt = select(JournalRow).order_by(JournalRow.id)
        if kind:
            stmt = stmt.where(JournalRow.kind == kind)
        with self.session() as s:
            return list(s.scalars(stmt).all())

    def emotions(self) -> list[EmotionRow]:
        with self.session() as s:
            return list(s.scalars(select(EmotionRow).order_by(EmotionRow.id)).all())

    def drawings(self, symbol: str) -> list[DrawingRow]:
        with self.session() as s:
            return list(s.scalars(select(DrawingRow).where(DrawingRow.symbol == symbol)
                                  .order_by(DrawingRow.id)).all())

    def delete_drawing(self, drawing_id: int) -> bool:
        with self.session() as s, s.begin():
            row = s.get(DrawingRow, drawing_id)
            if row is None:
                return False
            s.delete(row)
        return True

    def stop_changes(self, trade_id: int) -> list[StopChangeRow]:
        with self.session() as s:
            return list(s.scalars(select(StopChangeRow).where(
                StopChangeRow.trade_id == trade_id).order_by(StopChangeRow.id)).all())

    def fills(self) -> list[FillRow]:
        with self.session() as s:
            return list(s.scalars(select(FillRow).order_by(FillRow.id)).all())

    def alerts(self, limit: int = 100) -> list[AlertRow]:
        with self.session() as s:
            return list(s.scalars(select(AlertRow).order_by(AlertRow.id.desc())
                                  .limit(limit)).all())
