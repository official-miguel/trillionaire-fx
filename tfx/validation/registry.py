"""Validation registry: which strategy-timeframe pairs may produce signals.

* A pair gives signals ONLY with a current PASS. No record, a FAIL, or a PASS older
  than ``max_age_days`` (the gate is re-run monthly) means locked.
* Every backtest trial is recorded so the Deflated Sharpe Ratio uses the true
  number of trials, across all timeframes.
* :meth:`timeframe_menu` drives the UI selector: validated timeframes are
  selectable; the others are shown as "not validated" (view only, no signals).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import DateTime, Integer, String, Text, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from tfx.core_types import Timeframe
from tfx.db import sqlite_engine

Verdict = Literal["PASS", "FAIL"]


class _Base(DeclarativeBase):
    pass


class ValidationResult(_Base):
    __tablename__ = "validation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(40), index=True)
    timeframe: Mapped[str] = mapped_column(String(4))
    verdict: Mapped[str] = mapped_column(String(4))
    run_id: Mapped[str] = mapped_column(String(64))
    validated_at: Mapped[datetime] = mapped_column(DateTime)
    reasons: Mapped[str] = mapped_column(Text, default="[]")
    metrics: Mapped[str] = mapped_column(Text, default="{}")


class Trial(_Base):
    __tablename__ = "backtest_trials"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    strategy: Mapped[str] = mapped_column(String(40))
    timeframe: Mapped[str] = mapped_column(String(4))
    params: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class MenuItem:
    timeframe: Timeframe
    validated: bool
    label: str


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None)


class ValidationRegistry:
    def __init__(self, url_or_path: str | Path = ":memory:", max_age_days: int = 40) -> None:
        p = str(url_or_path)
        if p != ":memory:" and not p.startswith("sqlite"):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        url = p if p.startswith("sqlite") else f"sqlite:///{p}"
        self.engine = sqlite_engine(url)
        _Base.metadata.create_all(self.engine)
        self.max_age = timedelta(days=max_age_days)

    def close(self) -> None:
        self.engine.dispose()

    # ---------------------------------------------------------------- writes
    def record_result(self, strategy: str, tf: Timeframe, verdict: Verdict, run_id: str,
                      when: datetime, reasons: list[str] | None = None,
                      metrics: dict[str, Any] | None = None) -> None:
        if verdict not in ("PASS", "FAIL"):
            raise ValueError("verdict must be PASS or FAIL")
        if verdict == "FAIL" and not reasons:
            raise ValueError("a FAIL must state its reasons")
        with Session(self.engine) as s, s.begin():
            s.add(ValidationResult(strategy=strategy, timeframe=tf.value, verdict=verdict,
                                   run_id=run_id, validated_at=_naive(when),
                                   reasons=json.dumps(reasons or []),
                                   metrics=json.dumps(metrics or {}, default=str)))

    def record_trial(self, run_id: str, strategy: str, tf: Timeframe, params: dict[str, Any],
                     when: datetime) -> None:
        with Session(self.engine) as s, s.begin():
            s.add(Trial(run_id=run_id, strategy=strategy, timeframe=tf.value,
                        params=json.dumps(params, sort_keys=True, default=str),
                        created_at=_naive(when)))

    # ----------------------------------------------------------------- reads
    def trial_count(self, strategy: str | None = None) -> int:
        """Distinct trials (strategy, timeframe, params), counted across ALL timeframes."""
        stmt = select(func.count(func.distinct(Trial.strategy + "|" + Trial.timeframe + "|"
                                               + Trial.params)))
        if strategy is not None:
            stmt = stmt.where(Trial.strategy == strategy)
        with Session(self.engine) as s:
            return int(s.scalar(stmt) or 0)

    def latest(self, strategy: str, tf: Timeframe) -> ValidationResult | None:
        stmt = (select(ValidationResult)
                .where(ValidationResult.strategy == strategy,
                       ValidationResult.timeframe == tf.value)
                .order_by(ValidationResult.validated_at.desc(), ValidationResult.id.desc())
                .limit(1))
        with Session(self.engine) as s:
            row = s.scalars(stmt).first()
            if row is not None:
                s.expunge(row)
            return row

    def gate(self, strategy: str, tf: Timeframe, now: datetime) -> GateDecision:
        """The ONLY way a strategy-timeframe pair is cleared to produce signals."""
        row = self.latest(strategy, tf)
        if row is None:
            return GateDecision(False, f"{strategy} on {tf}: not validated yet")
        if row.verdict != "PASS":
            reasons = "; ".join(json.loads(row.reasons)) or "failed validation"
            return GateDecision(False, f"{strategy} on {tf}: FAILED validation ({reasons})")
        age = _naive(now) - row.validated_at
        if age > self.max_age:
            return GateDecision(False, f"{strategy} on {tf}: validation expired "
                                       f"({age.days} days old; re-run the monthly gate)")
        return GateDecision(True, f"{strategy} on {tf}: validated {row.validated_at:%Y-%m-%d}")

    def timeframe_menu(self, strategy: str, candidates: list[Timeframe],
                       now: datetime) -> list[MenuItem]:
        items = []
        for tf in candidates:
            ok = self.gate(strategy, tf, now).allowed
            items.append(MenuItem(tf, ok, tf.value if ok else f"{tf.value} (not validated)"))
        return items
