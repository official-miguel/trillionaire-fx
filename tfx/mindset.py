"""Trader mindset: discipline enforced in software.

Rules (cannot be bypassed without a logged DISCIPLINE BREACH):
* No entry is ever logged without a stop (never overridable).
* Stops may only be tightened toward profit (a widening attempt is refused + logged).
* Pre-trade checklist must be completed; emotion check-in first.
* Stress or urgency >= 4 -> "Wait. Re-read the plan." and a 15-minute delay.
* Max 3 new trades per week.
* After 2 consecutive losing days: 24-hour cooldown AND a journal review.
* Risk % is locked for the calendar month (it may be LOWERED, never raised).
* Off-signal trades are tagged DISCIPLINE BREACH.
* After the drawdown breaker or a big loss (<= -2R) a written reflection is required.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from tfx.store import EmotionRow, JournalRow, StopChangeRow, Store, TradeRow

PRINCIPLES = [
    "Protect capital first.",
    "Follow the plan, not emotion.",
    "One trade means nothing; the series is the edge.",
    "Losses are a business cost.",
    "Missing a trade is cheaper than breaking a rule.",
]

CHECKLIST = {
    "signal_from_todays_close": "The signal is from today's close.",
    "stop_at_displayed_level": "My stop is set at the displayed level.",
    "size_within_cap": "My position size is within the cap shown.",
    "spread_acceptable": "The spread is acceptable (not wider than 3x typical).",
    "no_news_30m": "There is no tier-1 news within 30 minutes (informational check).",
    "calm_not_chasing": "I am calm and I am not chasing a loss.",
}
MIN_REFLECTION_CHARS = 80


class DisciplineError(ValueError):
    """An action that the rules never allow (e.g. an entry without a stop)."""


@dataclass
class Gate:
    allowed: bool
    blockers: list[str] = field(default_factory=list)
    wait_until: datetime | None = None


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


class Mindset:
    def __init__(self, store: Store, cfg_mindset: dict[str, Any],
                 clock: Callable[[], datetime] | None = None) -> None:
        self.store = store
        self.c = cfg_mindset
        self.clock = clock or (lambda: datetime.now(UTC))

    # ---------------------------------------------------------- emotions
    def check_in(self, confidence: int, stress: int, urgency: int) -> Gate:
        for v in (confidence, stress, urgency):
            if not 1 <= int(v) <= 5:
                raise ValueError("emotion scores are 1-5")
        self.store.add(EmotionRow(at=_naive(self.clock()), confidence=confidence,
                                  stress=stress, urgency=urgency))
        return self.emotion_gate()

    def emotion_gate(self) -> Gate:
        rows = self.store.emotions()
        if not rows:
            return Gate(False, ["Complete the emotion check-in first."])
        last = rows[-1]
        thr = int(self.c.get("emotion_wait_threshold", 4))
        if last.stress >= thr or last.urgency >= thr:
            until = last.at + timedelta(minutes=int(self.c.get("emotion_wait_minutes", 15)))
            if _naive(self.clock()) < until:
                return Gate(False, ["Wait. Re-read the plan."], until.replace(tzinfo=UTC))
        return Gate(True)

    # ---------------------------------------------------------- rule state
    def trades_this_week(self) -> int:
        now = _naive(self.clock())
        monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0,
                                                               microsecond=0)
        return sum(1 for t in self.store.trades() if t.opened_at >= monday)

    def losing_day_streak(self) -> tuple[int, datetime | None]:
        by_day: dict[str, float] = {}
        last_loss: datetime | None = None
        for t in self.store.trades("closed"):
            if t.closed_at is None or t.r_result is None:
                continue
            k = t.closed_at.date().isoformat()
            by_day[k] = by_day.get(k, 0.0) + t.r_result
            if t.r_result < 0:
                last_loss = max(last_loss or t.closed_at, t.closed_at)
        streak = 0
        for k in sorted(by_day, reverse=True):
            if by_day[k] < 0:
                streak += 1
            else:
                break
        return streak, last_loss

    def cooldown(self) -> Gate:
        streak, last_loss = self.losing_day_streak()
        if streak < int(self.c.get("cooldown_after_losing_days", 2)) or last_loss is None:
            return Gate(True)
        until = last_loss + timedelta(hours=int(self.c.get("cooldown_hours", 24)))
        reviewed = any(j.at >= last_loss for j in self.store.journal("cooldown_review"))
        blockers = []
        if _naive(self.clock()) < until:
            blockers.append(f"Cooldown after {streak} losing days until {until:%Y-%m-%d %H:%M} "
                            "UTC.")
        if not reviewed:
            blockers.append("Write the cooldown journal review before the next entry.")
        return Gate(not blockers, blockers, until.replace(tzinfo=UTC) if blockers else None)

    def reflection_required(self) -> bool:
        flag = self.store.get_setting("reflection_required_since")
        if not flag:
            return False
        since = datetime.fromisoformat(flag)
        return not any(j.at >= since for j in self.store.journal("reflection"))

    def require_reflection(self, reason: str) -> None:
        self.store.set_setting("reflection_required_since", _naive(self.clock()).isoformat())
        self.store.set_setting("reflection_reason", reason)

    def submit_reflection(self, text: str) -> None:
        if len(text.strip()) < MIN_REFLECTION_CHARS:
            raise ValueError(f"Write at least {MIN_REFLECTION_CHARS} characters: what happened, "
                             "what the plan said, and what you will do next time.")
        self.store.add(JournalRow(at=_naive(self.clock()), kind="reflection", text=text))

    def submit_cooldown_review(self, text: str) -> None:
        if len(text.strip()) < 40:
            raise ValueError("The cooldown review needs at least 40 characters.")
        self.store.add(JournalRow(at=_naive(self.clock()), kind="cooldown_review", text=text))

    # -------------------------------------------------------------- risk lock
    def set_risk_pct(self, new_pct: float) -> float:
        """Risk is locked per calendar month: lowering is allowed, raising is not."""
        month = _naive(self.clock()).strftime("%Y-%m")
        lock = self.store.get_setting("risk_lock") or {}
        cur = self.store.get_setting("risk_pct")
        if lock.get("month") == month and cur is not None and new_pct > float(cur):
            self.store.breach("RISK_INCREASE_ATTEMPT",
                              f"Tried to raise risk from {cur}% to {new_pct}% mid-month.",
                              at=_naive(self.clock()))
            raise DisciplineError(f"Risk % is locked for {month}. You can lower it, not raise it.")
        self.store.set_setting("risk_pct", new_pct)
        self.store.set_setting("risk_lock", {"month": month, "value": new_pct})
        return new_pct

    # ------------------------------------------------------------- entries
    def entry_gate(self, checklist: dict[str, bool], max_per_week: int | None = None) -> Gate:
        blockers: list[str] = []
        missing = [CHECKLIST[k] for k in CHECKLIST if not checklist.get(k)]
        if missing:
            blockers += [f"Checklist not confirmed: {m}" for m in missing]
        eg = self.emotion_gate()
        blockers += eg.blockers
        cap = max_per_week or int(self.c.get("max_new_trades_per_week", 3))
        if self.trades_this_week() >= cap:
            blockers.append(f"Weekly limit reached ({cap} new trades).")
        cd = self.cooldown()
        blockers += cd.blockers
        if self.reflection_required():
            blockers.append("A written reflection is required before new entries.")
        wait = eg.wait_until or cd.wait_until
        return Gate(not blockers, blockers, wait)

    def open_trade(self, trade: TradeRow, checklist: dict[str, bool],
                   override_reason: str = "") -> TradeRow:
        if trade.stop is None or trade.initial_stop is None or trade.stop <= 0:
            raise DisciplineError("No entry is ever logged without a stop.")
        if trade.direction * (trade.entry - trade.stop) <= 0:
            raise DisciplineError("The stop must be on the losing side of the entry.")
        gate = self.entry_gate(checklist)
        if not gate.allowed and not override_reason:
            raise DisciplineError("; ".join(gate.blockers))
        trade.opened_at = _naive(self.clock())  # same clock as the weekly/cooldown rules
        trade.checklist = json.dumps(checklist)
        if self.store.emotions():
            e = self.store.emotions()[-1]
            trade.emotions = json.dumps({"confidence": e.confidence, "stress": e.stress,
                                         "urgency": e.urgency})
        trade.off_signal = trade.signal_id is None
        saved: TradeRow = self.store.save(trade)
        if trade.off_signal:
            self.store.breach("DISCIPLINE BREACH", "Off-signal trade (no system signal).",
                              saved.id, at=_naive(self.clock()))
        if not gate.allowed:
            self.store.breach("DISCIPLINE BREACH",
                              f"Rules overridden ({override_reason}): " + "; ".join(
                                  gate.blockers), saved.id, at=_naive(self.clock()))
        return saved

    def update_stop(self, trade_id: int, new_stop: float) -> TradeRow:
        t = self.store.trade(trade_id)
        if t is None:
            raise KeyError(trade_id)
        ok = t.direction * (new_stop - t.stop) >= 0
        self.store.add(StopChangeRow(trade_id=trade_id, at=_naive(self.clock()), old=t.stop,
                                     new=new_stop, accepted=ok,
                                     reason="" if ok else "widening refused"))
        if not ok:
            self.store.breach("STOP_WIDEN_ATTEMPT",
                              f"Tried to move the stop from {t.stop} to {new_stop}.", trade_id,
                              at=_naive(self.clock()))
            raise DisciplineError("Stops may only be tightened toward profit.")
        t.stop = new_stop
        saved: TradeRow = self.store.save(t)
        return saved

    def close_trade(self, trade_id: int, exit_price: float, followed_rules: bool,
                    lesson: str, mfe_r: float | None = None, mae_r: float | None = None
                    ) -> TradeRow:
        t = self.store.trade(trade_id)
        if t is None:
            raise KeyError(trade_id)
        risk = abs(t.entry - t.initial_stop)
        t.exit_price = exit_price
        t.r_result = t.direction * (exit_price - t.entry) / risk if risk else 0.0
        t.closed_at = _naive(self.clock())
        t.followed_rules = followed_rules
        t.lesson = lesson
        if mfe_r is not None:
            t.mfe_r = mfe_r
        if mae_r is not None:
            t.mae_r = mae_r
        saved: TradeRow = self.store.save(t)
        if not followed_rules:
            self.store.breach("EXIT_NOT_PER_RULES", "Exit did not follow the rules.", trade_id,
                              at=_naive(self.clock()))
        if t.r_result <= -2:
            self.require_reflection(f"Big loss on trade {trade_id} ({t.r_result:+.2f}R).")
        return saved

    # ------------------------------------------------------------- review
    def weekly_review(self, week_start: datetime | None = None) -> dict[str, Any]:
        now = _naive(self.clock())
        start = week_start or (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        trades = [t for t in self.store.trades() if start <= t.opened_at < end]
        breaches = [b for b in self.store.breaches() if start <= b.at < end]
        breached_ids = {b.trade_id for b in breaches if b.trade_id is not None}
        clean = [t for t in trades if t.id not in breached_ids and not t.off_signal]
        adherence = 100.0 * len(clean) / len(trades) if trades else 100.0
        closed = [t for t in trades if t.r_result is not None]
        kinds: dict[str, int] = {}
        for b in breaches:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
        best = "Followed every exit rule." if closed and all(
            t.followed_rules for t in closed) else (
            "Kept stops in place." if not kinds.get("STOP_WIDEN_ATTEMPT") else "")
        worst = max(kinds, key=lambda k: kinds[k]) if kinds else ""
        return {"week_start": start.date().isoformat(), "plan_adherence_pct": round(adherence, 1),
                "trades": len(trades), "breaches": len(breaches), "breach_kinds": kinds,
                "result_r": round(sum(t.r_result or 0 for t in closed), 2),
                "best_habit": best, "worst_habit": worst}
