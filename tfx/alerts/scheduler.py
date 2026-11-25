"""Daily jobs: evaluate after each UTC day completes, and a pre-session reminder.

* 00:35 UTC Mon-Sat: refresh data (Yahoo + MT5), evaluate every market, alert on new
  signals, exit triggers, stop-to-breakeven prompts, regime changes and health events.
* 06:30 UTC Mon-Fri: reminder of pending orders and levels.

Windows Task Scheduler alternative (if the app is not left running):
    schtasks /Create /SC DAILY /ST 03:35 /TN TrillionaireFX ^
      /TR "\\"<project>\\.venv\\Scripts\\python.exe\\" -m tfx daily"
(03:35 local time in Kenya = 00:35 UTC). MetaTrader 5 must be open and signed in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tfx.alerts.notify import Notifier, signal_message
from tfx.logging_setup import fields, get_logger
from tfx.pipeline import Pipeline

log = get_logger(__name__)


def run_daily(pipeline: Pipeline, notifier: Notifier, refresh: bool = True,
              intraday: Any = None) -> dict[str, Any]:
    sent, results = [], []
    store = pipeline.store
    for sym in pipeline.cfg.symbols:
        if intraday is not None and refresh:
            try:  # hourly bars feed the H1/H4 views and the spread-by-hour execution score
                intraday.refresh(sym)
            except Exception as exc:
                log.warning("intraday refresh failed", extra=fields(symbol=sym, error=str(exc)))
        try:
            ev = pipeline.evaluate(sym, refresh=refresh)
        except Exception as exc:  # one market failing must not stop the others
            log.exception("evaluate failed", extra=fields(symbol=sym, error=str(exc)))
            continue
        results.append(ev)
        key = f"alerted:{sym}:{ev['data'].get('last_bar')}:{ev['status']}"
        msg = signal_message(ev)
        if msg and not store.get_setting(key):
            notifier.send(ev["status"], msg)
            store.set_setting(key, True)
            sent.append(sym)
        regime = ev["card"]["regime"]["regime"]
        prev = store.get_setting(f"regime:{sym}")
        if prev and prev != regime:
            notifier.send("REGIME", f"{sym}: regime changed {prev} -> {regime}. "
                                    f"{ev['card']['regime']['guidance']}")
        store.set_setting(f"regime:{sym}", regime)
    store.set_setting("last_daily_run", datetime.now(UTC).isoformat())
    return {"evaluated": len(results), "alerts": sent}


def pre_session_reminder(pipeline: Pipeline, notifier: Notifier) -> str | None:
    pending = [s for s in pipeline.store.signals(limit=50) if s.status in ("BUY", "SELL")
               and not s.outcome]
    if not pending:
        return None
    lines = [f"{s.symbol} {s.status} {s.order_type} {s.order_price} stop {s.stop} "
             f"(expires {s.expiry})" for s in pending[:8]]
    text = "Pre-session reminder - pending orders:\n" + "\n".join(lines)
    notifier.send("REMINDER", text)
    return text


def start_scheduler(pipeline: Pipeline, notifier: Notifier,
                    intraday: Any = None) -> Any:  # pragma: no cover
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(run_daily, "cron", day_of_week="mon-sat", hour=0, minute=35,
                  args=[pipeline, notifier, True, intraday], id="daily", max_instances=1,
                  coalesce=True)
    sched.add_job(pre_session_reminder, "cron", day_of_week="mon-fri", hour=6, minute=30,
                  args=[pipeline, notifier], id="reminder", max_instances=1, coalesce=True)
    pub = pipeline.cfg.section("publish")
    if pub.get("enabled", False):   # public view-only site (pushes only when data changed)
        from tfx.publish import publish_job

        sched.add_job(publish_job, "cron", minute=50,
                      hour=f"*/{int(pub.get('every_hours', 3))}",
                      args=[pipeline.svc], id="publish", max_instances=1, coalesce=True)
    sched.start()
    return sched
