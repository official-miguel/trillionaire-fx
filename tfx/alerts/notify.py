"""Alerts: Telegram (free bot) and local desktop notifications. Send-only.

The Telegram token and chat id are read from the local ``.env`` (never from the repo)
and are never logged. Every signal alert includes entry, stop and amount at risk.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import requests

from tfx.coach import assert_no_promises
from tfx.logging_setup import fields, get_logger
from tfx.store import AlertRow, Store

log = get_logger(__name__)
FOOTER = "Decision support only. Trading carries risk of loss."


def signal_message(ev: dict[str, Any]) -> str | None:
    card = ev.get("card", {})
    status = ev.get("status")
    if status in ("BUY", "SELL"):
        size = card.get("size") or {}
        order = card.get("order") or {}
        if card.get("entry") is None or card.get("stop") is None or not size:
            return None  # never send a signal alert without entry, stop and risk
        text = (f"{card['headline']}\nSetup: {card.get('setup', '')} (score "
                f"{(card.get('score') or {}).get('score', '-')})\n"
                f"Order: {order.get('text', '')}\nEntry ~ {card['entry']:.5g} | Stop "
                f"{card['stop']:.5g}\nAmount at risk: {size['amount_at_risk']} "
                f"{size['currency']} ({size['risk_pct']}%), {size['lots']} lots\n{FOOTER}")
    elif status == "EXIT":
        text = f"{card.get('headline')}\n{card.get('why', '')}\n{FOOTER}"
    elif status == "HOLD" and any(a["code"] == "TAKE_PARTIAL" for a in card.get("actions", [])):
        text = f"{ev['symbol']}: " + next(a["text"] for a in card["actions"]
                                          if a["code"] == "TAKE_PARTIAL") + f"\n{FOOTER}"
    else:
        return None
    assert_no_promises(text)
    return text


class Notifier:
    def __init__(self, store: Store,
                 http_post: Callable[..., Any] | None = None,
                 desktop: Callable[[str, str], bool] | None = None) -> None:
        self.store = store
        self._post = http_post or requests.post
        self._desktop = desktop or desktop_toast

    def telegram_configured(self) -> bool:
        return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))

    def send(self, kind: str, text: str) -> dict[str, bool]:
        assert_no_promises(text)
        out: dict[str, bool] = {}
        if self.telegram_configured():
            ok = False
            try:
                r = self._post(
                    f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
                    data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text}, timeout=15)
                ok = getattr(r, "status_code", 0) == 200
            except Exception as exc:  # never crash on alert failures
                log.warning("telegram failed", extra=fields(error=type(exc).__name__))
            self.store.add(AlertRow(channel="telegram", kind=kind, text=text, ok=ok))
            out["telegram"] = ok
        ok_d = False
        try:
            ok_d = self._desktop("TrillionaireFX", text)
        except Exception as exc:
            log.warning("desktop notification failed", extra=fields(error=type(exc).__name__))
        self.store.add(AlertRow(channel="desktop", kind=kind, text=text, ok=ok_d))
        out["desktop"] = ok_d
        return out


def desktop_toast(title: str, text: str) -> bool:  # pragma: no cover - OS integration
    """Windows balloon notification via PowerShell (no extra dependency)."""
    if sys.platform != "win32":
        return False
    safe_t = title.replace("'", "''")
    safe_b = text.replace("'", "''")[:240]
    ps = ("Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
          "$n = New-Object System.Windows.Forms.NotifyIcon; "
          "$n.Icon = [System.Drawing.SystemIcons]::Information; $n.Visible = $true; "
          f"$n.ShowBalloonTip(8000, '{safe_t}', '{safe_b}', "
          "[System.Windows.Forms.ToolTipIcon]::Info); Start-Sleep -Seconds 9; $n.Dispose()")
    subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return True
