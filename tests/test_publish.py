"""Public view-only site: no personal data can be published, and pushes are minimal."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import tfx.publish as pub
from tests.test_api_flows import Env, env  # noqa: F401  (fixture re-export)
from tfx.mindset import CHECKLIST


def _all_keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _all_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _all_keys(v)}
    return set()


def test_scrub_removes_private_keys_at_any_depth() -> None:
    raw = {"card": {"size": {"lots": 1}, "setup": {"box": {"risk_amount": 5, "rr": 2}}},
           "markers": [1], "rows": [{"balance": 9, "ok": 1}], "signal_id": 3}
    out = pub.scrub(raw)
    assert out == {"card": {"setup": {"box": {"rr": 2}}}, "rows": [{"ok": 1}]}


def test_published_site_has_no_personal_data(env: Env, tmp_path: Path,  # noqa: F811
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    c = env.client
    # put personal data into the REAL account: a trade, a journal note, balance, a drawing
    assert c.put("/api/settings", json={"balance": 98765, "risk_pct": 0.75}).status_code == 200
    c.post("/api/emotions", json={"confidence": 3, "stress": 2, "urgency": 2})
    sig = c.get("/api/signal/EURUSD").json()
    t = c.post("/api/trades", json={"signal_id": sig.get("signal_id"), "symbol": "EURUSD",
                                    "direction": 1, "entry": 1.10, "stop": 1.09,
                                    "checklist": {k: True for k in CHECKLIST}})
    assert t.status_code == 200, t.text
    c.post("/api/journal", json={"kind": "note", "text": "SECRET-JOURNAL-TEXT"})
    c.post("/api/drawings/EURUSD", json={"kind": "hline", "data": {"price": 1.2345678}})
    assert c.get("/api/signal/EURUSD").json()["status"] in ("HOLD", "EXIT")

    res = pub.build(env.svc, tmp_path, intraday_zones=False)
    assert set(res["symbols"]) == {"EURUSD", "GBPUSD"}
    files = list((tmp_path / "data").rglob("*.json"))
    assert len(files) >= 2 + 2 * (2 + 3)
    blob = "".join(p.read_text(encoding="utf-8") for p in files)
    for secret in ("SECRET-JOURNAL-TEXT", "1.2345678", "HOLD", "trade_r_now"):
        assert secret not in blob, secret
    assert not re.search(r"(?<![\d.])98765(\.0+)?(?!\d)", blob)     # the balance
    keys: set[str] = set()
    for p in files:
        keys |= _all_keys(json.loads(p.read_text(encoding="utf-8")))
    assert not keys & pub.PRIVATE_KEYS
    meta = json.loads((tmp_path / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["static"] and "Yahoo" in meta["live_note"]
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert pub.STATIC_FLAG in html and "/static/" not in html
    assert html.index(pub.STATIC_FLAG) < html.index('src="/app.js"')
    assert (tmp_path / "vendor").is_dir() and (tmp_path / "app.js").exists()
    # same data -> same fingerprint (publish time ignored) -> nothing to push
    again = pub.build(env.svc, tmp_path, intraday_zones=False)
    assert again["digest"] == res["digest"]
    assert pub.push(res["digest"], again) == "unchanged: nothing to push"
    assert pub.previous_digest(tmp_path) == again["digest"]
    assert pub.previous_digest(tmp_path / "missing") is None


def test_push_commits_only_site(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(pub, "_git", lambda *a: calls.append(a) or R())
    assert pub.push("old", {"digest": "new", "published_at": "t"}) == "pushed"
    assert calls[0] == ("add", "--", "site")
    assert calls[1][:1] == ("commit",) and calls[1][-2:] == ("--", "site")
    assert calls[2] == ("push",)
