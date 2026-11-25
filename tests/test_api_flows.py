"""End-to-end flows through the API (what the web app does) on a throwaway database."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import tfx.pipeline as pipeline_mod
from tests.conftest import FakeProvider, MutableClock, synthetic_candles
from tfx.alerts.notify import Notifier, signal_message
from tfx.alerts.scheduler import pre_session_reminder, run_daily
from tfx.api.app import create_app
from tfx.config import AppConfig
from tfx.core_types import Timeframe
from tfx.data.cache import CandleCache
from tfx.data.service import MarketDataService
from tfx.mindset import CHECKLIST
from tfx.pipeline import Pipeline
from tfx.store import Store
from tfx.validation.registry import ValidationRegistry

NOW = datetime(2026, 11, 21, 1, 0, tzinfo=UTC)  # Friday's UTC day complete
FULL = synthetic_candles("2003-01-01", "2026-11-20", seed=12)


class Env:
    def __init__(self, cfg: AppConfig, validated: bool, fire: bool) -> None:
        self.cache = CandleCache(":memory:")
        provs = {"yfinance": FakeProvider("yfinance", FULL), "mt5": FakeProvider("mt5", FULL)}
        self.svc = MarketDataService(cfg, self.cache, provs, clock=MutableClock(NOW))
        for s in ("EURUSD", "GBPUSD"):
            self.svc.get(s, refresh=True)
        self.store = Store(":memory:")
        self.reg = ValidationRegistry(":memory:")
        if validated:
            from tfx.versioning import rule_hash
            self.reg.record_result("trend", Timeframe.D1, "PASS", "t", NOW,
                                   metrics={"rule_hash": rule_hash("trend", cfg)})
        self.pipe = Pipeline(self.svc, self.store, self.reg)
        self.sent: list[str] = []
        self.notifier = Notifier(self.store, desktop=lambda t, b: self.sent.append(b) or True)
        cfg.symbols = {k: v for k, v in cfg.symbols.items() if k in ("EURUSD", "GBPUSD")}
        self.client = TestClient(create_app(self.pipe, self.notifier, None,
                                            clock=lambda: NOW))
        self.fire = fire


@pytest.fixture
def env(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> Any:
    real = pipeline_mod.signals_for

    def fake(strategy: str, f: pd.DataFrame, p: Any) -> pd.DataFrame:
        out = real(strategy, f, p)
        if strategy == "trend":
            out = out.copy()
            out.iloc[-1, out.columns.get_loc("direction")] = 1
            out.iloc[-1, out.columns.get_loc("stop")] = float(f["close"].iloc[-1] - 0.01)
            out.iloc[-1, out.columns.get_loc("grade")] = "A"
            out.iloc[-1, out.columns.get_loc("confidence")] = 1.0
            out.iloc[-1, out.columns.get_loc("reason")] = "test breakout"
        else:
            out = out.copy()
            out["direction"] = 0
            out["grade"] = ""
        return out

    monkeypatch.setattr(pipeline_mod, "signals_for", fake)
    e = Env(cfg, validated=True, fire=True)
    yield e
    e.store.close()


def test_meta_health_and_localhost_only(env: Env) -> None:
    c = env.client
    assert c.get("/api/health").json()["ok"]
    meta = c.get("/api/meta").json()
    assert meta["risk_notice"].startswith("Educational decision support")
    trend_menu = {m["timeframe"]: m for m in meta["timeframes"]["trend"]}
    assert trend_menu["D1"]["validated"] and trend_menu["H4"]["label"] == "H4 (not validated)"
    assert c.get("/api/health", headers={"host": "evil.example.com"}).status_code == 400


def test_flow_select_symbol_view_signal_checklist_log_trade(env: Env) -> None:
    c = env.client
    wl = c.get("/api/watchlist").json()
    assert {r["symbol"] for r in wl} == {"EURUSD", "GBPUSD"}
    candles = c.get("/api/candles/EURUSD?tf=D1").json()
    assert len(candles["candles"]) > 1000 and "ema_200" in candles["overlays"]
    assert candles["validated_view"] and candles["regimes"]
    assert "no signals" in c.get("/api/candles/EURUSD?tf=W1").json()["note"]
    sig = c.get("/api/signal/EURUSD").json()
    assert sig["status"] == "BUY", sig["card"].get("blocks")
    card = sig["card"]
    assert candles.get("setup") is None or candles["setup"]["box"]["entry"] == card["entry"]
    if sig["status"] == "BUY":
        assert card["size"]["amount_at_risk"] > 0 and card["order"]["order_type"] == "STOP-ENTRY"
        assert card["headline"] == "BUY EURUSD at next open"
        assert sig["signal_id"]
    # emotion check-in, then checklist and log
    r = c.post("/api/emotions", json={"confidence": 3, "stress": 2, "urgency": 2}).json()
    assert r["allowed"]
    body = {"signal_id": sig.get("signal_id"), "symbol": "EURUSD", "direction": 1,
            "entry": 1.10, "stop": 1.09, "checklist": {k: True for k in CHECKLIST}}
    t = c.post("/api/trades", json=body)
    assert t.status_code == 200, t.text
    trade = t.json()
    assert trade["amount_at_risk"] == pytest.approx(5.0) and trade["mode"] == "paper"
    assert c.get("/api/signal/EURUSD").json()["status"] in ("HOLD", "EXIT")
    # stops: tighten ok, widen refused
    assert c.patch(f"/api/trades/{trade['id']}/stop", json={"stop": 1.095}).status_code == 200
    bad = c.patch(f"/api/trades/{trade['id']}/stop", json={"stop": 1.08})
    assert bad.status_code == 409 and "tightened" in bad.json()["detail"]
    closed = c.post(f"/api/trades/{trade['id']}/close",
                    json={"exit_price": 1.12, "followed_rules": True, "lesson": "patience"})
    assert closed.status_code == 200
    rv = closed.json()["review"]
    assert rv["r_result"] == pytest.approx(2.0) and rv["plan_adherence"] is True
    assert c.get(f"/api/trades/{trade['id']}/review").json()["lesson"] == "patience"
    assert c.get("/api/settings").json()["balance"] == pytest.approx(1010.0)


def test_entry_without_checklist_or_stop_is_refused(env: Env) -> None:
    c = env.client
    c.post("/api/emotions", json={"confidence": 3, "stress": 1, "urgency": 1})
    no_check = {"symbol": "EURUSD", "direction": 1, "entry": 1.10, "stop": 1.09}
    assert c.post("/api/trades", json=no_check).status_code == 409
    wrong_stop = {**no_check, "stop": 1.12, "checklist": {k: True for k in CHECKLIST}}
    assert c.post("/api/trades", json=wrong_stop).status_code in (409, 422)
    stressed = c.post("/api/emotions", json={"confidence": 3, "stress": 5, "urgency": 1}).json()
    assert not stressed["allowed"] and stressed["blockers"] == ["Wait. Re-read the plan."]


def test_off_signal_trade_logged_as_breach(env: Env) -> None:
    c = env.client
    c.post("/api/emotions", json={"confidence": 3, "stress": 1, "urgency": 1})
    t = c.post("/api/trades", json={"symbol": "GBPUSD", "direction": -1, "entry": 1.30,
                                    "stop": 1.31, "checklist": {k: True for k in CHECKLIST}})
    assert t.status_code == 200 and t.json()["off_signal"]
    m = c.get("/api/mindset").json()
    assert any(b["kind"] == "DISCIPLINE BREACH" for b in m["breaches"])
    assert m["weekly_review"]["plan_adherence_pct"] < 100


def test_settings_risk_lock_and_validation(env: Env) -> None:
    c = env.client
    assert c.put("/api/settings", json={"risk_pct": 0.5}).status_code == 200
    assert c.put("/api/settings", json={"risk_pct": 0.25}).status_code == 200
    raised = c.put("/api/settings", json={"risk_pct": 0.75})
    assert raised.status_code == 409
    assert c.put("/api/settings", json={"risk_pct": 1.5}).status_code == 422  # hard cap
    assert c.put("/api/settings", json={"mode": "yolo"}).status_code == 422
    s = c.put("/api/settings", json={"currency": "kes", "theme": "light"}).json()
    assert s["currency"] == "KES" and s["theme"] == "light"


def test_drawings_persist_per_symbol(env: Env) -> None:
    c = env.client
    d = c.post("/api/drawings/EURUSD", json={"kind": "hline", "data": {"a": {"time": 1,
                                                                            "price": 1.1}}})
    assert d.status_code == 200
    assert len(c.get("/api/drawings/EURUSD").json()) == 1
    assert c.get("/api/drawings/GBPUSD").json() == []
    assert c.post("/api/drawings/EURUSD", json={"kind": "laser", "data": {}}).status_code == 422
    assert c.delete(f"/api/drawings/item/{d.json()['id']}").status_code == 200
    assert c.delete("/api/drawings/item/999").status_code == 404
    assert c.get("/api/candles/BTCUSD").status_code == 404


def test_other_tabs_load(env: Env) -> None:
    c = env.client
    for path in ("/api/validation", "/api/mindset", "/api/quality", "/api/paper",
                 "/api/execution/EURUSD", "/api/signals", "/api/missed", "/api/journal",
                 "/api/alerts", "/api/trades?status=open"):
        r = c.get(path)
        assert r.status_code == 200, (path, r.text)
    assert c.post("/api/journal", json={"kind": "note", "text": "hello"}).status_code == 200
    assert c.post("/api/journal", json={"kind": "reflection", "text": "too short"}
                  ).status_code == 422
    paper = c.get("/api/paper").json()
    assert paper["ready_for_small_live_use"] is False


def test_unvalidated_strategy_never_produces_buy(cfg: AppConfig,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    real = pipeline_mod.signals_for

    def fake(strategy: str, f: pd.DataFrame, p: Any) -> pd.DataFrame:
        out = real(strategy, f, p).copy()
        out.iloc[-1, out.columns.get_loc("direction")] = 1
        out.iloc[-1, out.columns.get_loc("stop")] = float(f["close"].iloc[-1] - 0.01)
        out.iloc[-1, out.columns.get_loc("grade")] = "A"
        return out

    monkeypatch.setattr(pipeline_mod, "signals_for", fake)
    e = Env(cfg, validated=False, fire=True)
    sig = e.client.get("/api/signal/EURUSD").json()
    assert sig["status"] == "WAIT" and sig["locked"]
    assert "not validated" in sig["card"]["headline"] or sig["card"]["blocks"]
    assert all(r["status"] not in ("BUY", "SELL") for r in e.client.get("/api/watchlist").json())


def test_daily_run_alerts_include_entry_stop_and_risk(env: Env) -> None:
    out = run_daily(env.pipe, env.notifier, refresh=False)
    assert out["evaluated"] == 2
    for text in env.sent:
        if "BUY" in text or "SELL" in text:
            assert "Stop" in text and "Amount at risk" in text and "Entry" in text
    again = run_daily(env.pipe, env.notifier, refresh=False)
    assert again["alerts"] == []  # no duplicate alerts for the same bar
    pre_session_reminder(env.pipe, env.notifier)


def test_signal_message_requires_risk_fields() -> None:
    ev = {"status": "BUY", "symbol": "EURUSD", "card": {"headline": "BUY EURUSD at next open",
                                                        "entry": 1.1, "stop": 1.09}}
    assert signal_message(ev) is None  # no size -> no alert
    ev["card"]["size"] = {"amount_at_risk": 5, "currency": "USD", "risk_pct": 0.5, "lots": 0.01}
    msg = signal_message(ev)
    assert msg is not None and "Amount at risk: 5 USD" in msg and "Stop 1.09" in msg
    assert signal_message({"status": "WAIT", "card": {}}) is None
    hold = {"status": "HOLD", "symbol": "EURUSD", "card": {"actions": [
        {"code": "TAKE_PARTIAL", "text": "+1R reached: take half"}]}}
    assert "take half" in (signal_message(hold) or "")


def test_telegram_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(":memory:")
    posted: list[dict[str, Any]] = []

    class R:
        status_code = 200

    def post(url: str, data: dict[str, Any], timeout: float) -> R:
        posted.append({"url": url, **data})
        return R()

    n = Notifier(store, http_post=post, desktop=lambda t, b: True)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert not n.telegram_configured()
    assert n.send("TEST", "hello") == {"desktop": True}
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert n.send("TEST", "hello")["telegram"] is True
    assert posted[0]["chat_id"] == "42" and "bottok" in posted[0]["url"]
    with pytest.raises(ValueError):
        n.send("TEST", "guaranteed profit")
    assert len(store.alerts()) == 3  # desktop; then telegram + desktop
    store.close()


def test_candle_quality_helper() -> None:
    from tfx.pipeline import candle_quality

    strong = pd.Series({"open": 1.0, "high": 1.1, "low": 0.99, "close": 1.095})
    weak = pd.Series({"open": 1.05, "high": 1.1, "low": 1.0, "close": 1.049})
    assert candle_quality(strong, 1) == 1.0 and candle_quality(weak, 1) == 0.0
    assert candle_quality(pd.Series({"open": 1, "high": 1, "low": 1, "close": 1}), 1) == 0.0
    assert np.isfinite(candle_quality(strong, -1))
