"""Deterministic validators: no AI in the loop, reproducibility, hashes, golden files,
traceability and setup rendering consistency."""

from __future__ import annotations

import ast
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import tfx.pipeline as pipeline_mod
from tests.conftest import FakeProvider, MutableClock, synthetic_candles
from tests.golden.cases import EXPECTED_DIR, cases, compute, header
from tfx.config import AppConfig
from tfx.consensus import Vote, consensus
from tfx.core_types import Timeframe
from tfx.data.cache import CandleCache
from tfx.data.service import MarketDataService
from tfx.pipeline import Pipeline
from tfx.store import Store
from tfx.validation.registry import ValidationRegistry
from tfx.validation.validators import Criteria, StrategyValidator, ValidatorStatus, validators
from tfx.versioning import RULE_VERSIONS, canonical_json, code_hash, rule_hash

PKG = Path(__file__).resolve().parents[1] / "tfx"
NOW = datetime(2026, 11, 21, 1, 0, tzinfo=UTC)
FORBIDDEN_MODULES = ("openai", "anthropic", "langchain", "transformers", "google.generativeai",
                     "google.genai", "cohere", "mistralai", "ollama", "llama_cpp",
                     "huggingface_hub", "vertexai", "litellm", "groq", "replicate")
FORBIDDEN_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
                   "api.cohere", "api.mistral.ai", "api-inference.huggingface.co")


# ------------------------------------------------------------- 1. no AI in the loop
def test_no_llm_imports_or_endpoints_anywhere_in_tfx() -> None:
    offenders = []
    for path in PKG.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if any(n == m or n.startswith(m + ".") for m in FORBIDDEN_MODULES):
                    offenders.append(f"{path.name}: import {n}")
        offenders += [f"{path.name}: {h}" for h in FORBIDDEN_HOSTS if h in src]
    assert offenders == []


def test_signal_path_loads_no_ai_module() -> None:
    import tfx.api.app
    import tfx.pipeline  # noqa: F401
    loaded = [m for m in sys.modules if any(m == f or m.startswith(f + ".")
                                            for f in FORBIDDEN_MODULES)]
    assert loaded == []


# ------------------------------------------------------------------ 4. golden files
@pytest.mark.parametrize("name", sorted(cases()))
def test_golden_examples_reproduce_exactly(name: str) -> None:
    path = EXPECTED_DIR / f"{name}.json"
    assert path.exists(), "golden file missing: run python -m tests.golden.regen"
    expected = json.loads(path.read_text(encoding="utf-8"))
    now = header()
    for strat, ver in now["rule_versions"].items():
        if expected["code_hashes"][strat] != now["code_hashes"][strat]:
            assert expected["rule_versions"][strat] != ver, (
                f"{strat}: rule code changed but RULE_VERSIONS was not bumped")
        assert expected["rule_versions"][strat] == ver, (
            f"{strat}: rule version {ver} but golden files are for "
            f"{expected['rule_versions'][strat]}: regenerate with python -m tests.golden.regen")
    got = json.loads(json.dumps(compute(cases()[name]), sort_keys=True))
    assert got == expected["outputs"], f"{name}: rule outputs changed - explicit update needed"


def test_golden_cover_every_strategy_setup() -> None:
    covered = set()
    for p in EXPECTED_DIR.glob("*.json"):
        for strat, out in json.loads(p.read_text(encoding="utf-8"))["outputs"].items():
            if out.get("setup"):
                covered.add(strat)
    assert covered == {"trend", "pullback", "mean_reversion"}


# --------------------------------------------------------------- 2. reproducibility
def test_rule_hash_stable_and_sensitive(cfg: AppConfig) -> None:
    h = rule_hash("trend", cfg)
    assert h == rule_hash("trend", cfg) and len(h) == 64
    cfg.section("strategies")["trend"]["adx_min"] = 21
    assert rule_hash("trend", cfg) != h           # a config change changes the hash
    assert code_hash("trend") != code_hash("mean_reversion")
    with pytest.raises(KeyError):
        rule_hash("smc", cfg)
    assert canonical_json({"b": 1, "a": [1.5]}) == '{"a":[1.5],"b":1}'


class World:
    def __init__(self, cfg: AppConfig, record: str | None) -> None:
        full = synthetic_candles("2003-01-01", "2026-11-20", seed=12)
        self.cache = CandleCache(":memory:")
        self.svc = MarketDataService(cfg, self.cache, {"yfinance": FakeProvider("yfinance", full),
                                                       "mt5": FakeProvider("mt5", full)},
                                     clock=MutableClock(NOW))
        self.svc.get("EURUSD", refresh=True)
        self.store = Store(":memory:")
        self.reg = ValidationRegistry(":memory:")
        if record is not None:
            self.reg.record_result("trend", Timeframe.D1, "PASS", "r", NOW,
                                   metrics={"rule_hash": record})
        self.pipe = Pipeline(self.svc, self.store, self.reg)


@pytest.fixture
def fire_trend(monkeypatch: pytest.MonkeyPatch) -> None:
    real = pipeline_mod.signals_for

    def fake(strategy: str, f: pd.DataFrame, p: Any) -> pd.DataFrame:
        out = real(strategy, f, p).copy()
        if strategy == "trend":
            out.iloc[-1, out.columns.get_loc("direction")] = 1
            out.iloc[-1, out.columns.get_loc("stop")] = float(f["close"].iloc[-1] - 0.01)
            out.iloc[-1, out.columns.get_loc("grade")] = "A"
            out.iloc[-1, out.columns.get_loc("confidence")] = 1.0
            out.iloc[-1, out.columns.get_loc("reason")] = "test breakout"
        else:
            out["direction"] = 0
            out["grade"] = ""
        return out

    monkeypatch.setattr(pipeline_mod, "signals_for", fake)


def test_same_input_byte_identical_output(cfg: AppConfig, fire_trend: None) -> None:
    outs = []
    for _ in range(2):
        w = World(cfg, rule_hash("trend", cfg))
        ev = w.pipe.evaluate("EURUSD", now=NOW, persist=False)
        outs.append(canonical_json(ev).encode())
    assert outs[0] == outs[1]


def test_hash_mismatch_blocks_signals(cfg: AppConfig, fire_trend: None) -> None:
    ok = World(cfg, rule_hash("trend", cfg)).pipe.evaluate("EURUSD", now=NOW)
    assert ok["status"] == "BUY" and not ok["locked"]
    bad = World(cfg, "0" * 64).pipe.evaluate("EURUSD", now=NOW)
    assert bad["status"] == "WAIT" and bad["locked"]
    assert any("changed since validation" in b for b in bad["card"]["blocks"])
    assert bad["card"]["trace"]["validator_state"] == "HASH MISMATCH"
    none = World(cfg, None).pipe.evaluate("EURUSD", now=NOW)
    assert none["status"] == "WAIT" and none["card"]["trace"]["validator_state"] == "NOT VALIDATED"


# ----------------------------------------------------- 3. traceability + 6. rendering
def test_trace_stored_and_setup_matches_card(cfg: AppConfig, fire_trend: None) -> None:
    w = World(cfg, rule_hash("trend", cfg))
    ev = w.pipe.evaluate("EURUSD", now=NOW)
    card = ev["card"]
    lv, setup = card["levels"], card["setup"]
    # chart objects == card numbers
    assert setup["box"]["entry"] == card["entry"] == lv["entry"]
    assert setup["box"]["stop"] == card["stop"] == lv["stop"]
    assert setup["box"]["target"] == card["targets"][0]["price"] == lv["target"]
    by_id = {line["id"]: line["price"] for line in setup["lines"]}
    assert by_id["entry"] == card["entry"] and by_id["stop"] == card["stop"]
    assert by_id["target"] == lv["target"]
    assert setup["box"]["risk_amount"] == card["size"]["amount_at_risk"]
    assert "R:R" in setup["summary"] and str(lv["stop"]) in setup["summary"]
    assert {"trail", "breakeven", "channel_hi", "ema_200"} <= set(by_id)
    # stored signal == card, with the trace and rule version/hash
    row = w.store.signal(ev["signal_id"])
    assert row is not None and row.entry_ref == card["entry"] and row.stop == card["stop"]
    trace = json.loads(row.trace)
    assert row.rule_hash == trace["rule_hash"] == rule_hash("trend", cfg)
    assert row.rule_version == RULE_VERSIONS["trend"]
    assert trace["levels"] == lv and len(trace["bars"]) == 5
    assert {"adx_14", "atr_20", "dch_20"} <= set(trace["indicators"])
    assert trace["thresholds"]["adx_min"] == 20.0


def test_locked_setup_labelled_view_only(cfg: AppConfig, fire_trend: None) -> None:
    ev = World(cfg, None).pipe.evaluate("EURUSD", now=NOW)
    assert ev["card"]["setup"]["summary"].startswith("NOT VALIDATED - view only")
    assert ev["card"]["setup"]["validated"] is False


def test_render_each_strategy_and_reject_unknown() -> None:
    from tfx.setup_render import levels, render
    from tfx.strategies.features import build_features
    from tfx.strategies.rules import StrategyParams

    f = build_features(synthetic_candles("2018-01-01", "2020-12-31"))
    p = StrategyParams()
    row = f.iloc[-1]
    for strat in ("trend", "pullback", "mean_reversion"):
        lv = levels(strat, 1, row, float(row["close"]), float(row["close"] - 0.01), 1.0, 1e-4, 5)
        s = render(strat, 1, f, lv, p, 1.0, 5, True, 5.0, "USD")
        assert s["box"]["entry"] == lv["entry"] and s["summary"]
        assert all(o.get("label") and o.get("icon") for o in s["lines"] + s["zones"])
    mr = render("mean_reversion", 1, f, lv, p, 1.0, 5, True, None, "USD", bars_held=3)
    assert any("Time stop in 2 bars" in m["label"] for m in mr["markers"])
    with pytest.raises(ValueError):
        render("smc", 1, f, lv, p, 1.0, 5, True)


# ------------------------------------------------------- 5. validators + consensus
def _status(name: str, ok: bool) -> ValidatorStatus:
    return ValidatorStatus(name, "D1", "PASS" if ok else "FAIL", ok, "", "1.0.0", "h", "h")


def test_validator_fixed_criteria(cfg: AppConfig) -> None:
    v = validators(cfg)["trend"]
    good = {"pre_holdout": {"trades": 300}, "holdout": {"sharpe": 0.6, "expectancy_r": 0.2},
            "pbo": 0.02, "dsr": 0.95, "neighbours_profitable": 0.8,
            "cost_stress": {"1.5x": 0.3}, "total_trials_counted": 54}
    assert v.evaluate(good) == ("PASS", [])
    verdict, reasons = v.evaluate({**good, "pbo": 0.3, "cost_stress": {"1.5x": -0.1}})
    assert verdict == "FAIL" and len(reasons) == 2
    assert StrategyValidator("trend", Criteria()).criteria.pbo_max == 0.05
    with pytest.raises(KeyError):
        StrategyValidator("smc", Criteria())


def test_consensus_only_validated_strategies_vote() -> None:
    st = {"trend": _status("trend", True), "pullback": _status("pullback", False),
          "mean_reversion": _status("mean_reversion", True)}
    c = consensus([Vote("trend", 1, 80), Vote("pullback", -1, 90)], st)
    assert c.direction == 1 and c.voters == ["trend"] and c.excluded == {"pullback": "FAIL"}
    clash = consensus([Vote("trend", 1, 80), Vote("mean_reversion", -1, 70)], st)
    assert clash.conflict and clash.direction == 0 and "disagree" in clash.text
    nobody = consensus([Vote("pullback", 1, 99)], st)
    assert nobody.voters == [] and "No validated" in nobody.text
    both = consensus([Vote("trend", 1, 60), Vote("mean_reversion", 1, 75)], st)
    assert both.leader == "mean_reversion" and "agree" in both.text


def test_store_migration_adds_new_columns(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, created_at DATETIME, "
                "session_date VARCHAR(10), symbol VARCHAR(12), strategy VARCHAR(32), "
                "timeframe VARCHAR(4), status VARCHAR(12), direction INTEGER, grade VARCHAR(8),"
                " score FLOAT, size_mult FLOAT, entry_ref FLOAT, stop FLOAT, order_type "
                "VARCHAR(16), order_price FLOAT, expiry VARCHAR(32), reasons TEXT, factors TEXT,"
                " outcome VARCHAR(16))")
    con.commit()
    con.close()
    s = Store(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(signals)")}
    assert {"rule_hash", "rule_version", "trace"} <= cols
    s.close()
