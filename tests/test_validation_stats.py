"""Validation statistics and the proof-phase runner."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tfx.validation.proof as proof
from tests.conftest import FakeProvider, MutableClock, synthetic_candles
from tfx.config import AppConfig
from tfx.core_types import Timeframe
from tfx.data.cache import CandleCache
from tfx.data.service import MarketDataService
from tfx.validation import stats
from tfx.validation.registry import ValidationRegistry


def test_basic_metrics() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=pd.bdate_range("2020-01-01", periods=5))
    assert stats.max_drawdown(r) == pytest.approx(-0.02)
    assert stats.sharpe(np.zeros(10)) == 0.0
    assert stats.sharpe([0.01, 0.02, 0.0]) > 0
    assert stats.cagr(pd.Series(dtype=float)) == 0.0
    assert stats.per_year(r) == {2020: pytest.approx(float(np.prod(1 + r) - 1))}
    m = stats.trade_metrics([2.0, -1.0, -1.0, 1.5], [0.05] * 4, months=2)
    assert m["win_rate"] == 0.5 and m["payoff"] == pytest.approx(1.75)
    assert m["expectancy_r"] == pytest.approx(0.375) and m["profit_factor"] == pytest.approx(1.75)
    assert m["longest_losing_streak"] == 2 and m["trades_per_month"] == 2.0
    assert stats.trade_metrics([])["trades"] == 0


def test_psr_and_dsr_behave() -> None:
    assert stats.probabilistic_sharpe(0.1, 1000, 0, 3) > 0.99
    assert stats.probabilistic_sharpe(0.0, 1000, 0, 3) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    trials = [float(x) for x in rng.normal(0, 0.03, 50)]
    noise = rng.normal(0, 0.01, 2000)
    strong = rng.normal(0.002, 0.01, 2000)
    assert stats.deflated_sharpe(noise, trials, 50) < 0.5
    assert stats.deflated_sharpe(strong, trials, 50) > 0.9
    # more trials -> harder to pass (deflation)
    assert stats.deflated_sharpe(strong, trials, 5000) <= stats.deflated_sharpe(strong, trials, 5)
    assert stats.deflated_sharpe(np.zeros(5), trials, 5) == 0.0


def test_pbo_noise_vs_genuine_edge() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (1500, 12))
    assert 0.2 < stats.pbo_cscv(noise, 10, 5).pbo < 0.8
    edge = noise.copy()
    edge[:, 3] += 0.003  # one configuration genuinely better everywhere
    res = stats.pbo_cscv(edge, 10, 5)
    assert res.pbo < 0.05 and res.n_splits == 252
    assert stats.pbo_cscv(noise[:20], 10).pbo == 1.0  # too little data: treated as overfit


def test_bootstrap_and_shuffles_are_deterministic() -> None:
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 800))
    a = stats.block_bootstrap(r, 200, 20, seed=5)
    b = stats.block_bootstrap(r, 200, 20, seed=5)
    assert a == b and a["sharpe"][0] < a["sharpe"][1] and a["max_drawdown"][1] <= 0
    assert stats.block_bootstrap(r.iloc[:10]) == {}
    mc = stats.shuffle_drawdowns([1, -1, -1, 2, -1, 0.5] * 10, 0.005, 300, seed=1)
    assert mc["max_drawdown_p95"] <= mc["max_drawdown_p50"] <= 0
    assert mc["losing_streak_p95"] >= mc["losing_streak_p50"]
    assert stats.shuffle_drawdowns([], 0.005) == {}


def test_proof_runner_on_synthetic_data(cfg: AppConfig, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proof, "REPORT_PATH", tmp_path / "report.json")
    full = synthetic_candles("2003-01-01", "2026-11-20", seed=31)
    cache = CandleCache(":memory:")
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", full),
                                         "mt5": FakeProvider("mt5", full)},
                            clock=MutableClock(datetime(2026, 11, 21, 1, tzinfo=UTC)))
    svc.get("EURUSD", refresh=True)
    cfg.symbols = {"EURUSD": cfg.symbols["EURUSD"]}
    raw = cfg.model_extra or {}
    raw["parameter_grid"] = {"trend": {"initial_stop_atr": [1.5, 2.5]},
                             "pullback": {"exit_channel": [10, 20]},
                             "mean_reversion": {"mr_time_stop": [3, 5]},
                             "xs_momentum": {"formation_months": [3], "n_each_side": [1]},
                             "ts_momentum": {"lookback_weeks": [12]},
                             "ict": {"swing_n": [3]}, "ict_flex": {"swing_n": [3]},
                             "supply_demand": {"departure_atr": [1.0]}}
    raw["validation"]["bootstrap_samples"] = 40
    reg = ValidationRegistry(":memory:")
    report = proof.ProofRunner(cfg, cache, reg).run_all()
    assert (tmp_path / "report.json").exists()
    d1 = [p for p in report["pairs"] if p["full"]]
    assert {p["strategy"] for p in d1} == {"trend", "pullback", "mean_reversion",
                                          "xs_momentum", "ts_momentum", "ict", "ict_flex",
                                          "supply_demand"}
    assert reg.trial_count() == 11  # every grid combination counted
    assert {p["timeframe"] for p in d1 if "momentum" in p["strategy"]} == {"W1"}
    for p in d1:
        # random-walk data has no edge: the gate must not pass it
        assert p["verdict"] == "FAIL" and p["reasons"]
        assert p["holdout"]["start"] > p["pre_holdout"]["start"]
        assert set(p["cost_stress"]) == {"1.0x", "1.5x", "2.0x"}
        assert "DSR" in p["edge_derivation"]
    assert not reg.gate("trend", Timeframe.D1, datetime.now(UTC)).allowed
    assert not reg.gate("trend", Timeframe.H4, datetime.now(UTC)).allowed
    assert "Nothing passed" in report["summary"]
    assert proof.load_report(tmp_path / "report.json")["run_id"] == report["run_id"]
    assert proof.load_report(tmp_path / "missing.json") is None
    # --only: re-prove one strategy, keep the others from the report, count only new trials
    later = datetime(2030, 1, 1, tzinfo=UTC)
    again = proof.ProofRunner(cfg, cache, reg, clock=lambda: later).run_all(
        only=["supply_demand"])
    assert reg.trial_count() == 11   # same settings again: not a new trial
    assert {p["strategy"] for p in again["pairs"] if p["full"]} == {p["strategy"] for p in d1}
    assert proof.costs_for(cfg, "EURUSD").spread == pytest.approx(0.8 * 0.0001)
