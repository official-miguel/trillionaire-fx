"""Hypothetical replay of a past window (e.g. last week from $100)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import tfx.validation.proof as proof
from tests.conftest import FakeProvider, MutableClock, synthetic_candles
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.service import MarketDataService
from tfx.replay import DISCLAIMER, MIN_LOT, replay


@pytest.fixture
def world(cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CandleCache:
    monkeypatch.setattr(proof, "REPORT_PATH", tmp_path / "none.json")
    full = synthetic_candles("2003-01-01", "2026-11-20", seed=21)
    cache = CandleCache(":memory:")
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", full),
                                         "mt5": FakeProvider("mt5", full)},
                            clock=MutableClock(datetime(2026, 11, 21, 1, tzinfo=UTC)))
    cfg.symbols = {k: v for k, v in cfg.symbols.items() if k in ("EURUSD", "GBPUSD")}
    for s in cfg.symbols:
        svc.get(s, refresh=True)
    return cache


def test_replay_window_and_accounting(cfg: AppConfig, world: CandleCache) -> None:
    r = replay(cfg, world, date(2026, 1, 5), date(2026, 3, 27), 100.0, 0.5)
    assert r.disclaimer == DISCLAIMER and "HYPOTHETICAL" in r.disclaimer
    bal = 100.0
    for t in r.trades:
        assert "2026-01-05" <= t.signal_date <= "2026-03-27"   # only signals in the window
        assert t.risk_amount <= 0.005 * bal + 0.01             # 0.5% of the running balance
        bal += t.pnl
    assert r.ending_balance == pytest.approx(100 + sum(t.pnl for t in r.trades), abs=0.05)
    assert r.return_pct == pytest.approx((r.ending_balance / 100 - 1) * 100, abs=0.01)


def test_small_account_min_lot_flagged(cfg: AppConfig, world: CandleCache) -> None:
    r = replay(cfg, world, date(2025, 1, 6), date(2026, 3, 27), 100.0, 0.5)
    assert r.trades, "expected some trades over a year of synthetic data"
    small = [t for t in r.trades if t.lots < MIN_LOT]
    assert small and all(not t.tradable for t in small)
    assert r.untradable == len(small) and "minimum" in r.notes[0]
    big = replay(cfg, world, date(2025, 1, 6), date(2026, 3, 27), 1_000_000.0, 0.5)
    assert big.untradable == 0


def test_replay_rejects_risk_above_cap(cfg: AppConfig, world: CandleCache) -> None:
    with pytest.raises(ValueError):
        replay(cfg, world, date(2026, 1, 5), date(2026, 1, 9), 100.0, 2.0)
    import json

    from tfx.replay import as_dict

    full = replay(cfg, world, date(2025, 1, 6), date(2026, 3, 27), 100.0, 0.5)
    json.dumps(as_dict(full))  # must be plain-JSON serialisable for the API
    empty = replay(cfg, world, date(2026, 11, 21), date(2026, 11, 22), 100.0, 0.5)
    assert empty.trades == [] and "no trades" in empty.notes[0]


def test_compare_each_strategy_separately(cfg: AppConfig, world: CandleCache) -> None:
    from tfx.replay import compare

    c = compare(cfg, world, date(2025, 6, 2), date(2026, 3, 27), 100.0, 0.5,
                include_momentum=False)
    rows = {r["strategy"]: r for r in c["rows"]}
    assert set(rows) == {"trend", "pullback", "mean_reversion", "ict", "ict_flex",
                         "supply_demand", "ict_then_sd", "all"}
    for r in rows.values():
        assert r["starting_balance"] == 100.0 and r["kind"] == "per-trade"
        assert r["trades"] == r["closed"] + r["open"]
    single = {s: rows[s]["trades"] for s in ("trend", "pullback", "mean_reversion")}
    assert rows["all"]["trades"] <= sum(single.values())  # caps can only remove trades
    only_trend = replay(cfg, world, date(2025, 6, 2), date(2026, 3, 27), 100.0, 0.5,
                        strategies=("trend",))
    assert all(t.strategy == "trend" for t in only_trend.trades)
    assert len(only_trend.trades) == rows["trend"]["trades"]


def test_zone_strategies_and_ict_then_sd_stack(cfg: AppConfig, world: CandleCache) -> None:
    from tfx.replay import max_drawdown_pct

    s, e = date(2010, 1, 1), date(2026, 11, 20)
    ict = replay(cfg, world, s, e, 100.0, 0.5, strategies=("ict",))
    sd = replay(cfg, world, s, e, 100.0, 0.5, strategies=("supply_demand",))
    stack = replay(cfg, world, s, e, 100.0, 0.5, strategies=("ict_then_sd",))
    assert sd.trades and {t.strategy for t in sd.trades} == {"supply_demand"}
    assert {t.strategy for t in stack.trades} <= {"ict", "supply_demand"}
    # every ICT trade is kept; S&D only fills the gaps
    assert sum(t.strategy == "ict" for t in stack.trades) >= sum(
        1 for t in ict.trades if t.status == "closed") - 2
    assert len(stack.trades) <= len(ict.trades) + len(sd.trades)
    for t in stack.trades:
        assert "2010-01-01" <= t.signal_date <= "2026-11-20"
    dd = max_drawdown_pct(stack)
    assert -100 <= dd <= 0
