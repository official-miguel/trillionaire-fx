"""Supply & demand zones, the shared limit simulator, the missed-trade study, the
ICT -> supply & demand fallback and the top-down market read."""

from __future__ import annotations

import itertools
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.analysis import DISCLAIMER, market_read
from tfx.backtest.engine import Costs
from tfx.config import AppConfig
from tfx.setup_render import render_sd
from tfx.strategies.features import build_features
from tfx.strategies.limit_sim import LimitOrder, fill_and_manage
from tfx.strategies.supply_demand import (
    SDParams,
    active_order,
    chart_zones,
    find_zones,
    orders,
    simulate,
)
from tfx.study_missed import CONDITIONS, big_moves, run_study

ZERO = Costs(0.0, 0.0, 0.0, 0.0)


def demand_chart() -> pd.DataFrame:
    """Drift, a 2-candle base at 1.1000, a strong rally (leg-out), then a first return."""
    rows = []
    for k in range(30):
        cl = 1.0900 + 0.0003 * k
        rows.append((cl - 0.0002, cl + 0.0006, cl - 0.0006, cl))
    rows.append((1.0990, 1.1004, 1.0985, 1.0995))     # base 1
    rows.append((1.0995, 1.1006, 1.0988, 1.1000))     # base 2
    rows.append((1.1000, 1.1105, 1.0998, 1.1100))     # leg-out (big up body)
    for k in range(6):                                 # drift away
        cl = 1.1110 + 0.0004 * k
        rows.append((cl - 0.0002, cl + 0.0006, cl - 0.0006, cl))
    for cl in (1.1080, 1.1040, 1.1010):                # return into the zone
        rows.append((cl + 0.0010, cl + 0.0012, cl - 0.0012, cl))
    for k in range(10):
        cl = 1.1030 + 0.0010 * k
        rows.append((cl - 0.0005, cl + 0.0008, cl - 0.0006, cl))
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_demand_zone_found_with_correct_edges_and_first_touch() -> None:
    df = demand_chart()
    p = SDParams(departure_atr=1.5, trend_filter=False)
    zones = [z for z in find_zones(df, p) if z.direction == 1 and z.created_bar == 32]
    assert zones, "expected a demand zone at the leg-out bar"
    z = zones[0]
    assert z.proximal == pytest.approx(1.1000)          # highest body top of the base
    assert z.distal == pytest.approx(df["low"].iloc[z.base_start:33].min())  # base + leg-out
    assert 30 <= z.base_start <= 31 or z.base_start == 29   # 1-3 small-bodied candles
    assert z.touched_bar is not None and z.touched_bar > z.created_bar
    od = next(o for o in orders(df, p) if o.known_bar == 32)
    assert od.entry == pytest.approx(z.proximal) and od.stop < z.distal
    assert od.target == pytest.approx(od.entry + 2 * (od.entry - od.stop))


def test_zones_have_no_lookahead() -> None:
    df = synthetic_candles("2012-01-01", "2016-12-31", seed=3)
    p = SDParams(departure_atr=1.0)
    full = find_zones(df, p)
    assert len(full) > 10
    for z in full[:20]:
        cut = df.iloc[: z.created_bar + 1]
        again = [y for y in find_zones(cut, p) if y.created_bar == z.created_bar]
        assert again
        a, b = asdict(again[0]), asdict(z)
        for k in ("touched_bar", "broken_bar"):       # only known later
            a.pop(k), b.pop(k)
        assert a == b


def test_trend_filter_blocks_counter_trend_zones() -> None:
    df = synthetic_candles("2005-01-01", "2016-12-31", seed=8)
    from tfx.strategies import indicators as ind
    e50, e200 = ind.ema(df["close"], 50).to_numpy(), ind.ema(df["close"], 200).to_numpy()
    for od in orders(df, SDParams(departure_atr=1.0)):
        t = od.known_bar
        assert (e50[t] > e200[t]) if od.direction == 1 else (e50[t] < e200[t])


def test_simulation_fixed_stop_costs_and_one_at_a_time() -> None:
    df = synthetic_candles("2005-01-01", "2020-12-31", seed=11)
    p = SDParams(departure_atr=1.0)
    trades = simulate("EURUSD", df, p, ZERO)
    assert trades
    for a, b in itertools.pairwise(trades):
        assert b.signal_date > a.exit_date or b.entry_date > a.exit_date
    for t in trades:
        assert t.stop_path == [t.stop0] and t.strategy == "supply_demand"
        assert t.entry_date > t.signal_date
        assert t.exit_reason in ("stop", "time stop", "end of test", "target (2R)")
    costly = simulate("EURUSD", df, p, Costs(0.0002, 0.00007, 0.00002, 0.00001))
    assert sum(t.r_net for t in costly) < sum(t.r_net for t in trades)
    opp = simulate("EURUSD", df, SDParams(departure_atr=1.0, target="opposing"), ZERO)
    assert all(t.exit_reason != "target (2R)" for t in opp)


def test_limit_sim_cancels_when_stop_comes_first_and_no_target_on_fill_bar() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"open": [1.10, 1.10, 1.10, 1.10, 1.10],
                       "high": [1.101, 1.101, 1.130, 1.101, 1.101],
                       "low": [1.099, 1.099, 1.080, 1.099, 1.099],
                       "close": [1.10, 1.10, 1.10, 1.10, 1.10]}, index=idx)
    # bar 2 touches entry, stop AND target: opened above the entry -> filled, stop first
    tr = fill_and_manage("EURUSD", "x", df, LimitOrder(1, 1, 1.095, 1.090, 1.105, 4), ZERO,
                         10, "2R")
    assert tr is not None and tr.exit_reason == "stop"
    never = fill_and_manage("EURUSD", "x", df, LimitOrder(1, 1, 1.05, 1.04, 1.07, 4), ZERO,
                            10, "2R")
    assert never is None


def test_active_order_and_chart_zones_and_render() -> None:
    df = demand_chart().iloc[:36]                       # before the return
    p = SDParams(departure_atr=1.5, trend_filter=False)
    act = active_order(df, p)
    assert act is not None
    z, od = act
    assert z.direction == 1 and z.touched_bar is None
    zs = chart_zones(df, p)
    assert zs and all(x["label"] and x["icon"] for x in zs)
    f = build_features(df)
    r = render_sd(z, od, f, 5, False, 2.0, "USD")
    assert r["summary"].startswith("NOT VALIDATED") and r["box"]["entry"] == round(od.entry, 5)
    assert {line["id"] for line in r["lines"]} == {"entry", "stop", "target"}
    assert active_order(demand_chart(), p) is None or active_order(demand_chart(), p)[0] != z


def test_params_from_config(cfg: AppConfig) -> None:
    p = SDParams.from_config(cfg.section("strategies")["supply_demand"],
                             {"departure_atr": 1.0, "nonsense": 5})
    assert p.departure_atr == 1.0 and p.trend_filter and p.time_stop == 15


# --------------------------------------------------------------------- study
def test_study_uses_training_data_only_and_counts_losers(cfg: AppConfig) -> None:
    data = {"EURUSD": synthetic_candles("2003-01-01", "2026-11-20", seed=21),
            "GBPUSD": synthetic_candles("2003-01-01", "2026-11-20", seed=22, base=1.3)}
    res = run_study(cfg, data, {s: ZERO for s in data})
    hs = pd.Timestamp(res.holdout_start)
    assert hs == pd.Timestamp("2026-11-20") - pd.DateOffset(years=3)
    assert all(pd.Timestamp(e["mss_date"]) < hs for e in res.examples)
    missed = res.groups["missed (all)"]
    assert missed.wins + missed.losses == missed.filled
    assert res.profitable_missed + res.losing_missed == missed.filled
    for cond in CONDITIONS:
        g = res.groups[f"missed: only {cond}"]
        if g.filled < 30:
            assert not g.adopted                      # pre-declared minimum sample
    assert set(res.adopted) <= set(CONDITIONS)
    assert "holdout" not in res.summary or "validation" in res.summary
    assert res.big_moves["moves"] >= 0 and 0 <= res.big_moves["base_rate"] <= 1


def test_big_moves_counts_hindsight_runs() -> None:
    idx = pd.bdate_range("2024-01-01", periods=60)
    close = [1.0] * 30 + [1.0 + 0.01 * k for k in range(1, 31)]
    df = pd.DataFrame({"open": close, "high": [c + 0.001 for c in close],
                       "low": [c - 0.001 for c in close], "close": close}, index=idx)
    out = big_moves(df, idx[-1], {"none": set(), "all": {(1, b) for b in range(60)}})
    assert out["moves"] >= 1
    assert out["had_a_setup"]["none"] == 0 and out["had_a_setup"]["all"] == out["moves"]


# ------------------------------------------------------------ market read
def test_market_read_is_deterministic_and_labelled() -> None:
    df = synthetic_candles("2018-01-01", "2026-11-20", seed=4)
    a, b = market_read(df), market_read(df)
    assert a == b and a["available"]
    assert [s["step"][:2] for s in a["steps"]] == ["1.", "2.", "3.", "4.", "5.", "6.", "7.",
                                                "8."]
    assert a["disclaimer"] == DISCLAIMER and "not a signal" in DISCLAIMER
    assert 0 <= a["range_position"] <= 1
    assert a["bias"] in ("bullish", "bearish", "neutral / mixed")
    assert not market_read(df.iloc[:100])["available"]


def test_analysis_endpoint_and_fallback_card(cfg: AppConfig,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import tfx.pipeline as pm
    from tests.test_api_flows import NOW, Env

    e = Env(cfg, validated=False, fire=False)
    r = e.client.get("/api/analysis/EURUSD").json()
    assert r["symbol"] == "EURUSD" and r["available"] and len(r["steps"]) == 8
    # ICT has nothing -> supply & demand is shown (locked: not validated)
    df = e.svc.get("EURUSD").candles
    z_od: Any = (pm.Zone(1, len(df) - 3, len(df) - 5, float(df["close"].iloc[-1]) - 0.002,
                         float(df["close"].iloc[-1]) - 0.004, float(df["close"].iloc[-1]) - 0.004,
                         float(df["close"].iloc[-1]) - 0.002, 1.6),
                 LimitOrder(1, len(df) - 3, float(df["close"].iloc[-1]) - 0.002,
                            float(df["close"].iloc[-1]) - 0.0045,
                            float(df["close"].iloc[-1]) + 0.003, len(df) + 30))
    monkeypatch.setattr(pm.Pipeline, "ict_active", lambda self, df, variant="ict": None)
    monkeypatch.setattr(pm, "active_order", lambda df, p: z_od)
    monkeypatch.setattr(pm, "signals_for", lambda s, f, p: pd.DataFrame(
        {"direction": 0, "stop": float("nan"), "grade": "", "confidence": 0.0, "reason": "",
         "watch_level": float("nan")}, index=f.index))
    e.pipe.clear_cache()
    ev = e.pipe.evaluate("EURUSD", now=datetime(2026, 11, 21, 1, 0, tzinfo=UTC))
    assert ev["strategy"] == "supply_demand" and ev["locked"] and ev["status"] == "WAIT"
    assert ev["card"]["setup"]["strategy"] == "supply_demand"
    assert ev["card"]["trace"]["supply_demand"]["strength_atr"] == 1.6
    assert NOW
    e.store.close()
