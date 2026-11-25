"""Move precursors and the move study (start from the moves, count the false alarms)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.analysis import move_warnings
from tfx.backtest.engine import Costs
from tfx.config import AppConfig
from tfx.strategies.limit_sim import fill_and_manage
from tfx.strategies.precursors import (
    ALL_RULES,
    BASE_RULES,
    BracketParams,
    bracket_orders,
    signals,
    simulate,
)
from tfx.study_moves import move_labels, run_move_study

ZERO = Costs(0.0, 0.0, 0.0, 0.0)


def test_signals_have_no_lookahead() -> None:
    df = synthetic_candles("2010-01-01", "2016-12-31", seed=5)
    full = signals(df)
    assert set(full) == set(ALL_RULES) and len(ALL_RULES) == 14
    for cut in (400, 900, 1500):
        part = signals(df.iloc[: cut + 1])
        for r in ALL_RULES:
            assert part[r][-1] == full[r][cut], (r, cut)
    for r in BASE_RULES:   # weekly versions only keep signals of the base rule
        w = full[f"{r}+weekly"]
        assert np.all((w == 0) | (w == full[r]))


def test_market_entry_on_a_stop_bar_is_a_loss_not_cancelled() -> None:
    idx = pd.bdate_range("2024-01-01", periods=30)
    c = np.full(30, 1.10)
    df = pd.DataFrame({"open": c, "high": c + 0.001, "low": c - 0.001, "close": c}, index=idx)
    df.iloc[25, df.columns.get_loc("low")] = 1.05          # crash through the stop on entry bar
    d = np.zeros(30, dtype=int)
    d[24] = 1
    od = bracket_orders(df, d, BracketParams())[0]
    assert od.known_bar == 24 and od.entry == pytest.approx(1.10)
    tr = fill_and_manage("EURUSD", "x", df, od, ZERO, 10, "3R")
    assert tr is not None and tr.exit_reason == "stop" and tr.r_net == pytest.approx(-1, abs=0.05)


def test_bracket_simulation() -> None:
    df = synthetic_candles("2005-01-01", "2020-12-31", seed=9)
    p = BracketParams(stop_atr=1.0, target_r=3.0, time_stop=10)
    trades = simulate("EURUSD", df, ("donchian20",), p, ZERO)
    assert trades
    for t in trades:
        assert t.exit_reason in ("stop", "time stop", "target (3R)", "end of test")
        assert t.r_net >= -1.6 and t.entry_date > t.signal_date
    wins = [t for t in trades if t.exit_reason == "target (3R)"]
    assert all(t.r_net == pytest.approx(3.0, abs=0.05) for t in wins)


def test_move_labels() -> None:
    idx = pd.bdate_range("2024-01-01", periods=60)
    close = [1.0] * 30 + [1.0 + 0.01 * k for k in range(1, 31)]
    df = pd.DataFrame({"open": close, "high": [x + 0.001 for x in close],
                       "low": [x - 0.001 for x in close], "close": close}, index=idx)
    lab = move_labels(df)
    assert (lab == 1).any() and not (lab == -1).any()


def test_move_study_training_only_and_strict_adoption(cfg: AppConfig) -> None:
    data = {"EURUSD": synthetic_candles("2003-01-01", "2026-11-20", seed=31),
            "GBPUSD": synthetic_candles("2003-01-01", "2026-11-20", seed=32, base=1.3)}
    res = run_move_study(cfg, data, {s: ZERO for s in data})
    assert res.period[1] == str((pd.Timestamp("2026-11-20") - pd.DateOffset(years=3)).date())
    assert res.tests == 14 and res.boot_percentile == pytest.approx(100 * 0.05 / 14, abs=1e-3)
    assert res.moves == res.up_moves + res.down_moves
    for r in res.rules.values():
        assert r.moves_captured <= res.moves and 0 <= r.hit_rate <= 1
        if r.adopted:
            assert r.trades >= 100 and (r.mean_r or 0) > 0 and (r.boot_low or 0) > 0
            assert (r.mean_r_first_half or 0) > 0 and (r.mean_r_second_half or 0) > 0
    # random walks have no edge: nothing should survive the adjusted bootstrap
    assert not res.adopted and "No precursor" in res.summary


def test_move_warnings_are_labelled_unproven(tmp_path: object,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import tfx.study_moves as sm

    monkeypatch.setattr(sm, "STUDY_PATH", __import__("pathlib").Path(str(tmp_path)) / "x.json")
    df = synthetic_candles("2015-01-01", "2026-11-20", seed=2)
    for cut in range(len(df) - 40, len(df)):
        w = move_warnings(df.iloc[:cut])
        for x in w:
            assert x["direction"] in ("up", "down") and x["text"]
            assert "NOT proven" not in x["text"]          # no study file -> no history claim
