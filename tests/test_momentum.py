"""Momentum sleeves: next-open execution, costs, ranking, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.strategies.momentum import USD_LEGS, PairCosts, _simulate, ts_momentum, xs_momentum

START = pd.Timestamp("2006-01-01")


def universe(seed0: int = 1) -> dict[str, pd.DataFrame]:
    out = {}
    for i, pair in enumerate(p for p, _ in USD_LEGS.values()):
        base = 100.0 if pair.endswith("JPY") else 1.2
        out[pair] = synthetic_candles("2003-01-01", "2012-12-31", seed=seed0 + i, base=base)
    return out


def zero_costs(data: dict[str, pd.DataFrame]) -> dict[str, PairCosts]:
    return {k: PairCosts(0.0, 0.0) for k in data}


def test_weights_take_effect_at_next_open() -> None:
    idx = pd.bdate_range("2020-01-01", periods=4)
    w = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0]}, index=idx)
    gaps = pd.DataFrame({"X": [0.0, 0.0, 0.10, 0.0]}, index=idx)
    sess = pd.DataFrame({"X": [0.0, 0.50, 0.02, 0.0]}, index=idx)
    r = _simulate(w, gaps, sess, {"X": PairCosts(0.0, 0.0)}, 1.0)
    # decided at close of day 1 -> earns nothing on day 1 (not even the 50% session),
    # earns the session of day 2 but NOT day 2's overnight gap, then everything after
    assert list(r.round(6)) == [0.0, 0.0, 0.02, 0.0]


def test_costs_only_reduce_returns() -> None:
    data = universe()
    free, _ = ts_momentum(data, zero_costs(data), 26, START)
    costly_c = {k: PairCosts(0.0002, 0.00001) for k in data}
    paid, _ = ts_momentum(data, costly_c, 26, START)
    stressed, _ = ts_momentum(data, costly_c, 26, START, cost_mult=2.0)
    assert paid.sum() < free.sum() and stressed.sum() < paid.sum()


def test_xs_ranking_longs_strongest_shorts_weakest() -> None:
    data = universe()
    eur = data["EURUSD"].copy()
    drift = np.exp(np.linspace(0, 0.8, len(eur)))     # EUR strongly up vs everything
    for c in ("open", "high", "low", "close"):
        eur[c] = eur[c] * drift
    data["EURUSD"] = eur
    r, trades = xs_momentum(data, zero_costs(data), 3, 1, START)
    eur_trades = [t for t in trades if t.symbol == "EURUSD"]
    assert eur_trades and sum(t.direction for t in eur_trades) > 0  # mostly long EUR
    assert r.index[0] >= START and np.isfinite(r).all()


def test_no_lookahead_future_shock_does_not_change_past() -> None:
    data = universe()
    base, _ = ts_momentum(data, zero_costs(data), 12, START)
    shocked = {k: v.copy() for k, v in data.items()}
    cut = pd.Timestamp("2010-06-01")
    for df in shocked.values():
        later = df.index >= cut
        for c in ("open", "high", "low", "close"):
            df.loc[later, c] = df.loc[later, c] * 1.5   # change only the future
    after, _ = ts_momentum(shocked, zero_costs(shocked), 12, START)
    before = base.index < cut - pd.Timedelta(days=3)
    pd.testing.assert_series_equal(base[before], after[before])  # past results unchanged


def test_trade_r_definition() -> None:
    data = universe()
    _, trades = ts_momentum(data, zero_costs(data), 26, START)
    assert trades and all(np.isfinite(t.r_net) for t in trades)
    assert all(t.r_cost == pytest.approx(0.0) for t in trades)
    assert all(t.exit_reason == "rebalance" and t.bars >= 1 for t in trades)
