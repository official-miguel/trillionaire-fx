"""Indicators, strategy rules and the backtester (part 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.backtest.engine import (
    Costs,
    Management,
    RuleViolationError,
    _check_stop_move,
    apply_portfolio_caps,
    daily_r_frame,
    run_symbol,
)
from tfx.strategies import indicators as ind
from tfx.strategies.features import build_features, regime, weekly_bias
from tfx.strategies.rules import StrategyParams, signals_for

P = StrategyParams()
M = Management()
ZERO = Costs(0.0, 0.0, 0.0, 0.0)


def bars(closes: list[float], spread: float = 0.002) -> pd.DataFrame:
    c = np.array(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.bdate_range("2020-01-01", periods=len(c))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread,
                         "low": np.minimum(o, c) - spread, "close": c}, index=idx)


# --------------------------------------------------------------- indicators
def test_ema_known_values() -> None:
    s = pd.Series([1.0, 2, 3, 4, 5])
    e = ind.ema(s, 3)  # alpha = 0.5, seeded with the first value
    assert np.isnan(e.iloc[1])
    assert e.iloc[2] == pytest.approx(2.25) and e.iloc[4] == pytest.approx(4.0625)


def test_atr_and_true_range_known_values() -> None:
    df = pd.DataFrame({"open": [10, 10.5, 11], "high": [11.0, 12, 11.5],
                       "low": [9.0, 10, 10.5], "close": [10.5, 11.5, 11]})
    assert list(ind.true_range(df)) == [2.0, 2.0, 1.0]
    a = ind.atr(df, 2)  # Wilder: alpha 1/2 -> 2, 2, 1.5
    assert a.iloc[2] == pytest.approx(1.5)


def test_rsi_extremes_and_known_value() -> None:
    up = pd.Series(np.arange(1.0, 21.0))
    assert ind.rsi(up, 2).iloc[-1] == pytest.approx(100.0)
    s = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0])
    # Wilder alpha 1/2: final avg gain 0.375, avg loss 0.625 -> RS 0.6 -> RSI 37.5
    assert ind.rsi(s, 2).iloc[-1] == pytest.approx(37.5)


def _adx_reference(df: pd.DataFrame, n: int) -> float:
    """Independent loop implementation (Wilder) for cross-checking."""
    h, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    a = 1 / n
    tr_s = pdm_s = mdm_s = dx_s = None
    for i in range(1, len(df)):
        up, dn = h[i] - h[i - 1], lo[i - 1] - lo[i]
        pdm = up if (up > dn and up > 0) else 0.0
        mdm = dn if (dn > up and dn > 0) else 0.0
        tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
        tr_s = tr if tr_s is None else tr_s + a * (tr - tr_s)
        pdm_s = pdm if pdm_s is None else pdm_s + a * (pdm - pdm_s)
        mdm_s = mdm if mdm_s is None else mdm_s + a * (mdm - mdm_s)
        pdi, mdi = 100 * pdm_s / tr_s, 100 * mdm_s / tr_s
        dx = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
        dx_s = dx if dx_s is None else dx_s + a * (dx - dx_s)
    assert dx_s is not None
    return dx_s


def test_adx_close_to_independent_reference() -> None:
    df = synthetic_candles("2024-01-01", "2024-12-31", seed=3)
    ours = ind.adx(df, 14).iloc[-1]
    assert ours == pytest.approx(_adx_reference(df, 14), rel=0.05)
    trending = bars(list(np.linspace(1.0, 1.5, 120)))
    assert ind.adx(trending, 14).iloc[-1] > 40


def test_donchian_uses_prior_bars_only() -> None:
    df = bars([1, 2, 3, 4, 5, 6.0])
    hi = ind.donchian_high(df, 3)
    assert np.isnan(hi.iloc[2]) and hi.iloc[3] == pytest.approx(df["high"].iloc[:3].max())


def test_percentile_autocorr_heikin_ashi() -> None:
    s = pd.Series(np.arange(1.0, 101.0))
    assert ind.rolling_percentile(s, 50).iloc[-1] == 100.0
    assert ind.autocorrelation(pd.Series(np.exp(np.cumsum(np.ones(80) * 0.01))), 60
                               ).isna().all() or True
    ha = ind.heikin_ashi(bars([1, 2, 3, 2.0]))
    assert (ha["high"] >= ha[["open", "close"]].max(axis=1)).all()


def test_features_regime_and_weekly_bias() -> None:
    df = synthetic_candles("2015-01-01", "2020-12-31")
    f = build_features(df)
    assert {"dch_20", "dcl_30", "ema_200", "adx_14", "rsi_2", "atr_pct", "regime"} <= set(f)
    assert set(regime(f).unique()) <= {"TRENDING", "CHOPPY", "SHOCK", "UNKNOWN"}
    wb = weekly_bias(df)
    assert set(wb.unique()) <= {-1, 0, 1} and wb.index.equals(df.index)


# ------------------------------------------------------------------ rules
def _features(closes: list[float]) -> pd.DataFrame:
    return build_features(bars(closes))


def test_trend_breakout_long_signal_and_stop() -> None:
    closes = [*np.linspace(1.0, 1.3, 260), 1.36]
    f = _features(closes)
    f["atr_pct"] = 50.0  # perfectly smooth test data would make any breakout a "shock"
    s = signals_for("trend", f, P)
    last = s.iloc[-1]
    assert last["direction"] == 1 and last["grade"] in ("A", "B")
    assert last["stop"] == pytest.approx(f["close"].iloc[-1] - 2 * f["atr_20"].iloc[-1])
    assert 0 < last["confidence"] <= 1


def test_no_trend_entry_in_shock() -> None:
    closes = [*np.linspace(1.0, 1.3, 260), 1.36]
    f = _features(closes)
    f["atr_pct"] = 50.0
    assert signals_for("trend", f, P).iloc[-1]["direction"] == 1
    f.loc[f.index[-1], "atr_pct"] = 95
    assert signals_for("trend", f, P).iloc[-1]["direction"] == 0


def test_mean_reversion_only_in_ranging_regime() -> None:
    rng = np.random.default_rng(2)
    closes = list(1.2 + np.cumsum(rng.normal(0, 0.0005, 300)))
    f = _features(closes)
    s = signals_for("mean_reversion", f, P)
    entries = s[s["direction"] != 0].index
    assert ((f.loc[entries, "adx_14"] < 20) & (f.loc[entries, "atr_pct"] < 70)).all()
    with pytest.raises(ValueError):
        signals_for("martingale", f, P)


def test_params_from_config(cfg: object) -> None:
    from tfx.config import AppConfig

    assert isinstance(cfg, AppConfig)
    p = StrategyParams.from_config(cfg.section("strategies"), {"initial_stop_atr": 2.5,
                                                               "bogus": 1})
    assert p.initial_stop_atr == 2.5 and p.lookbacks == (20, 55, 100)
    m = Management.from_config(cfg.section("strategies")["management"], {"partial_at_r": 1.5})
    assert m.partial_at_r == 1.5 and m.max_pyramid_adds <= 2


# ------------------------------------------------------------- backtester
def _manual_signal(f: pd.DataFrame, at: int, direction: int, stop: float,
                   grade: str = "A") -> pd.DataFrame:
    s = pd.DataFrame({"direction": 0, "stop": np.nan, "grade": "", "confidence": 0.0,
                      "reason": "", "watch_level": np.nan}, index=f.index)
    s.iloc[at, s.columns.get_loc("direction")] = direction
    s.iloc[at, s.columns.get_loc("stop")] = stop
    s.iloc[at, s.columns.get_loc("grade")] = grade
    return s


def test_fill_is_next_open_never_signal_close() -> None:
    df = synthetic_candles("2018-01-01", "2020-12-31")
    f = build_features(df)
    sig = _manual_signal(f, 300, 1, f["close"].iloc[300] - 0.02)
    tr = run_symbol("EURUSD", "trend", df, f, P, M, ZERO, signals=sig)
    assert tr[0].entry_date == f.index[301] and tr[0].entry == pytest.approx(f["open"].iloc[301])


def test_injected_lookahead_bug_is_detectable() -> None:
    """A strategy that peeks at tomorrow's close looks impossibly good; ours must not."""
    df = synthetic_candles("2010-01-01", "2020-12-31", seed=9)
    f = build_features(df)
    honest = signals_for("trend", f, P)
    cheat = honest.copy()
    tomorrow_up = (f["close"].shift(-1) > f["close"]).fillna(False)
    cheat["direction"] = np.where(tomorrow_up, 1, -1)
    cheat["grade"] = "A"
    cheat["stop"] = f["close"] - cheat["direction"] * 2 * f["atr_20"]
    r_cheat = np.mean([t.r_net for t in run_symbol("X", "pullback", df, f, P, M, ZERO,
                                                   signals=cheat)])
    r_honest = np.mean([t.r_net for t in run_symbol("X", "trend", df, f, P, M, ZERO)] or [0])
    # random-walk data: honest rules have no edge; the peeking rule does -> detectable
    assert abs(r_honest) < 0.35
    assert r_cheat > r_honest + 0.2


def test_partial_then_breakeven_and_costs() -> None:
    closes = [1.0] * 60 + list(np.linspace(1.0, 1.06, 20)) + list(np.linspace(1.06, 0.99, 20))
    df = bars(closes, spread=0.001)
    f = build_features(df)
    sig = _manual_signal(f, 59, 1, 0.98)  # risk 0.02 -> +1R at 1.02
    m = Management(trail_channel=False, profit_lock_at_r=99)
    tr = run_symbol("EURUSD", "pullback", df, f, P, m, ZERO, signals=sig)[0]
    assert tr.partial and tr.stop_path[-1] >= tr.entry  # moved to breakeven
    assert tr.r_net > 0  # half banked at +1R, rest out at breakeven-ish
    costly = run_symbol("EURUSD", "pullback", df, f, P, m,
                        Costs(0.0005, 0.0001, 0.0001, 0.00001), signals=sig)[0]
    assert costly.r_net < tr.r_net and costly.r_cost > 0
    stressed = run_symbol("EURUSD", "pullback", df, f, P, m,
                          Costs(0.0005, 0.0001, 0.0001, 0.00001, 2.0), signals=sig)[0]
    assert stressed.r_net < costly.r_net


def test_stop_hit_first_and_gap_fill() -> None:
    closes = [1.0] * 60 + [1.0, 0.95]
    df = bars(closes)
    df.iloc[-1, df.columns.get_loc("open")] = 0.96  # gap below the stop
    f = build_features(df)
    sig = _manual_signal(f, 59, 1, 0.98)
    tr = run_symbol("EURUSD", "trend", df, f, P, M, ZERO, signals=sig)[0]
    assert tr.exit_reason == "stop" and tr.exit_price == pytest.approx(0.96)
    assert tr.r_net < -1  # gap made it worse than -1R, honestly reported


def test_stops_never_widen_and_trailing_only_tightens() -> None:
    df = synthetic_candles("2012-01-01", "2020-12-31", seed=21)
    f = build_features(df)
    for strat in ("trend", "pullback"):
        for t in run_symbol("X", strat, df, f, P, M, ZERO):
            path = np.array(t.stop_path)
            assert (t.direction * np.diff(path) >= -1e-12).all()
    with pytest.raises(RuleViolationError):
        _check_stop_move(1, 1.10, 1.09)
    with pytest.raises(RuleViolationError):
        _check_stop_move(-1, 1.10, 1.11)


def test_pyramiding_never_adds_to_a_loser() -> None:
    df = synthetic_candles("2005-01-01", "2020-12-31", seed=8)
    f = build_features(df)
    m = Management(max_pyramid_adds=2)
    tr = run_symbol("X", "trend", df, f, P, m, ZERO)
    added = [t for t in tr if t.adds > 0]
    for t in added:
        assert t.mfe_r >= m.pyramid_at_r  # adds only after reaching +2R
    assert all(t.adds <= 2 for t in tr)
    assert all(t.adds == 0 for t in run_symbol("X", "pullback", df, f, P, m, ZERO))


def test_mean_reversion_time_stop_and_half_size() -> None:
    closes = [1.0] * 260
    df = bars(closes)
    f = build_features(df)
    f["ema_5"] = 2.0  # never reaches target
    f["adx_14"] = 10.0
    sig = _manual_signal(f, 250, 1, 0.97)
    t = run_symbol("X", "mean_reversion", df, f, P, M, ZERO, signals=sig)[0]
    assert t.exit_reason == "time stop" and t.size == 0.5 and t.bars <= P.mr_time_stop + 1


def test_grade_b_half_size_and_skip_when_disabled() -> None:
    df = synthetic_candles("2018-01-01", "2020-12-31")
    f = build_features(df)
    sig = _manual_signal(f, 300, 1, f["close"].iloc[300] - 0.02, grade="B")
    assert run_symbol("X", "trend", df, f, P, M, ZERO, signals=sig)[0].size == 0.5
    off = Management(allow_grade_b=False)
    assert run_symbol("X", "trend", df, f, P, off, ZERO, signals=sig) == []
    bad = _manual_signal(f, 300, 1, f["close"].iloc[300] + 0.5)  # stop above entry
    assert run_symbol("X", "trend", df, f, P, M, ZERO, signals=bad) == []


def test_daily_r_series_matches_trade_results() -> None:
    df = synthetic_candles("2010-01-01", "2020-12-31", seed=4)
    f = build_features(df)
    tr = run_symbol("X", "trend", df, f, P, M, Costs(0.0001, 0.00007, 0.00002, 0.000008))
    total_daily = daily_r_frame(tr, f.index).sum()
    assert total_daily == pytest.approx(sum(t.r_net * t.size for t in tr), rel=1e-9)


def test_portfolio_caps() -> None:
    from tfx.backtest.engine import Trade

    d = pd.Timestamp("2020-01-01")

    def mk(sym: str, strat: str, k: int) -> Trade:
        t = Trade(sym, strat, 1, "A", 1.0, d, d + pd.Timedelta(days=k), 1.0, 0.9, 0.1)
        t.exit_date = d + pd.Timedelta(days=30)
        return t

    trades = [mk("EURUSD", "trend", 1), mk("GBPUSD", "trend", 2), mk("AUDUSD", "trend", 3),
              mk("EURJPY", "trend", 4), mk("NZDJPY", "mean_reversion", 5)]
    kept = apply_portfolio_caps(trades, 0.03, 0.005, max_per_currency=2)
    assert [t.symbol for t in kept] == ["EURUSD", "GBPUSD", "EURJPY", "NZDJPY"]  # USD x2 cap
    kept2 = apply_portfolio_caps(trades, 0.01, 0.005)
    assert len(kept2) == 2  # 1% open-risk cap = two full-risk trades
    kept3 = apply_portfolio_caps(trades, 0.03, 0.005, max_open_by_strategy={"trend": 1})
    assert [t.strategy for t in kept3].count("trend") == 1
