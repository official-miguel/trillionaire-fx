"""ICT primitives, setups, simulation and chart objects."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from tests.conftest import synthetic_candles
from tfx.backtest.engine import Costs
from tfx.setup_render import render_ict
from tfx.strategies.features import build_features
from tfx.strategies.ict import ICTParams, chart_annotations, find_setups, simulate, swings

ZERO = Costs(0.0, 0.0, 0.0, 0.0)


def textbook_long() -> pd.DataFrame:
    """Range with a swing low at 1.0950 and a swing high at 1.1050, then:
    sweep below 1.0950 closing back inside, displacement up leaving an FVG, and a close
    above 1.1050 (MSS)."""
    rows = []
    base = [1.1000, 1.1010, 1.1030, 1.1050, 1.1030, 1.1010, 1.0990, 1.0970, 1.0950, 1.0970,
            1.0990, 1.1000, 1.0990, 1.0980]
    for cl in base * 3:
        rows.append((cl - 0.0003, cl + 0.0008, cl - 0.0008, cl))
    rows.append((1.0975, 1.0978, 1.0920, 1.0960))   # sweep: low 1.0920 < 1.0942, close back in
    rows.append((1.0960, 1.0995, 1.0955, 1.0992))
    rows.append((1.0992, 1.1060, 1.0990, 1.1035))   # displacement (MSS close)
    rows.append((1.1036, 1.1075, 1.1010, 1.1070))   # gap: low 1.1010 > high two bars ago 1.0995
    for k in range(12):                              # drift so swings confirm + room to fill
        cl = 1.1060 - 0.0004 * k
        rows.append((cl + 0.0002, cl + 0.0006, cl - 0.0006, cl))
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_swings_need_bars_on_both_sides() -> None:
    df = textbook_long()
    sh, sl = swings(df, 3)
    assert not sh[-3:].any() and not sl[-3:].any()     # last n bars can never be swings yet
    assert sh.any() and sl.any()


def test_textbook_setup_detected_with_correct_levels() -> None:
    df = textbook_long()
    p = ICTParams(swing_n=3, fvg_expiry=10)
    setups = [s for s in find_setups(df, p) if s.direction == 1]
    assert setups, "expected a bullish sweep -> MSS -> FVG setup"
    st = setups[-1]
    assert st.sweep_price == pytest.approx(1.0920)
    assert st.fvg_low < st.entry < st.fvg_high
    assert st.entry == pytest.approx((st.fvg_low + st.fvg_high) / 2)
    assert st.stop < st.sweep_price                     # beyond the sweep extreme
    assert st.entry <= (st.range_low + st.range_high) / 2 + 1e-12   # discount
    assert st.target == pytest.approx(st.entry + 2 * (st.entry - st.stop))
    assert 0 <= st.quality <= 100


def test_flex_tolerances_accept_near_textbook_setups() -> None:
    """Real markets rarely form the textbook pattern: the tolerant variant must accept a
    setup whose FVG sits just above the strict 50% line, the strict one must not."""
    df = textbook_long()
    df.iloc[44, df.columns.get_loc("high")] = 1.1040      # range mid now just below the CE
    strict = [s for s in find_setups(df, ICTParams(swing_n=3, fvg_expiry=10))
              if s.direction == 1 and s.sweep_bar == 42]
    flex = ICTParams(swing_n=3, fvg_expiry=10, sweep_close_tol_atr=0.1, pool_tol_atr=0.1,
                     gap_tol_atr=0.05, discount_max=0.618)
    tolerant = [s for s in find_setups(df, flex) if s.direction == 1 and s.sweep_bar == 42]
    assert not strict and tolerant
    st = tolerant[-1]
    assert (st.entry - st.range_low) / (st.range_high - st.range_low) <= 0.618


def test_params_from_config(cfg: object) -> None:
    from tfx.config import AppConfig

    assert isinstance(cfg, AppConfig)
    strict = ICTParams.from_config(cfg.section("strategies")["ict"], {"swing_n": 5})
    flex = ICTParams.from_config(cfg.section("strategies")["ict_flex"])
    assert strict.discount_max == 0.5 and strict.gap_tol_atr == 0 and strict.swing_n == 5
    assert flex.discount_max == 0.618 and flex.sweep_close_tol_atr == 0.1


def test_no_lookahead_each_setup_rebuilt_from_truncated_data() -> None:
    df = synthetic_candles("2010-01-01", "2016-12-31", seed=5)
    p = ICTParams(swing_n=3)
    full = find_setups(df, p)
    assert len(full) > 5
    for st in full[:15]:
        cut = df.iloc[: st.mss_bar + 1]
        again = [s for s in find_setups(cut, p) if s.mss_bar == st.mss_bar]
        assert again and asdict(again[-1]) == asdict(st)


def test_simulation_conservative_and_stops_fixed() -> None:
    df = synthetic_candles("2005-01-01", "2020-12-31", seed=7)
    for target in ("2R", "liquidity"):
        trades = simulate("EURUSD", df, ICTParams(target=target), ZERO)
        assert trades
        for t in trades:
            assert t.stop_path == [t.stop0]                 # ICT stop never moves
            assert t.r_net >= -1.5                          # zero-cost loss ~ -1R (gaps)
            assert t.exit_reason in ("stop", "time stop", "end of test",
                                     "target (2R)", "target (liquidity)")
            assert t.entry_date > t.signal_date             # limit filled AFTER the MSS bar
    wins = [t for t in simulate("EURUSD", df, ICTParams(target="2R"), ZERO)
            if t.exit_reason == "target (2R)"]
    assert all(t.r_net == pytest.approx(2.0, abs=0.05) for t in wins)
    costly = simulate("EURUSD", df, ICTParams(), Costs(0.0002, 0.00007, 0.00002, 0.00001))
    free = simulate("EURUSD", df, ICTParams(), ZERO)
    assert sum(t.r_net for t in costly) < sum(t.r_net for t in free)


def test_chart_annotations_have_labels_and_icons() -> None:
    df = synthetic_candles("2023-01-01", "2024-12-31", seed=9)
    a = chart_annotations(df, ICTParams())
    assert set(a) >= {"liquidity", "fvgs", "events", "zones", "kill_zones_utc"}
    for group in ("liquidity", "fvgs", "events", "zones"):
        for obj in a[group]:
            assert obj["label"] and obj["icon"]
    assert all(x["label"].startswith("LQ") for x in a["liquidity"])
    assert chart_annotations(df.iloc[:5], ICTParams())["fvgs"] == []


def test_render_ict_box_matches_setup() -> None:
    df = textbook_long()
    st = [s for s in find_setups(df, ICTParams(swing_n=3, fvg_expiry=10)) if s.direction == 1][-1]
    f = build_features(df)
    r = render_ict(st, f, 5, False, 5.0, "USD")
    assert r["box"]["entry"] == round(st.entry, 5) and r["box"]["stop"] == round(st.stop, 5)
    assert r["box"]["target"] == round(st.target, 5)
    ids = {line["id"] for line in r["lines"]}
    assert {"entry", "stop", "target", "lq", "mss"} <= ids
    assert r["zones"][0]["label"].startswith("FVG")
    assert r["summary"].startswith("NOT VALIDATED") and "Sweep of" in r["summary"]
    assert "R:R" in r["summary"] and np.isfinite(r["box"]["rr"])
