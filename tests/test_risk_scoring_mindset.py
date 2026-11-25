"""Risk engine, trade manager, scoring, mindset, coach, execution, missed trades, paper."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tfx import coach
from tfx.config import AppConfig
from tfx.execution import cost_in_r, execution_quality, order_plan, slippage_report
from tfx.mindset import CHECKLIST, DisciplineError, Mindset
from tfx.missed import missed_report, reentry_allowed, second_chance
from tfx.paper import paper_status
from tfx.risk.engine import (
    FxConverter,
    Limits,
    OpenPosition,
    RiskContext,
    hard_rule_violations,
    health_check,
    in_rollover_window,
    kelly_fraction,
    position_size,
    vol_target_scale,
)
from tfx.risk.manager import ManagedTrade, next_actions
from tfx.scoring import factor_values, fit_weights, score_setup
from tfx.store import Store, TradeRow

FX = FxConverter({"EURUSD": 1.10, "USDJPY": 150.0, "GBPUSD": 1.30, "USDKES": 129.0})
ALL_OK = {k: True for k in CHECKLIST}


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    yield s  # type: ignore[misc]
    s.close()


# ---------------------------------------------------------------- sizing
def test_fx_conversion_paths() -> None:
    assert FX.rate("USD", "USD") == 1
    assert FX.rate("EUR", "USD") == pytest.approx(1.10)
    assert FX.rate("JPY", "USD") == pytest.approx(1 / 150)
    assert FX.rate("JPY", "KES") == pytest.approx(129 / 150)
    with pytest.raises(KeyError):
        FX.rate("XYZ", "USD")


def test_position_size_usd_quote(cfg: AppConfig) -> None:
    ps = position_size(1000, 0.5, 1.1000, 1.0950, cfg.symbol("EURUSD"), "USD", FX)
    assert ps.amount_at_risk == pytest.approx(5.0)
    assert ps.units == pytest.approx(1000) and ps.stop_distance_pips == pytest.approx(50)
    assert ps.units * 0.0050 == pytest.approx(5.0)  # loss at the stop = amount at risk


def test_position_size_jpy_quote_and_kes_account(cfg: AppConfig) -> None:
    ps = position_size(1000, 0.5, 150.00, 149.00, cfg.symbol("USDJPY"), "USD", FX)
    assert ps.units * 1.0 / 150 == pytest.approx(5.0)
    kes = position_size(100_000, 0.5, 1.1, 1.09, cfg.symbol("EURUSD"), "KES", FX)
    assert kes.amount_at_risk == pytest.approx(500) and kes.account_ccy == "KES"
    half = position_size(1000, 0.5, 1.1, 1.095, cfg.symbol("EURUSD"), "USD", FX, 0.5)
    assert half.amount_at_risk == pytest.approx(2.5)
    capped = position_size(1000, 0.5, 1.1, 1.095, cfg.symbol("EURUSD"), "USD", FX, 3.0)
    assert capped.amount_at_risk == pytest.approx(5.0)  # multiplier can never exceed 1


def test_position_size_rejects_bad_inputs(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="hard cap"):
        position_size(1000, 1.5, 1.1, 1.09, cfg.symbol("EURUSD"), "USD", FX)
    with pytest.raises(ValueError, match="stop"):
        position_size(1000, 0.5, 1.1, 1.1, cfg.symbol("EURUSD"), "USD", FX)
    with pytest.raises(ValueError):
        position_size(1000, 0.5, 1.1, float("nan"), cfg.symbol("EURUSD"), "USD", FX)


# ---------------------------------------------------------------- hard rules
def _ctx(**kw: object) -> RiskContext:
    base: dict[str, object] = {"risk_pct": 0.5, "equity_peak": 1000, "equity_now": 1000}
    base.update(kw)
    return RiskContext(**base)  # type: ignore[arg-type]


def test_hard_rules_block_each_violation() -> None:
    lim = Limits()
    assert hard_rule_violations("EURUSD", 1.09, _ctx(), lim) == []
    assert any("No stop" in v for v in hard_rule_violations("EURUSD", None, _ctx(), lim))
    assert any("two sources" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(data_confirmed=False), lim))
    assert any("validation" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(pair_validated=False), lim))
    assert any("health" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(strategy_health="DISABLED"), lim))
    assert any("hard cap" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(), lim, new_risk_pct=1.2))
    assert any("Kelly" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(kelly_fraction=0.01), lim))
    assert any("PAUSED - REVIEW" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(equity_now=800), lim))
    assert hard_rule_violations("EURUSD", 1.09, _ctx(equity_now=800, breaker_reviewed=True),
                                lim) == []
    six = [OpenPosition("AUDJPY", "trend", 1, 0.5)] * 6
    assert any("Open risk" in v for v in hard_rule_violations(
        "NZDCAD", 1.0, _ctx(open_positions=six), lim))
    two_usd = [OpenPosition("GBPUSD", "trend", 1, 0.5), OpenPosition("AUDUSD", "trend", 1, 0.5)]
    assert any("exposed to USD" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(open_positions=two_usd), lim))
    assert any("Spread" in v for v in hard_rule_violations(
        "EURUSD", 1.09, _ctx(spread_now=3.0, typical_spread=0.8), lim))
    roll = datetime(2026, 11, 20, 21, 50, tzinfo=UTC)
    assert in_rollover_window(roll)
    assert any("rollover" in v for v in hard_rule_violations("EURUSD", 1.09, _ctx(now=roll), lim))


def test_limits_from_config(cfg: AppConfig) -> None:
    lim = Limits.from_config(cfg.section("risk"), cfg.section("strategies"), cfg.section("costs"))
    assert lim.max_open_risk_pct == 3.0 and lim.drawdown_breaker_pct == 15.0
    assert lim.max_per_currency == 2 and lim.kelly_cap == 0.25


def test_kelly_and_vol_target_never_increase_size() -> None:
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.25)
    assert kelly_fraction(0.4, 1.0) < 0  # no edge
    assert kelly_fraction(0.5, 0) == 0.0
    rng = np.random.default_rng(1)
    assert vol_target_scale(rng.normal(0, 0.001, 300)) == 1.0  # low vol: never scaled UP
    assert vol_target_scale(rng.normal(0, 0.02, 300)) < 1.0
    assert vol_target_scale(np.zeros(5)) == 1.0


def test_health_monitor() -> None:
    bt = list(np.random.default_rng(2).normal(0.2, 1.0, 300))
    similar = list(np.random.default_rng(7).normal(0.2, 1.0, 30))
    assert health_check(similar, [0.001] * 200, bt, -0.1).status == "OK"
    bad = health_check([-1.0] * 30, [0.0] * 10, bt, -0.1)
    assert bad.status == "PAUSED" and bad.size_factor == 0
    neg = health_check([], list(np.random.default_rng(3).normal(-0.002, 0.005, 260)), bt, -0.5)
    assert neg.status == "DISABLED"
    dd = health_check([], [-0.02] * 20, bt, -0.1)
    assert dd.status == "DISABLED" and "1.5x" in dd.reasons[0]
    mild = health_check(list(np.random.default_rng(5).normal(-0.25, 1.0, 30)), [], bt, -0.1)
    assert mild.status in ("REDUCE", "PAUSED", "OK")


# -------------------------------------------------------------- trade manager
def _bar(**kw: float) -> pd.Series:
    b = {"open": 1.10, "high": 1.11, "low": 1.095, "close": 1.105, "dcl_20": 1.08,
         "dch_20": 1.12, "ema_5": 1.2, "adx_14": 22}
    b.update(kw)
    return pd.Series(b)


def test_manager_actions() -> None:
    t = ManagedTrade("EURUSD", "trend", 1, 1.10, 1.09, 1.09, pd.Timestamp("2026-11-01"))
    assert next_actions(t, _bar(low=1.085), 3, 0.0001)[0].code == "STOP_HIT"
    acts = {a.code for a in next_actions(t, _bar(high=1.111), 3, 0.0001)}
    assert "TAKE_PARTIAL" in acts
    assert next_actions(t, _bar(close=1.07, low=1.091), 3, 0.0001)[0].code == "EXIT_RULE"
    trail = next_actions(t, _bar(dcl_20=1.095, low=1.096), 3, 0.0001)
    assert any(a.code == "TRAIL" and a.new_stop == pytest.approx(1.095) for a in trail)
    assert next_actions(t, _bar(high=1.104, low=1.096, close=1.1), 3, 0.0001)[0].code == "HOLD"


def test_manager_never_suggests_add_to_loser_or_wider_stop() -> None:
    t = ManagedTrade("EURUSD", "trend", 1, 1.10, 1.09, 1.10, pd.Timestamp("2026-11-01"),
                     partial_done=True)
    up = next_actions(t, _bar(high=1.125, low=1.101, close=1.12, dcl_20=1.05), 9, 0.0001)
    assert any(a.code == "PYRAMID_OK" for a in up)
    loser = ManagedTrade("EURUSD", "trend", 1, 1.10, 1.09, 1.09, pd.Timestamp("2026-11-01"))
    down = next_actions(loser, _bar(high=1.121, low=1.091, close=1.095, dcl_20=1.05), 9, 0.0001)
    assert not any(a.code == "PYRAMID_OK" for a in down)  # not in profit -> never add
    for a in up + down:
        if a.new_stop is not None:
            assert a.new_stop >= 1.09  # long stops only move up
    mr = ManagedTrade("EURUSD", "mean_reversion", 1, 1.10, 1.09, 1.09,
                      pd.Timestamp("2026-11-01"))
    assert next_actions(mr, _bar(high=1.104, low=1.096, close=1.1), 5, 0.0001)[0].code == \
        "TIME_STOP"
    assert next_actions(mr, _bar(high=1.104, low=1.096, close=1.1, ema_5=1.09), 1, 0.0001
                        )[0].code == "TARGET"
    assert next_actions(mr, _bar(high=1.104, low=1.096, close=1.1, adx_14=30), 1, 0.0001
                        )[0].code == "TREND_EMERGING"


def test_no_martingale_or_size_increase_after_losses() -> None:
    """Static guard: sizing never reads past losses to INCREASE size."""
    import tfx.risk.engine as eng
    import tfx.scoring as sc

    src = inspect.getsource(eng) + inspect.getsource(sc)
    for bad in ("martingale", "double_after_loss", "average_down", "remove_stop"):
        assert bad not in src.lower()
    v = factor_values(_bar(atr_pct=50), 1, 1.0, 2.0, 0, 0, 1, 1.0, 1.0, False)
    v_losses = factor_values(_bar(atr_pct=50), 1, 1.0, 2.0, 0, 4, 1, 1.0, 1.0, False)
    assert v_losses["recent_loss_streak"] < v["recent_loss_streak"]  # losses only REDUCE


# ---------------------------------------------------------------- scoring
def test_scoring_maps_to_size_and_hard_rules_block(cfg: AppConfig) -> None:
    w = cfg.scoring_factors()
    great = factor_values(_bar(adx_14=40, atr_pct=50), 1, 1.0, 3.0, 0, 0, 1, 1.0, 1.0, False)
    s = score_setup(great, w, [])
    assert s.score >= 70 and s.grade == "A" and s.size_multiplier == 1.0
    weak = factor_values(_bar(adx_14=10, atr_pct=95), 1, 0.0, 0.2, 2, 4, -1, 0.0, 0.0, True)
    assert score_setup(weak, w, []).grade == "WAIT"
    mid = {k: 0.6 for k in great}
    sm = score_setup(mid, w, [])
    assert sm.grade == "B" and 0.5 <= sm.size_multiplier <= 0.75
    probe = score_setup({k: 0.4 for k in great}, w, [])
    assert probe.grade == "PROBE" and probe.size_multiplier == 0.25
    blocked = score_setup(great, w, ["No stop defined"])
    assert blocked.blocked and blocked.size_multiplier == 0 and blocked.score >= 70
    assert score_setup(great, w, [], "WATCH").grade == "WATCH"
    assert score_setup(great, w, [], "B").size_multiplier <= 0.5
    assert s.factors["news_proximity"]["mode"] == "informational"


def test_soft_factors_only_change_size_never_block(cfg: AppConfig) -> None:
    w = cfg.scoring_factors()
    worst = {k: 0.0 for k in factor_values(_bar(), 1, 0, 0, 0, 0, 0, 0, 0, None)}
    s = score_setup(worst, w, [])
    assert not s.blocked and s.hard_blocks == []


def test_weights_frozen_before_holdout(cfg: AppConfig) -> None:
    w = cfg.scoring_factors()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({k: rng.random(60) for k in w})
    df["r"] = df["trend_clarity"] * 2 - 1 + rng.normal(0, 0.1, 60)
    df["entry_date"] = pd.date_range("2015-01-01", periods=60, freq="W")
    out = fit_weights(df, w, pd.Timestamp("2023-01-01"))
    assert out["trend_clarity"] > w["trend_clarity"].weight * 0.9
    assert out["news_proximity"] == w["news_proximity"].weight  # informational stays fixed
    leaky = df.copy()
    leaky.loc[59, "entry_date"] = pd.Timestamp("2024-06-01")
    with pytest.raises(ValueError, match="holdout"):
        fit_weights(leaky, w, pd.Timestamp("2023-01-01"))


# ---------------------------------------------------------------- mindset
def _trade(direction: int = 1, stop: float = 1.09, signal_id: int | None = 1) -> TradeRow:
    return TradeRow(signal_id=signal_id, symbol="EURUSD", strategy="trend",
                    direction=direction, entry=1.10, units=1000, initial_stop=stop, stop=stop,
                    risk_pct=0.5, amount_at_risk=5.0)


def test_emotion_check_wait_15_minutes(store: Store, cfg: AppConfig) -> None:
    clock = Clock(datetime(2026, 11, 20, 10, tzinfo=UTC))
    m = Mindset(store, cfg.section("mindset"), clock)
    assert not m.emotion_gate().allowed  # no check-in yet
    g = m.check_in(3, 4, 2)
    assert not g.allowed and g.blockers == ["Wait. Re-read the plan."]
    clock.t += timedelta(minutes=16)
    assert m.emotion_gate().allowed
    with pytest.raises(ValueError):
        m.check_in(0, 1, 1)


def test_entry_requires_stop_and_checklist(store: Store, cfg: AppConfig) -> None:
    clock = Clock(datetime(2026, 11, 17, 10, tzinfo=UTC))
    m = Mindset(store, cfg.section("mindset"), clock)
    m.check_in(3, 2, 2)
    with pytest.raises(DisciplineError, match="stop"):
        m.open_trade(_trade(stop=1.11), ALL_OK)          # stop on the wrong side
    with pytest.raises(DisciplineError, match="Checklist"):
        m.open_trade(_trade(), {**ALL_OK, "calm_not_chasing": False})
    t = m.open_trade(_trade(), ALL_OK)
    assert t.id and not t.off_signal and '"stress": 2' in t.emotions


def test_stops_only_tighten_and_attempts_are_logged(store: Store, cfg: AppConfig) -> None:
    m = Mindset(store, cfg.section("mindset"), Clock(datetime(2026, 11, 17, 10, tzinfo=UTC)))
    m.check_in(3, 2, 2)
    t = m.open_trade(_trade(), ALL_OK)
    assert m.update_stop(t.id, 1.095).stop == 1.095
    with pytest.raises(DisciplineError, match="tightened"):
        m.update_stop(t.id, 1.08)
    assert store.trade(t.id).stop == 1.095  # type: ignore[union-attr]
    kinds = [b.kind for b in store.breaches()]
    assert "STOP_WIDEN_ATTEMPT" in kinds
    assert [c.accepted for c in store.stop_changes(t.id)] == [True, False]


def test_weekly_limit_cooldown_and_breach_logging(store: Store, cfg: AppConfig) -> None:
    clock = Clock(datetime(2026, 11, 16, 9, tzinfo=UTC))  # Monday
    m = Mindset(store, cfg.section("mindset"), clock)
    m.check_in(3, 1, 1)
    ids = [m.open_trade(_trade(), ALL_OK).id for _ in range(3)]
    with pytest.raises(DisciplineError, match="Weekly limit"):
        m.open_trade(_trade(), ALL_OK)
    forced = m.open_trade(_trade(), ALL_OK, override_reason="I insist")
    assert any(b.kind == "DISCIPLINE BREACH" and b.trade_id == forced.id
               for b in store.breaches())
    # two consecutive losing days -> 24h cooldown + review
    clock.t = datetime(2026, 11, 17, 15, tzinfo=UTC)
    m.close_trade(ids[0], 1.095, True, "")
    clock.t = datetime(2026, 11, 18, 15, tzinfo=UTC)
    m.close_trade(ids[1], 1.095, True, "")
    cd = m.cooldown()
    assert not cd.allowed and any("Cooldown" in b for b in cd.blockers)
    clock.t = datetime(2026, 11, 19, 16, tzinfo=UTC)
    assert not m.cooldown().allowed  # time passed, but no review yet
    m.submit_cooldown_review("Two stops in a row, both per plan. Volatility rose; I keep size.")
    assert m.cooldown().allowed


def test_off_signal_trade_is_tagged(store: Store, cfg: AppConfig) -> None:
    m = Mindset(store, cfg.section("mindset"), Clock(datetime(2026, 11, 17, 10, tzinfo=UTC)))
    m.check_in(3, 1, 1)
    t = m.open_trade(_trade(signal_id=None), ALL_OK)
    assert t.off_signal and store.breaches()[-1].detail.startswith("Off-signal")
    review = m.weekly_review()
    assert review["plan_adherence_pct"] == 0.0 and review["breaches"] == 1


def test_risk_locked_for_month_and_reflection(store: Store, cfg: AppConfig) -> None:
    clock = Clock(datetime(2026, 11, 3, tzinfo=UTC))
    m = Mindset(store, cfg.section("mindset"), clock)
    m.set_risk_pct(0.5)
    assert m.set_risk_pct(0.25) == 0.25          # lowering is allowed
    with pytest.raises(DisciplineError, match="locked"):
        m.set_risk_pct(0.75)
    clock.t = datetime(2026, 12, 1, tzinfo=UTC)
    assert m.set_risk_pct(0.75) == 0.75          # new month
    m.check_in(3, 1, 1)
    t = m.open_trade(_trade(), ALL_OK)
    m.close_trade(t.id, 1.07, False, "moved my stop")   # -3R and rules broken
    assert m.reflection_required()
    assert any("reflection" in b for b in m.entry_gate(ALL_OK).blockers)
    with pytest.raises(ValueError):
        m.submit_reflection("short")
    m.submit_reflection("I broke the plan by not exiting on the rule. Next time I exit at the "
                        "next open, no matter how I feel. Losses are a business cost.")
    assert not m.reflection_required()


def test_discipline_records_are_append_only(store: Store) -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError, OperationalError

    store.breach("TEST", "x")
    with store.engine.begin() as c, pytest.raises((IntegrityError, OperationalError)):
        c.execute(text("DELETE FROM breaches"))
    assert len(store.breaches()) == 1


# ---------------------------------------------------------------- coach
def test_coach_card_and_no_promises() -> None:
    card = coach.signal_card("EURUSD", "BUY", "trend", 1, "breakout", 1.10, 1.09,
                             {"amount_at_risk": 5}, {"text": "BUY STOP"}, None,
                             {"win_rate": 0.4}, None, 0.01, [], "TRENDING")
    assert card["headline"] == "BUY EURUSD at next open"
    assert card["targets"][0]["price"] == pytest.approx(1.11)
    assert "not a forecast" in card["historical"]["label"]
    assert len(card["skip_if"]) >= 3 and "invalid" in card["invalidation"]
    text = str(card)
    coach.assert_no_promises(text)
    with pytest.raises(ValueError):
        coach.assert_no_promises("This trade is a sure thing, guaranteed profit")
    assert coach.regime_banner("SHOCK")["size_factor"] == 0.0
    assert coach.headline("LOCKED", "X").startswith("WAIT")
    watch = coach.signal_card("EURUSD", "WATCH", "trend", 0, "", None, None, None, None, None,
                              None, 1.1, 0.01, [], "CHOPPY")
    assert "1.1" in watch["why"]


def test_post_trade_review() -> None:
    t = TradeRow(id=1, symbol="EURUSD", direction=1, entry=1.1, units=1, initial_stop=1.09,
                 stop=1.09, risk_pct=0.5, amount_at_risk=5, r_result=-1.0, mfe_r=0.4,
                 mae_r=-1.0, off_signal=False, lesson="")
    r = coach.post_trade_review(t, 0, True)
    assert r["plan_adherence"] and "business cost" in r["lesson"]


# -------------------------------------------------------------- execution
def test_order_plans() -> None:
    from datetime import date

    s = order_plan("trend", 1, 1.10, 0.01, 1.099, 1.095, date(2026, 11, 20))
    assert s.order_type == "STOP-ENTRY" and s.price > 1.10 and "Mon 23 Nov" in s.expiry
    p = order_plan("pullback", -1, 1.10, 0.01, None, 1.102, date(2026, 11, 20))
    assert p.order_type == "LIMIT" and p.price == 1.102
    m = order_plan("mean_reversion", 1, 1.10, 0.01, None, None, date(2026, 11, 20))
    assert m.order_type == "MARKET-AT-OPEN"


def test_execution_quality_and_costs() -> None:
    sbh = pd.Series({h: (40.0 if h in (21, 22) else 8.0) for h in range(24)})
    good = execution_quality(sbh, None, datetime(2026, 11, 20, 13, tzinfo=UTC))
    bad = execution_quality(sbh, None, datetime(2026, 11, 20, 21, 45, tzinfo=UTC))
    assert good["score"] > bad["score"] and bad["score"] <= 10
    assert execution_quality(pd.Series(dtype=float), None, datetime(2026, 11, 20, 13,
                                                                     tzinfo=UTC))["score"] > 50
    assert cost_in_r(0.8, 0.7, 0.2, 40) == pytest.approx(1.9 / 40)
    assert cost_in_r(1, 1, 1, 0) == float("inf")


def test_slippage_loop_triggers_revalidation() -> None:
    ok = slippage_report([("EURUSD", 0.2)] * 6, 0.2)
    assert not ok.revalidate
    worse = slippage_report([("EURUSD", 0.4)] * 6 + [("GBPUSD", 0.1)] * 6, 0.2)
    assert worse.revalidate and "EURUSD" in worse.over_assumption
    assert worse.over_assumption["EURUSD"] == pytest.approx(2.0)
    assert not slippage_report([], 0.2).revalidate


# ----------------------------------------------------------------- missed
def test_missed_report_matches_log() -> None:
    sigs = [{"id": 1, "status": "BUY", "symbol": "EURUSD", "size_mult": 1.0, "reason": ""},
            {"id": 2, "status": "SELL", "symbol": "GBPUSD", "size_mult": 1.0,
             "reason": "exposure cap"},
            {"id": 3, "status": "BUY", "symbol": "AUDUSD", "size_mult": 0.5, "reason": ""},
            {"id": 4, "status": "WAIT", "symbol": "NZDUSD", "size_mult": 0, "reason": ""}]
    rep = missed_report(sigs, {1, 3}, {2: -1.0, 3: 2.0})
    assert rep.valid_setups == 3 and rep.taken == 2
    assert rep.capture_rate == pytest.approx(2 / 3)
    ids = {r["id"] for r in rep.rows}
    assert ids == {2, 3}  # missed #2, reduced #3; taken #1 and WAIT #4 excluded
    assert rep.by_reason["exposure cap"]["sum_r"] == -1.0
    assert rep.net_effect_r == pytest.approx(1.0)


def test_second_chance_and_reentry_size_caps() -> None:
    ok = second_chance(1, 1.10, 1.105, 0.01, True, 1.08)
    assert ok is not None and ok["size_multiplier"] <= 0.25 and ok["stop"] == 1.08
    assert second_chance(1, 1.10, 1.12, 0.01, True, 1.08) is None   # > 1 ATR away
    assert second_chance(1, 1.10, 1.105, 0.01, False, 1.08) is None
    day = pd.Timestamp("2026-11-20")
    assert reentry_allowed(day, day, True)[0] is False                # same day: never
    allowed, size, _ = reentry_allowed(day, day + pd.Timedelta(days=3), True)
    assert allowed and size == 1.0                                     # never larger


# ----------------------------------------------------------------- paper
def test_paper_readiness() -> None:
    now = datetime(2026, 11, 20)
    early = paper_status([0.1] * 20, now - timedelta(weeks=3), now, (-0.1, 0.3))
    assert not early["ready_for_small_live_use"] and "8 weeks" in early["message"]
    ready = paper_status([0.1] * 20, now - timedelta(weeks=9), now, (-0.1, 0.3))
    assert ready["ready_for_small_live_use"] and "SMALL account" in ready["message"]
    outside = paper_status([-0.5] * 20, now - timedelta(weeks=9), now, (-0.1, 0.3))
    assert not outside["ready_for_small_live_use"]
    none = paper_status([], None, now, None)
    assert not none["ready_for_small_live_use"]
