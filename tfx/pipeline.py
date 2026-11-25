"""Signal pipeline: from confirmed data to a coached, risk-checked signal card.

Order of checks for every market (latest CLOSED, cross-checked bar only):
data confirmed -> strategy-timeframe validated -> rule fires -> hard rules -> soft score
-> size in account currency -> order plan -> coach card. Any failed hard rule means
the card says WAIT/BLOCKED with the reasons, never BUY/SELL.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from tfx import coach
from tfx.config import AppConfig
from tfx.consensus import Consensus, Vote, consensus
from tfx.core_types import Timeframe
from tfx.data.service import MarketData, MarketDataService
from tfx.execution import order_plan
from tfx.risk.engine import (
    FxConverter,
    Limits,
    OpenPosition,
    RiskContext,
    hard_rule_violations,
    kelly_fraction,
    position_size,
)
from tfx.risk.manager import ManagedTrade, next_actions
from tfx.scoring import factor_values, score_setup
from tfx.setup_render import levels as render_levels
from tfx.setup_render import render as render_setup
from tfx.setup_render import render_ict, render_sd
from tfx.store import SignalRow, Store
from tfx.strategies.features import build_features, weekly_bias
from tfx.strategies.ict import ICTParams, Setup, find_setups
from tfx.strategies.limit_sim import LimitOrder
from tfx.strategies.rules import STRATEGIES, StrategyParams, signals_for
from tfx.strategies.supply_demand import SDParams, Zone, active_order
from tfx.validation.proof import load_report
from tfx.validation.registry import ValidationRegistry
from tfx.validation.validators import ValidatorStatus, validators
from tfx.versioning import canonical_json


@dataclass
class Account:
    balance: float
    risk_pct: float
    currency: str


def account(cfg: AppConfig, store: Store) -> Account:
    return Account(float(store.get_setting("balance", cfg.account.starting_balance)),
                   float(store.get_setting("risk_pct", cfg.account.risk_per_trade_pct)),
                   str(store.get_setting("currency", cfg.account.currency)))


def fx_converter(svc: MarketDataService, store: Store) -> FxConverter:
    rates: dict[str, float] = {}
    for sym in svc.cfg.symbols:
        meta = svc.cache.meta(sym)
        if meta is not None:
            df = svc.cache.load(sym, start=meta.last_date)
            if not df.empty:
                rates[sym] = float(df["close"].iloc[-1])
    rates.update({k: float(v) for k, v in (store.get_setting("extra_fx_rates") or {}).items()})
    return FxConverter(rates)


def candle_quality(row: pd.Series, direction: int) -> float:
    rng = float(row["high"] - row["low"])
    if rng <= 0:
        return 0.0
    body = abs(float(row["close"] - row["open"])) / rng
    pos = (float(row["close"]) - float(row["low"])) / rng
    outer = pos >= 2 / 3 if direction == 1 else pos <= 1 / 3
    return 1.0 if (outer and body >= 0.5) else (0.5 if outer or body >= 0.5 else 0.0)


class Pipeline:
    def __init__(self, svc: MarketDataService, store: Store, registry: ValidationRegistry
                 ) -> None:
        self.svc = svc
        self.cfg = svc.cfg
        self.store = store
        self.registry = registry
        self._features: dict[tuple[str, str | None], pd.DataFrame] = {}
        self._eval_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.cache_seconds = 60.0

    def clear_cache(self) -> None:
        """Forget cached evaluations (called after any settings/trade change)."""
        self._eval_cache.clear()

    # ------------------------------------------------------------ helpers
    def features(self, md: MarketData) -> pd.DataFrame:
        key = (md.symbol, md.data_version)
        if key not in self._features:
            self._features = {k: v for k, v in self._features.items() if k[0] != md.symbol}
            self._features[key] = build_features(md.candles) if not md.candles.empty else \
                pd.DataFrame()
        return self._features[key]

    def hist_stats(self) -> dict[str, dict[str, Any]]:
        rep = load_report() or {}
        out = {}
        for p in rep.get("pairs", []):
            if p.get("full"):
                out[f"{p['strategy']}:{p['timeframe']}"] = {
                    "verdict": p["verdict"], "win_rate": p["full"]["win_rate"],
                    "avg_r": p["full"]["expectancy_r"], "payoff": p["full"]["payoff"],
                    "trades": p["full"]["trades"], "edge_confidence": p["edge_confidence"]}
        return out

    def open_positions(self, acct: Account) -> list[OpenPosition]:
        out = []
        for t in self.store.trades("open"):
            at_risk = 0.0 if t.direction * (t.stop - t.entry) >= 0 else t.risk_pct
            out.append(OpenPosition(t.symbol, t.strategy, t.direction, at_risk))
        return out

    def loss_streak(self) -> int:
        n = 0
        for t in reversed(self.store.trades("closed")):
            if (t.r_result or 0) < 0:
                n += 1
            else:
                break
        return n

    # ----------------------------------------------------------- per symbol
    def evaluate(self, symbol: str, refresh: bool = False, now: datetime | None = None,
                 persist: bool = True) -> dict[str, Any]:
        import time as _time

        hit = self._eval_cache.get(symbol)
        if not refresh and hit is not None and _time.monotonic() - hit[0] < self.cache_seconds:
            return hit[1]
        result = self._evaluate(symbol, refresh, now, persist)
        self._eval_cache[symbol] = (_time.monotonic(), result)
        return result

    def _evaluate(self, symbol: str, refresh: bool, now: datetime | None,
                  persist: bool) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        md = self.svc.get(symbol, refresh=refresh)
        f = self.features(md)
        acct = account(self.cfg, self.store)
        regime = str(f["regime"].iloc[-1]) if len(f) else "UNKNOWN"
        hist = self.hist_stats()
        base = {"symbol": symbol, "data": {"status": md.status.value, "message": md.message,
                                           "last_bar": str(md.last_bar or ""),
                                           "proxy": md.is_proxy},
                "last_closes": [float(x) for x in md.candles["close"].iloc[-2:]]
                if not md.candles.empty else []}
        # open trades first: HOLD / EXIT guidance
        open_trades = [t for t in self.store.trades("open") if t.symbol == symbol]
        if open_trades and len(f):
            t = open_trades[0]
            mt = ManagedTrade(t.symbol, t.strategy, t.direction, t.entry, t.initial_stop, t.stop,
                              pd.Timestamp(t.opened_at), t.partial_done, t.adds)
            held = int((f.index > pd.Timestamp(t.opened_at.date())).sum())
            sym_cfg = self.cfg.symbol(symbol)
            acts = next_actions(mt, f.iloc[-1], held, sym_cfg.typical_spread_pips * sym_cfg.pip)
            status = "EXIT" if any(a.code in ("STOP_HIT", "EXIT_RULE", "TARGET", "TIME_STOP",
                                              "TREND_EMERGING") for a in acts) else "HOLD"
            card = coach.signal_card(symbol, status, t.strategy, t.direction,
                                     acts[0].text, None, None, None, None, None, None, None,
                                     None, [], regime)
            card["actions"] = [{"code": a.code, "text": a.text, "new_stop": a.new_stop}
                               for a in acts]
            card["trade_r_now"] = round(mt.r_at(float(f["close"].iloc[-1])), 2)
            return {**base, "status": status, "card": card, "grade": "", "score": None}
        if not md.signal_allowed or len(f) < 220:
            card = coach.signal_card(symbol, "NO DATA", None, 0, md.message, None, None, None,
                                     None, None, None, None, None, [md.message], regime)
            return {**base, "status": "WAIT", "card": card, "grade": "", "score": None}

        params = StrategyParams.from_config(self.cfg.section("strategies"))
        mgmt = self.cfg.section("strategies").get("management", {})
        partial_at_r = float(mgmt.get("partial_at_r", 1.0))
        enabled = [s for s in STRATEGIES if self.cfg.section("strategies").get(s, {}).get(
            "enabled", False)]
        session = md.last_bar
        row = f.iloc[-1]
        sym_cfg = self.cfg.symbol(symbol)
        digits = 3 if sym_cfg.pip >= 0.01 else 5
        spread = sym_cfg.typical_spread_pips * sym_cfg.pip
        ict_variants = [v for v in ("ict", "ict_flex")
                        if self.cfg.section("strategies").get(v, {}).get("enabled", False)]
        vals_by = {s: v for s, v in validators(self.cfg).items()
                   if s in enabled or s in ict_variants or s == "supply_demand"}
        statuses = {s: v.status(self.registry, self.cfg, Timeframe.D1, now)
                    for s, v in vals_by.items()}
        watch: dict[str, Any] | None = None
        fired: dict[str, pd.Series] = {}
        for strat in enabled:
            sig = signals_for(strat, f, params).iloc[-1]
            if sig["grade"] == "WATCH" and watch is None:
                watch = {"strategy": strat, "level": float(sig["watch_level"])}
            if int(sig["direction"]) != 0 and np.isfinite(float(sig["stop"])):
                fired[strat] = sig
        # ICT: use a variant with a current PASS; otherwise show the tolerant variant's
        # setups for awareness (locked, view only).
        ict_name = next((v for v in ict_variants if statuses[v].can_vote),
                        "ict_flex" if "ict_flex" in ict_variants else
                        (ict_variants[0] if ict_variants else None))
        ict_setup = self.ict_active(md.candles, ict_name) if ict_name else None
        sd_on = bool(self.cfg.section("strategies").get("supply_demand", {}).get("enabled"))
        sd_setup = (active_order(md.candles, self.sd_params())
                    if sd_on and ict_setup is None and len(md.candles) > 220 else None)
        if fired or ict_setup is not None or sd_setup is not None:
            votes = [Vote(s, int(sig["direction"]), 0.0) for s, sig in fired.items()]
            if ict_setup is not None and ict_name is not None:
                votes.append(Vote(ict_name, ict_setup.direction, 0.0))
            if sd_setup is not None:
                votes.append(Vote("supply_demand", sd_setup[1].direction, 0.0))
            cons = consensus(votes, statuses)
            # Priority when no validated leader: ICT first, then supply & demand, then the
            # older rule strategies.
            strat = cons.leader or (
                ict_name if ict_setup is not None and ict_name else
                "supply_demand" if sd_setup is not None else next(iter(fired)))
            if cons.conflict:
                card = coach.signal_card(symbol, "WAIT", None, 0, cons.text, None, None, None,
                                         None, None, None, None, None, [cons.text], regime)
                card["consensus"] = {"text": cons.text, "voters": cons.voters,
                                     "excluded": cons.excluded}
                return {**base, "status": "WAIT", "card": card, "grade": "", "score": None}
            if strat in ("ict", "ict_flex") and ict_setup is not None:
                return self._ict_card(base, symbol, ict_setup, f, md, acct, digits,
                                      statuses[strat], cons, hist, regime, session, now,
                                      persist, strat)
            if strat == "supply_demand" and sd_setup is not None:
                return self._sd_card(base, symbol, sd_setup[0], sd_setup[1], f, md, acct,
                                     digits, statuses["supply_demand"], cons, hist, regime,
                                     session, now, persist)
            return self._fired_card(base, symbol, strat, fired[strat], f, md, acct, params,
                                    partial_at_r, digits, spread, statuses[strat], cons, hist,
                                    regime, session, now, persist)
        if watch is not None:
            card = coach.signal_card(symbol, "WATCH", watch["strategy"], 0, "", None, None, None,
                                     None, None, hist.get(f"{watch['strategy']}:D1"),
                                     watch["level"], float(row["atr_20"]), [], regime)
            return {**base, "status": "WATCH", "card": card, "grade": "WATCH", "score": None}
        card = coach.signal_card(symbol, "WAIT", None, 0, "", None, None, None, None, None, None,
                                 None, None, [], regime)
        return {**base, "status": "WAIT", "card": card, "grade": "", "score": None}

    def _fired_card(self, base: dict[str, Any], symbol: str, strat: str, sig: pd.Series,
                    f: pd.DataFrame, md: MarketData, acct: Account, params: StrategyParams,
                    partial_at_r: float, digits: int, spread: float, vstat: ValidatorStatus,
                    cons: Consensus, hist: dict[str, dict[str, Any]], regime: str,
                    session: Any, now: datetime, persist: bool) -> dict[str, Any]:
        row = f.iloc[-1]
        d = int(sig["direction"])
        stop = round(float(sig["stop"]), digits)
        entry_ref = round(float(row["close"]), digits)
        lv = render_levels(strat, d, row, entry_ref, stop, partial_at_r, spread, digits)
        stats_ = hist.get(f"{strat}:D1")
        kf = kelly_fraction(stats_["win_rate"], stats_["payoff"]) if stats_ else None
        positions = self.open_positions(acct)
        ctx = RiskContext(
            risk_pct=acct.risk_pct, open_positions=positions,
            equity_peak=float(self.store.get_setting("equity_peak", acct.balance)),
            equity_now=acct.balance, breaker_reviewed=not self._breaker_needs_review(acct),
            data_confirmed=md.signal_allowed, pair_validated=vstat.can_vote,
            strategy_health=str(self.store.get_setting(f"health:{strat}", "OK")),
            kelly_fraction=kf, now=None)
        limits = Limits.from_config(self.cfg.section("risk"), self.cfg.section("strategies"),
                                    self.cfg.section("costs"))
        blocks = hard_rule_violations(symbol, stop, ctx, limits)
        same = sum(1 for p in positions if p.symbol[:3] in symbol or p.symbol[3:] in symbol)
        rr = lv["rr"] if strat == "mean_reversion" else (
            stats_["payoff"] if stats_ and math.isfinite(stats_["payoff"]) else 2.0)
        vals = factor_values(row, d, float(sig["confidence"]), rr, same, self.loss_streak(),
                             int(weekly_bias(md.candles).iloc[-1]), candle_quality(row, d),
                             None, None)
        sc = score_setup(vals, self.cfg.scoring_factors(), blocks, str(sig["grade"]))
        status = ("BUY" if d == 1 else "SELL") if not sc.blocked and sc.grade in (
            "A", "B", "PROBE") else ("WATCH" if sc.grade == "WATCH" else "WAIT")
        locked_reason = ""
        if not vstat.can_vote:
            status, locked_reason = "LOCKED", vstat.reason
        size: dict[str, Any] | None = None
        if status in ("BUY", "SELL"):
            try:
                ps = position_size(acct.balance, acct.risk_pct, entry_ref, stop,
                                   self.cfg.symbol(symbol), acct.currency,
                                   fx_converter(self.svc, self.store), sc.size_multiplier)
                size = {"units": round(ps.units), "lots": round(ps.lots, 2),
                        "amount_at_risk": round(ps.amount_at_risk, 2),
                        "currency": ps.account_ccy, "risk_pct": round(ps.risk_pct, 3),
                        "stop_pips": round(ps.stop_distance_pips, 1)}
            except (KeyError, ValueError) as exc:
                status, blocks = "WAIT", [*blocks, f"Cannot size the position: {exc}"]
        op = order_plan(strat, d, entry_ref, float(row["atr_20"]),
                        float(row.get(f"dch_{params.lookbacks[0]}" if d == 1 else
                                      f"dcl_{params.lookbacks[0]}", entry_ref)),
                        float(row["ema_20"]), session or now.date(), digits)
        score_d = {"score": sc.score, "grade": sc.grade, "size_multiplier": sc.size_multiplier,
                   "factors": sc.factors}
        all_blocks = blocks + ([locked_reason] if locked_reason else [])
        card = coach.signal_card(symbol, status, strat, d, str(sig["reason"]), entry_ref, stop,
                                 size, op.__dict__, score_d, stats_, None, float(row["atr_20"]),
                                 all_blocks, regime, partial_at_r, levels=lv)
        card["levels"] = lv
        card["setup"] = render_setup(
            strat, d, f, lv, params, partial_at_r, digits, vstat.can_vote,
            float(size["amount_at_risk"]) if size else None, acct.currency)
        trace = self._trace(strat, sig, f, md, params, vstat, cons, lv)
        card["trace"] = trace
        card["consensus"] = {"text": cons.text, "voters": cons.voters, "excluded": cons.excluded}
        sid = None
        if persist and session is not None:
            sid = self._persist(symbol, strat, str(session), status, d, sc, entry_ref, stop, op,
                                all_blocks, trace)
        return {**base, "status": "WAIT" if status == "LOCKED" else status, "card": card,
                "grade": sc.grade, "score": sc.score, "strategy": strat,
                "locked": status == "LOCKED", "signal_id": sid}

    def ict_params(self, variant: str = "ict") -> ICTParams:
        rep = load_report() or {}
        chosen: dict[str, Any] = next((p.get("selected_params", {}) for p in rep.get("pairs", [])
                                       if p.get("strategy") == variant
                                       and p.get("timeframe") == "D1"), {})
        return ICTParams.from_config(self.cfg.section("strategies").get(variant, {}), chosen)

    def ict_active(self, df: pd.DataFrame, variant: str = "ict") -> Setup | None:
        """The latest ICT setup whose limit order is still pending (not filled/expired/hit)."""
        if len(df) < 60:
            return None
        setups = find_setups(df, self.ict_params(variant))
        if not setups:
            return None
        st = setups[-1]
        last = len(df) - 1
        if st.expiry_bar < last:
            return None
        lo, h = df["low"].to_numpy(float), df["high"].to_numpy(float)
        for t in range(st.mss_bar + 1, last + 1):
            touched = lo[t] <= st.entry if st.direction == 1 else h[t] >= st.entry
            if touched:
                return None
        return st

    def _ict_card(self, base: dict[str, Any], symbol: str, st: Setup, f: pd.DataFrame,
                  md: MarketData, acct: Account, digits: int, vstat: ValidatorStatus,
                  cons: Consensus, hist: dict[str, dict[str, Any]], regime: str,
                  session: Any, now: datetime, persist: bool,
                  variant: str = "ict") -> dict[str, Any]:
        d = st.direction
        reason = (f"Liquidity sweep of {round(st.swept_level, digits)} then market structure "
                  f"shift at {round(st.mss_level, digits)}; fair value gap "
                  f"{round(st.fvg_low, digits)}-{round(st.fvg_high, digits)} in "
                  f"{'discount' if d == 1 else 'premium'}.")
        extra = {"ict": {"sweep_date": str(f.index[st.sweep_bar].date()),
                         "mss_date": str(f.index[st.mss_bar].date()),
                         "fvg": [round(st.fvg_low, 6), round(st.fvg_high, 6)],
                         "quality": st.quality}}
        return self._limit_card(
            base, symbol, variant, d, st.entry, st.stop, st.target, st.expiry_bar,
            "FVG midpoint", reason, lambda risk_amt: render_ict(
                st, f, digits, vstat.can_vote, risk_amt, acct.currency),
            asdict(self.ict_params(variant)), extra, f, md, acct, digits, vstat, cons, hist,
            regime, session, now, persist)

    def sd_params(self) -> SDParams:
        rep = load_report() or {}
        chosen: dict[str, Any] = next((p.get("selected_params", {}) for p in rep.get("pairs", [])
                                       if p.get("strategy") == "supply_demand"
                                       and p.get("timeframe") == "D1"), {})
        return SDParams.from_config(self.cfg.section("strategies").get("supply_demand", {}),
                                    chosen)

    def _sd_card(self, base: dict[str, Any], symbol: str, zone: Zone, od: LimitOrder,
                 f: pd.DataFrame, md: MarketData, acct: Account, digits: int,
                 vstat: ValidatorStatus, cons: Consensus, hist: dict[str, dict[str, Any]],
                 regime: str, session: Any, now: datetime, persist: bool) -> dict[str, Any]:
        kind = "demand" if od.direction == 1 else "supply"
        reason = (f"No ICT setup, so supply & demand: fresh {kind} zone "
                  f"{round(zone.low, digits)}-{round(zone.high, digits)} left with a "
                  f"{zone.strength:.1f} ATR leg-out, with the EMA50/200 trend; waiting for the "
                  "first return.")
        extra = {"supply_demand": {"zone": [round(zone.low, 6), round(zone.high, 6)],
                                   "created": str(f.index[zone.created_bar].date()),
                                   "strength_atr": zone.strength}}
        return self._limit_card(
            base, symbol, "supply_demand", od.direction, od.entry, od.stop, od.target,
            od.expiry_bar, f"{kind} zone edge", reason, lambda risk_amt: render_sd(
                zone, od, f, digits, vstat.can_vote, risk_amt, acct.currency),
            asdict(self.sd_params()), extra, f, md, acct, digits, vstat, cons, hist, regime,
            session, now, persist)

    def _limit_card(self, base: dict[str, Any], symbol: str, variant: str, d: int,
                    entry_px: float, stop_px: float, target_px: float, expiry_bar: int,
                    entry_label: str, reason: str, render: Any, thresholds: dict[str, Any],
                    extra: dict[str, Any], f: pd.DataFrame, md: MarketData, acct: Account,
                    digits: int, vstat: ValidatorStatus, cons: Consensus,
                    hist: dict[str, dict[str, Any]], regime: str, session: Any,
                    now: datetime, persist: bool) -> dict[str, Any]:
        entry, stop, target = (round(entry_px, digits), round(stop_px, digits),
                               round(target_px, digits))
        risk = abs(entry - stop)
        lv = {"entry": entry, "stop": stop, "target": target, "breakeven": entry,
              "rr": round(abs(target - entry) / risk, 2) if risk else 0.0,
              "risk": round(risk, digits)}
        stats_ = hist.get(f"{variant}:D1")
        positions = self.open_positions(acct)
        ctx = RiskContext(
            risk_pct=acct.risk_pct, open_positions=positions,
            equity_peak=float(self.store.get_setting("equity_peak", acct.balance)),
            equity_now=acct.balance, breaker_reviewed=not self._breaker_needs_review(acct),
            data_confirmed=md.signal_allowed, pair_validated=vstat.can_vote,
            strategy_health=str(self.store.get_setting(f"health:{variant}", "OK")),
            kelly_fraction=kelly_fraction(stats_["win_rate"], stats_["payoff"])
            if stats_ else None, now=None)
        limits = Limits.from_config(self.cfg.section("risk"), self.cfg.section("strategies"),
                                    self.cfg.section("costs"))
        blocks = hard_rule_violations(symbol, stop, ctx, limits)
        status = ("BUY" if d == 1 else "SELL") if not blocks else "WAIT"
        locked_reason = ""
        if not vstat.can_vote:
            status, locked_reason = "LOCKED", vstat.reason
        size: dict[str, Any] | None = None
        if status in ("BUY", "SELL"):
            try:
                ps = position_size(acct.balance, acct.risk_pct, entry, stop,
                                   self.cfg.symbol(symbol), acct.currency,
                                   fx_converter(self.svc, self.store))
                size = {"units": round(ps.units), "lots": round(ps.lots, 2),
                        "amount_at_risk": round(ps.amount_at_risk, 2),
                        "currency": ps.account_ccy, "risk_pct": round(ps.risk_pct, 3),
                        "stop_pips": round(ps.stop_distance_pips, 1)}
            except (KeyError, ValueError) as exc:
                status, blocks = "WAIT", [*blocks, f"Cannot size the position: {exc}"]
        bars_left = expiry_bar - (len(f) - 1)
        order = {"order_type": "LIMIT", "price": entry,
                 "expiry": f"{bars_left} more session(s)",
                 "text": f"{'BUY' if d == 1 else 'SELL'} LIMIT at {entry} ({entry_label}); "
                         f"cancel after {bars_left} more session(s) or if price trades "
                         f"through {stop} first."}
        all_blocks = blocks + ([locked_reason] if locked_reason else [])
        card = coach.signal_card(symbol, status, variant, d, reason, entry, stop, size, order,
                                 None, stats_, None, float(f["atr_20"].iloc[-1]), all_blocks,
                                 regime, 1.0, levels=lv)
        card["setup_name"] = coach.STRATEGY_TEXT.get(variant, variant)
        card["levels"] = lv
        card["setup"] = render(float(size["amount_at_risk"]) if size else None)
        trace = {"strategy": variant, "rule_version": vstat.rule_version,
                 "rule_hash": vstat.live_hash, "validated_hash": vstat.validated_hash,
                 "validator_state": vstat.state, "validator_reason": vstat.reason,
                 "data_version": md.data_version, "data_source": md.source,
                 "bar_date": str(f.index[-1].date()),
                 "bars": [{"date": str(ix.date()), **{c: round(float(f.loc[ix, c]), 6)
                                                      for c in ("open", "high", "low", "close")}}
                          for ix in f.index[-5:]],
                 "indicators": {"atr_20": round(float(f["atr_20"].iloc[-1]), 6)},
                 "thresholds": thresholds, "levels": lv, "grade": "A",
                 "confidence": 1.0, "reason": reason, "consensus": cons.text, **extra}
        card["trace"] = trace
        card["consensus"] = {"text": cons.text, "voters": cons.voters, "excluded": cons.excluded}
        sid = None
        if persist and session is not None:
            from types import SimpleNamespace

            sc = SimpleNamespace(grade="A", score=0.0, size_multiplier=1.0, factors={})
            op = SimpleNamespace(order_type="LIMIT", price=entry, expiry=order["expiry"])
            sid = self._persist(symbol, variant, str(session), status, d, sc, entry, stop, op,
                                all_blocks, trace)
        return {**base, "status": "WAIT" if status == "LOCKED" else status, "card": card,
                "grade": "A", "score": None, "strategy": variant,
                "locked": status == "LOCKED", "signal_id": sid}

    def _trace(self, strat: str, sig: pd.Series, f: pd.DataFrame, md: MarketData,
               params: StrategyParams, vstat: ValidatorStatus, cons: Consensus,
               lv: dict[str, float]) -> dict[str, Any]:
        """Exact evidence for 'Why this signal' (bars, indicators, thresholds, versions)."""
        row = f.iloc[-1]
        keys = [k for k in ("close", "dch_20", "dcl_20", "dch_55", "dcl_55", "dch_100",
                            "dcl_100", "ema_5", "ema_20", "ema_100", "ema_200", "atr_14",
                            "atr_20", "adx_14", "rsi_2", "atr_pct") if k in row]
        bars = [{"date": str(ix.date()), **{c: round(float(f.loc[ix, c]), 6)
                                            for c in ("open", "high", "low", "close")}}
                for ix in f.index[-5:]]
        thresholds = {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in asdict(params).items() if k != "extra"}
        return {"strategy": strat, "rule_version": vstat.rule_version,
                "rule_hash": vstat.live_hash, "validated_hash": vstat.validated_hash,
                "validator_state": vstat.state, "validator_reason": vstat.reason,
                "data_version": md.data_version, "data_source": md.source,
                "bar_date": str(f.index[-1].date()), "bars": bars,
                "indicators": {k: round(float(row[k]), 6) for k in keys
                               if np.isfinite(float(row[k]))},
                "thresholds": thresholds, "levels": lv, "grade": str(sig["grade"]),
                "confidence": round(float(sig["confidence"]), 4), "reason": str(sig["reason"]),
                "consensus": cons.text}

    def _breaker_needs_review(self, acct: Account) -> bool:
        peak = float(self.store.get_setting("equity_peak", acct.balance))
        dd = (peak - acct.balance) / peak * 100 if peak > 0 else 0
        if dd > float(self.cfg.section("risk").get("drawdown_breaker_pct", 15)):
            flag = self.store.get_setting("breaker_reviewed_at")
            return flag is None
        return False

    def _persist(self, symbol: str, strat: str, session: str, status: str, d: int,
                 sc: Any, entry: float, stop: float, op: Any, reasons: list[str],
                 trace: dict[str, Any]) -> int:
        for s in self.store.signals(symbol, 20):
            if s.session_date == session and s.strategy == strat:
                return s.id
        row = self.store.add(SignalRow(session_date=session, symbol=symbol, strategy=strat,
                                 status=status, direction=d, grade=sc.grade, score=sc.score,
                                 size_mult=sc.size_multiplier, entry_ref=entry, stop=stop,
                                 order_type=op.order_type, order_price=op.price,
                                 expiry=op.expiry, reasons=json.dumps(reasons),
                                 factors=json.dumps(sc.factors),
                                 rule_version=trace["rule_version"],
                                 rule_hash=trace["rule_hash"],
                                 trace=canonical_json(trace)))
        return int(row.id)

    def watchlist(self, refresh: bool = False) -> list[dict[str, Any]]:
        out = []
        for sym in self.cfg.symbols:
            ev = self.evaluate(sym, refresh=refresh)
            c = pd.Series(ev.get("last_closes", []), dtype=float)
            chg = float((c.iloc[-1] / c.iloc[-2] - 1) * 100) if len(c) > 1 else None
            out.append({"symbol": sym, "status": ev["status"], "grade": ev.get("grade", ""),
                        "score": ev.get("score"), "last": float(c.iloc[-1]) if len(c) else None,
                        "change_pct": round(chg, 3) if chg is not None else None,
                        "data": ev["data"]["status"], "locked": ev.get("locked", False)})
        return out
