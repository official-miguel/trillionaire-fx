"""Strategy proof phase: the validation gate run on real, confirmed data.

For every strategy-timeframe pair:

1. Backtest every combination of the strategy's PRE-DECLARED grid on the whole
   universe (2005 → today), with realistic costs and the live portfolio caps.
   Every combination is recorded as a trial.
2. Choose parameters on data BEFORE the untouched 3-year holdout only.
3. Overfitting: PBO via CSCV with embargo (< 0.05) and the Deflated Sharpe
   probability (>= 0.90) using ALL trials counted so far.
4. Robustness: share of grid neighbours profitable after costs (>= 70%), cost stress
   (must stay profitable at 1.5x costs; 2x reported), stress periods, walk-forward
   with stitched out-of-sample years, block-bootstrap ranges, Monte Carlo drawdowns.
5. Holdout: the chosen parameters are run ONCE on the last 3 years.
6. Verdict PASS / FAIL with plain reasons, written to the validation registry. FAIL
   means locked (no signals). If nothing passes, the report says so — no re-tuning.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import (
    Costs,
    Management,
    Trade,
    apply_portfolio_caps,
    daily_r_frame,
    run_symbol,
)
from tfx.config import PROJECT_ROOT, AppConfig
from tfx.core_types import Timeframe
from tfx.data.cache import CandleCache
from tfx.logging_setup import fields, get_logger
from tfx.strategies.features import build_features
from tfx.strategies.momentum import PairCosts, ts_momentum, xs_momentum
from tfx.strategies.rules import StrategyParams
from tfx.validation import stats
from tfx.validation.registry import ValidationRegistry
from tfx.validation.validators import validators
from tfx.versioning import RULE_VERSIONS, all_rule_hashes, rule_hash

log = get_logger(__name__)

PROOF_STRATEGIES = ("trend", "pullback", "mean_reversion")
MOMENTUM_STRATEGIES = ("xs_momentum", "ts_momentum")
CUSTOM_STRATEGIES = ("ict", "ict_flex", "supply_demand")
EXCLUDED = {"XAUUSD": "excluded pending the gold data decision (Yahoo futures vs MT5 spot "
                      "agree on only ~18% of days)"}
REPORT_PATH = PROJECT_ROOT / "var" / "validation_report.json"
MGMT_KEYS = {"partial_at_r", "partial_fraction", "pyramid_at_r", "max_pyramid_adds"}


@dataclass
class PairResult:
    strategy: str
    timeframe: str
    verdict: str
    reasons: list[str]
    selected_params: dict[str, Any] = field(default_factory=dict)
    trials: int = 0
    total_trials_counted: int = 0
    pbo: float | None = None
    dsr: float | None = None
    neighbours_profitable: float | None = None
    pre_holdout: dict[str, Any] = field(default_factory=dict)
    holdout: dict[str, Any] = field(default_factory=dict)
    full: dict[str, Any] = field(default_factory=dict)
    cost_stress: dict[str, float] = field(default_factory=dict)
    stress_periods: dict[str, float] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    monte_carlo: dict[str, Any] = field(default_factory=dict)
    per_year: dict[int, float] = field(default_factory=dict)
    by_symbol: dict[str, float] = field(default_factory=dict)
    exit_reasons: dict[str, int] = field(default_factory=dict)
    edge_confidence: float = 0.0
    edge_derivation: str = ""
    expectancy_ci: tuple[float, float] | None = None
    rule_version: str = ""
    rule_hash: str = ""


def costs_for(cfg: AppConfig, symbol: str, multiplier: float = 1.0) -> Costs:
    sym = cfg.symbol(symbol)
    prof = cfg.section("costs").get("profiles", {}).get(cfg.account.broker_profile, {})
    pip = sym.pip
    spread = sym.typical_spread_pips * pip
    commission = float(prof.get("commission_pips", 0.7)) * pip * (10 if symbol == "XAUUSD"
                                                                   else 1)
    slippage = float(prof.get("slippage_pips", 0.2)) * pip
    swap = float(prof.get("swap_spread_fraction", 0.1)) * spread
    return Costs(spread, commission, slippage, swap, multiplier)


def grid_for(cfg: AppConfig, strategy: str) -> list[dict[str, Any]]:
    g = cfg.section("parameter_grid").get(strategy, {})
    keys = sorted(g)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(
        *(g[k] for k in keys))] or [{}]


def _split(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return ({k: v for k, v in params.items() if k not in MGMT_KEYS},
            {k: v for k, v in params.items() if k in MGMT_KEYS})


class ProofRunner:
    def __init__(self, cfg: AppConfig, cache: CandleCache, registry: ValidationRegistry,
                 clock: Any = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self.registry = registry
        self.clock = clock or (lambda: datetime.now(UTC))
        self.vcfg = cfg.section("validation")
        self.risk_pct = cfg.account.risk_per_trade_pct / 100
        self.run_id = f"proof-{self.clock():%Y%m%dT%H%M%S}"
        self._data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    # ------------------------------------------------------------------ data
    def universe(self) -> list[str]:
        return [s for s in self.cfg.symbols if s not in EXCLUDED]

    def data(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if symbol not in self._data:
            df = self.cache.load(symbol)
            self._data[symbol] = (df, build_features(df))
        return self._data[symbol]

    # ------------------------------------------------------------- backtests
    def run_combo(self, strategy: str, params: dict[str, Any], cost_mult: float = 1.0,
                  symbols: list[str] | None = None) -> tuple[pd.Series, list[Trade]]:
        sp, mp = _split(params)
        p = StrategyParams.from_config(self.cfg.section("strategies"), sp)
        m = Management.from_config(self.cfg.section("strategies").get("management", {}), mp)
        start = pd.Timestamp(self.vcfg.get("backtest_start", "2005-01-01"))
        trades: list[Trade] = []
        index: pd.DatetimeIndex | None = None
        for sym in symbols or self.universe():
            df, f = self.data(sym)
            if df.empty:
                continue
            trades += run_symbol(sym, strategy, df, f, p, m,
                                 costs_for(self.cfg, sym, cost_mult), start=start)
            sidx = f.index[f.index >= start]
            index = sidx if index is None else index.union(sidx)
        fcfg = self.cfg.section("strategies").get("filters", {})
        caps = {"mean_reversion": int(self.cfg.section("strategies").get(
            "mean_reversion", {}).get("max_open", 2))}
        kept = apply_portfolio_caps(trades, self.cfg.section("risk").get(
            "max_open_risk_pct", 3.0) / 100, self.risk_pct,
            int(fcfg.get("max_trades_per_currency", 2)), caps)
        assert index is not None
        returns = daily_r_frame(kept, index) * self.risk_pct
        return returns, kept

    def run_momentum(self, strategy: str, params: dict[str, Any], cost_mult: float = 1.0
                     ) -> tuple[pd.Series, list[Trade]]:
        start = pd.Timestamp(self.vcfg.get("backtest_start", "2005-01-01"))
        data = {s: self.data(s)[0] for s in self.universe() if not self.data(s)[0].empty}
        costs = {}
        for s, df in data.items():
            c = costs_for(self.cfg, s)
            px = float(df["close"].iloc[-1])
            costs[s] = PairCosts((c.spread + c.commission + 2 * c.slippage) / px,
                                 c.swap_per_night / px)
        if strategy == "xs_momentum":
            return xs_momentum(data, costs, int(params["formation_months"]),
                               int(params["n_each_side"]), start, cost_mult)
        return ts_momentum(data, costs, int(params["lookback_weeks"]), start, cost_mult)

    def run_ict(self, params: dict[str, Any], cost_mult: float = 1.0, variant: str = "ict"
                ) -> tuple[pd.Series, list[Trade]]:
        from tfx.strategies import ict, supply_demand

        strat = self.cfg.section("strategies").get(variant, {})
        if variant == "supply_demand":
            sp = supply_demand.SDParams.from_config(strat, params)

            def simulate(sym: str, df: pd.DataFrame, _p: Any, costs: Costs,
                         start: pd.Timestamp) -> list[Trade]:
                return supply_demand.simulate(sym, df, sp, costs, start)
            p: Any = sp
        else:
            p = ict.ICTParams.from_config(strat, params)
            simulate = ict.simulate  # type: ignore[assignment]
        start = pd.Timestamp(self.vcfg.get("backtest_start", "2005-01-01"))
        trades: list[Trade] = []
        index: pd.DatetimeIndex | None = None
        for sym in self.universe():
            df, f = self.data(sym)
            if df.empty:
                continue
            trades += simulate(sym, df, p, costs_for(self.cfg, sym, cost_mult), start)
            sidx = f.index[f.index >= start]
            index = sidx if index is None else index.union(sidx)
        fcfg = self.cfg.section("strategies").get("filters", {})
        kept = apply_portfolio_caps(trades, self.cfg.section("risk").get(
            "max_open_risk_pct", 3.0) / 100, self.risk_pct,
            int(fcfg.get("max_trades_per_currency", 2)))
        assert index is not None
        return daily_r_frame(kept, index) * self.risk_pct, kept

    def run_any(self, strategy: str, params: dict[str, Any], cost_mult: float = 1.0
                ) -> tuple[pd.Series, list[Trade]]:
        if strategy in CUSTOM_STRATEGIES:
            return self.run_ict(params, cost_mult, strategy)
        if strategy in MOMENTUM_STRATEGIES:
            return self.run_momentum(strategy, params, cost_mult)
        return self.run_combo(strategy, params, cost_mult)

    # ------------------------------------------------------------- the gate
    def prove(self, strategy: str) -> PairResult:
        grid = grid_for(self.cfg, strategy)
        now = self.clock()
        tf = Timeframe.W1 if strategy in MOMENTUM_STRATEGIES else Timeframe.D1
        results: list[tuple[dict[str, Any], pd.Series, list[Trade]]] = []
        for params in grid:
            r, tr = self.run_any(strategy, params)
            self.registry.record_trial(self.run_id, strategy, tf, params, now)
            results.append((params, r, tr))
        index = results[0][1].index
        holdout_start = index[-1] - pd.DateOffset(years=int(self.vcfg.get("holdout_years", 3)))
        pre = index < holdout_start
        matrix = np.column_stack([r.to_numpy()[pre] for _, r, _ in results])
        pre_sr = [stats.sharpe(matrix[:, k]) for k in range(matrix.shape[1])]
        best = int(np.argmax(pre_sr))
        params, r_best, tr_best = results[best]
        res = PairResult(strategy, tf.value, "FAIL", [], selected_params=params,
                         trials=len(grid))
        res.total_trials_counted = self.registry.trial_count()

        pre_trades = [t for t in tr_best if t.entry_date < holdout_start]
        ho_trades = [t for t in tr_best if t.entry_date >= holdout_start]
        months_pre = pre.sum() / 21
        months_ho = (~pre).sum() / 21
        r_pre, r_ho = r_best[pre], r_best[~pre]
        res.pre_holdout = self._summary(r_pre, pre_trades, months_pre)
        res.holdout = self._summary(r_ho, ho_trades, months_ho)
        res.full = self._summary(r_best, tr_best, len(index) / 21)

        pbo = stats.pbo_cscv(matrix, int(self.vcfg.get("cpcv_groups", 10)),
                             int(self.vcfg.get("embargo_days", 10)))
        res.pbo = pbo.pbo
        trial_srs = [stats.sharpe(matrix[:, k], annualise=False)
                     for k in range(matrix.shape[1])]
        res.dsr = stats.deflated_sharpe(matrix[:, best], trial_srs,
                                        max(res.total_trials_counted, len(grid)))
        res.neighbours_profitable = float(np.mean([matrix[:, k].sum() > 0
                                                   for k in range(matrix.shape[1])]))
        for m in (1.5, 2.0):
            rr, _ = self.run_any(strategy, params, cost_mult=m)
            res.cost_stress[f"{m}x"] = stats.sharpe(rr)
        res.cost_stress["1.0x"] = stats.sharpe(r_best)
        for name, (a, b) in self.vcfg.get("stress_periods", {}).items():
            seg = r_best[(r_best.index >= pd.Timestamp(a)) & (r_best.index <= pd.Timestamp(b))]
            res.stress_periods[name] = float((1 + seg).prod() - 1)
        res.walk_forward = self._walk_forward(results, holdout_start)
        res.bootstrap = {k: list(v) for k, v in stats.block_bootstrap(
            r_best, int(self.vcfg.get("bootstrap_samples", 2000)) // 4,
            int(self.vcfg.get("bootstrap_block_days", 20)),
            self.cfg.random_seed).items()}
        rs = [t.r_net for t in tr_best]
        res.monte_carlo = stats.shuffle_drawdowns(rs, self.risk_pct, 1000, self.cfg.random_seed)
        res.expectancy_ci = self._expectancy_ci(rs)
        res.per_year = stats.per_year(r_best)
        res.by_symbol = {s: float(np.sum([t.r_net * t.size for t in tr_best if t.symbol == s]))
                         for s in self.universe()}
        res.exit_reasons = dict(pd.Series([t.exit_reason for t in tr_best]).value_counts())

        verdict, reasons = validators(self.cfg)[strategy].evaluate(asdict(res))
        res.reasons = reasons
        res.verdict = verdict
        res.rule_version = RULE_VERSIONS[strategy]
        res.rule_hash = rule_hash(strategy, self.cfg)
        ho_ok = res.holdout["sharpe"] > 0
        res.edge_confidence = round(100 * (res.dsr or 0) * (1 - (res.pbo or 1))
                                    * (1.0 if ho_ok else 0.5), 1)
        res.edge_derivation = (f"100 x DSR {res.dsr:.2f} x (1 - PBO {res.pbo:.2f}) x "
                               f"holdout factor {1.0 if ho_ok else 0.5} = "
                               f"{res.edge_confidence}")
        self.registry.record_result(strategy, tf, res.verdict,  # type: ignore[arg-type]
                                    self.run_id, now, reasons or None,
                                    {"edge_confidence": res.edge_confidence,
                                     "params": params, "rule_hash": res.rule_hash,
                                     "rule_version": res.rule_version})
        log.info("proof result", extra=fields(strategy=strategy, verdict=res.verdict,
                                              reasons=reasons))
        return res

    def _summary(self, r: pd.Series, trades: list[Trade], months: float) -> dict[str, Any]:
        m: dict[str, Any] = dict(stats.trade_metrics([t.r_net for t in trades],
                                                    [t.r_cost / t.size for t in trades],
                                                    months))
        m.update({"sharpe": stats.sharpe(r), "cagr": stats.cagr(r),
                  "max_drawdown": stats.max_drawdown(r),
                  "start": str(r.index[0].date()) if len(r) else "",
                  "end": str(r.index[-1].date()) if len(r) else ""})
        return m

    def _walk_forward(self, results: list[tuple[dict[str, Any], pd.Series, list[Trade]]],
                      holdout_start: pd.Timestamp) -> dict[str, Any]:
        """Re-select yearly on past data only; stitch the following year's returns."""
        idx = results[0][1].index
        first = idx[0].year + 5
        pieces = []
        chosen = {}
        for year in range(first, holdout_start.year + 1):
            cut = pd.Timestamp(f"{year}-01-01")
            end = min(pd.Timestamp(f"{year + 1}-01-01"), holdout_start)
            if cut >= end:
                break
            past = [stats.sharpe(r[r.index < cut]) for _, r, _ in results]
            k = int(np.argmax(past))
            seg = results[k][1]
            pieces.append(seg[(seg.index >= cut) & (seg.index < end)])
            chosen[year] = results[k][0]
        if not pieces:
            return {}
        stitched = pd.concat(pieces)
        return {"oos_sharpe": stats.sharpe(stitched), "oos_cagr": stats.cagr(stitched),
                "oos_max_drawdown": stats.max_drawdown(stitched), "years": len(pieces),
                "selections_changed": len({json.dumps(v, sort_keys=True)
                                           for v in chosen.values()})}

    def _expectancy_ci(self, rs: list[float]) -> tuple[float, float] | None:
        if len(rs) < 20:
            return None
        rng = np.random.default_rng(self.cfg.random_seed)
        arr = np.asarray(rs)
        means = [rng.choice(arr, len(arr)).mean() for _ in range(1000)]
        return float(np.percentile(means, 5)), float(np.percentile(means, 95))

    # ---------------------------------------------------- non-daily candidates
    def lock_intraday(self) -> list[PairResult]:
        """H4/H1 candidates need confirmed deep intraday history; without it they FAIL."""
        out = []
        now = self.clock()
        for name, tfs in self.cfg.strategy_timeframes().items():
            if name not in (*PROOF_STRATEGIES, *CUSTOM_STRATEGIES):
                continue
            for tf in tfs.candidates:
                if tf is Timeframe.D1:
                    continue
                reason = (f"{tf} needs >= 8 years of confirmed intraday history; the broker's "
                          "MT5 hourly history starts mid-2019 and Yahoo keeps ~2 years")
                self.registry.record_result(name, tf, "FAIL", self.run_id, now, [reason])
                out.append(PairResult(name, tf.value, "FAIL", [reason]))
        return out

    def run_all(self, only: list[str] | None = None) -> dict[str, Any]:
        """Prove every enabled strategy, or only ``only`` (earlier results are kept from
        the existing report, so already-tested strategies are not re-run or re-counted)."""
        pairs: list[PairResult] = []
        enabled = {k for k, v in self.cfg.section("strategies").items()
                   if isinstance(v, dict) and v.get("enabled", False)}
        kept: list[dict[str, Any]] = []
        if only:
            old = load_report() or {}
            kept = [p for p in old.get("pairs", []) if p.get("strategy") not in only]
        for s in (*PROOF_STRATEGIES, *MOMENTUM_STRATEGIES, *CUSTOM_STRATEGIES):
            if s in enabled and (not only or s in only):
                pairs.append(self.prove(s))
        pairs += [p for p in self.lock_intraday() if not only or p.strategy in only]
        if kept:
            known = {f.name for f in PairResult.__dataclass_fields__.values()}
            pairs = [PairResult(**{k: v for k, v in p.items() if k in known})
                     for p in kept] + pairs
        passing = [p for p in pairs if p.verdict == "PASS"]
        combined = self._combined(passing)
        report = {
            "run_id": self.run_id, "generated_at": self.clock().isoformat(),
            "universe": self.universe(), "excluded": EXCLUDED,
            "risk_per_trade_pct": self.cfg.account.risk_per_trade_pct,
            "holdout_years": self.vcfg.get("holdout_years", 3),
            "pairs": [asdict(p) for p in pairs], "combined": combined,
            "rule_hashes": all_rule_hashes(self.cfg), "rule_versions": RULE_VERSIONS,
            "disabled_modules": {k: "disabled by default; not validated" for k in
                                 ("carry", "xs_momentum", "ml_scorer")},
            "summary": (f"{len(passing)} of {len([p for p in pairs if p.full])} "
                        "strategies passed. " + (
                            "Nothing passed: no live signals will be shown. The strategies "
                            "are NOT re-tuned to force a pass." if not passing else
                            "Only passing strategy-timeframe pairs may produce signals.")),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return report

    def _combined(self, passing: list[PairResult]) -> dict[str, Any]:
        if not passing:
            return {"note": "no passing strategies to combine"}
        series = {p.strategy: self.run_any(p.strategy, p.selected_params)[0] for p in passing}
        df = pd.DataFrame(series).fillna(0.0)
        holdout_start = df.index[-1] - pd.DateOffset(years=3)
        vol = df[df.index < holdout_start].std().replace(0, np.nan)
        w = (1 / vol) / (1 / vol).sum()
        comb = (df * w).sum(axis=1)
        return {"weights": {k: float(v) for k, v in w.items()},
                "sharpe": stats.sharpe(comb), "cagr": stats.cagr(comb),
                "max_drawdown": stats.max_drawdown(comb),
                "holdout_sharpe": stats.sharpe(comb[comb.index >= holdout_start])}


def load_report(path: Path | None = None) -> dict[str, Any] | None:
    path = path or REPORT_PATH       # looked up at call time (tests redirect it)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
