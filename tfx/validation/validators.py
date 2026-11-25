"""Per-strategy validators: fixed pass/fail criteria and a live status.

Each strategy has its own :class:`StrategyValidator`. Its criteria come from the
validation gate and are fixed in code + ``config.yaml`` (never tuned per result).
:meth:`StrategyValidator.status` is what the consensus engine and the signal
pipeline read: a strategy may vote ONLY with a current PASS whose rule hash equals
the live rule hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tfx.config import AppConfig
from tfx.core_types import Timeframe
from tfx.validation.registry import ValidationRegistry
from tfx.versioning import RULE_VERSIONS, rule_hash


@dataclass(frozen=True)
class Criteria:
    pbo_max: float = 0.05
    dsr_min: float = 0.90
    neighbours_min: float = 0.70
    min_trades: int = 100
    cost_stress_key: str = "1.5x"

    @classmethod
    def from_config(cls, v: dict[str, Any]) -> Criteria:
        return cls(pbo_max=float(v.get("pbo_max", 0.05)),
                   dsr_min=float(v.get("dsr_min_prob", 0.90)),
                   neighbours_min=float(v.get("neighbour_profitable_min", 0.70)))


@dataclass(frozen=True)
class ValidatorStatus:
    strategy: str
    timeframe: str
    state: str           # PASS | FAIL | NOT VALIDATED | EXPIRED | HASH MISMATCH
    can_vote: bool
    reason: str
    rule_version: str
    live_hash: str
    validated_hash: str | None


class StrategyValidator:
    def __init__(self, strategy: str, criteria: Criteria) -> None:
        if strategy not in RULE_VERSIONS:
            raise KeyError(strategy)
        self.strategy = strategy
        self.criteria = criteria

    # ------------------------------------------------------- fixed criteria
    def evaluate(self, r: dict[str, Any]) -> tuple[str, list[str]]:
        """Verdict from a proof result (dict with pre_holdout/holdout/pbo/dsr/...)."""
        c = self.criteria
        reasons: list[str] = []
        pre, ho = r.get("pre_holdout") or {}, r.get("holdout") or {}
        if pre.get("trades", 0) < c.min_trades:
            reasons.append(f"too few trades before the holdout ({pre.get('trades', 0)})")
        if r.get("pbo") is None or r["pbo"] >= c.pbo_max:
            reasons.append(f"probability of backtest overfitting {r.get('pbo', 1.0):.2f} >= "
                           f"{c.pbo_max}")
        if r.get("dsr") is None or r["dsr"] < c.dsr_min:
            reasons.append(f"deflated Sharpe probability {r.get('dsr', 0.0):.2f} < {c.dsr_min} "
                           f"after {r.get('total_trials_counted', 0)} trials")
        if ho.get("sharpe", 0) <= 0 or ho.get("expectancy_r", 0) <= 0:
            reasons.append(f"not profitable on the untouched holdout (Sharpe "
                           f"{ho.get('sharpe', 0):.2f}, expectancy "
                           f"{ho.get('expectancy_r', 0):+.2f}R)")
        nb = r.get("neighbours_profitable")
        if nb is None or nb < c.neighbours_min:
            reasons.append(f"only {nb or 0:.0%} of grid settings profitable after costs "
                           f"(need {c.neighbours_min:.0%})")
        cs = (r.get("cost_stress") or {}).get(c.cost_stress_key)
        if cs is None or cs <= 0:
            reasons.append(f"not profitable at {c.cost_stress_key} costs (Sharpe "
                           f"{(cs or 0):.2f})")
        return ("FAIL" if reasons else "PASS"), reasons

    # ---------------------------------------------------------- live status
    def status(self, registry: ValidationRegistry, cfg: AppConfig, tf: Timeframe,
               now: datetime) -> ValidatorStatus:
        import json

        live = rule_hash(self.strategy, cfg)
        ver = RULE_VERSIONS[self.strategy]
        gate = registry.gate(self.strategy, tf, now)
        row = registry.latest(self.strategy, tf)
        validated = json.loads(row.metrics or "{}").get("rule_hash") if row else None
        if row is None:
            return ValidatorStatus(self.strategy, tf.value, "NOT VALIDATED", False, gate.reason,
                                   ver, live, None)
        if row.verdict != "PASS":
            return ValidatorStatus(self.strategy, tf.value, "FAIL", False, gate.reason, ver,
                                   live, validated)
        if validated != live:
            return ValidatorStatus(
                self.strategy, tf.value, "HASH MISMATCH", False,
                f"{self.strategy} on {tf}: rules or settings changed since validation "
                f"(validated {str(validated)[:12]}, live {live[:12]}). Re-run validation.",
                ver, live, validated)
        if not gate.allowed:
            return ValidatorStatus(self.strategy, tf.value, "EXPIRED", False, gate.reason, ver,
                                   live, validated)
        return ValidatorStatus(self.strategy, tf.value, "PASS", True, gate.reason, ver, live,
                               validated)


def validators(cfg: AppConfig) -> dict[str, StrategyValidator]:
    crit = Criteria.from_config(cfg.section("validation"))
    return {s: StrategyValidator(s, crit) for s in RULE_VERSIONS}
