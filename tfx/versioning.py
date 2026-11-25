"""Rule versions and rule hashes (reproducibility and drift control).

Every strategy has an explicit ``RULE_VERSIONS`` entry and a ``rule_hash``: SHA-256 of
(a) the source code of the functions that define its rules and exits and (b) the
config sections that parameterise it, serialised canonically. The hash is stored
with every validation result and every signal. If the live hash differs from the
hash that was validated, the strategy is BLOCKED until it is re-validated.

Changing a rule therefore needs two explicit steps: bump ``RULE_VERSIONS`` and
re-generate the golden files (see ``tests/golden``), then re-run validation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

from tfx.backtest import engine
from tfx.config import AppConfig
from tfx.strategies import features, ict, indicators, limit_sim, momentum, rules
from tfx.strategies import supply_demand as sd

RULE_VERSIONS: dict[str, str] = {
    "trend": "1.0.0",
    "pullback": "1.0.0",
    "mean_reversion": "1.0.0",
    "xs_momentum": "1.0.0",
    "ts_momentum": "1.0.0",
    "ict": "1.0.0",
    "ict_flex": "1.0.0",
    "supply_demand": "1.0.0",
}

_SHARED: list[Callable[..., Any] | type] = [
    indicators.ema, indicators.true_range, indicators.atr, indicators.adx, indicators.rsi,
    indicators.donchian_high, indicators.donchian_low, indicators.rolling_percentile,
    features.build_features, features.regime, rules.StrategyParams,
]
_ENGINE: list[Callable[..., Any] | type] = [engine.run_symbol, engine.Costs, engine.Management,
                                             engine.apply_portfolio_caps]
_CODE: dict[str, list[Callable[..., Any] | type]] = {
    "trend": [*_SHARED, *_ENGINE, rules.trend_signals, rules.trend_exit],
    "pullback": [*_SHARED, *_ENGINE, rules.pullback_signals, rules.trend_exit],
    "mean_reversion": [*_SHARED, *_ENGINE, rules.mean_reversion_signals],
    "xs_momentum": [momentum._simulate, momentum._vol, momentum._holdings_to_trades,
                    momentum.xs_momentum],
    "ts_momentum": [momentum._simulate, momentum._vol, momentum._holdings_to_trades,
                    momentum.ts_momentum],
    "ict": [indicators.atr, ict.ICTParams, ict.swings, ict.find_setups, ict.simulate,
            *_ENGINE],
    "ict_flex": [indicators.atr, ict.ICTParams, ict.swings, ict.find_setups, ict.simulate,
                 *_ENGINE],
    "supply_demand": [indicators.atr, indicators.ema, sd.SDParams, sd.find_zones, sd.orders,
                      sd.simulate, limit_sim.fill_and_manage, limit_sim.simulate_orders,
                      *_ENGINE],
}
_CONFIG_KEYS = ("grades", "filters", "management")


def canonical_json(obj: Any) -> str:
    """Byte-stable JSON: sorted keys, no whitespace, floats via repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _source(objs: list[Callable[..., Any] | type]) -> str:
    return "\n".join(inspect.getsource(o) for o in objs)


def rule_hash(strategy: str, cfg: AppConfig) -> str:
    if strategy not in RULE_VERSIONS:
        raise KeyError(f"unknown strategy {strategy!r}")
    strat = cfg.section("strategies")
    conf = {"strategy": strat.get(strategy, {}),
            "shared": {k: strat.get(k, {}) for k in _CONFIG_KEYS},
            "grid": cfg.section("parameter_grid").get(strategy, {}),
            "costs": cfg.section("costs"),
            "validation": cfg.section("validation")}
    h = hashlib.sha256()
    h.update(f"{strategy}|{RULE_VERSIONS[strategy]}\n".encode())
    h.update(_source(_CODE[strategy]).encode())
    h.update(canonical_json(conf).encode())
    return h.hexdigest()


def all_rule_hashes(cfg: AppConfig) -> dict[str, str]:
    return {s: rule_hash(s, cfg) for s in RULE_VERSIONS}


def code_hash(strategy: str) -> str:
    """Hash of the rule SOURCE CODE only (used by golden tests to catch unversioned edits)."""
    if strategy not in _CODE:
        raise KeyError(strategy)
    return hashlib.sha256(_source(_CODE[strategy]).encode()).hexdigest()
