"""Fixed chart examples for golden tests (deterministic: fixed seeds, no randomness at run).

Regenerate expected files ONLY when a rule change is intended:
    .venv\\Scripts\\python.exe -m tests.golden.regen
and bump ``tfx.versioning.RULE_VERSIONS`` for every strategy whose output changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfx.config import load_config
from tfx.scoring import factor_values, score_setup
from tfx.setup_render import levels, render
from tfx.strategies.features import build_features
from tfx.strategies.ict import ICTParams, find_setups
from tfx.strategies.rules import STRATEGIES, StrategyParams, signals_for
from tfx.versioning import RULE_VERSIONS, code_hash

EXPECTED_DIR = Path(__file__).resolve().parent / "expected"
ROOT = Path(__file__).resolve().parents[2]


def _frame(close: np.ndarray, seed: int, wick: float = 0.0015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=len(close), name="date")
    o = np.concatenate([[close[0]], close[:-1]])
    w = np.abs(rng.normal(0, wick, len(close))) * close
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + w,
                         "low": np.minimum(o, close) - w, "close": close}, index=idx)


def _walk(n: int, seed: int, drift: float, vol: float, start: float = 1.10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(start * np.exp(np.cumsum(rng.normal(drift, vol, n))))


def cases() -> dict[str, pd.DataFrame]:
    up = _walk(420, 1, 0.0008, 0.004)
    up[-1] = up[-30:].max() * 1.004                     # clean breakout on the last bar
    chop = _walk(420, 2, 0.0, 0.004)
    gap = _walk(420, 3, 0.0005, 0.004)
    gap[-3:] = gap[-3:] * np.array([1.03, 1.035, 1.02])  # gap / volatility shock at the end
    down = _walk(420, 4, -0.0008, 0.004)
    down[-4:] = down[-5] * np.array([1.006, 1.010, 1.006, 0.999])  # rally into EMA 20 then down
    rng_ = 1.10 + 0.01 * np.sin(np.arange(420) / 6.0) + _walk(420, 5, 0, 0.0006, 1.0) - 1.0
    rng_[-2:] = rng_[-3] * np.array([0.994, 0.989])      # quiet range, two-day selloff
    return {"clean_uptrend_breakout": _frame(up, 11),
            "messy_chop": _frame(chop, 12, wick=0.004),
            "edge_gap_shock": _frame(gap, 13),
            "downtrend_pullback": _frame(down, 14),
            "quiet_range_selloff": _frame(rng_, 15, wick=0.0008)}


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, 6)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def compute(case: pd.DataFrame) -> dict[str, Any]:
    cfg = load_config(ROOT / "config.yaml")
    p = StrategyParams.from_config(cfg.section("strategies"))
    f = build_features(case)
    out: dict[str, Any] = {}
    for strat in STRATEGIES:
        sig = signals_for(strat, f, p)
        tail = sig.iloc[-40:]
        rows = [[str(ix.date()), int(r.direction), str(r.grade), _round(float(r.stop)),
                 _round(float(r.confidence))] for ix, r in tail.iterrows()
                if r.direction != 0 or r.grade]
        entry: dict[str, Any] = {"signals": rows,
                                 "total_fired": int((sig["direction"] != 0).sum())}
        fired = sig.index[(sig["direction"] != 0) & sig["stop"].notna()]
        if len(fired):
            # render the most recent firing bar using ONLY data up to that bar
            at = fired[-1]
            fx = f.loc[:at]
            last = sig.loc[at]
            d = int(last.direction)
            row = fx.iloc[-1]
            lv = levels(strat, d, row, round(float(row["close"]), 5), round(float(last.stop), 5),
                        1.0, 0.00008, 5)
            vals = factor_values(row, d, float(last.confidence), 2.0, 0, 0, 0, 0.5, None, None)
            sc = score_setup(vals, cfg.scoring_factors(), [], str(last.grade))
            setup = render(strat, d, fx, lv, p, 1.0, 5, False, None, "USD")
            entry.update({"setup_bar": str(at.date()), "levels": lv, "score": sc.score,
                          "grade": sc.grade, "size_multiplier": sc.size_multiplier,
                          "setup": setup})
        out[strat] = _round(entry)
    for variant in ("ict", "ict_flex"):
        ip = ICTParams.from_config(cfg.section("strategies").get(variant, {}))
        setups = find_setups(case, ip)
        out[variant] = _round({"count": len(setups), "setups": [
            {"dir": s.direction, "sweep": str(case.index[s.sweep_bar].date()),
             "mss": str(case.index[s.mss_bar].date()), "entry": float(s.entry),
             "stop": float(s.stop), "target": float(s.target), "quality": float(s.quality)}
            for s in setups[-3:]]})
    return out


GOLDEN_STRATEGIES = (*STRATEGIES, "ict", "ict_flex")


def header() -> dict[str, Any]:
    return {"rule_versions": {s: RULE_VERSIONS[s] for s in GOLDEN_STRATEGIES},
            "code_hashes": {s: code_hash(s) for s in GOLDEN_STRATEGIES}}


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)
