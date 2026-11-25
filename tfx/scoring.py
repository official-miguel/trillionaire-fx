"""Setup scoring: hard rules block, soft factors only change size and grade.

Score = 100 x sum(w_i x f_i) / sum(w_i) over ENABLED factors, each f_i in [0, 1].
Score -> size: >= 70 full risk (A); 50-69 0.5-0.75 (B); 30-49 0.25 probe or WATCHLIST;
< 30 WAIT. A setup does not need every factor; imperfect setups get smaller size.

Weights: config values are starting points. :func:`fit_weights` re-estimates them
from TRAINING trades only and refuses any trade on/after the holdout start, so the
weights are frozen before the holdout test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tfx.config import SoftFactorConfig


@dataclass(frozen=True)
class Score:
    score: float
    grade: str              # A | B | PROBE | WATCH | WAIT | BLOCKED
    size_multiplier: float
    factors: dict[str, dict[str, Any]]
    hard_blocks: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.hard_blocks)


def _clip(x: float) -> float:
    return 0.0 if not math.isfinite(x) else float(min(1.0, max(0.0, x)))


def factor_values(row: pd.Series, direction: int, confidence: float, planned_rr: float,
                  same_ccy_open: int, loss_streak: int, htf_bias: int,
                  confirmations: float, spread_quality: float | None,
                  news_within_30m: bool | None) -> dict[str, float]:
    """Each soft factor as a 0-1 value from measured conditions (no guesses)."""
    adx = float(row.get("adx_14", np.nan))
    atr_pct = float(row.get("atr_pct", np.nan))
    vol = 0.0 if atr_pct > 90 else (1.0 if 20 <= atr_pct <= 80 else 0.6)
    return {
        "trend_clarity": _clip((adx - 15) / 20),
        "reward_to_risk": _clip(planned_rr / 3),
        "ensemble_agreement": _clip(confidence),
        "volatility_regime": vol if math.isfinite(atr_pct) else 0.5,
        "exposure_load": _clip(1 - same_ccy_open / 2),
        "recent_loss_streak": _clip(1 - loss_streak / 4),
        "confirmations": _clip(confirmations),
        "htf_bias": 1.0 if htf_bias == direction else (0.5 if htf_bias == 0 else 0.0),
        "session_spread": 0.5 if spread_quality is None else _clip(spread_quality),
        "news_proximity": 0.5 if news_within_30m is None else (0.0 if news_within_30m else 1.0),
        "rate_differential": 0.5,
        "positioning_cot": 0.5,
    }


def score_setup(values: dict[str, float], weights: dict[str, SoftFactorConfig],
                hard_blocks: list[str], base_grade: str = "A",
                thresholds: tuple[float, float, float] = (70, 50, 30)) -> Score:
    a, b, probe = thresholds
    used = {k: w for k, w in weights.items() if w.enabled and k in values}
    total_w = sum(w.weight for w in used.values()) or 1.0
    s = 100 * sum(w.weight * values[k] for k, w in used.items()) / total_w
    factors = {k: {"value": round(values[k], 3), "weight": w.weight, "mode": w.mode,
                   "label": w.display_label,
                   "status": "good" if values[k] >= 0.67 else ("mixed" if values[k] >= 0.34
                                                               else "weak")}
               for k, w in used.items()}
    if hard_blocks:
        return Score(round(s, 1), "BLOCKED", 0.0, factors, hard_blocks)
    if base_grade == "WATCH":
        return Score(round(s, 1), "WATCH", 0.0, factors)
    if s >= a:
        grade, mult = "A", 1.0
    elif s >= b:
        grade, mult = "B", 0.5 + 0.25 * (s - b) / (a - b)
    elif s >= probe:
        grade, mult = "PROBE", 0.25
    else:
        grade, mult = "WAIT", 0.0
    if base_grade == "B":
        mult = min(mult, 0.5)
    return Score(round(s, 1), grade, round(mult, 3), factors)


def fit_weights(train: pd.DataFrame, defaults: dict[str, SoftFactorConfig],
                holdout_start: pd.Timestamp, shrink: float = 0.5) -> dict[str, float]:
    """Estimate candidate weights from training trades only (frozen before the holdout).

    ``train``: one row per trade with factor columns, ``entry_date`` and ``r``.
    Weight ~ max(0, corr(factor, R)), shrunk toward the defaults. Informational
    factors keep their fixed weight.
    """
    if (pd.to_datetime(train["entry_date"]) >= holdout_start).any():
        raise ValueError("training data overlaps the holdout: weights must be frozen first")
    out: dict[str, float] = {}
    cand = {k: w for k, w in defaults.items() if w.enabled and w.mode == "candidate"}
    raw = {}
    for k in cand:
        if k in train and train[k].std() > 0 and len(train) >= 30:
            raw[k] = max(0.0, float(np.corrcoef(train[k], train["r"])[0, 1]))
        else:
            raw[k] = 0.0
    total_raw = sum(raw.values())
    total_def = sum(w.weight for w in cand.values()) or 1.0
    for k, w in cand.items():
        est = raw[k] / total_raw * total_def if total_raw > 0 else w.weight
        out[k] = shrink * w.weight + (1 - shrink) * est
    for k, w in defaults.items():
        if w.mode == "informational":
            out[k] = w.weight
    return out
