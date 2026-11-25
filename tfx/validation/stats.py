"""Performance metrics and overfitting statistics.

* Probabilistic / Deflated Sharpe Ratio — Bailey & López de Prado (2012, 2014).
* Probability of Backtest Overfitting via CSCV — Bailey, Borwein, López de Prado &
  Zhu (2015), with an embargo between in-sample and out-of-sample blocks.
* Stationary block bootstrap and Monte Carlo trade shuffles for confidence ranges.
All randomness is seeded (deterministic results).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------------------ metrics
def sharpe(r: np.ndarray | pd.Series, annualise: bool = True) -> float:
    x = np.asarray(r, float)
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return 0.0
    s = float(np.mean(x) / np.std(x, ddof=1))
    return s * math.sqrt(TRADING_DAYS) if annualise else s


def equity_curve(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def max_drawdown(r: pd.Series | np.ndarray) -> float:
    eq: Any = np.cumprod(1 + np.asarray(r, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(((eq - peak) / peak).min()) if len(eq) else 0.0


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    years = len(r) / TRADING_DAYS
    final = float(np.prod(1 + r.to_numpy()))
    return final ** (1 / years) - 1 if final > 0 and years > 0 else -1.0


def longest_losing_streak(rs: list[float]) -> int:
    best = cur = 0
    for x in rs:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def trade_metrics(r_multiples: list[float], cost_r: list[float] | None = None,
                  months: float | None = None) -> dict[str, float]:
    r: Any = np.asarray(r_multiples, float)
    if len(r) == 0:
        return {"trades": 0, "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
                "payoff": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0,
                "longest_losing_streak": 0, "trades_per_month": 0.0, "cost_per_trade_r": 0.0}
    wins, losses = r[r > 0], r[r <= 0]
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    gross_l = float(-losses.sum())
    return {
        "trades": len(r), "win_rate": float(len(wins) / len(r)),
        "avg_win_r": avg_w, "avg_loss_r": avg_l,
        "payoff": float(avg_w / -avg_l) if avg_l < 0 else float("inf"),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins.sum() / gross_l) if gross_l > 0 else float("inf"),
        "longest_losing_streak": longest_losing_streak(list(r)),
        "trades_per_month": float(len(r) / months) if months else 0.0,
        "cost_per_trade_r": float(np.mean(cost_r)) if cost_r else 0.0,
    }


def per_year(r: pd.Series) -> dict[int, float]:
    g = (1 + r).groupby(pd.DatetimeIndex(r.index).year).prod() - 1
    return {int(k): float(v) for k, v in g.items()}


# ------------------------------------------------------------ Sharpe stats
def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float:
    """P(true SR > benchmark). ``sr`` and benchmark are PER-PERIOD (not annualised)."""
    if n < 3:
        return 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(trial_srs: list[float], n_trials: int) -> float:
    """Expected maximum per-period SR of ``n_trials`` unskilled trials."""
    n_trials = max(2, n_trials)
    var = float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 1 else 0.0
    sd = math.sqrt(var)
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(sd * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: np.ndarray, trial_srs: list[float], n_trials: int) -> float:
    """DSR: probability the selected strategy's SR beats the best of ``n_trials`` flukes."""
    x = np.asarray(returns, float)
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    sr = sharpe(x, annualise=False)
    sr0 = expected_max_sharpe(trial_srs, n_trials)
    return probabilistic_sharpe(sr, len(x), float(stats.skew(x)),
                                float(stats.kurtosis(x, fisher=False)), sr0)


# ---------------------------------------------------------------- PBO/CSCV
@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    median_oos_sharpe_of_is_best: float
    logits: list[float]


def pbo_cscv(matrix: np.ndarray, n_blocks: int = 10, embargo: int = 10) -> PBOResult:
    """Probability that the in-sample best configuration ranks below median out-of-sample.

    ``matrix``: T x N daily returns of N trials. Blocks adjacent to the OOS blocks lose
    ``embargo`` rows on the in-sample side (purging serial leakage).
    """
    m = np.asarray(matrix, float)
    t, n = m.shape
    if n < 2 or t < n_blocks * 5:
        return PBOResult(1.0, 0, 0.0, [])
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    logits, oos_best = [], []
    for is_ids in itertools.combinations(range(n_blocks), n_blocks // 2):
        oos_ids = [b for b in range(n_blocks) if b not in is_ids]
        oos_rows = np.concatenate([blocks[b] for b in oos_ids])
        banned: set[int] = set()
        for b in oos_ids:
            lo, hi = int(blocks[b][0]), int(blocks[b][-1])
            banned.update(range(max(0, lo - embargo), lo))
            banned.update(range(hi + 1, min(t, hi + 1 + embargo)))
        is_rows = np.array([r for b in is_ids for r in blocks[b] if r not in banned])
        is_sr = np.array([sharpe(m[is_rows, k], False) for k in range(n)])
        oos_sr = np.array([sharpe(m[oos_rows, k], False) for k in range(n)])
        best = int(np.argmax(is_sr))
        rank = stats.rankdata(oos_sr)[best]          # 1 = worst ... n = best
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
        oos_best.append(oos_sr[best] * math.sqrt(TRADING_DAYS))
    lg = np.array(logits)
    return PBOResult(float((lg <= 0).mean()), len(lg), float(np.median(oos_best)), list(lg))


# ------------------------------------------------------- bootstrap / Monte Carlo
def block_bootstrap(r: pd.Series, n_samples: int = 2000, block: int = 20,
                    seed: int = 42) -> dict[str, tuple[float, float]]:
    """Stationary block bootstrap 5%-95% ranges for CAGR, Sharpe and max drawdown."""
    x = r.to_numpy(float)
    t = len(x)
    if t < block * 2:
        return {}
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {"cagr": [], "sharpe": [], "max_drawdown": []}
    p = 1 / block
    n_blocks = int(t * p * 3) + 10
    for _ in range(n_samples):
        lengths = rng.geometric(p, size=n_blocks)
        starts = rng.integers(0, t, size=n_blocks)
        idx = np.concatenate([(s + np.arange(n)) % t for s, n in zip(starts, lengths,
                                                                        strict=True)])[:t]
        s = pd.Series(x[idx])
        out["cagr"].append(cagr(s))
        out["sharpe"].append(sharpe(s))
        out["max_drawdown"].append(max_drawdown(s))
    return {k: (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
            for k, v in out.items()}


def shuffle_drawdowns(r_multiples: list[float], risk_pct: float, n: int = 2000,
                      seed: int = 42) -> dict[str, Any]:
    """Max drawdown and losing-streak percentiles from random trade orderings."""
    r: Any = np.asarray(r_multiples, float)
    if len(r) == 0:
        return {}
    rng = np.random.default_rng(seed)
    dds, streaks = [], []
    for _ in range(n):
        s = rng.permutation(r)
        dds.append(max_drawdown(s * risk_pct))
        streaks.append(longest_losing_streak(list(s)))
    return {"max_drawdown_p50": float(np.percentile(dds, 50)),
            "max_drawdown_p95": float(np.percentile(dds, 5)),   # 5th pct = worse DD
            "losing_streak_p50": float(np.percentile(streaks, 50)),
            "losing_streak_p95": float(np.percentile(streaks, 95))}
