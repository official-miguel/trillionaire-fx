"""Move study: start from the big moves, find what came before them, and test whether
those precursors would have made money - counting every time they appeared, not only
the times a move followed.

Pre-declared 2026-11-24, before it was run (config.yaml ``move_study``):

* Data: EURUSD and GBPUSD daily, ``backtest_start`` to the START of the 3-year holdout.
  The holdout is not used, so anything adopted here still gets a fair final test.
* Big move: from the close of bar t, price runs >= 3 ATR within 10 bars before going
  1 ATR against (the same definition as the missed-trade study).
* For each of the 14 precursor rules (tfx/strategies/precursors.py):
  - capture: how many of the big moves it fired for (0-5 bars before the start);
  - hit rate: P(big move in the rule's direction | rule fired) vs the base rate;
  - trades: the rule traded on its own (next open, 1 ATR stop, 3R target, 10-bar time
    stop, one at a time, real costs) -> mean R, both halves of the period, bootstrap.
* Adoption (all must hold): >= ``min_trades`` trades; mean R > 0 after costs; mean R > 0
  in BOTH halves of the period; the bootstrap lower bound at the multiple-testing
  adjusted level (5% / 14 rules) above 0. Adopted rules form the strategy
  ``move_hunter``, which must still pass the full validation gate (all pairs, holdout).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import Costs
from tfx.config import PROJECT_ROOT, AppConfig
from tfx.strategies import indicators as ind
from tfx.strategies.limit_sim import simulate_orders
from tfx.strategies.precursors import (
    ALL_RULES,
    BracketParams,
    bracket_orders,
    signals,
)

STUDY_PATH = PROJECT_ROOT / "var" / "move_study.json"


@dataclass
class RuleResult:
    rule: str
    signals: int = 0
    moves_captured: int = 0
    hit_rate: float = 0.0          # P(big move in its direction within 10 bars | fired)
    base_rate: float = 0.0         # same, for any bar, same direction mix
    trades: int = 0
    win_rate: float = 0.0
    mean_r: float | None = None
    total_r: float = 0.0
    mean_r_first_half: float | None = None
    mean_r_second_half: float | None = None
    boot_low: float | None = None
    adopted: bool = False
    why: str = ""


@dataclass
class MoveStudy:
    symbols: list[str]
    period: tuple[str, str]
    split: str
    moves: int
    up_moves: int
    down_moves: int
    bars: int
    rules: dict[str, RuleResult]
    adopted: list[str]
    tests: int
    boot_percentile: float
    summary: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)


def move_labels(df: pd.DataFrame, run_atr: float = 3.0, against_atr: float = 1.0,
                horizon: int = 10) -> np.ndarray:
    """+1/-1 if a big move in that direction starts at the close of bar t, else 0."""
    h, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    a = ind.atr(df, 14).to_numpy(float)
    n = len(df)
    out: np.ndarray = np.zeros(n, dtype=int)
    for t in range(n - 1):
        if not np.isfinite(a[t]):
            continue
        for d in (1, -1):
            for k in range(t + 1, min(n, t + horizon + 1)):
                adverse = (c[t] - lo[k]) if d == 1 else (h[k] - c[t])
                fav = (h[k] - c[t]) if d == 1 else (c[t] - lo[k])
                if adverse >= against_atr * a[t]:
                    break
                if fav >= run_atr * a[t]:
                    out[t] = d
                    break
            if out[t]:
                break
    return out


def _episodes(lab: np.ndarray, horizon: int = 10) -> list[tuple[int, int]]:
    eps, t = [], 0
    while t < len(lab):
        if lab[t]:
            eps.append((t, int(lab[t])))
            t += horizon
        else:
            t += 1
    return eps


def run_move_study(cfg: AppConfig, data: dict[str, pd.DataFrame],
                   costs: dict[str, Costs]) -> MoveStudy:
    mcfg = cfg.section("move_study")
    min_trades = int(mcfg.get("min_trades", 100))
    p = BracketParams.from_config(mcfg.get("bracket", {}))
    seed = cfg.random_seed
    start = pd.Timestamp(cfg.section("validation").get("backtest_start", "2005-01-01"))
    end_all = min(df.index[-1] for df in data.values())
    holdout = end_all - pd.DateOffset(years=int(cfg.section("validation").get(
        "holdout_years", 3)))
    split = start + (holdout - start) / 2
    per_rule: dict[str, dict[str, Any]] = {r: {"sig": 0, "hit": 0, "cap": 0, "r": [],
                                               "dates": []} for r in ALL_RULES}
    moves = up = down = bars = 0
    dir_base = {1: 0, -1: 0}
    for sym, df in data.items():
        df = df[df.index < holdout]
        lab = move_labels(df)
        sig = signals(df)
        in_win = np.asarray(df.index >= start)
        bars += int(in_win.sum())
        dir_base[1] += int(((lab == 1) & in_win).sum())
        dir_base[-1] += int(((lab == -1) & in_win).sum())
        eps = [(t, d) for t, d in _episodes(lab) if in_win[t]]
        moves += len(eps)
        up += sum(1 for _, d in eps if d == 1)
        down += sum(1 for _, d in eps if d == -1)
        for r in ALL_RULES:
            s = np.where(in_win, sig[r], 0)
            acc = per_rule[r]
            fired = np.flatnonzero(s != 0)
            acc["sig"] += len(fired)
            acc["hit"] += int(sum(1 for t in fired if lab[t] == s[t]))
            acc["cap"] += sum(1 for t, d in eps
                              if any(s[b] == d for b in range(max(0, t - 5), t + 1)))
            for tr in simulate_orders(sym, r, df, bracket_orders(df, s, p), costs[sym],
                                      p.time_stop, f"{p.target_r:g}R", start):
                if tr.exit_reason == "end of test":
                    continue
                acc["r"].append(tr.r_net)
                acc["dates"].append(tr.entry_date)
    tests = len(ALL_RULES)
    pct = 100 * 0.05 / tests
    rng = np.random.default_rng(seed)
    base_rate = (dir_base[1] + dir_base[-1]) / max(bars, 1) / 2   # one direction
    results: dict[str, RuleResult] = {}
    adopted = []
    for r in ALL_RULES:
        acc = per_rule[r]
        rs = np.asarray(acc["r"], dtype=float)
        dates = pd.DatetimeIndex(acc["dates"])
        rr = RuleResult(r, acc["sig"], acc["cap"],
                        round(acc["hit"] / acc["sig"], 4) if acc["sig"] else 0.0,
                        round(base_rate, 4), len(rs))
        if len(rs):
            rr.win_rate = round(float((rs > 0).mean()), 3)
            rr.mean_r, rr.total_r = round(float(rs.mean()), 3), round(float(rs.sum()), 1)
            a, b = rs[dates < split], rs[dates >= split]
            rr.mean_r_first_half = round(float(a.mean()), 3) if len(a) else None
            rr.mean_r_second_half = round(float(b.mean()), 3) if len(b) else None
            boot = rng.choice(rs, (4000, len(rs))).mean(axis=1)
            rr.boot_low = round(float(np.percentile(boot, pct)), 3)
        checks = [
            (rr.trades >= min_trades, f">= {min_trades} trades (got {rr.trades})"),
            ((rr.mean_r or 0) > 0, f"mean R > 0 after costs (got {rr.mean_r})"),
            ((rr.mean_r_first_half or 0) > 0 and (rr.mean_r_second_half or 0) > 0,
             f"positive in both halves (got {rr.mean_r_first_half} / "
             f"{rr.mean_r_second_half})"),
            ((rr.boot_low or -1) > 0, f"bootstrap {pct:.2f}th percentile > 0 "
                                      f"(got {rr.boot_low})"),
        ]
        failed = [txt for ok, txt in checks if not ok]
        rr.adopted = not failed
        rr.why = "ADOPTED" if not failed else "not adopted: needs " + "; ".join(failed)
        if rr.adopted:
            adopted.append(r)
        results[r] = rr
    res = MoveStudy(list(data), (str(start.date()), str(holdout.date())), str(split.date()),
                    moves, up, down, bars, results, adopted, tests, round(pct, 3))
    best = max(results.values(), key=lambda x: x.moves_captured)
    share = 100 * best.moves_captured / max(moves, 1)
    res.summary = (
        f"{moves} big moves in {bars} trading days. The rule that fired before the most of "
        f"them ({best.rule}) caught {best.moves_captured} ({share:.0f}%) "
        f"but fired {best.signals} times, a move followed {100 * best.hit_rate:.1f}% of the "
        f"time versus {100 * base_rate:.1f}% for any day. "
        + (f"Adopted for 'move_hunter': {', '.join(adopted)} (must still pass the "
           "validation gate)." if adopted else
           "No precursor met the pre-declared adoption rule: none of them predicted moves "
           "well enough to make money after costs once the false alarms are counted."))
    return res


def save(res: MoveStudy) -> None:
    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_text(json.dumps(asdict(res), indent=2, default=str), encoding="utf-8")
