"""Missed-trade study: which near-miss ICT setups were skipped, and were they good?

Pre-declared 2026-11-24 (before the study was run); see config.yaml ``missed_study``.

* Study data: the study symbols (EURUSD, GBPUSD), D1, from ``backtest_start`` to the
  START of the 3-year holdout. The holdout is never looked at here, so it stays a
  fair final test for anything this study suggests.
* Candidates: ICT setups found with deliberately LOOSE rules (sweep close may stay
  0.5 ATR beyond the level, near-gaps overlapping up to 0.3 ATR, any range position).
  Each candidate is traded on its own (same limit-order execution, real costs).
* A candidate is "taken" if the strict ICT rules produce the same setup (same
  direction and structure-shift bar). Otherwise it is "missed" and tagged with every
  strict condition it failed: sweep did not close back inside, no true gap, entry
  not in discount/premium.
* Winners AND losers are counted. Judging only the missed winners is hindsight and
  would build a system that loses money live.
* Adoption rule (fixed in advance): a relaxation is adopted into ``ict_learned`` only
  if its missed group has >= ``min_trades`` filled trades, a positive mean R after
  costs, AND the 5th percentile of a bootstrap of the mean R is above 0.
* Big-move coverage: bars from which price ran >= 3 ATR in one direction within 10
  bars before moving 1 ATR against. Reported: how many such moves each rule set had
  a setup for (they are only visible afterwards; the base rate is shown too).
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
from tfx.strategies.ict import ICTParams, Setup, find_setups
from tfx.strategies.limit_sim import LimitOrder, fill_and_manage

STUDY_PATH = PROJECT_ROOT / "var" / "missed_study.json"
CONDITIONS = ("sweep_close", "true_gap", "discount")
CONDITION_TEXT = {
    "sweep_close": "the sweep candle did not close back inside the swept level",
    "true_gap": "the gap was a near-gap (wicks overlapped slightly), not a true FVG",
    "discount": "the entry was not in the discount (long) / premium (short) half",
}
#: relaxation adopted per failed condition -> ICTParams tolerance values
RELAXATION = {"sweep_close": {"sweep_close_tol_atr": 0.5},
              "true_gap": {"gap_tol_atr": 0.3},
              "discount": {"discount_max": 1.0}}
LOOSE = {"sweep_close_tol_atr": 0.5, "gap_tol_atr": 0.3, "discount_max": 1.0}


@dataclass
class Candidate:
    symbol: str
    direction: int
    mss_date: str
    entry: float
    stop: float
    target: float
    taken: bool
    failed: list[str]
    filled: bool
    r: float | None
    exit_reason: str = ""


@dataclass
class Group:
    name: str
    candidates: int = 0
    filled: int = 0
    wins: int = 0
    losses: int = 0
    mean_r: float | None = None
    total_r: float = 0.0
    boot_p5: float | None = None
    adopted: bool = False
    why: str = ""


@dataclass
class StudyResult:
    symbols: list[str]
    period: tuple[str, str]
    holdout_start: str
    groups: dict[str, Group]
    big_moves: dict[str, Any]
    adopted: list[str]
    learned_params: dict[str, float]
    profitable_missed: int
    losing_missed: int
    examples: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


def _tags(df: pd.DataFrame, st: Setup) -> list[str]:
    h, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    d, s, k = st.direction, st.sweep_bar, st.fvg_bar
    failed = []
    if not ((d == 1 and c[s] > st.swept_level) or (d == -1 and c[s] < st.swept_level)):
        failed.append("sweep_close")
    if not ((d == 1 and lo[k] > h[k - 2]) or (d == -1 and h[k] < lo[k - 2])):
        failed.append("true_gap")
    span = st.range_high - st.range_low
    pos = ((st.entry - st.range_low) / span if d == 1 else (st.range_high - st.entry) / span)
    if pos > 0.5 + 1e-12:
        failed.append("discount")
    return failed


def candidates(symbol: str, df: pd.DataFrame, base: ICTParams, costs: Costs,
               end: pd.Timestamp) -> list[Candidate]:
    loose = ICTParams(**{**asdict(base), **LOOSE})
    a = ind.atr(df, 14).to_numpy(float)
    strict_keys = {(s.direction, s.mss_bar) for s in find_setups(df, base, a)}
    out = []
    for st in find_setups(df, loose, a):
        if df.index[st.mss_bar] >= end:
            continue
        taken = (st.direction, st.mss_bar) in strict_keys
        failed = [] if taken else _tags(df, st)
        od = LimitOrder(st.direction, st.mss_bar, st.entry, st.stop, st.target, st.expiry_bar)
        tr = fill_and_manage(symbol, "ict", df.loc[:end], od, costs, base.time_stop, "2R") \
            if st.mss_bar < len(df.loc[:end]) - 1 else None
        out.append(Candidate(symbol, st.direction, str(df.index[st.mss_bar].date()),
                             round(st.entry, 5), round(st.stop, 5), round(st.target, 5),
                             taken, failed, tr is not None,
                             None if tr is None else round(tr.r_net, 3),
                             "" if tr is None else tr.exit_reason))
    return out


def _group(name: str, rows: list[Candidate], seed: int, min_trades: int) -> Group:
    g = Group(name, candidates=len(rows))
    rs = np.array([r.r for r in rows if r.r is not None], dtype=float)
    g.filled = len(rs)
    if not len(rs):
        g.why = "no filled trades"
        return g
    g.wins, g.losses = int((rs > 0).sum()), int((rs <= 0).sum())
    g.mean_r, g.total_r = round(float(rs.mean()), 3), round(float(rs.sum()), 2)
    rng = np.random.default_rng(seed)
    boot = rng.choice(rs, (2000, len(rs))).mean(axis=1)
    g.boot_p5 = round(float(np.percentile(boot, 5)), 3)
    return g


def big_moves(df: pd.DataFrame, end: pd.Timestamp, known: dict[str, set[tuple[int, int]]],
              run_atr: float = 3.0, against_atr: float = 1.0, horizon: int = 10,
              lead: int = 5) -> dict[str, Any]:
    """Count hindsight 'obvious' moves and how many each rule set had a setup for."""
    h, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    a = ind.atr(df, 14).to_numpy(float)
    n = int(df.index.searchsorted(end))
    moves, t, total_bars = [], 20, 0
    while t < n - horizon:
        total_bars += 1
        hit = 0
        for d in (1, -1):
            for k in range(t + 1, t + horizon + 1):
                adverse = (c[t] - lo[k]) if d == 1 else (h[k] - c[t])
                fav = (h[k] - c[t]) if d == 1 else (c[t] - lo[k])
                if adverse >= against_atr * a[t]:
                    break
                if fav >= run_atr * a[t]:
                    hit = d
                    break
            if hit:
                break
        if hit:
            moves.append((t, hit))
            t += horizon          # one move counted once
        else:
            t += 1
    cover = {}
    for name, keys in known.items():
        cover[name] = sum(1 for t, d in moves
                          if any((d, b) in keys for b in range(t - lead, t + 1)))
    return {"moves": len(moves), "bars_checked": total_bars,
            "base_rate": round(len(moves) / max(total_bars, 1), 3),
            "had_a_setup": cover}


def run_study(cfg: AppConfig, data: dict[str, pd.DataFrame],
              costs: dict[str, Costs]) -> StudyResult:
    scfg = cfg.section("missed_study")
    seed, min_trades = cfg.random_seed, int(scfg.get("min_trades", 30))
    base = ICTParams.from_config(cfg.section("strategies").get("ict", {}),
                                 scfg.get("base_params", {}))
    end_all = min(df.index[-1] for df in data.values())
    holdout_start = end_all - pd.DateOffset(years=int(cfg.section("validation").get(
        "holdout_years", 3)))
    start = pd.Timestamp(cfg.section("validation").get("backtest_start", "2005-01-01"))
    rows: list[Candidate] = []
    moves_total: dict[str, Any] = {"moves": 0, "bars_checked": 0, "had_a_setup": {}}
    for sym, df in data.items():
        df = df[df.index >= start - pd.DateOffset(years=1)]
        c = candidates(sym, df, base, costs[sym], holdout_start)
        rows += [x for x in c if pd.Timestamp(x.mss_date) >= start]
        a = ind.atr(df, 14).to_numpy(float)
        loose = ICTParams(**{**asdict(base), **LOOSE})
        from tfx.strategies.supply_demand import SDParams
        from tfx.strategies.supply_demand import orders as sd_orders
        sdp = SDParams.from_config(cfg.section("strategies").get("supply_demand", {}))
        known = {"strict ICT": {(s.direction, s.mss_bar) for s in find_setups(df, base, a)},
                 "loose ICT": {(s.direction, s.mss_bar) for s in find_setups(df, loose, a)},
                 "supply & demand": {(o.direction, o.known_bar) for o in sd_orders(df, sdp, a)}}
        bm = big_moves(df, holdout_start, known)
        moves_total["moves"] += bm["moves"]
        moves_total["bars_checked"] += bm["bars_checked"]
        for k, v in bm["had_a_setup"].items():
            moves_total["had_a_setup"][k] = moves_total["had_a_setup"].get(k, 0) + v
    moves_total["base_rate"] = round(moves_total["moves"] / max(moves_total["bars_checked"], 1), 3)

    groups = {"taken (strict ICT)": _group("taken (strict ICT)",
                                           [r for r in rows if r.taken], seed, min_trades)}
    missed = [r for r in rows if not r.taken]
    groups["missed (all)"] = _group("missed (all)", missed, seed, min_trades)
    adopted = []
    for cond in CONDITIONS:
        only = [r for r in missed if r.failed == [cond]]
        g = _group(f"missed: only {cond}", only, seed, min_trades)
        ok = (g.filled >= min_trades and (g.mean_r or 0) > 0 and (g.boot_p5 or -1) > 0)
        g.adopted = ok
        g.why = (f"ADOPTED: {g.filled} trades, mean {g.mean_r:+.3f}R, bootstrap 5th pct "
                 f"{g.boot_p5:+.3f}R > 0" if ok else
                 f"not adopted: needs >= {min_trades} trades, mean R > 0 and bootstrap 5th "
                 f"percentile > 0 (got {g.filled} trades, mean {g.mean_r}, p5 {g.boot_p5})")
        groups[g.name] = g
        if ok:
            adopted.append(cond)
    multi = [r for r in missed if len(r.failed) > 1]
    groups["missed: 2+ conditions"] = _group("missed: 2+ conditions", multi, seed, min_trades)
    learned: dict[str, float] = {}
    for cond in adopted:
        learned.update(RELAXATION[cond])
    filled_missed = [r for r in missed if r.r is not None]
    prof = sum(1 for r in filled_missed if (r.r or 0) > 0)
    examples = sorted(filled_missed, key=lambda r: r.mss_date)[-12:]
    res = StudyResult(list(data), (str(start.date()), str(holdout_start.date())),
                      str(holdout_start.date()), groups, moves_total, adopted, learned,
                      prof, len(filled_missed) - prof,
                      [asdict(e) for e in examples])
    res.summary = (
        f"{len(missed)} near-miss setups were skipped by the strict rules; "
        f"{len(filled_missed)} would have filled: {prof} winners and "
        f"{len(filled_missed) - prof} losers. "
        + (f"Adopted relaxations: {', '.join(adopted)} -> strategy 'ict_learned' "
           "(must still pass the full validation gate, holdout included)."
           if adopted else "No relaxation met the pre-declared adoption rule: taking the "
           "skipped setups would not have improved results after costs."))
    return res


def save(res: StudyResult) -> None:
    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_text(json.dumps(asdict(res), indent=2, default=str), encoding="utf-8")


def load() -> dict[str, Any] | None:
    if not STUDY_PATH.exists():
        return None
    out: dict[str, Any] = json.loads(STUDY_PATH.read_text(encoding="utf-8"))
    return out
