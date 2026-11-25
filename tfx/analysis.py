"""Top-down market read: how a discretionary trader looks at a chart, as fixed steps.

Deterministic (same candles -> same text) and template-only: no AI. Step 8 lists the
move precursors that fired today with their measured (unproven) history. It is ANALYSIS,
not a signal: nothing here is validated, it never sizes a trade and it never turns a
locked strategy into BUY/SELL. Steps, in the order a pro works:

1. Higher-timeframe bias (weekly trend, daily EMA 50/200).
2. Daily market structure from confirmed swings (HH/HL, LH/LL, or range).
3. Dealing range: where price sits (premium / discount / equilibrium).
4. Liquidity: nearest unswept buy-side above and sell-side below (where stops rest).
5. Points of interest: nearest fresh supply above / demand below, open fair value gaps.
6. Volatility and regime.
7. Plan: what would have to happen for a long or a short, and what invalidates the bias.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from tfx.strategies import indicators as ind
from tfx.strategies.features import build_features, weekly_bias
from tfx.strategies.ict import ICTParams, chart_annotations, swings
from tfx.strategies.supply_demand import SDParams, chart_zones

DISCLAIMER = ("Analysis only - not a signal. These reads are not validated and do not "
              "unlock any strategy.")


def _structure(df: pd.DataFrame, n: int = 3) -> tuple[str, list[float], list[float]]:
    sh, sl = swings(df, n)
    last = len(df) - 1
    highs = [float(df["high"].iloc[i]) for i in np.flatnonzero(sh) if i + n <= last][-2:]
    lows = [float(df["low"].iloc[i]) for i in np.flatnonzero(sl) if i + n <= last][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "unclear", highs, lows
    if highs[1] > highs[0] and lows[1] > lows[0]:
        return "bullish (higher highs and higher lows)", highs, lows
    if highs[1] < highs[0] and lows[1] < lows[0]:
        return "bearish (lower highs and lower lows)", highs, lows
    return "ranging (mixed swings)", highs, lows


def move_warnings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Precursors that fired on the last closed bar, with their MEASURED history.

    Awareness only: none passed the move study's adoption rule, so none is a signal.
    The hit rate is how often a big move in that direction followed (training data,
    EURUSD/GBPUSD), next to the base rate for any day; mean R is the bracket trade.
    """
    from tfx.strategies.precursors import BASE_RULES, RULE_TEXT, signals
    from tfx.study_moves import STUDY_PATH

    stats: dict[str, Any] = {}
    try:
        stats = json.loads(STUDY_PATH.read_text(encoding="utf-8")).get("rules", {})
    except (OSError, ValueError):
        stats = {}
    sig = signals(df)
    out = []
    for rule in BASE_RULES:
        d = int(sig[rule][-1])
        if d == 0:
            continue
        aligned = int(sig[f"{rule}+weekly"][-1]) != 0
        key = f"{rule}+weekly" if aligned else rule
        s = stats.get(key, {})
        hist = (f" - historically a big move followed {100 * s['hit_rate']:.0f}% of the time "
                f"(any day: {100 * s['base_rate']:.0f}%), {s['trades']} trades averaged "
                f"{s['mean_r']:+.2f}R, NOT proven" if s.get("mean_r") is not None else "")
        out.append({"rule": key, "direction": "up" if d == 1 else "down",
                    "with_weekly_trend": aligned,
                    "text": f"{RULE_TEXT[rule]} ({'up' if d == 1 else 'down'}"
                            f"{', with the weekly trend' if aligned else ''}){hist}"})
    return out


def market_read(df: pd.DataFrame, digits: int = 5, sd: SDParams | None = None
                ) -> dict[str, Any]:
    if len(df) < 220:
        return {"available": False, "text": "Not enough history for a top-down read.",
                "disclaimer": DISCLAIMER}
    f = build_features(df)
    row = f.iloc[-1]
    px = float(row["close"])

    def r(x: float) -> float:
        return round(float(x), digits)

    wb = int(weekly_bias(df).iloc[-1])
    weekly = {1: "up", -1: "down", 0: "neutral"}[wb]
    daily_up = float(ind.ema(df["close"], 50).iloc[-1]) > float(ind.ema(df["close"], 200).iloc[-1])
    structure, highs, lows = _structure(df)
    s_score = 1 if structure.startswith("bullish") else -1 if structure.startswith(
        "bearish") else 0
    if lows and px < lows[-1]:
        structure += f"; price has broken below the last swing low {r(lows[-1])} " \
                     "(bearish break of structure)"
        s_score = -1
    elif highs and px > highs[-1]:
        structure += f"; price has broken above the last swing high {r(highs[-1])} " \
                     "(bullish break of structure)"
        s_score = 1
    score = wb + (1 if daily_up else -1) + s_score
    bias = "bullish" if score >= 2 else "bearish" if score <= -2 else "neutral / mixed"

    # dealing range: last swings, stretched to include where price has gone since
    recent_hi, recent_lo = float(df["high"].iloc[-20:].max()), float(df["low"].iloc[-20:].min())
    rng_hi = max(highs[-1], recent_hi) if highs else recent_hi
    rng_lo = min(lows[-1], recent_lo) if lows else recent_lo
    pos = (px - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5
    zone = ("premium (upper half)" if pos > 0.55 else "discount (lower half)" if pos < 0.45
            else "equilibrium (middle)")

    ann = chart_annotations(df, ICTParams(swing_n=3))
    above = sorted([x["price"] for x in ann["liquidity"]
                    if x["side"] == "buy-side" and x["price"] > px])
    below = sorted([x["price"] for x in ann["liquidity"]
                    if x["side"] == "sell-side" and x["price"] < px], reverse=True)
    zones = chart_zones(df, sd or SDParams(), max_items=10)
    supply = sorted([z for z in zones if z["dir"] == "supply" and z["low"] > px],
                    key=lambda z: z["low"])
    demand = sorted([z for z in zones if z["dir"] == "demand" and z["high"] < px],
                    key=lambda z: -z["high"])
    fvgs = ann["fvgs"][:3]
    atr = float(row["atr_20"])
    vol_pct = float(row["atr_pct"]) if np.isfinite(row["atr_pct"]) else float("nan")
    vol = ("high" if vol_pct >= 80 else "low" if vol_pct <= 20 else "normal") \
        if np.isfinite(vol_pct) else "unknown"

    steps = [
        {"step": "1. Higher-timeframe bias",
         "text": f"Weekly trend {weekly}; daily EMA50 is {'above' if daily_up else 'below'} "
                 f"EMA200. Overall bias: {bias}."},
        {"step": "2. Market structure (daily)",
         "text": f"Structure is {structure}."
                 + (f" Last swing high {r(highs[-1])}, last swing low {r(lows[-1])}."
                    if highs and lows else "")},
        {"step": "3. Dealing range",
         "text": f"Price {r(px)} is at {pos * 100:.0f}% of the range {r(rng_lo)}-{r(rng_hi)}: "
                 f"{zone}. Pros prefer buying in discount and selling in premium."},
        {"step": "4. Liquidity (where stops rest)",
         "text": (f"Buy-side liquidity above at {r(above[0])}" if above else
                  "No unswept buy-side liquidity nearby above")
                 + "; " + (f"sell-side liquidity below at {r(below[0])}." if below else
                           "no unswept sell-side liquidity nearby below.")},
        {"step": "5. Points of interest",
         "text": "; ".join([
             (f"nearest fresh supply {r(supply[0]['low'])}-{r(supply[0]['high'])}"
              if supply else "no fresh supply above"),
             (f"nearest fresh demand {r(demand[0]['low'])}-{r(demand[0]['high'])}"
              if demand else "no fresh demand below"),
             (f"{len(fvgs)} open fair value gap(s), newest {r(fvgs[0]['low'])}-"
              f"{r(fvgs[0]['high'])} ({fvgs[0]['dir']})" if fvgs else "no open FVGs"),
         ]) + "."},
        {"step": "6. Volatility",
         "text": f"ATR(20) {r(atr)} ({vol} versus the last year); regime {row['regime']}."},
    ]
    if bias == "bullish":
        poi = (f"demand {r(demand[0]['low'])}-{r(demand[0]['high'])}" if demand else
               f"the discount half below {r((rng_lo + rng_hi) / 2)}")
        plan = (f"Look for longs only: wait for price to come back into {poi}, ideally after "
                f"a sweep of sell-side liquidity{f' at {r(below[0])}' if below else ''} and a "
                f"shift back up. Targets: buy-side liquidity"
                f"{f' at {r(above[0])}' if above else ''}. Bias invalidated by a daily close "
                f"below {r(rng_lo)}.")
    elif bias == "bearish":
        poi = (f"supply {r(supply[0]['low'])}-{r(supply[0]['high'])}" if supply else
               f"the premium half above {r((rng_lo + rng_hi) / 2)}")
        plan = (f"Look for shorts only: wait for price to rally into {poi}, ideally after a "
                f"sweep of buy-side liquidity{f' at {r(above[0])}' if above else ''} and a "
                f"shift back down. Targets: sell-side liquidity"
                f"{f' at {r(below[0])}' if below else ''}. Bias invalidated by a daily close "
                f"above {r(rng_hi)}.")
    else:
        plan = (f"No clear bias: the timeframes disagree. A pro usually stands aside or waits "
                f"for a sweep of either side ({r(above[0]) if above else '-'} above / "
                f"{r(below[0]) if below else '-'} below) followed by a clear structure shift.")
    steps.append({"step": "7. Plan", "text": plan})
    warnings = move_warnings(df)
    steps.append({"step": "8. Move warning signs today",
                  "text": "; ".join(w["text"] for w in warnings) if warnings else
                  "None of the 14 tested move precursors fired on the last closed bar."})
    return {"available": True, "bias": bias, "price": r(px), "range": [r(rng_lo), r(rng_hi)],
            "range_position": round(pos, 3), "zone": zone, "steps": steps,
            "warnings": warnings,
            "liquidity": {"above": [r(x) for x in above[:2]], "below": [r(x) for x in below[:2]]},
            "supply": supply[:2], "demand": demand[:2], "disclaimer": DISCLAIMER,
            "bar_date": str(df.index[-1].date())}
