"""Per-strategy setup rendering objects for the chart (deterministic, template text).

The SAME numbers feed the signal card and the chart: entry, stop and target in the
position box are taken from the card's levels, never recomputed separately.
Only closed bars are used. Colours are chosen in the UI (colour-blind-safe) and every
object also carries a text label and an icon.

Not applicable here: SMC / supply-demand zones (LQ, LOF, FVG, BOS, sweeps). No such
strategy exists in TrillionaireFX; drawing them would imply analysis the system does
not perform or validate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tfx.strategies.rules import StrategyParams


def _r(x: float, digits: int) -> float:
    return round(float(x), digits)


def levels(strategy: str, direction: int, row: pd.Series, entry: float, stop: float,
           partial_at_r: float, spread: float, digits: int) -> dict[str, float]:
    """Entry / stop / target / breakeven shared by the card and the chart."""
    risk = abs(entry - stop)
    target = float(row["ema_5"]) if strategy == "mean_reversion" else \
        entry + direction * partial_at_r * risk
    rr = abs(target - entry) / risk if risk > 0 else 0.0
    return {"entry": _r(entry, digits), "stop": _r(stop, digits), "target": _r(target, digits),
            "breakeven": _r(entry + direction * spread, digits), "rr": round(rr, 2),
            "risk": _r(risk, digits)}


def _time(ts: pd.Timestamp) -> int:
    t = ts if ts.tzinfo is not None else ts.tz_localize("UTC")
    return int(t.timestamp())


def _broken_level(f: pd.DataFrame, direction: int, n: int = 20, lookback: int = 60
                  ) -> float | None:
    tail = f.iloc[-lookback:]
    col = f"dch_{n}" if direction == 1 else f"dcl_{n}"
    hit = (tail["close"] > tail[col]) if direction == 1 else (tail["close"] < tail[col])
    idx = tail.index[hit.fillna(False).to_numpy(bool)]
    return float(tail.loc[idx[-1], col]) if len(idx) else None


def render(strategy: str, direction: int, f: pd.DataFrame, lv: dict[str, float],
           p: StrategyParams, partial_at_r: float, digits: int, validated: bool,
           risk_amount: float | None = None, currency: str = "",
           bars_held: int | None = None) -> dict[str, Any]:
    row = f.iloc[-1]
    t = _time(f.index[-1])
    side = "long" if direction == 1 else "short"
    up = direction == 1
    lines: list[dict[str, Any]] = [
        {"id": "entry", "label": "Entry", "icon": "▲" if up else "▼", "price": lv["entry"]},
        {"id": "stop", "label": "Initial stop", "icon": "✕", "price": lv["stop"]},
    ]
    zones: list[dict[str, Any]] = []
    markers = [{"time": t, "label": f"{side.upper()} setup", "icon": "▲" if up else "▼",
                "price": lv["entry"]}]
    adx = float(row["adx_14"])
    if strategy == "trend":
        n = p.lookbacks[0]
        lvl = float(row[f"dch_{n}" if up else f"dcl_{n}"])
        trail = float(row[f"dcl_{p.exit_channel}" if up else f"dch_{p.exit_channel}"])
        lines += [
            {"id": "channel_hi", "label": f"{n}-day high", "icon": "⎺",
             "price": _r(row[f"dch_{n}"], digits)},
            {"id": "channel_lo", "label": f"{n}-day low", "icon": "⎽",
             "price": _r(row[f"dcl_{n}"], digits)},
            {"id": "ema_100", "label": "EMA 100", "icon": "∼", "price": _r(row["ema_100"], digits)},
            {"id": "ema_200", "label": "EMA 200", "icon": "∼", "price": _r(row["ema_200"], digits)},
            {"id": "trail", "label": f"Trailing stop ({p.exit_channel}-day)", "icon": "⇢",
             "price": _r(trail, digits)},
            {"id": "target", "label": f"+{partial_at_r:g}R partial (50%)", "icon": "◆",
             "price": lv["target"]},
            {"id": "breakeven", "label": "Breakeven after partial", "icon": "=",
             "price": lv["breakeven"]},
        ]
        summary = (f"Close {lv['entry']} broke the {n}-day {'high' if up else 'low'} "
                   f"{_r(lvl, digits)}; ADX {adx:.1f}; stop {lv['stop']} "
                   f"({p.initial_stop_atr:g} ATR); +{partial_at_r:g}R at {lv['target']}; "
                   f"R:R {lv['rr']:.1f}")
    elif strategy == "pullback":
        broken = _broken_level(f, direction)
        atr = float(row["atr_20"])
        ema20 = float(row["ema_20"])
        zones.append({"id": "retest", "label": "Retest zone (EMA 20 ± 0.25 ATR)", "icon": "▭",
                      "top": _r(ema20 + 0.25 * atr, digits),
                      "bottom": _r(ema20 - 0.25 * atr, digits)})
        if broken is not None:
            lines.append({"id": "broken", "label": "Broken level", "icon": "⤒" if up else "⤓",
                          "price": _r(broken, digits)})
        lines.append({"id": "target", "label": f"+{partial_at_r:g}R partial (50%)", "icon": "◆",
                      "price": lv["target"]})
        summary = (f"Pullback to EMA 20 {_r(ema20, digits)} in a {'up' if up else 'down'}trend "
                   f"(ADX {adx:.1f}); broken level "
                   f"{'-' if broken is None else _r(broken, digits)}; stop {lv['stop']}; "
                   f"R:R {lv['rr']:.1f}")
    elif strategy == "mean_reversion":
        held = bars_held or 0
        left = max(0, p.mr_time_stop - held)
        zones.append({"id": "band", "label": "Reversion band (entry to 5 EMA target)", "icon": "▭",
                      "top": max(lv["entry"], lv["target"]),
                      "bottom": min(lv["entry"], lv["target"])})
        lines.append({"id": "target", "label": "Target (5 EMA)", "icon": "◎",
                      "price": lv["target"]})
        markers.append({"time": t, "label": f"Time stop in {left} bars", "icon": "⏱",
                        "price": lv["target"]})
        summary = (f"RSI(2) {float(row['rsi_2']):.1f} in a quiet range (ADX {adx:.1f}); "
                   f"target 5 EMA {lv['target']}; stop {lv['stop']}; time stop in {left} bars; "
                   f"R:R {lv['rr']:.1f}")
    else:
        raise ValueError(f"no chart setup for {strategy!r}")
    if not validated:
        summary = "NOT VALIDATED - view only, no trade. " + summary
    box = {"side": side, "entry": lv["entry"], "stop": lv["stop"], "target": lv["target"],
           "rr": lv["rr"], "risk_amount": risk_amount, "currency": currency,
           "from_time": t}
    for obj in (*lines, *zones):
        for k in ("price", "top", "bottom"):
            if k in obj and not np.isfinite(obj[k]):
                obj[k] = None
    return {"strategy": strategy, "validated": validated, "lines": lines, "zones": zones,
            "markers": markers, "box": box, "summary": summary}


def render_ict(st: Any, f: pd.DataFrame, digits: int, validated: bool,
               risk_amount: float | None = None, currency: str = "") -> dict[str, Any]:
    """ICT setup: swept liquidity, MSS/BOS, FVG (CE entry), order block, box, summary."""
    up = st.direction == 1
    side = "long" if up else "short"
    times = [_time(ix) for ix in f.index]
    risk = abs(st.entry - st.stop)
    rr = abs(st.target - st.entry) / risk if risk > 0 else 0.0
    lines = [
        {"id": "entry", "label": "Entry (FVG midpoint, limit)", "icon": "▲" if up else "▼",
         "price": _r(st.entry, digits)},
        {"id": "stop", "label": "Stop (beyond sweep)", "icon": "✕", "price": _r(st.stop, digits)},
        {"id": "target", "label": "Target", "icon": "◆", "price": _r(st.target, digits)},
        {"id": "lq", "label": f"LQ swept ({'sell' if up else 'buy'}-side)", "icon": "≡",
         "price": _r(st.swept_level, digits)},
        {"id": "mss", "label": "MSS / BOS level", "icon": "⇧" if up else "⇩",
         "price": _r(st.mss_level, digits)},
    ]
    zones = [{"id": "fvg", "label": f"FVG ({'bullish' if up else 'bearish'})", "icon": "▱",
              "top": _r(st.fvg_high, digits), "bottom": _r(st.fvg_low, digits)}]
    if st.ob_low is not None and st.ob_high is not None:
        zones.append({"id": "ob", "label": f"{'Demand' if up else 'Supply'} OB (quality "
                                           f"{st.quality:.0f})", "icon": "▭",
                      "top": _r(st.ob_high, digits), "bottom": _r(st.ob_low, digits)})
    markers = [
        {"time": times[st.sweep_bar], "label": "Sweep", "icon": "⤓" if up else "⤒",
         "price": _r(st.sweep_price, digits)},
        {"time": times[st.mss_bar], "label": "MSS", "icon": "⇧" if up else "⇩",
         "price": _r(st.mss_level, digits)},
    ]
    summary = (f"Sweep of {_r(st.swept_level, digits)} then MSS at {_r(st.mss_level, digits)}; "
               f"{'demand' if up else 'supply'} zone quality {st.quality:.0f}; "
               f"entry (FVG CE) {_r(st.entry, digits)}; stop {_r(st.stop, digits)}; "
               f"R:R {rr:.1f}")
    if not validated:
        summary = "NOT VALIDATED - view only, no trade. " + summary
    box = {"side": side, "entry": _r(st.entry, digits), "stop": _r(st.stop, digits),
           "target": _r(st.target, digits), "rr": round(rr, 2), "risk_amount": risk_amount,
           "currency": currency, "from_time": times[st.mss_bar]}
    return {"strategy": "ict", "validated": validated, "lines": lines, "zones": zones,
            "markers": markers, "box": box, "summary": summary}


def render_sd(zone: Any, od: Any, f: pd.DataFrame, digits: int, validated: bool,
              risk_amount: float | None = None, currency: str = "") -> dict[str, Any]:
    """Supply & demand setup: the fresh zone, limit entry at its edge, stop, target, box."""
    up = od.direction == 1
    times = [_time(ix) for ix in f.index]
    risk = abs(od.entry - od.stop)
    rr = abs(od.target - od.entry) / risk if risk > 0 else 0.0
    kind = "Demand" if up else "Supply"
    lines = [
        {"id": "entry", "label": f"Entry ({kind.lower()} zone edge, limit)",
         "icon": "▲" if up else "▼", "price": _r(od.entry, digits)},
        {"id": "stop", "label": "Stop (beyond the zone)", "icon": "✕",
         "price": _r(od.stop, digits)},
        {"id": "target", "label": "Target", "icon": "◆", "price": _r(od.target, digits)},
    ]
    zones = [{"id": "sd", "label": f"{kind} zone (fresh, leg-out {zone.strength:.1f} ATR)",
              "icon": "▤", "top": _r(zone.high, digits), "bottom": _r(zone.low, digits)}]
    markers = [{"time": times[zone.created_bar], "label": "Leg-out", "icon": "⇧" if up else "⇩",
                "price": _r(od.entry, digits)}]
    summary = (f"Fresh {kind.lower()} zone {_r(zone.low, digits)}-{_r(zone.high, digits)} "
               f"(strong leg-out {zone.strength:.1f} ATR, with the trend); entry "
               f"{_r(od.entry, digits)} on the first return; stop {_r(od.stop, digits)}; "
               f"R:R {rr:.1f}")
    if not validated:
        summary = "NOT VALIDATED - view only, no trade. " + summary
    box = {"side": "long" if up else "short", "entry": _r(od.entry, digits),
           "stop": _r(od.stop, digits), "target": _r(od.target, digits), "rr": round(rr, 2),
           "risk_amount": risk_amount, "currency": currency,
           "from_time": times[zone.created_bar]}
    return {"strategy": "supply_demand", "validated": validated, "lines": lines,
            "zones": zones, "markers": markers, "box": box, "summary": summary}
