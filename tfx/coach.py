"""Trade coach: fixed plain-English templates tied to measured conditions.

No speculation, no promises. Historical figures are always labelled as historical.
"""

from __future__ import annotations

from typing import Any

REGIME_GUIDE = {
    "TRENDING": ("TRENDING", "Trend sleeves active. Follow breakouts and pullbacks; let winners "
                             "run on the trailing stop.", 1.0),
    "CHOPPY": ("CHOPPY", "No clear trend. Trend entries are rare; mean reversion only in quiet "
                         "ranges. WAIT is a valid answer.", 1.0),
    "SHOCK": ("SHOCK", "Volatility is extreme (top 10% of the last year). No new entries until "
                       "it calms down.", 0.0),
    "UNKNOWN": ("UNKNOWN", "Not enough history to classify the regime.", 0.0),
}

STRATEGY_TEXT = {
    "trend": "Trend breakout",
    "pullback": "Trend pullback",
    "mean_reversion": "Mean reversion",
    "ict": "ICT sweep + MSS + FVG (strict)",
    "ict_flex": "ICT sweep + MSS + FVG (real-market tolerances)",
    "supply_demand": "Supply & demand zone (first return, with the trend)",
    "ict_learned": "ICT with relaxations learned from the missed-trade study",
}

FORBIDDEN_PHRASES = ("guarantee", "sure thing", "can't lose", "cannot lose", "risk-free",
                     "risk free", "always win", "likely to win", "certain profit")


def regime_banner(regime: str) -> dict[str, Any]:
    name, text, size = REGIME_GUIDE.get(regime, REGIME_GUIDE["UNKNOWN"])
    return {"regime": name, "guidance": text, "size_factor": size}


def headline(status: str, symbol: str) -> str:
    return {
        "BUY": f"BUY {symbol} at next open",
        "SELL": f"SELL {symbol} at next open",
        "WATCH": f"WATCH {symbol}: close to a trigger",
        "EXIT": f"EXIT {symbol} at next open",
        "HOLD": f"HOLD {symbol}: manage the open trade",
        "WAIT": f"WAIT on {symbol}: no valid setup",
        "LOCKED": f"WAIT on {symbol}: strategy not validated",
        "NO DATA": f"DATA UNAVAILABLE - NO SIGNAL ({symbol})",
    }.get(status, f"WAIT on {symbol}")


def signal_card(symbol: str, status: str, strategy: str | None, direction: int, reason: str,
                entry: float | None, stop: float | None, size: dict[str, Any] | None,
                order: dict[str, Any] | None, score: dict[str, Any] | None,
                hist: dict[str, Any] | None, watch_level: float | None, atr: float | None,
                blocks: list[str], regime: str, partial_at_r: float = 1.0,
                levels: dict[str, float] | None = None) -> dict[str, Any]:
    side = "long" if direction == 1 else "short"
    card: dict[str, Any] = {"symbol": symbol, "status": status,
                            "headline": headline(status, symbol),
                            "regime": regime_banner(regime), "blocks": blocks}
    if strategy:
        card["setup"] = STRATEGY_TEXT.get(strategy, strategy)
    if status in ("BUY", "SELL") and entry is not None and stop is not None:
        risk = abs(entry - stop)
        t1 = levels["target"] if levels else entry + direction * partial_at_r * risk
        tlabel = ("Target (5 EMA)" if strategy == "mean_reversion"
                  else f"+{partial_at_r:g}R partial (50%)")
        card.update({
            "why": reason,
            "entry": entry, "stop": stop, "risk_per_unit": risk,
            "targets": [{"label": tlabel, "price": t1}],
            "playbook": [
                f"Enter {side} with the order below; the stop is placed at the same time.",
                f"At +{partial_at_r:g}R ({t1:.5g}) take half and move the stop to breakeven "
                "plus the spread.",
                "Trail the rest on the 20-day channel; the stop only moves in your favour."
                if strategy != "mean_reversion" else
                "Exit at the 5 EMA target, at the stop, or after 5 sessions (time stop).",
                "Exit at the next open when the exit rule triggers. Do not wait for a bounce.",
            ],
            "invalidation": f"The setup is invalid if price trades through {stop:.5g} "
                            "(stop) or the next open gaps beyond it.",
            "skip_if": ["the spread is more than 3x typical",
                        "it is within 30 minutes of rollover or tier-1 news",
                        "you are stressed, rushed or chasing a loss",
                        "the next open gaps beyond the stop"],
            "size": size, "order": order, "score": score,
        })
    elif status == "WATCH":
        card["why"] = (f"Price is within 0.5 ATR of the trigger at {watch_level:.5g}. No trade "
                       "yet: the order is pre-computed so the setup is not missed."
                       if watch_level is not None else "Close to a trigger. No trade yet.")
        card["watch_level"] = watch_level
        if watch_level is not None and atr:
            card["distance_atr"] = None
    else:
        card["why"] = reason or "No valid setup at today's close. WAIT is a valid answer."
    if levels:
        card["levels"] = levels
    if hist:
        card["historical"] = {**hist, "label": "Historical results of this setup type in "
                                               "validation (not a forecast)"}
    return card


def assert_no_promises(text: str) -> None:
    low = text.lower()
    bad = [p for p in FORBIDDEN_PHRASES if p in low]
    if bad:
        raise ValueError(f"UI text contains forbidden promise wording: {bad}")


def post_trade_review(trade: Any, stop_changes: int, rules_followed: bool | None
                      ) -> dict[str, Any]:
    r = trade.r_result or 0.0
    lesson = trade.lesson or (
        "Loss within plan: a normal business cost." if r < 0 and rules_followed else
        "Rule break cost more than the market did: re-read the plan." if not rules_followed
        else "Plan followed: repeat the process, not the outcome.")
    return {"trade_id": trade.id, "symbol": trade.symbol, "r_result": round(r, 2),
            "plan_adherence": bool(rules_followed) and not trade.off_signal,
            "mfe_r": round(trade.mfe_r, 2), "mae_r": round(trade.mae_r, 2),
            "exit_followed_rules": rules_followed, "stop_changes": stop_changes,
            "lesson": lesson}
