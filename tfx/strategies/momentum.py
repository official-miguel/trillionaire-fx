"""Momentum sleeves (spec section 4), backtested as daily return series.

* Cross-sectional currency momentum (``xs_momentum``): at each month-end close, rank
  the 8 currencies by their past ``formation_months`` return against USD; go long the
  ``n_each_side`` strongest and short the weakest, equal weight, volatility-scaled.
* Time-series momentum (``ts_momentum``): at each weekly close, hold each pair in the
  direction of its past ``lookback_weeks`` return, volatility-scaled.

Honesty rules (same as the daily engine)
---------------------------------------
* Decisions use CLOSED bars only; new weights take effect at the NEXT bar's OPEN:
  on a rebalance day the old weights earn the overnight gap, the new ones the session.
* Volatility used for sizing is trailing (60 days, known at the decision).
* Every weight change pays half the round-trip cost (spread + commission + 2x slippage)
  per side; open positions pay the nightly carry drag used everywhere else.
* "R" for a holding = holding return / (2 x its trailing volatility over the holding
  period at entry) - comparable to a 2-ATR stop. Used only for trade statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tfx.backtest.engine import Trade

USD_LEGS = {  # currency -> (pair, +1 if currency is the base of the pair, -1 if quote)
    "EUR": ("EURUSD", 1), "GBP": ("GBPUSD", 1), "AUD": ("AUDUSD", 1), "NZD": ("NZDUSD", 1),
    "JPY": ("USDJPY", -1), "CAD": ("USDCAD", -1), "CHF": ("USDCHF", -1),
}
VOL_WINDOW = 60
ANNUAL = 252


@dataclass(frozen=True)
class PairCosts:
    round_trip_frac: float   # (spread + commission + 2 x slippage) / price
    carry_frac: float        # nightly drag as a fraction of price


def _gap_and_session(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    gap = df["open"] / df["close"].shift(1) - 1
    session = df["close"] / df["open"] - 1
    return gap.fillna(0.0), session.fillna(0.0)


def _simulate(weights_at_close: pd.DataFrame, gaps: pd.DataFrame, sessions: pd.DataFrame,
              costs: dict[str, PairCosts], cost_mult: float) -> pd.Series:
    """Daily returns: weights decided at close t are held from the open of t+1."""
    w_new = weights_at_close.shift(1).fillna(0.0)        # effective from today's open
    w_old = weights_at_close.shift(2).fillna(0.0)        # held through last night
    ret = (w_old * gaps).sum(axis=1) + (w_new * sessions).sum(axis=1)
    rt = pd.Series({k: v.round_trip_frac for k, v in costs.items()})
    carry = pd.Series({k: v.carry_frac for k, v in costs.items()})
    turnover = (w_new - w_old).abs()
    trade_cost = (turnover * rt.reindex(turnover.columns).fillna(0) / 2).sum(axis=1)
    carry_cost = (w_new.abs() * carry.reindex(w_new.columns).fillna(0)).sum(axis=1)
    return ret - cost_mult * (trade_cost + carry_cost)


def _vol(r: pd.Series | pd.DataFrame) -> Any:
    return r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std() * np.sqrt(ANNUAL)


def _holdings_to_trades(weights: pd.DataFrame, gaps: pd.DataFrame, sessions: pd.DataFrame,
                        daily_vol: pd.DataFrame, strategy: str, cost_mult: float,
                        costs: dict[str, PairCosts], start: pd.Timestamp) -> list[Trade]:
    """One Trade per contiguous same-sign holding per pair (for trade statistics)."""
    trades: list[Trade] = []
    eff = weights.shift(1).fillna(0.0)
    for col in weights.columns:
        s = np.sign(eff[col])
        r = (eff[col].shift(1).fillna(0) * gaps[col] + eff[col] * sessions[col])
        change = s.ne(s.shift(1))
        run_id = change.cumsum()
        for _, idx in s[s != 0].groupby(run_id[s != 0]).groups.items():
            idx = pd.DatetimeIndex(idx)
            if idx[0] < start:
                continue
            d = int(s.loc[idx[0]])
            w0 = abs(float(eff[col].loc[idx[0]]))
            gross = float(r.loc[idx].sum()) / w0 if w0 else 0.0
            n = len(idx)
            vol_d = float(daily_vol[col].loc[idx[0]]) / np.sqrt(ANNUAL)
            unit = 2 * vol_d * np.sqrt(n) if vol_d > 0 else np.nan
            cost = cost_mult * (costs[col].round_trip_frac + costs[col].carry_frac * n)
            if not np.isfinite(unit) or unit <= 0:
                continue
            t = Trade(col, strategy, d, "A", 1.0, idx[0], idx[0], 0.0, 0.0, unit)
            t.exit_date, t.exit_reason, t.bars = idx[-1], "rebalance", n
            t.r_cost = cost / unit
            t.r_net = (gross - cost) / unit
            t.r_gross = gross / unit
            trades.append(t)
    return trades


def ts_momentum(data: dict[str, pd.DataFrame], costs: dict[str, PairCosts],
                lookback_weeks: int, start: pd.Timestamp, cost_mult: float = 1.0,
                pair_vol_target: float = 0.03) -> tuple[pd.Series, list[Trade]]:
    closes = pd.DataFrame({k: v["close"] for k, v in data.items()}).sort_index()
    gaps = pd.DataFrame({k: _gap_and_session(v)[0] for k, v in data.items()}).reindex(
        closes.index).fillna(0.0)
    sessions = pd.DataFrame({k: _gap_and_session(v)[1] for k, v in data.items()}).reindex(
        closes.index).fillna(0.0)
    daily = closes.pct_change(fill_method=None)
    vol = _vol(daily)
    fridays = closes.index[closes.index.to_series().dt.day_of_week == 4]
    mom = np.sign(np.log(closes / closes.shift(lookback_weeks * 5)))
    w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    target = (pair_vol_target / vol).clip(upper=3.0)
    w.loc[fridays] = (mom * target).loc[fridays]
    w = w.ffill().fillna(0.0)
    w[closes.isna()] = 0.0
    r = _simulate(w, gaps, sessions, costs, cost_mult)
    r = r[r.index >= start]
    return r, _holdings_to_trades(w, gaps, sessions, vol, "ts_momentum", cost_mult, costs,
                                  start)


def xs_momentum(data: dict[str, pd.DataFrame], costs: dict[str, PairCosts],
                formation_months: int, n_each_side: int, start: pd.Timestamp,
                cost_mult: float = 1.0, sleeve_vol_target: float = 0.06
                ) -> tuple[pd.Series, list[Trade]]:
    pairs = {ccy: (p, s) for ccy, (p, s) in USD_LEGS.items() if p in data}
    closes = pd.DataFrame({p: data[p]["close"] for p, _ in pairs.values()}).sort_index()
    idx = closes.index
    gaps = pd.DataFrame({p: _gap_and_session(data[p])[0] for p, _ in pairs.values()}
                        ).reindex(idx).fillna(0.0)
    sessions = pd.DataFrame({p: _gap_and_session(data[p])[1] for p, _ in pairs.values()}
                            ).reindex(idx).fillna(0.0)
    # currency value vs USD (USD itself = 1)
    value = pd.DataFrame({c: closes[p] ** s for c, (p, s) in pairs.items()})
    value["USD"] = 1.0
    month_end = idx.to_series().groupby(idx.to_period("M")).max().to_numpy()
    month_end = pd.DatetimeIndex(month_end)
    formation = np.log(value / value.shift(formation_months * 21))
    ccy_w = pd.DataFrame(0.0, index=idx, columns=value.columns)
    for t in month_end:
        f = formation.loc[t].dropna()
        if len(f) < 2 * n_each_side + 1:
            continue
        ranked = f.sort_values()
        ccy_w.loc[t, ranked.index[-n_each_side:]] = 1.0 / n_each_side
        ccy_w.loc[t, ranked.index[:n_each_side]] = -1.0 / n_each_side
    ccy_w.loc[~ccy_w.index.isin(month_end)] = np.nan
    ccy_w = ccy_w.ffill().fillna(0.0)
    # currency weights -> pair weights (USD exposure is implied by the other legs)
    pair_w = pd.DataFrame({p: ccy_w[c] * s for c, (p, s) in pairs.items()})
    raw = _simulate(pair_w, gaps, sessions, costs, 0.0)
    scale = (sleeve_vol_target / _vol(raw)).clip(upper=3.0)
    scale = scale.where(scale.index.isin(month_end)).ffill().fillna(0.0)
    pair_w = pair_w.mul(scale, axis=0)
    r = _simulate(pair_w, gaps, sessions, costs, cost_mult)
    r = r[r.index >= start]
    vol = _vol(closes.pct_change(fill_method=None))
    return r, _holdings_to_trades(pair_w, gaps, sessions, vol, "xs_momentum", cost_mult,
                                  costs, start)
