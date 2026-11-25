# TrillionaireFX build order (with the November 2026 update applied)

Global rule for every module: it can be switched off in `config.yaml`, it is tested on
its own, and it is kept **only** if it improves out-of-sample, after-cost expectancy in
R, Sharpe and max drawdown under the validation gate. Parameter grids are declared in
advance and every trial is counted. Modules that fail are reported and left disabled.
Nothing is tuned for win rate or trade count, and no UI text promises profit.

| # | Step | Status |
|---|------|--------|
| 1 | Data layer: providers, SQLite cache, session calendar, staleness guard, **cross-source check (>0.2 ATR = disputed, no signal)**, gap/spike/weekend flags, **immutable raw-data log** | done |
| 2 | Indicators and strategy engine: trend ensemble (fast 20/55 and slow 100/200 sleeves), pullback, mean reversion, rule-based regime detector (ADX, ATR percentile, autocorrelation) | next |
| 3 | Backtester with costs (spread, commission, slippage, swap), next-open fills | |
| 4 | Validation gate: CPCV + embargo, PBO, deflated Sharpe, 3-year holdout, sensitivity heatmap, bootstrap, stress periods, per-module and per-factor reports, edge-decay log | |
| 4a | Setup scoring: hard rules block, soft factors (0–100) set size and grade only; weights fitted on training data and frozen before the holdout | |
| 4b | New sleeves: risk-parity multi-speed trend, cross-sectional momentum, carry/rate differential (FRED) with crash filter, COT positioning (lagged to publication date), optional HMM regime, diversification, dynamic allocation with shrinkage | |
| 4c | Data/execution edge: hourly execution data, spread-by-hour, execution-quality score, order-type guidance (stop entry / limit + expiry), slippage learning loop (>25% over assumption triggers re-validation), broker cost checklist, event calendar | |
| 4d | Missed-trade tools: Missed Opportunities report, second-chance entries, pullback re-entry watch, re-entry after stop-out, watchlist automation, probe sizing, opportunity-capture rate | |
| 4e | Exit toolkit: MAE/MFE-driven exits, partial grid 1.0/1.5/2.0R, breakeven + buffer, chandelier and channel trails, structure targets, time stop, profit lock (+3R protects +1.5R), pyramiding (max 2 adds, never to a loser) | |
| 5 | Risk engine and trade manager | |
| 6 | API layer (localhost only) | |
| 7 | Frontend, including factor breakdown, execution-quality score, order-type card, edge-decay chart, capture rate, Missed Opportunities page | |
| 8 | Mindset module and journal | |
| 9 | Alerts and scheduler (signal alert after the close, pre-session order reminder) | |
| 10 | Paper-trading mode (at least 8 weeks, tracked against the backtest range) | |
| 11 | README: 10-minute setup, how to read signals, limitations and risk | |

## Open data issues found in step 1

* **Stooq** now sits behind a JavaScript proof-of-work bot check. TrillionaireFX does not
  bypass bot checks, so Stooq fails cleanly and is skipped.
* **Dukascopy**'s public datafeed returned HTTP 429 (rate limited) when tested.
* OANDA does not accept clients resident in Kenya, so it cannot be used here.
* **Chosen second source: the user's broker via a MetaTrader 5 demo account**
  (`tfx/data/mt5_provider.py`, read-only, terminal must be open and signed in). Its
  hourly bars are regrouped into London-midnight days to match Yahoo. Each bar's
  broker spread is kept in the raw log for the step-4c execution work. Until the
  terminal is signed in, every symbol is `UNVERIFIED` and no signal is shown.
* Later decision: the broker's MT5 D1 bars end at the New York close, which is what
  signals should be built on. Consider making MT5 the primary source for live signals
  (with Yahoo kept for deep history) once the broker's history depth is known.
* Yahoo daily FX bars end at London midnight, not at the 17:00 New York close. The MT5
  reference is therefore rebuilt on Yahoo's day boundary (from hourly bars) so the two
  sources are compared like for like. The disagreement rate must then be measured
  before the 0.2-ATR limit is trusted.
* Yahoo has visible bad ticks (for example EURUSD and USDJPY on 8–9 Oct 2008: an 8%+ move
  that reverses the next day). These are flagged as spikes. Step 3 must handle them,
  and step 4 must test with and without them.
* Yahoo gold (`GC=F`) is a futures proxy with 193 session gaps. AUDUSD history starts in
  May 2006.
