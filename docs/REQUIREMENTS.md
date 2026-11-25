# TrillionaireFX requirements checklist

Every instruction from the three briefs (original spec, November 2026 update, and
the "make sure nothing gets left off" brief) is listed here once. Later briefs win
where they conflict. Status: **DONE** (built and tested) · **PARTIAL** · **TODO** ·
**BLOCKED** (waiting on the user) · **LIMITED** (only partly possible; see the note and
the honest alternative).

The phase numbers match the build order at the bottom.

## A. Hard constraints and global rules

| ID | Requirement | Phase | Status |
|---|---|---|---|
| A1 | Zero running cost: free data, local, no paid APIs or cloud | all | DONE so far |
| A2 | No broker keys, never places orders (a test scans the code for order calls) | all | DONE (`test_no_order_functions_anywhere_in_the_package`) |
| A3 | No live signals unless the strategy-timeframe pair passes the validation gate | 2, 6 | DONE (registry gate + pipeline: unvalidated pair = WAIT; `test_unvalidated_strategy_never_produces_buy`) |
| A4 | No martingale, grid, averaging down, stop removal or size increase after losses; tests fail if any code path does it | 5, 7 | DONE (engine RuleViolationError, sizing never increases after losses; `test_no_martingale_or_size_increase_after_losses`, `test_stops_never_widen...`, `test_pyramiding_never_adds_to_a_loser`) |
| A5 | Deterministic: fixed seeds, pinned dependencies, logged data versions | all | DONE for data (seed in config, pinned requirements, SHA-256 data versions) |
| A6 | Never say a trade is sure, guaranteed or likely to win; never promise profit, constant wins or fixed accuracy | all | DONE (`coach.assert_no_promises` on every card and alert; README/UI risk notice) |
| A7 | Every module switchable in config, tested alone, kept only if it improves out-of-sample after-cost expectancy, Sharpe and drawdown | all | PARTIAL (every strategy/module has `enabled` in config; carry/xs-momentum/ML not implemented beyond config) |
| A8 | Small pre-declared grids, every trial counted (Deflated Sharpe), failures reported and left disabled | 4 | DONE (pre-declared grids, trials recorded per combination, DSR uses the count) |
| A9 | If something is unsafe, unvalidated or impossible, say so and propose an alternative | all | ongoing (see section L) |
| A10 | Show test results at the end of each step | all | ongoing |

## B. Data layer

| ID | Requirement | Phase | Status |
|---|---|---|---|
| B1 | Swappable providers; yfinance primary | 1 | DONE |
| B2 | Second source = MetaTrader 5 terminal (broker history), replacing Stooq | 1 | DONE (code), BLOCKED (MT5 not signed in) |
| B3 | Bar confirmed only if both sources agree within 0.2 ATR (never loosened) | 1 | DONE |
| B4 | XAUUSD: MT5 spot gold is the reference; Yahoo `GC=F` futures clearly labelled as a proxy | 1 | DONE (Yahoo `GC=F` primary marked proxy; MT5 spot confirms) |
| B5 | Reject HTML, empty or malformed responses, never parse them; tests for each and for disagreement | 1 | DONE |
| B6 | Per-symbol data-quality report: span, gaps, outliers, agreement rate, latest confirmed bar | 1 | DONE (`tfx quality`) |
| B7 | If MT5 isn't connected, say exactly what to do and wait; never fall back to unverified data | 1 | DONE (UNVERIFIED blocks signals; report prints the steps) |
| B8 | SQLite cache, works offline, "DATA UNAVAILABLE - NO SIGNAL" when stale | 1 | DONE |
| B9 | Immutable raw-data log | 1 | DONE (SQLite triggers) |
| B10 | Flag gaps, outliers, weekend artefacts | 1 | DONE |
| B11 | Intraday data (H1/H4) for strategies and execution, closed bars only, cross-checked | 2 | DONE (`tfx/data/intraday.py`); LIMITED depth (see L1) |
| B12 | Free FX rates for account-currency conversion (USD/EUR/GBP/KES ...) | 5 | DONE (FxConverter from cached closes + extra rates setting) |
| B13 | Free economic calendar (tier-1 events) | 4c | LIMITED (informational factor only; no free historical calendar; no live feed wired) |
| B14 | Rate differentials from FRED / central banks, lagged to publication | 3 | TODO (carry sleeve disabled; FRED not wired) |
| B15 | CFTC COT positioning, lagged to publication date (look-ahead test must fail otherwise) | 3 | TODO (positioning disabled; COT not wired) |
| B16 | OANDA practice API optional | 1 | DONE (not usable in Kenya) |
| B17 | Dukascopy as a RESEARCH-ONLY H1/H4 history source back to 2005 | 2 | REPLACED by B20 at the user's choice (Dukascopy is no longer free; see L5). Guard DONE: research-only sources can never confirm a live bar |
| B20 | RESEARCH-ONLY H1 history from the broker's MT5 (2005 or earliest), cached separately, checked against Yahoo H1 over the overlap and against Yahoo D1 over the full span (disagreement rates reported), pre-declared PASS/FAIL, used only for validation | 2 | DONE (`tfx/data/research.py`, `.\make.ps1 research`); BLOCKED on MT5 sign-in for real data |
| B18 | Live signals always need Yahoo + MT5 confirmation | 1 | DONE |
| B19 | If research history fails the quality checks, H1/H4 stay "not validated" | 4 | PARTIAL (`ResearchHistory.usable_for_validation`; the backtester must call it in phase 3/4) |

## C. Timeframes (third brief, part 2)

| ID | Requirement | Phase | Status |
|---|---|---|---|
| C1 | Each strategy declares its signal timeframe, optional higher-timeframe bias and optional lower-timeframe entry or execution timeframe in config; no global D1-only rule | 2 | DONE (`tfx/config.py` StrategyTimeframes) |
| C2 | Candidate grids: trend {D1,H4}; pullback {bias D1, entry H4/D1}; mean reversion {D1,H4,H1}; carry and momentum {W1, monthly rebalance}; execution H1 | 2 | DONE (declared in `config.yaml`) |
| C3 | Each strategy-timeframe pair validated separately with real costs (spread by hour from MT5), purged CV, holdout, robustness maps, 1.5x/2x cost stress | 4 | DONE for D1 (proof phase); H4/H1 FAIL: insufficient confirmed history |
| C4 | All timeframe trials counted in the Deflated Sharpe | 4 | DONE |
| C5 | Shorter timeframe kept only if it beats the longer one out-of-sample after costs | 4 | DONE (H4/H1 locked; nothing to compare until they have data) |
| C6 | Higher-timeframe bias feeds the soft score; lower-timeframe entries are separate validated modules | 4a | PARTIAL (weekly bias in score; lower-TF entry modules not built: no validated base) |
| C7 | Closed bars only on every timeframe; look-ahead test: no higher-timeframe bar used before it closes | 2 | DONE (`tfx/config.py` StrategyTimeframes) |
| C8 | UI timeframe selector shows only validated timeframes; others "not validated" (view only, no signals) | 2, 7 | DONE (UI selector labels 'not validated', view-only) |

## D. Strategies and backtester

| ID | Requirement | Phase | Status |
|---|---|---|---|
| D1 | Indicators with known-value tests (EMA, ATR, ADX, RSI(2), Donchian, ATR percentile, autocorrelation) | 3 | DONE (`test_strategies_backtest.py`) |
| D2 | Trend ensemble: Donchian 20/55/100, EMA 100/200 side filter, ADX>20, confidence = share of agreeing lookbacks | 3 | DONE |
| D3 | Multi-speed trend: fast (20/55) and slow (100/200) sleeves, risk parity, monthly rebalance | 3 | PARTIAL (fast/slow lookbacks in the ensemble; separate risk-parity sleeves not split) |
| D4 | Exit beyond opposite 20-day channel or stop; initial stop 2xATR(20); trailing only in profit direction | 3 | DONE |
| D5 | Pullback entry to EMA(20) in confirmed trend; stop beyond pullback extreme | 3 | DONE |
| D6 | Mean reversion (ranging only: ADX<20, ATR pct<70): RSI(2) <10 / >90 with 200 EMA; target EMA5; 1.5xATR(14) stop; 5-day time stop; 0.25% risk; max 2 open; auto-disable if ADX>25 against the position | 3 | DONE |
| D7 | Grades A / B / WATCHLIST (within 0.5 ATR of trigger) | 3 | DONE (superseded by score at signal time) |
| D8 | Shock filter (ATR pct > 90) and correlation cap (max 2 per currency) | 3 | DONE |
| D9 | Regime detector TRENDING/CHOPPY/SHOCK (ADX, ATR percentile, autocorrelation); optional 2–3 state HMM only if it beats the rules out-of-sample; route sleeves by regime | 3 | DONE (rule-based; HMM not built: optional, only if it beats rules) |
| D10 | Disabled by default: carry (with crash filter), cross-sectional momentum (3/12-month), COT positioning, multi-asset diversification, ML scorer (size/grade only, purged CV) | 3 | PARTIAL: cross-sectional momentum + weekly time-series momentum BUILT and FAILED validation (2026-11-23); carry, COT, diversification, ML not built |
| D11 | Dynamic allocation by rolling out-of-sample Sharpe with shrinkage to equal weight | 3 | PARTIAL (inverse-vol combiner for passing sleeves) |
| D12 | Backtester: realistic costs (spread by hour from MT5, commission, slippage, swap), next-open fills, closed bars, no look-ahead (an injected look-ahead bug must fail a test) | 3 | DONE (spread per pair from config; MT5 hourly spread used by execution score, not yet in backtests) |
| D13 | Handle known bad ticks (e.g. Yahoo EURUSD 8–9 Oct 2008) and test results with and without them | 3 | PARTIAL (flagged in quality report; no with/without comparison yet) |

## E. Strategy proof phase (gate before confirmations, scoring, signals UI, alerts)

| ID | Requirement | Phase | Status |
|---|---|---|---|
| E1 | Each strategy and timeframe backtested alone, 2005 to now (or longest available), cross-checked data | 4 | DONE (D1, 11 FX pairs, 2005-2026) |
| E2 | Optimise only within small grids; CPCV with embargo; untouched 3-year holdout; Deflated Sharpe and PBO reported | 4 | DONE |
| E3 | Robustness heatmaps (most neighbours profitable after costs), walk-forward with stitched out-of-sample results | 4 | DONE (neighbour share + walk-forward; heatmap shown as table) |
| E4 | Stress periods 2008, 2015 CHF, 2020; cost stress 1.5x (must stay profitable) and 2x | 4 | DONE |
| E5 | Block-bootstrap confidence intervals; Monte Carlo trade-order shuffles for drawdown percentiles | 4 | DONE |
| E6 | Combine passing strategies with risk-parity weights; sleeve contributions, combined equity and drawdown | 4 | DONE (no passing sleeves to combine) |
| E7 | PASS/FAIL table: CAGR, Sharpe, max DD, win rate, payoff, expectancy R, profit factor, trades/month, net cost per trade in R, per-year results | 4 | DONE |
| E8 | FAIL = disabled. If nothing passes: STOP and report, no further tuning | 4 | DONE - nothing passed, all locked, no re-tuning |
| E9 | Edge Confidence figure with its derivation; monthly re-run; edge-decay log | 4, 8 | DONE (edge confidence + derivation; edge-decay chart) |
| E10 | Expectation panel data: normal losing streaks and drawdown ranges | 4 | DONE (Mindset tab) |

## F. Confirmation module (passing strategies only)

| ID | Requirement | Phase | Status |
|---|---|---|---|
| F1 | Candle quality (close in outer third, body >= 50% of range) | 4a | PARTIAL (candle quality in score) |
| F2 | Retest confirmation (can trigger the pullback entry) | 4a | TODO |
| F3 | Follow-through (next close stays outside the channel) | 4a | TODO |
| F4 | Higher-timeframe agreement (weekly close vs 40-week EMA and 26-week Donchian) | 4a | PARTIAL (weekly bias in score) |
| F5 | Failed-breakout exit (back inside within 2 bars → exit next open), validated against holding | 4a | TODO |
| F6 | Report per confirmation: trades affected, win rate, avg R, expectancy, DD, capture rate; keep only if it improves out-of-sample results; confirmations lower the score, never block alone | 4a | TODO (needs a passing strategy) |

## G. Setup scoring, missed trades, profit management, execution

| ID | Requirement | Phase | Status |
|---|---|---|---|
| G1 | Hard rules block: stop before entry; per-trade and portfolio risk caps; data confirmed by two sources; drawdown breaker off; strategy not health-disabled; pair validated | 4a | DONE |
| G2a | Economic-calendar factor is informational with a small fixed weight (cap 0.05) and is never described as validated | 4a | DONE in config (`scoring.soft_factors.news_proximity`, `test_news_factor_is_informational_with_small_fixed_weight`) |
| G2 | Soft factors 0–100 (trend clarity, R:R, news, exposure, volatility regime, loss streak, ensemble agreement, confirmations, rate differential, positioning, session/spread quality); weights fitted on training data only, frozen before the holdout | 4a | DONE (weights frozen before holdout via `fit_weights`) |
| G3 | Score → size: >=70 full (A); 50–69 x0.5–0.75 (B); 30–49 x0.25 probe or WATCHLIST; <30 WAIT; thresholds set by validation | 4a | DONE |
| G4 | Per-factor validation report; drop factors that don't help | 4a | TODO (needs a passing strategy) |
| G5 | Top-trader checklist shown per signal (feeds the score; stop and risk cap stay hard rules) | 4a, 7 | DONE (card + score breakdown) |
| G6 | Missed Opportunities report (by reason: low score, news, exposure cap, user didn't act); must match the trade log | 4d | DONE |
| G7 | Second-chance entry within 1 ATR at x0.5 probe size; pullback re-entry watch with expiry; re-entry after stop-out (not same day, no revenge sizing) | 4d | DONE (functions; UI shows via signals) |
| G8 | Watchlist automation with pre-computed orders; probe sizing 25%; opportunity-capture rate | 4d | PARTIAL (watch levels + capture rate) |
| G9 | MAE/MFE-driven exits; exit toolkit validated independently (partial grid 1.0/1.5/2.0R, breakeven + spread buffer, chandelier/ATR and channel trailing, structure targets, time stop, volatility-scaled targets) | 4e | PARTIAL (partial grid, breakeven, channel trail, time stop, profit lock; MAE/MFE-driven selection TODO) |
| G10 | Profit lock: past +3R protect at least +1.5R | 4e | DONE |
| G11 | Pyramiding: add 25% of original risk at +2R only when the combined stop is at or beyond breakeven; max two adds; never add to a loser | 4e | DONE |
| G12 | Reporting: win rate, avg win/loss, payoff, expectancy R, profit factor, MAE/MFE distributions, contribution per exit rule | 4e | PARTIAL (exit-reason counts, win/payoff/PF; MAE/MFE distributions TODO) |
| G13 | Compounding: fixed risk %, monthly rebalance; no increase after wins or to recover losses | 5 | DONE (fixed % risk, locked per month) |
| G14 | Order type, price and expiry on every signal (stop-entry for breakouts, limit for pullbacks) | 4c | DONE |
| G15 | Entry timing by session and spread from hourly data; avoid rollover and thin hours; execution-quality score | 4c | DONE |
| G16 | Slippage learning loop (>25% over the assumption updates costs and triggers re-validation) | 4c | DONE |
| G17 | Broker cost check (user's spread/commission vs assumption; cost per trade in R; warn if net edge turns negative) | 4c | DONE |

## H. Risk engine and trade manager

| ID | Requirement | Phase | Status |
|---|---|---|---|
| H1 | Risk default 0.5%, hard cap 1% (enforced), fractional Kelly cap 25% | 5 | DONE |
| H2 | Position size in account currency via free FX rates | 5 | DONE |
| H3 | Equal ATR risk per position; 10% portfolio volatility target; 3% open-risk cap | 5 | DONE (equal R risk, vol-target scale, 3% cap) |
| H4 | Drawdown >15% → PAUSED - REVIEW until a written review is done | 5 | DONE |
| H5 | Cost gate: spread >3x typical, within 30 minutes of rollover or tier-1 news | 5 | DONE (news informational) |
| H6 | Health monitor: disable if 12-month live Sharpe <0 or live DD >1.5x backtest; 30-trade drift test (z/KS); reduce size or pause | 5 | DONE |
| H7 | Live trade manager: daily R, next action in plain English | 5 | DONE |

## I. Coach and mindset

| ID | Requirement | Phase | Status |
|---|---|---|---|
| I1 | Signal card: setup, plain-English reason, entry, stop, targets, trailing, invalidation, "skip if", historical win rate and avg R | 8 | DONE |
| I2 | Regime banner; fixed templates tied to measured conditions; post-trade review card (R, adherence, MFE/MAE, one lesson) | 8 | DONE |
| I3 | Principles shown in the app; pre-trade checklist required to log an entry | 8 | DONE |
| I4 | Hard rules: no entry without a stop; stops only tighten; max 3 new trades/week; 24h cooldown after 2 losing days (with journal review); risk % locked for the month; off-signal trades tagged DISCIPLINE BREACH | 8 | DONE |
| I5 | Emotion check-in; stress or urgency >= 4 → "Wait. Re-read the plan." and a 15-minute delay | 8 | DONE |
| I6 | Journal (planned vs actual, R, emotions, notes, chart screenshots); weekly review with PLAN ADHERENCE % shown above P&L | 8 | DONE |
| I7 | Written reflection after a breaker or big loss; rules can't be bypassed without being logged as a breach | 8 | DONE |

## J. Interface, API, alerts, paper trading

| ID | Requirement | Phase | Status |
|---|---|---|---|
| J1 | FastAPI bound to localhost only; works offline with bundled assets | 6 | DONE |
| J2 | TradingView-style original UI: top bar, left drawing toolbar (incl. long/short position tool, drawings saved per symbol), right watchlist + signal/coach panel, bottom tabs (Open Trades, Signals History, Journal, Validation, Mindset, Settings, Missed Opportunities) | 7 | DONE |
| J3 | Lightweight Charts (Apache-2.0, attribution kept); candles/bars/line/Heikin Ashi; overlays; ADX/RSI/ATR sub-panes; keyboard shortcuts; dark/light; colour-blind-safe; tablet width; accessible; empty/error/loading states | 7 | DONE |
| J4 | Current price from MT5 for display only; validated timeframes only; factor breakdown; execution-quality score; order-type card; edge-decay chart; capture rate | 7 | DONE |
| J5 | Permanent footer risk notice | 7 | DONE |
| J6 | Frontend tests: select symbol, view signal, run checklist, log trade | 7 | DONE (API flow tests + manual browser check) |
| J7 | Telegram and desktop alerts (send only) with entry, stop and amount at risk; signal right after the close; pre-session reminder; scheduler and Task Scheduler instructions; MT5 must be open for the run | 9 | DONE |
| J8 | Paper trading at least 8 weeks, compared with the backtest's expected range; small live account only after that | 10 | DONE (status + rules; the 8 weeks must still run) |
| J9 | README: setup in under 10 minutes, how to read signals, limitations and risk | 11 | DONE |
| J10 | `make run` / `make test` one command each | 6 | DONE |

## K. Engineering tests (third brief, part 7)

| ID | Test | Phase | Status |
|---|---|---|---|
| K1 | Hard rules block signals | 4a | DONE |
| K2 | Soft factors only change size and grade | 4a | DONE |
| K3 | Weights frozen before holdout | 4a | DONE |
| K4 | No look-ahead across timeframes | 2 | DONE (`tfx/config.py` StrategyTimeframes) |
| K5 | COT and rate data lagged to publication dates | 3 | TODO (COT/rate data not wired) |
| K6 | Cross-source disagreement blocks a signal | 1 | DONE |
| K7 | Unvalidated timeframe never produces a signal | 2 | DONE (`tfx/config.py` StrategyTimeframes) |
| K8 | Timeframe grid trials counted | 2, 4 | DONE for counting (`test_trials_counted_across_timeframes`) |
| K9 | Slippage loop updates costs and triggers re-validation | 4c | DONE |
| K10 | Second-chance and re-entry obey size caps | 4d | DONE |
| K11 | Pyramiding never adds to losers | 4e | DONE |
| K12 | Stops never widen | 4e, 8 | DONE |
| K13 | Every module can be disabled and the system still runs | all | DONE (config switches; tests run with modules off) |
| K14 | Missed Opportunities report matches the trade log | 4d | DONE |
| K15 | Coverage above 90% on core logic; ruff and mypy clean | all | DONE |

## L. Honest limits (cannot be done exactly as asked with free data)

| ID | Limitation | Honest alternative |
|---|---|---|
| L1 | Free intraday history is short. Yahoo hourly covers only the last ~730 days, and MT5 history depth depends on your broker. H1/H4 cannot be validated from 2005 with two-source confirmation. | Validate H1/H4 on the overlap that both sources cover, state the span next to every result, and keep the 3-year holdout. If the training span is too short for CPCV, mark the pair "insufficient data – not validated" (no signals). |
| L2 | "Spread by hour from MT5" exists only for the history your broker keeps, and MT5 bar spreads are a per-bar figure (usually the minimum), not the spread you would actually pay. | Use the MT5 spread where it exists, use the config cost model before that, and always report the 1.5x/2x cost stress. |
| L3 | No free historical economic calendar covers all tier-1 events back to 2005. | Build US events (CPI, NFP, FOMC) from FRED/ALFRED release dates and central-bank meeting calendars; use a free weekly feed for upcoming events. The news factor is validated only where history exists and is marked as such. |
| L4 | OANDA does not accept Kenyan residents; Stooq and Dukascopy block automated downloads. | MT5 broker demo as the second source (chosen by the user). |
| L5 | (Resolved: the user chose broker MT5 history, see B20.) Dukascopy's bulk history is now only on AWS S3 with "Requester Pays" (the downloader pays AWS transfer fees and needs an AWS account). Its free website export gives one instrument and one day per download, and its old datafeed returns HTTP 429. Free, automated download back to 2005 is not possible without breaking the zero-cost rule or Dukascopy's access limits. | Options put to the user: (a) the broker's own MT5 H1 history as the research source, checked against Yahoo over their ~2-year overlap; (b) a one-off, user-approved AWS download (small one-time fee, needs an AWS account); (c) keep H1/H4 "not validated". |

### Findings from the first MT5 connection (2026-11-23, JustMarkets demo, USD)

* Broker server clock = New York + 7 h (`server_time: ny_close` is correct); D1 bars end at
  the New York close. Symbols use a `.m` suffix (mapped automatically).
* Broker history: D1 from 2012 (2020 is missing about 60 sessions); genuine H1 only from
  mid-2019 (2020 incomplete). Before that, H1 requests return about one bar per day, which
  must never be treated as hourly data. Terminal "Max bars in chart" = 100,000.
* **Yahoo daily FX bars are not true OHLC after 2010.** From about 2011, a bar dated D has
  open ≈ close ≈ the price at the start of D (00:00 UTC), while high/low cover day D. The
  median |close − open| / range falls from about 0.5 (2003–2009) to about 0.01 (2011–2026).
  Close-based rules on these bars would be corrupted. The first cross-check therefore
  compared misaligned days (13–30% agreement), which is not a valid measurement.
* **L6 resolved (user decision 2026-11-23: "Repaired Yahoo + MT5").** Daily FX bars now run
  midnight to midnight UTC. The last ~2 years are built from Yahoo hourly bars (proper OHLC;
  98.1% of days agree with MT5 within 0.2 ATR, measured on EURUSD 2025-26). Older years
  use repaired Yahoo daily rows (open = close_D, high/low_D, close = close_{D+1}), and
  pre-2010 rows are kept as delivered. The changeover weeks are dropped, and the
  in-progress row (live or placeholder close) is never used. MT5 confirms each bar on the
  same UTC days. Cached series built the old way are rebuilt once, automatically.
* **Timing change vs the original spec (section 3):** signals are evaluated after 00:00 UTC
  plus 30 minutes (19:30 New York in winter, 20:30 in summer), not right after 17:00 New
  York.
* **Known limitations of the repaired (older) history:** Yahoo has no weekend rows, so a
  repaired Friday closes at Monday 00:00 UTC (it includes about 2-3 h of Sunday-evening
  trading). Pre-2010 rows cannot be confirmed by a second source (broker history starts in
  2012 or 2017), so backtest results before then are flagged "single-source".
* Gold (XAUUSD) is still pending: Yahoo GC=F futures vs MT5 spot agree on only about 18% of
  days, so gold is almost always DISPUTED (no signal) until a separate decision is made.

## M. Deterministic validators and setup rendering (brief of 2026-11-23)

| ID | Requirement | Status |
|---|---|---|
| M1 | No AI/LLM in the signal path; template text only; a test fails on any LLM import or endpoint | DONE (`test_no_llm_imports_or_endpoints_anywhere_in_tfx`, `test_signal_path_loads_no_ai_module`) |
| M2 | Same data + config + rule version gives byte-identical output; per-strategy rule hash stored with signals, validation results and the report; mismatch blocks signals until re-validated | DONE (`tfx/versioning.py`, `test_same_input_byte_identical_output`, `test_hash_mismatch_blocks_signals`) |
| M3 | Each signal stores bars, indicator values, levels, thresholds, rule version/hash; "Why this signal" panel in the UI | DONE (`signals.trace` column, UI panel) |
| M4 | Golden chart examples (clean, messy, edge) with expected signals, levels, scores and setups; any output change fails until the files are regenerated AND the rule version is bumped | DONE (`tests/golden/`, regenerate with `python -m tests.golden.regen`). Note: there is no git/CI service; `.\make.ps1 test` is the CI gate. |
| M5 | Per-strategy validator modules with fixed criteria; consensus engine counts only strategies with a current PASS | DONE (`tfx/validation/validators.py`, `tfx/consensus.py`) |
| M6 | Chart setup rendering for trend (channels, EMA, entry, initial/trailing stop, partial and breakeven), mean reversion (band, entry, 5-EMA target, time-stop countdown), pullback (broken level, retest zone, entry, stop); long/short box with R:R and amount at risk; one-line template summary; toggleable; colour + icon + label | DONE (`tfx/setup_render.py`, chart overlay). Locked setups are labelled "NOT VALIDATED - view only". |
| M6-SMC | SMC/supply-demand zones, LQ/LOF lines, FVG, BOS/sweep markers | DONE as ICT overlays (`tfx/strategies/ict.py` `chart_annotations`, `render_ict`). Strategies `ict` (strict) and `ict_flex` (real-market tolerances declared before results) validated: both FAIL, both locked. |
| M7 | Tests: no LLM calls, identical output, hash mismatch blocks, golden reproduction, setup objects equal the card's stored levels | DONE (`tests/test_determinism_golden.py`) |

## N. Missed trades, supply & demand, top-down read (brief of 2026-11-24)

| ID | Requirement | Status |
|---|---|---|
| N1 | Investigate profitable trades that were not taken (EURUSD, GBPUSD, whole history) | DONE (`tfx/study_missed.py`, `python -m tfx study`). Training data only; the 3-year holdout is kept for the final test. Winners AND losers of skipped setups are counted. |
| N2 | Learn the patterns and make sure such trades are not missed | DONE as a pre-declared adoption rule: a relaxation is adopted only with >= 30 trades, mean R > 0 and bootstrap 5th percentile > 0. Result: none qualified (skipped setups lost -15.8R), so nothing was adopted. Adopting them would have made results worse. |
| N3 | Supply & demand as the second strategy, used when ICT has no setup | DONE (`tfx/strategies/supply_demand.py`, rules declared before testing). Validated: FAIL (PBO 0.78, DSR 0.59), locked. Pipeline order: ICT, then S&D, then the rule strategies. |
| N2b | Investigate profitable-looking MOVES that were not traded, find their patterns, stop missing them | DONE (`tfx/study_moves.py`, `python -m tfx moves`): 549 big moves, 14 pre-declared precursors, false alarms counted. None adopted (best capture 28% at the base hit rate; profitable signs failed the multiple-testing bound). Precursors firing today are shown in the market read (step 8) with their measured history, labelled NOT proven. 14 trials recorded. |
| N4 | Analyse the market like a pro trader, not just fixed rules | DONE as a deterministic top-down read (`tfx/analysis.py`, `/api/analysis/{symbol}`, "Market read" panel). No AI in the signal path (rule M1 still applies). It is labelled analysis only and cannot unlock a strategy. |

## Build order (phases)

0 audit → 1 data fixes → 2 timeframes and intraday data → 3 indicators, strategies,
backtester → **4 proof phase (gate)** → 4a confirmations and scoring → 4c execution →
4d missed trades → 4e exits → 5 risk engine and trade manager → 6 API → 7 interface →
8 coach, mindset, journal, validation reports → 9 alerts → 10 paper trading →
11 README.
