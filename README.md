# TrillionaireFX

A free, local, rules-based FX signal and trade-coaching app with a charting interface.
It analyses 11 currency pairs and gold once a day, says in plain English when a
**validated** strategy wants to enter or exit, coaches you through each trade, and
enforces trading discipline.

> **Educational decision support. Trading carries risk of loss. Past performance does not
> guarantee future results.** TrillionaireFX never places trades and holds no broker
> credentials. Nothing in it promises profit, constant wins or a fixed accuracy.

## Current status (read this first)

| Area | Status |
|---|---|
| Data (Yahoo + your MT5 broker, cross-checked) | Working. 11 FX pairs confirmed daily, 95-98% source agreement |
| Charting app, watchlist, coach, journal, mindset rules, alerts, paper trading | Working |
| **Strategy proof phase (2005-2026, real costs)** | **All 8 strategies FAILED** (trend, pullback, mean reversion, cross-sectional momentum, weekly time-series momentum, ICT strict, ICT tolerant, supply & demand; 78 trials counted). Supply & demand was the most robust (1,224 trades, every grid setting profitable, +0.10R per trade on the holdout, profitable at 2x costs) but failed the overfitting tests (PBO 0.78, Deflated Sharpe 0.59) and its walk-forward was flat. The first five lost money on the untouched last 3 years. ICT strict was the best (+0.18R per trade, profitable on the holdout and at 2x costs) but had only 73 trades and failed the overfitting tests (PBO 0.37, Deflated Sharpe 0.21). ICT tolerant (+0.05R) failed robustness. |
| ICT / SMC chart overlay | Working on every timeframe (liquidity lines, FVGs, order blocks, sweeps, MSS). Display only. |
| Strategy order | ICT first; if ICT has no setup, supply & demand; then the older rule strategies. Each is shown LOCKED (view only) until it passes. |
| Missed-trade study (EURUSD, GBPUSD) | Done (`python -m tfx study`). The 172 setups the strict ICT rules skipped would have made 24 winners and 43 losers (-15.8R). No relaxation met the pre-declared rule, so none was adopted. See `docs/MISSED_TRADES_STUDY.md`. |
| Move study (EURUSD, GBPUSD) | Done (`python -m tfx moves`). Of 14 warning signs tested on 549 big moves, none predicted them well enough to profit after false alarms. Signs that fire today are shown in the market read, labelled NOT proven. |
| Market read (top-down) | Working: bias, structure, premium/discount, liquidity, fresh zones, FVGs, volatility and a written plan for every pair. Analysis only, never a signal. |
| Live BUY/SELL signals | **None.** Every strategy is locked until one passes validation. The app shows WAIT with the reasons. |
| Gold (XAUUSD) | Disputed data (Yahoo futures vs MT5 spot); excluded until a separate decision |
| H1/H4 strategies | Not validated: confirmed intraday history is too short |

The strategies are **not** re-tuned to force a pass. See "How reliable is this?" in the
app, or `var/validation_report.json`.

Re-validate only new strategies (earlier results kept, nothing re-counted):
`python -m tfx validate --only supply_demand`.

## Setup (under 10 minutes, Windows)

1. Install **Python 3.11+** (python.org; tick "Add to PATH").
2. Install **MetaTrader 5** and sign in to a free **demo** account at your broker
   (File > Login to Trade Account). Set Tools > Options > Charts > "Max bars in chart"
   to Unlimited. Keep MT5 open while TrillionaireFX runs.
3. In PowerShell, in the project folder:

```powershell
.\make.ps1 setup
```
```powershell
.\make.ps1 fetch
```
```powershell
.\make.ps1 run
```

4. Open **http://127.0.0.1:8765** in your browser. The app only listens on your own
   computer (localhost).

Optional: copy `.env.example` to `.env` and add `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` (free bot from @BotFather) for Telegram alerts. Desktop
notifications work without setup.

### Commands

| Command | What it does |
|---|---|
| `.\make.ps1 run` | Start the web app + daily scheduler |
| `.\make.ps1 daily` | One daily evaluation + alerts (for Task Scheduler) |
| `.\make.ps1 fetch` / `status` | Refresh / show data freshness |
| `.\make.ps1 quality` | Data-quality report per symbol |
| `.\make.ps1 validate` | Run the strategy proof phase (validation gate) |
| `.\make.ps1 research` | Download + check MT5 hourly research history |
| `.\make.ps1 replay` | Hypothetical replay of last week from $100 (`python -m tfx replay --start 2026-11-16 --end 2026-11-20 --balance 100`) |
| `.\make.ps1 test` | Full test suite with coverage |

**Daily schedule.** With `run` open, evaluation happens at 00:35 UTC (03:35 in Kenya)
after each daily bar completes, plus a 06:30 UTC pending-order reminder. If you do not
leave the app running, add a Windows Task Scheduler job that runs
`.venv\Scripts\python.exe -m tfx daily` at 03:35 local time (MT5 must be open).

## How to read the app

* **Watchlist chips**: ▲ BUY, ▼ SELL, ■ EXIT, ● HOLD, ◎ WATCH, – WAIT. A 🔒 means a
  strategy fired but is not validated, so the answer is WAIT. Colours are colour-blind-safe
  and always come with an icon and a word.
* **Signal and Coach panel**: the headline ("BUY EURUSD at next open"), why it triggered,
  entry, stop, position size and the exact amount at risk for your balance and risk %,
  the order type, price and expiry, the playbook (partial at +1R, breakeven, trailing), the
  invalidation level, "skip if" checks, the setup score with a factor-by-factor breakdown,
  and the setup's historical results (labelled as history, not a forecast).
* **Hard rules block a signal** (no stop, risk above 1% or the Kelly cap, open risk above
  3%, more than 2 trades in one currency, drawdown breaker, unconfirmed data, strategy
  not validated or switched off by the health monitor, wide spread or rollover).
  **Soft factors only change size and grade.**
* **Logging a trade**: emotion check-in (stress or urgency 4+ means "Wait. Re-read the
  plan." and a 15-minute pause), then the pre-trade checklist. A stop is always required.
  Stops can only be tightened. Off-signal trades and overrides are logged as DISCIPLINE
  BREACH.
* **Bottom tabs**: Open Trades, Signals History, Journal (plan adherence is shown above
  P&L), How reliable is this? (validation results and the edge-decay chart), Mindset,
  Missed Opportunities (with capture rate), Execution (execution quality, your cost per
  trade in R), Paper Trading, Data Quality, Settings.
* **Timeframes**: 1m 5m 15m 30m 1H 4H D W M. Intraday candles come live from your MT5
  terminal (updated every second, display only). Only D can carry signals, and only for
  validated strategies. Every other timeframe is labelled "view only".
* **Replay tab**: what the rules would have done over any past window from any starting
  balance (hypothetical). It flags trades below the 0.01-lot broker minimum: with $100,
  most trades would force 5-15% risk.
* **Chart**: candles/bars/line/Heikin Ashi; Donchian 20/55/100, EMA 20/100/200, ATR stop,
  regime strip, signal and watch levels, trade markers; ADX, RSI(2) and ATR panes; drawing
  tools saved per symbol. Shortcuts: Alt+H/T/R/M/N/L/S/E, Esc, +/-, ? for help.

## Determinism and traceability

* No AI or language model is used anywhere in signals, scores, validation or explanations.
  All on-screen text comes from fixed templates filled with measured values.
* Each strategy has a rule version and a rule hash (its code + config). Signals are
  blocked if the live hash differs from the hash that passed validation.
* Every signal stores the bars, indicator values, levels and thresholds that produced it
  (the "Why this signal" panel).
* Golden chart examples in `tests/golden/` must reproduce exactly. Intentional rule changes
  need `python -m tests.golden.regen`, a version bump in `tfx/versioning.py`, and a fresh
  validation run.

## How the data works

* Daily bars run midnight to midnight UTC. Yahoo's own daily FX bars are not true OHLC
  since about 2011, so the last ~2 years are built from Yahoo hourly bars and older years
  are repaired (see `docs/REQUIREMENTS.md`, L6).
* Every bar must be confirmed by your MT5 broker within 0.2 ATR. Otherwise the app shows
  "DATA UNAVAILABLE - NO SIGNAL". Stale data is never shown as current.
* Every provider response is kept unchanged in an append-only log in `var/tfx.sqlite`.

## Limitations and risk

* **No strategy currently has a validated edge.** The app is complete, but it will only
  show BUY/SELL after a strategy passes the gate: PBO < 0.05, Deflated Sharpe probability
  >= 0.90, profitable on the untouched 3-year holdout, >= 70% of neighbouring settings
  profitable, and profitable at 1.5x costs.
* Win rate on its own means little. A high win rate can hide large losses. Judge
  expectancy in R, drawdown and the holdout instead.
* Costs are assumptions (0.8-pip majors, 0.7-pip commission, 0.2-pip slippage, a carry
  approximation). Real swaps vary. Enter your real spread in Settings; the app shows your
  cost per trade in R.
* Older history is single-source before your broker's history starts (2012, or 2017
  for most pairs). Repaired Yahoo Friday bars include Sunday-evening trading.
* No free historical economic calendar exists, so the news factor is informational
  only, with a small fixed weight.
* Paper trading must run at least 8 weeks and stay inside the backtest's expected range
  before the app calls anything "ready for small live use". Even then: a small account
  first.

## Project layout

`tfx/data` data layer · `tfx/strategies` indicators and rules · `tfx/backtest` engine ·
`tfx/validation` statistics, proof phase, registry · `tfx/risk` sizing, caps, trade manager ·
`tfx/scoring.py` setup score · `tfx/execution.py` order guidance · `tfx/missed.py` ·
`tfx/mindset.py` · `tfx/coach.py` · `tfx/paper.py` · `tfx/alerts` · `tfx/api` · `web/` UI ·
`tests/` · `config.yaml` (every parameter) · `docs/REQUIREMENTS.md` (every requirement and
its status).

Charts by TradingView Lightweight Charts™ (Apache-2.0, see `web/vendor/`).
