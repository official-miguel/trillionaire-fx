# Missed-trade study (EURUSD, GBPUSD) – 2026-11-24

**Question.** Is TrillionaireFX skipping trades that a person would have taken and
that would have made money? If so, can it learn from them?

**How it was tested** (rules fixed before the study ran; `tfx/study_missed.py`):

- Daily bars, 2005-01-01 to 2023-11-20. The last 3 years (the holdout) were **not**
  used, so they stay a fair final test for anything this study suggests.
- Every near-miss ICT setup was collected with deliberately loose rules, then traded
  on its own with the same limit-order execution and real costs.
- Each skipped setup is tagged with the strict rule it broke: the sweep did not close
  back inside, the gap was only a near-gap, or the entry was not in discount/premium.
- **Winners and losers are both counted.** Picking only the trades that worked out
  is hindsight: every chart looks obvious afterwards, and a system built that way
  loses money live.
- Adoption rule: a relaxation is only adopted if its skipped trades number >= 30,
  have a positive mean R after costs, and stay positive at the 5th percentile of a
  bootstrap.

## Results

| Group | Setups | Filled | Wins | Losses | Mean R | Total R |
|---|---|---|---|---|---|---|
| Taken by strict ICT | 63 | 20 | 8 | 12 | -0.09 | -1.9 |
| **Skipped (all)** | 172 | 67 | 24 | 43 | **-0.24** | **-15.8** |
| Skipped: only the sweep close | 17 | 3 | 2 | 1 | +0.71 | +2.1 |
| Skipped: only the near-gap | 9 | 2 | 0 | 2 | -0.92 | -1.8 |
| Skipped: only discount/premium | 52 | 21 | 10 | 11 | -0.07 | -1.5 |
| Skipped: 2+ rules broken | 94 | 41 | 12 | 29 | -0.36 | -14.6 |

**No relaxation was adopted.** The skipped trades lost money as a group. The one group
with a positive result had only 3 trades, which is far too few to learn from.

**Big moves.** 573 times, price ran 3 ATR or more within 10 days without first
moving 1 ATR against (11% of all days). Strict ICT had a setup before 9 of them (2%),
loose ICT before 29 (5%), and supply & demand before 21 (4%). Moves like this are easy
to see afterwards and hard to see beforehand. Most of them had no clear pattern in
front of them, so they cannot be "not missed" without also taking many losers.

## What this means

The strict rules are not throwing away good trades. Taking the setups they skip
would have made results worse. Re-run any time with `python -m tfx study`; the full
output is written to `var/missed_study.json`.

# Move study: starting from the moves (2026-11-24)

**Question.** Forget the strategies: take every big move in history, look at what the
chart showed at the close before it started, and check whether those warning signs
could have caught the moves.

**How it was tested** (rules fixed before the run; `tfx/study_moves.py`,
`tfx/strategies/precursors.py`, `python -m tfx moves`). Same data and holdout as above.
There were 14 warning signs: squeeze breakout, NR7 breakout, 20-day breakout, liquidity
sweep, momentum candle, buying the dip in a trend, and first touch of a supply/demand
zone, each also tested only in the direction of the weekly trend. Each sign was
measured three ways:

1. how many of the 549 big moves it fired before;
2. how often a big move actually followed when it fired, against 13.8% for any day;
3. trading it: next open, 1 ATR stop, 3R target, 10-day time stop, real costs.

A sign is adopted only with >= 100 trades, profit after costs, profit in BOTH halves
(2005-2014 and 2014-2023), and a bootstrap lower bound above 0 after adjusting for
testing 14 signs at once. All 14 are counted as trials.

| Warning sign | Fired | Moves caught | Move followed | Trades | Mean R | Both halves + |
|---|---|---|---|---|---|---|
| NR7 breakout | 960 | **151 (28%)** | 13.8% | 646 | -0.05 | no |
| 20-day breakout | 1452 | 128 | 15.5% | 608 | -0.07 | no |
| Buy the dip in a trend | 2466 | 129 | 14.0% | 598 | +0.06 | yes |
| Liquidity sweep | 956 | 120 | 13.3% | 507 | +0.05 | yes |
| Squeeze breakout | 462 | 55 | 15.4% | 251 | +0.08 | yes |
| Momentum candle | 261 | 47 | 16.5% | 223 | +0.06 | yes |
| Momentum candle + weekly trend | 123 | 27 | 19.5% | 111 | +0.21 | yes |
| Buy the dip + weekly trend | 1462 | 99 | 15.5% | 412 | +0.09 | yes |
| S&D first touch (+ weekly) | 50 | 12 | 22.0% | 48 | +0.52 | yes (too few) |

**Nothing was adopted.** The signs that fire before the most moves (NR7, 20-day
breakouts) fire so often that a move follows only about as often as on a random day,
and trading them lost money. A few signs made a small profit in both halves, but with
this few trades none is distinguishable from luck once 14 signs are tested.

**What the system does with them now.** The market read (step 8) lists the warning
signs that fired on the last closed bar, with these measured numbers and the label
"NOT proven". You are not left unaware of a setup, and nothing unproven is turned
into a signal.

The leaders (momentum candle + weekly trend, squeeze breakout + weekly trend, S&D first
touch) need more data before they can be judged. The fair test is the 3-year holdout,
and it can only be used once.
