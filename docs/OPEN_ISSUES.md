# Open issues

Known, unresolved, and deliberately not fixed yet. Spans both repos:
`orb-live-trading` (live) and `../BacktestingGaps` (backtest/research).

Last reviewed 2026-08-21 (D1 rewritten after measurement; D1b added and fixed). Add to this file rather than relying on a session
transcript — everything below has already been lost to a crash once.

---

## Live defects

### L1. `RollingIndicators` are never seeded — every symbol, every session
`_run_post_orb` reads `self._cache.get_bars(symbol)`, but the market-data
subscription that fills that cache is deliberately deferred until *after* the
loop (subscribing all ~59 symbols at open exceeds IB's line cap and makes IB
silently deliver nothing). So `raw.empty` is always true there and every symbol
takes the `continue` branch before `indicator.seed_from_orb_bars` runs.

This is the same bug that hid the margin gate for months. `bacb27e` moved
`_prewarm_candidate_margin` out from behind it; the indicator seeding was left.

v1 does not use EMA confirmation (`require_ema_confirmation=False`), so this is
*probably* inert — but nobody has confirmed it, and `tp3_mode` is still
`ema_crossback` in `StrategyConfig`.

**Diagnostic**: `phase2_ready` never appears in any session log.

### L2. Unexplained `entry_rejected_or_unfilled`
2026-08-19: four consecutive rejections (BRZU, NUGT, JNUG, LABU) with no reason
logged. These are conventional 2x/3x names at measured rates 0.52–0.78, so
margin cannot be the cause. 2026-08-20 had three of the same plus one genuine
`entry_rejected_insufficient_buying_power` (SOLT).

Cause unknown. Until it is found, any backtest fill assumption is optimistic:
`entry_at_boundary=True` assumes every breakout fills at the ORB boundary.

### L3. `pattern_day_trader` is hardcoded `False`
`ib_client.get_account()` returns a literal `False` rather than reading the flag
from IB. Nothing currently consumes it, so this is latent — but the account's
PDT status is genuinely unknown to the system, and it matters for L4.

### L4. Unknown whether IB grants a lower intraday margin requirement
`BuyingPower` reads 4.00x equity, but that is 4x **AvailableFunds**, which
positions deplete — so it does not raise the ceiling. `whatIfOrder` returns full
Reg-T initial margin (TQQQ 0.79, not ~0.20), and live saturates at 3–4 positions,
consistent with a 1x-equity margin ceiling.

If IB *does* apply a lower requirement intraday for a PDT-flagged account and
`whatIfOrder` simply is not showing it, capacity roughly quadruples and most of
the margin analysis needs redoing at a much looser constraint. **Worth asking IB
directly.** This is the single highest-leverage open question.

### L5. `allow_partial_entries` defaults `True` and has never run live
New in `206342c`. Watch for `entry_partially_sized`. Backtest partials book P&L
linearly in size while paying a full spread and the commission minimum, so they
are flattered slightly (77 of 2,002 trades, ~4%).

### L6. BOIL, GUSH, KOLD have no `master_universe.csv` row
They are `FORCE_INCLUDE` entries defined in `v1_strategy.py`, so they cannot
carry measured margin columns. They fall back to a conservative 1.0 until
`prewarm_margin` measures them live each session — which over-reserves in the
window before that.

---

## Data quality

### D1. The cache stops covering the universe after 2026-05-20 (was: "AV/IBKR seam")
Measured 2026-08-21; this entry previously described a provenance seam at
2026-06-16. That framing was wrong. There is no clean seam — there is missing
data, and a backfill that replaced it selectively.

**Intraday cache**, universe symbols with any bars on a given day:

| period | symbols/day |
|---|---|
| through 2026-05-20 | 130-175 |
| 2026-05-21 .. 05-27 | 89 -> 43 |
| **2026-05-28 .. 06-15** | **0 — 12 consecutive trading days, no data at all** |
| 2026-06-16 onward | 2-32 |

By month: 176 symbols in 2026-05, **32** in 2026-06, 31 in 2026-07, 20 in
2026-08. 149 of 184 universe symbols have no intraday file after 2026-05.

The post-06-16 bars exist because `pull_ibkr_last_2mo.py` backfilled them — but
it only fetched `(symbol, date)` pairs that *could plausibly fire*, and merged
them into the existing files. So recent coverage is **conditioned on the signal
being measured**: the days a symbol gapped are present, the days it did not are
absent. That is not a thinner sample of the same population, it is a different
population.

**Consequence — measured, not assumed.** The first guess was that candidates
would be left with no intraday bars and so count as guaranteed misses. That was
checked and is **false**: 0 of 4,683 candidates in the traded span lack intraday
bars. Candidates and bars narrowed together, so P(fire) is not contaminated that
way.

What actually happens is a **composition shift plus a within-class inflation**:

| window | C1 crypto | C2 | C3 | candidates | symbols | days |
|---|---|---|---|---|---|---|
| pre-2026 | 12.8% | 43.2% | 44.1% | 4,132 | 54 | 1,276 |
| 2026-01..05 | 32.1% | 44.0% | 23.9% | 461 | 40 | 94 |
| 2026-06+ | 35.6% | 45.6% | 18.9% | **90** | **18** | 33 |

C3's underlyings are the ones that went stale (D1b), so C3 candidates fall away
and crypto — whose underlyings never went stale — goes from an eighth of the
candidate set to over a third. C1 has the highest P(fire), so the pooled number
drifts up: **0.432 pooled, 0.505 in 2026, 0.567 since 2026-06-01.**

And P(fire) is up *within every class* in the recent window (C1 .594, C2 .537,
C3 .588 vs .419/.465/.381 pre-2026) — that part is the selection-conditioned
backfill, since only plausibly-firing symbol-days were ever fetched.

So the recent window is **not** a like-for-like continuation of the sample: it is
thin (90 candidates over 33 days from 18 symbols), crypto-weighted, and biased
toward firing. Anything read off it — the 0.43 `reserve` default in
`run_allocation_policies.py`, per-year side preference (R3), and especially the
2026 column of any per-year table — should be treated as unreliable until the
caches are rebuilt. It also means recent results are crypto-dominated, which
makes R1 more urgent, not less.

**Options, since IB cannot backfill this.** 1-min history is pacing-limited to
roughly 60 requests per 10 minutes and capped around 6 months, so 184 symbols x
~65 trading days (~12,000 requests) is over a day of continuous pulling and
partly outside the window anyway. That leaves:

1. Keep backfilling only candidate symbol-days — cheap, but preserves exactly
   the selection conditioning that makes the window untrustworthy.
2. **Truncate the backtest sample at 2026-05-20** and stop treating the last
   three months as comparable. Costs 3 months of a 6.5-year sample, buys back a
   clean population. Needs an end-date bound on `run_v1_at_k`, which has no such
   parameter today.

Recommended: (2). Not implemented — capping the sample is a research-scope
decision, not a bug fix.

**Not yet done**: the original provenance question (do AV and IBKR bars actually
disagree on the 09:30-10:00 ORB high/low?) is still unanswered.
`compare_av_vs_ibkr_bars.py` is written and ready but has never been run to a
recorded verdict. Fractional-volume fingerprinting was tried as a cheap proxy
and **does not discriminate** — 2026-02 files sit at 5-26% fractional, mid-range
— so the question needs the actual IB comparison.

### D1b. The daily underlying cache stopped updating for 41 of 71 underlyings
Root cause found and fixed 2026-08-21. `scripts/refresh_daily_ul_cache.py` had a
hardcoded `EQUITY_ULS` list of **14** symbols while `master_universe.csv`
references **71** underlyings, so everything outside that list froze wherever the
last bulk build left it: 9 at 2026-04-29, 2 at 2026-05-06, 11 at 2026-05-13,
19 at 2026-05-15. Only 30 of 71 ran to 2026-08-18.

Those 41 stale underlyings feed **56 of 184** ETFs. By class the damage is
concentrated in **C3 (37.6% of rows stale)**, not C2 (10.3%); C1 crypto is
entirely clean. It is therefore *not* an explanation for R2.

It went unnoticed for months because of a **second, independent bug that hid
it**: at `--lookback 250` the script asked IB for `durationStr="500 D"`, and IB
answers an over-365 "D" duration with a *timeout rather than an error*. Every
symbol returned an empty frame, `merge_and_write` added 0 rows, and nothing in
the logs looked wrong. So the list was both too short and never actually
fetching. Fixed in `ib_client.get_daily_bars` (switch to a "Y" duration past
365 days) with a parametrized regression test.

Fixed by deriving the list from `master_universe.csv` (ticker-shaped entries
only — several `underlying` cells hold index names, see D6). Adding an ETF to
the universe now refreshes its underlying automatically.

**Result of the repair run** (2026-08-21, 250-day lookback): 11,221 new daily
rows across 89 equity underlyings; 94 of 106 cached underlyings now current
through 2026-08-21.

Still failing, both minor:
- **BRK-B** — not a valid IB symbol; IB wants `BRK B` (space, not hyphen).
- **VIX** — `IB returned 0 rows`. It is an index, not a `Stock` contract, so it
  needs a different contract type or a non-IB source.

Seven underlyings remain at May dates (GME, IEF, MELI, SMH, SNOW, TLT, VWO).
These are **orphans**: present in the daily cache but no longer referenced by
`master_universe.csv`, so the derived list correctly skips them. Harmless, but
they should be deleted rather than left looking stale.

**The intraday cache is NOT repaired by this, and cannot be widened by it.**
Checked after the repair run: the candidate set is unchanged (4,676 vs 4,683
candidates, still 100% intraday-covered, identical P(fire) by year). The reason
is structural — `etf_basis.compute_candidates_etf_basis` derives every gap from
the *intraday* ETF cache and returns an empty frame for a symbol that has none.
The daily UL cache feeds only the PS filter (`_ps_passes`). So intraday
coverage is the binding constraint on D1, and repairing daily can neither add
candidates nor strand existing ones.

The daily repair still matters for correctness: with 41 underlyings frozen at
May prices, `_ps_passes` was testing recent candidates against stale underlying
closes. The effect is small but real — 7 candidates changed status. Cached trade parquets (`scratchpad/_priority_trades.parquet`,
`_unpruned_trades.parquet`) were built against the *old* daily cache and are now
stale — regenerate them before comparing any number to one produced before
2026-08-21.

### D2. Split artifacts in the intraday ETF cache
> **Partly misattributed.** Four of the seven "recurring offenders" below
> (SQQQ, SOXS, DUST, FNGD) turned out to be the D7 timestamp shift, not splits.
> Re-derive this list after the D7 repair.
Guarded, not fixed. `apply_split_correction.py` covered the *daily underlying*
cache; the intraday ETF cache still stores raw prices, so a split leaves the
prior close on one side of the ratio and the 09:30 print on the other — SOXS
2020-08-28 reads as +13,813%. `etf_basis.MAX_PLAUSIBLE_GAP = 0.50` discards 31
in-sample sessions of 72,542 rather than repairing them. Recurring offenders:
SQQQ, SOXS, BOIL, DUST, FNGD, SPXS, LABD.

### D3. Cache holes fabricate gaps
Also guarded, not fixed. `shift(1)` pairs a session with whichever one precedes
it *in the file*, so a week of ordinary drift collapses into one "overnight"
move (GDXU 2026-08-07 read as +21.3%). `MAX_PRIOR_GAP_DAYS = 5` rejects those
pairs. Live never sees this — it reads a real prior daily close from the broker.

### D4. Wrong underlying mappings
Measured by daily log-return regression (`scripts/check_underlying_mapping.py`):

| ETF | current UL | R² | true index | R² |
|---|---|---|---|---|
| WEBL | XLC | 0.728 | **FDN** | **0.984** |
| FNGU | QQQ | 0.843 | FANG+ basket | 0.885 |
| FNGD | QQQ | 0.834 | FANG+ basket | 0.854 |

WEBL is decisive and fixable, but needs FDN added as an underlying with its own
daily cache and sigma. FNGU/FNGD track NYSE FANG+, which has no free ticker —
may have to be accepted and documented as a known bias.

### D5. FNGU sigma rests on ~18 months
yfinance has no FNGU history before 2025-02-20 (ETN redemption/relaunch). If the
daily cache was rebuilt from yfinance, its σ — and therefore its PS-filter
threshold — is computed on a short and unrepresentative window.

### D6. 25 underlyings have unresolved sigma
`BacktestingGaps/sigma_unresolved.csv` (2026-08-14). Mostly index names that are
not tickers — "AI and Big Data", "Dow Jones Transportation Average" — plus real
ones like BRK.B. Reason recorded as "Insufficient data". Unreviewed since the
split-correction work; unclear which are benign naming artifacts and which are
tradable instruments silently missing a filter.

---

### D7. Five symbols have 4-hour-shifted timestamps across their whole history
Found 2026-08-21 while running the D1 provenance comparison — which is how the
provenance question earned its keep, even though the answer to the question as
posed was "the values agree".

**SQQQ, SOXS, DUST, UVXY, BOIL, FNGD, UVIX, SBIT.** Their cached bars were
written with naive ET timestamps labelled UTC. Read back as UTC and converted to
ET they land 4h early under EDT, 5h under EST: a session that really ran
09:30-15:59 is stored as 05:30-11:59. **21,936 sessions across 1,077 files.**

The first scan of this reported only five symbols. It was wrong twice over: it
iterated `master_universe.csv`, which has no row for **BOIL** or **UVXY** (see
L6), and it required >50% of a symbol's sessions to be shifted, which hid
**SBIT** at 38 partially-shifted sessions. Scan the cache directory, not the
universe, and do not threshold on prevalence.

The prices are correct — DUST's cached 05:30 bar is byte-identical to IBKR's
09:30 bar (42.84/43.42/42.76/43.00 on 2026-04-20). Only the labels moved.

**Why it matters.** `load_etf_gaps` takes the bar labelled 09:30, which in a
shifted session is the **13:30** print, and divides it by the prior session's
last close — which is *correctly* placed, since the last bar of the file is the
last bar either way. So the "overnight gap" is really 09:30-to-13:30 intraday
drift stacked on a real gap. DUST 2026-04-21: 47.21 used as the open instead of
the true 44.18, giving a 9.66% gap where IBKR says 2.62%.

**Nothing catches it.** These phantom gaps run 2-5%, and `MAX_PLAUSIBLE_GAP` is
0.50 — 0 of 243 in-window gaps were rejected. 80 of them cleared their own
threshold and became candidates. Measured against IBKR on the overlap window,
6 of 9 candidate-membership flips were DUST, every one AV-candidate ->
IBKR-not-a-candidate. They are phantom trades.

**Impact**: **183 of 2,304 `_priority_trades` (7.9%, 7.5% of summed
`pnl_pct`)** and **795 of 4,153 `_unpruned_trades` (19.1%, -9.8%)**.

The largest single contributor is **BOIL** — 337 unpruned trades summing
`pnl_pct` -3.069 — which is a `FORCE_INCLUDE` symbol live actually trades, and
the one with no `master_universe.csv` row. So this is not a backtest-only
curiosity: the same shifted history feeds BOIL's sigma and its PS-filter
threshold.

**Fix**: `scripts/fix_shifted_timestamps.py` (in BacktestingGaps) re-labels the
wall-clock reading as ET. Detection is per session on the RTH start time, so it
is idempotent and backs up originals before writing. Regenerate the trade
caches afterwards.

---

## Backtest / live parity

### P1. 12.2% of engine trades are unreachable from the candidate set
2,022 of 2,304. Both paths nominally use `leverage × 2%` on the ETF's own gap,
so the residual is unexplained. Partly the D2/D3 guards, partly PS-filter
application, partly the most-extreme-wins reconstruction — never reconciled.

### P2. The golden-session gate is armed but not firing
`test_gap_scan_replays_identically` covers the pre-market layer as of `f8f211b`,
but only **2 real fixtures** exist (2026-08-19, 2026-08-20) and both are
schema-1, so the scan replay skips. It will start covering real sessions once
live writes schema-2 fixtures. Until then, live/backtest agreement on candidate
selection is asserted, not tested.

### P3. Backtest sizing is flat; live compounds
The backtest sizes every trade at `$1,000 × weight` against a fixed `$10,000`.
Live sizes at `base_notional_pct × current_equity`. Ratios (Sharpe, drawdown %)
carry over; **dollar figures do not**.

### P4. The margin budget is a static snapshot
`seed_margin_rates` snapshots AvailableFunds once. Live's real AvailableFunds
moves intraday with unrealised P&L, so the ceiling drifts during a session in a
way neither the backtest nor the gate models.

---

## Research questions

> **Methodology warning, learned the hard way 2026-08-21..23.** Two conclusions
> in this file were reversed on re-examination, both from the same two errors:
>
> 1. **Per-symbol P&L in the pruned trade set confounds the instrument with its
>    era.** skip-cheap picks one sibling per underlying per day, so siblings can
>    trade disjoint periods. GDXU and NUGT overlap on **zero** pruned days; on
>    the 90 unpruned days they share, the 3x beats the 2x by **3.5x**. "GDXU is
>    dead weight" was an artifact. Score symbols on the unpruned set.
> 2. **Mean return per margin dollar cannot see diversification.** It ranked C1
>    last when C1 is the class whose removal triples drawdown, and ranked BOIL
>    second-worst when BOIL is the most independent symbol in the book
>    (+0.003 mean pairwise P&L correlation over 330 trading days).
>
> Before removing any instrument, check both: score it unpruned, and check its
> marginal contribution with the margin allowed to reflow.

### R1. Does crypto earn its margin cost? — ANSWERED 2026-08-23: YES, decisively
Drop a class from the candidate set and re-run the real allocation policy at the
same budget, so the freed margin reflows to whatever else wanted it:

| | trades | Sharpe | vol | MaxDD | P&L |
|---|---|---|---|---|---|
| all classes | 1900 | **2.181** | 99.4 | **-8.91%** | 29,145 |
| without C1 | 1629 | 1.732 | 92.4 | **-23.02%** | 22,237 |
| without C2 | 1028 | 2.094 | 86.0 | -7.27% | 24,580 |
| without C3 | 1192 | 1.068 | 57.9 | -7.65% | 10,271 |

**Removing crypto triples drawdown and costs 21% of Sharpe.** Mechanism, from
pairwise daily-P&L correlation: C1 vs C3 is **-0.245 conditional on both
trading** (and -0.002 zero-filled). Crypto does not merely fail to correlate --
it moves opposite to broad-market instruments on shared days. It *raises*
volatility (99.4 vs 92.4) while *lowering* drawdown, i.e. it earns on the days
C3 loses, which no variance measure would surface.

The earlier reading here -- "C1 is the worst class per margin dollar (2.469) at
the highest rate (1.066)" -- was true and completely misleading. A first-moment
ranking cannot price a hedge. Do not use per-margin mean return to judge whether
an instrument belongs in the book.

**Still true and still actionable**: C1 flood is ~zero (+0.0004, n=55, 6 of 9
symbols negative), and C1 is internally redundant (+0.695 conditional
correlation among its own members). The hedge likely survives on a few crypto
names rather than eleven, which frees margin without giving up the drawdown
protection. That is the precise lever; cutting the class is not.

**C2 is the weak class**: dropping it *improves* drawdown (-7.27%) and costs
only 4% of Sharpe.

### R1-old. Original note, retained for the record
On repaired data C1 is the **worst class per margin dollar (2.469) while paying
the highest rate (median 1.066)**; C3 returns 9.663 and C2 5.634. But note the
measurement is a first moment and cannot see diversification, which is the
actual case for holding crypto — that still needs a variance/drawdown
contribution study, not a mean-return ranking.

What is clear is narrower and actionable: **C1 flood is the only ~zero cell in
the table (+0.0004, n=55) and 6 of its 9 symbols are negative there.** Crypto
also only trades 55 flood days against 241 quiet and 133 active, so it is not
mainly competing for flood-day margin — but the slice where it does compete is
the slice where it does not earn.

Original note follows.
C1 has the highest P(fire) (45.5%) but IB charges ~2x the modelled rate — near
or above full notional regardless of stated leverage. Three crypto positions
saturate the account where four conventional ones fit. Whether crypto deserves
its share of a constrained budget is unanswered.

### R2. C2 is incoherent — ANSWERED 2026-08-21
It is a residual bucket, not a cluster. Split by what the instruments actually
track, the sub-groups disagree on exactly the axis the regime notch operates on
(mean pnl_pct per margin dollar, per trade):

| sub-group | syms | trades | flood | active | quiet | win |
|---|---|---|---|---|---|---|
| gold miners | 5 | 289 | **+0.0265** | +0.0123 | +0.0037 | 54% |
| international | 8 | 338 | +0.0047 | +0.0083 | -0.0008 | 45% |
| single-stock | 4 | 328 | +0.0014 | +0.0103 | +0.0055 | 51% |

**Gold miners on flood days are the best cell in the strategy** — better than
C3 flood (+0.0181) — and the `("C2","flood"): 0.5` notch suppresses them because
they share a bucket with two groups that genuinely are weak on flood. The notch
is right about C2-as-labelled and wrong about two thirds of what is in it.

The same boundary shows up in the candidate screen: gold *bullion* (UGL -0.063,
GLL -0.034) is negative while gold *miners* (NUGT +0.0206, JNUG +0.0205) are
near the top. Same metal, opposite behaviour, because one is equity and one is
not. The category should follow the driver, not the sector.

Caveat before acting: the gold-miner flood cell is n=57 with 5/7-year sign
stability. The strongest argument for splitting rests on the thinnest data, and
splitting means three new weight cells estimated on thinner samples each.

**Removals need no theory and are independent of the split**: GDXU (-0.312 over
74 trades, worst in C2, and the symbol with the L-series 3x/threshold bug) and
the near-zero international cluster KORU (+0.0004, n=81), BRZU (-0.0008, n=74),
YINN (-0.0007, n=24) — 179 trades for nothing.

`CLASS_2_SYMS` also lists 13 symbols that have never produced a trade (AMDG,
AMUU, NVDG, NVDL, NVDW, NVDX, TSL, TSLG, TSLI, TSLR, TSLT, TSLW, UVIX). They
lose every sibling contest, so removing them is hygiene, not P&L.

### R3. Side preference is real but unstable
Shorts return ~1.5x per margin dollar pooled, but the ratio flips by year
(2.88 / 0.56 / 2.72 / 0.67 / 2.07 / 2.25 / −0.04 for 2020–2026) and the existing
regime variable does not explain it — shorts win in all three regimes.
Deliberately left unconditioned rather than fitted to three of seven years.
