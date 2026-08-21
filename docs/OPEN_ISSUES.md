# Open issues

Known, unresolved, and deliberately not fixed yet. Spans both repos:
`orb-live-trading` (live) and `../BacktestingGaps` (backtest/research).

Last reviewed 2026-08-21. Add to this file rather than relying on a session
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

### D1. Alpha Vantage / IBKR bar seam, never measured
`BacktestingGaps/scripts/compare_av_vs_ibkr_bars.py` was written to diff cached
AV-era bars against fresh IBKR pulls on the same (symbol, date), specifically
the 09:30–10:00 ORB high/low. **Its verdict was never recorded.**
`pull_ibkr_last_2mo.py` then backfilled IBKR bars from 2026-06-16 onward, so the
intraday cache is mixed-provenance with an unmeasured seam at that date.

A bar-source discontinuity mid-sample presents as a regime change that is not
one. Everything else is measured against these bars, so this is the top data
priority.

### D2. Split artifacts in the intraday ETF cache
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

### R1. Does crypto earn its margin cost?
C1 has the highest P(fire) (45.5%) but IB charges ~2x the modelled rate — near
or above full notional regardless of stated leverage. Three crypto positions
saturate the account where four conventional ones fit. Whether crypto deserves
its share of a constrained budget is unanswered.

### R2. C2 is incoherent
Largest class (~4,355 candidates) and the least well-behaved: the gap-size
signal is non-monotone within it, unlike C1 (crypto) and C3 (large
market-following ETFs). Reads as a universe-composition problem.

### R3. Side preference is real but unstable
Shorts return ~1.5x per margin dollar pooled, but the ratio flips by year
(2.88 / 0.56 / 2.72 / 0.67 / 2.07 / 2.25 / −0.04 for 2020–2026) and the existing
regime variable does not explain it — shorts win in all three regimes.
Deliberately left unconditioned rather than fitted to three of seven years.
