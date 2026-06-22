# Candidate Selection & Order Sizing — Architecture Trace

*Written 2026-06-22 after supervised paper session surfaced 4 vs 3 candidate divergence.*

---

## 1. The two "candidate" computations

There are two sequential stages, not two competing systems:

### Stage A — `plan_session()` (Phase 1, ~09:31 ET)

Called inside `PreMarketJob.run_phase1()`:

```
plan_session(universe, instruments, sigmas, overnight_gaps, prior_two_closes, prior_etf_close)
  → SessionPlan { candidates: list[str], multipliers: dict[str, float], regime, cap_factor, n_uls }
```

Applies: UL-gap threshold, direction filters, skip-cheap-top-2, regime/cap weighting.
This is the **authoritative v1 candidate list and multiplier map**.
`run_phase1()` then filters its `preliminary_p1` list to symbols in `plan.candidates`.

### Stage B — `PreFlightCheck` (Phase 2, ~10:01 ET)

Called inside `PreMarketJob.run_phase2()` for each Phase 1 survivor:

```
PreFlightCheck.check(symbol, session_date, direction, current_equity)
  → CandidateDecision { passed, reason, adv_dollars, yesterday_dv, … }
```

Three sequential gates:
- **Gate A**: operator overrides (excluded_symbols, date_exclusions, force_long_only)
- **Gate B**: broker asset status (tradable, active, shortable/HTB)
- **Gate C**: ADV / dollar-volume floor ← **now neutralised** (Task 1)

`is_candidate = preflight.passed` drives the Phase 2 result.

### What actually enters the order queue

`SessionRunner._run_post_orb()` calls `engine.on_orb_complete(symbol, p2, orb_df)`.
The engine transitions to `ORB_COMPLETE` only when both:
- `p2.is_candidate == True` (Phase 2 passed)
- `p2.size_mult != 0.0` (plan_session assigned nonzero weight)

So **the watch list = Phase 1 plan_session() candidates that also survive Phase 2 preflight**.
Phase 2 can only reduce, never expand, the set from plan_session().

---

## 2. Sizing at fire time

When a breakout bar fires, `strategy_engine.on_bar()` calls:

```python
compute_entry(
    bar, orb, gap_direction, scfg,
    current_equity=current_equity,
    size_mult=p2.size_mult,             # from plan_session().multipliers[symbol]
    v1_base_notional=self._v1_base_notional,  # LiveConfig.v1_base_notional = $1,000
)
```

Inside `compute_entry()` the branch taken is:

```python
if v1_base_notional is not None and v1_base_notional > 0:
    shares = floor(v1_base_notional * size_mult / entry_price)
```

So:  **shares = floor($1,000 × plan_multiplier / entry_price)**

This IS the v1 multiplier × base_notional path.  `StrategyConfig.daily_risk_pct = 0.0`
is an intentional sentinel — the `else` branch (equity-fraction sizing) is bypassed.

### On "LiveConfig.baseline_pct is None"

`baseline_pct` does not exist anywhere in the codebase.  The equivalent field
is `v1_base_notional` (default $1,000).  The confusion likely arose from the
reference backtester which uses a `baseline_notional` concept.  The live runner
is already on the v1 sizing path; no change is needed.

---

## 3. The 4 vs 3 divergence (today's session)

| Stage | Count | Blocker |
|---|---|---|
| plan_session() candidates | 4 | — |
| Phase 2 Gate C (ADV floor $1M) | −1 | `yesterday_dv_ratio_low` |
| Final watch list | 3 | — |

**Root cause**: Gate C's `min_dollar_volume_floor = $1M` and `min_yesterday_dv_ratio = 0.30`
were sized for equity positions using `daily_risk_pct × equity` (old framework).
Under v1, `intended_dollars = equity × 0.0 = $0`, so `intended_pct_of_adv = 0.0` always
passes the `max_pct_of_adv` check, but the absolute floor and ratio checks still fire
against thin ETFs.

**Fix (Task 1)**: neutralised Gate C defaults to `floor=0, max_pct=1.0, ratio=0.0`.
Telemetry is preserved; re-enable via `overrides.yaml` if needed.

---

## 4. What it takes to unify watch list and sizing into one plan_session() call

This is already essentially true — both come from the same Phase 1 `plan_session()`.
The remaining gap:

| Source of divergence | Consequence | Fix |
|---|---|---|
| Gate C (ADV / DV) | Blocks valid candidates | Neutralised in Task 1 |
| Gate A (operator exclusions) | Intentional; correct | Keep as-is |
| Gate B (broker asset status) | Live-check unavoidable | Keep as-is |

**To make plan_session() == watch list exactly**: remove Gate C from the runtime
`PreFlightCheck` call in Phase 2 entirely (it has no effect now that all three
thresholds are 0/1.0/0.0, but the broker round-trip still happens for every symbol).
Optionally keep a warning log for very-low-DV symbols without gating on it.

**Sizing is already single-source**: `plan_session().multipliers[symbol]` feeds
`p2.size_mult` in Phase 2 and flows unchanged into `compute_entry()` at fire time.
No second sizing computation exists.

---

## 5. Recommended next steps (do not implement without explicit sign-off)

1. Remove the Gate C broker call in Phase 2 entirely (replace with a DV warning log only).
   This saves `N_candidates × 1` broker API calls per session.
2. Consider raising `v1_base_notional` from $1,000 to match actual intended dollar
   exposure (the multiplier × 1k formula is currently producing very small share counts
   on high-priced ETFs like TQQQ at $185 → 5 shares at mult=1.0).
3. Crypto PS-filter weekend mismatch (documented in Task 3 diagnostic) is a separate
   item requiring a calendar-aware PS filter for crypto underlyings.
