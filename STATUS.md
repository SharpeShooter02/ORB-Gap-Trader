# ORB Live Runner — Current Status

_Last updated: 2026-06-22_

## Where things stand

The v1 strategy is transplanted into this IBKR framework. Migration complete,
parity gate passing, the four execution-hardening items done, the OCA
premature-commit bug fixed, and the delayed-data preflight passes green against
the paper account. Static gates are cleared. The dynamic, money-touching paths
(real-time quotes, streaming, order placement, fills) have NOT yet run against a
live broker — that's the go-live checklist below.

## Done

- **v1 migration** — strategy brain replaced; old `reference/` config, RTG, and
  pair-routing removed. Parity gate green.
- **Vendored, not imported** — `v1_strategy.py` + `master_universe.csv` +
  `sigma_master.csv` live in `orb_live/strategy/` (no cross-folder imports).
- **Execution hardening (4 items):** native IB OCA bracket for the TP1/stop pair;
  fail-loud-and-retain on cancel / OCA-sibling-cancel failure; `realized_exit_price`
  recorded alongside the target `exit_price`; expanded preflight (checks 6–10).
- **OCA TP1 fix** — TP1 commits only on confirmed `filled`/`partially_filled`
  status (never synthesized from bar price); partials book only newly-filled
  shares; bar-cross-without-fill is diagnostic-only and alerts after 2 bars;
  realized price re-polls once if `filled_avg_price` is 0.
- **FNGU/FNGD/BULZ → QQQ** — fixed a latent parity bug (backtest always used QQQ;
  live had drifted to NYFANG, which has no data and a looser sigma).
- **BTFX dropped** — added to `EXPLICIT_DROPS` (untradable on IB; BTC still
  covered by BITU/BITX/BTCL/BTCZ/SBIT). Universe = **59** symbols.
- **Preflight fixes** — pacing check now uses a real `(start, end)` window;
  subscription check honors `IB_ALLOW_DELAYED_DATA` (soft-pass for dress
  rehearsal); underlying staleness is calendar-aware (`prev_trading_day`), so
  weekends/holidays don't false-alarm; backfill command points at the real
  entrypoint (`scripts.backfill_underlyings`); `MSOS` mapped + `_yf_ticker`
  defaults to identity so a missing ticker can't crash the whole backfill.

## Preflight status: GREEN (delayed data)

All 10 checks pass with the delayed flags set (see Run reference). Sigma source
29 rolling / 0 seed. Untradable warnings for CLDL/OOTO/ARMU/SNPX are **noise**
(class-3 / single-stock names v1 never trades); BTFX is dropped.

What green preflight does NOT prove: real-time quotes, real-time bar streaming,
ORB breakout *timing*, order placement, or fills. All read-only. Those are the
point of the paper sessions below.

---

# GO-LIVE CHECKLIST (paper)

Definition of done = several clean supervised paper sessions, including at least
one observed OCA fill. A green preflight is NOT the bar.

## Stage 0 — pre-session (offline / anytime)

- [ ] `git add -A && git commit` the current known-good state (it's all
      uncommitted right now).
- [ ] `pytest orb_live/tests/ -q` green.
- [ ] `runner/dry_run.py` end-to-end on the migrated code (catches wiring errors
      without market hours).
- [ ] Optional cleanup: remove the dead NYFANG row from `sigma_master.csv`.

## Stage 1 — buy data + first supervised paper session

Real-time data is required here — delayed data cannot validate ORB *timing*.
Buy the IBKR real-time stream(s), then run ONE session start-to-finish, watched.

- [ ] Re-run preflight in **strict** mode (unset `IB_ALLOW_DELAYED_DATA`) during
      RTH — SPY must return a non-zero real-time quote (confirms the subscription).
- [ ] Start runner in paper mode; confirm pre-market scan (~9:20 ET) produces a
      candidate list and writes it to the state store.
- [ ] At 10:00 ET confirm real-time bars stream for the day's CANDIDATES only
      (not all 59) — check the line count.
- [ ] Confirm ORB high/low/EMA per candidate; watch for a breakout.
- [ ] **Watch the first real order placement** — entry, then the OCA bracket
      (TP1 limit + stop). Verify both legs resting at IB.
- [ ] **Observe at least one OCA fill** — ideally a *partial* TP1 — and confirm
      IB cancels the stop sibling and accounting books only filled shares.
      (Biggest unproven risk; no test can show it.)
- [ ] Confirm **EOD flatten at 16:00 ET** closes everything; no overnight position.
- [ ] `/health` 200 and `/status` sane throughout; no CRITICAL alerts.

## Stage 2 — resilience + variety (several more paper sessions)

- [ ] Kill IB Gateway mid-session → confirm bar_router reconnect + missed-bar
      replay + token refresh recover cleanly.
- [ ] See each of: a flood day (many candidates / cap_factor binding), a stop-out,
      and a half-day session (early close handled).
- [ ] Reconcile each day: paper blotter P&L vs `realized_exit_price` in
      `closed_trades`; investigate divergence beyond expected slippage.
- [ ] Confirm risk gate (session kill-switch, concurrent-position limits) behaves.

## Stage 3 — only after Stage 1–2 are clean across a week

- [ ] Decide real-money go-live separately. Do not flip live flags or unset paper
      mode until paper has run clean for multiple sessions including a real OCA
      fill and a reconnect.

---

## Run reference

```bash
# Preflight — delayed data (dress rehearsal). bash/MINGW64 syntax:
IB_ALLOW_DELAYED_DATA=1 IB_MARKET_DATA_TYPE=3 python tools/preflight_check.py

# Preflight — strict (real-time subscription check), run during RTH:
python tools/preflight_check.py

# Backfill underlyings (5y daily bars → cfg.data_dir)
python -m orb_live.scripts.backfill_underlyings

# Tests
pytest orb_live/tests/ -q
```

## Related docs

- `V1_MIGRATION_SPEC.md` — the migration plan (keep/replace/delete map, parity gate)

## Note on tooling

Commit frequently and let one tool own the working tree at a time (Cowork's
sync layer and Claude Code writing the same files simultaneously is asking for
trouble). Git commits are the safe handoff point between them.
