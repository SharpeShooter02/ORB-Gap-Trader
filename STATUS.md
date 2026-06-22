# ORB Live Runner — Current Status

_Last updated: 2026-06-22_

## Where things stand

The v1 strategy has been transplanted into this IBKR framework. Migration is
complete, the §4 parity gate passes, the four execution-hardening items are done,
the OCA premature-commit bug is fixed, and the **delayed-data preflight passes
green** against the paper account. Static gates are cleared; the dynamic,
money-touching paths have NOT yet run against a live broker — see the go-live
checklist.

## Done

- **v1 migration** — strategy brain replaced; old `reference/` config, RTG, and
  pair-routing removed. Parity gate green (328 passed).
- **Vendored, not imported** — `v1_strategy.py` + `master_universe.csv` +
  `sigma_master.csv` live in `orb_live/strategy/` (no cross-folder imports).
- **Execution hardening (4 items):** native IB OCA bracket for the TP1/stop pair;
  fail-loud-and-retain on cancel / OCA-sibling-cancel failure; `realized_exit_price`
  recorded alongside the target `exit_price`; expanded preflight (checks 6–10).
- **OCA TP1 fix** — TP1 now commits only on confirmed `filled`/`partially_filled`
  order status (never synthesized from bar price); partials book only newly-filled
  shares; bar-cross-without-fill is diagnostic-only and alerts after 2 bars;
  realized price re-polls once if `filled_avg_price` is 0.
- **FNGU/FNGD/BULZ → QQQ** — fixed a latent parity bug. Backtest (`config.py`)
  always used QQQ; live had drifted to NYFANG (no data, looser sigma). Changed in
  both `master_universe.csv` copies. NYFANG row in `sigma_master.csv` is now dead
  (optional cleanup).
- **BTFX dropped** — added to `EXPLICIT_DROPS` (untradable on IB; BTC still
  covered by BITU/BITX/BTCL/BTCZ/SBIT). Universe now **59** symbols.
- **Preflight pacing check fixed** — was calling `get_intraday_bars(lookback_days=…)`
  (wrong signature) so it never ran. Now uses a real `(start, end)` ~35-min window
  with a 7-day lookback so the rate-limit guard is actually exercised.

## Preflight: last run (2 FAILs, both expected) → next run should be clean

1. **Market data subscription (SPY) — FAIL.** Ran without the delayed flag. IB
   confirmed delayed data is available. Re-run with
   `IB_ALLOW_DELAYED_DATA=1 IB_MARKET_DATA_TYPE=3`. (This is also the real-time
   subscription you'd buy later.)
2. **Underlying parquets missing — FAIL.** EWZ, FEZ, FXI, IHE, IWM, MSOS, XOP
   need a backfill: `python -m orb_live.scripts.backfill_underlyings`. (NYFANG no
   longer needed after the QQQ switch. Note: `orb_live.data.underlying_data` has
   no __main__ and does nothing — the real entrypoint is the scripts module.)
3. Untradable warnings for CLDL/OOTO/ARMU/SNPX are **noise** — class-3 /
   single-stock names v1 doesn't trade. BTFX is now dropped.

## Open items before going live

- [ ] Backfill underlying parquets (command above), then re-run preflight with
      delayed data until it's fully green.
- [ ] **Watch a real OCA fill in paper** — especially a *partial* TP1 fill — to
      confirm IB's `ocaType=1` overfill/cancel behavior matches our accounting.
      This is the one thing unit tests can't validate.
- [ ] Optional: remove the dead NYFANG row from `sigma_master.csv`.
- [ ] Only after a clean delayed-data dress rehearsal: buy the two IBKR
      real-time data streams and do a real-time paper session (real-time is the
      only thing that validates ORB breakout *timing*).
- [ ] First real-time paper session: supervised, confirm EOD-flat at 16:00,
      watch `/health` and `/status`.

## Run reference

```bash
# Preflight (delayed data, paper)
IB_ALLOW_DELAYED_DATA=1 IB_MARKET_DATA_TYPE=3 python tools/preflight_check.py

# Backfill underlyings (5y daily bars → cfg.data_dir)
python -m orb_live.scripts.backfill_underlyings

# Tests
pytest orb_live/tests/ -q
```

## Related docs

- `V1_MIGRATION_SPEC.md` — the migration plan (keep/replace/delete map, parity gate)
- `HARDENING_PROMPT.md` — the four execution-hardening items
- `OCA_FIX_PROMPT.md` — the TP1 premature-commit fix
