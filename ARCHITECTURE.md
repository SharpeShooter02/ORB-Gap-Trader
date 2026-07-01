# ARCHITECTURE.md — orb-live-trading

Phase 1 inventory for the IBKR-native rebuild.  Read this before deleting or modifying anything.

---

## Hard Line

**Tier 1 tests green ≠ safe to trade.**
The bar for "ready" is Tier 3 (live paper integration) green during market hours PLUS one watched paper session with an observed fill and a clean EOD flatten.  A mock-passing test suite has been wrong before on this project — that is the lesson encoded here.

---

## File Classification

### ROOT LEVEL

| File | Classification | Reason |
|------|---------------|---------|
| `.env` | KEEP (gitignored) | Runtime secrets; not committed |
| `.env.example` | REBUILD | Strip Alpaca keys (3 lines); keep IB/webhook/backup keys |
| `.gitignore` | KEEP | Fine as-is |
| `BUGFIX_PROMPT.md` | DELETE | Planning artifact; bugs are fixed and committed |
| `DEPLOYMENT.md` | KEEP | Operational reference |
| `HARDENING_PROMPT.md` | DELETE | Superseded by this rebuild |
| `OCA_FIX_PROMPT.md` | DELETE | Superseded by Bug 11 fix |
| `README.md` | KEEP | Update after rebuild completes |
| `REBUILD_PROMPT.md` | DELETE after Phase 5 | Execution prompt; not a deliverable |
| `STATUS.md` | DELETE | Ephemeral; superseded by ARCHITECTURE.md |
| `STATUS_CANDIDATE_SIZING.md` | DELETE | Planning artifact |
| `V1_MIGRATION_SPEC.md` | DELETE | Migration is complete |
| `_production_run.py` | DELETE | Backtester launcher; imports `reference/`; not live system |
| `orb_backtester.py` | DELETE | Backtester; not part of live system |
| `pyproject.toml` | KEEP | No Alpaca deps; clean |
| `data/` | KEEP | Placeholder dirs for logs/archive/reports |
| `reports/` | KEEP | Session JSON reports |
| `reference/` | DELETE | Old backtester config (`config.py`, `_production_run.py`, `orb_backtester.py`); only imported by root-level `_production_run.py` which is itself deleted |
| `tools/gap_ps_diagnostic.py` | KEEP | Operator diagnostic |
| `tools/ib_bar_probe.py` | KEEP | IB-native bar delivery smoke test |
| `tools/preflight_check.py` | KEEP | Pre-session connectivity check |

---

### `orb_live/config/`

| File | Classification | Reason |
|------|---------------|---------|
| `live_config.py` | KEEP | All config lives here; no Alpaca references |
| `overrides.yaml` | KEEP | Operator overrides |

---

### `orb_live/core/`

| File | Classification | Reason |
|------|---------------|---------|
| `calendar.py` | KEEP — DO NOT MODIFY | Validated holiday/`prev_trading_day` logic |
| `clock.py` | KEEP | MarketClock; IB-aware |
| `logger.py` | KEEP | structlog wrapper |
| `state_store.py` | KEEP + Phase 4 migration | 20-table SQLite schema; Phase 4 adds init-time migration for any missing columns |

---

### `orb_live/data/`

| File | Classification | Reason |
|------|---------------|---------|
| `bar_cache.py` | KEEP | In-memory bar accumulator; inert |
| `broker_client.py` | KEEP | Abstract ABC defining the broker interface |
| `ib_client.py` | KEEP | Sole concrete broker; IB-native via `ib_async` |
| `sigma_runtime.yaml` | KEEP | Runtime sigma seeds |
| `underlying_data.py` | KEEP | Parquet-backed UL data store |
| `underlyings/*.parquet` | KEEP | Historical underlying daily data |

No Alpaca client remains in `data/`; it was removed in May 2026.

---

### `orb_live/execution/`

| File | Classification | Reason |
|------|---------------|---------|
| `indicators.py` | KEEP | EMA/rolling indicators; seeded from ORB bars |
| `order_policy.py` | REBUILD (Phase 3) | `MarketableLimitPolicy._submit_entry/exit/stop` all call `time.sleep` in a loop — blocks the asyncio event loop when called from a bar callback. Phase 3 replaces entry with IB native bracket (market order + OCA TP1+stop). Exit/stop legs become OCA siblings placed at entry; `_submit_exit`/`_submit_stop` are deleted. Only `compute_entry_limit` is kept as a price calculator for the stop level. |
| `position_manager.py` | REBUILD (Phase 3) | Currently: non-blocking `submit_limit_order` for entry + `_check_pending_entry` polling on each bar (Bug 11 fix); OCA via `submit_oca_pair`; `policy.sell/buy` with `time.sleep` for exit legs. Phase 3: entry becomes market order; bracket (TP1+stop) placed as OCA children transmit-chained; fill detection via `execDetailsEvent`/`trade.fillEvent` (not polling); exits auto-handled by IB when OCA fires. |
| `risk_gate.py` | KEEP | Session kill and exposure caps; no broker-specific logic |

---

### `orb_live/ops/`

| File | Classification | Reason |
|------|---------------|---------|
| `alerts.py` | KEEP | Webhook alerts |
| `calibration.py` | KEEP | Sigma recalibration from yfinance |
| `health_check.py` | KEEP + cleanup | `/status` exposes `ws_token_age_minutes` via `getattr` fallback (safe; returns None for IB). Clean up the dead field in Phase 4. |
| `long_term_logging.py` | KEEP | Audit log |
| `quarterly_calibration.py` | KEEP | Scheduled sigma recalibration |
| `reports.py` | KEEP + Phase 4 fix | Phase 4: fix `idxmax` crash on empty/no-closed-trades day |
| `runbook.md` | REBUILD | References `ws_token_age_minutes` and `BarRouter._token_refresh_watcher` — Alpaca-era. Rewrite for IB-native operation after Phase 3. |

---

### `orb_live/runner/`

| File | Classification | Reason |
|------|---------------|---------|
| `bar_router.py` | KEEP | Already IB-native: `reqRealTimeBars` on main thread, missed-bar REST replay, no WebSocket/token machinery |
| `dry_run.py` | KEEP | `DryRunBroker` wraps `IBClient` for market data; simulates fills locally; no Alpaca |
| `main.py` | KEEP | CLI entry; builds all components from env; daemon loop |
| `session_runner.py` | KEEP | Session orchestrator; phase timeline; LOAD-BEARING bar dispatch order |
| `strategy_engine.py` | KEEP | Per-symbol ORB state machine (WAITING→ORB_COMPLETE→IN_POSITION) |

---

### `orb_live/scripts/`

| File | Classification | Reason |
|------|---------------|---------|
| `backfill_underlyings.py` | KEEP | Operator tool for parquet refresh |
| `backup_state_store.py` | KEEP | Nightly DB backup |
| `force_ws_reconnect.py` | DELETE | Alpaca WebSocket sentinel script; writes a file for `_token_refresh_watcher` which does not exist in IB-native `BarRouter`. Completely dead. |
| `recalibrate_sigmas.py` | KEEP | Sigma recalibration runner |
| `refresh_universe.py` | KEEP | Universe refresh tool |
| `restore_from_backup.py` | KEEP | Restore from B2/R2 backup |
| `smoke_test.py` | KEEP | Pre-session connectivity sanity check |

---

### `orb_live/signals/`

| File | Classification | Reason |
|------|---------------|---------|
| `liquidity.py` | REBUILD (Phase 4) | Gate C (ADV/dollar-volume) is active but disabled in v1. Gate C calls `get_daily_bars` in a hot path at 10:00 ET. Add a config flag (`enable_gate_c: false`) or short-circuit Gate C to always pass. Gates A+B (operator overrides, asset eligibility) are correct and stay. |
| `pre_market.py` | KEEP | Phase 1/Phase 2 qualification; calls `plan_session()` for single definitive candidate list |
| `strategy_signals.py` | KEEP — DO NOT MODIFY | `compute_gap`, `check_prior_session_filter`, `check_breakout`, `compute_entry`, `compute_opening_range` — validated |

---

### `orb_live/strategy/`

| File | Classification | Reason |
|------|---------------|---------|
| `v1_strategy.py` | KEEP — DO NOT MODIFY | `plan_session`, gap/PS/ORB/skip-cheap, regime, sizing, WEIGHTS, class taxonomy, EXPLICIT_DROPS/FORCE_INCLUDE — parity-validated |
| `data/master_universe.csv` | KEEP — DO NOT MODIFY | Validated |
| `data/sigma_master.csv` | KEEP — DO NOT MODIFY | Validated |

---

### `orb_live/tests/`

| File | Classification | Reason |
|------|---------------|---------|
| `conftest.py` | KEEP + Phase 5 | `MockBroker` updated for async fills (Bug 11). Phase 5 adds IB-shaped mock objects (`RealTimeBar`, `Execution`, `OrderStatus`). |
| `test_bar_router.py` | KEEP | Tests `BarRouter` with IB-native flow |
| `test_clock.py` | KEEP | MarketClock tests |
| `test_daemon_loop.py` | KEEP | Daemon loop tests |
| `test_end_to_end_replay.py` | REBUILD (Phase 5) | Currently synchronous mocks; rebuild with real asyncio loop and `loop.call_later` fill events |
| `test_half_day_session.py` | KEEP | Half-day schedule tests |
| `test_ib_client.py` | KEEP | IBClient unit tests |
| `test_live_config_rolling.py` | KEEP | Config tests |
| `test_operational.py` | KEEP | Health/ops tests |
| `test_order_policy.py` | REBUILD (Phase 3) | Tests for `MarketableLimitPolicy` repeg; replaced by bracket tests |
| `test_position_manager.py` | REBUILD (Phase 3) | Tests for marketable-limit position management; replaced by bracket/OCA tests |
| `test_pre_market.py` | KEEP | Pre-market qualification tests |
| `test_ps_filter_parity.py` | KEEP — DO NOT MODIFY | v1 strategy parity |
| `test_session_runner.py` | KEEP | Session runner integration tests |
| `test_signal_parity.py` | KEEP — DO NOT MODIFY | v1 strategy parity |
| `test_state_store.py` | KEEP | DB schema tests |
| `test_underlying_data.py` | KEEP | UL data freshness tests |
| `test_universe_parity.py` | KEEP — DO NOT MODIFY | v1 strategy parity |

---

### `tests/integration/`

| File | Classification | Reason |
|------|---------------|---------|
| `test_ib_client_live.py` | KEEP (Tier 3) | Live connectivity; requires IB Gateway in market hours |
| `test_dress_rehearsal_live.py` | KEEP (Tier 3) | Full paper session dress rehearsal |

---

## Intended Clean Structure (post-rebuild)

```
orb-live-trading/
├── orb_live/
│   ├── config/           # LiveConfig + overrides — no broker logic
│   ├── core/             # calendar, clock, logger, state_store (SQLite)
│   ├── data/             # broker_client (ABC), ib_client (IB-native), bar_cache,
│   │                     #   underlying_data, underlyings/*.parquet
│   ├── execution/        # indicators (EMA), risk_gate
│   │                     #   order_policy → DELETED or reduced to price calculator
│   │                     #   position_manager → IB native bracket, event-driven fills
│   ├── ops/              # health_check, reports, alerts, calibration, runbook
│   ├── runner/           # bar_router (IB native), session_runner, strategy_engine,
│   │                     #   main (CLI entry), dry_run
│   ├── scripts/          # operator tools (no Alpaca scripts)
│   ├── signals/          # pre_market, strategy_signals, liquidity (Gate C disabled)
│   ├── strategy/         # v1_strategy.py + data CSVs — DO NOT MODIFY
│   └── tests/            # Tier 1: offline fast suite
├── tests/integration/    # Tier 3: live paper integration (requires Gateway)
├── tools/                # gap_ps_diagnostic, ib_bar_probe, preflight_check
├── data/                 # logs/, archive/, reports/ (gitkeep placeholders)
├── reports/              # session JSON reports
└── ARCHITECTURE.md       # this file
```

---

## Alpaca Remnants — Complete List

All Alpaca code is gone from `orb_live/` proper. Remnants are:

1. **`.env.example`** — 3 Alpaca env-var lines (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_LIVE`). Strip in Phase 2.
2. **`scripts/force_ws_reconnect.py`** — Alpaca WS sentinel; dead for IB. Delete in Phase 2.
3. **`ops/runbook.md`** — references `ws_token_age_minutes` and `_token_refresh_watcher`. Rewrite in Phase 4.
4. **`ops/health_check.py`** — `/status` has a `ws_token_age_minutes` field (returns None for IB). Remove in Phase 4.
5. **`_production_run.py`** + **`orb_backtester.py`** + **`reference/`** — backtester artefacts, not Alpaca per-se but dead weight for the live system. Delete in Phase 2.

`pyproject.toml` has no Alpaca dependency. Nothing else.

---

## What Has Already Been Rebuilt (pre-Phase 2)

- **Bar streaming** — `BarRouter` is IB-native (`reqRealTimeBars`); no threads, no WebSocket, no token refresh. ✓
- **Re-entry guard** — `open_position` checks broker position truth before persisting 'entering'. ✓ (Bug 11)
- **EOD flatten safety net** — `flatten_all` iterates `_universe` and closes untracked broker positions. ✓ (Bug 11)
- **Async fill detection** — `_pending_entries` dict + `_check_pending_entry` on each bar. ✓ (Bug 11)
- **Cancel-before-repeg** — prior order must reach terminal state before new one submitted. ✓ (Bug 11)
- **Single `plan_session()` call** — Phase 1 runs `plan_session()` once; Phase 2 reads multipliers from it. ✓
- **Holiday awareness** — `MarketClock` consults `calendar.py`. ✓

## What Remains (Phases 3–5)

| Phase | Work |
|-------|------|
| 3 — Execution rebuild | Replace marketable-limit + `time.sleep` + manual OCA with IB native bracket (market entry + OCA TP1+stop transmit-chained). Fill detection via `execDetailsEvent`/`trade.fillEvent`. `order_policy.py` reduced to price calculator or deleted. |
| 3 — Margin-aware sizing | Check `buying_power` before placing; skip/downsize if insufficient (leveraged ETFs ~1:1 margin at IB). |
| 4 — Operational fixes | Gate C disabled in `liquidity.py`. `reports.py` `idxmax` crash on empty day. Schema migration in `state_store`. IB loggers set to WARNING. `health_check.py` `ws_token_age_minutes` field removed. `runbook.md` rewritten. |
| 5 — Tests | Tier 1: real asyncio loop tests with `loop.call_later` fill events. Tier 2: one regression test per confirmed bug. Tier 3: adapt existing integration tests. |
