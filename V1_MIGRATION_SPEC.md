# V1 Migration Spec — Transplant v1 strategy into the existing IBKR live runner

**Purpose.** This repo (`orb-live-trading`) is a complete IBKR live-trading
framework, but its strategy layer encodes the *pre-v1* production config
(k=1.25, 47 symbols, RTG + pair-routing, multi-TP, look-ahead-biased sizing).
The goal is to keep the broker/IO/ops infrastructure and replace only the
strategy brain with **v1** (`../v1/v1_strategy.py`): k=1.00, ~60 symbols,
skip-cheap-top-2 pruning, 3-class × 3-regime causal sizing, TP1-only exit.

**Hand this file to Claude Code, running inside this repo, with TWS/IB Gateway
in paper mode on port 4002.** Cowork cannot reach a local Gateway, so the
build/test loop must run where the socket lives.

---

## 0. Ground truth: what v1 actually is

Read these first (in `../v1/`):
- `V1_SYSTEM_REPORT.md` — the canonical spec
- `v1_strategy.py` — the pure-function contract (`plan_session()`)
- `README.md` §"Locked config at a glance"
- `memory/feedback_lookahead_bias.md` — the hard line: every signal must be
  computable before 9:30 ET. No close-to-close, no fired-trade counts.

Locked v1 constants:

| Param | v1 value | old-code value (to remove) |
|---|---:|---:|
| PS filter k | **1.00** | 1.25 |
| Universe | **~60 ETFs / ~30 ULs** | 47 symbols |
| Gap threshold | **2% on UL** | per-instrument GAP_FILTER |
| Regime cuts | **active=5, flood=8** (on n_uls) | n/a (RTG instead) |
| Sizing | **weight[class,regime] × cap_factor**, cap_units=20 | apply_sizing / CLASS_A / FLOOD_LOSERS |
| TP | **TP1-only at 1.0× ORB range** | 0.35 / 0.05 / 0.60 |
| Stop | ORB tight (inner-quarter) | midpoint variants |
| DOW exclusions | **none** | DOW_EXCL (ETHU/ETHD Monday) |
| RTG / pair-routing | **none** | rtg.py + routing.py |

---

## 1. KEEP — strategy-agnostic infrastructure (do not rewrite)

| File | Why keep |
|---|---|
| `data/ib_client.py` | Full `ib_async` IBKR wrapper: orders, quotes w/ degraded-data guard, 5s→1min aggregation, streaming. The core asset. |
| `data/broker_client.py` | Clean `BrokerClient` ABC. Strategy never touches IB directly. |
| `runner/bar_router.py` | WS streaming + missed-bar replay + token refresh + REST poll fallback + degraded mode. |
| `core/clock.py`, `core/calendar.py` | RTH/half-day/effective_close, phase detection, market calendar. |
| `core/state_store.py` | SQLite persistence. Keep engine + generic tables; prune RTG-specific tables (see §3). |
| `core/logger.py` | Structured logging. |
| `execution/order_policy.py` | MarketableLimitPolicy w/ repeg + slippage guard. |
| `execution/risk_gate.py` | Session kill-switch + concurrent-position limits. |
| `execution/indicators.py` | Rolling EMA state seeded from ORB bars. |
| `runner/session_runner.py` | Top-level orchestrator + the load-bearing bar-dispatch-order invariant. |
| `runner/strategy_engine.py` | WAITING→ORB_COMPLETE→IN_POSITION state machine. |
| `runner/main.py` | Entrypoint + `build_client_from_env`. |
| `ops/*` | Health server, alerts, long-term logging, calibration. |
| `tests/` harness | 150-test synthetic-replay pattern. Reuse the scaffolding; rewrite the strategy-specific assertions. |

---

## 2. KEEP-WITH-EDITS — the signal primitives

`signals/strategy_signals.py` — `compute_gap`, `check_prior_session_filter`,
`compute_opening_range`, `check_breakout`, `compute_entry` are causal and match
v1's mechanics. Edits:

- `compute_entry`: collapse multi-TP to **TP1-only** (exit_ratio_tp1 = 1.0,
  tp2/tp3 = 0). Drop `size_mult` from RTG routing.
- `check_breakout`: v1 sets `min_profit_pct = min_increment_pct =
  min_entry_excess = 0` — every breakout is taken. Keep the EMA-side gate
  (`close > orb.ema` for longs). Verify the EMA(30) seeding matches v1.
- Stop: use v1 **tight** method (inner-quarter of ORB), confirm against
  `compute_entry`'s current `(midpoint + low)/2` formula — that *is* the tight
  method; keep it, drop the alternatives.

---

## 3. REPLACE — the strategy brain

### 3a. Config bridge — rewrite `config/live_config.py`
- **Delete** all `from reference._production_run import ...` (UNIVERSE, SIGMA,
  apply_sizing, CLASS_A_*, FLOOD_LOSERS, DOW_EXCL, GAP_FILTER, PS_FILTERS, etc.).
- **Vendor v1 into this repo — do NOT import across folders.** Copy
  `v1_strategy.py`, `master_universe.csv`, and `sigma_master.csv` from `../v1/`
  into the package (e.g. `orb_live/strategy/v1_strategy.py` and
  `orb_live/strategy/data/`). The live repo must be self-contained: no runtime
  dependency on the `../v1/` folder. (Reading `../v1/` for reference during the
  build is fine; runtime imports from it are not.) Record the vendored v1
  source commit/date in a header comment so drift is visible.
- Source config from the vendored copy: K_SIGMA=1.00, CAP_UNITS=20,
  GAP_THRESHOLD=0.02, ACTIVE_MIN=5, FLOOD_MIN=8, WEIGHTS, CLASS_1_SYMS,
  CLASS_2_SYMS, EXPLICIT_DROPS, FORCE_INCLUDE, DIRECTION_FILTERS.
- Universe built by the vendored `build_universe()` from the vendored
  `master_universe.csv` + `sigma_master.csv`.
- `ps_filter_k` default → **1.00**.

### 3b. Pre-market Phase 2 — gut and replace `signals/pre_market.py`
- Phase 1 (gap scan + PS filter) scaffolding is fine — repoint to v1 config.
- **Phase 2 is a different algorithm.** Remove RTG computation, RTG percentile
  history, RTG exclusion, and pair routing entirely. Replace with v1's pipeline:
  1. `compute_candidates()` — gap≥2% + PS(k=1.00) + direction filter +
     **skip-cheap-top-2** per UL (rank by prior ETF close).
  2. `n_uls` = distinct underlyings in candidates → `assign_regime()`.
  3. `compute_cap_factor()` = min(1, cap_units / Σ weight[class,regime]).
  4. `multipliers[sym]` = weight[class,regime] × cap_factor; drop zero-weight
     (e.g. C2 on flood days).
- The function v1 already exposes as the whole Phase-2 contract is
  `plan_session()` — wire the live Phase-2 to call it and return its
  `SessionPlan(candidates, regime, n_uls, cap_factor, multipliers)`.

### 3c. Sizing — replace position-size math in `execution/position_manager.py` / `compute_entry`
- Old: `apply_sizing` + risk-based/daily-risk sizing + CLASS_A/FLOOD_LOSERS.
- New (v1): per fire, `quantity = base_notional × multipliers[sym] / entry_price`
  where `base_notional` is the $1k-per-unit baseline. cap_factor is already
  baked into `multipliers`. No CLASS_A, no FLOOD_LOSERS, no purity_skip.

### 3d. DELETE
- `signals/rtg.py`, `signals/routing.py` (+ their tests).
- `reference/` (old k=1.25 backtester/config) — or repoint to v1 and keep only
  for historical-parity diffing, clearly quarantined.
- State-store tables: `rtg`, RTG-history, routing — drop from schema.
- `.env.example` Alpaca lines — replace with the IB block from `DEPLOYMENT.md`
  (`IB_HOST`, `IB_PORT=4002`, `IB_CLIENT_ID`).

---

## 4. Parity gate (non-negotiable before any paper run)

Build a parity test that, for a sample of historical dates, asserts the live
path reproduces v1's backtest decisions:

1. **Candidate parity** — live `plan_session()` candidates == v1
   `compute_causal_candidates()` for the same date. (See v1
   `scripts/run_final_config_causal.py`.)
2. **Multiplier parity** — live `multipliers` == v1 `causal_size()`.
3. **Signal parity** — gap, PS pass/block, ORB high/low/EMA match the
   backtester within tolerance.
4. **No look-ahead audit** — confirm every Phase-1/Phase-2 input is timestamped
   ≤ 9:30 ET (gap from open vs prior close; PS from prior two closes; skip-cheap
   from prior ETF closes; regime from candidate n_uls, NOT fired count).

Reuse the existing `test_signal_parity.py` / `test_universe_parity.py` /
`test_ps_filter_parity.py` scaffolding — repoint expected values to v1.

---

## 5. Suggested build order

1. Vendor `v1_strategy.py` + `master_universe.csv` + `sigma_master.csv` into the
   package (no cross-folder imports); rewrite `live_config.py`; get
   `load_live_config()` to return ~60 symbols at k=1.00.
2. Rewrite Phase 2 around `plan_session()`; delete RTG/routing.
3. Collapse exits to TP1-only; fix sizing to multiplier-based.
4. Make the parity tests pass (this is the real definition of done).
5. Dry-run replay (`runner/dry_run.py`) over a few historical sessions.
6. Paper run against IB Gateway (port 4002), EOD-flat at 16:00, watch
   `/health` and `/status`.

---

## 6. Kickoff prompt for Claude Code

> This repo is an IBKR live-trading framework whose strategy layer encodes an
> old, look-ahead-biased config. Read `V1_MIGRATION_SPEC.md`, then read
> `../v1/V1_SYSTEM_REPORT.md` and `../v1/v1_strategy.py`. Execute the migration
> in `V1_MIGRATION_SPEC.md` §5: replace the strategy brain with v1 while keeping
> the IBKR/IO/ops infrastructure. The definition of done is the §4 parity gate
> passing. Do not port anything from `reference/` or `signals/rtg.py` /
> `signals/routing.py` — they carry the old config and its bias. Work in small
> commits; run `pytest orb_live/tests/ -q` after each step.
