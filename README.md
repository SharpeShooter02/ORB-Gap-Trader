# ORB Live Trading System

Automated live execution of the Opening Range Breakout strategy across a
universe of leveraged and inverse ETFs.  Uses Interactive Brokers (IBClient)
for order routing and IB-native streaming (`reqRealTimeBars`, 5-sec bars).

## Architecture

```
orb_live/
├── config/           LiveConfig + StrategyConfig (all tunables in one place)
├── core/
│   ├── clock.py      MarketClock — RTH detection, half-day schedule, effective_close()
│   └── state_store.py  SQLite persistence (20 tables via SQLAlchemy Core)
├── data/
│   ├── ib_client.py       IB Gateway wrapper (IBClient, ib_async) — 5-sec real-time bars
│   └── underlying_data.py  Prior-session OHLCV via yfinance
├── signals/
│   ├── pre_market.py  Phase 1 (gap scan + PS filter) + Phase 2 (RTG + preflight)
│   └── ps_filter.py   Prior-session filter with per-symbol σ thresholds
├── execution/
│   ├── indicators.py      Rolling EMA state (seeded from ORB bars)
│   ├── order_policy.py    MarketableLimitPolicy with repeg + slippage guard
│   ├── position_manager.py  TP1/TP2/TP3/stop/EOD exit management
│   └── risk_gate.py       Session kill-switch + concurrent position limits
├── runner/
│   ├── bar_router.py    IB-native 5-sec bar streaming + missed-bar REST replay
│   ├── strategy_engine.py  State machine: WAITING → ORB_COMPLETE → IN_POSITION
│   └── session_runner.py   Top-level orchestrator (pre-market → trade → EOD)
└── ops/
    ├── health_check.py     HTTP server on :8080 (/health /status /positions /metrics)
    ├── alerts.py           Discord/Telegram/Slack webhook with rate limiting
    ├── long_term_logging.py  Daily rollup + monthly reports + log cleanup
    ├── quarterly_calibration.py  ADV baselines, metric drift, universe audit
    └── runbook.md          Operator manual
```

## Quickstart (paper trading)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Start IB Gateway / TWS in paper-trading mode

# 3. Run tests
python -m pytest orb_live/tests/ -q   # 453 tests, ~1s

# 4. Start session
python -m orb_live.runner.main --paper

# 5. Monitor
curl http://localhost:8080/status | python -m json.tool
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for production VPS setup.

## Bar Dispatch (two paths)

The runner processes two bar streams from `BarRouter`, both fed by the same
IB 5-sec `reqRealTimeBars` feed:

**1. Entry path — `SessionRunner._on_entry_bar_dispatch` (every 5-sec bar).**
Post-ORB 5-sec bars are forwarded straight to the strategy engine for breakout
detection, so an entry fires the instant price crosses the ORB boundary rather
than waiting for the 1-min bar to close. It deliberately touches nothing else
(no cache/indicator/exit work). The first path to see the breakout flips the
symbol to `IN_POSITION`; the other then no-ops via the engine's state guard, so
running both is idempotent.

**2. Main path — `SessionRunner._on_bar_dispatch` (aggregated 1-min bar).**
Load-bearing order — do not reorder steps 2/3/4:

```
1. bar_cache.add_bar(symbol, bar)         — accumulate ORB window
2. indicators[symbol].on_bar(bar)         — EMA update  ← MUST precede step 3
3. position_manager.on_bar(symbol, bar)   — exit management reads current EMA
4. strategy_engine.on_bar(symbol, bar)    — entry detection (ORB_COMPLETE only)
```

Reordering steps 2 and 3 causes position manager to use a one-bar-stale EMA.
This invariant is enforced by `test_i_indicator_updated_before_position_manager`.

## Configuration

All strategy parameters live in `orb_live/config/live_config.py`.  Production
config is locked at:

### Backtest results (2020–2026, locked v1 — boundary-fill variant, live-parity universe)

Live universe (58 symbols, drops BTFX + UBR as untradable/delisted).

| Metric | Value |
|--------|-------|
| Sharpe | 2.030 |
| Max Drawdown | -14.89% |
| Calmar | 4.27 |
| Flat P&L | $33,263 |
| CAGR | 63.67% |
| Starting equity | $10,000 |
| Trades (2020–2026) | 2,695 |

### Strategy configuration

| Parameter | Value |
|-----------|-------|
| Universe | 58 symbols (C1 crypto, C2 gold/international/singles/VIX/GUSH, C3 broad leveraged ETFs + biotech + UNG) |
| ORB window | 30 min |
| Gap filter | 2% move on underlying (GAP_THRESHOLD) |
| PS filter k | 1.00σ (K_SIGMA) |
| Entry trigger | ORB high/low crossed on a **5-sec** real-time bar (`entry_at_boundary=True`); reactive by default. Optional pre-placed resting stop-limit mode (`use_resting_entries=True`) |
| Entry order | Marketable limit at `boundary ± (entry_buffer_orb_frac × ORB range)` — `entry_buffer_orb_frac=0.35`, auto-scales the fill room with each day's volatility (realized R:R ≈ 1.50 vs the 2.67 boundary-fill backtest) |
| EMA breakout gate | **Disabled** (`require_ema_confirmation=False`) |
| TP structure | TP1-only: 100% of position, class-based target (`tp1_target_multiple_by_class`) — **C1 (crypto) 2× ORB**, **C2/C3 1× ORB** — from boundary entry |
| Stop | midpoint of `orb_mid` and opposite ORB extreme (long: `(orb_mid + orb_low)/2`) |
| EOD flatten | 16:00 (`eod_exit_hour=16`, `eod_exit_minute=0`), lead `eod_flatten_lead_secs=60` |
| Base notional | **10% of live equity** per unit (`base_notional_pct=0.10`); falls back to fixed `v1_base_notional=$1,000` when equity is unavailable |
| Units cap | 20.0× (CAP_UNITS); gross exposure capped at `max_gross_exposure_pct=2.0` (200%) by the risk gate |
| Pruning | prune-all (skip-cheap-then-top-2 by prior daily close, applied to broad/BE as well) |
| Regime system | quiet (< 5 active underlyings), active (5–7), flood (≥ 8) |

**Operational upshot of the boundary-fill variant**: all three price levels (entry, stop, TP) are deterministic at 10:00 ET when the ORB completes. In the default reactive mode the entry limit is placed the moment a 5-sec bar crosses the boundary, with the stop and TP attached as OCA children; in `use_resting_entries` mode each candidate is instead submitted at 10:00 as a pre-placed stop-limit bracket. Either way, exits rest at IB (OCA stop-loss + take-profit) and residuals are flattened at ~15:59 ET (`eod_flatten_lead_secs`).

### Class × regime weight matrix

Structural 3-class × 3-regime causal sizing (locked 2026-06-20; cluster removed):

| Class | quiet | active | flood |
|-------|-------|--------|-------|
| **C1** (crypto) | 3.0 | 3.0 | 3.0 |
| **C2** (gold/intl/singles/VIX/GUSH) | 0.5 | 3.0 | 0.0 |
| **C3** (broad + sector pairs + biotech + UNG) | 3.0 | 3.0 | 3.0 |

`purity_skip = 1.0`. C2_flood=0 removes structurally unprofitable flood-day setups in the C2 class.

## Market Data Subscription Required

`IBClient.get_latest_quote()` raises `RuntimeError` if IB returns a degraded or
all-zero quote, rather than silently returning zeros and letting a session trade
on bad prices.

| Env var | Default | Purpose |
|---------|---------|---------|
| `IB_MARKET_DATA_TYPE` | `1` | IB data type: 1=live, 3=delayed |
| `IB_ALLOW_DELAYED_DATA` | (unset) | Set to `1` to permit delayed/zero quotes |

On `connect()`, IBClient registers an error handler for IB error code **10089**
(subscription required).  If 10089 fires, `_market_data_degraded` is set and
subsequent `get_latest_quote()` calls raise immediately.

`connect()` also calls `_validate_subscription()` which does a test quote on SPY.
If that raises `RuntimeError`, `connect()` translates it to `ConnectionError` so
the startup path fails loudly before any session begins.

To use delayed data intentionally (backtesting, monitoring without a live feed):

```bash
IB_ALLOW_DELAYED_DATA=1 IB_MARKET_DATA_TYPE=3 python -m orb_live.runner.main
```

## Operational Health

| Endpoint | Normal | Unhealthy |
|----------|--------|-----------|
| `/health` | 200 OK | 503 + reason |
| `/metrics` | Prometheus text | — |

During RTH, `/health` returns 503 if:
- Last bar received > 90 seconds ago
- Broker API not called in > 5 minutes
- `state_store.all_open_positions()` raises

## Operations

### Sigma calibration

The prior-session filter uses sigma values (daily-return volatility) of each
underlying, thresholded at `sigma × ps_filter_k` (k = 1.00). Live loads these
**verbatim** from the vendored long-history file
`orb_live/strategy/data/sigma_master.csv` — the same multi-year file the
backtest uses. There is **no** rolling 30-day recompute at startup and nothing
is written to `sigma_runtime.yaml` (that path diverged 2–3× from the backtest
for some underlyings and silently shifted PS thresholds; it was removed).

Sigma methodology: `sigma = close.pct_change().dropna().abs().std()` over the
**full adjusted** (split/dividend-adjusted) daily history. Adjusted prices are
deliberate — raw closes carry reverse-split-day jumps that inflate sigma 2–3×
for names like XOP/GDXJ/MSTR.

To refresh the CSV (self-contained, yfinance only — previews by default):

```bash
python -m orb_live.scripts.refresh_sigma_master            # preview universe
python -m orb_live.scripts.refresh_sigma_master --confirm  # write
python -m orb_live.scripts.refresh_sigma_master --all      # every CSV row
```

Because sigma is measured over many years, a few weeks of new data barely moves
it — run this monthly/quarterly so values drift slowly without window noise.
Recomputed rows are merged into the CSV, preserving untouched underlyings; a
`>20%` move vs the current value is flagged (re-check the backtest if
unexpected). Parity between live and the CSV is locked by
`test_sigma_source_parity.py`. Per-symbol overrides can still be layered via
`config/sigma_override.yaml`.

## Tests

```bash
python -m pytest orb_live/tests/ -v
```

453 tests in ~1 second.  All synchronous — no broker credentials required.

Key test files:
- `test_session_runner.py` — bar dispatch order invariant, state machine, 5-sec entry path, equity-scaled sizing
- `test_half_day_session.py` — half-day schedule correctness
- `test_operational.py` — health server, alerts, daily rollup, drift detection
- `test_end_to_end_replay.py` — full session replay with synthetic bars
- `test_signal_parity.py` — backtest ↔ live parity for indicators and filters
- `test_sigma_source_parity.py` — live sigma is the CSV, bit-for-bit (no rolling recompute)
- `test_order_policy.py` — ORB-range entry buffer math

## Equity curves (2020–2026)

Flat ($1k fixed notional per unit multiplier) and compounded (notional scales
with equity) equity curves for the locked v1 boundary-fill variant on the
58-symbol live-parity universe. Drawdowns shown below each curve.

![v1 equity curves — flat and compounded, 2020–2026](docs/img/v1_boundary_equity_curves.png)

- Flat: $10k → $43,263 (+$33,263, +332.6%), MaxDD -8.33%
- Compounded: $10k → $234,412 (23.44×, 63.72% CAGR), MaxDD -14.89%
