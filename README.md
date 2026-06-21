# ORB Live Trading System

Automated live execution of the Opening Range Breakout strategy across a
universe of leveraged and inverse ETFs.  Uses Interactive Brokers (IBClient)
for order routing and WebSocket bar streaming.

## Architecture

```
orb_live/
├── config/           LiveConfig + StrategyConfig (all tunables in one place)
├── core/
│   ├── clock.py      MarketClock — RTH detection, half-day schedule, effective_close()
│   └── state_store.py  SQLite persistence (20 tables via SQLAlchemy Core)
├── data/
│   ├── ib_client.py       IB Gateway REST + WebSocket wrapper (IBClient)
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
│   ├── bar_router.py    WebSocket streaming + missed-bar replay + token refresh
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
python -m pytest orb_live/tests/ -q   # 150 tests, ~3s

# 4. Start session
python -m orb_live.runner.main --paper

# 5. Monitor
curl http://localhost:8080/status | python -m json.tool
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for production VPS setup.

## Bar Dispatch Order (load-bearing invariant)

Within `SessionRunner._on_bar_dispatch`, bars are always processed in this order:

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

| Parameter | Value |
|-----------|-------|
| Sharpe (backtest 2017–2024) | 2.717 |
| Max Drawdown | -8.0% |
| Calmar | 11.259 |
| Universe | 10 instruments (TQQQ/SQQQ/UPRO/SPXS/URTY/TZA/UDOW/SDOW/FNGU+FNGD, LABD/LABU, BITX, BOIL/KOLD, UVXY, KORU, ETHU/ETHD) |
| TP weights | 0.35 / 0.05 / 0.60 |
| EMA period | 30 |
| RTG exclusion | Class A only (KOLD excepted) |

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
underlying. The live system computes these dynamically from full historical
data at startup, and refreshes them weekly via cron.

- **Seed values:** `reference/_production_run.py`'s `SIGMA` dict — used as
  fallback when historical parquet data is unavailable.
- **Live values:** `data/sigma_runtime.yaml` — auto-generated at every
  startup and after each weekly recalibration. Human-readable; safe to inspect.
- **History:** `state_store.sigma_history` table — append-only record of
  every sigma computation, for forensic analysis.

To force frozen-seed mode (for backtest parity validation):

```bash
USE_ROLLING_SIGMAS=0 python -m orb_live.runner.main
```

To manually recalibrate now:

```bash
python -m orb_live.scripts.recalibrate_sigmas --confirm
python -m orb_live.scripts.recalibrate_sigmas --confirm --db /path/to/orb_live.db
```

Underlyings whose parquet is missing will fall back to seed values and emit a
`WARN`. Run `backfill_underlyings` if WARNs appear frequently:

```bash
python -m orb_live.scripts.backfill_underlyings
```

## Tests

```bash
python -m pytest orb_live/tests/ -v
```

150 tests in ~3 seconds.  All synchronous — no broker credentials required.

Key test files:
- `test_session_runner.py` — bar dispatch order invariant, state machine
- `test_half_day_session.py` — half-day schedule correctness
- `test_operational.py` — health server, alerts, daily rollup, drift detection
- `test_end_to_end_replay.py` — full session replay with synthetic bars
- `test_signal_parity.py` — backtest ↔ live parity for indicators and filters
