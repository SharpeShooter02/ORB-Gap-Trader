# Claude Code prompt — gut the Alpaca-era execution layer, rebuild the live runner IBKR-native

This repo is a v1 ORB live trader. The **strategy brain is correct and
parity-validated**, but the **execution/runner layer is inherited Alpaca
plumbing** (WebSocket streaming with threads/token-refresh, a marketable-limit
repeg policy, hand-rolled fill/OCA management, `alpaca_*` columns) retrofitted to
`ib_async`. That mismatch is the source of nearly every bug: off-thread
subscribe, blocking-wait re-entrancy, repeg overshoot, undetected fills, OCA
cancel races, botched EOD flatten. We are ripping out that layer and rebuilding
it IBKR-native, **without touching the validated strategy.**

Target broker: **Interactive Brokers via `ib_async` only.** Remove all Alpaca.

Work in **staged phases, each its own commit. Do Phase 0 and Phase 1 first and
STOP for review before deleting anything.**

---

## PRESERVE — do NOT modify logic in these (they're validated; changing them
## re-opens solved bugs: look-ahead, inverse-ETF sign flip, skip-cheap N==2,
## FNGU/FNGD→QQQ, BTFX drop, data-freshness)

- `orb_live/strategy/v1_strategy.py` (plan_session, gap/PS/ORB/skip-cheap, regime,
  sizing, WEIGHTS, class taxonomy, EXPLICIT_DROPS/FORCE_INCLUDE)
- `orb_live/strategy/data/master_universe.csv`, `sigma_master.csv`
- sigma computation + `core/calendar.py` (holiday/`prev_trading_day` logic)
- the strategy/parity tests that assert v1 behavior (universe, PS, skip-cheap,
  regime) — keep them green

---

## PHASE 0 — safety (do first)
- Create a branch `rebuild/ibkr-native`. Commit the current working tree as-is
  first, so everything is reversible.

## PHASE 1 — inventory & classify (STOP for review after this)
- Walk every file under `orb_live/` (and top-level). For each, classify:
  **KEEP / DELETE (Alpaca-era or dead) / REBUILD**, with a one-line reason.
- Explicitly flag every Alpaca remnant: `.env`/`.env.example` Alpaca keys,
  `BarRouter` WebSocket/token/rest-poll machinery (`WS_TOKEN_REFRESH`,
  `_ws_connected_at`, `_token_refresh_watcher`, `_rest_poll_loop`, `_stream_loop`),
  `alpaca_id`/`alpaca_order_id` columns and refs, the `MarketableLimitPolicy`
  repeg logic, any `reference/` old-config backtester, `DryRunBroker` if
  Alpaca-shaped, unused scripts/ops.
- Write **`ARCHITECTURE.md`**: the intended clean structure and what each KEEP
  file does (the operator does not currently understand the layout — this doc is
  a deliverable). Then STOP and report the classification before deleting.

## PHASE 2 — delete the cruft
- Remove everything classified DELETE. Strip Alpaca from `.env`/`.env.example`
  and dependencies (`pyproject.toml`/requires). Drop `alpaca_*` columns from the
  schema. The result should import and run with only `ib_async` + data/strategy
  deps.

## PHASE 3 — rebuild execution IBKR-native (the core of this work)
Replace the deleted execution layer with a minimal, ib_async-native design.

1. **Bar streaming** — `reqRealTimeBars` on the **main thread**; deliver 5-sec
   bars via the event loop, aggregated to 1-min by `BarAggregator`. NO background
   threads, NO WebSocket/token machinery. The runner's waits use `ib.sleep()` so
   the loop is pumped. (Confirmed: off-thread subscribe or `time.sleep` = zero
   bars. And `BarAggregator` must read `bar.open_` — ib_async's RealTimeBar field
   — not `bar.open`.)
2. **Orders — use IB NATIVE BRACKET orders.** For each fire: a **parent entry**
   (a **market order** at the breakout, matching the backtest's "market on close
   of the breakout bar") with **two OCA children** — TP1 limit and protective
   stop — submitted transmit-chained so IB holds the children until the parent
   fills, then activates them and cancels the sibling on fill. This replaces the
   repeg policy, the manual fill detection, the manual OCA cancel, and the
   overshoot/stacking — IB owns the lifecycle. Round ALL prices to the contract's
   `minTick` (`reqContractDetails().minTick`, cached). Set `order.tif` explicitly
   (`'DAY'`) on every order so IB doesn't cancel via preset (Error 10349).
3. **Fills & positions from broker truth.** Subscribe to `ib.execDetailsEvent`/
   `trade.fillEvent` to record fills; treat the **broker's positions** as the
   source of truth, not an in-memory guess. Persist positions/fills to the state
   store on the actual fill.
4. **Re-entry guard.** Block a new entry on any symbol with an open broker
   position or a non-terminal order.
5. **EOD flatten** — start **before** the close (e.g. 15:55 ET) with a hard
   deadline; **reconcile against the broker** and close every position it holds
   in the universe (not just tracked ones); use market orders with explicit TIF
   that actually execute before the bell. Also run this reconcile-and-adopt/flatten
   at **startup** so a restart never leaves stale positions unmanaged.
6. **Session runner** — pre-market scan → ORB window → post-ORB watch → EOD.
   Drive BOTH the candidate watch-list AND sizing from a **single `plan_session()`
   call** (no parallel phase-1/phase-2 gate; that divergence caused wrong regimes
   and double computation). Subscribe real-time bars only for the day's
   **candidates** at 10:00, not the whole universe.
7. **Sizing** — apply v1 multipliers × base_notional from `plan_session`, and make
   it **margin-aware**: check available funds/buying power before placing and
   skip/downsize rather than firing an order IB will reject (leveraged ETFs margin
   near 1:1 at IB, so the 200% cap needs adequate equity).

## PHASE 4 — fold in the known operational fixes
- Liquidity Gate C stays disabled (not part of v1).
- Daily report must handle an empty/no-closed-trades day without crashing
  (`idxmax` on all-NA).
- Schema: an init-time migration that adds missing columns to an existing DB.
- Logging: set `ib_async` loggers to WARNING so portfolio/orderStatus spam is
  suppressed; keep the runner's own structured events at INFO.

## PHASE 5 — tests that exercise REAL paths (not mocks that mirror bugs)
This is non-negotiable — false-green mocks hid every bug this project.
- Streaming/fill/bracket tests must run on a **real asyncio loop** with
  **real-shaped `RealTimeBar`/fill objects** (a synchronous mock won't catch the
  loop, thread, `open_`, or async-fill bugs).
- End-to-end: a simulated breakout → market entry fills → bracket (TP1+stop) rests
  → TP1 or stop fills → sibling cancelled → EOD reconcile closes anything left.
- Keep all v1 strategy/parity tests green.

---

## GUARDRAILS
- IBKR/`ib_async` only. No Alpaca, no cross-folder imports from `../v1/`.
- Do NOT change `v1_strategy.py` logic or the vendored data/CSVs.
- Commit per phase; keep the strategy parity tests green throughout.
- Prefer IB-native primitives (bracket/OCA orders, event loop, execDetails) over
  hand-rolled order management — smaller and fewer places to be wrong.
- After Phase 5: summarize what was deleted, the new file layout (in
  ARCHITECTURE.md), and confirm a full end-to-end test passes on a live loop.
```
