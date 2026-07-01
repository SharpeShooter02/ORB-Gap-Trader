# Claude Code bug-fix prompt — v1 live runner (2026-06-23)

Paste into Claude Code, running in this repo. These are issues found during live
paper sessions on 2026-06-22 and 2026-06-23, several confirmed against the v1
backtest source. Do tasks in priority order, **commit after each**, run
`pytest orb_live/tests/ -q` after every change, never introduce look-ahead, and
keep the parity gate green. Where a fix must match the backtest, match the
**specific** function named — not a similarly-named variant.

Authoritative spec for this project: `../v1/V1_SYSTEM_REPORT.md` and the v1
backtest scripts in `../v1/scripts/` (read-only reference; do NOT add a runtime
import to them — this repo is vendored/self-contained).

---

## BUG 0 (P0 — DO FIRST; gates the entire strategy) — breakout detection never runs

**The strategy has never attempted a trade in any live paper session.** Across
multiple sessions, `breakout_signal`, `pending_orders`, `fills`, `open_positions`,
and `closed_trades` are all EMPTY — while `candidates` shows fully-qualified
watched setups (e.g. 2026-06-24 XRPT: phase 2, direction -1 short,
ps_filter_passed=1, preflight_passed=1, decision='candidate'). ~16 candidate-days,
zero breakout detections. The detection loop is not firing.

**Root cause (confirm, then fix):** `SessionRunner._run_pre_market()` calls
`self._router.subscribe(list(self._cfg.symbols))` — it subscribes real-time bars
for ALL ~59 symbols at the open. That likely exceeds IB's simultaneous
market-data line cap, and IB then silently delivers NO bars rather than erroring,
so `_on_bar_dispatch` → `engine.on_bar()` never runs and no ORB breakout is ever
evaluated. The preflight already warns: "subscribe real-time bars per-CANDIDATE
at 10:00 ET, not at open." The code does the opposite.

**Fix:**
1. Subscribe to real-time bars only for the **day's candidates**, at ORB close
   (~10:00 ET), not all 59 symbols at the open. (Pre-open you only need the data
   for gap/ORB seeding, which can come from historical bars, not 59 live streams.)
2. Add a **bars-received heartbeat + RTH watchdog**: count bars delivered per
   watched symbol; if zero bars arrive for any watched candidate during RTH, emit
   a CRITICAL alert. A dead detection loop must never again hide behind a quiet
   `waiting_for_eod`.
3. Verify `_on_bar_dispatch` → `engine.on_bar` is actually invoked (add a debug
   counter / log) and that `reqRealTimeBars` is delivering for the candidates.

**Test:** an end-to-end test that feeds streamed 1-min bars crossing `orb_low`/
`orb_high` for a watched candidate **through the live bar-dispatch path** (not a
directly-mocked engine) and asserts a `breakout_signal` row is written AND an
order is placed. The current tests pass while the real bar path delivers nothing —
they mock the engine and never exercise `reqRealTimeBars` → dispatch → on_bar.

---

## BUG 0b (P0 — THE actual blocker) — blocking sleep starves the ib_async event loop; reqRealTimeBars never delivers

After Bug 0's candidate-only subscription, the watchdog still fired CRITICAL for
**all 8** candidates ("No bars 5 min post-ORB"). 8 subscriptions is far under any
line cap, so it is NOT a line-limit problem — `reqRealTimeBars` is delivering
nothing.

**Root cause:** `ib_async` is asyncio-based; `reqRealTimeBars` `updateEvent`
callbacks only fire while the event loop is being pumped. The runner blocks its
thread with `time.sleep()` (`SessionRunner._sleep` in `orb_window_wait` and
`waiting_for_eod`), which is the *same thread* running IB's loop — so during the
trade window the loop never processes incoming bar messages and no callback
reaches `_on_bar_dispatch`. Evidence: `updatePortfolio` messages arrive only in a
single burst when active code briefly runs the loop, then total silence through
the blocking sleep. This is the canonical ib_async gotcha: **use `ib.sleep()`,
never `time.sleep()`, while streaming.**

**Fix:** replace the blocking sleeps in the runner's wait phases with
`ib.sleep(seconds)` (which pumps the loop while waiting), OR run the IBClient on a
dedicated event-loop thread so streaming continues regardless of the main
thread. Keep the loop alive for the entire ORB-watch / wait-until-EOD window.

**Tests (must actually run the loop — see "how to test" below):**
1. **Live probe** (run during market hours, any day — does not need the open):
   a ~15-line script that connects, `reqRealTimeBars` on SPY + one candidate, and
   `ib.sleep(60)` in a loop; assert bars print. Swapping `ib.sleep` for
   `time.sleep` must produce zero bars (reproduces the bug).
2. **Regression test (no market):** drive the runner's wait phase on a real
   asyncio loop with a fake IB that schedules a bar event during the wait; assert
   it reaches `_on_bar_dispatch`. A pure mock will NOT catch this — the test must
   exercise blocking-vs-pumping on an actual loop. (This is the gap `test_k` left:
   it hand-feeds `_on_bar_dispatch` and never runs the loop.)

---

## BUG 0d (P0 — THE ACTUAL ROOT CAUSE of zero bars) — wrong field name on RealTimeBar

`BarAggregator.add_5sec_bar` reads `bar.open`, but ib_async's **`RealTimeBar`
names that field `open_`** (trailing underscore; only `open` is renamed — `high`,
`low`, `close`, `volume`, `time` are normal). Every real 5-sec bar therefore
raises `AttributeError` inside the `reqRealTimeBars` `updateEvent` handler, which
ib_async swallows — so the whole stream fails **silently**: zero bars, no error.

Why it hid for so long / fits all evidence:
- Raw `reqRealTimeBars` probe worked → its callback only touched `.time`/`.close`.
- The aggregator unit test passed → it used a fake bar with `.open`, matching the
  buggy code (false green).
- ORB/historical works → `reqHistoricalData` returns `BarData`, which has `.open`.
- Only `subscribe_bars` feeds real `RealTimeBar` objects into `add_5sec_bar`.

Proven by isolation: raw `reqRealTimeBars` on SOXL/SOLT streams every 5s, while
the identical symbols through `subscribe_bars` deliver nothing (same clientId,
same moment, runner running).

**Fix** in `add_5sec_bar` (both the first-bar and new-minute branches):
```python
_open = getattr(bar, "open_", None)
if _open is None:
    _open = bar.open      # fallback for BarData / synthetic test bars
self.open = float(_open)
```

**Test:** the aggregator regression test must use an object with `open_` (a real
RealTimeBar shape), not `open` — otherwise it keeps giving a false pass. Better:
an end-to-end test that runs an actual `reqRealTimeBars` stream (or a faithful
RealTimeBar mock) through `subscribe_bars` and asserts a dispatched bar.

---

## BUG 0e (P0 — order path) — order submission blocks the event loop inside the bar callback

After 0d, a real breakout fired and `placeOrder` reached IB (BOIL SELL, status
PendingSubmit) — but the runner immediately threw
`entry_order_exception: "This event loop is already running"`.

**Cause:** the order-submission path (`submit_limit_order` →
`_wait_for_submit_terminal`, which does a blocking `ib.sleep`/run to wait for
terminal status) is invoked from *inside* the `reqRealTimeBars` `updateEvent`
callback (`_on_bar_dispatch` → `engine.on_bar` → `open_position` → submit). That
callback already runs on the event loop, so a nested `ib.sleep`/run raises "loop
already running." `placeOrder` itself worked (non-blocking); only the synchronous
wait blew up.

**Fix:** make order submission **non-blocking** in the callback context.
`ib.placeOrder(...)` and return; confirm fill asynchronously — either attach
`trade.statusEvent`/`filledEvent`/`fillEvent` handlers, or rely on the position
manager's existing per-bar `get_order` polling to detect the fill on subsequent
bars. Never call `ib.sleep`/`ib.run`/`_wait_for_submit_terminal` from within a bar
(or any ib_async event) callback.

**Test:** drive a breakout through the real callback path on an actual loop and
assert the order is placed with no re-entrancy exception.

---

## BUG 7 (P1 — order path) — limit/stop prices not rounded to the contract min tick

Same breakout: `Error 110: The price does not conform to the minimum price
variation for this contract.` The limit was `26.603369999999998` — sub-penny —
and BOIL's tick is $0.01, so IB rejected it.

**Cause:** `MarketableLimitPolicy._compute_limit` computes `reference × (1 ± bps)`
with full float precision and never rounds to the contract's tick.

**Fix:** round every submitted price (entry limit, exit limit, stop, and both OCA
bracket legs) to the contract's `minTick`. Fetch it once via
`reqContractDetails().minTick` and cache per symbol; fallback to $0.01 / 2 dp for
standard US ETFs. Round in the direction that stays marketable (round a buy limit
up, a sell limit down, etc.).

**Test:** assert all computed prices are exact multiples of the symbol's minTick.

---

## BUG 8 (P0 — order path) — failed entry leaves a stale open_positions row, blocking re-entry

A real BOIL breakout fired and crashed with
`sqlite3.IntegrityError: UNIQUE constraint failed: open_positions.symbol`.
`open_position` persists an `'entering'` row *before* submitting the order, and
`open_positions` is UNIQUE on `symbol`. A prior failed entry (yesterday's BOIL,
killed by the re-entrancy/tick bugs) left its `'entering'` row behind, so the next
attempt for the same symbol collides.

**Fixes:**
1. **Delete the `'entering'` row on any entry failure/abort.** The persist-then-
   submit path must roll back its row if submission throws or the order is
   rejected/cancelled — no zombie rows.
2. **Reconcile at session start.** On startup, compare `open_positions` against
   the broker's actual positions and purge orphans (DB rows with no matching
   broker position), and/or clear any non-`'open'` (`'entering'`) rows. Don't make
   `--recover` optional for this — stale rows must never block a fresh session.
3. **Close the memory-vs-DB gap.** The risk gate's "already_in_position" check
   reads the in-memory position manager, which is empty on a fresh process while
   the DB still holds a row. Either check the DB too, or guarantee #2 runs first.
4. Make `_persist_new` robust to a pre-existing row (delete-then-insert / upsert)
   as a backstop.

**Test:** simulate a failed entry (submission raises), assert no row remains in
`open_positions`; and a startup reconcile that purges a stale `'entering'` row
with no matching broker position.

---

## BUG 11 (P0 — CRITICAL) — fills are never detected → unmanaged, unprotected, oversized overnight positions

**Confirmed 2026-06-30.** IB held SBIT 70, AMDL 62, BOIL 267 (real fills) with
**no stop/TP orders and no working orders**, while the runner's `fills` table
marked every entry `'unfilled'` (qty 0, avg 0), `open_positions` was empty, and
the report said "0 trades." Orders filled asynchronously at IB; the runner read
`filled=0` at `PendingSubmit`, recorded `'unfilled'`, and never reconciled.

Consequences (all dangerous live):
- No OCA bracket placed → positions had **no stop, no TP**.
- EOD `flatten_all` iterates *tracked* positions (empty) → **nothing flattened,
  positions held overnight** on leveraged ETFs.
- **Re-entry stacking:** BOIL 267 = 3×89 — repeg churn + failed cancels
  (10147/10148) filled multiple orders; the runner saw "no position" and
  re-entered each time.

Root: the non-blocking submission (Bug 0e) removed the blocking wait but never
added async fill detection.

**Fixes:**
1. **Detect each order's real terminal outcome.** After placing non-blocking,
   track the order via `trade.filledEvent`/`statusEvent` (preferred) or poll
   `get_order` until `Filled`/`Cancelled`. On fill: write the real fill (qty,
   avg price), create the `open_position`, and **place the OCA bracket**. Never
   write `'unfilled'` until the order is actually terminal-and-not-filled.
2. **Confirm cancels before repeg.** The marketable-limit repeg must verify the
   prior order reached `Cancelled` at IB before placing the next; if the cancel
   failed (order working/filled), reconcile the fill instead of placing another.
   This stops the 3× stacking.
3. **Re-entry guard from broker truth.** Block re-entry on a symbol with an open
   broker position OR a non-terminal order — check broker/order state, not just
   in-memory `self._positions`.
4. **EOD flatten + startup reconcile from the broker.** Flatten every position
   the BROKER holds in the universe (`reconcile_from_broker`), not just tracked
   ones, so undetected fills still close. Run reconcile at startup too (ties to
   Bug 8). This is the safety net that must never be optional.

**Test:** an order that fills *after* submission → assert the runner records the
fill, creates the position, and places the bracket; and that EOD flatten closes a
broker position the manager never tracked. (A mock that returns Filled
synchronously won't catch this — the fill must arrive asynchronously.)

---

## BUG 0c (P0 — CONFIRMED ROOT CAUSE of no bars) — BarRouter subscribes off-thread

**Reproduced in isolation.** A probe calling `reqRealTimeBars` + `ib.sleep` on
the **main thread** streams bars every 5s. The identical probe with only the
`reqRealTimeBars` call moved onto a `threading.Thread` streams **nothing**.
`ib_async` is not thread-safe — a streaming subscription registered off the loop
thread never gets its callbacks serviced.

The runner hits exactly this: `BarRouter.subscribe()` spawns a daemon thread
(`_stream_loop`) and calls `broker.subscribe_bars()` there, expecting it to block
(the comment literally says "blocks until the stream terminates"). That's Alpaca
WebSocket plumbing — `WS_TOKEN_REFRESH`, `_ws_connected_at`,
`_token_refresh_watcher`, `_rest_poll_loop` are all WS-era. But
`IBClient.subscribe_bars` uses `reqRealTimeBars`, returns immediately, and
delivers bars via the **main-thread event loop**. So the subscription is stranded
on the wrong thread and no bar ever reaches `_on_bar_dispatch`.

Diagnostic that proves it's not entitlement/connection: in the runner, ORB
high/low (built from `reqHistoricalData`, a request→reply call on the main
thread) and portfolio updates (IB auto-push) both work — only the off-thread
streaming subscription fails.

**Fix:** subscribe on the MAIN thread. `BarRouter.subscribe()` should call
`broker.subscribe_bars(symbols, self._on_stream_bar)` directly/synchronously (it
returns immediately for IB), with NO `_stream_loop` thread. The runner's
`ib.sleep` waits (Bug 0b) pump the loop and deliver the `updateEvent` callbacks →
`_dispatch` → `_on_bar_dispatch`. Remove/guard the WS token-refresh / `_stream_loop`
/ `_rest_poll_loop` machinery — it's Alpaca-only and actively breaks the IB path.
Keep `register_listener`/`_dispatch` fan-out and missed-bar replay.

**Follow-up (not blocking):** handle ib_async `disconnectedEvent`/reconnect to
re-issue `reqRealTimeBars` on reconnect (this replaces the WS-token reconnect
logic the old code used).

**Test:** the live watchdog staying quiet is the real confirmation. Also keep the
threaded-probe reproduction documented, and the Bug 0b loop-based regression test
should subscribe through the same path the runner uses (not a directly-mocked
engine).

---

## BUG 1 (P0) — skip-cheap keeps 2 when it should keep 1 at N==2

**Confirmed against backtest.** The locked run uses
`../v1/scripts/skip_cheap_by_class.py :: skip_cheap_then_top2_when_3plus()`,
whose rule is (it literally prints `"rule: skip-cheap @ N=2, top2 @ N>=3"`):
- N == 1 → keep 1
- N == 2 → **drop the cheaper, keep 1** (classic skip-cheap)
- N >= 3 → keep top 2 by prior close

The vendored live code in `orb_live/strategy/v1_strategy.py :: compute_candidates()`
does `group_sorted[:2]`, i.e. it keeps **both** when N==2. That over-trades every
2-vehicle underlying (today: XRP→UXRP/XRPT, QQQ→SQQQ/TQQQ, SOXX→SOXL/SOXS,
XLK→TECL/TECS) and doubles same-direction exposure (UXRP and XRPT are *both*
2x-long XRP). It also inflates candidate count and depresses cap_factor vs the
locked numbers. The manual's §5 ("if 2 fire → drop the cheaper one") is correct;
the "keep up to 2" wording in §1/config-glance is loose.

**Fix** (in `compute_candidates`, the per-UL pruning loop):
```python
group_sorted = sorted(group, key=lambda c: -c.prior_close)
keep_n = 1 if len(group) == 2 else 2   # N==2: skip-cheap (keep 1); N>=3: top-2
final.extend(group_sorted[:keep_n])
```
**Test:** add a parity test that runs the pruning on UL groups of size 1, 2, 3,
and 4 and asserts kept counts of 1, 1, 2, 2 and that the kept symbols are the
highest by prior close. Reference the backtest rule in a comment (the backtest
script is not importable from this repo).

---

## BUG 2 (P0) — underlying data is stale at runtime; no boot refresh, no freshness gate

**Evidence (2026-06-23 session):** equity underlyings used `c_t1 d_t1=2026-06-18`
(Thursday) on a **Tuesday** session — Monday 6/22 was missing, so gaps were
measured Thu→Tue and inflated, pushing the regime to **flood** (n_uls=14) and
zeroing all C2 names. Crypto had Monday but still skipped Sunday
(`BTC c_t2=2026-06-20 Sat`, jumping 6/21). The two data paths even had different
freshness. This has been distorting gaps/regime for multiple sessions.

**Fix:**
1. **Refresh underlying data at startup**, before pre-market, for every universe
   underlying (equity + crypto). Don't rely on a manual/cron backfill.
2. **Fail-loud freshness assertion** at boot (abort the session, alert CRITICAL,
   do NOT trade on stale data):
   - equity ULs must be current through `prev_trading_day(today)` (use
     `core/calendar.py`);
   - crypto ULs must be current through the prior **calendar** day, i.e. include
     Saturday AND Sunday — the Sunday bar must be present on a Monday.
3. Add a test for the freshness gate: stale equity (missing prior trading day)
   and missing-Sunday crypto each cause a loud abort, not a silent stale run.

This is the root cause behind the inflated gaps/regime — fix it before trusting
any regime classification.

---

## BUG 3 (P1) — old-framework liquidity gates still present (not in v1)

The per-trade `PreFlightCheck` "Gate C" in `signals/liquidity.py`
(min_dollar_volume_floor / max_pct_of_adv / min_yesterday_dv_ratio) is
old-framework; v1's only liquidity filter is the universe-build ADV floor. On
2026-06-22 it blocked every candidate.
1. Neutralize via config — set LiveConfig defaults `min_dollar_volume_floor=0.0`,
   `max_pct_of_adv=1.0`, `min_yesterday_dv_ratio=0.0`. Keep the metric logging.
   (If already done, verify and move on.)
2. Fix the latent partial-bar bug regardless: `yesterday_dv = bars["dv"].iloc[-1]`
   grabs the in-progress current-day bar intraday. Use only sessions with
   `date < session_date` for both `yesterday_dv` and the ADV window. Add a test.

---

## BUG 4 (P2) — investigate & document only (no refactor yet)

The runner computes candidates twice: v1 `plan_session()` (drives regime /
cap_factor / multipliers) and the old `PreMarketJob` phase-1/phase-2 gate (drives
the actual watch list). They can diverge. Also `LiveConfig.baseline_pct is None`,
so fire-time sizing may not use v1's `multiplier × base_notional` path. Trace
which candidate set and which sizing actually reach order placement, and write up
(`NOTES.md`) the plan to make the watch list AND sizing both derive from a single
`plan_session()` call. **Do not change behavior in this task** — document only.

---

## BUG 5 (P1) — no schema migration; model column additions break existing DBs

A live session crashed at EOD with `no such column: closed_trades.realized_exit_price`.
The column was added to the model during hardening, but `init_db` only creates
missing tables — it never `ALTER`s existing ones — so any `live.db` created before
the column existed lacks it, and the EOD report's `SELECT` crashes. The test
suite never catches this because it builds a fresh DB (current schema) every run.

**Fix:**
1. Add a lightweight init-time migration: after `init_db`, for each table compare
   the model's columns against `PRAGMA table_info` and `ALTER TABLE ... ADD COLUMN`
   any that are missing (SQLite supports add-column). Run it on startup so model
   additions auto-apply to existing DBs. (Or adopt Alembic.)
2. Immediate hotfix for the current DB:
   `ALTER TABLE closed_trades ADD COLUMN realized_exit_price FLOAT`.
3. **Test:** open a DB created with an older schema (drop a column), run init, and
   assert the migration adds it back and the EOD report query succeeds. This is
   the path the fresh-DB tests never exercise.

---

## Guardrails
- Commit after each bug; `pytest orb_live/tests/ -q` green after each.
- Don't weaken the parity gate to make tests pass — fix logic, then update
  expected values only where the logic became *more* correct.
- No look-ahead. No runtime imports from `../v1/`.

## Already fixed this session (regression-check only, don't redo)
- Inverse-ETF `ul_gap` sign flip in `run_phase1` (was corrupting direction).
- Crypto PS filter reverted to raw parquet (weekend bars) for backtest parity.
- FNGU/FNGD/BULZ → QQQ underlying; BTFX added to EXPLICIT_DROPS.
- Preflight: subscription flag honored, pacing check fixed, calendar-aware
  staleness, correct backfill command; `MSOS` mapped + `_yf_ticker` identity default.
