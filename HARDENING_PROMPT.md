# Claude Code prompt — pre-live execution hardening

Paste the block below into Claude Code, running inside this repo. It assumes the
v1 migration is complete and `pytest orb_live/tests/ -q` is green (328 passing).
Work in small commits; keep the parity gate passing after every change.

---

The v1 migration is done and all tests pass, but three execution paths are only
exercised against a live broker and need hardening before any paper run. Read
`execution/position_manager.py` (`on_bar` TP1 block ~L406-481, `_exit_partial`,
`_exit_all`, `reconcile_from_broker`), `execution/order_policy.py`, and
`data/ib_client.py` (`submit_stop_order`, `modify_stop_order`, `cancel_order`,
`get_order`) before changing anything. Then implement:

## 1. Replace the client-side TP/stop race with a native IB bracket (OCA)
Today the entry, TP1 limit, and protective stop are managed as independent
orders, and on a TP1 full-exit the code submits the TP1 exit and then cancels
the stop in a separate call — a race where a single bar can fill both and flip
the position. Change entry submission so the TP1 exit and the stop are placed in
one **IB OCA group** (one-cancels-all) via `ib_async` (set `ocaGroup`/`ocaType`,
or use a parent bracket with `transmit` chaining). When one leg fills, IB must
cancel the sibling at the exchange — no client-side cancel race. Update the TP1
full-exit block to rely on OCA cancellation rather than an explicit
`cancel_order`, and update `_recover_stop_after_modify_failure` accordingly.
Add a unit test that simulates TP1 fill and asserts the stop leg is cancelled
(mock the broker's OCA behavior).

## 2. Stop treating a failed cancel as success
In the TP1-only early-close block, `cancel_order` is wrapped in
`except Exception: pass` and the position is then marked closed and deleted from
`self._positions`. If the cancel fails (or the stop already triggered), this
orphans a live stop and creates a phantom position. Change this so:
- `cancel_order`'s return value (and/or a `get_order` status re-check) is
  verified.
- On failure, emit a `CRITICAL` alert via the ops alert path and DO NOT delete
  the position; leave it for `reconcile_from_broker` to resolve on the next
  cycle, then close it once the broker confirms flat.
- Add a test for the cancel-failure branch (broker returns False / raises):
  assert an alert fires and the position is retained, not deleted.
(If item 1's OCA approach is adopted, this logic moves to verifying the OCA
sibling actually cancelled; keep the same fail-loud-and-reconcile behavior.)

## 3. Record the realized fill price, not just the target
`save_closed_trade` logs `exit_price = pos.tp1_price` (the target), so the live
blotter can't show real slippage. Keep `record_realized_pnl` using the actual
fill (it already does), but also persist the actual `fill.avg_price` from
`_exit_partial`/`_exit_all` — e.g. add a `realized_exit_price` column (keep
`exit_price` = target for backtest parity, add the realized field alongside).
Update the closed-trade write path and one test.

## 4. Add a live-readiness preflight (delayed-data dress rehearsal)
These fail only against IB and must be checked before a real-time run. Add a
`tools/preflight_check.py` (or extend the existing one) that, against a paper
IB connection with delayed data (`IB_ALLOW_DELAYED_DATA=1 IB_MARKET_DATA_TYPE=3`):
- Qualifies every universe contract and reports which tickers DON'T resolve
  (expect issues on newer ETFs: UXRP, XRPT, NVDU, AMDL, BITX, etc.). Untradable
  names should be logged loudly, not silently skipped.
- Verifies yfinance returns ≥2 prior closes for every PS-filter underlying
  (crypto as BTC-USD etc., VIX as ^VIX); reports fallbacks to seed sigma.
- Confirms the sigma source is wired: rolling `{ul}.parquet` vs the vendored
  `sigma_master.csv` — make the resolution order explicit and logged.
- Estimates simultaneous market-data lines needed and confirms streaming is
  scoped to the day's CANDIDATES, not the full ~60-symbol universe.
- Adds basic pacing/throttle handling (or a guard + retry) around the burst of
  `reqHistoricalData` calls at 9:30/10:00.
Wire it as `python -m orb_live.tools.preflight_check` and document it in
`DEPLOYMENT.md` as a required step before the first real-time session.

After each item: run `pytest orb_live/tests/ -q` and confirm still-green plus the
new tests. Do not weaken the §4 parity gate to make anything pass.
