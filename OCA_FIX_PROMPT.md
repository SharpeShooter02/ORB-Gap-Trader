# Claude Code prompt — fix premature TP1 commit on the OCA path

Paste the block below into Claude Code, running inside this repo. Context: the
v1 migration and the four hardening items are done and tests are green. A review
of `execution/position_manager.py` found that the OCA TP1 path commits the exit
on a bar-price cross even when the resting OCA limit hasn't actually filled,
which double-counts P&L and leaves a zombie position. Fix it.

---

In `execution/position_manager.py`, the `on_bar` TP1 block (~L433-601) has a bug
on the OCA path. Read that block, `_handle_stop_fired`, and `open_position`'s OCA
placement (~L292-340) first.

**The bug.** When an OCA TP1 limit is resting (`pos.tp1_order_id` set) and the
bar's high/low crosses `tp1_price` but the order status is NOT yet
`filled`/`partially_filled`, the code currently sets `tp1_hit = True`, books P&L
at the target price (L484-485), and decrements `remaining` to 0 (L493). With a
resting OCA limit, `on_bar` should only OBSERVE the fill, never synthesize one
from bar price — IB fills the limit when the market reaches it. The wick-to-TP1-
then-reverse-to-stop case then books a TP1 gain AND (when the stop later fires) a
stop loss for the same position, and leaves a `remaining=0, tp1_hit=True,
status="open"` zombie.

**Required behavior.**

1. **Commit TP1 strictly on confirmed fill (OCA path).** When
   `pos.tp1_order_id` is set, only set `tp1_hit = True` / book P&L / decrement
   `remaining` / run the `remaining == 0` close block when
   `tp1_status in ("filled", "partially_filled")`. Read the realized fill from
   the order, not from bar price.
   - For `partially_filled`, commit only the actually-filled quantity (use the
     order's filled qty), not the full `tp1_shares`. Leave the remainder for the
     next poll. (Document this as an edge case if you keep it simple, but do not
     book full-size P&L on a partial.)

2. **Make the bar-price branch diagnostic only (OCA path).** If price has
   crossed `tp1_price` but the order is still working, do NOT treat it as a hit.
   Instead, count consecutive "crossed-but-not-filled" bars on the Position
   (add a field, e.g. `tp1_crossed_unfilled_bars`). After a small threshold
   (e.g. 2 bars), emit a `WARNING`/`CRITICAL` alert to `state_store` (category
   `tp1_limit_not_filling`) so a non-filling limit (thin book / bad price) is
   surfaced — but still do not synthesize a fill. The OCA limit remains the
   source of truth; if the market truly traded through, IB will fill it.

3. **Do not regress the non-OCA path.** When `pos.tp1_order_id` is None (legacy
   multi-leg / no OCA support), keep the existing bar-price trigger that submits
   a marketable-limit exit via `_exit_partial`. That path actively places the
   order, so bar-price triggering is correct there.

4. **Realized-price re-poll (point 2).** Where the OCA fill is confirmed but
   `filled_avg_price` reads falsy (0/None) due to event timing, re-poll
   `get_order(pos.tp1_order_id)` once after a short wait before falling back to
   `pos.tp1_price`. Persist the true `realized_exit_price` when available; only
   fall back to the target if the re-poll is still empty.

5. **No un-reverted mutations.** Ensure that on any early `return` that retains
   the position (e.g. the OCA sibling-not-cancelled CRITICAL branch), the
   position's `remaining` / `tp1_hit` are not left in a committed state unless
   the fill was actually confirmed. The cleanest way is to only mutate them
   inside the confirmed-fill path (item 1), so the retain/alert branches never
   run with synthetic state.

**Tests to add (mock broker):**
- OCA limit resting, bar price crosses TP1, order status still `submitted`:
  assert NO P&L booked, `remaining` unchanged, `tp1_hit` False, position still
  open; after N such bars, assert a `tp1_limit_not_filling` alert fired.
- OCA limit reported `filled` with a real `filled_avg_price`: assert P&L booked
  at the fill price, stop sibling verified cancelled, position closed,
  `realized_exit_price` == fill price.
- Wick-then-reverse: price crosses TP1 (unfilled), then a later bar fires the
  stop: assert exactly ONE realized-P&L booking (the stop), no TP1 gain booked,
  no zombie position.
- `filled` but `filled_avg_price` == 0 on first read, populated on re-poll:
  assert `realized_exit_price` == the re-polled price, not the target.

Run `pytest orb_live/tests/ -q` after each change; keep the §4 parity gate and
all prior hardening tests green. Work in small commits.
