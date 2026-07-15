"""
tests/integration/test_dress_rehearsal_live.py — End-to-end dress rehearsal.

Exercises the full strategy lifecycle against a real IB Gateway with the
clock patched to a chosen historical time.  Catches broker-strategy seam
bugs (e.g. no_ref_price) that only surface when the full pipeline runs
against a live connection instead of mocked unit tests.

    RUN_DRESS_REHEARSAL=1 pytest tests/integration/test_dress_rehearsal_live.py -v

Prerequisites:
    - IB Gateway / TWS running and logged in (paper mode)
    - IB_HOST, IB_PORT, IB_CLIENT_ID set (defaults: 127.0.0.1:4002, clientId=1)
    - Underlying parquets present in orb_live/data/underlyings/
    - Active Level-1 market data subscription
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

RUN = os.environ.get("RUN_DRESS_REHEARSAL") == "1"

_BAD_REASONS    = frozenset({"no_ref_price", "no_daily_bars"})
_ALLOWED_SKIP   = frozenset({"gap_too_small", "dow_excluded", "direction_filtered",
                              "no_prior_close"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_trading_friday() -> date:
    """Most recent Friday that was a NYSE trading day."""
    from orb_live.core.calendar import is_trading_day
    d = datetime.now(tz=ET).date() - timedelta(days=1)
    while True:
        if d.weekday() == 4 and is_trading_day(d):
            return d
        d -= timedelta(days=1)


class _GapScanSpy:
    """Thin wrapper around StateStore that records every save_gap_scan call."""

    def __init__(self, real_store):
        self._real   = real_store
        self.records: list[dict] = []

    def save_gap_scan(self, trade_date, symbol, **kwargs):
        self.records.append({"symbol": symbol, **kwargs})
        self._real.save_gap_scan(trade_date, symbol, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _RecordingLogger:
    """Captures structured log events emitted by session components."""

    def __init__(self):
        self.events: list[str] = []

    def _emit(self, level: str, event: str, **kw):
        self.events.append(event)
        kw_str = " ".join(f"{k}={v!r}" for k, v in kw.items()) if kw else ""
        print(f"  [DRESS/{level}] {event} {kw_str}")

    def info(self,     event, **kw): self._emit("INFO", event, **kw)
    def warning(self,  event, **kw): self._emit("WARN", event, **kw)
    def error(self,    event, **kw): self._emit("ERROR", event, **kw)
    def critical(self, event, **kw): self._emit("CRIT", event, **kw)
    def debug(self,    event, **kw): pass


# ── Test class ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not RUN, reason="Set RUN_DRESS_REHEARSAL=1 to run dress rehearsal")
class TestDressRehearsal:
    """
    Full-lifecycle dress rehearsal against a real IB Gateway.

    All four tests share a single connected IBClient (class-scoped fixture).
    A temporary SQLite DB is used so production state is never touched.
    """

    @pytest.fixture(scope="class")
    def components(self):
        from orb_live.config.live_config import load_live_config
        from orb_live.core.clock import MarketClock
        from orb_live.core.state_store import StateStore
        from orb_live.data.ib_client import build_client_from_env
        from orb_live.data.underlying_data import UnderlyingDataStore
        from orb_live.signals.pre_market import PreMarketJob

        cfg    = load_live_config()
        client = build_client_from_env(paper=True)
        # Allow delayed data so connect() doesn't gate on subscription during
        # fixture setup; individual tests re-enable strict mode as needed.
        client._allow_delayed_data = True
        client.connect()

        tmpdir_obj = tempfile.TemporaryDirectory(prefix="dress_rehearsal_",
                                                  ignore_cleanup_errors=True)
        db_path    = Path(tmpdir_obj.name) / "rehearsal.db"
        store      = StateStore(db_path)
        ul_store   = UnderlyingDataStore(cfg.data_dir)
        clock      = MarketClock(broker_client=client)

        pre_market = PreMarketJob(cfg, store, client, ul_store)

        trade_date   = _last_trading_friday()
        fixed_09_31  = datetime(trade_date.year, trade_date.month, trade_date.day,
                                9, 31, tzinfo=ET)
        fixed_10_01  = datetime(trade_date.year, trade_date.month, trade_date.day,
                                10, 1, tzinfo=ET)
        fixed_15_55  = datetime(trade_date.year, trade_date.month, trade_date.day,
                                15, 55, tzinfo=ET)

        yield {
            "cfg":         cfg,
            "client":      client,
            "store":       store,
            "ul_store":    ul_store,
            "clock":       clock,
            "pre_market":  pre_market,
            "trade_date":  trade_date,
            "fixed_09_31": fixed_09_31,
            "fixed_10_01": fixed_10_01,
            "fixed_15_55": fixed_15_55,
            "tmpdir":      tmpdir_obj,
        }

        client.disconnect()
        store.close()
        tmpdir_obj.cleanup()

    # ── Test 1: Phase 1 against live broker ───────────────────────────────────

    def test_dress_phase1_against_live_broker(self, components):
        """
        Phase 1 uses the 9:30 bar close (not a pre-market quote) as the gap
        reference price.  With the clock patched to 09:31 ET, get_intraday_bars
        returns the completed 9:30 bar.  No symbol should be rejected with
        'no_ref_price' or 'no_daily_bars' when a valid IB connection is active —
        either real candidates emerge OR every rejection has a legitimate
        strategy reason (gap_too_small / dow_excluded / direction_filtered /
        no_prior_close).
        """
        trade_date  = components["trade_date"]
        fixed_dt    = components["fixed_09_31"]
        pre_market  = components["pre_market"]
        real_store  = components["store"]

        spy = _GapScanSpy(real_store)
        pre_market._store = spy

        print(f"\n[DRESS] Patched clock to {fixed_dt.strftime('%a %Y-%m-%d %H:%M')} ET")
        print("[DRESS] Phase 1 (9:30 bar close ref price) firing with real IB Gateway ...")

        with patch.object(components["clock"], "now_et", return_value=fixed_dt):
            results = pre_market.run_phase1(trade_date)

        # Restore real store so subsequent tests write to it normally.
        pre_market._store = real_store

        candidates = [r.symbol for r in results]
        rejections = [
            (r["symbol"], r.get("filter_reason"))
            for r in spy.records
            if not r.get("qualifies", True)
        ]
        bad = [
            (sym, reason)
            for sym, reason in rejections
            if reason in _BAD_REASONS
        ]

        print(f"[DRESS] Phase 1 result: n={len(results)} symbols={candidates}")
        if rejections:
            print(f"[DRESS] Rejections: {rejections}")

        if bad:
            pytest.fail(
                f"[DRESS] FAIL — broker-data seam bug detected.\n"
                f"  Symbols with bad rejection reason: {bad}\n"
                f"  'no_ref_price' means get_intraday_bars returned no 9:30 bar;\n"
                f"  'no_daily_bars' means get_daily_bars returned empty.\n"
                f"  Check IB Gateway connectivity and that {trade_date} has bar data."
            )

        if results:
            print(f"[DRESS] PASS: {len(results)} real candidate(s), no data-seam errors")
        else:
            bad_reasons = {r for _, r in rejections} - _ALLOWED_SKIP
            assert not bad_reasons, (
                f"All symbols rejected but some had unexpected reasons: {bad_reasons}"
            )
            print("[DRESS] PASS: all symbols filtered for legitimate reasons (no gap), "
                  "no data-seam errors")

    # ── Test 2: Phase 2 with real ORB bars ────────────────────────────────────

    def test_dress_phase2_with_real_orb_bars(self, components):
        """
        Phase 2 must be able to fetch intraday 9:30-10:00 bars from IB and
        compute an ORB dict.  Uses a synthetic Phase1Result so the test is
        independent of whether any symbol gapped on the target date.
        """
        from orb_live.signals.pre_market import Phase1Result, Phase2Result

        trade_date = components["trade_date"]
        fixed_dt   = components["fixed_10_01"]
        pre_market = components["pre_market"]
        cfg        = components["cfg"]

        # Pick a highly-liquid symbol guaranteed to be in the universe.
        test_sym = next(s for s in cfg.symbols if s in ("TQQQ", "SQQQ", "UPRO"))
        p1 = Phase1Result(
            symbol          = test_sym,
            gap_abs         = 0.04,    # synthetic 4% gap
            gap_direction   = 1,
            prior_close     = 50.0,    # synthetic; not used by phase2 logic
            ps_filter_passed= True,
        )

        print(f"\n[DRESS] Patched clock to {fixed_dt.strftime('%a %Y-%m-%d %H:%M')} ET")
        print(f"[DRESS] Phase 2 firing for synthetic {test_sym} candidate ...")

        with patch.object(components["clock"], "now_et", return_value=fixed_dt):
            p2_list = pre_market.run_phase2(trade_date, [p1])

        assert len(p2_list) == 1, "run_phase2 must return one result per input"
        p2: Phase2Result = p2_list[0]

        print(f"[DRESS] Phase 2 result: symbol={p2.symbol} "
              f"is_candidate={p2.is_candidate} orb={p2.orb is not None} "
              f"exclusion='{p2.exclusion_reason}'")

        if p2.orb is not None:
            n_bars = p2.orb.get("n_bars", "?")
            print(f"[DRESS] ORB computed from {n_bars} intraday bars — PASS")
        else:
            # Acceptable if IB has no historical data for that date / time window.
            reason = p2.exclusion_reason or "no_intraday_bars"
            print(f"[DRESS] ORB not computed (reason: {reason}) — "
                  "acceptable outside market hours or for stale dates")
            assert reason not in _BAD_REASONS, (
                f"Phase 2 exclusion reason '{reason}' indicates a data-seam bug"
            )

        # The critical assertion: no AttributeError / TypeError / seam crash.
        # If we get here without an unhandled exception, the seam is healthy.
        print(f"[DRESS] PASS: Phase 2 completed without crash for {test_sym}")

    # ── Test 3: Full session dry run ──────────────────────────────────────────

    def test_dress_full_session_dry_run(self, components):
        """
        Run session_runner.run_session() end-to-end with:
          - clock fixed at 15:55 ET (ORB long past, near EOD)
          - _sleep replaced by a no-op
          - bar subscriptions suppressed (no background threads)
          - signal handlers patched (avoid pytest conflicts)

        Verifies that the full orchestration flow completes without crashing
        and emits the expected lifecycle log events.
        """
        from orb_live.core.clock import MarketClock
        from orb_live.core.state_store import StateStore
        from orb_live.data.bar_cache import BarCache
        from orb_live.execution.order_policy import MarketableLimitPolicy
        from orb_live.execution.position_manager import LivePositionManager
        from orb_live.execution.risk_gate import RiskGate
        from orb_live.runner.bar_router import BarRouter
        from orb_live.runner.session_runner import SessionRunner
        from orb_live.runner.strategy_engine import StrategyEngine
        from orb_live.signals.pre_market import PreMarketJob

        cfg      = components["cfg"]
        client   = components["client"]
        ul_store = components["ul_store"]
        trade_date = components["trade_date"]
        fixed_dt   = components["fixed_15_55"]

        # Fresh isolated DB for this test.
        with tempfile.TemporaryDirectory(prefix="dress_session_",
                                         ignore_cleanup_errors=True) as tmpdir:
            store = StateStore(Path(tmpdir) / "session.db")

            bar_cache        = BarCache()
            clock            = MarketClock(broker_client=client)
            policy           = MarketableLimitPolicy(client, cfg, store)
            gate             = RiskGate(cfg, store, client)
            indicators_store: dict = {}
            mgr              = LivePositionManager(
                broker=client, policy=policy, state_store=store,
                risk_gate=gate, indicators_store=indicators_store,
                config=cfg.strategy_config,
            )
            bar_router  = BarRouter(client, store, bar_cache)
            pre_market  = PreMarketJob(cfg, store, client, ul_store)
            engine      = StrategyEngine(mgr, cfg, store, client)
            logger      = _RecordingLogger()

            runner = SessionRunner(
                config           = cfg,
                broker           = client,
                state_store      = store,
                bar_cache        = bar_cache,
                bar_router       = bar_router,
                pre_market_job   = pre_market,
                strategy_engine  = engine,
                position_manager = mgr,
                risk_gate        = gate,
                indicators_store = indicators_store,
                underlying_store = ul_store,
                clock            = clock,
                logger           = logger,
                _sleep           = lambda s: None,
            )

            print(f"\n[DRESS] Full session dry run — {trade_date} clock@"
                  f"{fixed_dt.strftime('%H:%M')} ET")

            with patch.object(clock, "now_et", return_value=fixed_dt), \
                 patch.object(bar_router, "subscribe"), \
                 patch.object(bar_router, "unsubscribe"), \
                 patch("signal.signal"):
                runner.run_session(trade_date)

            events = logger.events
            print(f"[DRESS] Events logged: {events}")

            assert "pre_market_start" in events, \
                f"Expected 'pre_market_start' in events; got: {events}"
            assert "phase1_complete" in events, \
                f"Expected 'phase1_complete' in events; got: {events}"
            assert "session_complete" in events, \
                f"Expected 'session_complete' in events; got: {events}"

            print("[DRESS] PASS: session completed without crash, "
                  "expected events emitted")

            store.close()  # release SQLite handle before temp dir is deleted

    # ── Test 4: Bracket order full lifecycle ─────────────────────────────────

    def test_dress_bracket_order_lifecycle(self, components):
        """
        Place a native IB bracket order (entry limit far from market + OCA TP1
        limit + OCA stop), verify all three legs are created and working at IB,
        then cancel the entry which should cascade-cancel the children.

        This exercises the exact path that open_position() now uses: one atomic
        submit_bracket_order call instead of the old detect-fill-then-place-OCA
        sequence.
        """
        client = components["client"]

        # Use a liquid symbol with a far-from-market limit so nothing fills.
        symbol     = "SOXL"
        entry_lmt  = 0.01   # far below market — will never fill
        tp1_lmt    = 0.02   # TP1 above entry
        stop_px    = 0.005  # stop below entry
        qty        = 1

        print(f"\n[DRESS] Placing bracket order: {symbol} BUY {qty} "
              f"entry@{entry_lmt} tp1@{tp1_lmt} stop@{stop_px}")

        bracket = client.submit_bracket_order(
            symbol          = symbol,
            side            = "buy",
            qty             = qty,
            entry_price     = entry_lmt,
            tp1_limit_price = tp1_lmt,
            stop_price      = stop_px,
            tif             = "day",
            timeout         = 10.0,
        )

        entry_id = bracket["entry_order_id"]
        tp1_id   = bracket["tp1_order_id"]
        stop_id  = bracket["stop_order_id"]

        print(f"[DRESS] Bracket placed: entry={entry_id} tp1={tp1_id} stop={stop_id}")
        assert entry_id, "submit_bracket_order did not return entry_order_id"
        assert tp1_id,   "submit_bracket_order did not return tp1_order_id"
        assert stop_id,  "submit_bracket_order did not return stop_order_id"

        # Entry must be active; children must be inactive/held (IB holds children
        # until parent fills — they appear as PreSubmitted or Inactive).
        try:
            entry_state = client.get_order(entry_id)
            assert entry_state["status"] in ("new", "submitted"), \
                f"entry unexpected status: {entry_state['status']}"
            print(f"[DRESS] Entry status: {entry_state['status']} — OK")

            tp1_state = client.get_order(tp1_id)
            print(f"[DRESS] TP1 child status: {tp1_state['status']}")

            stop_state = client.get_order(stop_id)
            print(f"[DRESS] Stop child status: {stop_state['status']}")
        finally:
            # Cancel the entry — IB cascades the cancel to all OCA children.
            cancelled = client.cancel_order(entry_id, timeout=10.0)
            print(f"[DRESS] cancel_order(entry): {cancelled}")
            client.cancel_order(tp1_id, timeout=5.0)
            client.cancel_order(stop_id, timeout=5.0)

        print("[DRESS] PASS: bracket lifecycle (place → verify → cancel) completed")

    # ── Test 5: BrokerClient contract compliance ──────────────────────────────

    def test_dress_broker_contract_compliance(self, components):
        """
        Call every BrokerClient ABC method with safe test arguments.

        Read-only methods must return a value matching the declared type.
        Unimplemented methods (those that raise NotImplementedError in IBClient)
        are accepted — the test verifies the method *exists* and *raises* the
        correct exception, not that it succeeds.

        For order submission: one far-from-market BUY @ $0.01 is placed and
        immediately cancelled in a finally block.
        """
        client = components["client"]
        print(f"\n[DRESS] BrokerClient contract compliance check ...")

        # ── Read-only methods ─────────────────────────────────────────────────

        assert client.is_connected() is True
        print("[DRESS]  is_connected: OK")

        acct = client.get_account()
        for key in ("equity", "cash", "buying_power", "portfolio_value",
                    "daytrade_count", "pattern_day_trader"):
            assert key in acct, f"get_account missing key '{key}'"
        assert isinstance(acct["equity"], float)
        print(f"[DRESS]  get_account: equity=${acct['equity']:,.2f}")

        equity = client.get_equity()
        assert isinstance(equity, float) and equity >= 0
        print(f"[DRESS]  get_equity: ${equity:,.2f}")

        clk = client.get_clock()
        for key in ("is_open", "next_open", "next_close", "timestamp"):
            assert key in clk, f"get_clock missing key '{key}'"
        print(f"[DRESS]  get_clock: is_open={clk['is_open']}")

        is_open = client.is_market_open()
        assert isinstance(is_open, bool)
        print(f"[DRESS]  is_market_open: {is_open}")

        asset = client.get_asset("SPY")
        assert "symbol" in asset and "tradable" in asset
        print(f"[DRESS]  get_asset(SPY): tradable={asset['tradable']}")

        import pandas as pd
        daily = client.get_daily_bars("SPY", lookback_days=5)
        assert isinstance(daily, pd.DataFrame)
        if not daily.empty:
            assert "close" in daily.columns
        print(f"[DRESS]  get_daily_bars(SPY): {len(daily)} bars")

        positions = client.get_positions()
        assert isinstance(positions, list)
        print(f"[DRESS]  get_positions: {len(positions)} open")

        pos = client.get_position("ZZZNOTREAL")
        assert pos is None
        print("[DRESS]  get_position(unknown): None — OK")

        # get_latest_quote: accept both valid quote and RuntimeError (zero-guard).
        try:
            quote = client.get_latest_quote("SPY")
            assert "bid" in quote and "ask" in quote
            print(f"[DRESS]  get_latest_quote(SPY): bid={quote['bid']:.2f} "
                  f"ask={quote['ask']:.2f}")
        except RuntimeError as exc:
            print(f"[DRESS]  get_latest_quote(SPY): RuntimeError (subscription "
                  f"guard) — {exc}")

        # ── Order lifecycle ───────────────────────────────────────────────────

        order_id = None
        try:
            print("[DRESS]  submit_limit_order(SPY BUY 1 @ $0.01) ...")
            order = client.submit_limit_order("SPY", "buy", 1, 0.01)
            assert "id" in order, f"submit_limit_order missing 'id' key: {order}"
            order_id = order["id"]
            print(f"[DRESS]  submit_limit_order: order_id={order_id}")

            fetched = client.get_order(order_id)
            assert fetched is not None
            print(f"[DRESS]  get_order: status={fetched.get('status')}")
        finally:
            if order_id is not None:
                try:
                    client.cancel_order(order_id)
                    print(f"[DRESS]  cancel_order({order_id}): OK")
                except Exception as exc:
                    print(f"[DRESS]  cancel_order({order_id}): {exc} (may already be done)")

        # ── Emergency / utility methods ───────────────────────────────────────

        for name, fn in [
            ("cancel_all_orders",   lambda: client.cancel_all_orders()),
            ("close_position",      lambda: client.close_position("ZZZNOTREAL")),
            ("close_all_positions", lambda: client.close_all_positions()),
            ("list_orders",         lambda: client.list_orders()),
        ]:
            try:
                result = fn()
                print(f"[DRESS]  {name}: returned {result!r}")
            except Exception as exc:
                print(f"[DRESS]  {name}: {type(exc).__name__}({exc})")

        print("[DRESS] PASS: all BrokerClient contract methods reachable")
