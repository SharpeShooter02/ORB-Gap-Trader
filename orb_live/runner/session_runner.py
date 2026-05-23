"""
runner/session_runner.py — Top-level session orchestrator.

Production session timeline (all times ET):
  ~08:30  pre-market:   Phase 1 gap scan + PS filter
   09:30  ORB window:   subscribe_bars begins; bars accumulate in bar_cache
   10:00  ORB closes:   Phase 2 (RTG + routing + preflight); seed indicators
  10:00+  trade_active: per-bar dispatch loop (LOAD-BEARING ORDER below)
  16:00   eod_exit:     wait for eod_exit_hour, then flatten_all("eod_sweep")
  16:30   report:       generate_daily_report

LOAD-BEARING BAR DISPATCH ORDER (do not reorder):
  1. bar_cache.add_bar(symbol, bar)
  2. indicators[symbol].on_bar(bar)           ← EMA update; MUST precede step 3
  3. position_manager.on_bar(symbol, bar, ts) ← exit management
  4. strategy_engine.on_bar(symbol, bar, ts)  ← entry detection (ORB_COMPLETE only)
"""

from __future__ import annotations

import signal
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class SessionRunner:
    """
    Wires all sub-components together for one trading session.

    All components are injected at construction so the class is fully
    testable without Alpaca credentials.  In production, build the components
    via `runner.main` and pass them in.
    """

    def __init__(
        self,
        config,                # LiveConfig
        broker,                # BrokerClient | DryRunAlpaca | MockAlpaca
        state_store,           # StateStore
        bar_cache,             # BarCache
        bar_router,            # BarRouter
        pre_market_job,        # PreMarketJob
        strategy_engine,       # StrategyEngine
        position_manager,      # LivePositionManager
        risk_gate,             # RiskGate
        indicators_store,      # dict[str, RollingIndicators] (mutable, session-scoped)
        underlying_store,      # UnderlyingDataStore
        clock,                 # MarketClock
        logger=None,
        _sleep=None,           # injectable for tests; defaults to time.sleep
    ):
        self._cfg         = config
        self._broker      = broker
        self._store       = state_store
        self._cache       = bar_cache
        self._router      = bar_router
        self._pre_market  = pre_market_job
        self._engine      = strategy_engine
        self._mgr         = position_manager
        self._gate        = risk_gate
        self._indicators  = indicators_store
        self._ul          = underlying_store
        self._clock       = clock
        self._log         = logger
        self._sleep       = _sleep or time.sleep

        self._session_date: Optional[date] = None
        self._phase1_results = []

        # Register bar dispatch with router
        self._router.register_listener(self._on_bar_dispatch)

    # ── Main entry point ───────────────────────────────────────────────────────

    def run_session(self, session_date: date) -> None:
        """Run a complete session.  Installs SIGTERM/SIGINT handlers."""
        self._session_date = session_date
        self._engine.new_session(session_date)

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

        try:
            self._run_pre_market(session_date)
            self._run_orb_window(session_date)
            self._run_post_orb(session_date)
            self._run_eod(session_date)
        except SystemExit:
            raise
        except Exception as exc:
            if self._log:
                self._log.critical("session_crash", exc=str(exc), date=str(session_date))
            self._shutdown("crash")
            raise
        finally:
            self._router.unsubscribe()

    def recover(self, session_date: date) -> None:
        """
        Resume a crashed session.  Reconciles open positions from Alpaca before
        re-entering the session loop at the current phase.
        """
        if self._log:
            self._log.info("session_recovery_start", date=str(session_date))
        self._mgr.reconcile_from_broker()
        self.run_session(session_date)

    # ── Phase runners ──────────────────────────────────────────────────────────

    def _run_pre_market(self, session_date: date) -> None:
        equity = float(self._broker.get_account().get("equity", 0.0))
        self._gate.session_start(equity, session_date)
        self._store.upsert_day_state(session_date, phase="pre_market")

        is_half  = self._clock.is_half_day()
        close_t  = self._clock.effective_close()
        if self._log:
            self._log.info(
                "pre_market_start", equity=equity, date=str(session_date),
                is_half_day=is_half,
                todays_close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
            )
        if is_half and self._log:
            self._log.warning(
                "half_day_session",
                close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
                date=str(session_date),
            )

        self._phase1_results = self._pre_market.run_phase1(session_date)
        p1_symbols = [r.symbol for r in self._phase1_results]

        self._store.upsert_day_state(
            session_date, phase="pre_market", n_prequalified=len(p1_symbols)
        )
        if self._log:
            self._log.info("phase1_complete", n=len(p1_symbols), symbols=p1_symbols)

        if p1_symbols:
            self._router.subscribe(p1_symbols)

    def _run_orb_window(self, session_date: date) -> None:
        """Sleep until the ORB window closes at 10:00 ET."""
        self._store.upsert_day_state(session_date, phase="orb_window")
        orb_end = self._clock.orb_end_et(self._cfg.orb_minutes)
        secs    = (orb_end - self._clock.now_et()).total_seconds()
        if secs > 0:
            if self._log:
                self._log.info("orb_window_wait", seconds=round(secs, 1))
            self._sleep(secs)

    def _run_post_orb(self, session_date: date) -> None:
        """Run Phase 2, seed indicators, then wait until eod_exit_hour."""
        self._store.upsert_day_state(session_date, phase="trade_active")

        if not self._phase1_results:
            if self._log:
                self._log.info("no_phase1_candidates", date=str(session_date))
            self._wait_until_eod(session_date)
            return

        p2_results = self._pre_market.run_phase2(session_date, self._phase1_results)

        from orb_live.execution.indicators import RollingIndicators

        for p2 in p2_results:
            symbol = p2.symbol

            if p2.orb is None:
                self._engine.on_orb_complete(symbol, p2, None)
                continue

            # Get ORB bars from cache and seed the indicator
            raw = self._cache.get_bars(symbol)
            if raw.empty:
                self._engine.on_orb_complete(symbol, p2, None)
                continue

            raw = raw.copy()
            raw["timestamp"] = _to_et(raw["timestamp"])
            raw = raw.set_index("timestamp")

            # Filter to ORB window only for seeding
            orb_end_dt   = self._clock.orb_end_et(self._cfg.orb_minutes)
            orb_start_dt = orb_end_dt - timedelta(minutes=self._cfg.orb_minutes)
            mask = (raw.index >= _aware(orb_start_dt)) & (raw.index < _aware(orb_end_dt))
            orb_df = raw.loc[mask]

            scfg      = self._cfg.strategy_config
            indicator = RollingIndicators(symbol, scfg)

            try:
                if not orb_df.empty:
                    indicator.seed_from_orb_bars(orb_df)
                    self._indicators[symbol] = indicator
            except Exception as exc:
                if self._log:
                    self._log.error("indicator_seed_failed", symbol=symbol, exc=str(exc))
                self._engine.on_orb_complete(symbol, p2, None)
                continue

            self._engine.on_orb_complete(symbol, p2, orb_df)

            if self._log:
                self._log.info(
                    "phase2_ready", symbol=symbol,
                    is_candidate=p2.is_candidate,
                    rtg_excluded=p2.rtg_excluded,
                )

        self._wait_until_eod(session_date)

    def _wait_until_eod(self, session_date: date) -> None:
        # Respect half-day schedules (e.g. July 3 closes at 13:00 ET not 16:00).
        close_t = self._clock.effective_close()
        now     = self._clock.now_et()
        eod     = now.replace(hour=close_t.hour, minute=close_t.minute,
                               second=0, microsecond=0)
        secs    = (eod - now).total_seconds()

        if self._clock.is_half_day():
            if self._log:
                self._log.warning(
                    "half_day_detected",
                    close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
                    date=str(session_date),
                )

        if secs > 0:
            if self._log:
                self._log.info("waiting_for_eod", seconds=round(secs, 1),
                               close_et=f"{close_t.hour:02d}:{close_t.minute:02d}")
            self._sleep(secs)

    def _run_eod(self, session_date: date) -> None:
        """16:00:30 ET: flatten all positions, record equity, generate report."""
        self._sleep(30)  # 30 seconds past eod boundary
        self._mgr.flatten_all("eod_sweep")
        self._store.upsert_day_state(session_date, phase="closed")

        equity = float(self._broker.get_account().get("equity", 0.0))
        self._store.record_equity(session_date, equity, equity, 0.0)

        from orb_live.ops.reports import generate_daily_report
        generate_daily_report(self._store, session_date, logger=self._log)

        if self._log:
            self._log.info("session_complete", date=str(session_date))

    # ── Bar dispatch (LOAD-BEARING ORDER) ─────────────────────────────────────

    def _on_bar_dispatch(self, symbol: str, bar: dict) -> None:
        """
        Called by BarRouter for every incoming bar (live or replayed).

        ORDER IS LOAD-BEARING — do not reorder steps 2/3/4:
          1. bar_cache.add_bar       — persist to cache
          2. indicators.on_bar       — update EMA  ← MUST precede position_manager
          3. position_manager.on_bar — exit checks for open positions
          4. strategy_engine.on_bar  — entry detection (ORB_COMPLETE symbols only)
        """
        # 1. Cache
        self._cache.add_bar(symbol, bar)

        # Only process post-ORB bars through the engine
        ts = bar.get("timestamp")
        if ts is None:
            return

        bar_time = ts.time() if hasattr(ts, "time") else dtime(0, 0)
        orb_end  = dtime(
            self._cfg.strategy_config.market_open_hour,
            self._cfg.strategy_config.market_open_minute,
        )
        # Advance orb_end by orb_minutes to get actual ORB close time
        orb_end_minutes = (
            self._cfg.strategy_config.market_open_hour * 60
            + self._cfg.strategy_config.market_open_minute
            + self._cfg.orb_minutes
        )
        orb_end = dtime(orb_end_minutes // 60, orb_end_minutes % 60)

        if bar_time < orb_end:
            return  # ORB window bar — cached but not dispatched to engine

        # 2. Indicators
        indicator = self._indicators.get(symbol)
        if indicator is not None and indicator.is_seeded:
            indicator.on_bar(bar)

        # 3. Position management (exits)
        self._mgr.on_bar(symbol, bar, ts)

        # 4. Entry detection
        self._engine.on_bar(symbol, bar, ts)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _handle_signal(self, signum, frame) -> None:
        reason = "sigterm" if signum == signal.SIGTERM else "sigint"
        self._shutdown(reason)
        sys.exit(0)

    def _shutdown(self, reason: str) -> None:
        if self._log:
            self._log.warning("shutdown", reason=reason)
        try:
            self._mgr.flatten_all(reason)
        except Exception as exc:
            if self._log:
                self._log.error("shutdown_flatten_failed", exc=str(exc))
        self._router.unsubscribe()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_et(series):
    """Coerce a timestamp Series to ET-aware datetimes."""
    s = series.copy()
    if hasattr(s, "dt"):
        if s.dt.tz is None:
            s = s.dt.tz_localize(ET)
        else:
            s = s.dt.tz_convert(ET)
    return s


def _aware(dt: datetime) -> datetime:
    """Return dt as an ET-aware datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)
