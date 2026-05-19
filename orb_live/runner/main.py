"""
runner/main.py — CLI entry point for the live ORB session runner.

Usage:
    python -m orb_live.runner.main --paper
    python -m orb_live.runner.main --live
    python -m orb_live.runner.main --dry-run
    python -m orb_live.runner.main --paper --session-date 2026-01-07
    python -m orb_live.runner.main --paper --recover

Flags:
    --paper          Use Alpaca paper-trading endpoint (default)
    --live           Use Alpaca live-trading endpoint (requires confirmation)
    --dry-run        Simulate fills locally; use real market data
    --session-date   Override today's date (YYYY-MM-DD; for replay / testing)
    --recover        Reconcile positions from broker before running
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live ORB session runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper",   action="store_true", default=True,
                      help="Paper trading (default)")
    mode.add_argument("--live",    action="store_true",
                      help="Live trading (requires ALPACA_LIVE=1 env var)")
    mode.add_argument("--dry-run", dest="dry_run", action="store_true",
                      help="Simulate fills locally; use real market data")

    p.add_argument("--session-date", metavar="YYYY-MM-DD",
                   help="Override trading date (default: today ET)")
    p.add_argument("--recover", action="store_true",
                   help="Reconcile broker positions before running")
    return p.parse_args()


def _confirm_live() -> None:
    import os
    if os.environ.get("ALPACA_LIVE") != "1":
        print(
            "ERROR: --live requires ALPACA_LIVE=1 in environment.\n"
            "Set this only after completing all paper-trading validation."
        )
        sys.exit(1)
    answer = input("You are about to trade with REAL MONEY. Type 'yes' to confirm: ")
    if answer.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)


def _build_components(args: argparse.Namespace):
    """
    Construct all session components from config + env.

    Returns (runner, session_date).
    """
    import orb_live  # noqa: F401 — path setup

    from orb_live.config.live_config import load_live_config
    from orb_live.core.state_store import StateStore
    from orb_live.core.logger import get_logger
    from orb_live.core.clock import MarketClock
    from orb_live.data.alpaca_client import build_client_from_env
    from orb_live.data.bar_cache import BarCache
    from orb_live.data.underlying_data import UnderlyingDataStore
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.signals.pre_market import PreMarketJob
    from orb_live.runner.bar_router import BarRouter
    from orb_live.runner.strategy_engine import StrategyEngine
    from orb_live.runner.session_runner import SessionRunner

    cfg    = load_live_config()
    logger = get_logger(__name__)

    # Session date
    if args.session_date:
        session_date = datetime.strptime(args.session_date, "%Y-%m-%d").date()
    else:
        session_date = datetime.now(tz=ET).date()

    # Alpaca client
    paper = not args.live
    real_client = build_client_from_env(paper=paper)

    if args.dry_run:
        from orb_live.runner.dry_run import DryRunAlpaca
        starting_equity = float(real_client.get_account().get("equity", 100_000.0))
        alpaca = DryRunAlpaca(real_client, starting_equity=starting_equity)
    else:
        alpaca = real_client

    # State store
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    store = StateStore(cfg.db_path)

    # Supporting components
    bar_cache    = BarCache()
    clock        = MarketClock(alpaca_client=real_client)
    ul_store     = UnderlyingDataStore(cfg.data_dir)

    # Execution layer
    policy = MarketableLimitPolicy(alpaca, cfg, store)
    gate   = RiskGate(cfg, store, alpaca, logger=logger)

    indicators_store: dict = {}

    mgr = LivePositionManager(
        alpaca=alpaca,
        policy=policy,
        state_store=store,
        risk_gate=gate,
        indicators_store=indicators_store,
        config=cfg.strategy_config,
        logger=logger,
    )

    # Runner components
    bar_router   = BarRouter(alpaca, store, bar_cache, logger=logger)
    pre_market   = PreMarketJob(cfg, store, alpaca, ul_store, logger=logger)
    engine       = StrategyEngine(mgr, cfg, store, alpaca, logger=logger)

    runner = SessionRunner(
        config=cfg,
        alpaca=alpaca,
        state_store=store,
        bar_cache=bar_cache,
        bar_router=bar_router,
        pre_market_job=pre_market,
        strategy_engine=engine,
        position_manager=mgr,
        risk_gate=gate,
        indicators_store=indicators_store,
        underlying_store=ul_store,
        clock=clock,
        logger=logger,
    )

    return runner, clock, session_date


def _run_daemon(
    runner,
    clock,
    recover: bool = False,
    _sleep=None,
    _shutdown=None,
    _sleep_interval: float = 60.0,
) -> None:
    """
    Daemon loop: sleep until 08:30 ET on the next market day, run a session,
    then repeat until SIGTERM/SIGINT.

    Parameters prefixed with _ are injection points for testing only.
    """
    import time as _time
    import signal as _signal

    _sleep_fn = _sleep or _time.sleep
    shutdown   = _shutdown if _shutdown is not None else [False]

    if _shutdown is None:
        def _on_signal(signum, frame):
            shutdown[0] = True
        _signal.signal(_signal.SIGTERM, _on_signal)
        _signal.signal(_signal.SIGINT,  _on_signal)

    while not shutdown[0]:
        next_pm = clock.next_premarket_start()
        now     = clock.now_et()
        secs    = (next_pm - now).total_seconds()

        if secs > 1:
            remaining = secs
            while not shutdown[0] and remaining > 0:
                chunk = min(_sleep_interval, remaining)
                _sleep_fn(chunk)
                remaining -= chunk
            if shutdown[0]:
                break

        if shutdown[0]:
            break

        session_date = next_pm.date()
        try:
            if recover:
                runner.recover(session_date)
                recover = False
            else:
                runner.run_session(session_date)
        except SystemExit:
            break


def main() -> None:
    args = _parse_args()

    if args.live:
        _confirm_live()

    runner, clock, session_date = _build_components(args)

    if args.session_date:
        # One-shot: run the specified date immediately
        if args.recover:
            runner.recover(session_date)
        else:
            runner.run_session(session_date)
    else:
        # Daemon mode: sleep until next 08:30 ET pre-market, then loop
        _run_daemon(runner, clock, recover=args.recover)


if __name__ == "__main__":
    main()
