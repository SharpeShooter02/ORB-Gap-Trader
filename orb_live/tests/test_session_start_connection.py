"""IB Gateway restarts daily. The session must notice before it trades.

The daemon sleeps between sessions with a plain time.sleep (main.py
_run_daemon), which neither pumps the asyncio loop nor checks the socket. So a
Gateway restart at 23:45 leaves a dead connection that nothing observes until
08:30 -- and a dead IB socket does NOT raise. It degrades quietly:

    get_account()   accountValues() returns []  -> equity 0.0 (logged, not raised)
    check_margin()  is_connected() False        -> full-notional fallback, 100%
    get_positions() portfolio() returns []      -> looks flat

A session started that way runs to completion, has every entry rejected by the
risk gate on zero equity, and books nothing. No crash, no ConnectionError, and
the daemon's `except ConnectionError` recovery -- which exists and works -- is
never reached. A silent no-trade day is the worst failure mode available here,
because it looks exactly like a day with no qualifying gaps.

run_session therefore reconnects first, and raises ConnectionError if it
cannot, so the daemon's existing handler retries the same day rather than
proceeding on a dead socket.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _runner(broker):
    """SessionRunner with every collaborator stubbed except the broker."""
    from orb_live.runner.session_runner import SessionRunner

    mgr = MagicMock()
    return SessionRunner(
        config=SimpleNamespace(eod_flatten_lead_secs=30, symbols=[]),
        broker=broker,
        state_store=MagicMock(),
        bar_cache=MagicMock(),
        bar_router=MagicMock(),
        pre_market_job=MagicMock(),
        strategy_engine=MagicMock(),
        position_manager=mgr,
        risk_gate=MagicMock(),
        indicators_store={},
        underlying_store=SimpleNamespace(),
        clock=MagicMock(),
        _sleep=lambda _: None,
    ), mgr


class TestSessionRefusesToStartDisconnected:
    def test_reconnects_when_the_socket_is_down(self):
        broker = MagicMock()
        broker.is_connected.side_effect = [False, True]
        broker.reconnect.return_value = True

        runner, mgr = _runner(broker)
        runner._ensure_connected()

        broker.reconnect.assert_called_once()

    def test_raises_connection_error_when_reconnect_fails(self):
        """ConnectionError specifically -- that is what the daemon catches and
        retries the same day. Any other exception kills the daemon loop."""
        broker = MagicMock()
        broker.is_connected.return_value = False
        broker.reconnect.return_value = False

        runner, _ = _runner(broker)
        with pytest.raises(ConnectionError):
            runner._ensure_connected()

    def test_no_reconnect_when_already_connected(self):
        broker = MagicMock()
        broker.is_connected.return_value = True
        runner, _ = _runner(broker)
        runner._ensure_connected()
        broker.reconnect.assert_not_called()

    def test_run_session_checks_before_touching_the_broker(self):
        """The guard must run BEFORE startup_reconcile, which asks the broker
        for positions and would read [] off a dead socket as 'flat'."""
        broker = MagicMock()
        broker.is_connected.return_value = False
        broker.reconnect.return_value = False

        runner, mgr = _runner(broker)
        with pytest.raises(ConnectionError):
            runner.run_session(date(2026, 8, 26))
        mgr.startup_reconcile.assert_not_called()

    def test_broker_without_is_connected_is_tolerated(self):
        """dry_run and test brokers do not implement it; absence must not
        become a hard failure."""
        broker = SimpleNamespace()
        runner, _ = _runner(broker)
        runner._ensure_connected()      # must not raise
