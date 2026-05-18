"""
tests/test_order_policy.py — 8 cases for MarketableLimitPolicy.

Each test uses MockAlpaca from conftest.py (injected via mock_alpaca fixture)
and a minimal policy_cfg SimpleNamespace so tests stay fast.  _sleep is
always injected as a no-op; MockAlpaca returns the final order status on the
first get_order call, so poll loops exit immediately.
"""

from datetime import date
from types import SimpleNamespace

import pytest

TRADE_DATE = date(2026, 1, 5)
SYMBOL = "TQQQ"

_NO_SLEEP = lambda _: None  # noqa: E731


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def policy_cfg():
    return SimpleNamespace(
        entry_slippage_bps=10,
        entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3,
        entry_slippage_max_bps=30,
        exit_slippage_bps=5,
        stop_order_type="market",
    )


@pytest.fixture
def policy(mock_alpaca, policy_cfg, tmp_store):
    from orb_live.execution.order_policy import MarketableLimitPolicy
    return MarketableLimitPolicy(
        mock_alpaca, policy_cfg, tmp_store, _sleep=_NO_SLEEP,
    )


# ── Test cases ─────────────────────────────────────────────────────────────────

def test_a_entry_full_fill_first_attempt(policy, mock_alpaca):
    """Entry fills on the first attempt: qty==requested, reason=='filled', attempts==1."""
    fill = policy.buy(SYMBOL, 100, "entry",
                      reference_price=100.0, session_date=TRADE_DATE)
    assert fill.qty == 100
    assert fill.reason == "filled"
    assert fill.attempts == 1
    # avg_price == limit price (entry_slippage_bps=10 → 100 * 1.001 = 100.10)
    assert abs(fill.avg_price - 100.10) < 1e-9


def test_b_entry_partial_first_repeg_full(policy, mock_alpaca):
    """Entry partially fills on attempt 1, full fill on repeg attempt 2."""
    # First submit: 50/100 (partial). Second submit: 50/50 (full).
    mock_alpaca.set_fill_sequence(0.5, 1.0)
    fill = policy.buy(SYMBOL, 100, "entry",
                      reference_price=100.0, session_date=TRADE_DATE)
    assert fill.qty == 100
    assert fill.reason == "filled"
    assert fill.attempts == 2
    # Weighted avg: (50*100.10 + 50*100.30) / 100 = 100.20
    assert abs(fill.avg_price - 100.20) < 1e-6


def test_c_entry_partial_after_max_repegs(policy, mock_alpaca):
    """
    Entry fills half on attempt 1, zero on attempts 2 and 3.
    Result: 50 shares filled, reason='partial_unfilled'.
    """
    mock_alpaca.set_fill_sequence(0.5, 0.0, 0.0)
    fill = policy.buy(SYMBOL, 100, "entry",
                      reference_price=100.0, session_date=TRADE_DATE)
    assert fill.qty == 50
    assert fill.reason == "partial_unfilled"
    assert fill.attempts == 3


def test_d_entry_all_repegs_unfilled(policy, mock_alpaca):
    """All 3 attempts produce zero fills → qty==0, reason=='unfilled'."""
    mock_alpaca.set_fill_fraction(0.0)
    fill = policy.buy(SYMBOL, 100, "entry",
                      reference_price=100.0, session_date=TRADE_DATE)
    assert fill.qty == 0
    assert fill.reason == "unfilled"
    assert fill.attempts == 3


def test_e_exit_leg_full_fill(policy, mock_alpaca):
    """TP exit (tp1 leg) fills on first attempt."""
    fill = policy.sell(SYMBOL, 50, "tp1",
                       reference_price=101.0, session_date=TRADE_DATE)
    assert fill.qty == 50
    assert fill.reason == "filled"
    assert fill.attempts == 1
    # exit_slippage_bps=5 → 101.0 * (1 - 0.0005) = 100.9495
    assert abs(fill.avg_price - 100.9495) < 1e-6


def test_f_exit_leg_raises_after_three_attempts(policy, mock_alpaca):
    """_submit_exit raises RuntimeError when all 3 repeg attempts produce zero fills."""
    mock_alpaca.set_fill_fraction(0.0)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        policy.sell(SYMBOL, 50, "tp1",
                    reference_price=101.0, session_date=TRADE_DATE)


def test_g_stop_market_order_type(policy, mock_alpaca, policy_cfg):
    """stop_order_type='market' submits a market order (not a limit order)."""
    # Default policy_cfg already has stop_order_type='market'
    assert policy_cfg.stop_order_type == "market"
    fill = policy.sell(SYMBOL, 100, "stop",
                       reference_price=99.0, session_date=TRADE_DATE)
    # Market order fills at bid (99.90)
    assert fill.qty == 100
    assert abs(fill.avg_price - 99.90) < 1e-9
    # Verify a market order was stored (no limit price in the order record)
    order_ids = list(mock_alpaca._orders.keys())
    assert len(order_ids) == 1
    # Market orders use the quote price, not a computed limit
    stored = mock_alpaca._orders[order_ids[0]]
    assert float(stored["filled_avg_price"]) == 99.90


def test_h_stop_limit_falls_back_to_market(mock_alpaca, tmp_store):
    """
    stop_order_type='stop_limit': if limit exits fail after repegs, falls back
    to a market order and returns a filled Fill.
    """
    cfg = SimpleNamespace(
        entry_slippage_bps=10,
        entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3,
        entry_slippage_max_bps=30,
        exit_slippage_bps=5,
        stop_order_type="stop_limit",  # limit-first mode
    )
    from orb_live.execution.order_policy import MarketableLimitPolicy
    policy = MarketableLimitPolicy(
        mock_alpaca, cfg, tmp_store, _sleep=_NO_SLEEP,
    )
    mock_alpaca.set_fill_fraction(0.0)   # all limit orders fail
    fill = policy.sell(SYMBOL, 100, "stop",
                       reference_price=99.0, session_date=TRADE_DATE)
    # Despite limit failures, market fallback succeeds
    assert fill.qty == 100
    # Market fill uses bid price (99.90)
    assert abs(fill.avg_price - 99.90) < 1e-9
