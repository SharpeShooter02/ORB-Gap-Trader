"""
tests/test_order_policy.py — Tests for MarketableLimitPolicy price calculator and Fill.

Phase 3 rebuild: MarketableLimitPolicy no longer submits orders or sleeps.
Tests cover compute_entry_limit() and the Fill dataclass.
"""

from datetime import datetime, timezone

import pytest

UTC = timezone.utc
SYMBOL = "TQQQ"


def test_fill_dataclass_fields():
    """Fill can be constructed and has default raw_response_json."""
    from orb_live.execution.order_policy import Fill

    f = Fill(
        symbol="TQQQ", side="buy", qty=100, avg_price=100.5,
        order_id="abc-123", leg="entry", attempts=1, reason="filled",
        submitted_at=datetime.now(UTC), filled_at=datetime.now(UTC),
    )
    assert f.symbol           == "TQQQ"
    assert f.side             == "buy"
    assert f.qty              == 100
    assert f.avg_price        == pytest.approx(100.5)
    assert f.leg              == "entry"
    assert f.reason           == "filled"
    assert f.raw_response_json == "{}"   # default


def test_compute_entry_limit_buy_adds_bps(mock_broker):
    """Buy limit = reference * (1 + bps/10000)."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=10)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    limit  = policy.compute_entry_limit("buy", SYMBOL, reference_price=100.0)

    assert abs(limit - 100.10) < 1e-9   # 100.0 × 1.001


def test_compute_entry_limit_sell_subtracts_bps(mock_broker):
    """Sell limit = reference * (1 - bps/10000)."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=5)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    limit  = policy.compute_entry_limit("sell", SYMBOL, reference_price=100.0)

    assert abs(limit - 99.95) < 1e-9    # 100.0 × 0.9995


def test_compute_entry_limit_uses_nbbo_when_no_reference(mock_broker):
    """Without reference_price, falls back to NBBO ask (buy) or bid (sell)."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=0)
    policy = MarketableLimitPolicy(mock_broker, cfg)

    # mock ask = 100.10, bps=0 → limit = ask exactly
    limit_buy  = policy.compute_entry_limit("buy",  SYMBOL)
    limit_sell = policy.compute_entry_limit("sell", SYMBOL)

    assert abs(limit_buy  - 100.10) < 1e-9
    assert abs(limit_sell -  99.90) < 1e-9


def test_compute_entry_limit_bps_override(mock_broker):
    """bps keyword overrides the config value."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=10)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    limit  = policy.compute_entry_limit("buy", SYMBOL, reference_price=100.0, bps=30)

    assert abs(limit - 100.30) < 1e-9   # bps override (30) takes precedence over cfg (10)


def test_entry_buffer_orb_frac_buy_uses_range(mock_broker):
    """With entry_buffer_orb_frac and orb_range, buy limit = boundary + frac×range."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=10, entry_buffer_orb_frac=0.35)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    # boundary=100, ORB range=2.0 → 100 + 0.35×2.0 = 100.70
    limit  = policy.compute_entry_limit("buy", SYMBOL, reference_price=100.0, orb_range=2.0)

    assert abs(limit - 100.70) < 1e-9


def test_entry_buffer_orb_frac_sell_uses_range(mock_broker):
    """Short side: sell limit = boundary − frac×range."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=10, entry_buffer_orb_frac=0.35)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    # boundary=100, ORB range=2.0 → 100 − 0.35×2.0 = 99.30
    limit  = policy.compute_entry_limit("sell", SYMBOL, reference_price=100.0, orb_range=2.0)

    assert abs(limit - 99.30) < 1e-9


def test_entry_buffer_orb_frac_falls_back_to_bps_when_no_range(mock_broker):
    """No orb_range → range-relative buffer inert, fixed-bps path is used."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=10, entry_buffer_orb_frac=0.35)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    limit  = policy.compute_entry_limit("buy", SYMBOL, reference_price=100.0)

    assert abs(limit - 100.10) < 1e-9   # bps path (10 bps), range buffer not applied


def test_entry_buffer_orb_frac_zero_uses_bps(mock_broker):
    """frac=0 disables the range buffer even when orb_range is supplied."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg    = SimpleNamespace(entry_slippage_bps=10, entry_buffer_orb_frac=0.0)
    policy = MarketableLimitPolicy(mock_broker, cfg)
    limit  = policy.compute_entry_limit("buy", SYMBOL, reference_price=100.0, orb_range=2.0)

    assert abs(limit - 100.10) < 1e-9   # bps path


def test_legacy_kwargs_accepted_without_error(mock_broker, tmp_store):
    """Extra legacy kwargs (_sleep, state_store) are accepted without raising."""
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from types import SimpleNamespace

    cfg       = SimpleNamespace(entry_slippage_bps=10)
    _NO_SLEEP = lambda _: None  # noqa: E731

    policy = MarketableLimitPolicy(mock_broker, cfg,
                                   state_store=tmp_store, _sleep=_NO_SLEEP)
    assert policy is not None
    assert abs(policy.compute_entry_limit("buy", SYMBOL, reference_price=50.0) - 50.05) < 1e-9
