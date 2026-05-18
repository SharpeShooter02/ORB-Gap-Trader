"""
reference/orb_backtester.py — re-exports from orb_backtester.py.

Allows:  from reference.orb_backtester import StrategyConfig, DEFAULT_CONFIG
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# orb_backtester.py replaces sys.stdout with a UTF-8 TextIOWrapper at module
# load time (needed for Windows console box-drawing characters).  This side
# effect corrupts pytest's SpooledTemporaryFile capture buffer, so we hide
# .buffer from stdout during the first import to suppress the replacement.
if "orb_backtester" not in sys.modules:
    _real_stdout = sys.stdout
    _guard = type("_NoBuffer", (), {})()   # plain object — no .buffer attr
    sys.stdout = _guard
    try:
        import orb_backtester as _ob  # noqa: F401
    finally:
        sys.stdout = _real_stdout
        del _real_stdout, _guard

from orb_backtester import (  # noqa: F401, E402
    StrategyConfig,
    DEFAULT_CONFIG,
    run_backtest,
    compute_metrics,
    simulate_trade,
    compute_gap,
    compute_opening_range,
    check_breakout,
    check_prior_session_filter,
    compute_entry,
)
