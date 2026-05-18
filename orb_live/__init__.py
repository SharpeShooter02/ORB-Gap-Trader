"""
orb_live — Live trading system for the ORB leveraged-ETF strategy.

Imports reference.* (backtest parity layer). Adds BacktestingGaps root to
sys.path on first import so that `import reference` resolves correctly
regardless of where Python is invoked from.
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
