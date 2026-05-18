"""
reference/_production_run.py — re-exports from _production_run.py.

Allows:  from reference._production_run import ACTIVE, UNIVERSE, SIGMA, ps_filter, ...
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from _production_run import (  # noqa: F401, E402
    ACTIVE,
    UNIVERSE,
    ETH_SYMS,
    GAP_FILTER,
    DOW_EXCL,
    SYMS,
    SIGMA,
    ps_filter,
    PS_FILTERS,
    CLASS_A_EXCL_SYMS,
    CRYPTO_UL,
    SINGLESTK_UL,
    NATGAS_UL,
    INTERNATIONAL_SYMS,
    BIOTECH_SYMS,
    BANKING_SYMS,
    DRIFT_SYMS,
    CLASS_A_UL,
    CLASS_A_SYMS,
    FLOOD_LOSERS,
    _is_class_a,
    BASELINE_PCT,
    CAP_UNITS,
    apply_sizing,
    cfg,
)
