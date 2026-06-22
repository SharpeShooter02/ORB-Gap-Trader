"""
orb_live/tools/preflight_check.py — Module entry point.

    python -m orb_live.tools.preflight_check

Delegates to tools/preflight_check.py (the canonical implementation).
"""

from __future__ import annotations

import os
import sys

# Resolve tools/ relative to this package so the import works regardless of
# the working directory from which the module is invoked.
_TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from preflight_check import run_preflight  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_preflight())
