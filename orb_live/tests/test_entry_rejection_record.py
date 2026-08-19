"""A rejected entry must leave a durable trace.

On 2026-08-18 six candidates broke out and four were submitted; KORU and
GDXU were dropped by the margin budget and left no record in the database at
all — the reason had to be reconstructed from arithmetic. These tests pin
the fix: every rejection writes a phase-3 candidates row (matching how
risk-gate rejections were already recorded) plus an alert_log row carrying
the numbers behind the decision.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text

from orb_live.core.logger import configure_logging, default_log_file
from orb_live.core.state_store import StateStore, create_db_engine, init_db


@pytest.fixture
def store(tmp_path) -> StateStore:
    db = tmp_path / "t.db"
    init_db(create_db_engine(db))
    return StateStore(db)


class _Recorder:
    """Minimal stand-in exposing only what _record_entry_rejection touches."""

    def __init__(self, store):
        self._store = store
        self._log = None

    from orb_live.execution.position_manager import LivePositionManager

    _record_entry_rejection = LivePositionManager._record_entry_rejection


def test_margin_rejection_is_queryable(store):
    d = date(2026, 8, 18)
    _Recorder(store)._record_entry_rejection(
        d, "KORU", "margin_budget", -1,
        "est_margin=2676.20 committed=10704.85 budget=10696.23 rate=0.9000",
    )

    cands = store.get_candidates(d) if hasattr(store, "get_candidates") else []
    rows = [c for c in cands if (c.get("symbol") if isinstance(c, dict) else c[2]) == "KORU"]
    assert rows, "rejected entry must appear in the candidates table"

    with store.conn() as c:
        alerts = list(c.execute(text(
            "SELECT symbol, category, message FROM alert_log "
            "WHERE category LIKE 'entry_rejected%'"
        )))
    assert len(alerts) == 1
    sym, cat, msg = alerts[0]
    assert sym == "KORU"
    assert cat == "entry_rejected_margin_budget"
    # The numbers must survive — they are what makes the record actionable.
    assert "rate=0.9000" in msg and "budget=10696.23" in msg


def test_buying_power_rejection_is_distinguishable(store):
    d = date(2026, 8, 18)
    r = _Recorder(store)
    r._record_entry_rejection(d, "KORU", "margin_budget", -1, "a=1")
    r._record_entry_rejection(d, "GDXU", "insufficient_buying_power", -1, "b=2")
    with store.conn() as c:
        cats = sorted(x[0] for x in c.execute(text(
            "SELECT category FROM alert_log WHERE category LIKE 'entry_rejected%'")))
    assert cats == ["entry_rejected_insufficient_buying_power",
                    "entry_rejected_margin_budget"]


def test_recording_never_raises_on_broken_store():
    """Bookkeeping must not be able to kill the entry path."""
    class Broken:
        def save_candidate(self, **kw): raise RuntimeError("db gone")
        def log_alert(self, **kw): raise RuntimeError("db gone")

    r = _Recorder.__new__(_Recorder)
    r._store, r._log = Broken(), None
    r._record_entry_rejection(date(2026, 8, 18), "KORU", "margin_budget", -1, "x")


def test_logging_writes_to_a_file_by_default(tmp_path):
    """The 2026-08-18 rejections were invisible because nothing wrote a file."""
    f = tmp_path / "out.jsonl"
    configure_logging(log_file=f)
    from orb_live.core.logger import get_logger
    get_logger("t").warning("entry_rejected_margin_budget", symbol="KORU")
    assert f.exists() and "entry_rejected_margin_budget" in f.read_text(encoding="utf-8")


def test_default_log_file_is_dated():
    p = default_log_file(date(2026, 8, 18))
    assert p.name == "orb_live_2026-08-18.jsonl"
    assert p.parent.name == "logs"
