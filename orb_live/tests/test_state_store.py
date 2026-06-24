"""
tests/test_state_store.py — StateStore CRUD and schema tests.
"""

from datetime import date, datetime

import pytest


def test_tables_created(tmp_store):
    from sqlalchemy import inspect
    inspector = inspect(tmp_store.engine)
    tables = inspector.get_table_names()
    expected = {
        "universe", "day_state", "gap_scan", "ps_filter_result",
        "orb_window", "breakout_signal", "pending_orders", "open_positions",
        "closed_trades", "equity_curve", "sigma_calibration",
        "alert_log", "system_events", "underlying_bars",
        "rtg_history", "liquidity_metrics", "candidates", "sigma_history",
        "fills", "indicator_state",
    }
    assert expected == set(tables), f"Missing tables: {expected - set(tables)}"


def test_day_state_upsert(tmp_store):
    d = date(2026, 1, 2)
    tmp_store.upsert_day_state(d, phase="pre_market", n_prequalified=5)
    row = tmp_store.get_day_state(d)
    assert row["phase"] == "pre_market"
    assert row["n_prequalified"] == 5

    tmp_store.upsert_day_state(d, phase="orb_window", n_prequalified=7)
    row = tmp_store.get_day_state(d)
    assert row["phase"] == "orb_window"
    assert row["n_prequalified"] == 7


def test_kill_switch(tmp_store):
    d = date(2026, 1, 3)
    assert not tmp_store.is_kill_triggered(d)
    tmp_store.set_kill_switch(d, reason="session_loss_exceeded")
    assert tmp_store.is_kill_triggered(d)


def test_gap_scan_qualify(tmp_store):
    d = date(2026, 1, 4)
    tmp_store.save_gap_scan(d, "TQQQ", gap_pct=0.08, qualifies=True)
    tmp_store.save_gap_scan(d, "SQQQ", gap_pct=0.03, qualifies=False, filter_reason="below_threshold")
    syms = tmp_store.get_qualifying_symbols(d)
    assert "TQQQ" in syms
    assert "SQQQ" not in syms


def test_ps_filter(tmp_store):
    d = date(2026, 1, 5)
    tmp_store.save_ps_filter(d, "TQQQ", passed=True, underlying="QQQ", ul_move_pct=0.012)
    tmp_store.save_ps_filter(d, "JNUG", passed=False, underlying="GDXJ", ul_move_pct=0.08)
    passed = tmp_store.get_ps_passed_symbols(d)
    assert "TQQQ" in passed
    assert "JNUG" not in passed


def test_open_close_position(tmp_store):
    d = date(2026, 1, 6)
    tmp_store.save_open_position("TQQQ", d, direction=1, entry_price=100.0, qty=10.0)
    pos = tmp_store.get_open_position("TQQQ")
    assert pos is not None
    assert pos["entry_price"] == 100.0
    tmp_store.close_position("TQQQ")
    assert tmp_store.get_open_position("TQQQ") is None


def test_equity_record(tmp_store):
    d = date(2026, 1, 7)
    tmp_store.record_equity(d, 100_000.0, 101_500.0, 1_500.0)
    from orb_live.core.state_store import equity_curve
    with tmp_store.conn() as c:
        rows = c.execute(equity_curve.select()).mappings().all()
    assert len(rows) == 1
    assert rows[0]["session_pnl"] == pytest.approx(1500.0)


def test_alert_log(tmp_store):
    tmp_store.log_alert("WARN", "test alert", category="smoke", symbol="TQQQ",
                        trade_date=date.today())
    from orb_live.core.state_store import alert_log
    with tmp_store.conn() as c:
        rows = c.execute(alert_log.select()).mappings().all()
    assert len(rows) == 1
    assert rows[0]["level"] == "WARN"


def test_system_events(tmp_store):
    tmp_store.log_event("startup", "test run")
    from orb_live.core.state_store import system_events
    with tmp_store.conn() as c:
        rows = c.execute(system_events.select()).mappings().all()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "startup"


def test_migrate_adds_missing_column(tmp_path):
    """migrate_db must ADD COLUMN for any model column absent from existing tables.

    Simulates a live.db created before realized_exit_price was added to
    closed_trades — the crash path the suite never exercises because it always
    builds a fresh schema.
    """
    from sqlalchemy import create_engine
    from sqlalchemy import text as sa_text
    from orb_live.core.state_store import create_db_engine, init_db, migrate_db

    db_path = tmp_path / "legacy.db"
    engine = create_db_engine(db_path)

    # Create closed_trades WITHOUT realized_exit_price (legacy schema)
    with engine.begin() as conn:
        conn.execute(sa_text("""
            CREATE TABLE closed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date DATE NOT NULL,
                symbol VARCHAR(16) NOT NULL,
                direction INTEGER,
                entry_price FLOAT,
                exit_price FLOAT,
                qty FLOAT,
                pnl_pct FLOAT,
                dollar_pnl FLOAT,
                exit_reason VARCHAR(32),
                opened_at DATETIME,
                closed_at DATETIME
            )
        """))

    # Confirm column is missing before migration
    with engine.connect() as conn:
        pragma = conn.execute(sa_text("PRAGMA table_info(closed_trades)")).fetchall()
    col_names = {row[1] for row in pragma}
    assert "realized_exit_price" not in col_names

    # Run init_db (which calls migrate_db internally)
    init_db(engine)

    # Column must now exist
    with engine.connect() as conn:
        pragma = conn.execute(sa_text("PRAGMA table_info(closed_trades)")).fetchall()
    col_names = {row[1] for row in pragma}
    assert "realized_exit_price" in col_names, (
        "migrate_db must add realized_exit_price to legacy closed_trades table"
    )

    # Query must succeed (no 'no such column' error)
    with engine.connect() as conn:
        conn.execute(sa_text("SELECT realized_exit_price FROM closed_trades LIMIT 1"))
