"""
tests/test_operational.py — Unit tests for ops/ layer.

Covers:
  - HealthServer endpoint logic (no real HTTP)
  - AlertManager rate-limiting and burst digest
  - long_term_logging.daily_rollup idempotency and was_half_day field
  - long_term_logging.cleanup_old_logs retention policy
  - quarterly_calibration.detect_metric_drift on synthetic data
  - BarRouter WS token refresh cadence and degraded fallback
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import orb_live  # noqa: F401

ET = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════════════════════════
# HealthServer tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_store_stub(positions=None):
    class _S:
        def all_open_positions(self): return positions or []
        def conn(self): return None
    return _S()


def _make_clock_stub(rth=True):
    class _C:
        def is_rth_open(self): return rth
        def is_half_day(self): return False
        def effective_close(self):
            from datetime import time as dtime
            return dtime(16, 0)
    return _C()


def _make_router_stub(last_bar_ts=None):
    class _R:
        _last_bar_ts = {}
        _bars_received = {}
        _last_reconnect_ts = None
        _last_reconnect_duration_s = None
        _ws_reconnect_count = 0
        ws_token_age_minutes = None
    r = _R()
    if last_bar_ts is not None:
        r._last_bar_ts = {"TQQQ": last_bar_ts}
    return r


def _make_alpaca_stub(equity=100_000.0):
    class _A:
        def get_account(self): return {"equity": equity}
    return _A()


def _make_health_server(
    store=None, broker=None, router=None, clock=None, port=19999
):
    from orb_live.ops.health_check import HealthServer
    return HealthServer(
        state_store=store or _make_store_stub(),
        broker=broker or _make_alpaca_stub(),
        bar_router=router or _make_router_stub(),
        clock=clock or _make_clock_stub(rth=False),
        port=port,
    )


def test_op01_health_200_outside_rth():
    """Outside RTH the health check returns 200 even with no recent bars."""
    hs = _make_health_server(clock=_make_clock_stub(rth=False))
    code, body = hs.get_health_response()
    assert code == 200
    assert "OK" in body


def test_op02_health_503_stale_bar_during_rth():
    """During RTH, if the last bar is >90s old, health returns 503."""
    stale_ts = datetime.now(ET) - timedelta(seconds=200)
    router   = _make_router_stub(last_bar_ts=stale_ts)
    hs       = _make_health_server(clock=_make_clock_stub(rth=True), router=router)

    code, body = hs.get_health_response()
    assert code == 503
    assert "stale" in body.lower()


def test_op03_health_200_fresh_bar_during_rth():
    """During RTH, a bar received <90s ago produces 200."""
    fresh_ts = datetime.now(ET) - timedelta(seconds=30)
    router   = _make_router_stub(last_bar_ts=fresh_ts)
    hs       = _make_health_server(clock=_make_clock_stub(rth=True), router=router)
    hs.record_api_call()  # mark Alpaca alive

    code, body = hs.get_health_response()
    assert code == 200, f"Expected 200 with fresh bar, got {code}: {body}"


def test_op04_metrics_prometheus_format(tmp_store):
    """GET /metrics returns valid Prometheus text with expected metric names."""
    router = _make_router_stub()
    router._bars_received = {"TQQQ": 42, "SQQQ": 17}

    class _FakeAlpaca:
        def get_account(self): return {"equity": 123456.78}

    class _FakeStore:
        def all_open_positions(self): return [{"symbol": "TQQQ"}]
        def conn(self):
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                class _Conn:
                    def execute(self, *a, **kw):
                        class _R:
                            def mappings(self): return self
                            def all(self): return []
                        return _R()
                yield _Conn()
            return _cm()

    hs = _make_health_server(
        store=_FakeStore(), broker=_FakeAlpaca(), router=router
    )
    hs.record_api_call()
    text = hs.get_metrics_response()

    assert "orb_bars_received_total" in text
    assert 'symbol="TQQQ"} 42' in text
    assert "orb_open_positions_gauge" in text
    assert "orb_account_equity_gauge" in text
    assert "# HELP" in text
    assert "# TYPE" in text


# ═══════════════════════════════════════════════════════════════════════════════
# AlertManager tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_alert_manager(min_level="INFO", info_opt_in=True):
    from orb_live.ops.alerts import AlertManager
    return AlertManager(
        webhook_url="http://fake-webhook",
        min_level=min_level,
        info_opt_in=info_opt_in,
    )


def test_op05_alert_dedup_suppresses_repeat_warn():
    """Sending the same WARN twice within 5 min should only post once."""
    mgr = _make_alert_manager()
    posts: list[tuple] = []
    mgr._post = lambda lvl, msg, ctx: posts.append((lvl, msg))

    mgr.send_alert("WARN", "disk space low")
    mgr.send_alert("WARN", "disk space low")  # duplicate

    assert len(posts) == 1, f"Expected 1 post (dedup), got {len(posts)}"


def test_op06_alert_critical_never_deduped():
    """CRITICAL alerts bypass dedup and always fire."""
    mgr = _make_alert_manager()
    posts: list[tuple] = []
    mgr._post = lambda lvl, msg, ctx: posts.append((lvl, msg))

    mgr.send_alert("CRITICAL", "connection lost")
    mgr.send_alert("CRITICAL", "connection lost")  # same message

    assert len(posts) == 2, (
        f"CRITICAL must never be deduplicated; expected 2 posts, got {len(posts)}"
    )


def test_op07_alert_burst_digest():
    """
    Sending >10 WARNs within 60s triggers burst mode and produces a digest
    rather than individual posts.
    """
    from orb_live.ops.alerts import _BURST_THRESHOLD
    mgr = _make_alert_manager()
    posts: list[tuple] = []
    mgr._post = lambda lvl, msg, ctx: posts.append((lvl, msg))

    # Fire enough WARNs to trigger burst (threshold + 5 more)
    n = _BURST_THRESHOLD + 5
    for i in range(n):
        mgr.send_alert("WARN", f"warn event {i}")

    # Some posts from before burst + at least one digest
    assert len(posts) < n, (
        f"Burst mode should collapse {n} WARNs into fewer posts; got {len(posts)}"
    )
    # The digest post should mention count
    digest_posts = [p for p in posts if "WARN events" in p[1]]
    assert digest_posts, f"Expected a digest post, got: {posts}"


def test_op08_alert_info_below_min_level_dropped():
    """INFO alerts are dropped when min_level=WARN."""
    mgr = _make_alert_manager(min_level="WARN", info_opt_in=False)
    posts: list[tuple] = []
    mgr._post = lambda lvl, msg, ctx: posts.append((lvl, msg))

    mgr.send_alert("INFO", "session started")
    assert len(posts) == 0, "INFO must be dropped when min_level=WARN"


# ═══════════════════════════════════════════════════════════════════════════════
# long_term_logging tests
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_store_for_rollup(store, session_date: date, n_trades: int = 2) -> None:
    """Insert minimal rows so daily_rollup has real data to archive."""
    from datetime import timezone
    UTC = timezone.utc
    for i in range(n_trades):
        store.save_closed_trade(
            trade_date=session_date,
            symbol="TQQQ",
            direction=1,
            entry_price=100.0,
            exit_price=101.0 + i,
            qty=100.0,
            pnl_pct=0.01 * (i + 1),
            dollar_pnl=100.0 * (i + 1),
            exit_reason="tp1",
            opened_at=datetime(session_date.year, session_date.month,
                               session_date.day, 10, 1, 0, tzinfo=UTC),
        )
    store.record_equity(session_date,
                        start_equity=100_000.0,
                        end_equity=100_500.0,
                        session_pnl=500.0)


def test_op09_daily_rollup_idempotent(tmp_path, tmp_store):
    """
    Running daily_rollup twice for the same session_date produces the same
    parquet rows (no duplicates).
    """
    from orb_live.ops.long_term_logging import daily_rollup

    archive_base = tmp_path / "archive"
    session_date = date(2026, 3, 15)
    _seed_store_for_rollup(tmp_store, session_date, n_trades=3)

    daily_rollup(session_date, tmp_store, archive_base)
    daily_rollup(session_date, tmp_store, archive_base)  # second run — must dedup

    summary_path = archive_base / "2026" / "daily_summary.parquet"
    assert summary_path.exists(), "daily_summary.parquet should be written"
    df = pd.read_parquet(summary_path)
    # Exactly one row for session_date
    mask = pd.to_datetime(df["date"]).dt.date == session_date
    assert mask.sum() == 1, (
        f"Expected 1 row for {session_date} after idempotent rollup, "
        f"got {mask.sum()}"
    )


def test_op10_daily_rollup_was_half_day(tmp_path, tmp_store):
    """
    daily_rollup with is_half_day=True must persist was_half_day=True
    in daily_summary.parquet.
    """
    from orb_live.ops.long_term_logging import daily_rollup

    archive_base = tmp_path / "archive"
    session_date = date(2026, 7, 3)
    _seed_store_for_rollup(tmp_store, session_date, n_trades=1)

    daily_rollup(session_date, tmp_store, archive_base, is_half_day=True)

    summary_path = archive_base / "2026" / "daily_summary.parquet"
    df = pd.read_parquet(summary_path)
    mask = pd.to_datetime(df["date"]).dt.date == session_date
    assert mask.any(), "Row not found in daily_summary.parquet"
    row = df.loc[mask].iloc[0]
    assert bool(row["was_half_day"]) is True, (
        f"Expected was_half_day=True, got {row['was_half_day']}"
    )


def test_op11_cleanup_old_logs_deletes_expired(tmp_path):
    """
    cleanup_old_logs deletes .log files older than retain_days (default 90).
    """
    import gzip
    from orb_live.ops.long_term_logging import cleanup_old_logs

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create files with different ages
    old_file = log_dir / "old_2024_01_01.log"
    new_file = log_dir / "new_today.log"
    old_file.write_text("old log data")
    new_file.write_text("new log data")

    # Back-date the old file to 100 days ago
    old_ts = time.time() - 100 * 86400
    import os
    os.utime(old_file, (old_ts, old_ts))

    cleanup_old_logs(log_dir, retain_days=90, compress_after_days=30)

    assert not old_file.exists(), "Old log (100d) should be deleted"
    assert new_file.exists(),     "New log file should be retained"


def test_op12_cleanup_old_logs_compresses_medium_age(tmp_path):
    """
    Log files older than compress_after_days but < retain_days are compressed.
    """
    import gzip
    import os
    from orb_live.ops.long_term_logging import cleanup_old_logs

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    medium_file = log_dir / "medium_age.log"
    medium_file.write_text("medium age log data")

    # 45 days ago: past compress threshold (30d) but within retain (90d)
    medium_ts = time.time() - 45 * 86400
    os.utime(medium_file, (medium_ts, medium_ts))

    cleanup_old_logs(log_dir, retain_days=90, compress_after_days=30)

    gz_path = medium_file.with_suffix(".log.gz")
    assert gz_path.exists(), "Medium-age log should be compressed to .log.gz"
    assert not medium_file.exists(), "Original .log file should be removed after compression"


# ═══════════════════════════════════════════════════════════════════════════════
# quarterly_calibration tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_op13_detect_metric_drift_flags_outlier(tmp_path):
    """
    detect_metric_drift should flag 'sharpe' as drifted when the current
    quarter's value is far outside the historical mean ± 2σ.
    """
    from orb_live.ops.quarterly_calibration import detect_metric_drift
    from orb_live.ops.long_term_logging import _parquet_path, _save_parquet, _archive_dir

    archive_base = tmp_path / "archive"

    # Build synthetic daily_summary: 4 years of historical data with varying PnL
    # Variation is needed so std_h > 0 and z-score is computable.
    rows = []
    import random
    rng = random.Random(42)
    for year in range(2021, 2025):
        for month in range(1, 13):
            for day in range(1, 21):
                try:
                    d = date(year, month, day)
                except ValueError:
                    continue
                # PnL varies by quarter: Q1=180, Q2=210, Q3=195, Q4=215, etc.
                # This gives std_h > 0 across quarters so z-scores are meaningful.
                quarter = (month - 1) // 3
                base_pnl = [180.0, 210.0, 195.0, 215.0][quarter]
                pnl = base_pnl + rng.uniform(-5.0, 5.0)
                rows.append({
                    "date": d,
                    "n_entered": 5,
                    "realized_pnl": pnl,
                    "ending_equity": 100_000.0 + len(rows) * pnl,
                    "n_candidates": 10,
                    "n_passed_preflight": 8,
                    "n_exited_tp1": 2, "n_exited_tp2": 1, "n_exited_tp3": 1,
                    "n_stopped": 1, "n_eod": 0,
                    "n_partial_fills": 0, "n_repegs": 0, "n_reconnects": 0,
                    "max_reconnect_duration": 0.0, "max_open_positions": 5,
                    "was_half_day": False,
                })

    # Current quarter (last 90 days) with extreme negative PnL → drift
    today = date(2025, 4, 1)
    for i in range(60):
        d = today - timedelta(days=60 - i)
        rows.append({
            "date": d,
            "n_entered": 5,
            "realized_pnl": -2000.0,   # extreme negative — should trigger drift
            "ending_equity": 50_000.0,
            "n_candidates": 10,
            "n_passed_preflight": 8,
            "n_exited_tp1": 0, "n_exited_tp2": 0, "n_exited_tp3": 0,
            "n_stopped": 5, "n_eod": 0,
            "n_partial_fills": 0, "n_repegs": 0, "n_reconnects": 0,
            "max_reconnect_duration": 0.0, "max_open_positions": 5,
            "was_half_day": False,
        })

    df   = pd.DataFrame(rows)
    path = _parquet_path(archive_base, 2025, "daily_summary")
    _archive_dir(archive_base, 2021)
    _archive_dir(archive_base, 2022)
    _archive_dir(archive_base, 2023)
    _archive_dir(archive_base, 2024)

    # Save all rows into 2025 parquet (detect_metric_drift reads all years)
    for yr in range(2021, 2026):
        yr_rows = [r for r in rows if r["date"].year == yr]
        if yr_rows:
            p = _parquet_path(archive_base, yr, "daily_summary")
            _save_parquet(pd.DataFrame(yr_rows), p)

    result = detect_metric_drift(archive_base, today=today)

    assert "mean_pnl" in result, f"Expected 'mean_pnl' in drift result, got {list(result.keys())}"
    assert result["mean_pnl"]["drifted"] is True, (
        "Expected mean_pnl to be flagged as drifted with extreme negative PnL"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BarRouter WS token refresh tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_op14_token_refresh_calls_stop_bars_stream():
    """
    _do_token_refresh should call alpaca.stop_bars_stream() to trigger reconnect.
    """
    from orb_live.runner.bar_router import BarRouter

    stop_called: list[bool] = []

    class _FakeAlpaca:
        def stop_bars_stream(self): stop_called.append(True)

    class _FakeStore: pass
    class _FakeCache: pass

    router = BarRouter(_FakeAlpaca(), _FakeStore(), _FakeCache())
    router._do_token_refresh(age_s=43200.0)

    assert len(stop_called) == 1, "stop_bars_stream should be called once on token refresh"


def test_op15_token_refresh_failure_enters_degraded_mode():
    """
    If stop_bars_stream raises, _do_token_refresh should enter degraded mode
    and log a CRITICAL event.
    """
    from orb_live.runner.bar_router import BarRouter

    log_events: list[str] = []

    class _Log:
        def critical(self, event, **kw): log_events.append(event)
        def info(self, event, **kw): log_events.append(event)
        def warning(self, event, **kw): log_events.append(event)
        def error(self, event, **kw): log_events.append(event)

    class _BrokenAlpaca:
        def stop_bars_stream(self): raise RuntimeError("auth error")

    class _FakeStore: pass
    class _FakeCache: pass

    router = BarRouter(_BrokenAlpaca(), _FakeStore(), _FakeCache(), logger=_Log())
    router._do_token_refresh(age_s=43200.0)

    assert router._degraded is True, (
        "Router should enter degraded mode after stop_bars_stream failure"
    )
    assert "ws_token_refresh_failed" in log_events, (
        f"Expected 'ws_token_refresh_failed' CRITICAL log; got: {log_events}"
    )
