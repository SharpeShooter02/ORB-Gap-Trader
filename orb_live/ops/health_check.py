"""
ops/health_check.py — Lightweight HTTP health / metrics server.

Started by SessionRunner at process startup; runs in a background daemon thread.

Endpoints
---------
  GET /health   — 200 OK when healthy, 503 with reason when not
  GET /status   — JSON snapshot of session + operational state
  GET /positions — JSON list of open positions from state_store
  GET /metrics  — Prometheus-format metrics scrape

Usage
-----
    server = HealthServer(store, broker, bar_router, clock)
    server.start()   # non-blocking
    ...
    server.stop()    # graceful shutdown
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_RTH_MAX_BAR_GAP_S = 90.0   # stale if no bar received within this many seconds
_BROKER_STALE_S    = 300.0   # stale if no successful API call within this window


class HealthServer:
    """
    HTTP server providing health, status, positions, and Prometheus metrics.

    All endpoint logic is exposed as plain methods (get_health_response, etc.)
    so they can be unit-tested without starting an actual HTTP server.
    """

    def __init__(
        self,
        state_store,
        broker,
        bar_router,
        clock,
        session_state_fn: Optional[Callable[[], dict]] = None,
        port: int = 8080,
        logger=None,
    ):
        self._store            = state_store
        self._broker           = broker
        self._router           = bar_router
        self._clock            = clock
        self._session_state_fn = session_state_fn or (lambda: {})
        self._port             = port
        self._log              = logger
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._last_api_call_ts: Optional[datetime] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        _self = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # suppress default request log
                pass

            def do_GET(self):
                if self.path == "/health":
                    code, body = _self.get_health_response()
                    self._write(code, "text/plain", body)
                elif self.path == "/status":
                    body = json.dumps(_self.get_status_response(), default=str)
                    self._write(200, "application/json", body)
                elif self.path == "/positions":
                    body = json.dumps(_self.get_positions_response(), default=str)
                    self._write(200, "application/json", body)
                elif self.path == "/metrics":
                    self._write(200, "text/plain; version=0.0.4",
                                _self.get_metrics_response())
                else:
                    self._write(404, "text/plain", "Not Found")

            def _write(self, code: int, content_type: str, body: str) -> None:
                data = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = HTTPServer(("127.0.0.1", self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="HealthServer",
        )
        self._thread.start()
        if self._log:
            self._log.info("health_server_started", port=self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def record_api_call(self) -> None:
        """Notify the health server that a broker API call succeeded."""
        self._last_api_call_ts = datetime.now(ET)

    # ── Endpoint logic (public so tests call them without HTTP) ───────────────

    def get_health_response(self) -> tuple[int, str]:
        """
        Return (status_code, body).

        During RTH: 200 only if last bar < 90s old AND broker connection < 5min old
                    AND state_store reachable.
        Outside RTH: 200 if process alive and state_store reachable.
        """
        issues: list[str] = []

        # Store reachability (always checked)
        try:
            self._store.all_open_positions()
        except Exception as exc:
            issues.append(f"state_store_unreachable: {exc}")

        if self._clock.is_rth_open():
            now = datetime.now(ET)

            # Last bar freshness
            last_bar_ts = self._get_last_bar_ts()
            if last_bar_ts is not None:
                gap_s = (now - last_bar_ts).total_seconds()
                if gap_s > _RTH_MAX_BAR_GAP_S:
                    issues.append(f"last_bar_stale: {gap_s:.0f}s ago")
            # Only flag missing bars after the first minute of RTH
            # (bars don't exist at exactly 09:30)

            # Broker connection freshness
            if self._last_api_call_ts is not None:
                api_age_s = (now - self._last_api_call_ts).total_seconds()
                if api_age_s > _BROKER_STALE_S:
                    issues.append(f"broker_stale: {api_age_s:.0f}s since last API call")

        if issues:
            return 503, "UNHEALTHY: " + "; ".join(issues)
        return 200, "OK"

    def get_status_response(self) -> dict:
        session_state = self._session_state_fn()
        is_half       = self._clock.is_half_day()
        eff_close     = self._clock.effective_close()
        last_bar_ts   = self._get_last_bar_ts()

        last_reconnect_ts  = getattr(self._router, "_last_reconnect_ts", None)
        last_reconnect_dur = getattr(self._router, "_last_reconnect_duration_s", None)

        return {
            "session_date":       str(session_state.get("session_date",
                                       datetime.now(ET).date())),
            "session_state":      session_state.get("session_state", "idle"),
            "is_half_day":        is_half,
            "todays_close_et":    f"{eff_close.hour:02d}:{eff_close.minute:02d}",
            "active_candidates":  session_state.get("active_candidates", 0),
            "open_positions":     len(self._store.all_open_positions()),
            "todays_realized_pnl":   self._get_todays_pnl(),
            "todays_unrealized_pnl": session_state.get("todays_unrealized_pnl", 0.0),
            "session_kill_active":   session_state.get("session_kill_active", False),
            "last_bar_received_ts":  last_bar_ts.isoformat() if last_bar_ts else None,
            "last_reconnect_ts":     last_reconnect_ts.isoformat()
                                     if last_reconnect_ts else None,
            "last_reconnect_duration_seconds": last_reconnect_dur,
            "sigma_calibration_age_days": self._sigma_age_days(),
            "underlying_data_freshness":  self._underlying_freshness(),
        }

    def get_positions_response(self) -> list:
        return self._store.all_open_positions()

    def get_metrics_response(self) -> str:
        lines: list[str] = []

        # Bar counter per symbol
        bars_recv: dict = getattr(self._router, "_bars_received", {})
        if bars_recv:
            lines += [
                "# HELP orb_bars_received_total Total 1-min bars received",
                "# TYPE orb_bars_received_total counter",
            ]
            for sym, cnt in sorted(bars_recv.items()):
                lines.append(f'orb_bars_received_total{{symbol="{sym}"}} {cnt}')

        # Orders / fills from fills table
        try:
            today = datetime.now(ET).date()
            from orb_live.core.state_store import fills as fills_tbl
            with self._store.conn() as c:
                rows = c.execute(
                    fills_tbl.select().where(fills_tbl.c.session_date == today)
                ).mappings().all()
            orders_by: dict = defaultdict(int)
            fills_by:  dict = defaultdict(int)
            for r in rows:
                key = (r["symbol"], r["leg"])
                orders_by[key] += 1
                if r["reason"] == "filled":
                    fills_by[key] += 1
            if orders_by:
                lines += ["# HELP orb_orders_submitted_total Orders submitted",
                          "# TYPE orb_orders_submitted_total counter"]
                for (sym, leg), cnt in sorted(orders_by.items()):
                    lines.append(
                        f'orb_orders_submitted_total{{symbol="{sym}",leg="{leg}"}} {cnt}'
                    )
            if fills_by:
                lines += ["# HELP orb_fills_received_total Orders confirmed filled",
                          "# TYPE orb_fills_received_total counter"]
                for (sym, leg), cnt in sorted(fills_by.items()):
                    lines.append(
                        f'orb_fills_received_total{{symbol="{sym}",leg="{leg}"}} {cnt}'
                    )
        except Exception:
            pass

        # Gauges
        try:
            n_pos = len(self._store.all_open_positions())
            lines += ["# HELP orb_open_positions_gauge Current open positions",
                      "# TYPE orb_open_positions_gauge gauge",
                      f"orb_open_positions_gauge {n_pos}"]
        except Exception:
            pass

        try:
            pnl = self._get_todays_pnl()
            lines += ["# HELP orb_today_pnl_gauge Today realized P&L ($)",
                      "# TYPE orb_today_pnl_gauge gauge",
                      f"orb_today_pnl_gauge {pnl:.4f}"]
        except Exception:
            pass

        try:
            equity = float(self._broker.get_account().get("equity", 0.0))
            self.record_api_call()
            lines += ["# HELP orb_account_equity_gauge Account equity ($)",
                      "# TYPE orb_account_equity_gauge gauge",
                      f"orb_account_equity_gauge {equity:.4f}"]
        except Exception:
            pass

        lrt = getattr(self._router, "_last_reconnect_ts", None)
        if lrt is not None:
            age_s = (datetime.now(ET) - lrt).total_seconds()
            lines += ["# HELP orb_last_reconnect_age_seconds_gauge Seconds since last WS reconnect",
                      "# TYPE orb_last_reconnect_age_seconds_gauge gauge",
                      f"orb_last_reconnect_age_seconds_gauge {age_s:.1f}"]

        rcount = getattr(self._router, "_ws_reconnect_count", None)
        if rcount is not None:
            lines += ["# HELP orb_ws_reconnects_total Total WebSocket reconnects",
                      "# TYPE orb_ws_reconnects_total counter",
                      f"orb_ws_reconnects_total {rcount}"]

        return "\n".join(lines) + "\n"

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_last_bar_ts(self) -> Optional[datetime]:
        ts_map: dict = getattr(self._router, "_last_bar_ts", {})
        candidates = [v for v in ts_map.values() if v is not None]
        return max(candidates) if candidates else None

    def _sigma_age_days(self) -> int:
        try:
            from orb_live.core.state_store import sigma_calibration as sct
            with self._store.conn() as c:
                row = c.execute(
                    sct.select().order_by(sct.c.calibrated_at.desc()).limit(1)
                ).mappings().first()
            if row and row["calibrated_at"]:
                cal_dt = row["calibrated_at"]
                if hasattr(cal_dt, "date"):
                    cal_dt = cal_dt.date()
                return (datetime.now(ET).date() - cal_dt).days
        except Exception:
            pass
        return -1

    def _underlying_freshness(self) -> dict:
        try:
            from orb_live.core.state_store import underlying_bars as ubt
            with self._store.conn() as c:
                rows = c.execute(
                    ubt.select().order_by(ubt.c.bar_date.desc())
                ).mappings().all()
            result: dict = {}
            for r in rows:
                if r["underlying"] not in result:
                    result[r["underlying"]] = str(r["bar_date"])
            return result
        except Exception:
            return {}

    def _get_todays_pnl(self) -> float:
        try:
            today = datetime.now(ET).date()
            from orb_live.core.state_store import closed_trades as ctt
            with self._store.conn() as c:
                rows = c.execute(
                    ctt.select().where(ctt.c.trade_date == today)
                ).mappings().all()
            return sum(float(r["dollar_pnl"] or 0) for r in rows)
        except Exception:
            return 0.0
