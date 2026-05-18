"""
scripts/force_ws_reconnect.py — Operator tool: signal the live session to
reconnect its WebSocket stream immediately.

How it works
------------
The live session's HealthServer exposes /health and /status.  There is no
direct reconnect endpoint (attack surface), so this script forces a reconnect
by writing a sentinel file that bar_router._token_refresh_watcher checks.
Alternatively, it can simply verify that the process is alive and show stream
age before you perform a manual restart.

Usage:
    python -m orb_live.scripts.force_ws_reconnect --status
    python -m orb_live.scripts.force_ws_reconnect --reconnect
    python -m orb_live.scripts.force_ws_reconnect --reconnect --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Force or inspect WS reconnect for live session")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--status",    action="store_true",
                   help="Print current WS stream status from /status endpoint")
    g.add_argument("--reconnect", action="store_true",
                   help="Signal bar_router to reconnect via sentinel file")
    p.add_argument("--port", type=int, default=8080,
                   help="Health server port (default: 8080)")
    p.add_argument("--sentinel-dir", default=None,
                   help="Directory for sentinel file (default: system temp)")
    return p.parse_args()


def _fetch_status(port: int) -> dict:
    url = f"http://127.0.0.1:{port}/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        print(f"ERROR: could not reach health server at {url}: {exc}",
              file=sys.stderr)
        sys.exit(1)


def cmd_status(port: int) -> None:
    status = _fetch_status(port)
    print("=== ORB Live Session Status ===")
    for k, v in status.items():
        print(f"  {k:40s} {v}")

    ws_age = status.get("ws_token_age_minutes")
    if ws_age is not None:
        if ws_age > 600:
            print(f"\nWARN: WS token age is {ws_age} minutes — approaching 12h refresh")
        else:
            print(f"\nWS token age: {ws_age} min (auto-refresh at 720 min)")


def cmd_reconnect(port: int, sentinel_dir: str | None) -> None:
    # Verify the session is alive first
    status = _fetch_status(port)
    print(f"Session alive: date={status.get('session_date')} "
          f"state={status.get('session_state')}")

    # Write sentinel file that bar_router._token_refresh_watcher checks
    if sentinel_dir:
        base = Path(sentinel_dir)
    else:
        import tempfile
        base = Path(tempfile.gettempdir())

    sentinel = base / "orb_live_force_reconnect.sentinel"
    sentinel.touch()
    print(f"Sentinel written: {sentinel}")
    print("bar_router will detect this on its next refresh-watcher tick "
          "(within 60 s) and reconnect the WebSocket.")
    print("Monitor /status → ws_token_age_minutes resets to 0 on success.")


def main() -> None:
    args = _parse_args()
    if args.status:
        cmd_status(args.port)
    else:
        cmd_reconnect(args.port, args.sentinel_dir)


if __name__ == "__main__":
    main()
