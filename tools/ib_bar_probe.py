"""
tools/ib_bar_probe.py — live probe for BUG 0b (ib.sleep vs time.sleep).

Run during market hours (any day, no open needed):
    python tools/ib_bar_probe.py [--symbol SPY] [--seconds 60] [--paper]

Connects to IB Gateway, subscribes reqRealTimeBars on the given symbol,
and uses ib.sleep() to pump the event loop.  Bars should print within
~5 seconds of market open.  Swap ib.sleep → time.sleep in _wait() below
to reproduce the BUG 0b regression (zero bars arrive).
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orb_live.data.ib_client import build_client_from_env


def _wait(ib, seconds: float, use_ib_sleep: bool) -> None:
    if use_ib_sleep:
        ib.sleep(seconds)
    else:
        time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="IB bar delivery probe")
    parser.add_argument("--symbol",    default="SPY",  help="Symbol to subscribe")
    parser.add_argument("--seconds",   type=float, default=60.0, help="Listen duration")
    parser.add_argument("--paper",     action="store_true", default=True)
    parser.add_argument("--time-sleep", action="store_true",
                        help="Use time.sleep() instead of ib.sleep() (reproduces bug)")
    args = parser.parse_args()

    use_ib_sleep = not args.time_sleep
    print(f"Connecting (paper={args.paper}) …")
    client = build_client_from_env(paper=args.paper)
    client.connect()
    print(f"Connected.  Subscribing {args.symbol} for {args.seconds}s "
          f"using {'ib.sleep' if use_ib_sleep else 'time.sleep (BUG MODE)'} …")

    bars_seen: list = []

    def _on_bar(bar: dict) -> None:
        bars_seen.append(bar)
        ts = bar.get("timestamp", "?")
        close = bar.get("close", "?")
        print(f"  BAR  {ts}  close={close}")

    client.subscribe_bars([args.symbol], _on_bar)

    _wait(client._ib, args.seconds, use_ib_sleep)

    client.stop_bars_stream()
    client.disconnect()

    print(f"\nResult: {len(bars_seen)} bar(s) received in {args.seconds}s")
    if not bars_seen and use_ib_sleep:
        print("WARN: ib.sleep but zero bars — market may be closed or subscription failed")
    elif bars_seen and not use_ib_sleep:
        print("UNEXPECTED: time.sleep delivered bars — loop may be on a separate thread")
    elif not bars_seen and not use_ib_sleep:
        print("BUG CONFIRMED: time.sleep starves the IB event loop")
    sys.exit(0 if bars_seen else 1)


if __name__ == "__main__":
    main()
