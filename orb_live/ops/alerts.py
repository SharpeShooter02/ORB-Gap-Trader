"""
ops/alerts.py — Webhook alert poster for Discord / Telegram / Slack.

Configuration via environment:
    ALERT_WEBHOOK_URL  — full webhook URL
    ALERT_LEVEL        — minimum level to post: INFO | WARN | CRITICAL (default WARN)
    ALERT_INFO_OPT_IN  — set to "true" to enable INFO-level posts (default false)

Rate-limiting rules
-------------------
  INFO / WARN:
    - Duplicate (same level + first 80 chars of message) suppressed for 5 min.
    - Burst guard: if more than 10 WARNs arrive within 60 s, subsequent messages
      are buffered.  A single digest is sent (at most once per 60 s) that
      summarises all buffered WARNs.
  CRITICAL:
    - Never rate-limited.  Every CRITICAL fires immediately as a separate post.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from datetime import datetime, timedelta
from typing import Optional


_LEVEL_NUM = {"INFO": 0, "WARN": 1, "CRITICAL": 2}

_DEDUP_WINDOW_S  = 300.0   # 5 min duplicate suppression
_BURST_WINDOW_S  = 60.0    # burst-detection window
_BURST_THRESHOLD = 10      # WARNs per burst window before digest kicks in
_DIGEST_COOLDOWN_S = 60.0  # min seconds between digest sends


class AlertManager:
    """
    Thread-safe alert poster with deduplication and burst grouping.

    Replace ``_post`` on instances in tests to capture outgoing calls without
    making real HTTP requests.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        min_level:   str = "WARN",
        info_opt_in: bool = False,
        logger=None,
    ):
        self._webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL", "")
        self._min_num     = _LEVEL_NUM.get(
            os.getenv("ALERT_LEVEL", min_level).upper(), 1
        )
        self._info_opt_in = info_opt_in or (
            os.getenv("ALERT_INFO_OPT_IN", "false").lower() == "true"
        )
        self._log = logger

        self._lock           = threading.Lock()
        # (level, msg_prefix) → last posted datetime
        self._recent:        dict[tuple, datetime] = {}
        # Timestamps of recent WARNs for burst detection
        self._warn_window:   list[datetime] = []
        # Buffered messages while burst is active
        self._digest_buffer: list[str] = []
        self._last_digest_ts: Optional[datetime] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def send_alert(self, level: str, message: str, **context) -> None:
        """Post an alert.  Thread-safe."""
        level_num = _LEVEL_NUM.get(level.upper(), 1)
        if level_num < self._min_num:
            return
        if level.upper() == "INFO" and not self._info_opt_in:
            return

        with self._lock:
            self._send_locked(level.upper(), message, context)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _send_locked(self, level: str, message: str, context: dict) -> None:
        now = datetime.now()

        # CRITICAL: always send, never rate-limit
        if level == "CRITICAL":
            self._post(level, message, context)
            return

        # WARN burst detection
        if level == "WARN":
            cutoff = now - timedelta(seconds=_BURST_WINDOW_S)
            self._warn_window = [t for t in self._warn_window if t > cutoff]
            self._warn_window.append(now)

            if len(self._warn_window) > _BURST_THRESHOLD:
                # We are in burst mode — buffer and send a digest
                self._digest_buffer.append(message)
                if (self._last_digest_ts is None or
                        (now - self._last_digest_ts).total_seconds()
                        >= _DIGEST_COOLDOWN_S):
                    self._flush_digest(now)
                return

        # Duplicate suppression (WARN + INFO)
        key = (level, message[:80])
        last = self._recent.get(key)
        if last is not None and (now - last).total_seconds() < _DEDUP_WINDOW_S:
            return

        self._recent[key] = now
        self._post(level, message, context)

    def _flush_digest(self, now: datetime) -> None:
        count  = len(self._digest_buffer)
        sample = self._digest_buffer[:5]
        digest = (
            f"[{count} WARN events in {int(_BURST_WINDOW_S)}s] "
            + "; ".join(sample)
            + (f" ... +{count - 5} more" if count > 5 else "")
        )
        self._digest_buffer.clear()
        self._last_digest_ts = now
        self._post("WARN", digest, {})

    def _post(self, level: str, message: str, context: dict) -> None:
        """Send payload to webhook.  Override in tests."""
        if not self._webhook_url:
            return
        payload = {"level": level, "message": message, **{
            k: str(v) for k, v in context.items()
        }}
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            if self._log:
                self._log.error("alert_post_failed", exc=str(exc))


def build_alert_manager_from_env(logger=None) -> AlertManager:
    return AlertManager(
        webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        min_level=os.getenv("ALERT_LEVEL", "WARN"),
        info_opt_in=os.getenv("ALERT_INFO_OPT_IN", "false").lower() == "true",
        logger=logger,
    )
