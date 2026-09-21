"""Idle pacing policy. No backend access, worker thread, or accumulated credit."""
from __future__ import annotations

import math


def requested_profile(action, config):
    if not config.idle_enabled:
        raise ValueError("idle activity is not enabled by the host")
    mode = action.get("mode")
    if mode == "focus":
        if set(action) - {"op", "mode"}:
            raise ValueError("focus takes no pacing arguments")
        return {"mode": "focus"}
    if mode != "idle" or set(action) - {"op", "mode", "burst_tokens", "interval_seconds"}:
        raise ValueError("activity requires mode focus or idle and supported pacing fields")
    burst = action.get("burst_tokens", config.idle_max_burst_tokens)
    interval = action.get("interval_seconds", config.idle_min_interval_seconds)
    if type(burst) is not int or not 1 <= burst <= config.idle_max_burst_tokens:
        raise ValueError(f"burst_tokens must be an integer from 1 to {config.idle_max_burst_tokens}")
    if (isinstance(interval, bool) or not isinstance(interval, (int, float)) or
            not math.isfinite(interval) or interval < config.idle_min_interval_seconds):
        raise ValueError(f"interval_seconds must be finite and at least {config.idle_min_interval_seconds}")
    return {"mode": "idle", "burst_tokens": burst, "interval_seconds": interval}


class ActivityPacer:
    def __init__(self, monotonic):
        self.monotonic = monotonic
        self.profile = {"mode": "focus"}
        self.remaining = 0
        self.next_at = None

    def select(self, profile, *, restored=False):
        self.profile = dict(profile or {"mode": "focus"})
        self.remaining = 0
        self.next_at = None
        if self.profile["mode"] == "idle":
            # Restart grants no burst credit. A fresh deliberate choice does.
            self.remaining = 0 if restored else self.profile["burst_tokens"]
            self.next_at = self.monotonic() + self.profile["interval_seconds"] if restored else None

    def ready(self):
        if self.profile["mode"] == "focus":
            return True
        if not self.remaining and self.monotonic() >= self.next_at:
            self.remaining = self.profile["burst_tokens"]
            self.next_at = None
        return self.remaining > 0

    def consumed(self):
        if self.profile["mode"] == "idle":
            self.remaining = max(0, self.remaining - 1)
            if not self.remaining:
                self.next_at = self.monotonic() + self.profile["interval_seconds"]

    def status(self):
        return {**self.profile, "remaining_tokens": self.remaining,
                "next_in_seconds": None if self.next_at is None else max(0, self.next_at - self.monotonic())}
