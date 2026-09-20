"""Scheduling policy only; the runtime remains the sole owner of native state."""
from __future__ import annotations


class CheckpointSchedule:
    def __init__(self, config, monotonic):
        self.config, self.monotonic = config, monotonic
        self.saved_generated = 0
        self.completed_at = None
        self.captured_at = None
        self.dirty_since = None

    def changed(self):
        if self.dirty_since is None:
            self.dirty_since = self.monotonic()

    def committed(self, generated, captured_at):
        self.saved_generated = generated
        self.captured_at = captured_at
        self.completed_at = self.monotonic()
        self.dirty_since = None

    def due(self, generated):
        if self.dirty_since is None:
            return None
        if self.config.checkpoint_tokens and generated - self.saved_generated >= self.config.checkpoint_tokens:
            return "token_limit"
        if (self.config.checkpoint_interval_seconds and self.completed_at is not None and
                self.monotonic() - self.completed_at >= self.config.checkpoint_interval_seconds):
            return "time_limit"
        return None

    def status(self, generated):
        now = self.monotonic()
        return {"policy": self.config.checkpoint_policy,
                "interval_seconds": self.config.checkpoint_interval_seconds,
                "token_limit": self.config.checkpoint_tokens,
                "dirty": self.dirty_since is not None,
                "unsaved_generated_tokens": max(0, generated - self.saved_generated),
                "unsaved_seconds": None if self.dirty_since is None else max(0, now - self.dirty_since),
                "age_seconds": None if self.captured_at is None else max(0, now - self.captured_at)}
