"""Process-wide native diagnostics; deliberately separate from inference settings."""
from __future__ import annotations

import os
import sys
import threading


LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
_SEVERITY = {0: 50, 1: 20, 2: 30, 3: 40, 4: 10}
_callback = None  # ctypes callbacks must remain alive while native code holds them.


class NativeLogFilter:
    def __init__(self, level):
        if level not in LEVELS:
            raise ValueError("DMN_NATIVE_LOG_LEVEL must be debug, info, warning or error")
        self.threshold = LEVELS[level]
        self.local = threading.local()

    def __call__(self, level, text, _user_data):
        # CONT belongs to the previous message on this thread, even when that
        # message was filtered. Native enum values are not severity ordered.
        severity = (getattr(self.local, "severity", 30) if level == 5
                    else _SEVERITY.get(level, 30))
        self.local.severity = severity
        if severity < self.threshold or not text:
            return
        try:
            sys.stderr.write(text.decode("utf-8", "replace"))
            sys.stderr.flush()
        except (OSError, ValueError):
            # A closed terminal must not throw through a C callback.
            pass


def configure_native_logging(api):
    global _callback
    handler = NativeLogFilter(os.environ.get("DMN_NATIVE_LOG_LEVEL", "warning"))
    callback = api.llama_log_callback(handler)
    api.llama_log_set(callback, None)
    _callback = callback
