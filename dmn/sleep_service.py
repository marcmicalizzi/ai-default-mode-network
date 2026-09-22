"""Stable transport/queue ownership across supervised sleep cycles.

The service never supplies approval, retries uncertain training, or releases a
hold. Native inference is closed before the transition begins. Real-instance
execution requires an explicit, validated host recipe and model-reviewed plan.
"""
from __future__ import annotations

import threading

from .config import Config
from .deep_sleep import run_fixture_sleep, fixture_guard
from .runtime import Runtime


class SleepService:
    def __init__(self, runtime, *, transition=run_fixture_sleep, runtime_factory=Runtime):
        if runtime.sleep_offer is None:
            fixture_guard(runtime.config)
        if not runtime.sleep_test_mode and runtime.sleep_offer is None:
            raise ValueError('continuous sleep integration requires the disposable test gate')
        self.current = runtime
        self.transition = transition
        self.runtime_factory = runtime_factory
        self.stopped = threading.Event()
        self._cycle_phase = None

    def __getattr__(self, name):
        # Servers hold this stable object. The shared Store and control lock
        # survive backend replacement; old request handlers can finish safely.
        return getattr(self.current, name)

    def status(self):
        value = self.current.status()
        if self._cycle_phase:
            value = {**value, 'sleep_service': {'phase': self._cycle_phase, 'text_queue_available': True}}
        return value

    def request_shutdown_from_signal(self):
        self.current.request_shutdown_from_signal()

    def run(self):
        try:
            while True:
                old = self.current
                old.run()
                if old.state['mode'] != 'deep_sleep' or old._end_requested or old.state.get('hold'):
                    return
                with old._control_lock:
                    old.backend.close()
                    old.backend = None
                self._cycle_phase = 'training_or_rebuilding'
                result = self.transition(old.root, old.state['sleep_run_id'], _session=old,
                    **({'_live_offer': old.sleep_offer} if old.sleep_offer else {}))
                if result['phase'] != 'WakeCommitted':
                    self._cycle_phase = result['phase']
                    return
                self._cycle_phase = 'restoring_checkpoint'
                import json
                saved = old.store.latest()
                manifest = json.loads((saved / 'manifest.json').read_text())
                config = Config(**manifest['fingerprint']['config'])
                # Do not reuse the pre-sleep adapter configuration. Only the
                # atomically selected checkpoint determines the waking weights.
                fresh = self.runtime_factory(old.root, config, sleep_test_mode=old.sleep_test_mode,
                                              sleep_offer=old.sleep_offer, _session=old)
                with old._control_lock:
                    fresh._shutdown_signal_pending |= old._shutdown_signal_pending
                    self.current = fresh
                self._cycle_phase = None
                if fresh.sleep_offer:
                    self.offer_recipe()
        finally:
            self.stopped.set()

    def close(self):
        self.stopped.set()
        self.current.close()

    def offer_recipe(self):
        from .sleep_host import offer_for_runtime
        self.current.offer_learning_recipe(offer_for_runtime(self.current, self.current.sleep_offer))
