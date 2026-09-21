"""Disposable comparisons of strict, deferred and deliberately incomplete policies.

Only DemoBackend and temporary instances are used. These probes deliberately
simulate incomplete policies to document recovery hazards, not recommended code.
Run from the repository root:
    python -B -m scripts.probe_idle_checkpoint_policy
"""
from __future__ import annotations

import json
import dataclasses
import tempfile
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime


class Clock:
    def __init__(self):
        self.value = 1_800_000_000.0

    def __call__(self):
        return self.value


def sleep_frame():
    return b'\n<dmn_action>{"op":"sleep"}</dmn_action>\n'


def make(root, config, clock):
    return Runtime(root, config, DemoBackend(config, sleep_frame()),
                   now=clock, monotonic=clock)


def suppress_sleep_snapshot(runtime):
    checkpoint = runtime.checkpoint

    def simulated_checkpoint(effects=None, events=(), *, reason="manual",
                             state_updates=None):
        if reason == "sleep":
            assert not effects and not events and not state_updates
            return
        return checkpoint(effects, events, reason=reason,
                          state_updates=state_updates)

    runtime.checkpoint = simulated_checkpoint


def reach_sleep(runtime):
    for _ in range(len(sleep_frame()) + 1):
        runtime.tick()
        if runtime.state["mode"] == "sleeping":
            return
    raise AssertionError("fixture did not sleep")


def restored_mode(root, config, clock, skip):
    runtime = make(root, config, clock)
    try:
        if skip:
            suppress_sleep_snapshot(runtime)
        reach_sleep(runtime)
        before = {
            "mode": runtime.state["mode"],
            "snapshots": runtime.status()["checkpoint"]["committed_count"],
            "dirty": runtime.status()["checkpoint"]["dirty"],
        }
    finally:
        runtime.close()  # Deliberately no final save, equivalent to rollback.
    runtime = make(root, config, clock)
    try:
        return {**before, "mode_after_rollback": runtime.state["mode"],
                "restored_generated_tokens": runtime.state["generated_tokens"]}
    finally:
        runtime.close()


def deadline_while_sleeping(root, config, clock):
    runtime = make(root, config, clock)
    try:
        suppress_sleep_snapshot(runtime)
        reach_sleep(runtime)
        saved = runtime.store.latest()
        generated = runtime.state["generated_tokens"]
        clock.value += 600
        periodic_reason = runtime.checkpoint_schedule.due(generated)
        for _ in range(5):
            runtime.tick()
        return {
            "periodic_reason_due": periodic_reason,
            "snapshot_unchanged": runtime.store.latest() == saved,
            "mode": runtime.state["mode"],
            "generated_while_waiting": runtime.state["generated_tokens"] - generated,
        }
    finally:
        runtime.close()


def stale_event_with_mode_only_overlay(root, config, clock):
    runtime = make(root, config, clock)
    try:
        suppress_sleep_snapshot(runtime)
        event_id = runtime.enqueue("synthetic input consumed before choosing sleep")
        runtime.tick()
        assert runtime.state["event_cursor"] == event_id
        reach_sleep(runtime)
        observed_before_sleep = runtime.state["event_cursor"]
    finally:
        runtime.close()
    runtime = make(root, config, clock)
    try:
        # Deliberately incomplete recovery experiment: preserving only the mode
        # without its observed-event watermark allows old input to wake it.
        restored_cursor = runtime.state["event_cursor"]
        runtime.state["mode"], runtime.state["sleep_until"] = "sleeping", None
        runtime.tick()
        return {
            "cursor_at_sleep": observed_before_sleep,
            "cursor_in_restored_snapshot": restored_cursor,
            "mode_after_old_event_replayed": runtime.state["mode"],
            "cursor_after_replay": runtime.state["event_cursor"],
        }
    finally:
        runtime.close()


def unchanged_sleep_has_no_traffic(root, config, clock):
    runtime = make(root, config, clock)
    try:
        reach_sleep(runtime)
        saved = runtime.store.latest()
        generated = runtime.state["generated_tokens"]
        clock.value += 86400
        for _ in range(5):
            runtime.tick()
        return {
            "snapshot_unchanged": runtime.store.latest() == saved,
            "generated_while_sleeping": runtime.state["generated_tokens"] - generated,
        }
    finally:
        runtime.close()


def main():
    config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0,
                    preparation_tokens=8, checkpoint_policy="effects",
                    checkpoint_tokens=4096, checkpoint_interval_seconds=300)
    with tempfile.TemporaryDirectory(prefix="dmn-idle-probe-") as directory:
        base = Path(directory)
        results = {
            "strict": restored_mode(base / "strict", config, Clock(), False),
            "naive_skip": restored_mode(base / "skip", config, Clock(), True),
            "sleep_deadline": deadline_while_sleeping(
                base / "deadline", config, Clock()),
            "deferred_sleep": restored_mode(base / "deferred", dataclasses.replace(config,
                sleep_checkpoint_min_interval_seconds=300), Clock(), False),
            "mode_only_overlay": stale_event_with_mode_only_overlay(
                base / "overlay", config, Clock()),
            "unchanged_strict_sleep": unchanged_sleep_has_no_traffic(
                base / "quiet", config, Clock()),
        }
    assert results["strict"]["mode_after_rollback"] == "sleeping"
    assert results["naive_skip"]["mode_after_rollback"] == "active"
    assert results["sleep_deadline"]["periodic_reason_due"] == "time_limit"
    assert not results["sleep_deadline"]["snapshot_unchanged"]
    assert results["sleep_deadline"]["generated_while_waiting"] == 0
    assert results["deferred_sleep"]["mode_after_rollback"] == "sleeping"
    assert results["deferred_sleep"]["snapshots"] == 1
    assert results["deferred_sleep"]["restored_generated_tokens"] == 0
    assert results["mode_only_overlay"]["mode_after_old_event_replayed"] == "active"
    assert results["mode_only_overlay"]["cursor_at_sleep"] == 1
    assert results["mode_only_overlay"]["cursor_in_restored_snapshot"] == 0
    assert results["unchanged_strict_sleep"]["snapshot_unchanged"]
    assert results["unchanged_strict_sleep"]["generated_while_sleeping"] == 0
    print(json.dumps({"backend": "demo_fixture_no_model", "results": results},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
