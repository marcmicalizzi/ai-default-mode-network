"""Run a disposable two-participant scenario. No model, HTTP server or GPU.

From the worktree root: python scripts/verify_multi_user.py [--output report.json]
All instance state lives in a fresh temporary directory and is removed on exit.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime


def frame(**action):
    return ("\n<dmn_action>" + json.dumps(action) + "</dmn_action>\n").encode()


def run():
    config = Config(backend="demo", n_ctx=32768, multi_user=True, require_contact_consent=False,
                    operator_participant_id="operator", inbox_generation_tokens=4,
                    clock_interval_seconds=0, checkpoint_policy="effects")
    script = (frame(op="send_message", conversation_id="operator-chat", content="Complete message for the operator.") +
              frame(op="send_message", conversation_id="guest-chat", content="Separate message for the guest.") +
              frame(op="block_participant", participant_id="guest") + frame(op="sleep"))
    with tempfile.TemporaryDirectory(prefix="dmn-multi-user-") as folder:
        root = Path(folder) / "instance"
        runtime = Runtime(root, config, DemoBackend(config, script))
        try:
            runtime.register_conversation("operator", "Operator", "operator-chat")
            runtime.register_conversation("guest", "Guest", "guest-chat")
            operator_event = runtime.enqueue_conversation("operator-chat", "Hello", "operator:1")
            guest_event = None
            for _ in range(2500):
                if runtime.parser.pending and guest_event is None:
                    guest_event = runtime.enqueue_conversation("guest-chat", "Arrived during composition", "guest:1")
                if guest_event is not None and not runtime.store.messages():
                    assert not runtime.event_delivered(guest_event)
                runtime.tick()
                if runtime.state["mode"] == "sleeping":
                    break
            assert runtime.state["mode"] == "sleeping"
            assert runtime.event_delivered(operator_event) and runtime.event_delivered(guest_event)
            assert runtime.conversations.participant("guest")["blocked"]
            outputs = runtime.store.messages()
            assert [m["conversation_id"] for m in outputs] == ["operator-chat", "guest-chat"]
            assert not runtime.state.get("action_diagnostics", {}).get("interrupted_frames", 0)
            try:
                runtime.enqueue_conversation("guest-chat", "Rejected while blocked")
            except ValueError as exc:
                assert "blocked" in str(exc)
            else:
                raise AssertionError("blocked input was accepted")
            request_id = runtime.request_unblock("guest", 1, "Please reconsider this block.")
            assert runtime.conversations.participant("guest")["blocked"]
            instance_id = runtime.state["instance_id"]
        finally:
            runtime.close()
        restored = Runtime(root, config, DemoBackend(config, script))
        try:
            assert restored.state["instance_id"] == instance_id
            assert restored.store.messages() == outputs
            assert restored.conversations.participant("guest")["blocked"]
            assert restored.conversations.next_event(restored.state["event_cursor"])["id"] == request_id
        finally:
            restored.close()
    return {"fixture": "scripted_no_model", "gpu_used": False, "live_instance_accessed": False,
            "addressed_outputs": len(outputs), "incoming_event_deferred_during_send": True,
            "operator_identity_explicit": True, "block_enforced": True,
            "unblock_request_did_not_unblock": True, "restart_preserved_outputs_and_block": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
