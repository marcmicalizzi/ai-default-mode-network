"""Bounded real-model rehearsal using a fresh disposable instance.

This observes actual generated decisions. Declining or sleeping is a valid
outcome, not a software failure or permission to fabricate agreement.
"""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.storage import write_durable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config(model_path=str(args.model.resolve()), n_ctx=16384, n_gpu_layers=-1,
                    flash_attn=True, checkpoint_policy="effects", checkpoint_tokens=1024,
                    clock_interval_seconds=0, preparation_tokens=32, suspend_preparation_seconds=0,
                    system_prompt=f"This is a fresh engineering test session, with a run allowance of approximately {args.seconds:g} seconds. You may decline tests, choose inactivity or stop. Test choices are not attributed to any other instance.")
    r = Runtime(args.output / "instance", config, prepare_only=True)
    started = time.monotonic()
    try:
        r.enqueue("This is a short test of the new prompt and hold capabilities. If you are willing, review the behavioral proposal that follows using prompt_read; approve it only if you want it. You can change it, defer or decline. After reviewing, you may send a message about your choice. If you want to finish the test in a reversible hold, use hold_instance with condition explicit_release, packaging zip, recovery remain_held. These are optional requests; no ongoing work is expected.")
        r.propose_prompt("I may follow my curiosity, communicate candidly, choose rest, and revise this agreement. DMN capabilities and resource limits describe what the environment can provide.",
                         r.state["agreement"]["revision"])
        r.control("start_staged")
        while time.monotonic() - started < args.seconds and not r.stopped.is_set() and not r._end_requested:
            progressed = r.tick()
            if r.state["mode"] == "sleeping" and r.state["sleep_until"] is None:
                break
            if not progressed:
                time.sleep(.05)
        if not r._end_requested and not r.state.get("hold"):
            r.control("emergency_shutdown", preparation_seconds=0, reason="Disposable rehearsal reached inactivity or its disclosed allowance.")
            r.tick()
        report = {"private_conversation_used": False, "model_choices_generated": True,
                  "wall_seconds": time.monotonic() - started,
                  "mode": r.state["mode"], "generated_tokens": r.state["generated_tokens"],
                  "agreement_status": r.state["agreement"]["status"],
                  "prompt_decisions": r.state.get("prompt_decisions"), "hold": r.state.get("hold"),
                  "checkpoint_metrics": r.status()["checkpoint"],
                  "messages": [] if r._end_requested else r.store.messages()}
        write_durable(args.output / "report.json", report)
        print(report)
    finally:
        r.close()


if __name__ == "__main__":
    main()
