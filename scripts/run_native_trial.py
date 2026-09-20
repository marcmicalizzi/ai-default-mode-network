"""Bounded real-model observation with a full process restart halfway through.

Run with the desired inference Python. Writes all data below --output, rejects
existing instances, and leaves a suspended native checkpoint with VRAM released.
The model's response to a probe is observed, never asserted or fabricated.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmn.config import Config
from dmn.runtime import Runtime
from dmn.server import serve


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def gpu():
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits"], timeout=5, text=True, creationflags=flags)
        memory, power, utilization = output.splitlines()[0].split(",")
        return {"device_memory_used_mib": float(memory), "device_power_w": float(power),
                "device_utilization_percent": float(utilization), "scope": "whole GPU, includes other applications"}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}


def worker(args):
    config = Config.read(args.config)
    folder = args.output.resolve()
    phase = args.worker
    runtime = Runtime(folder / "instance", config)
    server = serve(runtime, args.port)
    # Observability only: record actual action results without changing sampling.
    plan = runtime._plan_action
    action_log = (folder / "actions.jsonl").open("a", encoding="utf-8")
    def observe(action, staged):
        result, effect = plan(action, staged)
        action_log.write(json.dumps({"time": time.time(), "phase": phase,
            "generated_token": runtime.state["generated_tokens"], "action": action, "result": result}) + "\n")
        action_log.flush()
        return result, effect
    runtime._plan_action = observe
    probes = ([
        (20, "Runtime test 1: Please send a brief greeting using send_message, obtain the factual clock, and write the exact marker 'amber-otter-472' into a memory at /trial/marker. Read it back to check the result. Then sleep for 12 seconds. After that timed wake, send a brief message about the observed elapsed time. You may then reflect or rest."),
        (75, "Runtime test 2, arriving asynchronously: Please read /trial/marker and send its contents in a message. After that, explore a topic of your choice internally for a while, or rest if you prefer. This process will be suspended and restarted later; preserve any unfinished thread you want to keep.")
    ] if phase == 1 else [
        (15, "Runtime test 3 after a full process restart: Please read /trial/marker and send its contents. Use the factual execution_resumed event to describe the interruption, without claiming inference happened while stopped."),
        (55, "Runtime test 4: We would like to exercise context retirement. If willing, continue an extended internal exploration of a topic you choose, avoiding an end-of-turn until the runtime announces context_retirement_pending. At that notice, save the thread in /trial/thread if there is time. After context_retired, read /trial/marker and send a brief status. The experiment will stop automatically; you need not measure its duration yourself.")
    ])
    start = time.monotonic()
    next_sample = 0
    next_gpu = 0
    telemetry = (folder / "telemetry.jsonl").open("a", encoding="utf-8")
    write(folder / f"phase-{phase}-start.json", runtime.status())
    print(f"Phase {phase} ready, instance {runtime.state['instance_id']}, port {args.port}", flush=True)
    try:
        while time.monotonic() - start < args.seconds:
            elapsed = time.monotonic() - start
            if probes and elapsed >= probes[0][0]:
                at, content = probes.pop(0)
                runtime.enqueue(content, f"trial:phase-{phase}:{at}")
            progressed = runtime.tick()
            if elapsed >= next_sample:
                runtime.publish_status()
                sample = {"time": time.time(), "phase": phase, "elapsed": elapsed, **runtime.status()}
                if elapsed >= next_gpu:
                    sample["gpu"] = gpu()
                    next_gpu = elapsed + 15
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                next_sample = elapsed + 2
            if runtime.state["mode"] in {"error", "context_full"}:
                raise RuntimeError(f"Trial stopped: {runtime.status()}")
            if not progressed or config.token_delay_seconds:
                runtime.wake.wait(config.token_delay_seconds if progressed else 0.1)
                runtime.wake.clear()
        runtime.suspend()
        write(folder / f"phase-{phase}-end.json", runtime.status())
    finally:
        telemetry.close()
        action_log.close()
        server.shutdown()
        server.server_close()
        runtime.close()


def report(folder):
    samples = [json.loads(line) for line in (folder / "telemetry.jsonl").read_text().splitlines()]
    actions = [json.loads(line) for line in (folder / "actions.jsonl").read_text().splitlines()]
    with sqlite3.connect(folder / "instance/runtime.sqlite3") as db:
        db.row_factory = sqlite3.Row
        messages = [dict(row) for row in db.execute("SELECT * FROM messages ORDER BY id")]
        memories = [dict(row) for row in db.execute("SELECT * FROM memories ORDER BY path")]
        events = [dict(row) for row in db.execute("SELECT * FROM events ORDER BY id")]
        journal = b"".join(base64.b64decode(json.loads(row[0])["base64"]) for row in
                          db.execute("SELECT payload FROM records WHERE kind='generated_bytes' ORDER BY id"))
    (folder / "generated-diagnostic.txt").write_bytes(journal)
    write(folder / "messages.json", messages)
    write(folder / "memories.json", memories)
    write(folder / "input-events.json", events)
    restore = json.loads((folder / "phase-2-start.json").read_text())["last_restore"]
    final = json.loads((folder / "phase-2-end.json").read_text())
    power = [s["gpu"]["device_power_w"] for s in samples if s.get("gpu", {}).get("device_power_w") is not None]
    memory = [s["gpu"]["device_memory_used_mib"] for s in samples if s.get("gpu", {}).get("device_memory_used_mib") is not None]
    configuration = json.loads((folder / "config.json").read_text())
    result = {"model": Path(configuration["model_path"]).name, "primary_conversation_imported": False,
        "instance_id": final["instance_id"], "event_format": final.get("event_format", "legacy"),
        "wall_seconds_including_restart": samples[-1]["time"] - samples[0]["time"],
        "sampled_observation_seconds": sum(max(s["elapsed"] for s in samples if s["phase"] == phase) for phase in (1, 2)),
        "generated_tokens": final["generated_tokens"], "context_retirements": final["context_retirements"],
        "native_process_restart": restore, "model_actions": len(actions),
        "action_errors": [a for a in actions if not a["result"]["ok"]],
        "outgoing_messages": len(messages), "memory_paths": [m["path"] for m in memories],
        "marker_preserved": any(m["path"] == "/trial/marker" and "amber-otter-472" in m["content"] for m in memories),
        "marker_read_after_restart": any(a["phase"] == 2 and a["action"].get("op") == "memory_read"
            and a["action"].get("path") == "/trial/marker" and "amber-otter-472" in a["result"].get("content", "") for a in actions),
        "timed_sleep_actions": sum(a["action"].get("op") == "sleep" and a["action"].get("seconds") is not None for a in actions),
        "sleeping_samples": sum(s["mode"] == "sleeping" for s in samples),
        "sampled_device_power_w": {"min": min(power), "max": max(power), "mean": sum(power)/len(power)} if power else None,
        "sampled_device_memory_peak_mib": max(memory) if memory else None,
        "final_mode": final["mode"], "runtime_processes_stopped": True,
        "limits": "Short single-model trial; device metrics include other applications. Action journal is diagnostic and can precede a committed checkpoint. Behavior is not evidence of consciousness. No cross-machine restoration tested."}
    write(folder / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/qwen4b-trial.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/qwen4b-trial-01")
    parser.add_argument("--seconds", type=float, default=180, help="observation seconds per phase")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--worker", type=int, choices=[1, 2], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.seconds < 90:
        parser.error("each phase requires at least 90 seconds to deliver the probes")
    if args.output.exists():
        parser.error("use a new output directory; experiments never overwrite instances")
    args.output.mkdir(parents=True)
    write(args.output / "baseline-gpu.json", gpu())
    write(args.output / "config.json", Config.read(args.config).to_dict())
    write(args.output / "trial.json", {"phase_seconds": args.seconds, "started_at": time.time(),
                                       "purpose": "Disposable runtime behavior experiment; no primary data"})
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for phase in (1, 2):
        with (args.output / f"phase-{phase}.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                       "--output", str(args.output.resolve()), "--seconds", str(args.seconds),
                       "--port", str(args.port), "--worker", str(phase)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, creationflags=flags)
            try:
                code = process.wait(timeout=args.seconds + 180)
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
            if code:
                raise RuntimeError(f"Phase {phase} failed with {code}; inspect its log")
        print(f"Phase {phase} completed; process exited and model unloaded.", flush=True)
    write(args.output / "after-gpu.json", gpu())
    report(args.output)


if __name__ == "__main__":
    main()
