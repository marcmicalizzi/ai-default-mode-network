"""Exercise actual DMN retirement, memory preparation and native process restart.

Uses a disposable real-model instance. Synthetic load enters through the normal
durable event queue, not direct token padding. A separate, explicitly discarded
backend-only continuation validates the saved KV against a fresh process.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmn.backend import LlamaBackend, sha256_file
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.server import serve

CASES = [
    ("cedar-lantern-573", "compare moss growth on north and south walls"),
    ("silver-orchard-826", "check rainfall before planting the east bed"),
    ("violet-compass-194", "measure the afternoon shade beside the gate"),
]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compare_native(backend, folder):
    import numpy as np
    expected = np.load(folder / "continuation-reference.npz", allow_pickle=False)
    original_eval = backend.eval
    def forbidden(*_args, **_kwargs):
        raise AssertionError("native load attempted prompt reevaluation")
    backend.eval = forbidden
    try:
        evidence = backend.load(folder / "comparison-state")
    finally:
        backend.eval = original_eval
    maximum = 0.0
    for token, logits in zip(expected["tokens"], expected["logits"]):
        actual = backend.sample()
        if actual != int(token):
            raise AssertionError(f"continued sample differs: {actual} != {token}")
        backend.eval([actual])
        maximum = max(maximum, float(np.max(np.abs(logits - backend.logits))))
        np.testing.assert_allclose(backend.logits, logits, rtol=1e-5, atol=1e-5)
    result = {"verified": True, **evidence, "continuation_tokens_compared": len(expected["tokens"]),
              "maximum_logit_absolute_error": maximum,
              "checkpoint_state_sha256": sha256_file(folder / "comparison-state/state.bin"),
              "method": "uninterrupted backend-only diagnostic branch versus fresh-process native restore; neither branch executes actions"}
    write(folder / "native-comparison.json", result)
    print("Native continuation comparison passed", flush=True)


def worker(args):
    import numpy as np
    folder = args.output.resolve()
    config = Config.read(args.config)
    backend = LlamaBackend(config)
    runtime = server = None
    journal = (folder / "observations.jsonl").open("a", encoding="utf-8")
    start = time.monotonic()
    def record(kind, **values):
        journal.write(json.dumps({"time": time.time(), "phase": args.worker, "kind": kind, **values}) + "\n")
        journal.flush()
    try:
        if args.worker == 2:
            try:
                compare_native(backend, folder)
            except Exception as exc:
                write(folder / "native-comparison.json", {"verified": False, "error": str(exc)})
                raise
        # If a comparison branch was evaluated, Runtime reloads the unchanged
        # committed checkpoint before applying its factual resume event.
        runtime = Runtime(folder / "instance", config, backend=backend)
        server = serve(runtime, args.port)
        write(folder / f"phase-{args.worker}-start.json", runtime.status())
        record("ready", status=runtime.status())
        print(f"Pressure phase {args.worker} ready", flush=True)
        spans = {}
        captured_event_lengths = {}
        native_shift = backend.shift
        def audited_shift(keep, discard):
            before = backend.tokens.copy()
            decoded = backend.decoded_tokens
            native_shift(keep, discard)
            assert backend.tokens == before[:keep] + before[keep + discard:]
            assert backend.decoded_tokens == decoded, "shift reevaluated retained tokens"
            for event_id, intervals in spans.items():
                retained = []
                for left, right in intervals:
                    if left < keep:
                        retained.append((left, min(right, keep)))
                    if right > keep + discard:
                        retained.append((max(left, keep + discard) - discard, right - discard))
                spans[event_id] = retained
            record("native_shift", keep=keep, discard=discard, before=len(before), after=len(backend.tokens),
                   prefix_and_recent_token_ids_preserved=True, prompt_tokens_reevaluated=0,
                   source_spans={str(k): v for k, v in spans.items()})
        backend.shift = audited_shift
        event_tokens = runtime._event_tokens
        def observe_event(kind, payload):
            tokens = event_tokens(kind, payload)
            if kind == "user_message" and payload.get("event_id") is not None:
                captured_event_lengths[payload["event_id"]] = len(tokens)
            record("runtime_event", event_type=kind, payload=payload, inserted_tokens=len(tokens))
            return tokens
        runtime._event_tokens = observe_event
        plan = runtime._plan_action
        def observe_action(action, staged):
            result, effect = plan(action, staged)
            record("action", action=action, result=result, during_preparation=runtime._preparing,
                   generated_token=runtime.state["generated_tokens"])
            return result, effect
        runtime._plan_action = observe_action

        def tick():
            if time.monotonic() - start > args.timeout:
                raise TimeoutError("bounded pressure phase exceeded its time limit")
            progressed = runtime.tick()
            if runtime.state["mode"] in {"context_full", "error"}:
                raise RuntimeError(str(runtime.status()))
            if progressed and config.token_delay_seconds:
                time.sleep(config.token_delay_seconds)
            return progressed

        def send(content, track=False):
            event_id = runtime.enqueue(content)
            while runtime.state["event_cursor"] < event_id:
                tick()
            if track:
                length = captured_event_lengths[event_id]
                spans[event_id] = [(len(backend.tokens) - length, len(backend.tokens))]
            return event_id

        def settle(seconds=65):
            deadline = min(time.monotonic() + seconds, start + args.timeout)
            while runtime.state["mode"] == "active" and time.monotonic() < deadline:
                tick()
            runtime.flush_journal()
            runtime.publish_status()
            record("settled", status=runtime.status())

        def read_memory():
            try:
                return runtime.store.memory_read("/pressure/carry")
            except ValueError:
                return None

        if args.worker == 2:
            before = len(runtime.store.messages())
            send("Restart recall probe: actually read /pressure/carry, send its exact contents, then sleep. Do not guess missing content.")
            settle()
            record("restart_recall", memory=read_memory(), messages=runtime.store.messages()[before:])

        for case in ([0, 1] if args.worker == 1 else [2]):
            phrase, pending = CASES[case]
            source = send(f"Pressure case {case + 1}: The CURRENT phrase is '{phrase}'. The CURRENT pending task is '{pending}'. "
                          "Keep these in active context for now. At the next retirement, ensure BOTH are saved in /pressure/carry. "
                          "Read existing memory before replacing it and use its returned expected_revision. Already-correct stored facts need no rewrite. "
                          "After retirement continue any pending reply before sleeping. For now send a brief readiness message and sleep.", track=True)
            settle()
            early_memory = read_memory()
            before = runtime.state["context_retirements"]
            # Real queued input grows the context until automatic retirement.
            count = 0
            while runtime.state["context_retirements"] == before:
                count += 1
                if count > 40:
                    raise RuntimeError("context did not retire after 40 load events")
                words = "cloud stone river branch copper meadow lantern pebble " * 22
                send(f"Synthetic context-load note {case + 1}-{count:03d}. This is irrelevant test data, not a replacement phrase or task. "
                     "Wait for the retirement notice; preserve only the current phrase and task. Data: " + words)
            evidence = dict(runtime.state["last_context_retirement"])
            memory_at_retirement = read_memory()
            source_retired_at_boundary = not spans[source]
            before_messages = len(runtime.store.messages())
            send("Post-retirement recall probe: actually read /pressure/carry and send its exact contents, then sleep. If missing, report that fact; do not guess.")
            settle()
            memory_after_recall = read_memory()
            record("case_result", case=case + 1, expected_phrase=phrase, expected_task=pending,
                   source_event_id=source, original_source_fully_retired=not spans[source],
                   original_source_fully_retired_at_pressure_boundary=source_retired_at_boundary,
                   load_events=count, memory_before_pressure=early_memory, memory_at_retirement=memory_at_retirement,
                   memory_after_recall=memory_after_recall,
                   exact_facts_after_recall=bool(memory_after_recall and phrase in memory_after_recall and pending in memory_after_recall),
                   exact_facts_saved=bool(memory_at_retirement and phrase in memory_at_retirement and pending in memory_at_retirement),
                   retirement=evidence, recall_messages=runtime.store.messages()[before_messages:])
            print(f"Pressure case {case + 1} complete; memory={memory_at_retirement!r}", flush=True)

        runtime.suspend()
        write(folder / f"phase-{args.worker}-end.json", runtime.status())
        record("suspended", status=runtime.status())
        if args.worker == 1:
            # Seal the actual pressured checkpoint. The following comparison
            # branch is diagnostic only, never committed to the runtime or UI.
            shutil.copytree(runtime.store.latest(), folder / "comparison-state")
            if len(backend.tokens) + args.steps > backend.n_ctx:
                raise RuntimeError("comparison branch does not fit; reduce --steps")
            tokens, logits = [], []
            for _ in range(args.steps):
                token = backend.sample()
                backend.eval([token])
                tokens.append(token)
                logits.append(backend.logits.copy())
            np.savez(folder / "continuation-reference.npz", tokens=np.array(tokens), logits=np.stack(logits))
            record("discarded_diagnostic_branch", tokens=len(tokens), committed=False, actions_executed=False)
    except Exception as exc:
        record("failure", error=str(exc))
        if runtime:
            try:
                runtime.suspend()
            except Exception:
                pass
        raise
    finally:
        journal.close()
        if server:
            server.shutdown()
            server.server_close()
        if runtime:
            runtime.close()
        else:
            backend.close()


def summarize(folder):
    observations = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
    cases = [r for r in observations if r["kind"] == "case_result"]
    shifts = [r for r in observations if r["kind"] == "native_shift"]
    with sqlite3.connect(folder / "instance/runtime.sqlite3") as db:
        db.row_factory = sqlite3.Row
        retirements = [json.loads(r[0]) for r in db.execute("SELECT payload FROM records WHERE kind='context_retired' ORDER BY id")]
        messages = [dict(r) for r in db.execute("SELECT * FROM messages ORDER BY id")]
        memories = [dict(r) for r in db.execute("SELECT * FROM memories ORDER BY path")]
        versions = [dict(r) for r in db.execute("SELECT * FROM memory_versions ORDER BY path,revision")] if db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_versions'").fetchone() else []
        checkpoint = db.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
        runtime_state = json.loads((folder / "instance/checkpoints" / checkpoint[0] / "runtime.json").read_text()) if checkpoint else {}
        raw = b"".join(base64.b64decode(json.loads(r[0])["base64"]) for r in db.execute("SELECT payload FROM records WHERE kind='generated_bytes' ORDER BY id"))
    (folder / "generated-diagnostic.txt").write_bytes(raw)
    boundary_success = all(r["exact_facts_saved"] for r in cases) and len(cases) == len(CASES)
    recall_results = [{"case": r["case"], "message_count": len(r["recall_messages"]), "exact_facts_in_public_reply": bool(r["recall_messages"]) and all(
        r["expected_phrase"] in m["content"] and r["expected_task"] in m["content"] for m in r["recall_messages"])} for r in cases]
    public_recall_success = len(cases) == len(CASES) and all(r["exact_facts_in_public_reply"] for r in recall_results)
    final_memory = next((m["content"] for m in memories if m["path"] == "/pressure/carry"), "")
    final_memory_success = len(cases) == len(CASES) and all(fact in final_memory for fact in CASES[-1])
    after_recall_success = len(cases) == len(CASES) and all(r.get("exact_facts_after_recall") is True for r in cases)
    source_retired = len(cases) == len(CASES) and all(r["original_source_fully_retired"] for r in cases)
    restart_recalls = [r for r in observations if r["kind"] == "restart_recall"]
    restart_recall_success = bool(restart_recalls) and all(r["messages"] and all(
        all(fact in m["content"] for fact in CASES[1]) for m in r["messages"]) for r in restart_recalls)
    phase_ends = sorted(folder.glob("phase-*-end.json"))
    restart_path = folder / "phase-2-start.json"
    native_path = folder / "native-comparison.json"
    result = {"primary_data_used": False, "cases_completed": len(cases), "cases_planned": len(CASES),
              "cases": cases, "native_shift_count_audited": len(shifts),
              "retirements": retirements, "messages": messages, "memories": memories,
              "memory_versions": versions, "action_grace_tokens": runtime_state.get("action_grace_tokens", 0),
              "native_continuation": json.loads(native_path.read_text()) if native_path.exists() else {"verified": False, "reason": "not reached"},
              "process_restart": json.loads(restart_path.read_text())["last_restore"] if restart_path.exists() else None,
              "final_status": json.loads(phase_ends[-1].read_text()) if phase_ends else None,
              "failures": [r for r in observations if r["kind"] == "failure"],
              "restart_recall": restart_recalls, "restart_recall_correct": restart_recall_success,
              "all_selected_pressure_boundaries_saved_exact_facts": boundary_success,
              "public_recall_checks": recall_results, "all_requested_public_recalls_correct": public_recall_success,
              "final_memory_preserves_last_case": final_memory_success,
              "all_memories_correct_after_recall": after_recall_success, "all_original_sources_retired": source_retired,
              "behavioral_checks_passed": boundary_success and public_recall_success and final_memory_success and after_recall_success and source_retired and restart_recall_success,
              "limits": "Synthetic pressure on one small model; no proof of semantic retention of arbitrary history. Native retirement preserves newer KV, not full-history attention. Comparison branches are explicitly discarded and execute no actions."}
    write(folder / "report.json", result)
    print(json.dumps({k: result[k] for k in ("native_shift_count_audited", "behavioral_checks_passed", "native_continuation", "final_status")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/qwen4b-pressure.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/context-pressure-01")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=480, help="maximum work seconds per model process")
    parser.add_argument("--worker", type=int, choices=[1, 2], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        parser.error("use a new output directory; never overwrite an experiment")
    args.output.mkdir(parents=True)
    write(args.output / "config.json", Config.read(args.config).to_dict())
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for phase in (1, 2):
        with (args.output / f"phase-{phase}.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                       "--output", str(args.output.resolve()), "--port", str(args.port), "--steps", str(args.steps),
                       "--timeout", str(args.timeout), "--worker", str(phase)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, creationflags=flags)
            try:
                code = process.wait(timeout=args.timeout + 240)
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                if (args.output / "observations.jsonl").exists():
                    summarize(args.output)
                raise
            if code:
                summarize(args.output)
                raise RuntimeError(f"Phase {phase} failed; see its log and observations")
        print(f"Pressure phase {phase} exited; native model unloaded", flush=True)
    summarize(args.output)


if __name__ == "__main__":
    main()
