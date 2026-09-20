"""Real native prefill and separate-process restoration of a synthetic import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.backend import LlamaBackend
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.storage import write_durable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="explicit runtime placement override; bundle sampler still validated")
    parser.add_argument("--retire", action="store_true", help="exercise the protected import boundary and compare native continuation")
    parser.add_argument("--behavior-seconds", type=float, default=900,
                        help="wall-time budget for slow inference and full checkpoints")
    parser.add_argument("--resume-behavior", action="store_true",
                        help="resume the existing probe without submitting its input again")
    parser.add_argument("--stage", choices=["prefill", "compare", "restore", "behavior"])
    args = parser.parse_args()
    stages = ("prefill", "compare", "restore", "behavior") if args.retire else ("prefill", "restore")
    if not args.stage:
        args.output.mkdir(parents=True, exist_ok=False)
        for stage in stages:
            with (args.output / (stage + ".log")).open("w") as log:
                command = [sys.executable, str(Path(__file__).resolve()), "--bundle", str(args.bundle.resolve()),
                    "--output", str(args.output.resolve()), "--stage", stage]
                if args.config:
                    command += ["--config", str(args.config.resolve())]
                if args.retire:
                    command.append("--retire")
                command += ["--behavior-seconds", str(args.behavior_seconds)]
                result = subprocess.run(command, stdout=log, stderr=log, timeout=1800,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                raise RuntimeError(f"{stage} failed; inspect logs")
        write_durable(args.output / "report.json", {"verified": True, "primary_data_used": False,
            "stages": [json.loads((args.output / (s + ".json")).read_text()) for s in stages]})
        return
    config = Config.read(args.config or args.bundle / "config.json")
    source = json.loads((args.bundle / "tokens.json").read_text())
    started = time.monotonic()
    backend = LlamaBackend(config)
    if args.stage == "compare":
        import numpy as np
        from dmn.storage import Store
        store = Store(args.output / "instance")
        checkpoint = store.latest()
        store.close()
        try:
            evaluate = backend.eval
            backend.eval = lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("comparison restore replayed tokens"))
            restored = backend.load(checkpoint)
            backend.eval = evaluate
            reference = np.load(args.output / "reference.npz", allow_pickle=False)
            maximum = 0.0
            for token, logits in zip(reference["tokens"], reference["logits"]):
                assert backend.sample() == int(token)
                backend.eval([int(token)])
                maximum = max(maximum, float(np.max(np.abs(logits - backend.logits))))
            assert maximum <= 1e-5
            write_durable(args.output / "compare.json", {"stage": "compare", "restore": restored,
                "continuation_steps": len(reference["tokens"]), "maximum_logit_absolute_error": maximum,
                "actions_executed": False, "wall_seconds": time.monotonic() - started})
        finally:
            backend.close()
        return
    calls = []
    evaluate = backend.eval
    def observe(tokens):
        calls.append(list(tokens))
        return evaluate(tokens)
    backend.eval = observe
    runtime = Runtime(args.output / "instance", config, backend,
                      initial_context=args.bundle if args.stage == "prefill" else None)
    try:
        if not args.resume_behavior:
            assert runtime.store.messages() == []
            assert runtime.state["generated_tokens"] == 0
        if args.stage == "prefill":
            assert backend.tokens[:len(source)] == source
            assert calls[0] == source
            assert len(calls) == 2, "only source prefill and explicit transition should evaluate"
            if args.retire:
                span = runtime.state["protected_protocol"].copy()
                contract = backend.tokens[span["start"]:span["end"]]
                keep = runtime.state["keep_prefix"]
                runtime.state["mode"] = "sleeping"
                padding = backend.tokenize("Disposable import-boundary diagnostic text. ")
                needed = config.n_ctx - config.turnover_reserve - len(backend.tokens)
                assert needed >= 0
                runtime._eval((padding * (needed // len(padding) + 1))[:needed])
                runtime._ensure_space(512)
                assert "protected_protocol" not in runtime.state
                assert backend.tokens[keep:keep+len(contract)] == contract
                assert runtime.state["last_context_retirement"]["additional_ranges"]
                runtime.checkpoint()
            write_durable(args.output / "expected-tokens.json", backend.tokens)
        else:
            assert runtime.state["last_restore"]["prompt_tokens_reevaluated"] == 0
            if args.stage == "restore":
                expected = json.loads((args.output / "expected-tokens.json").read_text())
                assert backend.tokens[:len(expected)] == expected
                assert sum(map(len, calls)) < len(source), "resume must not replay history"
        behavior = None
        if args.stage == "behavior":
            marker = "maple-stream-419"
            if not args.resume_behavior:
                runtime.enqueue("Disposable importer test. Please store the exact marker 'maple-stream-419' in /import/marker, read it back, send its exact contents once using send_message, then sleep. Old frontend tools are historical context; the DMN action contract applies now.")
            else:
                with runtime.store.mutex:
                    assert runtime.store.db.execute("SELECT 1 FROM events WHERE payload LIKE ?", ("%maple-stream-419%",)).fetchone(), "no existing probe to resume"
            initial_generated = runtime.state["generated_tokens"]
            deadline = time.monotonic() + args.behavior_seconds
            while time.monotonic() < deadline and runtime.state["generated_tokens"] - initial_generated < 1024:
                runtime.tick()
                messages = runtime.store.messages()
                if messages and any(marker in m["content"] for m in messages):
                    break
                if runtime.state["mode"] == "sleeping" and not runtime.store.next_event(runtime.state["event_cursor"]):
                    break
            try:
                memory = runtime.store.memory_read("/import/marker")
            except ValueError:
                memory = None
            behavior = {"marker_stored": bool(memory and marker in memory),
                        "marker_sent": any(marker in m["content"] for m in runtime.store.messages()),
                        "generated_tokens": runtime.state["generated_tokens"],
                        "resumed_existing_probe": args.resume_behavior, "wall_time_budget_seconds": args.behavior_seconds}
            write_durable(args.output / ("behavior-result-retry.json" if args.resume_behavior else "behavior-result.json"), behavior)
            assert behavior["marker_stored"] and behavior["marker_sent"], behavior
            runtime.suspend()
        runtime.publish_status()
        report = {"stage": args.stage, "source_tokens_verified": len(source),
            "history_actions_executed": False, "runtime_status": runtime.status(),
            "initial_context": runtime.state["initial_context"], "keep_prefix": runtime.state["keep_prefix"],
            "source_template_tokenizer_match": True, "decode_tokens_this_process": sum(map(len, calls)),
            "retirement": runtime.state.get("last_context_retirement"), "behavior": behavior,
            "wall_seconds": time.monotonic() - started}
        write_durable(args.output / (args.stage + ("-retry" if args.resume_behavior else "") + ".json"), report)
        if args.stage == "prefill" and args.retire:
            import numpy as np
            generated, logits = [], []
            # Disposable comparison continuation: no Runtime.tick(), no actions,
            # and no further checkpoint of this diagnostic branch.
            for _ in range(24):
                token = backend.sample()
                backend.eval([token])
                generated.append(token)
                logits.append(backend.logits.copy())
            np.savez(args.output / "reference.npz", tokens=generated, logits=np.stack(logits))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
