"""Plan or run sequential synthetic thread/placement trials on an idle machine.

Without --execute this only writes the plan. It never opens an instance. Each
measured run owns one subprocess, with a fresh context and deterministic seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.config import Config
from scripts.benchmark_inference import require_idle


def make_plan(config, threads, layers, tokens, warmup, steps, repeats):
    if config.backend != "llama" or config.system_prompt:
        raise ValueError("use a llama diagnostic config with an empty system prompt")
    if (tokens < 1 or warmup < 0 or steps < 8 or tokens + warmup + steps > config.n_ctx
            or not 1 <= repeats <= 10 or any(t < 1 for t in threads) or any(g < -1 for g in layers)):
        raise ValueError("invalid or overflowing benchmark parameters")
    return {"schema": 1, "synthetic_only": True, "completed": False,
            "base_config": config.to_dict(), "tokens": tokens, "warmup": warmup, "steps": steps,
            "repeats": repeats, "threads": list(dict.fromkeys([config.n_threads, *threads])),
            "gpu_layers": list(dict.fromkeys([config.n_gpu_layers, *layers])),
            "order": "Thread trials at original placement, then placement trials at fastest measured thread count, then baseline recheck.",
            "runs": [], "case_summaries": [],
            "limits": "Synthetic throughput only; does not validate changed-placement KV restore, retirement, actions or shutdown."}


def fastest(cases):
    # Prefer the earlier/lower-resource candidate on an exact tie.
    return max(cases, key=lambda case: case["median_tokens_per_second"])


def execute_plan(plan, output, python, guard_port):
    config = Config(**plan["base_config"])

    def persist():
        (output / "report.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def trial(stage, candidate):
        speeds = []
        for repeat in range(plan["repeats"]):
            require_idle(candidate, guard_port)
            name = f"{stage}-g{candidate.n_gpu_layers}-t{candidate.n_threads}-r{repeat + 1}"
            config_path = output / (name + ".json")
            config_path.write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")
            command = [str(python), str(ROOT / "scripts/benchmark_inference.py"),
                       "--config", str(config_path), "--output", str(output / name),
                       "--tokens", str(plan["tokens"]), "--warmup", str(plan["warmup"]),
                       "--steps", str(plan["steps"]), "--guard-port", str(guard_port)]
            print(f"Running {name}", flush=True)
            # Native models and CUDA allocations exit before the next trial.
            with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            row = {"name": name, "returncode": result.returncode}
            plan["runs"].append(row)
            persist()
            if result.returncode:
                raise RuntimeError(f"{name} failed; remaining trials were not started; inspect its log")
            report = json.loads((output / name / "report.json").read_text(encoding="utf-8"))
            if not report.get("completed"):
                raise RuntimeError(f"{name} did not complete")
            speeds.append(report["steady_tokens_per_second"])
            row["tokens_per_second"] = speeds[-1]
            persist()
        case = {"stage": stage, "n_threads": candidate.n_threads, "n_gpu_layers": candidate.n_gpu_layers,
                "median_tokens_per_second": statistics.median(speeds), "samples": speeds}
        plan["case_summaries"].append(case)
        persist()
        return case

    try:
        thread_cases = [trial("threads", dataclasses.replace(config, n_threads=t)) for t in plan["threads"]]
        best_thread = fastest(thread_cases)
        placement_cases = [best_thread]
        for layers in plan["gpu_layers"]:
            if layers != config.n_gpu_layers:
                placement_cases.append(trial("placement", dataclasses.replace(config,
                    n_threads=best_thread["n_threads"], n_gpu_layers=layers)))
        recheck = trial("baseline-recheck", config)
        best = fastest(placement_cases)
        initial = thread_cases[0]["median_tokens_per_second"]
        plan.update(completed=True, best_measured=best,
                    speedup_vs_initial_baseline=best["median_tokens_per_second"] / initial,
                    baseline_recheck_ratio=recheck["median_tokens_per_second"] / initial,
                    promotion_performed=False)
    except BaseException as exc:
        plan["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        persist()
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--gpu-layers", type=int, nargs="+", default=[27, 30])
    parser.add_argument("--tokens", type=int, default=25000)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--guard-port", type=int, default=8765)
    parser.add_argument("--execute", action="store_true", help="run now; only after maintenance shutdown")
    args = parser.parse_args(argv)
    config = Config.read(args.config)
    try:
        plan = make_plan(config, args.threads, args.gpu_layers, args.tokens, args.warmup, args.steps, args.repeats)
        if not 1 <= args.guard_port <= 65535:
            raise ValueError("guard-port must be between 1 and 65535")
    except ValueError as exc:
        parser.error(str(exc))
    if args.execute:
        require_idle(config, args.guard_port)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.execute:
        report = execute_plan(plan, output, sys.executable, args.guard_port)
        print(json.dumps({k: report[k] for k in
              ("best_measured", "speedup_vs_initial_baseline", "baseline_recheck_ratio")}, indent=2))
    else:
        print(f"Plan only; no model loaded. Saved to {output / 'plan.json'}")


if __name__ == "__main__":
    main()
