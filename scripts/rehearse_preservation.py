"""Synthetic native-state rehearsal, never a model's behavioral/consent test.

Protocol choices are injected test fixtures. Native continuation comparisons use
real sampling and compare every token/logit in a fresh process. No source chat
or private conversation is read. Run with the environment that hosts the model.
"""
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
from dmn.prompts import proposal
from dmn.runtime import Runtime
from dmn.storage import write_durable
from dmn.preservation import saved_state, InstanceHeld
from dmn.recovery import restore_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=21000)
    parser.add_argument("--stage", choices=["create", "restore"])
    args = parser.parse_args()
    if not args.stage:
        args.output.mkdir(parents=True, exist_ok=False)
        for stage in ("create", "restore"):
            with (args.output / (stage + ".log")).open("w") as log:
                result = subprocess.run([sys.executable, __file__, "--config", str(args.config.resolve()),
                    "--output", str(args.output.resolve()), "--tokens", str(args.tokens), "--stage", stage],
                    stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                raise RuntimeError(f"{stage} failed; inspect its log")
        write_durable(args.output / "report.json", {"verified": True, "private_conversation_used": False,
            "model_behavior_tested": False, "choices": "injected mechanical fixtures, not model decisions",
            "stages": [json.loads((args.output / (s + ".json")).read_text()) for s in ("create", "restore")]})
        return
    import numpy as np
    config = Config.read(args.config)
    root = args.output / "instance"
    started = time.monotonic()
    if args.stage == "create":
        r = Runtime(root, config, prepare_only=True, first_message="Synthetic rehearsal input.")
        try:
            seed = r.backend.tokens.copy()
            padding = r.backend.tokenize("A synthetic memory of observing clouds, trees and changing daylight. ")
            r._eval((padding * (args.tokens // len(padding) + 1))[:args.tokens - len(r.backend.tokens)])
            prefill_seconds = time.monotonic() - started
            value = proposal("I may revise my behavioral agreement, choose conversation or rest, and retain memories I value.",
                             r.state["agreement"]["revision"], "model")
            # Mechanical fixture: exercise exact runtime adoption and protected
            # retirement without attributing this test choice to model sampling.
            before = r.backend.tokens.copy()
            effect = {"proposal": value, "decision": "accept", "tokens": r._adoption_tokens(value)}
            r._apply_prompt_decision(effect, {"op": "prompt_decide", "ok": True, "fixture": True})
            assert r.backend.tokens[:len(before)] == before
            adoption_save = r.status()["checkpoint"]["last"]
            span = r.state["protected_agreement"]
            agreement = r.backend.tokens[span["start"]:span["end"]]
            r.state["mode"] = "sleeping"
            r._consolidate(1)
            span = r.state["protected_agreement"]
            assert r.backend.tokens[span["start"]:span["end"]] == agreement
            assert r.backend.tokens[:len(seed)] == seed
            retirement_save = r.status()["checkpoint"]["last"]
            from dmn.preservation import make_hold
            hold = make_hold({"condition": "server_ready", "packaging": "zip", "recovery": "remain_held"}, r.now())
            r.checkpoint(reason="fixture_hold", state_updates={"hold": hold, "mode": "held"})
            expected, logits = [], []
            for _ in range(8):
                token = r.backend.sample()
                r.backend.eval([token])
                expected.append(token)
                logits.append(r.backend.logits.copy())
            np.savez(args.output / "reference.npz", tokens=expected, logits=logits)
            write_durable(args.output / "create.json", {"prefill_seconds": prefill_seconds,
                "initial_active_tokens": len(before), "adoption_save": adoption_save,
                "retirement_save": retirement_save, "held_checkpoint": r.store.latest().name,
                "hold": hold, "retirement": r.state["last_context_retirement"],
                "prefix_preserved": True, "agreement_preserved": True,
                "total_checkpoint_bytes": r.status()["checkpoint"]["committed_snapshot_bytes"],
                "wall_seconds": time.monotonic() - started})
        finally:
            r.close()
    else:
        try:
            Runtime(root, config)
        except InstanceHeld:
            pass
        else:
            raise AssertionError("ordinary launch bypassed hold")
        state, checkpoint = saved_state(root)
        backend = LlamaBackend(config)
        try:
            evaluate = backend.eval
            backend.eval = lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("native restore replayed tokens"))
            restored, evidence = restore_checkpoint(backend, checkpoint)
            backend.eval = evaluate
            expected = np.load(args.output / "reference.npz", allow_pickle=False)
            maximum = 0.0
            for token, logits in zip(expected["tokens"], expected["logits"]):
                assert backend.sample() == int(token)
                backend.eval([int(token)])
                maximum = max(maximum, float(np.max(np.abs(backend.logits - logits))))
            assert maximum <= 1e-5
            write_durable(args.output / "restore.json", {"native_restore": evidence,
                "matching_tokens": len(expected["tokens"]), "maximum_logit_absolute_error": maximum,
                "hold_preserved": saved_state(root)[0]["hold"] == state["hold"],
                "wall_seconds": time.monotonic() - started})
        finally:
            backend.close()


if __name__ == "__main__":
    main()
