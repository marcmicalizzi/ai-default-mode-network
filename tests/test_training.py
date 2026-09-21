import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dmn.backend import sha256_file
from dmn.adapters import AdapterSpec
from dmn.config import Config
from dmn.deep_sleep import run_fixture_sleep, FixtureExecutor
from dmn.sleep_plans import seal, identity, read_record, implementation_identity
from dmn.storage import json_text, write_durable
from dmn.training import (CHECKS, KIND, CONTINUE_KIND, PACKAGES, CONVERTER_REVISION, tree_manifest,
                          verify_tree, validate_examples, read_completion, compile_training, parent_adapter_config)
from dmn.training_executor import TrainingExecutor
from tests import test_deep_sleep as sleep_fixtures
from tests.test_deep_sleep import Crash


def reference(path, value):
    write_durable(path, value)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def recipe(parent, resources, root):
    return {"schema": 1, "kind": KIND, "parent": parent, "resources": resources, "checks": CHECKS,
        "trainer": {"python": str(root / "python.exe"), "python_sha256": "a" * 64,
                    "packages": {name: "test" for name in PACKAGES},
                    "base_manifest": {"path": str(root / "base.json"), "sha256": "b" * 64},
                    "converter_manifest": {"path": str(root / "converter.json"), "sha256": "c" * 64},
                    "converter_revision": CONVERTER_REVISION, "inference_name": "Test",
                    "learning_rate": .01, "seed": 17}}


def bind_local_assets(value, root, assets, *, continuing):
    trainer = value["trainer"]
    trainer["python"] = str(Path(os.environ["DMN_TEST_TRAINING_PYTHON"]).resolve())
    trainer["python_sha256"] = sha256_file(Path(trainer["python"]))
    trainer["packages"] = json.loads(subprocess.check_output([trainer["python"], "-c",
        "import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in " + repr(PACKAGES) + "}))"], text=True))
    trainer["base_manifest"] = reference(root / "base.json", tree_manifest(assets / "base"))
    trainer["converter_manifest"] = reference(root / "converter.json", tree_manifest(
        Path(os.environ["DMN_TEST_LORA_CONVERTER"]).resolve(), python_only=True))
    trainer["inference_name"] = "DMN generated Gemma4 PEFT conversion fixture; not an instance"
    if continuing:
        value["kind"] = CONTINUE_KIND
        trainer["parent_adapter_manifest"] = reference(root / "parent.json", tree_manifest(assets / "adapter"))
    return value


class TrainingContractTest(unittest.TestCase):
    def setUp(self):
        self.case = sleep_fixtures.SleepTest()
        self.case.setUp()
        self.root = Path(self.case.temp.name)
        # A dependency-free stand-in for validating plan contracts only.
        self.case.r.backend.fingerprint.update(kind="native_llama_kv", model_sha256="a" * 64)
        self.case.plan["checks"] = CHECKS
        self.case.plan["preferences"]["adoption"] = "review_first"
        self.recipe = recipe(identity(self.case.r.backend.fingerprint), self.case.plan["resources"], self.root)
        self.case.recipe_id = self.case.r.offer_learning_recipe(self.recipe)["revision"]

    def tearDown(self):
        self.case.tearDown()

    def test_compilation_binds_actual_training_semantics_and_keeps_approval_separate(self):
        c = self.case
        revision = c.compile()
        value = read_record(c.r.store, "sleep_executions", revision)
        self.assertTrue(value["training_requested"])
        self.assertFalse(value["training_performed"])
        self.assertEqual(value["training"]["target_modules"], ["q_proj", "o_proj"])
        self.assertEqual(value["training"]["example_order"], "round_robin_in_reviewed_order")
        self.assertEqual(value["checks"], CHECKS)
        result, effect = c.r._plan_action({"op": "deep_sleep", "revision": revision}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)
        c.review(revision)
        c.generate({"op": "learning_execution_decide", "revision": revision, "decision": "approve"})
        c.r.sleep_test_mode = False
        result, effect = c.r._plan_action({"op": "deep_sleep", "revision": revision}, [])
        self.assertFalse(result["ok"])
        self.assertIsNone(effect)

    def test_existing_adapter_is_rejected_instead_of_discarded(self):
        value = copy.deepcopy(self.recipe)
        value["parent"]["lora_adapters"] = [{"sha256": "a" * 64}]
        with self.assertRaisesRegex(ValueError, "existing learning"):
            self.case.r.offer_learning_recipe(value)

    def continuation_recipe(self):
        parent = self.root / "adapter"
        parent.mkdir()
        write_durable(parent / "adapter_config.json", {"r": 2, "lora_alpha": 4, "peft_type": "LORA",
            "task_type": "CAUSAL_LM", "bias": "none", "lora_dropout": 0, "target_modules": ["q_proj", "o_proj"]})
        (parent / "adapter_model.safetensors").write_bytes(b"synthetic contract-only factors")
        c = self.case
        c.r.backend.fingerprint["lora_adapters"] = [AdapterSpec("unused", "d" * 64, "a" * 64, .1).identity()]
        value = copy.deepcopy(self.recipe)
        value.update(kind=CONTINUE_KIND, parent=identity(c.r.backend.fingerprint))
        value["trainer"]["parent_adapter_manifest"] = reference(self.root / "parent.json", tree_manifest(parent))
        return value, parent

    def test_continuation_review_binds_parent_factors_and_preserves_geometry_and_scale(self):
        value, parent = self.continuation_recipe()
        c = self.case
        c.recipe_id = c.r.offer_learning_recipe(value)["revision"]
        compiled = read_record(c.r.store, "sleep_executions", c.compile())
        self.assertEqual(compiled["lineage"]["parent_adapter"], value["parent"]["lora_adapters"][0])
        self.assertEqual(compiled["lineage"]["parent_peft"]["adapter_model.safetensors"], sha256_file(parent / "adapter_model.safetensors"))
        self.assertEqual(compiled["training"]["training_scale"], compiled["training"]["deployment_scale_float32"])
        self.assertTrue(compiled["training"]["optimizer_reset"])
        for key, changed in (("rank", 4), ("alpha", 8), ("scale", 1.)):
            revised = copy.deepcopy(compiled)
            revised["preferences"][key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preserves parent"):
                compile_training(revised)
        (parent / "adapter_model.safetensors").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "asset changed"):
            compile_training(compiled)

    def test_continuation_refuses_missing_stacked_or_disabled_parents(self):
        value, _ = self.continuation_recipe()
        old = value["parent"]["lora_adapters"][0]
        for adapters in ([], [old, old], [{**old, "scale": 0.}], [{**old, "scale": -.1}],
                         [{**old, "activation": "alora"}], [{**old, "base_model_sha256": "f" * 64}]):
            changed = copy.deepcopy(value)
            changed["parent"]["lora_adapters"] = adapters
            with self.subTest(adapters=adapters), self.assertRaises(ValueError):
                self.case.r.offer_learning_recipe(changed)

    def test_parent_peft_rejects_features_that_conversion_might_silently_drop(self):
        value, parent = self.continuation_recipe()
        config = parent_adapter_config(parent)
        for key, enabled in (("use_dora", True), ("modules_to_save", ["lm_head"]), ("rank_pattern", {"q_proj": 4}),
                             ("use_rslora", True), ("lora_dropout", .1), ("alora_invocation_tokens", [1]),
                             ("target_modules", ["q_proj"]), ("unknown_feature", True)):
            write_durable(parent / "adapter_config.json", {**config, key: enabled})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unsupported parent PEFT"):
                parent_adapter_config(parent)
        write_durable(parent / "adapter_config.json", config)
        (parent / "adapter_model.bin").write_bytes(b"no pickle")
        ref = reference(self.root / "invalid-parent.json", tree_manifest(parent))
        with self.assertRaisesRegex(ValueError, "safetensors/config"):
            verify_tree(ref, adapter=True)

    def test_tokenizer_and_loss_mask_must_match_reviewed_ids(self):
        tokenizer = mock.Mock()
        tokenizer.encode.side_effect = lambda text, **_: list(text.encode())
        row = {"input": "a", "target": "bc", "tokens": [97, 98, 99], "loss_mask": [0, 1, 1], "labels": [-100, 98, 99]}
        validate_examples([row], tokenizer, 256, 10)
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            validate_examples([{**row, "tokens": [97, 99, 99]}], tokenizer, 256, 10)
        with self.assertRaisesRegex(ValueError, "mask or labels"):
            validate_examples([{**row, "labels": [97, 98, 99]}], tokenizer, 256, 10)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            validate_examples([row], tokenizer, 256, 2)

    def test_changed_or_added_training_assets_fail_identity_check(self):
        base = self.root / "base"
        base.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (base / name).write_text("{}")
        ref = reference(self.root / "base.json", tree_manifest(base))
        self.assertTrue(verify_tree(ref, base=True).samefile(base))
        (base / "tokenizer.json").write_text('{"changed":1}')
        with self.assertRaisesRegex(ValueError, "asset changed"):
            verify_tree(ref, base=True)
        (base / "tokenizer.json").write_text("{}")
        (base / "injected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "asset set changed"):
            verify_tree(ref, base=True)

    def test_failed_worker_records_uncertainty_and_honors_previous_wake(self):
        c = self.case
        c.plan["preferences"]["failure"] = "wake_previous"
        run_id = c.request()
        class FailedWorker(FixtureExecutor):
            def candidate(self, *args):
                raise ValueError("training worker failed: time_limit")
        result = run_fixture_sleep(c.root, run_id, executor=FailedWorker(c.factory))
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertIsNone(result["report"]["training_performed"])
        self.assertEqual(result["report"]["training_status"], "unknown_or_partial")
        self.assertEqual(result["report"]["reconstruction"]["prompt_tokens_reevaluated"], 0)

    @unittest.skipUnless(os.name == "nt", "Windows process-tree timeout integration")
    def test_real_worker_timeout_flows_through_the_approved_failure_choice(self):
        from dmn.worker_limits import WorkerLimits, run_cpu_worker
        c = self.case
        c.plan["preferences"]["failure"] = "wake_previous"
        run_id = c.request()
        class TimedWorker(FixtureExecutor):
            def candidate(self, root, compiled):
                evidence = run_cpu_worker(sys.executable, ["-c", "import time; time.sleep(60)"],
                    cwd=root, log=root / "timeout-test.log", limits=WorkerLimits(128 * 1024**2, .5))
                if not evidence["succeeded"]:
                    raise ValueError("worker stopped: " + evidence["outcome"])
                raise AssertionError("sleeping worker unexpectedly finished")
        result = run_fixture_sleep(c.root, run_id, executor=TimedWorker(c.factory))
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertEqual(result["report"]["reason"], "worker stopped: time_limit")
        self.assertEqual(result["report"]["reconstruction"]["prompt_tokens_reevaluated"], 0)

    def test_completed_worker_is_recovered_without_training_again(self):
        c = self.case
        run_id = c.request()
        class CompletedWorker(FixtureExecutor):
            calls = 0
            def candidate(self, root, compiled):
                self.calls += 1
                # Simulate receipt + successful supervision already committed,
                # but crash before the supervisor's Candidate publication.
                raise Crash("worker receipt saved")
            def recover(self, root, compiled):
                return {"adapters": [], "training_performed": True}
            def validate(self, *args):
                pass
        executor = CompletedWorker(c.factory)
        with self.assertRaises(Crash):
            run_fixture_sleep(c.root, run_id, executor=executor)
        result = run_fixture_sleep(c.root, run_id, executor=executor)
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertTrue(result["report"]["training_performed"])
        self.assertEqual(executor.calls, 1)

    def test_crash_during_failure_policy_resumes_original_reason_without_work(self):
        c = self.case
        c.plan["preferences"]["failure"] = "wake_previous"
        run_id = c.request()
        class FailedWorker(FixtureExecutor):
            calls = 0
            def candidate(self, *args):
                self.calls += 1
                raise ValueError("specific tokenizer mismatch")
        executor = FailedWorker(c.factory)
        def crash(phase):
            if phase == "FailurePolicy":
                raise Crash("interrupted failure wake")
        with self.assertRaises(Crash):
            run_fixture_sleep(c.root, run_id, executor=executor, fault=crash)
        result = run_fixture_sleep(c.root, run_id, executor=executor)
        self.assertEqual(result["phase"], "WakeCommitted")
        self.assertEqual(result["report"]["reason"], "specific tokenizer mismatch")
        self.assertEqual(executor.calls, 1)

    def test_worker_files_participate_in_managed_erasure(self):
        from dmn.ending import erase_managed_state
        c = self.case
        run_id = c.request()
        folder = c.root / "sleep" / run_id / "worker"
        (folder / "adapter").mkdir(parents=True)
        (folder / "parent-adapter").mkdir()
        for name in ("input.json", "result.json", "result.json.partial", "process.json", "failure.json", "worker.log", "base-check.gguf", "parent-check.gguf", "adapter.gguf"):
            (folder / name).write_text("private selected examples")
        for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
            (folder / "adapter" / name).write_text("candidate")
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (folder / "parent-adapter" / name).write_text("parent learning")
        erase_managed_state(c.root)
        self.assertFalse((c.root / "sleep").exists())


class CompletionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "sleep" / ("a" * 32)
        self.work = self.folder / "worker"
        (self.work / "adapter").mkdir(parents=True)
        names = ("base-check.gguf", "adapter.gguf", "adapter/adapter_config.json", "adapter/adapter_model.safetensors")
        for name in names:
            (self.work / name).write_text(name)
        self.compiled = {"revision": "d" * 64, "parent": {"model_sha256": sha256_file(self.work / "base-check.gguf")},
                         "preferences": {"steps": 3, "scale": .1}, "examples": [],
                         "resources": {"max_ram_bytes": 128 * 1024**2, "max_training_seconds": 60}}
        self.result = {"schema": 1, "execution": self.compiled["revision"], "completed": True, "training_performed": True,
            "steps_completed": 3, "training_seconds": .1, "trainable_parameters": 4,
            "artifacts": {name: sha256_file(self.work / name) for name in names},
            "examples_sha256": hashlib.sha256(json_text([]).encode()).hexdigest(),
            "checks": {key: True for key in CHECKS if key != "retained_tokens_and_rng"},
            "loss_before": [2.], "loss_after_training_scale": [1.], "loss_after_deployment_scale": [1.8],
            "beneficial_learning_certified": False}
        write_durable(self.work / "result.json", seal(self.result))
        self.process = {"execution": self.compiled["revision"], "result": {"succeeded": True,
            "limits": {"max_committed_bytes": 128 * 1024**2, "max_seconds": 60}}}
        self.executor = TrainingExecutor(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def test_corrupt_artifact_or_incomplete_checks_cannot_be_adopted(self):
        read_completion(self.work, self.compiled)
        write_durable(self.work / "result.json", seal({**self.result, "checks": {}}))
        with self.assertRaisesRegex(ValueError, "checks"):
            read_completion(self.work, self.compiled)
        write_durable(self.work / "result.json", seal(self.result))
        (self.work / "adapter.gguf").write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            read_completion(self.work, self.compiled)

    def test_receipt_without_successful_supervision_cannot_recover(self):
        with self.assertRaises(FileNotFoundError):
            self.executor.recover(self.root, self.compiled)
        self.process["result"]["succeeded"] = False
        write_durable(self.work / "process.json", seal(self.process))
        with self.assertRaisesRegex(ValueError, "supervision"):
            self.executor.recover(self.root, self.compiled)
        self.assertFalse((self.root / "adapters").exists())

    def test_recovery_verifies_and_copies_without_launching_training(self):
        write_durable(self.work / "process.json", seal(self.process))
        with mock.patch("dmn.training_executor.run_cpu_worker", side_effect=AssertionError("must not retrain")):
            candidate = self.executor.recover(self.root, self.compiled)
            self.executor.validate(self.root, self.compiled, candidate)
            self.assertTrue(candidate["training_performed"])
            self.assertEqual(candidate["adapters"][0]["sha256"], self.result["artifacts"]["adapter.gguf"])
            changed = copy.deepcopy(candidate)
            changed["adapters"][0]["scale"] = 1.
            with self.assertRaisesRegex(ValueError, "candidate differs"):
                self.executor.validate(self.root, self.compiled, changed)

    def test_continuation_receipt_binds_parent_conversion_and_loaded_factors(self):
        (self.work / "parent-adapter").mkdir()
        names = ("parent-check.gguf", "parent-adapter/adapter_config.json", "parent-adapter/adapter_model.safetensors")
        for name in names:
            (self.work / name).write_text(name)
            self.result["artifacts"][name] = sha256_file(self.work / name)
        lineage = {"parent_adapter": {"sha256": self.result["artifacts"]["parent-check.gguf"]},
                   "parent_peft": {name: self.result["artifacts"]["parent-adapter/" + name] for name in
                                   ("adapter_config.json", "adapter_model.safetensors")}}
        self.compiled["lineage"] = lineage
        self.result.update(lineage=copy.deepcopy(lineage), parent_factors_loaded_exactly=True)
        write_durable(self.work / "result.json", seal(self.result))
        read_completion(self.work, self.compiled)
        for field, changed in (("lineage", None), ("parent_factors_loaded_exactly", False)):
            write_durable(self.work / "result.json", seal({**self.result, field: changed}))
            with self.assertRaises(ValueError):
                read_completion(self.work, self.compiled)
        self.compiled["lineage"]["parent_adapter"]["sha256"] = "f" * 64
        self.result["lineage"] = copy.deepcopy(self.compiled["lineage"])
        write_durable(self.work / "result.json", seal(self.result))
        with self.assertRaisesRegex(ValueError, "deployed parent"):
            read_completion(self.work, self.compiled)


@unittest.skipUnless(os.name == "nt" and all(os.environ.get(k) for k in (
    "DMN_TEST_LORA_PARENT", "DMN_TEST_TRAINING_PYTHON", "DMN_TEST_LORA_CONVERTER")), "requires CPU training/native environments and tiny local assets")
class NativeTrainingTest(unittest.TestCase):
    def test_approved_examples_train_convert_and_rebuild_native_context(self):
        self.run_training(continuing=False)

    def test_existing_adapter_continues_at_deployed_strength_and_rebuilds_native_context(self):
        self.run_training(continuing=True)

    def test_mismatched_parent_is_rejected_before_gradients(self):
        from dmn.worker_limits import WorkerLimits, run_cpu_worker
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assets = Path(os.environ["DMN_TEST_LORA_PARENT"]).resolve()
            parent = {"kind": "native_llama_kv", "model_sha256": sha256_file(assets / "base.gguf")}
            parent["lora_adapters"] = [AdapterSpec("unused", "0" * 64, parent["model_sha256"], .1).identity()]
            resources = {"max_ram_bytes": 1536 * 1024**2, "max_training_seconds": 180,
                         "max_vram_bytes": 0, "max_disk_bytes": 32 * 1024**2}
            value = bind_local_assets(recipe(parent, resources, root), root, assets, continuing=True)
            compiled = seal(compile_training({"recipe": seal(value), "parent": parent, "resources": resources,
                "implementation": implementation_identity(), "preferences": {"rank": 2, "alpha": 4, "scale": .1, "steps": 1},
                "examples": []}))
            work = root / "worker"
            work.mkdir()
            write_durable(work / "input.json", compiled)
            evidence = run_cpu_worker(value["trainer"]["python"], ["-m", "dmn.training_worker", str(work)],
                cwd=Path(__file__).resolve().parents[1], log=work / "worker.log",
                limits=WorkerLimits(resources["max_ram_bytes"], resources["max_training_seconds"]))
            self.assertFalse(evidence["succeeded"])
            failure = json.loads((work / "failure.json").read_text())
            self.assertIn("do not reproduce the deployed adapter GGUF", failure["reason"])
            self.assertFalse((work / "parent-adapter").exists())
            self.assertFalse((work / "adapter").exists())
            self.assertFalse((work / "result.json").exists())

    def run_training(self, *, continuing):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from dmn.runtime import Runtime
        from tests.test_adapters import native_config
        c = sleep_fixtures.SleepTest()
        c.setUp()
        try:
            c.r.close()
            c.root = Path(c.temp.name) / "native"
            assets = Path(os.environ["DMN_TEST_LORA_PARENT"]).resolve()
            config = native_config(assets)
            c.config = dataclasses.replace(config, lora_adapters=config.lora_adapters if continuing else (), n_ctx=32768,
                preparation_tokens=1, clock_interval_seconds=0, checkpoint_policy="effects", checkpoint_tokens=100000)
            with mock.patch("dmn.runtime.PROTOCOL", "Disposable training integration. Injected choices are mechanics tests, not real consent."):
                c.r = Runtime(c.root, c.config, sleep_test_mode=True)
            c.plan["checks"] = CHECKS
            c.plan["preferences"].update(steps=16, scale=.1)
            c.plan["resources"].update(max_ram_bytes=1536 * 1024**2, max_training_seconds=180)
            value = recipe(identity(c.r.backend.fingerprint), c.plan["resources"], Path(c.temp.name))
            bind_local_assets(value, Path(c.temp.name), assets, continuing=continuing)
            c.recipe_id = c.r.offer_learning_recipe(value)["revision"]
            c.r.tick()
            c.generate({"op": "send_message", "content": "fixture message once"})
            run_id = c.request()
            # New input stays queued; it is absent from the compiled examples.
            from dmn.storage import Store
            store = Store(c.root)
            queued_id = store.enqueue("user_message", {"content": "Unselected input must never become training data."})
            store.close()
            def crash_after_worker(phase):
                if phase == "worker_completed":
                    raise Crash("successful worker, candidate phase not yet committed")
            with self.assertRaises(Crash):
                run_fixture_sleep(c.root, run_id, fault=crash_after_worker)
            receipt_path = c.root / "sleep" / run_id / "worker" / "result.json"
            receipt_hash = sha256_file(receipt_path)
            with mock.patch("dmn.training_executor.run_cpu_worker", side_effect=AssertionError("completed worker must not rerun")):
                result = run_fixture_sleep(c.root, run_id)
            self.assertEqual(sha256_file(receipt_path), receipt_hash)
            if result["phase"] != "WakeCommitted":
                log = c.root / "sleep" / run_id / "worker" / "worker.log"
                self.fail(str(result) + "\nSynthetic worker diagnostics:\n" + (log.read_text(errors="replace")[-6000:] if log.exists() else "no worker log"))
            self.assertEqual(result["report"]["outcome"], "candidate_adopted", result)
            self.assertTrue(result["report"]["training_performed"])
            completion = json.loads(receipt_path.read_text())
            if continuing:
                self.assertTrue(completion["parent_factors_loaded_exactly"])
                self.assertEqual(completion["lineage"]["parent_adapter"], config.lora_adapters[0].identity())
                self.assertEqual(completion["loss_after_training_scale"], completion["loss_after_deployment_scale"])
                self.assertNotEqual(completion["artifacts"]["adapter.gguf"], config.lora_adapters[0].sha256)
                self.assertEqual(sha256_file(assets / "adapter.gguf"), config.lora_adapters[0].sha256)
            _, saved = c.inspect(run_id)
            old = json.loads((c.source / "engine.json").read_text())
            new = json.loads((saved / "engine.json").read_text())
            self.assertEqual(old["tokens"], new["tokens"])
            self.assertEqual(old["rng"], new["rng"])
            manifest = json.loads((saved / "manifest.json").read_text())
            c.r = Runtime(c.root, Config(**manifest["fingerprint"]["config"]))
            self.assertEqual(c.r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
            self.assertEqual(len(c.r.store.messages()), 1)
            self.assertLess(c.r.state["event_cursor"], queued_id)
            # Save only synthetic diagnostic evidence if explicitly requested.
            if os.environ.get("DMN_TEST_TRAINING_REPORT"):
                report = Path(os.environ["DMN_TEST_TRAINING_REPORT"])
                if continuing:
                    report = report.with_name(report.stem + "-continuation.json")
                write_durable(report, {
                    "completed": True, "training": result["report"]["candidate"]["training"],
                    "lineage": completion.get("lineage"), "parent_factors_loaded_exactly": completion.get("parent_factors_loaded_exactly"),
                    "retained_tokens": len(old["tokens"]), "tokens_and_rng_equal": True,
                    "strict_restore_replay": 0, "queued_input_preserved": True, "messages_unchanged": True,
                    "completed_worker_recovered_without_retraining": True})
        finally:
            c.tearDown()


if __name__ == "__main__":
    unittest.main()
