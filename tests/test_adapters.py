import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from dmn.adapters import AdapterSpec
from dmn.backend import DemoBackend, sha256_file
from dmn.config import Config
from dmn.ending import erase_managed_state
from dmn.packaging import package_instance
from dmn.recovery import reconstruction_compatible, same_native_environment
from dmn.runtime import Runtime


def fingerprint(specs=()):
    config = Config(model_path="base.gguf", lora_adapters=specs)
    return {"kind": "native_llama_kv", "model_sha256": "b" * 64,
            "config": config.to_dict(), "lora_adapters": [s.identity() for s in config.lora_adapters]}


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.spec = AdapterSpec("adapter.gguf", "a" * 64, "b" * 64, .1)

    def test_identity_includes_order_strength_base_and_content_but_not_location(self):
        original = fingerprint([self.spec])
        relocated = fingerprint([dataclasses.replace(self.spec, path="elsewhere.gguf", scale=.1000000001)])
        self.assertTrue(same_native_environment(original, relocated))
        reconstruction_compatible(original, relocated)
        for specs in ([], [dataclasses.replace(self.spec, scale=0)],
                      [dataclasses.replace(self.spec, sha256="c" * 64)],
                      [dataclasses.replace(self.spec, base_model_sha256="c" * 64)]):
            other = fingerprint(specs)
            self.assertFalse(same_native_environment(original, other))
            with self.assertRaisesRegex(ValueError, "adapter identity"):
                reconstruction_compatible(original, other)
        second = dataclasses.replace(self.spec, sha256="c" * 64)
        with self.assertRaisesRegex(ValueError, "adapter identity"):
            reconstruction_compatible(fingerprint([self.spec, second]), fingerprint([second, self.spec]))

    def test_legacy_empty_identity_and_incomplete_or_research_identity(self):
        old = fingerprint()
        old.pop("lora_adapters")
        old["config"].pop("lora_adapters")
        self.assertTrue(same_native_environment(old, fingerprint()))
        reconstruction_compatible(old, fingerprint())
        invalid = fingerprint([self.spec])
        invalid.pop("lora_adapters")
        with self.assertRaisesRegex(ValueError, "disagree"):
            same_native_environment(invalid, invalid)
        with self.assertRaisesRegex(ValueError, "adapter identity"):
            reconstruction_compatible({**old, "research_lora": {"sha256": "x"}}, old)

    def test_bad_config_and_relative_paths(self):
        for changes in ({"scale": float("nan")}, {"scale": float("inf")}, {"scale": True},
                        {"scale": 1e50}, {"scale": 1e-50}, {"sha256": "bad"}, {"path": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(self.spec, **changes)
        with self.assertRaises(ValueError):
            Config(lora_adapters=[self.spec, self.spec])
        with self.assertRaises(ValueError):
            Config(backend="demo", lora_adapters=[self.spec])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(Config(lora_adapters=[self.spec]).to_dict()))
            self.assertEqual(Config.read(path).lora_adapters[0].path, str((Path(folder) / "adapter.gguf").resolve()))

    def test_package_binds_active_external_adapter_and_erasure_only_removes_owned_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "instance"
            external = Path(folder) / "outside.gguf"
            external.write_bytes(b"unit-test artifact; no native model")
            digest = sha256_file(external)
            config = Config(backend="demo", n_ctx=16384)
            r = Runtime(root, config, DemoBackend(config, b"quiet"))
            hold = {"id": "fixture", "packaging": "zip", "condition": "server_ready", "recovery": "remain_held"}
            r.checkpoint(state_updates={"hold": hold})
            saved = r.store.latest()
            r.close()
            managed = root / "adapters"
            managed.mkdir()
            (managed / (digest + ".gguf")).write_bytes(external.read_bytes())
            manifest_path = saved / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["fingerprint"] = fingerprint([AdapterSpec(str(external), digest, "b" * 64)])
            manifest_path.write_text(json.dumps(manifest))
            output = Path(folder) / "held.zip"
            self.assertTrue(package_instance(root, output)["verified"])
            with zipfile.ZipFile(output) as archive:
                inventory = json.loads(archive.read("preservation.json"))
                adapter = inventory["adapter_files"][0]
                self.assertEqual(archive.read(adapter["archive_path"]), external.read_bytes())
                self.assertFalse(adapter["inside_instance"])
            external.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "adapter does not match"):
                package_instance(root, Path(folder) / "bad.zip")
            unknown = managed / "user-file.txt"
            unknown.write_text("retain unknown files")
            failures = erase_managed_state(root)
            self.assertFalse((managed / (digest + ".gguf")).exists())
            self.assertTrue(unknown.exists())
            self.assertTrue(external.exists())
            self.assertIn(str(Path("adapters") / "user-file.txt"), failures)


def native_config(folder, strength=.1):
    folder = Path(folder)
    model, adapter = folder / "base.gguf", folder / "adapter.gguf"
    if model.stat().st_size > 4 * 1024 * 1024 or adapter.stat().st_size > 128 * 1024:
        raise ValueError("this test accepts only the generated tiny training fixture")
    return Config(model_path=str(model), lora_adapters=[AdapterSpec(str(adapter), sha256_file(adapter),
                  sha256_file(model), strength)], n_ctx=2048, n_batch=64, n_threads=1,
                  n_gpu_layers=0, offload_kqv=False, flash_attn=True, swa_full=False,
                  experimental_compact_swa=True, pack_checkpoints=True, type_k="q8_0", type_v="q8_0",
                  prompt_format="plain", checkpoint_reserve_bytes=0)


@unittest.skipUnless(os.environ.get("DMN_TEST_LORA_PARENT"), "set DMN_TEST_LORA_PARENT to the completed tiny training probe")
class NativeAdapterTest(unittest.TestCase):
    def test_production_config_cold_restore_and_changed_weights_rejected(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.recovery import restore_checkpoint
        from scripts.probe_lora_wake import continuation, save
        config = native_config(Path(os.environ["DMN_TEST_LORA_PARENT"]).resolve())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with_backend = LlamaBackend(config)
            try:
                with_backend.eval([1, 11, 6, 7] * 80)
                with_backend.shift(8, 64)
                with_backend.eval([12])
                adapted_logits = with_backend.logits.copy()
                save(with_backend, root / "saved", 1)
                expected, logits = continuation(with_backend)
            finally:
                with_backend.close()
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            child = subprocess.run([sys.executable, "-m", "tests.test_adapters", "--restore",
                                    str(config_path), str(root)], capture_output=True, text=True,
                                   timeout=60, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
            self.assertEqual(child.returncode, 0, child.stderr[-4000:])
            restored = json.loads((root / "result.json").read_text())
            self.assertEqual(restored["tokens"], expected)
            self.assertTrue(np.array_equal(np.load(root / "logits.npy"), logits))
            self.assertEqual(restored["evidence"]["prompt_tokens_reevaluated"], 0)
            changed = LlamaBackend(dataclasses.replace(config, lora_adapters=()))
            try:
                for policy in ("strict", "fallback", "rebuild"):
                    with self.subTest(policy=policy), self.assertRaises(ValueError):
                        restore_checkpoint(changed, root / "saved", policy)
                self.assertEqual(changed.decode_calls, 0)
                changed.eval([1, 11, 6, 7] * 80)
                changed.shift(8, 64)
                changed.eval([12])
                self.assertGreater(float(np.max(np.abs(changed.logits - adapted_logits))), 1e-5)
            finally:
                changed.close()
            bad_spec = dataclasses.replace(config.lora_adapters[0], sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "declared SHA"):
                LlamaBackend(dataclasses.replace(config, lora_adapters=[bad_spec]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        import numpy as np
        from dmn.backend import LlamaBackend
        from dmn.recovery import restore_checkpoint
        from scripts.probe_lora_wake import continuation
        config, root = Config.read(Path(sys.argv[2])), Path(sys.argv[3])
        backend = LlamaBackend(config)
        try:
            _, evidence = restore_checkpoint(backend, root / "saved")
            tokens, logits = continuation(backend)
            (root / "result.json").write_text(json.dumps({"tokens": tokens, "evidence": evidence}))
            np.save(root / "logits.npy", logits)
        finally:
            backend.close()
    else:
        unittest.main()
