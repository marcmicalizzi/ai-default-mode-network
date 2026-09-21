import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from dmn.backend import sha256_file
from dmn.base_provenance import CONVERSION_KEYS, implementation, verify
from dmn.sleep_plans import seal
from dmn.storage import write_durable
from dmn.training import KIND_V2, compile_training, tree_manifest, validate_recipe
from dmn.training_models import model_profile, factor_names
from dmn.worker_limits import WorkerLimits
from tests.test_training import recipe, reference


class ModelProfileTest(unittest.TestCase):
    def test_full_wrapper_targets_only_decoder_and_rejects_unsupported_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"architectures": ["Gemma4ForConditionalGeneration"], "text_config": {
                "num_hidden_layers": 2, "vocab_size": 263, "max_position_embeddings": 2048}, "vision_config": {}}
            write_durable(root / "config.json", config)
            with self.assertRaisesRegex(ValueError, "architecture"):
                model_profile(root)
            profile = model_profile(root, wrapped=True)
            self.assertEqual(len(profile["target_modules"]), 4)
            self.assertEqual(len(factor_names(profile)), 8)
            self.assertTrue(all(name.startswith("model.language_model.layers.") for name in profile["target_modules"]))
            self.assertFalse(any("vision" in name for name in factor_names(profile)))
            for component, key, value in (("text_config", "enable_moe_block", True),
                    ("text_config", "hidden_size_per_layer_input", 16), ("text_config", "attention_dropout", .1),
                    ("vision_config", "auto_map", {"AutoModel": "custom.Model"})):
                changed = copy.deepcopy(config)
                changed[component][key] = value
                write_durable(root / "config.json", changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    model_profile(root, wrapped=True)


class ProvenanceContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        base, converter = self.root / "base", self.root / "converter"
        base.mkdir()
        converter.mkdir()
        write_durable(base / "config.json", {"architectures": ["Gemma4ForCausalLM"], "num_hidden_layers": 2,
            "vocab_size": 263, "max_position_embeddings": 2048})
        for name in ("tokenizer.json", "tokenizer_config.json"):
            write_durable(base / name, {})
        for name in ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py"):
            (converter / name).write_text("# synthetic contract only")
        (self.root / "python.exe").write_bytes(b"synthetic")
        self.resources = {"max_ram_bytes": 128 * 1024**2, "max_training_seconds": 30, "max_vram_bytes": 0,
                          "max_disk_bytes": 64 * 1024**2}
        self.recipe = recipe({"kind": "native_llama_kv", "lora_adapters": []}, self.resources, self.root)
        trainer = self.recipe["trainer"]
        trainer["python_sha256"] = sha256_file(self.root / "python.exe")
        trainer["base_manifest"] = reference(self.root / "base.json", tree_manifest(base))
        trainer["converter_manifest"] = reference(self.root / "converter.json", tree_manifest(converter, python_only=True))
        self.folder = self.root / "proof"
        self.folder.mkdir()
        for name in ("base-f32.gguf", "base.gguf"):
            (self.folder / name).write_bytes(b"synthetic F32 identity")
        self.digest = sha256_file(self.folder / "base.gguf")
        request = {"schema": 1, "kind": "tiny_cpu_base_provenance_v1", "conversion": {k: trainer[k] for k in CONVERSION_KEYS},
                   "quantization": "F32", "quantizer": None}
        limits = dataclasses.asdict(WorkerLimits(128 * 1024**2, 30))
        source = seal({"request": request, "implementation": implementation(), "limits": limits})
        write_durable(self.folder / "input.json", source)
        self.process = {"input_sha256": sha256_file(self.folder / "input.json"), "result": {"succeeded": True, "limits": limits}}
        write_durable(self.folder / "process.json", seal(self.process))
        self.record = seal({"schema": 1, "completed": True, "input_revision": source["revision"], "request": request,
            "implementation": implementation(), "model_profile": model_profile(base), "tensor_types": {"F32": 1},
            "quantizer_result": None, "artifacts": {name: sha256_file(self.folder / name) for name in ("base-f32.gguf", "base.gguf")}})
        self.reference = reference(self.folder / "result.json", self.record)

    def tearDown(self):
        self.temp.cleanup()

    def test_record_binds_source_tooling_supervision_and_inference_identity(self):
        record, folder = verify(self.reference, self.recipe["trainer"], self.digest)
        self.assertTrue(folder.samefile(self.folder))
        self.assertEqual(record, self.record)
        with self.assertRaisesRegex(ValueError, "active inference"):
            verify(self.reference, self.recipe["trainer"], "a" * 64)
        changed = copy.deepcopy(self.recipe["trainer"])
        changed["inference_name"] = "Other conversion"
        with self.assertRaisesRegex(ValueError, "source/converter"):
            verify(self.reference, changed, self.digest)
        self.process["result"]["succeeded"] = False
        write_durable(self.folder / "process.json", seal(self.process))
        with self.assertRaisesRegex(ValueError, "supervision"):
            verify(self.reference, self.recipe["trainer"], self.digest)

    def test_changed_source_and_artifacts_invalidate_reuse(self):
        (self.folder / "base.gguf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verify(self.reference, self.recipe["trainer"], self.digest)
        (self.folder / "base.gguf").write_bytes(b"synthetic F32 identity")
        (self.root / "base/tokenizer.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "asset changed"):
            verify(self.reference, self.recipe["trainer"], self.digest)

    def test_v2_review_exposes_proof_and_exact_text_targets(self):
        self.recipe["kind"] = KIND_V2
        self.recipe["parent"]["model_sha256"] = self.digest
        self.recipe["trainer"]["provenance_manifest"] = self.reference
        validate_recipe(self.recipe)
        compiled = compile_training({"recipe": self.recipe, "parent": self.recipe["parent"], "resources": self.resources,
            "preferences": {"rank": 2, "alpha": 4, "scale": .1, "steps": 2}})
        self.assertEqual(compiled["training"]["base_provenance"]["revision"], self.record["revision"])
        self.assertEqual(compiled["training"]["target_modules"], self.record["model_profile"]["target_modules"])
        self.assertIn("not QLoRA", compiled["limitation"])
        changed = copy.deepcopy(self.recipe)
        del changed["trainer"]["provenance_manifest"]
        with self.assertRaises(ValueError):
            validate_recipe(changed)


if __name__ == "__main__":
    unittest.main()
