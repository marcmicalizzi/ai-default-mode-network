import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from dmn.backend import DemoBackend, sha256_file
from dmn.config import Config
from dmn.initial_context import (CHAIN, DISABLED, SOURCE_BUILD, prepare_initial_context,
                                 resolve_sampler, validate_request)
from dmn.runtime import Runtime


class TextFixture(DemoBackend):
    """Codepoint tokens verify orchestration only, not native template parity."""
    kind = "native_llama_kv"
    template = "fixture-template"

    def __init__(self, config):
        super().__init__(config, b"a")
        self.fingerprint["model_sha256"] = sha256_file(Path(config.model_path))

    def save(self, directory):
        super().save(directory)
        (directory / "state.bin").write_bytes(b"fixture only")
        (directory / "logits.npy").write_bytes(b"fixture only")


class InitialContextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        model = self.root / "fixture.gguf"
        model.write_bytes(b"not a native model; orchestration fixture only")
        self.config = Config(model_path=str(model), n_ctx=24576, turnover_reserve=1536,
                             preparation_tokens=1, clock_interval_seconds=0)
        self.props = {"model_path": str(model), "build_info": SOURCE_BUILD, "chat_template": TextFixture.template,
            "default_generation_settings": {"params": {**DISABLED, "samplers": CHAIN, "seed": 42,
                "temperature": 0.7, "top_k": 20, "top_p": .8, "min_p": 0, "repeat_penalty": 1,
                "repeat_last_n": 64}}}
        self.request = {"model": "fixture", "messages": [{"role": "system", "content": "ORIGINAL SYSTEM"},
            {"role": "user", "content": "[CONVERSATION SUMMARY]\nsaved summary"},
            {"role": "assistant", "content": '<dmn_action>{"op":"send_message","content":"never execute this history"}</dmn_action>'}]}
        self.input = self.root / "request.json"
        self.input.write_text(json.dumps(self.request))
        self.prompt = "BOS\nSYSTEM ORIGINAL\n" + "archived conversation " * 270 + self.request["messages"][-1]["content"] + "\nASSISTANT\n"
        self.routes = []

    def tearDown(self):
        self.temp.cleanup()

    def api(self, url, route, body=None):
        self.routes.append(route)
        if route == "/props":
            return self.props
        if route == "/apply-template":
            self.assertEqual(body, self.request)
            return {"prompt": self.prompt}
        if route == "/tokenize":
            self.assertTrue(body["add_special"] and body["parse_special"])
            return {"tokens": list(map(ord, body["content"]))}
        if route == "/v1/responses/input_tokens":
            return {"input_tokens": len(self.prompt)}
        self.fail("inference endpoint must never be called")

    def bundle(self):
        folder = self.root / "bundle"
        prepare_initial_context(self.input, folder, self.config, "http://127.0.0.1:9000", 20, self.api)
        return folder, Config.read(folder / "config.json")

    def test_exact_source_tokens_precede_contract_without_historical_actions(self):
        bundle, config = self.bundle()
        runtime = Runtime(self.root / "instance", config, TextFixture(config), initial_context=bundle)
        try:
            source = list(map(ord, self.prompt))
            self.assertEqual(runtime.backend.tokens[:len(source)], source)
            self.assertEqual(runtime.state["continuity"], "initial_context_reconstruction")
            self.assertEqual(runtime.store.messages(), [])
            self.assertEqual(runtime.state["generated_tokens"], 0)
            self.assertEqual(self.routes, ["/props", "/apply-template", "/tokenize"])
            self.assertEqual((runtime.root / "import/provider-request.json").read_bytes(), self.input.read_bytes())
            saved_tokens = runtime.backend.tokens.copy()
        finally:
            runtime.close()

        runtime = Runtime(self.root / "instance", config, TextFixture(config))
        try:
            self.assertEqual(runtime.backend.tokens[:len(saved_tokens)], saved_tokens)
            self.assertEqual(runtime.state["last_restore"]["prompt_tokens_reevaluated"], 0)
            self.assertEqual(runtime.store.messages(), [])
        finally:
            runtime.close()

    def test_staged_import_and_repeated_retirement_preserve_both_contract_and_agreement(self):
        from dmn.prompts import proposal
        bundle, config = self.bundle()
        r = Runtime(self.root / "staged", config, TextFixture(config), initial_context=bundle,
                    prepare_only=True, first_message="Discuss the environment before continuing.")
        try:
            self.assertEqual(r.state["mode"], "staged")
            self.assertEqual(r.state["generated_tokens"], 0)
            self.assertEqual(r.state["agreement"]["base"], [self.request["messages"][0]])
            original = r.backend.tokens[:20]
            region = r.state["protected_protocol"]
            contract = r.backend.tokens[region["start"]:region["end"]]
            # Mechanical protection fixture; generated approval is tested separately.
            value = proposal("A new approved behavioral agreement.", r.state["agreement"]["revision"], "model")
            r._apply_prompt_decision({"proposal": value, "decision": "accept", "tokens": r._adoption_tokens(value)},
                                     {"op": "prompt_decide", "ok": True})
            region = r.state["protected_agreement"]
            agreement = r.backend.tokens[region["start"]:region["end"]]
            for _ in range(12):
                r.state["mode"] = "sleeping"
                r._eval([120] * (config.n_ctx - config.turnover_reserve - len(r.backend.tokens)))
                r._ensure_space(512)
                self.assertEqual(r.backend.tokens[:20], original)
                region = r.state.get("protected_protocol", {"start": 20, "end": 20 + len(contract)})
                self.assertEqual(r.backend.tokens[region["start"]:region["end"]], contract)
                region = r.state["protected_agreement"]
                self.assertEqual(r.backend.tokens[region["start"]:region["end"]], agreement)
        finally:
            r.close()

    def test_retirement_preserves_contract_until_it_joins_prefix(self):
        bundle, config = self.bundle()
        runtime = Runtime(self.root / "instance", config, TextFixture(config), initial_context=bundle)
        try:
            region = runtime.state["protected_protocol"].copy()
            contract = runtime.backend.tokens[region["start"]:region["end"]]
            for _ in range(16):
                runtime.state["mode"] = "sleeping"
                target = config.n_ctx - config.turnover_reserve - 10
                runtime._eval([ord("x")] * (target - len(runtime.backend.tokens)))
                runtime._ensure_space(400)
                self.assertLessEqual(len(runtime.backend.tokens) + 400, config.n_ctx - config.turnover_reserve)
                region = runtime.state.get("protected_protocol")
                if region:
                    self.assertEqual(runtime.backend.tokens[region["start"]:region["end"]], contract)
                else:
                    self.assertEqual(runtime.backend.tokens[20:20 + len(contract)], contract)
                    break
            self.assertNotIn("protected_protocol", runtime.state)
            self.assertEqual(runtime.state["keep_prefix"], 20 + len(contract))
            runtime._consolidate(400)
            self.assertEqual(runtime.backend.tokens[20:20 + len(contract)], contract)
        finally:
            runtime.close()

    def test_corrupt_tokens_and_sampler_mismatch_are_rejected_before_eval(self):
        bundle, config = self.bundle()
        wrong = dataclasses.replace(config, temperature=.2)
        backend = TextFixture(wrong)
        with self.assertRaisesRegex(ValueError, "sampler differs"):
            Runtime(self.root / "bad-sampler", wrong, backend, initial_context=bundle)
        self.assertEqual(backend.tokens, [])
        (bundle / "tokens.json").write_text("[1]")
        backend = TextFixture(config)
        with self.assertRaisesRegex(ValueError, "integrity"):
            Runtime(self.root / "corrupt", config, backend, initial_context=bundle)
        self.assertEqual(backend.tokens, [])

    def test_preparation_actions_leave_notice_room_at_narrow_contract_boundary(self):
        config = dataclasses.replace(self.config, preparation_tokens=768)
        bundle = self.root / "narrow-bundle"
        # A one-token gap must not be mistaken for a large reclaimable span.
        prepare_initial_context(self.input, bundle, config, "http://127.0.0.1:9000",
                                len(self.prompt) - 1, self.api)
        config = Config.read(bundle / "config.json")
        backend = TextFixture(config)
        backend.script = b'<dmn_action>{"op":"clock"}</dmn_action>'
        runtime = Runtime(self.root / "narrow-instance", config, backend,
                          initial_context=bundle, now=lambda: 1800000000.0)
        try:
            span = runtime.state["protected_protocol"].copy()
            contract = backend.tokens[span["start"]:span["end"]]
            runtime._eval([ord("x")] * (config.n_ctx - config.turnover_reserve - len(backend.tokens)))
            # Real action results consume reserve during preparation. Before
            # the fix, the post-retirement notice overflowed the hard context.
            runtime._ensure_space(512)
            self.assertLessEqual(len(backend.tokens) + 512, config.n_ctx - config.turnover_reserve)
            self.assertNotIn("protected_protocol", runtime.state)
            self.assertEqual(backend.tokens[len(self.prompt)-1:len(self.prompt)-1+len(contract)], contract)
            self.assertGreaterEqual(runtime.state["context_retirements"], 1)
            self.assertTrue(runtime.state["last_context_retirement"]["additional_ranges"])
            self.assertEqual(runtime.store.messages(), [])
        finally:
            runtime.close()

    def test_tokenizer_disagreement_and_overflow_do_not_truncate(self):
        bundle, config = self.bundle()
        backend = TextFixture(config)
        backend.tokenize = lambda *_args, **_kwargs: [7]
        with self.assertRaisesRegex(ValueError, "tokenizers disagree"):
            Runtime(self.root / "different-tokens", config, backend, initial_context=bundle)
        self.assertEqual(backend.tokens, [])
        small = dataclasses.replace(config, n_ctx=8192)
        backend = TextFixture(small)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            Runtime(self.root / "overflow", small, backend, initial_context=bundle)
        self.assertEqual(backend.tokens, [])

    def test_unsupported_features_fail_instead_of_silent_mapping(self):
        for extra in ({"tools": "invalid"}, {"logit_bias": {"1": 2}}, {"reasoning_budget": 100},
                      {"response_format": {"type": "json_object"}}, {"n": 2}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_request({**self.request, **extra})
        with self.assertRaisesRegex(ValueError, "multimodal"):
            validate_request({**self.request, "messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
        with self.assertRaisesRegex(ValueError, "active sampler"):
            resolve_sampler({"presence_penalty": 1}, self.props, self.config)
        with self.assertRaisesRegex(ValueError, "sampler order"):
            resolve_sampler({"samplers": list(reversed(CHAIN))}, self.props, self.config)

    def test_source_defaults_and_request_overrides_are_explicit(self):
        config, evidence = resolve_sampler({"temperature": .25, "seed": -1}, self.props, self.config)
        self.assertEqual(config.temperature, .25)
        self.assertEqual(config.repeat_penalty, 1)
        self.assertEqual(config.sampler_order, "llama_default_v1")
        self.assertFalse(evidence["original_rng_restored"])
        self.assertNotEqual(config.seed, -1)

    def test_historical_frontend_tool_attempt_identifies_the_active_actions(self):
        bundle, config = self.bundle()
        runtime = Runtime(self.root / "tool-scope", config, TextFixture(config), initial_context=bundle)
        try:
            result, effect = runtime._plan_action({"op": "search_notes", "query": "fixture"}, [])
            self.assertFalse(result["ok"])
            self.assertIsNone(effect)
            self.assertIn("historical", result["error"])
            self.assertIn("memory_read", result["available_operations"])
            self.assertNotIn("search_notes", result["available_operations"])
            self.assertEqual(runtime.store.messages(), [])
        finally:
            runtime.close()

    def test_responses_bundle_preserves_wire_body_and_checks_native_count(self):
        from dmn.responses import responses_to_chat
        wire = {"model": "fixture", "input": [{"role": "user", "content": "question"}],
                "instructions": "original", "temperature": .7}
        self.input.write_text(json.dumps(wire))
        self.request = responses_to_chat(wire)
        bundle, _ = self.bundle()
        self.assertEqual(json.loads((bundle / "provider-request.json").read_text()), wire)
        self.assertEqual(json.loads((bundle / "normalized-chat-request.json").read_text()), self.request)
        evidence = json.loads((bundle / "conversion.json").read_text())
        self.assertTrue(evidence["native_token_count_matched"])
        self.assertIn("/v1/responses/input_tokens", self.routes)


if __name__ == "__main__":
    unittest.main()
