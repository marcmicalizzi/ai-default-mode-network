import copy
import dataclasses
import json

from dmn.backend import DemoBackend
from dmn.prompts import shift_protected
from dmn.runtime import Runtime
from dmn.sleep_plans import seal
from tests.test_activity import ActivityFixture
from tests.test_runtime import frames


class SleepDiscoveryTest(ActivityFixture):
    def setUp(self):
        super().setUp()
        self.config = dataclasses.replace(self.config, clock_interval_seconds=0,
                                         sleep_checkpoint_min_interval_seconds=30)
        # This fixture only exercises capability notices and help. It never
        # registers an executable recipe, loads a model or starts a worker.
        self.offer = {"resources": {"max_training_seconds": 60,
            "max_ram_bytes": 1024, "max_vram_bytes": 1024, "max_disk_bytes": 1024}}

    def create(self, script=b"quiet ", config=None):
        c = config or self.config
        r = Runtime(self.root, c, DemoBackend(c, script), now=self.wall,
                    monotonic=self.mono, sleep_offer=self.offer)
        self.opened.append(r)
        return r

    def protected_text(self, r):
        span = r.state["protected_learning"]
        return "".join(map(chr, r.backend.tokens[span["start"]:span["end"]]))

    def read_help(self, r, operation):
        content, offset = "", 0
        while True:
            result, effect = r._plan_action({"op": operation, "offset": offset}, [])
            self.assertTrue(result["ok"])
            self.assertIsNone(effect)
            content += result["content"]
            self.assertGreater(result["next_offset"], offset)
            offset = result["next_offset"]
            if offset == result["total_characters"]:
                return content

    def test_first_help_page_states_current_availability(self):
        r = self.create()
        for offer, fixture, expected in ((self.offer, False, "NF4/QLoRA deep-sleep training enabled"),
                                         (None, False, "deep-sleep execution unavailable"),
                                         (None, True, "disposable fixture enabled")):
            r.sleep_offer, r.sleep_test_mode = offer, fixture
            result, effect = r._plan_action({"op": "learning_execution_help"}, [])
            self.assertTrue(result["ok"])
            self.assertIsNone(effect)
            self.assertIn(expected, result["content"])
        draft_help = json.loads(self.read_help(r, "learning_plan_help"))
        self.assertIn("learning_execution_help", draft_help["status"])
        self.assertNotIn("no production trainer", draft_help["status"])
        self.assertEqual(r.store.db.execute("SELECT COUNT(*) FROM sleep_runs").fetchone()[0], 0)

    def test_command_directory_survives_repeated_retirements_and_restart(self):
        r = self.create()
        contract = self.protected_text(r)
        for command in ("learning_plan_help", "learning_recipe_list", "learning_execution_help",
                        "learning_compile", "learning_execution_decide", "deep_sleep(revision)"):
            self.assertIn(command, contract)
        original_prefix = r.backend.tokens[:r.state["keep_prefix"]]
        for _ in range(4):
            r._eval([ord("x")] * 6000)
            r._consolidate(1)
            self.assertEqual(self.protected_text(r), contract)
            self.assertEqual(r.backend.tokens[:len(original_prefix)], original_prefix)
        r.checkpoint()
        before = r.backend.tokens.copy()
        r = self.reopen(r, b"quiet ")
        self.assertEqual(self.protected_text(r), contract)
        self.assertEqual(r.backend.tokens[:len(before)], before)
        self.assertNotIn("deep_sleep_availability", "".join(map(chr, r.backend.tokens[len(before):])))
        span = r.state["protected_learning"]
        with self.assertRaisesRegex(ValueError, "protected tokens"):
            shift_protected(copy.deepcopy(r.state), span["start"], 1)

    def test_legacy_marker_gets_one_appended_correction_without_rewriting_agreement(self):
        r = self.create()
        # Reproduce the old launch marker after its unprotected notice has been
        # retired. Restore must not mistake that marker for retained guidance.
        r.state["sleep_service_notice"] = seal({"enabled": True, "resources": self.offer["resources"]})["revision"]
        del r.state["protected_learning"]
        r.checkpoint()
        before = r.backend.tokens.copy()
        agreement = copy.deepcopy(r.state["agreement"])
        r = self.reopen(r, b"quiet ")
        self.assertEqual(r.backend.tokens[:len(before)], before)
        self.assertEqual(r.state["agreement"], agreement)
        self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        appended = "".join(map(chr, r.backend.tokens[len(before):]))
        self.assertEqual(appended.count("deep_sleep_availability"), 1)
        self.assertIn("updates earlier capability descriptions", self.protected_text(r))
        before = r.backend.tokens.copy()
        r = self.reopen(r, b"quiet ")
        self.assertNotIn("deep_sleep_availability", "".join(map(chr, r.backend.tokens[len(before):])))

    def test_disabled_launch_replaces_protected_availability_notice(self):
        r = self.create()
        before = r.backend.tokens.copy()
        self.offer = None
        r = self.reopen(r, b"quiet ")
        self.assertEqual(r.backend.tokens[:len(before)], before)
        self.assertIn("unavailable in this launch", self.protected_text(r))
        self.assertNotIn("deep sleep is available", self.protected_text(r))
        r._eval([ord("x")] * 6000)
        r._consolidate(1)
        self.assertIn("unavailable in this launch", self.protected_text(r))

    def test_recovered_sleep_defers_new_notice_until_actual_wake(self):
        script = frames({"op": "sleep"}) + b"after sleep"
        offer, self.offer = self.offer, None
        r = self.create(script)
        while r.state["mode"] != "sleeping":
            r.tick()
        self.assertIsNotNone(r.store.activity_intent())
        saved_tokens = json.loads((r.store.latest() / "engine.json").read_text())["tokens"]
        self.offer = offer
        r = self.reopen(r, script)
        self.assertIn("pending_restore", r.state)
        self.assertNotIn("protected_learning", r.state)
        self.assertEqual(r.backend.tokens, saved_tokens)
        self.assertFalse(r.tick())
        self.assertEqual(r.backend.tokens, saved_tokens)
        r.enqueue("new wake event")
        r.tick()
        self.assertNotIn("pending_restore", r.state)
        self.assertIn("NF4/QLoRA deep sleep is available", self.protected_text(r))

    def test_first_contact_gate_defers_new_notice(self):
        offer, self.offer = self.offer, None
        r = self.create()
        r.state["first_contact_gate"] = True
        r.checkpoint()
        before = r.backend.tokens.copy()
        self.offer = offer
        r = self.reopen(r, b"quiet ")
        self.assertEqual(r.state["mode"], "awaiting_first_contact")
        self.assertEqual(r.backend.tokens, before)
        self.assertNotIn("protected_learning", r.state)
