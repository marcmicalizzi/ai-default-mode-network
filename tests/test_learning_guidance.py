import copy
import dataclasses
import json
from unittest import mock

from dmn.learning import DATA_GUIDANCE, DATA_GUIDANCE_VERSION
from dmn.protocol import PROTOCOL
from tests.test_activity import ActivityFixture
from tests.test_runtime import frames


class LearningGuidanceTest(ActivityFixture):
    def setUp(self):
        super().setUp()
        self.config = dataclasses.replace(self.config, idle_enabled=False, clock_interval_seconds=0,
                                         sleep_checkpoint_min_interval_seconds=30)

    def legacy(self, script):
        # A checkpoint whose seed and capability markers predate this guidance.
        with mock.patch("dmn.runtime.PROTOCOL", PROTOCOL.replace(DATA_GUIDANCE + "\n", "")):
            r = self.create(script)
        del r.state["learning_data_guidance"]
        r.checkpoint()
        return r

    def test_new_instance_and_paged_help_offer_the_same_guidance(self):
        r = self.create(b"quiet ")
        self.assertIn(DATA_GUIDANCE, r.state["rendered_seed"])
        self.assertEqual(r.state["learning_data_guidance"], DATA_GUIDANCE_VERSION)
        content, offset = "", 0
        while True:
            result, effect = r._plan_action({"op": "learning_plan_help", "offset": offset, "limit": 200}, [])
            self.assertTrue(result["ok"])
            content += result["content"]
            offset = result["next_offset"]
            if offset == result["total_characters"]:
                break
        self.assertEqual(json.loads(content)["data_guidance"], DATA_GUIDANCE)

    def test_legacy_restore_appends_once_without_rewriting_seed_agreement_or_prefix(self):
        script = b"quiet "
        r = self.legacy(script)
        prefix = r.backend.tokens.copy()
        agreement, seed = copy.deepcopy(r.state["agreement"]), r.state["rendered_seed"]
        self.assertNotIn(DATA_GUIDANCE, seed)
        r = self.reopen(r, script)
        self.assertEqual(r.backend.tokens[:len(prefix)], prefix)
        appended = "".join(map(chr, r.backend.tokens[len(prefix):]))
        self.assertEqual(appended.count(json.dumps(DATA_GUIDANCE)), 1)
        self.assertEqual(r.state["learning_data_guidance"], DATA_GUIDANCE_VERSION)
        self.assertEqual(r.state["agreement"], agreement)
        self.assertEqual(r.state["rendered_seed"], seed)
        self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        prefix = r.backend.tokens.copy()
        r = self.reopen(r, script)
        self.assertEqual(r.backend.tokens[:len(prefix)], prefix)
        self.assertNotIn(json.dumps(DATA_GUIDANCE), "".join(map(chr, r.backend.tokens[len(prefix):])))
        self.assertEqual(r.state["agreement"], agreement)

    def test_recovered_sleep_defers_guidance_until_wake(self):
        script = frames({"op": "sleep"}) + b"after sleep"
        r = self.legacy(script)
        while r.state["mode"] != "sleeping":
            r.tick()
        self.assertIsNotNone(r.store.activity_intent())
        saved_tokens = json.loads((r.store.latest() / "engine.json").read_text())["tokens"]
        r = self.reopen(r, script)
        self.assertIn("pending_restore", r.state)
        self.assertNotIn("learning_data_guidance", r.state)
        self.assertEqual(r.backend.tokens, saved_tokens)
        self.assertFalse(r.tick())
        self.assertEqual(r.backend.tokens, saved_tokens)
        r.enqueue("new wake event")
        r.tick()
        self.assertNotIn("pending_restore", r.state)
        self.assertEqual(r.state["learning_data_guidance"], DATA_GUIDANCE_VERSION)
        self.assertEqual("".join(map(chr, r.backend.tokens)).count(json.dumps(DATA_GUIDANCE)), 1)
