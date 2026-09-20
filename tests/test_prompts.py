import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime
from tests.test_runtime import frames


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config(backend="demo", n_ctx=24576, clock_interval_seconds=0,
                             checkpoint_policy="effects", checkpoint_tokens=100000, preparation_tokens=1)
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"thought "))

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def generate(self, action):
        raw = frames(action)
        self.runtime.backend.script, self.runtime.backend.index = raw, 0
        for _ in raw:
            self.runtime.tick()

    def propose(self, text="I choose this behavioral agreement."):
        r = self.runtime
        value = r.propose_prompt(text, r.state["agreement"]["revision"])
        r.tick()
        return value["revision"], text, r.state["agreement"]["revision"]

    def read(self, revision, text):
        while self.runtime._prompt_reads.get(revision, 0) < len(text):
            self.generate({"op": "prompt_read", "revision": revision,
                           "offset": self.runtime._prompt_reads.get(revision, 0), "limit": 2000})

    def decide(self, revision, base, decision="accept"):
        self.generate({"op": "prompt_decide", "revision": revision, "base_revision": base, "decision": decision})

    def reopen(self):
        self.runtime.close()
        self.runtime = Runtime(self.root, self.config, DemoBackend(self.config, b"thought "))

    def test_exact_approval_appends_and_survives_restore_and_retirement(self):
        revision, text, base = self.propose("New agreed wording. " * 50)
        self.read(revision, text)
        prefix = self.runtime.backend.tokens.copy()
        self.decide(revision, base)
        r = self.runtime
        self.assertEqual(r.state["agreement"]["text"], text)
        self.assertEqual(r.backend.tokens[:len(prefix)], prefix)
        span = r.state["protected_agreement"]
        protected = r.backend.tokens[span["start"]:span["end"]]
        self.reopen()
        r = self.runtime
        self.assertEqual(r.state["agreement"]["revision"], revision)
        for _ in range(4):
            r.state["mode"] = "sleeping"
            r._eval([120] * (r.backend.n_ctx - r.config.turnover_reserve - len(r.backend.tokens)))
            r._ensure_space(200)
            span = r.state["protected_agreement"]
            self.assertEqual(r.backend.tokens[span["start"]:span["end"]], protected)
        self.assertNotIn(revision, r._prompt_reads)

    def test_edit_stale_decline_defer_unread_and_external_text_do_not_activate(self):
        revision, text, base = self.propose()
        self.decide(revision, base)
        self.assertEqual(self.runtime.state["agreement"]["revision"], base)
        self.runtime.enqueue(frames({"op": "prompt_decide", "revision": revision,
                                     "base_revision": base, "decision": "accept"}).decode())
        self.runtime.tick()
        self.decide(revision, base, "defer")
        self.assertEqual(self.runtime.state["prompt_decisions"][revision], "deferred")
        self.read(revision, text)
        self.decide(revision, "stale")
        self.assertEqual(self.runtime.state["agreement"]["revision"], base)
        self.decide(revision, base, "decline")
        self.decide(revision, base)
        self.assertEqual(self.runtime.state["agreement"]["revision"], base)
        self.read(revision, text)
        self.decide(revision, base)
        self.assertEqual(self.runtime.state["agreement"]["revision"], revision)

    def test_new_revision_unpins_old_span_but_keeps_durable_history(self):
        first, text, base = self.propose()
        self.read(first, text)
        self.decide(first, base)
        second, text, base = self.propose("A different agreement.")
        self.read(second, text)
        self.decide(second, base)
        self.assertEqual(self.runtime.state["prompt_decisions"][first], "superseded")
        self.assertEqual(len(self.runtime.prompt_status()["proposals"]), 2)
        with self.runtime.store.mutex:
            self.assertEqual(self.runtime.store.db.execute("SELECT COUNT(*) FROM records WHERE kind='prompt_decision'").fetchone()[0], 2)

    def test_failed_checkpoint_never_publishes_new_active_agreement(self):
        revision, text, base = self.propose()
        self.read(revision, text)
        self.runtime.checkpoint()
        with mock.patch.object(self.runtime.backend, "save", side_effect=OSError("save failed")):
            with self.assertRaises(OSError):
                self.decide(revision, base)
        self.assertEqual(self.runtime.prompt_status()["active"]["revision"], base)
        self.reopen()
        self.assertEqual(self.runtime.state["agreement"]["revision"], base)
        self.assertEqual(self.runtime._prompt_reads, {})

    def test_model_authored_proposal_is_not_implicit_acceptance(self):
        base = self.runtime.state["agreement"]["revision"]
        self.generate({"op": "prompt_propose", "text": "My proposal", "base_revision": base})
        self.runtime.tick()
        item = self.runtime.prompt_status()["proposals"][0]
        self.assertEqual(item["author"], "model")
        self.assertEqual(self.runtime.state["agreement"]["revision"], base)
        self.read(item["revision"], item["text"])
        self.decide(item["revision"], base)
        self.assertEqual(self.runtime.state["agreement"]["revision"], item["revision"])

    def test_oversize_proposal_rejected_without_shortening(self):
        with self.assertRaises(ValueError):
            self.propose("x" * (self.config.max_event_bytes + 1))

    def test_truncated_page_after_partial_frame_cancellation_is_not_review(self):
        revision, text, base = self.propose("word " * 200)
        piece = frames({"op": "prompt_read", "revision": revision, "limit": 2000}) + b'<dmn_action>{"op":'
        # One token can finish a read and begin a second, incomplete frame.
        # Its cancellation notice can force an otherwise fitting page to truncate.
        with mock.patch.object(self.runtime.backend, "piece", return_value=piece):
            self.runtime.tick()
        self.assertEqual(self.runtime._prompt_reads.get(revision, 0), 0)
        self.decide(revision, base)
        self.assertEqual(self.runtime.state["agreement"]["revision"], base)
