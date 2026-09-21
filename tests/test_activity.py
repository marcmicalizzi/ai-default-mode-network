import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.protocol import ACTIVITY_CONTRACT
from tests.test_runtime import FakeClock, frames


class ActivityFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "instance"
        self.wall, self.mono = FakeClock(), FakeClock()
        self.config = Config(backend="demo", n_ctx=32768, clock_interval_seconds=1,
                             checkpoint_policy="effects", checkpoint_tokens=4096,
                             preparation_tokens=8, idle_enabled=True,
                             idle_max_burst_tokens=4, idle_min_interval_seconds=10)
        self.opened = []

    def tearDown(self):
        for r in self.opened:
            r.close()
        self.temp.cleanup()

    def create(self, script, config=None):
        c = config or self.config
        r = Runtime(self.root, c, DemoBackend(c, script), now=self.wall, monotonic=self.mono)
        self.opened.append(r)
        return r

    def reopen(self, r, script, config=None):
        r.close()
        self.opened.remove(r)
        return self.create(script, config)

    def idle(self, suffix=b"abcdefghijk"):
        enter = frames({"op": "activity", "mode": "idle"})
        script = enter + suffix
        r = self.create(script)
        while r.pacer.profile["mode"] != "idle":
            r.tick()
        return r, script


class ActivityTest(ActivityFixture):
    def test_idle_pauses_retain_context_and_parser_without_saves(self):
        r, _ = self.idle(b"\n<dmn_action>{\"op\":\"send_message\",\"content\":\"later\"}</dmn_action>\n")
        for _ in range(4):
            r.tick()
        tokens, parser, saved = r.backend.tokens.copy(), r.parser.state(), r.store.latest()
        self.assertTrue(r.parser.pending)
        self.wall.time += 1000
        for _ in range(10):
            self.assertFalse(r.tick())
        self.assertEqual(r.backend.tokens, tokens)
        self.assertEqual(r.parser.state(), parser)
        self.assertEqual(r.store.latest(), saved)
        self.assertEqual(r.store.messages(), [])
        self.assertEqual(r.status()["activity"]["idle_generated_tokens"], 4)
        self.mono.time += 1000
        for _ in range(4):
            self.assertTrue(r.tick())
        self.assertFalse(r.tick())  # No catch-up credit.

    def test_idle_checkpoint_deadline_does_not_generate(self):
        c = dataclasses.replace(self.config, checkpoint_interval_seconds=5)
        script = frames({"op": "activity", "mode": "idle"}) + b"abcdef"
        r = self.create(script, c)
        while r.pacer.profile["mode"] != "idle":
            r.tick()
        for _ in range(4):
            r.tick()
        tokens = r.backend.tokens.copy()
        self.mono.time += 5
        self.assertFalse(r.tick())
        self.assertEqual(r.state["checkpoint_reason"], "time_limit")
        self.assertEqual(r.backend.tokens, tokens)
        self.assertFalse(r.status()["checkpoint"]["dirty"])

    def test_input_preempts_wait_and_selects_focus(self):
        r, _ = self.idle()
        for _ in range(4):
            r.tick()
        r.enqueue("new input")
        r.tick()
        self.assertEqual(r.state["event_cursor"], 1)
        self.assertEqual(r.pacer.profile["mode"], "focus")
        generated = r.state["generated_tokens"]
        r.tick()
        self.assertEqual(r.state["generated_tokens"], generated + 1)

    def test_restart_rebases_idle_without_granting_credit(self):
        r, script = self.idle()
        prefix = r.backend.tokens[:r.state["keep_prefix"]]
        r = self.reopen(r, script)
        self.assertFalse(r.tick())
        self.assertEqual(r.backend.tokens[:len(prefix)], prefix)
        self.assertEqual(r.state["last_restore"]["prompt_tokens_reevaluated"], 0)
        self.mono.time += 10
        self.assertTrue(r.tick())

    def test_eog_sleep_overrides_idle_and_survives_restart(self):
        r, script = self.idle()
        r.backend.is_eog = lambda _: True
        r.tick()
        self.assertEqual(r.state["mode"], "sleeping")
        r = self.reopen(r, script)
        before = r.state["generated_tokens"]
        self.mono.time += 1000
        self.wall.time += 1000
        self.assertFalse(r.tick())
        self.assertEqual(r.state["generated_tokens"], before)

    def test_suspend_resume_preserves_profile_and_stops_generation(self):
        r, _ = self.idle()
        r.control("emergency_suspend", preparation_seconds=0)
        r.tick()
        self.assertEqual(r.state["mode"], "suspended")
        before = r.state["generated_tokens"]
        self.mono.time += 1000
        self.assertFalse(r.tick())
        self.assertEqual(r.state["generated_tokens"], before)
        r.control("resume")
        r.tick()
        self.assertEqual(r.pacer.profile["mode"], "idle")

    def test_contract_is_complete_and_protected_through_retirement(self):
        r, _ = self.idle()
        span = r.state["protected_activity"]
        contract = r.backend.tokens[span["start"]:span["end"]]
        self.assertIn(json.dumps(ACTIVITY_CONTRACT)[1:-1], "".join(map(chr, contract)))
        r._eval([ord("x")] * 16000)
        # Keep preparation deterministic without model actions.
        r._consolidate(1)
        span = r.state["protected_activity"]
        self.assertEqual(r.backend.tokens[span["start"]:span["end"]], contract)

    def test_invalid_or_disabled_requests_do_not_change_profile(self):
        r = self.create(b"x")
        for action in ({"mode": "idle", "burst_tokens": True}, {"mode": "idle", "burst_tokens": 5},
                       {"mode": "idle", "interval_seconds": 9}, {"mode": "idle", "interval_seconds": float("nan")},
                       {"mode": "focus", "burst_tokens": 1}, {"mode": "other"}):
            result, effect = r._plan_action({"op": "activity", **action}, [])
            self.assertFalse(result["ok"])
            self.assertIsNone(effect)
        r.config = dataclasses.replace(self.config, idle_enabled=False)
        self.assertFalse(r._plan_action({"op": "activity", "mode": "idle"}, [])[0]["ok"])
        self.assertEqual(r.pacer.profile["mode"], "focus")

    def test_failed_profile_checkpoint_does_not_adopt_idle(self):
        script = frames({"op": "activity", "mode": "idle"})
        r = self.create(script)
        r.backend.save = lambda _: (_ for _ in ()).throw(OSError("save failed"))
        with self.assertRaises(OSError):
            for _ in script:
                r.tick()
        self.assertEqual(r.pacer.profile["mode"], "focus")
        self.assertNotIn("activity", r.state)

    def test_config_rejects_invalid_limits(self):
        for values in ({"idle_enabled": 1}, {"idle_max_burst_tokens": True}, {"idle_max_burst_tokens": 0},
                       {"idle_min_interval_seconds": 0}, {"idle_min_interval_seconds": float("inf")},
                       {"sleep_checkpoint_min_interval_seconds": -1},
                       {"checkpoint_policy": "all_actions", "sleep_checkpoint_min_interval_seconds": 10}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                dataclasses.replace(self.config, **values)
