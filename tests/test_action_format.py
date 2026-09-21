import json
import tempfile
import unittest
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.protocol import ActionParser
from dmn.runtime import Runtime
from tests.test_runtime import frames


class ActionFormatTest(unittest.TestCase):
    def test_missing_action_fields_identify_what_to_correct(self):
        config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(Path(tmp), config, DemoBackend(config, b'quiet'))
            try:
                for action, missing in (({"op": "send_message"}, "content"),
                                        ({"op": "memory_write", "content": "text"}, "path")):
                    result, effect = runtime._plan_action(action, [])
                    self.assertFalse(result["ok"])
                    self.assertIsNone(effect)
                    self.assertIn(f"Missing required field '{missing}'", result["error"])
                    self.assertIn("Nothing was executed or sent", result["error"])
                self.assertEqual(runtime.store.messages(), [])
            finally:
                runtime.close()

    def test_literal_whitespace_preserved_at_every_byte_boundary(self):
        content = 'First café line.\n\nSecond 🌧️ line.\r\n\tIndented.'
        raw = ('\n<dmn_action>{"op":"send_message","content":"' + content + '"}</dmn_action>').encode()
        for split in range(len(raw) + 1):
            parser = ActionParser(8192)
            result = parser.feed(raw[:split])
            parser = ActionParser(8192, parser.state())
            result += parser.feed(raw[split:])
            self.assertEqual(result, [{"op": "send_message", "content": content}])

    def test_escaped_quotes_and_backslashes_keep_their_meaning(self):
        content = 'A "quoted" phrase and C:\\folder.\nSecond line.'
        body = json.dumps({"op": "send_message", "content": content}).replace('\\n', '\n')
        result = ActionParser(8192).feed(('<dmn_action>' + body + '</dmn_action>').encode())
        self.assertEqual(result[0]["content"], content)

    def test_no_structural_repairs_other_controls_or_inline_execution(self):
        malformed = [b'{"op":"send_message","content":"unclosed}',
                     b'{"op":"send_message","content":"a" "extra"}',
                     b'{"op":"send_message","content":"x\x00y"}',
                     b'{"op":"send_message","content":"x\x1by"}',
                     b'{"op":"send_message","content":"a\\\nb"}',
                     b'{"op":"send_message","content":"a\nb",}',
                     b'{"op":"send_message","content":"a\nb"}{"op":"sleep"}']
        for body in malformed:
            result = ActionParser(8192).feed(b'<dmn_action>' + body + b'</dmn_action>')
            self.assertEqual(result[0]["op"], "__invalid__")
            self.assertEqual(result[0]["attempted_op"], "send_message")
        parser = ActionParser(8192)
        self.assertEqual(parser.feed(b'private text <dmn_action>{"op":"sleep"}</dmn_action>'), [])

    def test_multiline_message_commits_once_and_rejections_have_private_content_free_status(self):
        content = 'A deliberately multiline message.\n\nA second paragraph.'
        raw = ('\n<dmn_action>{"op":"send_message","content":"' + content + '"}</dmn_action>\n').encode()
        secret = 'PRIVATE_PAYLOAD_SHOULD_NOT_APPEAR_IN_STATUS'
        raw += ('<dmn_action>{"op":"send_message","content":"' + secret + '",}</dmn_action>\n').encode()
        raw += frames({"op": "sleep"})
        config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0, checkpoint_policy="effects")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = Runtime(root, config, DemoBackend(config, raw))
            try:
                # Imported instances previously replaced parse errors with an
                # unrelated unavailable-operation warning.
                runtime.state["initial_context"] = {"synthetic": True}
                for _ in raw:
                    runtime.tick()
                self.assertEqual([m["content"] for m in runtime.store.messages()], [content])
                diagnostics = runtime.status()["action_diagnostics"]
                self.assertEqual(diagnostics["rejected_actions"], 1)
                self.assertEqual(diagnostics["rejected_message_attempts"], 1)
                self.assertNotIn(secret, json.dumps(runtime.status()))
                injected = ''.join(map(chr, runtime.backend.tokens))
                self.assertIn('Invalid action format; nothing was executed or sent.', injected)
                self.assertNotIn('Unavailable operation.', injected)
            finally:
                runtime.close()
            runtime = Runtime(root, config, DemoBackend(config, raw))
            try:
                self.assertEqual(len(runtime.store.messages()), 1)
                self.assertEqual(runtime.status()["action_diagnostics"]["rejected_actions"], 1)
                for _ in range(20):
                    runtime.tick()
                self.assertEqual(len(runtime.store.messages()), 1)
            finally:
                runtime.close()

    def test_partial_action_is_cancelled_and_diagnostic_categories_are_fixed(self):
        raw = b'\n<dmn_action>{"op":"send_message","content":"unfinished'
        config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(Path(tmp), config, DemoBackend(config, raw))
            try:
                for _ in raw:
                    runtime.tick()
                runtime.enqueue('A new event')
                runtime.tick()
                self.assertEqual(runtime.status()["action_diagnostics"]["interrupted_frames"], 1)
                self.assertEqual(runtime.store.messages(), [])
                script = frames({"op": "__invalid__", "error_code": "PRIVATE_VALUE", "error": "error"}, {"op":"sleep"})
                runtime.backend.script, runtime.backend.index = script, 0
                for _ in script:
                    runtime.tick()
                self.assertEqual(runtime.status()["action_diagnostics"]["last_problem"]["category"], "action_rejected")
                self.assertNotIn('PRIVATE_VALUE', json.dumps(runtime.status()))
            finally:
                runtime.close()

    def test_legacy_resume_announces_fix_without_resending_historical_actions(self):
        config = Config(backend="demo", n_ctx=16384, clock_interval_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = Runtime(root, config, DemoBackend(config, b'quiet'))
            del runtime.state["action_format_protocol"]
            runtime.checkpoint()
            prefix = runtime.backend.tokens.copy()
            runtime.close()
            runtime = Runtime(root, config, DemoBackend(config, b'quiet'))
            try:
                self.assertEqual(runtime.backend.tokens[:len(prefix)], prefix)
                self.assertIn('does not\\nresend earlier rejected messages', ''.join(map(chr, runtime.backend.tokens[len(prefix):])))
                self.assertEqual(runtime.store.messages(), [])
                self.assertEqual(runtime.state["action_format_protocol"], "literal_whitespace_v1")
            finally:
                runtime.close()
