import subprocess
import unittest
from unittest.mock import Mock, patch

from scripts.openwebui_sandbox import stop_children


class SandboxShutdownTest(unittest.TestCase):
    def test_timeout_keeps_native_owner_alive(self):
        native, webui = Mock(pid=42), Mock()
        native.poll.return_value = webui.poll.return_value = None
        native.wait.side_effect = subprocess.TimeoutExpired("dmn", 600)
        with patch("scripts.openwebui_sandbox.request", return_value={"process_id": 42}), \
             patch("scripts.openwebui_sandbox.build_opener"), patch("builtins.print") as output:
            stop_children([native, webui], 8766)
        native.terminate.assert_not_called()
        native.kill.assert_not_called()
        webui.terminate.assert_called_once()
        self.assertIn("left running", output.call_args.args[0])

    def test_different_process_on_port_receives_no_shutdown(self):
        native = Mock(pid=42)
        native.poll.return_value = None
        with patch("scripts.openwebui_sandbox.request", return_value={"process_id": 43}), \
             patch("scripts.openwebui_sandbox.build_opener") as opener, patch("builtins.print"):
            stop_children([native], 8766)
        opener.assert_not_called()
        native.terminate.assert_not_called()

    def test_successful_shutdown_stops_frontend_after_native_exits(self):
        native, webui = Mock(pid=42), Mock()
        native.poll.return_value = webui.poll.return_value = None
        with patch("scripts.openwebui_sandbox.request", return_value={"process_id": 42}), \
             patch("scripts.openwebui_sandbox.build_opener"), patch("builtins.print") as output:
            stop_children([native, webui], 8766, 30)
        native.wait.assert_called_once_with(timeout=30)
        native.terminate.assert_not_called()
        webui.terminate.assert_called_once()
        output.assert_not_called()
