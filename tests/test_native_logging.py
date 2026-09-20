import io
import threading
import unittest
from unittest.mock import patch

from dmn.native_logging import NativeLogFilter


class NativeLoggingTest(unittest.TestCase):
    def test_filters_routine_messages_but_keeps_warning_error_continuations(self):
        handler = NativeLogFilter("warning")
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            for level, text in ((4, b"graph reused"), (5, b" debug continuation"),
                                (1, b"info"), (2, b"warning"), (5, b" details\n"),
                                (3, b"error"), (5, b" more\n")):
                handler(level, text, None)
        self.assertEqual(stream.getvalue(), "warning details\nerror more\n")

    def test_continuation_severity_is_thread_local(self):
        handler = NativeLogFilter("warning")
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            handler(2, b"warning", None)
            worker = threading.Thread(target=lambda: handler(4, b"debug", None))
            worker.start()
            worker.join()
            handler(5, b" details", None)
        self.assertEqual(stream.getvalue(), "warning details")

    def test_debug_opt_in_and_closed_output(self):
        stream = io.StringIO()
        with patch("sys.stderr", stream):
            NativeLogFilter("debug")(4, b"graph reused\n", None)
            self.assertEqual(stream.getvalue(), "graph reused\n")
            stream.close()
            NativeLogFilter("warning")(3, b"failure", None)
        with self.assertRaises(ValueError):
            NativeLogFilter("invalid")
