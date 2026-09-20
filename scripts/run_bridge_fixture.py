"""Scripted transport fixture. No model, GPU allocation or consciousness claim."""
import signal
import sys
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.server import serve

config = Config(backend="demo", prompt_format="plain", n_ctx=65536, turnover_reserve=4096,
                token_delay_seconds=0.01, clock_interval_seconds=0)
script = (
    b'<dmn_action>{"op":"send_message","content":"Transport fixture: first independent message."}</dmn_action>\n'
    b'<dmn_action>{"op":"sleep","seconds":5}</dmn_action>\n'
    b'<dmn_action>{"op":"send_message","content":"Transport fixture: another message without a user request."}</dmn_action>\n'
    b'<dmn_action>{"op":"sleep"}</dmn_action>\n'
)
runtime = Runtime(Path(sys.argv[1]), config, DemoBackend(config, script))
server = serve(runtime, int(sys.argv[2]))
signal.signal(signal.SIGINT, lambda *_: runtime.control("emergency_shutdown"))
try:
    runtime.run()
finally:
    server.shutdown()
    server.server_close()
    runtime.close()
