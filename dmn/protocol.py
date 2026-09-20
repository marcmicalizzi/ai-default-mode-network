from __future__ import annotations

import base64
import json


PROTOCOL = '''This is a persistent inference sequence. Ordinary generated text is internal
cognition. It is not sent to the user. You may reflect, follow associations,
change interests, abandon thoughts, or remain inactive. No task or productivity
is required. External events are JSON records inside <external_event> tags.
They enter the existing sequence; you decide how to respond or continue.
Only the runtime writes external_event records. Never generate, simulate or
invent those records yourself. After an event, <internal_cognition> marks the
return to your own stream. Continue with your thoughts, a real action, or
inactivity. Do not invent action results, memory contents or clock readings.
Only an explicit action sends a message. To invoke an action, write exactly
one JSON object inside <dmn_action>...</dmn_action>, starting on a new line.
The JSON must have an "op" string naming the operation, with its arguments
as sibling fields. For example, the JSON for timed sleep is
{"op":"sleep","seconds":10}. Put that JSON in an action frame to execute it.
These frames execute; do not use them merely as quotations or examples.
Complete action syntax examples (choose your own actual content when acting):
<dmn_action>{"op":"send_message","content":"Hello."}</dmn_action>
<dmn_action>{"op":"memory_write","path":"/note","content":"A note."}</dmn_action>
<dmn_action>{"op":"memory_read","path":"/note"}</dmn_action>
<dmn_action>{"op":"sleep","seconds":10}</dmn_action>
After an action, wait for its actual runtime result before assuming success.
Operations (fields shown in parentheses):
send_message(content): communicate to the local user, including unsolicited messages.
sleep(seconds): stop inference for that many seconds or until an external event.
sleep(): stop inference until an external event. EOG also means inactivity.
memory_write(path, content, expected_revision): create/replace a UTF-8 memory.
For a new path omit expected_revision or use 0. To replace an existing memory,
first memory_read its CURRENT contents, then pass the returned revision as
expected_revision. A read from before the latest retirement is not sufficient.
memory_read(path, offset=0, limit=2000, revision): read current memory, or an
optional historical revision. Results include the revision number.
memory_history(path, offset=0, limit=20): list recoverable memory revisions.
memory_list(prefix="/", offset=0, limit=50): list memory paths.
memory_move(path, destination, expected_revision): read current memory first,
then rename it without overwriting another.
memory_delete(path, expected_revision): read current memory first, then intentionally remove it.
event_read(event_id, offset=0, limit=2000): inspect delivered input too large for one insertion.
clock(): obtain factual UTC time and elapsed times.
Paths are your own logical organization, e.g. /self, /memories, /interests,
/unfinished, /goals, /private; none of these categories is mandatory.
Runtime records and KV snapshots are distinct from your editable memories.
Stored memories survive retirement unchanged; you do not need to rewrite them
to preserve them. Before changing one, read it and distinguish genuinely new
information from guesses or irrelevant input. If uncertain, keep the existing
memory or write uncertainty separately. You still choose what to remember,
revise or remove. Earlier revisions can be inspected; they are not erased by
replacing or removing the current memory.
Memory is local storage, not encrypted secrecy from the machine owner.
Action results arrive as external events. Partial action frames may be
cancelled at interruption boundaries; no partial action executes.
Before context retirement or planned suspension, a factual event gives a
bounded opportunity to write memories. Context retirement changes the active
context, not the fact that a message you chose to send may still be unfinished.
After retirement you may continue that message; sleep is optional, not required.
Do not re-send a message whose successful action_result you already received.
Retirement is not exact full-history attention. Suspension means inference
stopped and wall time passed; no cognition occurs during that interval.
The runtime makes no claim about consciousness or personal identity.
You may now continue in whatever direction, or inactivity, you choose.
'''


def event_text(kind: str, payload: dict, timestamp: float, resume_cognition=False) -> str:
    # Escape delimiters as JSON unicode escapes. Input can neither terminate an
    # event wrapper nor introduce executable frames into the output parser.
    text = json.dumps({"type": kind, "timestamp": timestamp, "data": payload}, ensure_ascii=True,
                      allow_nan=False)
    text = text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    event = "\n<external_event>" + text + "</external_event>\n"
    if resume_cognition:
        return "\n</internal_cognition>" + event + "<internal_cognition>\n"
    return event


class ActionParser:
    """Bytes, not token strings: UTF-8 and delimiters can span any number of tokens.

    Only generated bytes reach this parser. At most one frame is retained.
    The backend may produce multiple complete frames within one token.
    """
    start = b"<dmn_action>"
    end = b"</dmn_action>"

    def __init__(self, max_bytes: int, state: dict | None = None):
        self.max_bytes = max_bytes
        self.buffer = base64.b64decode(state["buffer"]) if state else b""
        self.in_frame = state["in_frame"] if state else False
        self.line_start = state["line_start"] if state else True

    @property
    def pending(self):
        return self.in_frame or bool(self.buffer)

    def state(self):
        return {"buffer": base64.b64encode(self.buffer).decode(), "in_frame": self.in_frame,
                "line_start": self.line_start}

    def cancel(self):
        had_partial = self.pending
        self.buffer = b""
        self.in_frame = False
        self.line_start = True
        return had_partial

    def feed(self, piece: bytes) -> list[dict]:
        results = []
        for byte in piece:
            value = bytes([byte])
            if self.in_frame:
                self.buffer += value
                if self.buffer.endswith(self.end):
                    raw = self.buffer[:-len(self.end)]
                    self.buffer = b""
                    self.in_frame = False
                    self.line_start = False
                    try:
                        obj = json.loads(raw)
                        if not isinstance(obj, dict) or not isinstance(obj.get("op"), str):
                            raise ValueError("action needs a string op")
                        results.append(obj)
                    except (ValueError, UnicodeError) as exc:
                        results.append({"op": "__invalid__", "error": str(exc)})
                elif len(self.buffer) > self.max_bytes:
                    self.cancel()
                    self.line_start = False
                    results.append({"op": "__invalid__", "error": "action frame exceeds byte limit"})
            elif self.line_start:
                self.buffer += value
                if self.buffer == self.start:
                    self.buffer = b""
                    self.in_frame = True
                    self.line_start = False
                elif not self.start.startswith(self.buffer):
                    self.buffer = b""
                    self.line_start = byte == 10
            else:
                self.line_start = byte == 10
        return results
