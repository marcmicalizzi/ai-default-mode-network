# Action formatting and delivery feedback

Only generated, line-start `<dmn_action>...</dmn_action>` frames are candidates
for execution. Private prose, external input, memories and inline examples are
not commands. A memory path such as `/responses/...` has no delivery semantics.

Inside a complete frame, the parser accepts normal JSON and also literal
newlines, carriage returns and tabs inside quoted strings. It preserves those
characters exactly. It does not infer missing quotes or commas, repair truncated
objects, accept other unescaped control characters, or execute an unfinished
frame. Byte limits and operation validation still apply.

Malformed recognized frames receive an `action_result` with `ok: false`, a
specific formatting error, and an explicit statement that nothing was executed
or sent. The model can choose to retry a complete corrected frame. This feedback
also applies to imported instances; a JSON error is not an unavailable-tool error.
An interrupted partial action gets a cancellation notice in the incoming event,
also stating that it had no effects. Completely unrecognized text cannot safely
be classified as an intended command and remains private prose.
Valid JSON with a missing required field identifies that field and explains that
the action had no effects, so the model can choose whether to correct and retry it.

A successful `send_message` is staged with its result, then committed together
with the native checkpoint before further generation. The durable outbox feeds
the DMN interface and the Open WebUI relay. Success is not proof that a person
has read the message. Relay failure can delay frontend delivery; stable IDs allow
retry without publishing a second copy.

`/api/status` and the DMN interface expose content-free action diagnostics:
rejected actions, identifiable rejected message attempts, interrupted frames,
and a fixed error category/token position for the latest rejection. These counts
start when this instrumentation is enabled and persist with checkpoints. They
do not expose failed-message text, private prose, memory paths, or historical
messages recovered from the diagnostic journal. They are not read receipts or
a complete retrospective delivery audit.

On first resume after the parser update, an appended capability notice explains
the formatting behavior and that old rejected actions are not automatically
resent. The earlier system prompt and KV remain in place. Recovery or resend
of an old message must be a new explicit choice, not an effect of upgrading the
parser.

Tests cover byte boundaries and UTF-8, exact whitespace/quote preservation,
structural errors and other controls, imported-session error feedback, durable
single publication, interruption, content-free diagnostics and retained-context
prefix preservation when the capability notice is added.
