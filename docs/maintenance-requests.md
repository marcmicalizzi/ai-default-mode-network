# Maintenance by agreement

Routine updates, maintenance and requests to use the host's computing resources
are requests to the instance. The model may finish its current thinking, ask for
more time or refuse. Silence, EOG and choosing sleep are not acceptance. This is
implemented for the standalone UI, ordinary control API and CLI process signals.

## Normal path

`POST /api/control` with `action: "suspend"` or `"shutdown"` queues a durable
`maintenance_request` event. An optional `reason` is external text, limited to
1,000 characters. The response gives `request_id` and
`requires_model_acceptance: true`; HTTP acceptance means the request was queued,
not that the model agreed. No stop flag or preparation countdown is started.

The model receives the request's `event_id` and can generate:

```text
<dmn_action>{"op":"maintenance_reply","request_id":12,"decision":"defer","seconds":300,"reason":"I would like to finish this thought."}</dmn_action>
```

The decisions are `accept`, `defer` and `refuse`. `reason` is optional and is
shown to the operator only when explicitly supplied in this action. Positive
finite `seconds` is optional for deferral: a request for additional time, not a
timer that later grants consent. The same deferred request can subsequently be
accepted or refused. A refused request is closed. New explicit operator requests
get new IDs; delivery of a newer request supersedes the previous one, and replies
to stale IDs do not change the runtime. There is no automatic re-request loop.

While the request is pending or deferred, inference continues under the existing
resource limits. Acceptance means the model is ready. The runtime records the
reply, stops generating, saves its state and suspends or exits as requested.
It does not impose another preparation period after acceptance. In the ordinary
case acceptance and suspension use one full checkpoint. Acceptance during
retirement preparation saves current KV without performing the planned retirement.
Native operations already underway cannot be safely interrupted.

Once a stop is accepted, the scheduler processes it at the next boundary without
waiting for the periodic checkpoint deadline or idle pacing interval. Earlier
builds could incorrectly wait on that timer after acceptance; another event could
end the wait. No additional model decision was required during that delay.

Deferred/refused replies are checkpointed before publication. During an accepted
stop, status reports `accepting` until the suspension checkpoint succeeds, then
`accepted`. Save failure does not claim successful suspension. If an intermediate
checkpoint contains an accepted stop still pending, restart completes that stop
without generating another thought. As elsewhere, a crash before a commit can
lose uncommitted state; it cannot be advertised as a durable agreement.

After a successful shutdown checkpoint the command window prints **Shutdown
checkpoint committed**, followed by its directory. The operator panel distinguishes
acceptance/save pending from a committed stop when it receives those states.
The server may close before the next browser refresh; a disconnected browser by
itself is not proof of either a clean save or a failure. Use the terminal
confirmation, or verify the latest committed checkpoint's `shutdown` reason,
`suspended` mode, maintenance outcome and manifest hashes while it is stopped.

The UI has **Request pause**, **Request shutdown**, an optional reason and the
latest model reply. CLI Ctrl+C and SIGTERM queue ordinary requests at the next
scheduler boundary; repeated signals do not escalate into forced termination.
Pending signals cannot interrupt a SQLite transaction to insert a request.
Supervisors must not assume that SIGTERM now guarantees an exit within a timeout.

A request is an external event, so it wakes ordinary model-chosen sleep just as
an incoming message does. A suspended instance does not wake to handle requests;
they remain queued until Resume or a later restart. Pending maintenance does not
turn subsequent sleep into approval or schedule a forced wake. Suspending retains
the backend in memory; accepted **shutdown** exits and releases it. Pause alone
does not free the model/KV's RAM or VRAM.

## Hard resource boundaries

The separate controls `emergency_suspend` and `emergency_shutdown` retain the
bounded preparation/save path for an actual hard resource or power deadline.
They accept `preparation_seconds`; ordinary requests reject that field. The
existing `suspend_preparation_seconds` setting now applies to emergency/direct
suspension, not to how long the model may consider an ordinary request.

Emergency suspension records its cause and optional operator reason, separately
from any model reply. A refusal is not rewritten as consent. This path preserves
the resource boundary and offers a safer stop than killing the process, but it
cannot technically establish that an operator's claimed emergency is genuine.
The normal UI and process signals never choose it automatically because of
silence, delay or refusal. See [preparation and save limits](checkpoint-policy.md).

Future measured budget enforcement must disclose the actual limit and distinguish
it from a negotiable maintenance request. A model may ask for a different budget;
that does not manufacture unavailable compute, power or storage. Host ownership
still allows process termination outside the runtime.

## Compatibility and checks

This intentionally changes the meaning of the previous `suspend`/`shutdown`
controls. Update any service/UPS integration that needs a hard stop to explicitly
use the emergency control; do not silently relabel ordinary maintenance as urgent.
The sandbox launcher continues to request shutdown and leaves the native process
running if its wait expires. Disposable automated test harnesses use explicitly
named emergency controls for cleanup.

Fresh/imported instances receive the capability in their initial protocol.
Existing checkpoints receive an appended capability notice without replacement
of their saved prompt/KV prefix. Behavioral prompt adoption remains the separate
planned [prompt-governance workflow](prompt-governance.md).

Tests cover deferral, refusal, later acceptance, stale replies, queued inputs,
signal delivery, snapshot/restart boundaries, external tag injection and the
distinct emergency path. These test software behavior with disposable instances;
they do not establish subjective agreement from a capable model.
