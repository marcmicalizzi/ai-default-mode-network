# Experimental authenticated multi-user WebUI transport

Verified against the installed Open WebUI 0.11.0, with authentication, SQLite and
one WebUI worker. This includes first-contact consent and operator reconsideration
following the authenticated transport milestone of the
[multi-user prototype](multi-user-prototype.md). It uses the same Pipe and Event
functions as the single-user adapter, selected by an explicit backend manifest.
No installed WebUI source files are modified. The live instance is not migrated
or connected by the verification script.

## Reproduce the disposable verification

Run this with the Python environment that contains Open WebUI 0.11.0 and its
Socket.IO client dependency, from this branch's checkout:

```powershell
C:/path/to/openwebui/.venv/Scripts/python.exe scripts/verify_multi_user_webui.py
```

Each run creates `data/multi-webui-<random>/`, starts WebUI on a fresh loopback
port with a new authenticated database and disabled model providers, and creates
an operator and a regular account with identical display names. The regular
account receives an explicit read grant for the DMN model. Real authenticated
Socket.IO connections supply the session IDs used by the completion requests.
A scripted runtime exercises the transport without loading any language model
or using GPU memory. Only the fixture's own child process is stopped on exit.

The directory retains `report.json`, `webui.log`, fixture databases and local
bridge credentials for inspection. It is ignored by Git; do not publish the
credential files or copy this data into a live installation. The report itself
contains results, not credentials. Failed runs retain their logs.

The verification covers:

- Admin attempts to use another account's chat and mismatched socket owners.
- Stable operator identity despite identical names and contradictory message text.
- The same source message ID in different chats, with independent receipts.
- Addressed replies, incoming retries and preservation of delivered output.
- Browser-supplied ID collisions and forged delivery metadata.
- New input appended after an independently delivered reply.
- Saving a reply while its recipient's socket is disconnected, with factual feedback.
- Relay restart with rewound delivery cursors, without duplicate output.
- Blocking the operator, including retry rejection.
- A deleted destination producing failure feedback while another chat still receives replies.
- Creating a new chat through the normal completion route before its first saved message.
- Withholding first messages until explicit contact acceptance, including for the operator.
- Submitting operator reasoning while everyone is blocked, with no automatic unblock.

The default Python test suite separately checks credential/Origin/Host/instance
validation, endpoint scope, immutable identity mapping, report coalescing,
independent cursors, slow/failing destinations and notification/report failures.

Verification on 2026-09-21: the authenticated fixture passed all listed scenarios.
The full default suite ran 296 tests: 278 passed and 18 opt-in tests were skipped.
One additional consent-compatibility regression test was then added and passed
with the focused consent/recovery checks.
Native-model and training-test opt-ins were disabled for this run.

`--browser-hold` keeps the disposable fixture available for browser checks and
adds a third dummy account. Local fixture files can direct scripted consent and
block actions; these are test controls, not product operator endpoints. Browser
verification covered the normal New Chat flow, withheld/deferred status, release
after acceptance, rejection while blocked, preserved multiline reconsideration
reasoning, delivered requests leaving blocks intact, and resumed input after an
explicit unblock. Decisions in these checks were scripted, not model judgments.
Open WebUI 0.11.0 keeps a rejected send as a temporary browser error and prevents
another send from that page state. After contact is permitted again, reload the
saved chat to clear that local error before sending a fresh message.

## Identity and credential boundary

The host chooses a stable namespace for one WebUI installation and the operator's
authenticated WebUI account ID. `participant_id(namespace, user_id)` derives the
runtime participant ID. The runtime's `operator_participant_id` must match that
mapping before the bridge can start. It is not inferred from a display name,
admin role, submitted message or whichever chat sends first. An operator may
still be blocked by the instance.

`conversation_id(namespace, chat_id)` derives a separate address. The owner is
deliberately absent from that derivation: attempting to reassign a chat collides
with its immutable ownership record and is rejected. The runtime durably pins
the namespace/operator mapping and stores each original chat/account binding.
Reconfiguring a relay against another instance or namespace fails closed.

`serve_bridge(runtime, token=..., namespace=..., operator_user_id=..., port=...)`
opens a dedicated loopback endpoint. Its bearer credential is an installation
credential held only by the trusted WebUI backend; it is not a participant token.
That backend can submit authenticated accounts' inputs and access the outputs
for their bindings. Compromise of this credential compromises that installation's
conversation transport. It does not grant DMN memory, cognition, general event
reads, operator control, or unblock-request endpoints.

All bridge operations require the credential and exact `X-DMN-Instance` header.
Requests with browser `Origin` headers or foreign Host headers are rejected.
The endpoint accepts only binding, scoped contact state, scoped input, scoped
output and delivery reports, plus a minimal identity/protocol check. There is no implicit recipient,
global outbox feed, broadcast or control operation. Tokens should be generated
randomly; the server requires at least 32 non-whitespace ASCII characters.

The WebUI backend selects this adapter when `DMN_MULTI_USER_BRIDGE_CONFIG` points
to a local JSON manifest with these fields:

```json
{
  "url": "http://127.0.0.1:PORT",
  "instance_id": "EXACT-RUNTIME-UUID",
  "namespace": "STABLE-INSTALLATION-NAMESPACE",
  "token_file": "ABSOLUTE-BACKEND-ONLY-PATH"
}
```

The fixture constructs these values. Normal CLI startup deliberately remains
single-user until a supported multi-user launcher and deliberate migration are integrated.
Do not place the token or manifest in user-visible Pipe valves or model context.
An existing single-user relay database or adoption record is not silently reused.

## Admission and delivery

Before upstream placeholder writes, the adapter checks model access, account
identity, chat ownership and that the submitting socket belongs to that account.
The Pipe repeats the identity/ownership checks before enqueueing. Identity comes
from authenticated backend metadata; browser text cannot assert an operator role.
Retries still obey current closure and participant blocks. Registration never
reopens a conversation or clears a participant block.

Bindings are limited by the runtime's 128-conversation directory. Each chat has
its own lock, input receipts, placeholder IDs and outgoing cursor. Different chats
can reuse source message IDs without deduplicating one another's input. Existing
delivered nodes cannot be replaced using a new request's user or assistant ID.
New input is appended after the authoritative saved leaf; stale browser ancestry
cannot rewind an independently delivered reply. Delivery tags and other internal
message fields supplied by a browser are discarded.

Up to eight destinations are serviced concurrently. A failure leaves that chat's
cursor unchanged and retries in order; other destinations continue. Runtime
HTTP calls have a ten-second timeout. WebUI database and task-system stalls can
still delay a destination; this is not a general distributed queue service.

A reply's JSON history and normalized message row commit in one WebUI database
transaction. Stable output IDs make a retry after that commit idempotent. A
durable `delivery_status` report precedes cursor advancement. Socket reload
notifications are best effort: their failure cannot retract a successful save.

Reports identify the conversation, participant and committed outbox message.
`persisted` means saved in WebUI. `failed` means the save could not be confirmed.
`connected` means a fresh authenticated socket exists for that account;
`disconnected` means none was found; `unknown` means the lookup failed. Another
tab can keep an account connected after the addressed chat closes. There is no
claim of browser focus, chat visibility or a read receipt, and no continuous
presence subscription. At most one failure and one success are queued for each
output, so reconnects and retries do not produce repeated cognition events.

## First-contact consent

Fresh multi-user runtimes default to `require_contact_consent: true`, including
for the operator. The first message is held in runtime storage outside the event
stream. The model receives a `contact_request` containing authenticated identity,
operator status and the conversation address, with no message body or preview.
`event_read` cannot expose the held message. One first message may be held per
participant; further input is rejected while consent remains pending/deferred.

After receiving the request, the model may generate:

```text
<dmn_action>{"op":"contact_decide","participant_id":"p_...","expected_request_revision":1,"decision":"accept"}</dmn_action>
```

`accept` grants contact for that account across its chats and queues its held
message atomically with the decision checkpoint. `defer` or silence keeps it
withheld. `decline` prevents further contact and discards that message from future
delivery. An optional `reason` of up to 1,000 characters is shown to that person.
A later acceptance following decline permits new input but never replays the
discarded input. Closing or blocking also discards affected held input; consent
never reopens a chat or removes a block. Discarding is a delivery state, not
erasure from the operator's database.

WebUI displays waiting, deferred and declined states as transport status. The
relay updates that status without fabricating a model response. New Chat uses
the authenticated account/socket before WebUI creates the saved chat; the Pipe
rechecks its resulting owner before the first message is held. Accepted contact
does not promise an answer or prevent a later block.

## Separate operator reconsideration interface

`serve_operator(runtime, token=..., port=...)` serves a loopback contact directory
and request form, protected by a separate operator key. It rejects reuse of the
bridge key. The key stays in the page's memory for that connection; it is not
placed in WebUI, a URL, browser storage or model context. The interface remains
available when the operator and every other participant are blocked.

Select a blocked participant by stable ID and enter required reasoning of up to
4,000 characters. This can describe suspected mixed conversations, misattributed
statements and evidence for the instance to examine. Submitting queues an
`unblock_request` with the exact participant and block revision. The interface
shows the preserved reasoning and whether delivery has been checkpointed,
separately from whether the block remains in place.

There is one immutable request per participant/block revision. An identical
retry returns the same event; different reasoning for that revision is rejected
explicitly rather than silently discarded. A truncated context preview retains
the target, requesting operator, revision and event ID, with the full reasoning
available through `event_read` after delivery. Only the model's explicit
`unblock_participant` action changes the block. Declining, deferring, silence,
request delivery and restart leave access unchanged. This service has no direct
unblock, memory, cognition or general control endpoint.

## Experimental limits

The browser workflow and operator request form have been exercised with a
scripted fixture. A supported multi-user launcher, existing-instance migration
and participant-side refusal/leave controls remain future integration work.

The model still has one shared context. Passing transport tests does not show
that it distinguishes people reliably, keeps their information separate in its
reasoning, or improves through sleep learning. These require separate experiments
with an explicitly prepared instance. Native inference, migration from the live
instance and interaction with the parallel training workstream remain untested.
