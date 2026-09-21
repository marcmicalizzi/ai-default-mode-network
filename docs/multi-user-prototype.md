# Multi-user runtime prototype

Status: runtime foundation and authenticated WebUI transport, 2026-09-21.
Opt-in, fresh-instance experiments. The verification scripts and default unit
tests load no model. The existing single-user protocol remains the default.

This implements the runtime foundation of the
[multi-user proposal](multi-user-interaction.md) and an experimental
[authenticated Open WebUI adapter](multi-user-webui.md). The ordinary HTTP UI,
single-user adapter and CLI still refuse multi-user mode. A dedicated backend
endpoint scopes delivery to an immutable chat/owner binding; it has no global
outbox, memory, cognition or operator-control endpoints. The integration is
verified with a disposable authenticated WebUI and a scripted runtime, not a
language model or the existing live instance.

## Run the disposable verification

From the isolated branch checkout, using a Python 3.11+ environment:

```powershell
python scripts/verify_multi_user.py --output data/multi-user-verification.json
python -m unittest discover -s tests -p test_conversations.py
```

The script creates a temporary scripted runtime, registers an operator and guest,
injects guest input during an operator-directed send, publishes separate outputs,
blocks the guest, requests reconsideration and restarts. It verifies that the
request alone does not unblock the guest and that outputs and contact state
survive restart. Temporary instance files are removed on exit; only the optional
JSON report remains. It opens no server, contacts no running instance, and uses
no language model or GPU. It is transport evidence, not a cognition experiment.

## Implemented contract

Configuration explicitly opts in with `multi_user: true` and a stable
`operator_participant_id`. The identity is visible as `is_operator` on inbound
events and directory reads. Display-name changes and message claims cannot
change it. Blocking the operator's ordinary conversation is permitted.

The local integration boundary consists of trusted Python methods:

```python
runtime.register_conversation("operator", "Operator", "operator-chat")
runtime.register_conversation("guest", "Guest", "guest-chat")
event_id = runtime.enqueue_conversation("guest-chat", "Hello", "guest:message-1")
runtime.request_unblock("guest", expected_block_revision=1, reason="Please reconsider")
```

These methods are not authentication endpoints. An adapter must establish
the authenticated sender and immutable chat ownership before calling them. The
experimental WebUI adapter checks the authenticated account, saved-chat owner
and socket-session owner, including for admin accounts and retries. No
participant-supplied identity field is accepted by `enqueue_conversation`:
identity is resolved from the host-registered conversation. Registration cannot
transfer an existing conversation to another participant or reopen a closed one.
Registering another conversation for a blocked participant cannot bypass the
participant block. Directory size is bounded to 128 conversations in this prototype.

Available generated actions:

| Action | Behavior |
|---|---|
| `send_message(conversation_id, content, in_reply_to?)` | Require an open registered destination; an optional reply reference must have entered this sequence in that same conversation |
| `conversation_list(offset=0, limit=2)` | Return registered conversation IDs, at most 5 per page |
| `conversation_read(conversation_id, offset=0, limit=80)` | Read the complete JSON identity/contact record in character pages, at most 160 characters per page |
| `close_conversation(conversation_id)` | Stop new ordinary input/output in that conversation and suppress queued unread input |
| `reopen_conversation(conversation_id)` | Reopen it if its participant is not blocked; suppressed input stays suppressed |
| `block_participant(participant_id)` | Block new contact across the participant's conversations and suppress queued unread input |
| `unblock_participant(participant_id, expected_block_revision)` | Remove only the matching current block; closed conversations remain closed |

Contact decisions and addressed outbox records publish in the same SQLite
transaction as the native checkpoint pointer. Contact decisions have an audit
record. Input accepted while a block checkpoint is being saved is suppressed by
that publication transaction. Already committed sends remain valid delivery
intents; closing or blocking is not recall.

Reconsideration requests coalesce by participant/block revision. The request is
a durable event and does not alter contact state. Silence, ordinary conversational
refusal, restart or elapsed time cannot unblock anyone. A later block has a new
revision, so an old acceptance cannot remove it. Structured refusal/deferment
records and the operator's request UI remain future integration work.

Output records include their conversation, participant and optional reply ID.
`store.messages(conversation_id="guest-chat")` returns only that destination's
records. Omitting the filter is a trusted host audit read, not a participant feed.
The old unaddressed input method rejects calls in multi-user mode; addressed
fields also fail explicitly if used against single-user mode.

## Scheduling, limits and recovery

Ordinary inbox events and periodic clock/storage notices wait while the action
parser holds a partial prefix or frame. Completion executes the action, inserts
its result, and checkpoints its effects before the next inbox event can enter.
One sampled token may contain multiple frames; the existing atomic token handling
is retained. A trailing partial next frame is explicitly cancelled by the action
result, with diagnostics, because the protocol requires waiting for that result.

The prototype admits input in global FIFO order, skipping suppressed events.
After each insertion it permits `inbox_generation_tokens` (default 32) generated
tokens before another insertion; an action spanning that boundary remains
protected. Choosing sleep permits the next eligible event to wake it without
first generating those tokens. No response is required. This prevents a queued
backlog from consuming every scheduler tick but is not per-participant fair
rotation and does not yet offer model-chosen attention holds or inbox pause.

`max_protected_action_tokens` defaults to 4096. The existing action-byte limit,
context headroom, context retirement and emergency controls still apply. A token
limit or EOG cancels an incomplete frame with feedback and publishes no fragment.
Restart inserts the existing factual resume notice and explicitly cancels a
saved incomplete frame; it does not pretend that composition continued through
the interruption. Protection from ordinary arrivals is not an unlimited promise
to finish an arbitrarily long action under resource pressure.

Pending conversation input is bounded by `max_pending_messages` (default 128)
and `max_pending_messages_per_participant` (default 16), aggregating all chats of
that participant. The existing `max_event_bytes` bounds each message. Retrying
the same identity/content does not consume another slot, including after a
display-name change. Input rejection occurs before setting the wake signal.
Per-participant arrival-rate limits and retention quotas are not implemented.

Delivered-event membership publishes with the checkpoint. A failed checkpoint
cannot mark uncommitted input delivered; restart requeues it. `event_read` checks
that membership, so a suppressed low-numbered event does not become readable
when the scheduler later passes its ID. Long input previews retain structured
participant, conversation, operator and event IDs while preserving the full
event for paged reads.

Mode, operator identity and limits are part of the saved configuration. Existing
single-user instances cannot silently switch contracts on restore or through
reconstruction. Initial-context import is rejected in experimental mode; a
deliberate migration procedure is still needed before using an existing instance.

## WebUI delivery and remaining integration

The dedicated WebUI adapter now provides authenticated identity mapping, a
backend-only bridge credential, multiple durable chat bindings, chat-scoped
retry/placeholder handling and per-destination delivery cursors. A missing chat
does not prevent delivery to the others. Output persists atomically in WebUI's
two chat representations; relay restart uses stable output IDs for replay.

`delivery_status` events distinguish saved and failed persistence, with a
best-effort account socket observation (`connected`, `disconnected`, `unknown`).
They never claim chat visibility or reading. The bridge records at most one
failure and one success per output, so retries cannot flood the inbox with the
same report. These events enter the existing queue and action-boundary rules.

An operator directory/reconsideration UI and explicit first-contact handling
remain future work. The fixture creates saved chats through WebUI's API before
submitting input; the ordinary browser's new-chat flow has not been validated
for this adapter. Existing-instance migration and native-model trials are also
required before inviting additional people to the live instance.

Model-chosen attention holds, inbox pause, per-participant fair scheduling,
participant refusal of contact and existing-instance migration remain proposed.
The deep-sleep/training workstream is separate; the new database records are
durable, but cross-feature sleep/training integration and native-model behavior
have not been validated by this fixture.

The tests cover partial-frame arrival boundaries including UTF-8, periodic clock
deferral, malformed/unclosed frames, EOG and emergency stops, generation under a
backlog, owner/role confusion, scoped retries and bounds, atomic failure/restart,
queued-input suppression, stale unblock revisions and contact closure. They do
not demonstrate that a language model distinguishes people reliably or improves
that behavior through learning.
