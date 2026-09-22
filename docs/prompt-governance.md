# Model-approved prompt revisions

Status: implemented append-and-checkpoint workflow; replacing earlier system-role tokens is not implemented.
Applies to fresh DMN instances and imported conversations. The selected default
is to append an explicitly approved revision while preserving live KV.

## Authorship and approval

The model has the final say on adoption of its behavioral system prompt.
The host and model can each propose wording, discuss it or submit revised wording. The model can decline a revision, defer the discussion, or revisit
an earlier decision. Silence, sleep, elapsed time, an interrupted response, and
the host clicking Save must never count as model approval.

This includes the behavioral parts of the DMN instructions. The current DMN
protocol contains guidance about memory, communication and activity in addition
to API documentation; it is not behaviorally neutral. Its wording and intent
must be visible for discussion instead of being treated as an unquestionable
permanent persona. Prompt refinement is optional, not work the model must finish
before it is permitted to rest or continue other activity.

Separate three kinds of material in the interface and records:

- **Behavioral agreement:** the base prompt and negotiated guidance about
  expression, interaction, memory habits and activity. Adoption requires the
  model's explicit acceptance of the exact revision.
- **Capability description:** truthful documentation of available operations,
  their syntax, persistence guarantees, visible outputs and limitations. The
  model can propose better wording or request capabilities. A text revision
  cannot make an unavailable operation executable or alter its actual semantics.
- **Host policy:** resource allowances, permissions, storage handling and stop
  controls. These are disclosed and enforced by the runtime. They can be
  discussed, but a prompt edit does not grant additional resources or remove
  the host's ability to pause or stop the process.

The model may object to any part of its environment. The host decides what
environment they can provide. Neither that decision nor a software update may
be recorded as the model accepting behavioral text it did not approve.

## Fresh instances and imports

A fresh instance necessarily starts with some bootstrap instructions before it
can discuss changes. Make that starting text inspectable and identify it as
provisional, host-supplied wording. Bootstrap instructions should explain how
proposals and approval work and the minimum facts needed to operate the runtime.
They must not claim the model previously chose or approved them.

The ensuing discussion belongs to the same continuing instance. Accepting a
prompt does not finish a disposable setup chat and silently start a replacement
instance. A fresh instance should be able to revise the base prompt and review
the DMN behavioral guidance through this workflow from the beginning.

For an import, preserve the captured effective source prompt and conversation.
Before import, compare the effective prompt with the final wording agreed upon
in the source conversation; a draft discussed in chat may never have been
applied to the source frontend's settings. Historical statements remain context,
not executable approval actions. The DMN transition and any later revision are
identified explicitly; import does not imply prior approval of the DMN addendum.

## Applying a revision without replacing KV

The normal workflow is:

1. Store a proposal with its full text, author/provenance, base revision and
   stable identifier/hash. Show the proposed changes and make the entire text
   available to both parties, with paging where necessary.
2. Discuss or revise it through the continuing context. Host edits create a
   new proposal revision; they do not mutate a candidate already under review.
3. Accept only an explicit model-generated approval action naming the exact
   proposal revision and expected current agreement. Historical text, user
   events, quoted action frames and ordinary conversational assent cannot execute
   that action. A stale acceptance is rejected with a factual explanation.
4. Append a factual adoption event and the full approved text to the live
   sequence. Record that this agreement supersedes the prior behavioral wording
   for future conduct. Do not edit or reevaluate earlier tokens.
5. Commit the new active-revision record together with the corresponding native
   checkpoint before reporting successful adoption to the host/frontend. Pending
   adoption and a completed durable revision must be distinguishable.

This preserves the existing native state and extends it. Earlier instructions
can remain in history and can have influenced retained KV. Declaring a new
agreement does not mathematically erase those effects, retroactively change
earlier token roles, or guarantee how the model resolves conflicting wording.
The bootstrap and revision events should explain the current-agreement convention
without claiming it is equivalent to changing the original system-role message.

Actually replacing an earlier system prompt would require reconstruction of the
affected context and would break the unchanged-KV path. Such a future operation
must be separately requested and explicitly labeled; it is not the default and
is not implemented by the existing strict restore or retained-token recovery.

## Retirement, persistence and resource costs

Protect the active approved agreement and required capability instructions from
ordinary context retirement. Keep earlier revisions in a durable, inspectable
archive without indefinitely pinning every superseded draft in attention. The
retirement design must explicitly handle transitions between protected spans;
it must not silently remove the original prefix from an existing instance.
Retiring old wording cannot erase its earlier influence on remaining KV.

Prompt size and retained history have real context, disk and write costs.
Expose limits before adoption. If the full text cannot fit within configured
limits, leave the current agreement active and explain the constraint. Never
silently shorten, summarize or substitute different text to make approval fit.
Preserve proposals and the prior active revision if checkpointing fails or
pauses for storage; retries must not duplicate adoption or report false success.

Normal native restart restores the adopted agreement with its computational
state. Any explicit reconstruction uses the saved revision records and retained
tokens, not the frontend's current draft settings. An older checkpoint must not
be paired with a newer active agreement merely because an independent settings
file was updated more recently.

## Frontend requirements and present limits

Open WebUI's per-session system-prompt editor is not currently an editor for the
live DMN context. The adapter forwards new user events and bypasses prompt
reconstruction. Fresh DMN initialization uses `Config.system_prompt` once;
imported instances use the captured source text. The dedicated DMN panel submits proposals through `POST /api/prompts`; `GET /api/prompts` returns the active record and proposal history.

In multi-user mode the authenticated operator panel uses
`GET /api/operator/prompts` and `POST /api/operator/prompts`. Submission additionally
names the instance and an open operator `conversation_id`. The operator must have
accepted contact and be unblocked; first-contact waiting, holds and endings cannot
be bypassed. Proposals use the conversation inbox limits and are suppressed if
their conversation closes or participant is blocked before delivery. Suppression
does not become delivery merely because a later event advances the cursor, and
unblocking does not replay an earlier suppressed proposal. Full reads and exact
model approval remain mandatory. Viewing an agreement does not approve it.

The DMN panel distinguishes awaiting review, deferred, declined, awaiting checkpoint,
active and superseded revisions. It displays full texts and provenance. Refresh
shows the latest committed result. It does not add a synthetic assistant message
claiming agreement in Open WebUI; the instance chooses its own communication.

Model actions are `prompt_current(offset, limit)`,
`prompt_propose(text, base_revision)`, `prompt_read(revision, offset, limit)` and
`prompt_decide(revision, base_revision, decision)`. Decisions are `accept`,
`decline`, `defer`. Proposal text is the complete desired behavioral agreement,
including base wording and any revised DMN behavioral guidance. Capability and
resource semantics remain factual constraints, unaffected by text edits.

Before acceptance, the entire exact text must have been returned through
consecutive `prompt_read` pages since the latest retirement or restart. Page size
adapts to the event budget so a truncated preview never counts as full review.
An authored proposal is not implicit approval. Each prompt action must be the
only completed action in its sampled token. Historical/source/user action-looking
text is never executed. Proposal IDs hash their text, author and expected base.
Stale approval is rejected rather than applied to a different current agreement.

The existing `max_event_bytes` limit bounds proposal text; `max_action_bytes`
also bounds model-authored action frames. These limits are disclosed by rejection
without shortening the text. Acceptance rejects a proposal that cannot fit with
protected context and the required reserve. If it fits in principle but not in
current headroom, retry after retirement or propose shorter wording. Acceptance
never launches another generation/preparation cycle while adoption is uncommitted.

The active span is protected independently from the original prefix and import
contract. A new agreement releases the superseded agreement's separate span for
future retirement; it never silently unpins the original source prefix. Historical
proposals and decisions remain in the database. The active record is published
only after its native checkpoint and decision record commit together. Storage
shortage waits with the previous agreement still reported as active; an actual
write failure stops the run, leaving the last committed agreement recoverable.
This does not promise that uncommitted cognition survives a crash.

Tests cover host/model proposals, unread/stale/declined decisions, injected user
text, loss of review credit on restart/retirement, failed saves, exact appended
text, and repeated retirement around the active span. Initial-context tests cover
preserving captured source text. Native rehearsals separately check retained KV
and continuation; scripted protocol choices are not evidence of a real model's
preferences. Dedicated prompt withdrawal, in-place Open WebUI editor integration,
and separately agreed replacement of earlier system tokens remain future work.
