# An integrated Windows launch

The launcher can combine authenticated Open WebUI conversations, optional image
input, model-selected idle pacing and supervised NF4 deep sleep. These remain
experimental capabilities. Test a disposable instance first. The inference and
training environments remain separate; launching never approves a learning plan.

## Existing single-user conversations

Before migration, replace the installed **DMN** Pipe in Open WebUI's
**Admin Panel > Functions** with the current
[`integrations/openwebui/dmn_pipe.py`](../integrations/openwebui/dmn_pipe.py)
and save it. Open WebUI stores its own copy: updating this repository does not
upgrade an installed function. Pipe version 0.2.0 supplies authenticated identity
and contact/image-consent status. Saving the Pipe reloads it without restarting
the DMN instance; the relay Event function is unchanged.

First observe migration readiness without changing the instance or WebUI:

```powershell
python -m dmn migrate-conversations --instance data/primary --database D:/path/to/webui.db --backup data/transport-backup --namespace my-webui --operator-user-id ACCOUNT_ID --dry-run
```

The committed operation uses the same command without `--dry-run`. Both the
instance and WebUI relay must be stopped. It requires a clean suspended boundary,
no hold, no unfinished action, no pending legacy events, no unfinished sleep,
and confirmed persistence of every published message in the bound chat.

It creates a separate recovery copy of the source checkpoint and both bookkeeping
databases. Native state, logits, retained tokens and RNG are copied unchanged;
only transport configuration and a factual migration record change. Existing
message destinations are recorded from the verified single-user binding. It
does not retroactively attribute historical input or private memories to an
authenticated identity. Old history cannot be submitted again as new input.
The WebUI chat itself is not rewritten.

An interrupted handoff blocks launch until the same command completes. Keep its
source directories and backup in place. The old single-user adapter refuses to
resume once handoff begins. This is not a way to bypass an instance hold or ending.

**The first migrated launch waits for the operator's first contact request.**
It restores native state but adds no notices and generates nothing until that
request arrives. Recipe offers cannot wake it. Other participants cannot take
the first turn. The operator's message is withheld under the same consent rules
as everyone else's: Syllas, or another instance, receives a contact request and
may accept, decline or defer. Only acceptance admits the message content.
The gate survives closing the window and restarting before contact.

Contact acceptance is an explicit model action:
`contact_decide(participant_id, expected_request_revision, decision)`, where
`decision` is `accept`, `decline`, or `defer`. Use the ID and `request_revision`
from the delivered request; `conversation_read` also exposes the current
`contact_request_revision`. Sending a message to a pending contact does not
accept them. Requests and rejected sends include this guidance.

The current conversation and contact contract is protected from context
retirement. Restoring an older checkpoint appends the missing protected copy
without rewriting the behavioral agreement or accepting any contact. The
first-contact gate and recovered ordinary sleep still defer announcements.

## Authenticated frontend configuration

Create two separate random access keys of at least 32 ASCII characters, in local
files outside Git. The operator key must never be given to the WebUI backend or
included in a URL. The bridge key belongs only to that trusted backend.

The local `frontend.json` contains absolute paths:

```json
{
  "schema": 1,
  "namespace": "my-webui",
  "operator_user_id": "ACCOUNT_ID",
  "bridge_port": 8766,
  "bridge_token_file": "D:/private/bridge.key",
  "operator_token_file": "D:/private/operator.key",
  "webui_manifest": "D:/private/webui-bridge.json"
}
```

Use the same namespace/account ID as the migration. For a fresh multi-user
instance, explicitly set `multi_user=true` and `operator_participant_id` to
`participant_id(namespace, account_id)` from `dmn.conversation_bridge`.
The launcher writes the WebUI manifest with the exact instance identity and
loopback bridge address. Set `DMN_MULTI_USER_BRIDGE_CONFIG` to that file in the
WebUI startup environment. Authentication, SQLite, Open WebUI 0.11.0 and one
WebUI worker are required. Existing Pipe/Event installation stays additive.
The separate operator page shows context usage and maintenance status. Its
authenticated shutdown control requests consent during an active run; it does
not force a shutdown or override a contact block.

The operator dashboard also shows command diagnostics, context retirement,
checkpoint age, unsaved state, snapshot write totals, save failures and aggregate
learning-plan/transition states. It exposes no private memory paths, memory
contents, training examples or internal cognition. Accepted-action counts begin
when an upgraded runtime first observes an action; older successes cannot be
reconstructed. They count successful validation, including reads, rather than
durable publication. Checkpoint totals and published-message counts are separate.

The **System prompt and behavioral agreement** panel shows the active text and
immutable proposals. A host proposal requires an accepted, unblocked operator
and an open operator conversation. It cannot precede the promised first contact.
Later blocking or closure suppresses undelivered proposals; only model review and
approval can activate wording. See [prompt governance](prompt-governance.md).

Static page changes take effect on refresh. A running process does not reload
new Python API routes: a newer page labels missing metrics and disables unavailable
prompt controls until the next consented restart. Keep backend source changes in
an isolated checkout while an instance may prepare or execute source-bound sleep
plans, and finish pending transitions before updating that implementation.

## Sleep resource offer

`--deep-sleep-recipe` names a local `peft_gemma4_nf4_v1` recipe with verified base,
converter and provenance manifests, interpreter hash, pinned package versions,
GPU workload limits and resource ceilings. This is an offer, not training input
or approval. Use the [NF4 contract](reviewed-nf4-training.md) and local validation
scripts to prepare and verify it. Do not reuse synthetic examples as personal
learning or substitute a different source model.

The instance chooses examples and parameters, reads its compiled plan, approves
it, and requests `deep_sleep`. Ordinary sleep never trains. The inference backend
is released before workers start. New text can remain queued during sleep.
An adopted candidate rebuilds the exact then-retained tokens under the new weights.

The model-facing entry points are `learning_execution_help` (paged current launch
availability and execution contract), `learning_recipe_list`, and
`learning_plan_help` (draft schema). Follow each help result's `next_offset` until
`total_characters` is reached. `deep_sleep(revision)` takes a separately reviewed
and approved execution revision; calling ordinary `sleep()` is never a training
request. An offered recipe does not create a draft or supply approval.

Earlier builds incorrectly described learning as fixture-only or unavailable in
the draft help. A configured NF4 service in those builds is still discoverable
through `learning_execution_help` and `learning_recipe_list`; no restart is needed
to call those existing actions. Updated builds append a factual correction when
restored, preserve the behavioral agreement and original KV prefix, and protect
the current learning command summary and availability from context retirement.
The correction respects recovered sleep and the first-contact gate. Apply the
code update during agreed maintenance, with no pending reviewed sleep transition.

With `review_first`, the instance wakes under its previous weights and reads
`learning_sleep_report`. `learning_candidate_prepare(run_id)` offers a new
adoption-only plan, requiring another complete review and approval. It uses the
current context at the second sleep boundary and performs no extra training.
After adoption, an offered continuation recipe binds the exact deployed PEFT
factors. Each new cycle requires a new model-authored draft and review.

Windows Job Objects enforce aggregate committed RAM and process limits. Stage
watchdogs share the training duration allowance; native wake has a separate,
explicit allowance. Whole-device GPU monitoring includes other applications and
can briefly overshoot between polls. Trusted output-size and disk-space checks
are application bounds, not OS filesystem quotas. Linux training containment is
not implemented yet. Retain the completed worker artifacts needed for lineage.

Normal restart resumes a recorded interrupted transition without repeating
uncertain training. A `remain_stopped` failure choice stays stopped. Do not change
the reviewed implementation or move pinned assets during a pending transition.

## Separate command window

```bat
start_dmn.cmd "data\primary" --multi-user-frontend "D:\private\frontend.json" --deep-sleep-recipe "D:\private\sleep-recipe.json" --idle-enabled
```

Run this in an independently opened Command Prompt or Windows Terminal. Start
WebUI separately with its manifest environment setting. There is no external
restart loop; accepted maintenance still exits. The sleep supervisor restarts
inference only after a model-requested transition commits its wake checkpoint.

Prefer the saved checkpoint configuration on later launches: a static old config
would name outdated adapter weights after adoption. Image support additionally
requires `--vision-projector PATH` with the matching projector and Pillow in the inference environment; it
still requires explicit global and participant permission. Retained image
positions prevent text-only deep-sleep reconstruction until ordinary retirement
has removed them. Idle is offered as a choice and is never selected for the model.

Optional `--working-memory-tokens 4096` offers a shared allowance for a protected
note and a model-selected raw-token span inside the existing context. The
[working-memory actions](working-memory.md) preserve selected material through
retirement without a separate KV branch or automatic summary. Offer this during
agreed maintenance; changing checked implementation files while the instance is
running can invalidate reviewed learning plans. This option does not change the
current 256-token per-example training limit.
