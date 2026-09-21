# An integrated Windows launch

The launcher can combine authenticated Open WebUI conversations, optional image
input, model-selected idle pacing and supervised NF4 deep sleep. These remain
experimental capabilities. Test a disposable instance first. The inference and
training environments remain separate; launching never approves a learning plan.

## Existing single-user conversations

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
