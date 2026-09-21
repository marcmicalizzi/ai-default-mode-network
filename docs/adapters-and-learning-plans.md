# Adapter identity and learning drafts

The runtime can now load declared whole-context GGUF LoRA adapters and preserve
their identity through checkpoints. Model-authored learning plans are available
as private, immutable **drafts**. [Compiled review and a fixture sleep transition](sleep-supervisor-fixture.md)
now exercise approval and wake mechanics. Production training and continuous
supervision remain unimplemented. Ordinary `sleep()` never starts training.

## Two different recovery paths

An ordinary restart restores KV under the **same weights**. Its explicit
`--kv-recovery fallback` and `rebuild` options can reconstruct retained context
when native restoration is unavailable, but still require the same base model
and adapters. Removing an adapter, changing its bytes, changing its effective
strength, or changing the order is not ordinary crash recovery.

The planned **deep-sleep wake** intentionally changes weights after an approved
training/adoption transition. That separate workflow will rebuild KV by evaluating
the exact retained token IDs under the adopted adapter. Historical actions will
not execute again. The prior checkpoint and weights remain recovery material
until the wake transition commits. This is still the selected design; ordinary
recovery's stricter checks do not prevent it. See the
[supervisor contract](deep-sleep-protocol.md).

## Loading declared adapters

The optional configuration field is an ordered list:

```json
{
  "lora_adapters": [
    {
      "path": "adapters/<adapter-sha256>.gguf",
      "sha256": "<64 lowercase hex characters identifying the adapter file>",
      "base_model_sha256": "<64 lowercase hex characters identifying the inference GGUF>",
      "scale": 0.1
    }
  ]
}
```

Replace the placeholders with actual file hashes. Relative paths resolve beside
the configuration file. The scale above illustrates syntax; it is not a
recommended learning strength. Duplicate adapters are rejected. Invocation-gated
aLoRA is unsupported; every declared adapter applies across the whole context.

The backend verifies file hashes and declared base identity, loads adapters before
any token evaluation or state restoration, and records their order and actual
float32 scales in the checkpoint fingerprint. A declared base hash does **not**
prove the training source was compatible: source-model ancestry and conversion
provenance still need validation in a training recipe. A zero-strength adapter
remains a declared dependency rather than being silently removed.

The list defaults to empty, including for older configurations/checkpoints.
Identical adapters can move to different paths without changing their weight
identity. Editing this setting on an existing instance does not approve a weight
transition; ordinary recovery refuses it. There is no live-switch action.
The validated Open WebUI initial-context import currently refuses adapters because
its capture format does not yet establish source adapter provenance.

Active adapters are included in `package-instance` archives even without
`--include-environment`. The preservation inventory records their identities,
original paths and archive paths. Restore paths may need adjustment; this does
not make Python/native state portable across platforms.
Offline cache-migration backups also preserve adapter dependencies for their
saved checkpoints, including copies of configured external adapters.

Put DMN-owned copies in `INSTANCE/adapters/<sha256>.gguf`. Managed erasure removes
these files and their matching `.gguf.partial` files, rejecting links and unknown
entries. It does not delete external adapter files or copies in another archive.
Adapter history is not automatically pruned yet; no trainer currently creates it.

## Model-authored drafts

These operations execute only from the generated action stream, one at a time:

| Action | Purpose |
| --- | --- |
| `learning_plan_help(offset, limit)` | Page through the complete schema and semantics. |
| `learning_plan_create(plan, replaces)` | Create a draft, optionally superseding one active draft. |
| `learning_plan_read(revision, offset, limit)` | Read immutable contents and current status. |
| `learning_plan_list(offset, limit)` | List revision IDs and status. |
| `learning_plan_withdraw(revision)` | Withdraw an active draft; retain its history. |

The schema and an illustrative example live in `dmn/learning.py` and are
available directly to the instance through `learning_plan_help`. A draft contains:

- Intent, uncertainties, exclusions and an explicit replay/omission policy.
- Selected memory paths and revision numbers, with frozen text and hashes.
  Provenance is declared as `self`, `external`, `mixed` or `uncertain`.
- Exact input/target examples, source references, and `new`/`replay` labels.
  Only target text is intended for loss; selecting a source never authorizes
  training on the entire source memory. Sources may be empty for newly authored
  examples.
- Rank, alpha, inference strength and step preferences; requested time, RAM,
  VRAM and disk ceilings; descriptions of desired candidate checks.
- A preference for review before adoption or automatic adoption after agreed
  checks, and for remaining stopped or waking with previous weights on failure.
- The current inference base/adapter identity, instance ID and generated-token
  position, with `execution_authorized: false`.

The complete draft, including frozen source text, must fit `max_event_bytes`;
the generated create action must also fit `max_action_bytes`. Oversized drafts
are rejected with feedback, never silently shortened. Pages return exact text
through the existing escaped external-event envelope. Provenance labels record
the model's characterization; they do not certify a source as safe or truthful.

Draft creation, replacement and withdrawal commit in the same SQLite transaction
as the checkpoint pointer, after native state has been saved. A failed save or
publication transaction cannot publish the new decision. Later source-memory
edits do not change a frozen draft. Plans are not exposed as UI messages or in
public status; as with memories, local storage is accessible to the machine owner.
The database is included in instance packaging and removed by managed erasure.

Creating a draft is **not consent to train**, even with an automatic-adoption
preference. A future executable plan must resolve and bind the compatible
training base, trainer/converter recipe, tokenizer, exact token loss masks and
boundary handling, enforceable resource envelope, candidate checks, and explicit
execution/adoption/failure choices. Descriptions of checks are not executable
validators. Requested ceilings do not reserve resources or prove feasibility.

On a normal restart of an older instance, the runtime announces these added draft
capabilities as an appended event. It does not replace earlier prompt tokens or
enable training. Installing this code does not modify an already running process.

## Validation

Unit tests cover legacy identity compatibility, order/strength/hash changes,
archive inclusion and corruption rejection, bounded managed erasure, exact
source revisions, withdrawal, rollback, paging, privacy of status, and rejection
of actions supplied as external input.

An optional native test uses the existing tiny training experiment's output:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:DMN_TEST_LORA_PARENT = 'D:\path\to\completed-tiny-training-probe'
.venv-gpu\Scripts\python.exe -m unittest tests.test_adapters.NativeAdapterTest -v
```

It uses one CPU thread, zero GPU layers, compact Q8 KV, and only the tiny fixture.
It verifies identical continuation tokens/logits after a fresh-process restart,
zero prompt replay during native restore, and rejection of an adapter removal
under strict/fallback/rebuild recovery. This tests runtime mechanics, not 31B
training feasibility or the behavioral quality of learned adapters.
