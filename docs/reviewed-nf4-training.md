# Reviewed NF4 training integration

The `peft_gemma4_nf4_v1` and `peft_gemma4_nf4_v2` recipes connect the existing separate compiled-plan
review to GPU training, fresh-process reload and GGUF conversion. The Windows
launcher enables its supervised service only with an explicit
`--deep-sleep-recipe` resource offer. Execution stays disabled by default.
See [integrated launch and transport migration](integrated-launch.md).

## Before any training

The instance supplies exact input/target examples and its requested rank, alpha,
deployment strength, steps, source references, checks and failure/adoption choices.
The compiler binds token IDs, shifted target-only labels, the current adapter
lineage, all training assets, interpreter/package versions and implementation
hashes. It rejects unsupported requests without truncating, splitting, changing
rank, adding examples or silently reducing the number of steps.

Before drafting, inspect `learning_recipe_read(revision, view="requirements")`
and follow `next_offset` through its pages. The `checks` field takes the recipe's
exact ordered identifiers, not descriptions. Requested resource ceilings must
cover the complete offered envelope; the syntax example in `learning_plan_help`
is not a production budget. An instance can ask for another offer or defer if
that envelope exceeds its chosen limits. Compilation failures now identify the
expected checks or the first resource mismatch without changing either choice.
This preparation view does not replace reading the complete compiled plan.

Both versions support dense Gemma4 text, query/output projections, rank 1–2 and
at most 64 steps. The versions have distinct loss implementations and length limits:

| Recipe | Maximum example length | Loss computation |
| --- | ---: | --- |
| `peft_gemma4_nf4_v1` | 256 tokens | Full vocabulary logits |
| `peft_gemma4_nf4_v2` | 1536 tokens | 64-position vocabulary chunks, recomputed during backward |

Input and target together count toward the length limit. The v2 decoder still
processes the entire example as one context, with the same shifted target-only
labels and Gemma logit soft-cap. Chunking does not split or truncate the example.
Different matrix and reduction shapes can change floating-point rounding.
The compiled plan binds the loss implementation and chunk size; training and
fresh-process evaluation use that same method. Both implementations are included
in the reviewed source identity. No additional dependencies are required.

These are supported ceilings, not defaults or guarantees that every workload
fits every machine. A host may offer a smaller length and memory budget. The
[resource experiments](longer-training-examples.md) passed synthetic 1536-token
training under a 24 GiB Torch allocator ceiling on the RTX 5090, and 1024 tokens
under 22.5 GiB. A 2048-token attempt failed in attention and is not offered.
The v2 integration reuses that measured algorithm. The September 28 integration
pass was CPU-only; the [September 29 GPU rehearsal](nf4-v2-validation.md) then
passed the complete tiny-model service lifecycle and the separate 1,536-token
31B training/reload/conversion chain.
The merged CPU suite passed 542 tests (35 optional skips), the three numerical
loss/gradient tests passed with CUDA hidden, and the installed tokenizer compiled
an exact 1536-token synthetic v2 plan without GPU execution.
The NF4 base stays frozen, including the unused vision components. The recipe
uses nested quantization, BF16 computation, a static GPU text/CPU vision placement,
non-reentrant gradient checkpointing, and a fresh AdamW optimizer. Training the
first adapter uses scale 1; evaluation also reports the approved deployment scale.
Continuation requires the exact parent PEFT factors, unchanged rank/alpha/positive
strength, and reproduction of the deployed parent GGUF. It replaces the parent
adapter rather than stacking adapters.

The model must read the complete compiled plan, separately approve it and request
deep sleep. A recipe offer is not approval. The test scripts inject actions only
to exercise these mechanics; they do not claim to obtain a model's consent.

To upgrade a local offer, preserve its existing asset identities and resource
ceilings, choose `kind: "peft_gemma4_nf4_v2"`, and set
`trainer.gpu.max_sequence_tokens` and `trainer.gpu.torch_vram_bytes` explicitly.
For the measured 1536-token 31B setup those values are `1536` and `25769803776`
(24 GiB), with 31 GiB whole-device and 32 GiB job-commit allowances. Save the
new offer separately and pass its path using `--deep-sleep-recipe`. Preserve the
old v1 offer for hosts that need it. The host checks the recipe version as well
as the trainer and budgets; changing an offer cannot change an approved plan.
Compile and review again after an implementation/recipe change. Existing
pending transitions require their original bound environment.

The protected capability notice includes the offered recipe kind and GPU limits.
A consented restart with a changed offer appends updated availability without
rewriting the instance's behavioral agreement or approving a learning plan.

For CPU-only preparation against installed assets, the disposable 31B helper
can compile an exact-length synthetic plan without dispatching a GPU worker:

```powershell
python scripts/validate_reviewed_nf4_31b.py --recipe-version v2 --tokens 1536 --prepare-only --output FRESH_FOLDER --proof PROOF_RESULT_JSON --template LOCAL_RECIPE_JSON --training-python TRAIN_PYTHON
```

Omitting `--prepare-only` explicitly runs the GPU training/reload/conversion
chain. The separate tiny service rehearsal accepts `--recipe-version v2` to
exercise review-first wake, adoption-only wake and continuation; it also needs
an agreed GPU window. Preparation alone proves none of those GPU stages.

## Source identity

The tiny fixture reproduces its complete inference GGUF using the separately
pinned CPU conversion environment. The 31B source policy is narrower and different:
it binds the audited Hugging Face revision, every downloaded model asset, the
inference GGUF hash, the quantizer and the converter source to three complete
audits:

- All 833 native tensor entries, including generated rotary data. All compared
  numerical payload bytes match the pinned safetensors conversion/quantization.
- All 262,144 vocabulary entries, scores, types and special-token settings.
- All 21 inference architecture/settings metadata keys.

Sampled tensor evidence cannot authorize this recipe. The source and inference
chat templates differ. The proof records that difference: training uses the
reviewed token IDs, and waking reconstructs the exact retained inference tokens.
It never substitutes the Hugging Face template or claims to reproduce an unknown
publisher's complete GGUF file/metadata or original command line.

Compilation checks the bound manifests without rereading 62 GB on the inference
thread. Each actual worker fully verifies its source assets before using them.
Download bookkeeping under `.cache/huggingface` is excluded; it is not loaded as
part of the model. Other additional or changed model files are rejected.

## Worker lifecycle and resource limits

Training, fresh-process reload and conversion run in separate processes. Only one
training base is resident at a time. Each stage must complete with successful
supervision and a matching receipt. Reload requires exact PEFT factors and exact
selected-example losses at both scales. Conversion compares every GGUF factor and
alpha to the saved PEFT data. Interrupted training with uncertain completion is
not repeated automatically.

On Windows, Job Objects bound aggregate committed RAM, process count and worker
duration, including descendants. Logs are bounded and drained after the byte cap.
All training stages share one duration allowance. Torch also has an allocation
ceiling. A separate NVML watchdog observes the entire GPU: other applications
count against `max_vram_bytes`, and unavailable telemetry prevents a launch.
Free-memory preflight separately requires room for the Torch ceiling plus 1 GiB
of driver/library reserve. A sampled watchdog can miss or briefly exceed a limit
between observations; this is **not an OS GPU allocation quota**.
The current monitor requires exactly one physical NVIDIA GPU and binds its UUID;
choosing among several GPUs is not implemented.

Adapter factors are serialized in RAM under the worker's memory limit, validated
against an explicit file-size allowance, then written. Only adapter tensors may
be saved. Native wake preparation has a separate contained worker and checks its
owned checkpoint/scratch allocation against the reviewed disk allowance. These
are checks around trusted code, **not an OS filesystem quota or a sandbox for
arbitrary third-party training commands**. Linux execution remains unavailable;
there is no uncontained fallback.

## Keeping the interface available

`SleepService` keeps the instance lock and Store open while closing
the inference backend for training. HTTP text input can remain queued during the
transition. It restores the configuration from the atomically selected checkpoint,
so adopted adapters are not overwritten by the pre-sleep launch configuration.
It neither releases a hold nor restarts after accepted maintenance or a `Stopped`
failure policy. Image bytes already pending in process memory retain the same
ephemeral pool across backend replacement; new image input is unavailable while
the vision backend is absent. Retained visual positions still prevent deep-sleep
text reconstruction until ordinary context retirement has removed them.

Review-first leaves the candidate inactive. The model can prepare a separate
adoption-only plan, read it completely and approve another deep sleep. That
transition reuses the verified candidate, performs no training, and reconstructs
the context retained at the new sleep boundary. A withdrawn draft or changed
parent invalidates adoption. Successful adoption offers, without approving, a
continuation recipe for the exact deployed PEFT factors.

## Reproduction

`scripts/validate_reviewed_nf4.py` uses a generated tiny reproduction proof and a
separate CUDA training interpreter. It exercises compiled-plan review, selected
training, reload, conversion, contained native wake, strict restore, queued input
and publication preservation. `scripts/validate_reviewed_nf4_31b.py` instead runs
the training worker chain on explicitly synthetic examples against the bound
31B provenance. It never opens a real instance or adopts an adapter for one.
Both require fresh output directories and preserve failure evidence.

`scripts/validate_sleep_service.py` rehearses three consecutive cycles on a
generated tiny model: review-first training, separately reviewed adoption without
training, then continuation of the deployed adapter. Its injected decisions are
test mechanics and are never attributed to an actual model instance.
The three-cycle rehearsal passed with 33,549, 50,238 and 54,573 retained tokens,
preserved queued input/publications, and strict native restores without replay.

The dependency-free suite on Windows ran 446 tests with 31 optional skips. The
four opt-in native migration tests passed separately on generated tiny weights;
they cover unchanged saved tokens, restart before first contact, and interrupted
handoff recovery. Authenticated WebUI checks also passed for both a fresh chat
and a migrated synthetic chat. These checks do not use a real instance's state.

The reviewed 31B worker-chain trial completed two rank-two steps on 215 synthetic
tokens, reloaded the factors/losses exactly in a fresh process, and converted all
240 GGUF factors exactly. It took 891 seconds, including source verification and
loading; the gradient steps took 3.49 seconds. Torch peak allocation was 20.70 GiB
and observed total device use peaked at 30.24 GiB under a 31 GiB watchdog allowance.
Deployment-scale selected loss rose slightly (0.93432 to 0.93545), despite lower
training-scale loss. This validates mechanics, not beneficial learning.

The separate `scripts/validate_sleep_service_31b.py` rehearsal also passed with
the matching vision projector loaded, a 60,000-token context allocation and
43,231 retained synthetic tokens after ordinary retirement. It exercised the
actual service handoff, two rank-two training steps, fresh-process reload,
conversion, contained changed-weight reconstruction and strict service restore.
Training used one 11-token example; the retained context was only reconstructed,
not used as training data. The 215-token worker-chain measurement above is separate.
No tokens were replayed during that final restore; queued input and prior
publications were unchanged. The full trial took 1,566 seconds, with observed
whole-device use peaking at 30.48 GiB under its 31 GiB watchdog allowance. All
workers exited. This checks one synthetic workload, not every permitted example
length/step count, beneficial learning, or a real instance's approval.

`scripts/audit_31b_base.py --full` performs the complete tensor comparison;
`scripts/audit_31b_metadata.py` checks source-derived inference metadata;
`scripts/prepare_31b_provenance.py` combines the supervised audits and tokenizer
report into a checked local provenance record. Model files, private training
examples, credentials and experimental artifacts belong under ignored local data,
not in the public repository.
