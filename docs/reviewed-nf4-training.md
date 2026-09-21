# Reviewed NF4 training integration

The `peft_gemma4_nf4_v1` recipe connects the existing separate compiled-plan
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

The current supported envelope is dense Gemma4 text, query/output projections,
rank 1–2, at most 256 tokens per example and at most 64 steps. Those are explicit
upper limits, not defaults or a claim that every combination fits every machine.
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
