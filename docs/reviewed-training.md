# Training from a reviewed learning plan

The `peft_gemma4_cpu_v1` and `peft_gemma4_cpu_continue_v1` recipes connect
compiled-plan approval to actual PEFT training, GGUF conversion, durable candidate
validation and native context reconstruction. They use the same sleep transition engine as the prebuilt-adapter
fixture. **Execution is still restricted to the explicitly enabled tiny CPU
integration harness. Ordinary `dmn run` cannot execute it.** This is a working
part of the production implementation, not a supported 31B training service.

## What is bound before approval

The instance's draft supplies the exact input/target examples, source provenance,
new/replay designation, rank, alpha, deployment strength, number of steps, resource
ceilings, checks and adoption/failure choices. The compiler binds native token IDs
and target-only labels. It adds no prompt template, BOS, EOS, padding or truncated
examples. Cross-boundary tokenization is rejected.

The host offers a recipe; it cannot approve it. The recipe adds:

- A training interpreter path and SHA-256, plus explicit versions of PyTorch,
  Transformers, PEFT, safetensors, tokenizers and NumPy.
- Hash-bound manifests of local safetensors/config/tokenizer assets and the pinned
  llama.cpp converter source. Missing, changed and additional files are rejected.
- The expected inference model name, converter revision, learning rate and seed.
- The implemented checks, including tokenizer parity, unchanged base weights,
  adapter round-trip integrity and finite training.

The compiled plan makes the remaining recipe choices explicit: CPU float32, one
computation thread, batch size one, round-robin order through the reviewed
examples, AdamW with a fresh optimizer, no weight decay/dropout, gradient clipping,
and query/output projection LoRA factors. The instance must read the whole plan
and separately approve it before requesting deep sleep. Implementation source
hashes include the trainer and resource launcher; changes require new review.

First-adapter training uses scale 1; the approved deployment scale is applied
separately and both selected-example losses are reported. Continuation trains at
the existing positive deployment scale and preserves it when waking. A small
deployment strength is not a bound on unrelated behavioral change. Replay consists only of examples explicitly
included in the plan, not automatically retrieved earlier memories or messages.

## Continuing an existing adapter

`peft_gemma4_cpu_continue_v1` requires exactly one active whole-context adapter.
Its trainer has an additional `parent_adapter_manifest` reference, with the same
`{path, sha256}` form as the other manifests. It binds an existing directory with
`adapter_config.json`, `adapter_model.safetensors`, and optionally `README.md`.
The previous completed worker's `adapter/` directory can supply these files.
Neither an arbitrary similarly named adapter nor GGUF alone is sufficient.

The compiler exposes the parent adapter hash/base/strength and both PEFT file
hashes in a lineage record. Rank, alpha and deployment strength must match the
parent. This version refuses multiple adapters, zero/negative strength, rank or
alpha changes, merging, stacking, aLoRA, DoRA, variable-rank patterns and additional
trainable modules. Unsupported choices require a new recipe and review; they are
never silently dropped.

Before training, the worker converts the bound parent PEFT directory and requires
the resulting GGUF hash to match the active adapter. The original directory is
used for conversion because its model card can affect GGUF metadata. It copies
the exact config and factors to private `parent-adapter/` artifacts, loads those
factors as trainable, and checks every loaded tensor against the copy before any
gradient step. Earlier factors are continued, not randomly initialized again.

The optimizer starts fresh, as stated in the compiled plan. Training and wake
both use the parent's current strength (for example, the same float32 value of
0.1), while rank and alpha remain unchanged. Saving/reloading reapplies that
strength explicitly because deployment strength is not stored in PEFT factors.
The candidate replaces the one parent adapter on an approved wake. Source PEFT
and GGUF files remain unchanged, and the previous checkpoint remains available
for the approved failure policy. Replaying selected older learning is still the
instance's choice; continuing factors does not guarantee avoiding forgetting.

The private completion receipt binds the parent conversion, copied parent
factors and loaded-factor check, as well as the candidate artifacts. Recovery
checks these local artifacts without rerunning training or depending on the
original external parent directory. Parent copies participate in instance
archiving and managed erasure alongside candidate files.

## Worker and candidate checks

The source checkpoint and approval consumption commit before inference closes.
The worker receives the compiled examples and recipe, not the instance database,
full memory store, private journal or retained context. New input remains queued.

Before any gradient step, the worker re-converts the training base to F32 GGUF and
requires its bytes to match the inference model hash. This narrow provenance
proof deliberately rejects quantized inference bases and unrelated training
weights. Loading uses local safetensors and the explicit text-only
`Gemma4ForCausalLM` class; there is no download or custom model-code loading.

The HF tokenizer must reproduce every approved native example and boundary
exactly. Cross-entropy uses the causal shift and ignores all input labels (`-100`),
as described by the [Gemma4 model interface](https://huggingface.co/docs/transformers/model_doc/gemma4).
Only the declared [LoRA factors](https://huggingface.co/docs/peft/en/package_reference/lora)
are trainable. The worker checks frozen base tensors before/after, finite loss and
factors, PEFT save/reload, and exact converted GGUF factor values and alpha.
Selected-example loss is reported at training and deployment scales; there is no
held-out benefit guarantee, automatic early stopping or implicit extra epoch.

The Windows worker process tree uses the [committed-memory limit and watchdog](worker-containment.md).
Candidate files and reports remain private instance-managed artifacts, included
in archives and removed by managed erasure. Original source weights and converter
assets are external dependencies, not owned files to erase.

## Crash, failure and wake behavior

A hash-bound `worker/result.json` is written only after conversion and checks.
`worker/process.json` independently records successful supervision and the
reviewed resource limits. Both are required for candidate recovery. A crash after
those records but before the Candidate phase can resume verification/copying
without retraining. An interrupted worker without both records follows the
recorded failure choice; it is never automatically run again.

Known worker failures return a private reason through the sleep report. If a kill
leaves the amount of completed training unknown, the report says
`unknown_or_partial` rather than claiming no training occurred. A partial or
failed candidate never becomes active. Failure either prepares the previous
native state or keeps the instance stopped, according to its approved plan.

With preauthorized adoption, the supervisor installs the validated adapter and
rebuilds **the exact retained tokens** under those weights, preserving sampler
RNG and runtime state. It executes no historical action frames. The new native
checkpoint and completed transition publish atomically. Review-first instead
prepares the old native state and a candidate report. Neither path automatically
starts generation; the continuous supervisor remains separate work.

## Validation

Dependency-free tests cover plan review, first-adapter and continuation
restrictions, tokenizer and mask mismatch, changed assets, corrupted/incomplete completion, successful
supervision requirements, completed-work recovery, failure choices and erasure.

The optional integration test uses the generated tiny model and injected action
frames. These are tests of approval mechanics, not a real model's consent. It
trains selected text, converts it, adopts it, reconstructs retained tokens/RNG,
and verifies a strict native restore with zero replay, unchanged delivered
messages and preserved queued input. Its expanded native context is a mechanics
fixture beyond the random model's trained context, not a long-context capability
claim.

The first complete run on 2026-09-21 trained 5,120 rank-two parameters for 16
steps, rebuilt all 18,683 retained tokens with unchanged sampler RNG, and restored
the adopted native checkpoint with zero replay. The selected-example loss fell
from 5.6164 to 4.2255 at training scale, and to 5.3824 at deployment strength 0.1.
These are training-example mechanics measurements, not a generalization or
benefit assessment. The full test took about 149 seconds; the gradient steps
themselves took about 0.56 seconds. Native context review/rebuild and conversion
are also part of the cost.

A subsequent native run injected a process-death boundary after successful
worker completion but before Candidate publication. Recovery finished the same
adoption and strict restore with the worker launcher forbidden from running
again. The completion record was unchanged. A separate real Windows timeout
test verified that job termination flows through the approved previous-state
wake choice; a failure-policy interruption preserves the original failure reason.

The continuation integration run on 2026-09-21 loaded all previous factors exactly,
trained the same 5,120 rank-two parameters for 16 steps at strength 0.1, and rebuilt
22,964 retained tokens with unchanged RNG. A strict native restart required zero
replay; queued input and delivered messages were preserved. It also recovered a
completed worker without retraining after the injected supervisor interruption.
The selected-example loss changed from 5.5559 to 4.5603; gradient steps took about
0.58 seconds. These are synthetic mechanics measurements, not evidence of broad
benefit or retained prior learning. A separate actual worker test rejected a
mismatched deployed parent hash before gradient training or candidate creation.

```powershell
.venv\Scripts\python.exe -m unittest tests.test_training.TrainingContractTest tests.test_training.CompletionTest -v
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:DMN_TEST_LORA_PARENT = 'D:\path\to\completed-tiny-training-probe'
$env:DMN_TEST_TRAINING_PYTHON = 'D:\path\to\cpu-training-env\Scripts\python.exe'
$env:DMN_TEST_LORA_CONVERTER = 'D:\path\to\pinned-llama-source'
.venv-gpu\Scripts\python.exe -m unittest tests.test_training.NativeTrainingTest -v
```

## Remaining production gates

Both recipes require exact F32 base provenance. The first-adapter recipe refuses
existing adapters; the continuation recipe requires verified PEFT/GGUF lineage.
Quantized 31B provenance, CUDA training and its resource envelope still need
measured validation with maintenance consent.

For the quantized-base path, a future recipe needs more than matching repository
names: pin the source weights/tokenizer, converter and quantizer implementations,
conversion precision, quantization options and any importance matrix. Reproducing
the inference artifact is the preferred proof; a documented numerical
equivalence policy would need separate review if exact reproduction is not
possible. Existing tiny Q8_0/Q4_0 transfer experiments do not establish that proof
for Q4_K_M. The [published GGUF model card](https://huggingface.co/llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF/blob/main/README.md)
names the modified source model but, when checked on 2026-09-21, did not give a
complete pinned quantization command or importance-matrix provenance. The
[source configuration](https://huggingface.co/llmfan46/gemma-4-31B-it-uncensored-heretic/blob/main/config.json)
also uses `Gemma4ForConditionalGeneration`; the current integration recipe only
loads the validated text-only `Gemma4ForCausalLM` fixture. Production must address
both differences without substituting Google's unmodified weights.

The tiny harness currently regenerates the F32 proof on each run. Production
should establish a reusable, hash-bound provenance record before training, then
invalidate it whenever a bound input or tool changes. Rewriting a full 31B
conversion each sleep cycle would impose unnecessary storage and write costs;
the tiny test's per-cycle proof is not the proposed production storage policy.

Hard disk quotas, Linux containment, resource limits covering the full wake
phase, interactive adoption after review-first, continuous service/frontend
ownership and automatic return to inference remain unfinished. The tiny model
and test-mode gates stay in place until those resource and lifecycle obligations
are met. These limits are visible in the compiled plan; approval cannot enable
unsupported production execution.
