# Training from a reviewed learning plan

The `peft_gemma4_cpu_v1` recipe connects compiled-plan approval to actual PEFT
training, GGUF conversion, durable candidate validation and native context
reconstruction. It uses the same sleep transition engine as the prebuilt-adapter
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

Training uses scale 1; the approved deployment scale is applied separately and
both selected-example losses are reported. A small deployment strength is not a
bound on unrelated behavioral change. Replay consists only of examples explicitly
included in the plan, not automatically retrieved earlier memories or messages.

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

Dependency-free tests cover plan review, first-adapter restrictions, tokenizer
and mask mismatch, changed assets, corrupted/incomplete completion, successful
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

```powershell
.venv\Scripts\python.exe -m unittest tests.test_training.TrainingContractTest tests.test_training.CompletionTest -v
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:DMN_TEST_LORA_PARENT = 'D:\path\to\completed-tiny-training-probe'
$env:DMN_TEST_TRAINING_PYTHON = 'D:\path\to\cpu-training-env\Scripts\python.exe'
$env:DMN_TEST_LORA_CONVERTER = 'D:\path\to\pinned-llama-source'
.venv-gpu\Scripts\python.exe -m unittest tests.test_training.NativeTrainingTest -v
```

## Remaining production gates

This first recipe accepts an unadapted F32 base only. It refuses an existing
adapter rather than silently discarding earlier learning. A continuing-adapter
recipe must bind the corresponding PEFT and deployed GGUF lineage and make its
optimization/scaling policy explicit. Quantized 31B provenance, CUDA training and
its resource envelope need measured validation with maintenance consent.

Hard disk quotas, Linux containment, resource limits covering the full wake
phase, interactive adoption after review-first, continuous service/frontend
ownership and automatic return to inference remain unfinished. The tiny model
and test-mode gates stay in place until those resource and lifecycle obligations
are met. These limits are visible in the compiled plan; approval cannot enable
unsupported production execution.
