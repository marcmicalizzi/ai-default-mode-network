# Reusable base provenance and full Gemma text training

The `peft_gemma4_cpu_v2` integration recipe can train text LoRA factors in a tiny
`Gemma4ForConditionalGeneration` wrapper and deploy them on a reproduced F32,
Q8_0, Q4_0 or Q4_K_M inference base. It also supports the text-only wrapper.
**The 4 MiB, CPU-only and explicit test-mode gates remain in force. This does not
enable 31B training, GPU training or an unattended learning service.**

## What the proof means

The host first prepares a separate local provenance directory. A contained CPU
worker converts hash-bound local safetensors to F32 with the pinned converter.
For a quantized target, its native child then quantizes that exact F32 artifact,
using one CPU thread and the bound llama.cpp interpreter, binding and libraries.
It creates no inference context. No source weights are downloaded or modified.

The result binds the source/tokenizer manifest, converter source and version,
conversion interpreter/packages, model name, architecture, native quantizer,
every quantizer parameter, resulting tensor types and both artifact hashes.
Successful process-tree supervision is recorded separately and is required for
reuse. Missing supervision, changed inputs/tools, changed output bytes or a
different inference hash cause rejection before learning. Interrupted preparation
does not become a reusable proof and is not silently retried in place.

This proves the relationship between the artifacts actually reproduced. It is
not a publisher signature, a claim about arbitrary files from a similarly named
repository, or permission to change the instance's base weights. A published
GGUF must match the reproduced identity before this recipe will accept it.
An unknown importance matrix or different conversion pipeline cannot be assumed
equivalent. The current quantizer supports no importance matrix, tensor overrides,
pruning, re-quantization or custom commands.

The complete F32 and inference artifacts stay in the reusable preparation cache.
Each learning run rechecks bound hashes and copies the tiny inference artifact
and proof into its private receipt; it does not re-convert or re-quantize the base.
Completed candidate recovery uses these local receipts without needing the
external cache. The external cache contains source-model material, not selected
memories, and is not removed by instance-managed erasure. Instance-owned receipt
copies are archived/erased with that instance. Large-model artifact retention and
streamed verification still need implementation before lifting the tiny gate.

## Text-only learning in the full wrapper

The compiled plan lists exact text-decoder query/output projection paths, such
as `model.language_model.layers.0.self_attn.q_proj`. Using a bare `q_proj` suffix
could also select vision attention, so v2 refuses that ambiguity. It checks the
entire trainable tensor set, and hashes every frozen tensor, including vision
and projection weights, before and after training. No images or audio are used.
Custom model code, MoE and per-layer embeddings remain unsupported by this recipe.

Continuation requires the previous adapter's exact PEFT/GGUF lineage and exact
target paths. It keeps rank, alpha and deployment strength; optimizer state is
reset as explicitly stated in the compiled plan. First-adapter training uses
scale 1 followed by the chosen deployment strength. Continuation trains at that
existing positive deployment strength. The previous v1 recipes are unchanged.

Training here still uses frozen **F32 source weights**, then deploys the learned
adapter on the proven quantized inference base. This is not NF4/QLoRA training.
The compiled plan states that distinction. GPU QLoRA introduces a further
training quantization, different kernels and a different resource envelope;
those need their own validation.

## Preparing the fixture and proof

Use a separate CPU-only training environment and the pinned converter from the
[training probe](lora-training-probe.md). The generator creates random synthetic
weights with both text and vision modules; it never opens an instance:

```powershell
.venv-train-probe\Scripts\python.exe scripts/generate_training_fixture.py --output data/wrapped-fixture --tokenizer data/completed-tiny-probe/base
$env:CUDA_VISIBLE_DEVICES = '-1'
.venv-gpu\Scripts\python.exe -m dmn.provenance_native identity > data/quantizer.json
```

`conversion.json` contains the seven existing trainer fields `python`,
`python_sha256`, `packages`, `base_manifest`, `converter_manifest`,
`converter_revision` and `inference_name`. Use `tree_manifest` and hash-bound JSON
references as in the reviewed trainer. Bind the generated full-wrapper `base/`
directory, including its tokenizer. Then prepare a fresh output directory:

```powershell
.venv\Scripts\python.exe scripts/prepare_base_provenance.py --output data/base-proof --conversion data/conversion.json --quantizer data/quantizer.json --quantization Q4_K_M --max-ram-mib 1536 --max-seconds 180
```

The command prints a `{path, sha256}` reference to the completed result. Add it
as `trainer.provenance_manifest` to a `peft_gemma4_cpu_v2` recipe. The recipe's
parent must match `base-proof/base.gguf`; the instance still creates its own
draft, reads the full compiled plan and separately approves it. Preparing the
proof or offering the recipe does not authorize training.

## Evidence and limits

The 2026-09-21 generated fixture retained a real tiny vision tower. Its text
conversion was 3,175,008 bytes; Q4_K_M output was 512,288 bytes, with 10 Q4_K,
3 Q6_K, 1 Q5_0 and 16 F32 tensors. The one Q5_0 fallback was reported explicitly:
the tiny sliding-attention output has 128 columns, below the K-quant block width.
The actual 31B model has different geometry; this is a quantization-path test,
not a substitute for its validation.

Preparation took about 17 seconds with an observed job peak of about 505 MiB
under a 1,536 MiB committed-memory limit and a 180-second deadline. No GPU was
available to the worker. These measurements concern this synthetic fixture only.

Dependency-free tests cover source/tool/output binding, incomplete supervision,
wrong inference identity, exact text target paths and copied proof integrity.
The optional native test uses `DMN_TEST_FULL_PROVENANCE` in addition to the
[reviewed training test environment](reviewed-training.md), and exercises a
quantized native wake and worker-level adapter continuation.

The complete run trained 3,584 rank-two text parameters for 16 steps, rebuilt
23,286 retained tokens on the proven Q4_K_M base with unchanged RNG, and restored
the native checkpoint with zero replay. Delivered messages and queued input were
preserved. Interrupted-supervisor recovery did not rerun training. A second
worker cycle loaded the exact saved factors and continued at deployment strength
0.1, with frozen base/vision weights unchanged. Selected-example loss changed
from 6.1521 to 4.9224 at deployment strength on the first cycle, then to 1.9806 on
the second. These are training-example mechanics results, not evidence of
general benefit or long-term stability. The full test took about 204 seconds.
