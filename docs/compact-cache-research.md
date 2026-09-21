# Experimental compact Gemma4 cache

Compact sliding-window retirement is available behind `experimental_compact_swa`,
disabled by default. Full-to-compact conversion has passed synthetic native tests,
but remains a research harness: there is **no instance migration command** and
ordinary recovery rejects changing cache allocation. No valuable instance was
used in this investigation.

## Why it matters

The tested Gemma4 31B has ten global-attention layers and fifty local-attention
layers, with a 1,024-token local window. At Q8 K/V and 60,160 actual context
capacity, allocating 60,160 cells for both caches reserves about 26.82 GiB of KV.
Compact allocation uses 60,160 global cells and 1,280 local cells (window plus
batch/padding), about 2.96 GiB. Global context capacity stays the same. These are
allocation estimates, not checkpoint sizes.

On an RTX 5090, compact allocation fit the entire 31B Q4_K_M model on the GPU.
A 2026-09-20 synthetic benchmark with 25,000 occupied tokens, 60K requested
capacity, batch 256, Q8 K/V and 18 CPU threads measured **45.19 tokens/sec** over
64 steps after eight warmup steps. The earlier full-cache hybrid configuration
(30 GPU layers, 18 threads) measured **2.24 tokens/sec** under that workload.
These are single-run measurements, not a guarantee across hosts or workloads.
Other desktop GPU allocations varied between runs.

Whole-device snapshots around compact decoding reported 27,275 MiB used and
4,916 MiB free, including other applications. Load took 62.7 seconds, prefill
65.8 seconds and warmup 24.2 seconds; steady speed does not describe startup
latency. One post-decode power sample was 403 W, not an average or a configured
power limit. Existing token pacing remains available.

## Retirement policy

The pinned native [interleaved cache implementation](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-kv-cache-iswa.cpp)
constructs separate global and local caches. Its combined capability check
advertises shifting only when their allocated sizes match, although the two
individual caches receive position edits and build their own shift graphs.

The experimental policy deliberately overrides that combined guard only for
Gemma4 STANDARD SWA on the pinned llama-cpp-python 0.3.35 implementation
(llama.cpp `4df29be4f4c3673f428170fda944a5b19f743bb8`). It requires flash attention,
packed checkpoints and F16/Q8 KV, excludes shared-KV variants, and checks the
head geometry needed by quantized K shifting. It needs no upstream fork.
A custom native build must retain the reviewed implementation; the Python
binding's version string alone cannot certify a custom binary's source.

Every retirement preserves the **entire recent local window as a contiguous
suffix**. This keeps previously masked, possibly evicted local entries from
becoming relevant when positions shift. The planner also preserves the existing
initialization prefix, import protocol and adopted prompt agreement. It validates
all disjoint removal ranges before any removal, using progressively shifted
coordinates. If those protections leave insufficient space, the runtime saves
and pauses at `context_full`; it does not relax the window or replay missing KV.
Direct backend shifts enforce the same suffix restriction.

For a new disposable Gemma4 test, these settings enable the policy:

```json
{
  "swa_full": false,
  "experimental_compact_swa": true,
  "flash_attn": true,
  "type_k": "q8_0",
  "type_v": "q8_0",
  "pack_checkpoints": true
}
```

Use [the complete example](../examples/gemma4-compact-experimental.json) with your
own model path and hardware settings. `swa_full: false` alone does not opt into
the override. Full-cache defaults and existing configurations remain unchanged.
Strict restore, including the placement-change option, refuses a full-to-compact
change.

## Conversion evidence

The research codec reads the pinned full-context session-v9 format, requiring
Gemma4, one sequence, non-transposed F16/Q8 KV, complete global positions and a
complete recent local window. It rejects unsupported or malformed layouts.
It streams a new file, preserving the global section verbatim and copying every
still-applicable local K/V row byte for byte. Only local rows already masked by
STANDARD SWA are removed. It never overwrites its source or evaluates tokens.
Token IDs, logits and sampler RNG sidecars stay unchanged.

In the 31B synthetic test:

- 4,096 occupied tokens at the same 60K allocation were converted from full
  cache/30 GPU layers to compact cache/full GPU offload.
- The native file changed from 1,960,953,302 to 623,982,038 bytes. All 4,096 global
  cells and 1,024 recent local cells survived; 3,072 masked local cells were
  removed. Every retained K/V row was compared byte for byte.
- The compact target restored with zero decode calls or prompt replay. Its
  reserialized file exactly matched the converted native file.
- Three cycles each retired two disjoint 256-token ranges and refilled 512
  tokens. Surviving V rows stayed byte-identical; K deliberately undergoes native
  RoPE shifting. An additional offline check required all still-applicable rows
  to remain present.
- A fresh-process restart after retirement reproduced 16 sampled tokens and all
  logits exactly (maximum absolute difference 0.0).
- Conversion copying took 0.62 seconds; native load plus byte verification took
  3.89 seconds. The final 4K-occupied save took 2.14 seconds. These exclude model
  construction and are not estimates for a larger occupied context.

The first shift/decode took 26.4 seconds; later cycles took 1.82 and 1.46 seconds.
The native CUDA workspace grew from its initial estimate of about 544 MiB to
941 MiB during shifting. Keep headroom beyond the initial allocation; these
point samples did not measure peak VRAM, RAM or temporary disk use.

The first 31B retirement probe used an isolated research guard override. The
subsequent 25K benchmark used the implemented opt-in policy. Tiny random-weight
Gemma4 fixtures separately passed F16/Q8 conversion, eight retirement/refill
cycles, fresh-process restart, and runtime planning around protected spans.
A matched full-cache reference from the **same initial KV** also reproduced
the surviving original rows byte for byte after three shifts, including K.
This avoids treating independently reconstructed context as an exact reference.

These checks establish retained-state transfer and target restart, not identical
future generation between different cache allocations or GPU placements.
Allocation changes can change attention arithmetic even with identical K/V.
No cross-allocation continuation guarantee is made. Linux, other model variants
and long-duration operation still need their own validation.

## Reproducing the research

`scripts/generate_swa_fixture.py OUTPUT.gguf` creates a 1.4 MB deterministic
random-weight Gemma4-shaped fixture; it downloads no model and has no meaningful
language ability. Use it with a disposable full-cache config: 2,048 context,
batch 64, CPU only, one thread, `offload_kqv: false`, flash attention, Q8 K/V,
packed checkpoints and an empty system prompt.

```sh
python scripts/probe_compact_cache.py --config data/tiny-full.json --output data/tiny-compact-probe
```

The harness only creates fresh synthetic state; it accepts no instance or input
checkpoint. Output must be a new directory. It checks ports 8765 and 8766 before
heavy phases; configure `--guard-ports` for every local runtime if different.
Tiny single-thread CPU fixtures are exempt. The parent closes its native model
before launching the fresh-process restart check.

The standard-library suite tests format rejection, exact masking boundaries,
protected-span planning, preflight failure and recovery isolation. For native
runtime/reference tests, set `DMN_TEST_SWA_MODEL` to the generated tiny fixture
and run `python -m unittest tests.test_compact_native -v`. Raw local reports,
checkpoints, model paths and configurations are excluded from Git.

## Before converting an existing instance

A supported migration still needs a lifecycle-aware command that preserves the
source checkpoint and environment, checks model/build identity and all snapshot
hashes, preserves runtime/database state and holds, records the conversion, and
verifies target native bytes without starting inference. The research codec
alone does not provide those protections and must not be used to edit a live
instance. A disposable hands-on UI/maintenance trial and a longer pressure soak
remain necessary before valuable-instance adoption. No existing checkpoint or
launch configuration was converted as part of this work.
