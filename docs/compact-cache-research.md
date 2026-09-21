# Experimental compact Gemma4 cache

Compact sliding-window retirement is available behind `experimental_compact_swa`,
disabled by default. Full-to-compact conversion has passed synthetic native tests
and is available through the separate offline `migrate-cache` command. Ordinary
recovery still rejects changing cache allocation. Validation used disposable
instances and synthetic state; it did not use private cognition as test input.

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

## Pressure and interface trials

The full 31B compact backend completed twelve retirement/refill cycles near
55,000 occupied tokens in 663 seconds. It retired 267,135 tokens cumulatively,
preserved protected spans and the recent local window, and measured
37.01–38.21 tokens/sec in its decode segments. Saves and byte-verified reloads
passed on cycles 4, 8 and 12. A fresh-process final restore loaded 55,040 tokens
without decoding or replay, then matched sixteen sampled tokens and all logits
exactly (maximum difference 0.0). This is not a multi-day reliability test.

`scripts/soak_compact_cache.py --config FULL_COMPACT_CONFIG --output NEW_DIRECTORY
--tokens 55000 --cycles 12 --seconds 1800` reproduces that workload with an empty
system prompt. It uses only new synthetic state and the production retirement
planner. The same runtime-port guard as the research probe applies.

A separate disposable 31B instance passed DMN browser message delivery, paragraph
preservation and model-accepted shutdown with no rejected actions. The requested
reply arrived in about six seconds; the final checkpoint saved in about three
seconds. Startup took about 2m40s. This tested the DMN page, not the Open WebUI page.

## Offline instance migration

Stop the instance through its agreed maintenance path first. For the tested
5090 configuration (all layers on GPU, eighteen CPU threads):

```sh
python -m dmn migrate-cache --instance data/instance --backup data/before-compact --gpu-layers -1 --threads 18
```

Choose placement for your hardware. Omitting placement arguments keeps the saved
values. Configuration is derived from the latest committed checkpoint, with only
`swa_full` and `experimental_compact_swa` changed. An optional `--config` must
otherwise match; sampling, prompt, context capacity, KV precision, cadence and
all unrelated settings cannot change in this operation.

The command takes the instance lock and requires an open lifecycle with a saved
`suspended`, `staged` or `context_full` state. It refuses held and ended instances.
It verifies all registered snapshots and the model/native environment before
conversion, requires a new backup directory outside the instance, and checks
available disk space. It does not construct a Runtime, sample or evaluate tokens.

The backup includes a verified SQLite copy, lifecycle, all registered checkpoints,
the original import archive and DMN source. Its renamed database and lifecycle
make it a recovery artifact rather than an executable instance. `preservation.json`
contains the inventory and restoration instructions. **Model files and native
installations are preserved in place, not bundled in this backup.** Keep those
installations for rollback; this is not yet a portable Linux environment archive.

The new native file is compared against every retained source KV byte, loaded by
the compact backend with zero decode calls, and reserialized for exact byte
verification. Engine/token/RNG and logits sidecars are copied unchanged. Runtime
state and existing database contents remain intact. A single SQLite transaction
publishes the new checkpoint and migration record only after verification and
native teardown succeed. Failure keeps the old checkpoint selected; recovery
copies and any unreferenced candidate remain available for inspection.

Success leaves the instance stopped. The normal launch command then uses the
new saved configuration, with strict native recovery. At actual resume, a factual
event tells the instance what changed and that future arithmetic can differ.
The source checkpoint remains untouched; normal later checkpoint pruning may
retire its in-instance copy, while the separate backup remains until explicitly
removed. The migration itself performs no pruning or automatic restart.

Unit tests cover rollback, identity/configuration rejection, lifecycle/hold/lock
refusal and database/sidecar preservation. `tests.test_cache_migration_native`
adds a complete tiny native Runtime migration and verifies the one-time resume
notice. Set `DMN_TEST_SWA_MODEL` to the random-weight fixture to run it.
