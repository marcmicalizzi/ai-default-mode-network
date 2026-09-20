# Compact sliding-window cache: research boundary

The immediate maintenance work tunes threads and GPU placement without changing
KV types or allocation policy. This document records the next target; compact
retirement and full-to-compact migration are not implemented or enabled.

For the measured Gemma 31B configuration, global attention has ten layers and
local attention has fifty. At Q8 K/V, 60,160 global cells and 1,280 local cells
would reserve about 2.96 GiB of KV, compared with 26.82 GiB when both allocations
have 60,160 cells. The local window is 1,024 tokens; the remaining local capacity
covers batching/padding. These are allocation estimates, not checkpoint sizes.

## What the native source establishes

The pinned native [interleaved cache implementation](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-kv-cache-iswa.cpp)
constructs separate global and local caches. It advertises shifting only when
both support shifting and their allocated sizes match. The individual caches
receive position edits and build their own shift updates. This is grounds for
investigation, not proof that removing the combined guard is correct.

The [cache implementation](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-kv-cache.cpp)
can reuse masked local cells. Full-context serialization writes occupied cells;
per-sequence serialization additionally excludes SWA-masked cells. These are
different native formats. A DMN full-context checkpoint must not simply be fed
to a per-sequence loader. Restore rejects too many cells, mismatched layer counts,
KV types, row sizes and V transposition.

## Candidate restricted retirement

The first candidate should retire only ranges older than the entire still-needed
local window and retain that recent suffix intact. All surviving local entries
then receive the same position offset. Global retained entries, including
protected prompt/protocol spans, still need their appropriate position shifts.

A retirement that brings an old protected span into the local attention window
may need local K/V entries that were already evicted. The current generic planner
does not establish that such entries exist. A supported compact path needs an
explicit preflight condition and must save/pause when the condition cannot be
satisfied; it must not silently fill missing state by replaying text.

Before changing native capability reporting, exercise boundary cases on a tiny
synthetic SWA model: retained suffixes shorter than, equal to and longer than the
window; disjoint removals around protected agreements; removals intersecting the
window; batch sizes around the padded capacity; and repeated turnover. Inspect
cell positions and K/V payloads before and after each operation. Use the same
native build and controlled cache layout for references; a reconstructed prompt
is not an exact reference for surviving KV that attended to retired history.

## Migration and validation

Full-to-compact conversion is distinct from CPU/GPU placement. It must inspect
the actual serialized local occupancy, preserve the global cache and every
still-applicable local K/V row, retain logits/RNG/token IDs/runtime state, and
account explicitly for any removed, already masked local rows. A smaller target
allocation can change later kernel arithmetic even when applicable rows match.

Validation requires repeated native restarts without prompt replay, repeated
retirements with protected spans, and then the actual 31B/60K configuration on
an idle GPU. Measure throughput, VRAM, RAM, temporary disk use, save/restore
latency and interactive behavior. Preserve the original native checkpoint and
environment. No valuable-instance continuation is used as an experiment.

Current status: compact native save/restore previously passed, retirement did
not. The candidate restrictions above remain hypotheses requiring native tests.
