# Performance before adopting a valuable conversation

Try a disposable instance interactively with the intended model, placement and
occupied context before importing a valuable conversation. Measure input latency,
steady generation, message publication and an agreed shutdown/save. Correct
checkpoint restoration alone does not establish usable performance. A 60K
capacity setting is different from 60K occupied tokens.

## Native diagnostics and sampling

Native logging defaults to warnings and errors. Debug graph messages are filtered
before terminal writes; diagnostic continuations retain their original severity.
Use `--native-log-level debug` on `dmn run`, or set `DMN_NATIVE_LOG_LEVEL=debug`,
when investigating the native backend. `info`, `warning` and `error` are also
supported. Logging is process-wide and does not enter the inference fingerprint.
No native-library files are changed.
The filter uses the [pinned native log-level values](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/ggml/include/ggml.h),
which differ from the binding's outdated Python logger definitions. An actual
native-emission test verifies that debug/info are hidden and warning/error
messages and their continuations remain visible.

For nonzero top-k smaller than the vocabulary, the sampler selects candidates
without sorting the entire vocabulary. Boundary ties retain ascending token ID
order, matching the former stable sort. Penalties, filter order, probabilities
and RNG use are unchanged. Top-k zero still sorts the full vocabulary.

On a synthetic 262,144-entry score vector with top-k 64, 30 alternating runs
measured candidate selection at 32.36 ms before and 2.75 ms after. This is one
CPU measurement, not model throughput; it does not resolve CPU layer offload.
Ordering/tie tests and both sampler chains match the previous implementation.
A native CPU fixture preserves fingerprint, restored KV, RNG and eight subsequent
tokens/logits exactly across logging verbosity changes, with zero prompt replay.
The expanded Windows suite passes 184 tests, including native CPU tests, in 84.0 seconds.
A separate CPU checkpoint created by the original production code also restored
strictly under the proposed code and matched 24 subsequent tokens and logits
exactly, without prompt replay. These checks use disposable state only.

## A guarded diagnostic benchmark

Use a fresh output directory and a local model configuration with an empty
`system_prompt`. This harness never opens an instance or executes generated actions:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/benchmark_inference.py --config data/benchmark.local.json --output data/benchmark-01 --tokens 20000 --warmup 8 --steps 64
```

It records model load, synthetic prefill, decode warmup, steady generation,
Python sampling and native decoding/logit-copy time separately. Reports include
the native fingerprint and whether retirement is supported. They exclude snapshot
and message-publication costs, which require a separate interactive trial.
Debug output can still be expensive; compare logging levels explicitly if needed.

The harness refuses a competing model while the loopback DMN port (default 8765)
is open. Set `--guard-port` to the actual runtime port. The only exemption is a
single-thread CPU fixture with at most 4K context and a model no larger than
16 MiB. This guard is a preflight check, not a GPU reservation or a way to detect
every unrelated process. Check resource use and keep other model servers stopped
before running a large comparison. A tiny CPU fixture exercised the harness while
the primary instance remained active; that fixture alone establishes no 31B speedup.

For an idle-machine comparison, keep the model, occupied token count, quantization,
sampling settings, warmup, power settings and other workloads comparable. Record
the native build: the installed llama-server and Python binding may differ.
Compare the existing hybrid allocation, different CPU thread counts/placement,
and compact-cache full GPU offload as separate experiments. The compact case now
has bounded experimental retirement support; long-duration continuous operation
still needs validation, as detailed below.

### Sequential thread and placement trials

`benchmark_matrix.py` writes a plan without loading a model by default. After an
agreed shutdown, add `--execute` with a new output directory to measure it:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/benchmark_matrix.py --config data/benchmark.local.json --output data/performance-plan --threads 8 12 --gpu-layers 27 30
# Run only on an idle machine, using a fresh output directory:
.\.venv-gpu\Scripts\python.exe scripts/benchmark_matrix.py --config data/benchmark.local.json --output data/performance-measured --threads 8 12 --gpu-layers 27 30 --execute
```

The default occupied context is 25K tokens, with eight warmup and 64 measured
steps, repeated twice per case. The original configuration is included. Thread
counts are compared first at the original placement; GPU-layer trials use the
fastest measured thread count. A final original-baseline recheck exposes drift
from changing background load or thermals. Large drift calls for investigation,
not treating the fastest result as established. Each trial uses a fresh process,
rechecks the runtime port, and frees its model before the next trial. A failed
trial stops the matrix. Nothing promotes the winner or edits an instance.
These repeated prefills can take substantial time; `--repeats 1` is a preliminary
screen, not the same evidence as a repeated comparison. Individual reports now
include CPU time/core equivalents and instantaneous GPU memory/utilization/power.

For the measured 31B/60K layout, 24 to 30 GPU layers adds roughly 1.60 GiB of
weights plus 2.68 GiB of KV, excluding changes in compute buffers and backend
representation. This is a sizing estimate, not a verified fit or speedup.

### Measured desktop comparison (2026-09-20)

A preliminary screen on an RTX 5090 with an i9-10980XE (18 physical cores),
the pinned 31B Q4_K_M model, 60K capacity, full SWA allocation and Q8 K/V used
25K synthetic occupied tokens, eight warmup tokens and 64 measured tokens per
case. Every case used the proposed logging/sampling code; the baseline refers
to the original placement settings. Each model ran in a separate process.

| GPU layers | CPU threads | Tokens/s |
| --- | --- | --- |
| 24 | 6 | 1.10 |
| 24 | 8 | 1.46 |
| 24 | 12 | 1.68 |
| 27 | 12 | 1.91 |
| 30 | 12 | 2.03 |
| 24 | 6, baseline recheck | 1.08 |
| 30 | 18, additional comparison | 2.24 |

The baseline recheck differed by about 2%; the fastest case was about 2.04 times
the initial baseline. This was one measured run per setting, not a repeated
statistical comparison. Small CPU diagnostics overlapped parts of the initial
screen, including the 30-layer/12-thread decode measurement. Backup and suite
work finished before the final 18-thread decode measurement. Startup timings
are not controlled comparisons; repeat with no competing work before treating
small differences between nearby settings as established.

The 27-layer case left about 4.6 GiB of free VRAM, versus about 2.4 GiB at
30 layers. These instantaneous device readings include desktop applications.
Retirement can need additional working memory; fitting normal inference alone
does not establish a usable continuous-run placement. These results do not
recover the former full-GPU server throughput or validate compact cache.

### Adaptive placement direction

DMN currently uses an explicit GPU-layer count. A future startup policy could
choose among measured placements using free VRAM, a configurable VRAM reserve,
and a system-RAM ceiling. Fitting the most layers is not proof of best throughput;
benchmark the candidates and retain the resolved settings in each checkpoint.
Full-cache allocation depends on capacity, not just the occupied token count.

The pinned llama.cpp revision already has
[startup fitting support](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/common/fit.h)
in its common C++ helpers. Our Python backend does not invoke it. That fitter
assumes unlimited system memory and can adjust unset parameters, so directly
adopting its defaults would not establish DMN's resource or continuity contract.
Keep context capacity, cache format and inference identity fixed; change only
the supported placement settings and apply native-state verification on restore.

Changing placement during inference requires more work: reserve resources,
reach a consistent boundary, preserve state, reload with the new placement and
verify it before continuing. Frequent fluctuations should not trigger repeated
large checkpoint writes or reloads; such a policy would need separate thresholds,
a cooldown and accounting for temporary RAM/disk use. It must never use prompt
reconstruction as an automatic escape from a failed transfer. Exact saved-state
transfer also does not guarantee identical future CPU/GPU arithmetic.

These are design requirements, not an implemented auto-fit or live-migration
feature. The current opt-in below is the bounded primitive being tested first.

### Native restoration after a placement change

An explicit `run --allow-placement-change --config ...` permits changes only to
`n_threads` and `n_gpu_layers`, alongside the already supported scheduling
settings. It requires strict recovery and the same model, native binaries,
platform, cache types, capacity, batch size, prompt and sampler. Compact SWA,
new kernels/builds, and other layout changes remain excluded. Hold/end gates
still apply; this flag cannot claim a hold's `original_environment` condition.

Before any resume input or inference, the loader restores native state and
reserializes it to temporary storage. The complete native session file must
match the original SHA256, and tokens, RNG, decoded-token count and logits must
match their saved sidecars. Failure stops startup without reconstruction. The
temporary file costs one additional snapshot-sized write and free-space reserve
for this explicit verification. Ordinary strict restores do not add this write.
The resume event records the changes and verification evidence. Preserving
stored bytes does not guarantee identical future CPU/GPU floating-point results.

First run a disposable probe with the intended winning placement:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/probe_placement.py --config data/benchmark.local.json --output data/placement-check --threads 8 --gpu-layers 30
```

The numbers here are examples, not a selected winner. The probe creates only
synthetic state, verifies byte-preserving native transfer without replay,
reports numerical differences with identical subsequent token inputs, and
tests three retirements plus restart under the target placement. It runs no
runtime actions and never opens an instance. The heavy-run guard also applies.
Only after this check and an interactive disposable trial should a measured
placement be applied to valuable state. Preserve the original checkpoint and
environment before migration.

A tiny CPU fixture has passed thread-count migration, serialized-byte equality,
zero replay, retirement and subsequent restart. The 2026-09-20 31B probe also
passed transfer from 24 GPU layers/6 threads to 30 GPU layers/18 threads, using
4,096 synthetic occupied tokens and the same 60K allocation. Restore took
14.2 seconds including verification, preserved the full native file byte for
byte, and performed zero decode calls or prompt-token reevaluations during load.
Three retirements succeeded. Reloading the resulting checkpoint into a fresh
native context under the target placement matched all 16 sampled tokens and
logits exactly. This was native-context recreation within one diagnostic process;
the separate native CPU suite also covers process restart.

Across the old and new placements, one of 16 sample comparisons differed
(zero-based step 10), with a maximum absolute logit difference of 1.8165 under
identical forced subsequent token inputs. Saved-state equality is therefore not
identical future generation across placements, and this difference must not be
described as merely cosmetic. A changed sampled token would alter later inputs
in an ordinary uninterrupted continuation. Whole-device sampling begun during
retirement observed at least 2,233 MiB free VRAM, but may have missed transient
peaks or the start of the first cycle.

The 4K transfer/retirement probe and 25K throughput comparison establish different
things. Neither replaces actual checkpoint verification on an opted-in resume,
nor the hands-on fresh-instance trial before resuming valuable state.

## Experimental compact Gemma4 cache

The opt-in `experimental_compact_swa` policy now permits bounded retirement with
compact allocation. It preserves the entire recent sliding window, validates
all removal ranges before mutation, and pauses if protected spans leave too
little room. It is restricted to the reviewed pinned Gemma4 implementation and
F16/Q8 geometry, and requires flash attention and packed checkpoints.

A synthetic 31B Q4_K_M/Q8 test on the RTX 5090 measured **45.19 tokens/sec** with
full GPU offload, 25K occupied tokens and 60K requested capacity. The previous
full-cache hybrid configuration measured **2.24 tokens/sec**. Compact allocation
reduces estimated KV reservation from 26.82 GiB to 2.96 GiB while preserving the
global context capacity. These are individual runs, not a general speed guarantee.

Full-to-compact conversion also passed on a separate 4K-occupied synthetic
checkpoint: every retained K/V row was byte-identical, loading reevaluated zero
tokens, three retirements succeeded, and a fresh-process restart matched the next
16 tokens and logits exactly. This does not guarantee identical future arithmetic
between the old and new allocations or GPU placements.

See [compact-cache evidence and limitations](compact-cache-research.md) for
measurements, the opt-in example and reproduction. The conversion is still a
research harness, **not an existing-instance migration command**. Ordinary
strict restore, including the placement opt-in, continues to reject a cache
allocation change. A lifecycle-aware migration and disposable interactive trial
remain necessary before using this for valuable existing state.

## Shutdown latency

Ordinary shutdown first delivers a maintenance request. Its input evaluation and
the model's decision can take time; acceptance, deferral and refusal remain the
model's choices. Once accepted, the runtime saves once and stops without further
thought generation. The staged 31B run measured a roughly 60-second full save
around 20K occupied tokens. Later snapshots can also shrink as masked local-cache
cells are reused: an observed 25.8K-token sleep checkpoint was 1.84 GB and took
12.45 seconds. Occupied context alone is not enough to predict save size or time;
the large preallocated KV buffers remain allocated even when snapshots shrink.

Staged start and ordinary restore currently make full checkpoints too. Avoiding
those writes requires a separately validated durability change; this patch does
not remove them or weaken action/shutdown checkpoint boundaries. No fixed
shutdown deadline is promised, and closing the terminal bypasses cooperative
shutdown.
