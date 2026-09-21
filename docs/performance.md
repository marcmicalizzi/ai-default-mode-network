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
The expanded Windows suite passes 183 tests, including native CPU tests, in 76.1 seconds.
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
the primary instance remained active; no 31B performance improvement is claimed yet.

For an idle-machine comparison, keep the model, occupied token count, quantization,
sampling settings, warmup, power settings and other workloads comparable. Record
the native build: the installed llama-server and Python binding may differ.
Compare the existing hybrid allocation, different CPU thread counts/placement,
and compact-cache full GPU offload as separate experiments. The compact case is
a diagnostic candidate, not a validated continuous-run configuration.

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
zero replay, retirement and subsequent restart. GPU placement and 31B throughput
still require the idle-machine trials; they are not established by that fixture.

## The Gemma compact-cache obstacle

The 60K hybrid layout allocates about 26.8 GiB of Q8 KV across RAM and VRAM before
model weights and working buffers. Only 24 of 61 layers are on the GPU. The
original server used full model offload and the default compact sliding-window
allocation. These have very different resource costs despite the same model,
Q8 types and requested context capacity.

The [pinned native implementation](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-kv-cache-iswa.cpp)
requires equal global and sliding-window cache sizes before advertising context
shifting. The upstream main source checked on 2026-09-20 retains that condition.
Compact-cache native restore already passed earlier tests; retirement did not.
Removing this guard alone would not establish safe retirement.

There is also a conversion problem: the [native state reader](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-kv-cache.cpp)
rejects a saved cache containing more cells than the destination allocation.
Whether a particular full-cache snapshot exceeds compact capacity depends on
its serialized occupied cells, not its original allocated capacity. Small
snapshot size alone does not prove compatibility. Strict restore refuses this
configuration change even before attempting native load, including when the
placement opt-in is used. It must not silently reconstruct.

A candidate solution still needs these proofs on disposable state:

1. Retirement handles the compact cache's retained sliding window, protected
   spans, position changes and repeated turnover without reading evicted cells.
2. Full-to-compact conversion preserves every still-applicable K/V value, global
   cache, retained token IDs, logits, sampler RNG and runtime state, while
   explicitly accounting for local cache entries it removes.
3. Native restart and subsequent retirement preserve the converted state without
   prompt replay. Compare continuation against an appropriate reference, not a
   reconstructed prompt presented as exact restoration.
4. The 31B/60K configuration passes a hands-on trial with uncongested VRAM and
   its measured inference, input and snapshot costs are acceptable.

No such conversion or native guard override is enabled by this performance patch.
See [compact-cache-research.md](compact-cache-research.md) for the bounded native
retirement investigation. Ordinary placement changes still fail strict checks
unless explicitly opted into the byte-verifying path above; cache changes always
remain outside that path.

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
