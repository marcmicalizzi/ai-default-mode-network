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
The Windows suite passes 166 tests, including native CPU tests, in 69.6 seconds.
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
Changing `swa_full` cannot directly load the existing full-cache checkpoint into
the compact allocation. Strict restore currently refuses that configuration
change even before attempting native load. It must not silently reconstruct.

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
Existing-instance placement/cache changes still fail strict compatibility checks.

## Shutdown latency

Ordinary shutdown first delivers a maintenance request. Its input evaluation and
the model's decision can take time; acceptance, deferral and refusal remain the
model's choices. Once accepted, the runtime saves once and stops without further
thought generation. The staged 31B run measured a roughly 60-second full save
around 20K occupied tokens, and later saves may be larger/slower.

Staged start and ordinary restore currently make full checkpoints too. Avoiding
those writes requires a separately validated durability change; this patch does
not remove them or weaken action/shutdown checkpoint boundaries. No fixed
shutdown deadline is promised, and closing the terminal bypasses cooperative
shutdown.
