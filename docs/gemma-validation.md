# Gemma 31B validation

These experiments use fresh diagnostic contexts, never inference on General
Greeting. Evidence directories retain unsuccessful runs as well as passes.

## Pinned environment

- Windows, RTX 5090, 32 GiB VRAM, approximately 64 GiB system RAM.
- `llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF:Q4_K_M`;
  18,687,063,168 bytes, SHA-256
  `7c65a35e7c4e53cba6c5e02cc9eeb850eb4251f4d9ad120c2caa6de23c5a6395`.
- `.venv-gpu`, llama-cpp-python 0.3.35, CUDA 13.2, native source
  `4df29be4f4c3673f428170fda944a5b19f743bb8`, NumPy 2.4.6.
- Q8_0 K and V cache, Flash Attention enabled. Each report records full native
  binary hashes, actual context allocation and sampler settings.
- Separate Qwen/Open WebUI soak and desktop applications shared the machine.
  Whole-device GPU readings include those applications.

## Seed rendering

Gemma's full template is unsupported by the binding's simple
`llama_chat_apply_template` path. Fresh Gemma trials use the GGUF template through
`Jinja2ChatFormatter`, with its implementation hash and Jinja2 3.1.6 recorded.
The renderer avoids a duplicate BOS when native tokenization adds it. Four
plain/Unicode/thinking cases matched the installed native llama-server renderer
exactly; see `.test-artifacts/gemma-renderer/seed-parity.json`.

`prompt_format=jinja` and `jinja_thinking=true` are explicit settings for the
pressure trial. An imported seed instead uses its archived, server-rendered
prompt and token IDs, bypassing this fresh-seed rendering path.

## Retirement and checkpoint packing

A short 4K native test initially passed both unshifted and shifted restoration.
The stronger repeated-retirement trial then found a mismatch: after normal
sliding-window decoding between retirements, restored continuation differed by
up to 1.513956 in logits. That run is preserved in
`data/gemma31b-pressure-02`; it is a failed continuity test.

The controlled layout probe reproduced a 1.148633 maximum difference, despite
the 24 sampled tokens happening to match. Packing the current native state
before saving removed the discrepancy in three probe cycles, each with zero
logit difference. Serialized native bytes were unchanged and no prompt tokens
were reevaluated (`data/gemma-layout-probe-01/report.json`). The evidence
isolates physical layout as relevant; it does not identify a particular CUDA
kernel operation as the cause.

Gemma configurations now set `pack_checkpoints=true`. This packs through native
state serialization before every save, as well as after explicit retirement.
It retains K/V values and positions without replay. It does change the live
physical layout, so the validated continuation is the runtime using this policy,
not a promise to reproduce an unmodified server's arithmetic. Packing requires
a temporary state buffer and adds checkpoint latency. Buffers above 256 MiB
now use an owned temporary file mapping beside the checkpoints, allowing the OS
to page them without another equally large private RAM allocation. A native test
forces that path and verifies identical serialized bytes, RNG, tokens and
24-step restored continuation, plus temporary-file cleanup. Existing instances retain
their saved policy unless explicitly migrated.

The revised `data/gemma31b-pressure-03/report.json` passes all three behavioral
cases and restart recall. Each case removes the original facts from active
context, then checks the model's actual memory and outgoing recall. All three
memory revisions are preserved. The trial records 16 retirement events, of
which 15 occurred while the shift-auditing wrapper was installed. A fresh
process restored 2,430 tokens with zero replay and matched the next 24 sampled
tokens and logits exactly. This remains one bounded behavior experiment, not
a general guarantee of memory judgment.

## Intended 60K context

`scripts/verify_context_scale.py` fills a fresh context with 55,000 synthetic
tokens, saves it, compares a 24-step continuation in another process, retires
old positions and repeats the comparison in a third process. No sampled output
is parsed as actions. This checks occupied context at scale rather than merely
allocating a large, mostly empty cache.

The configured 60,000 tokens become **60,160** in the native allocation.
Gemma has 50 sliding-window layers and 10 global-attention layers. The compact
sliding-window allocation (`swa_full=false`) and full allocation
(`swa_full=true`) have materially different resource and retirement behavior.

### Compact sliding-window allocation, all model layers on GPU

`data/gemma31b-60k-rolling-01/report.json`:

- 55,000-token prefill: 87.0 seconds, after a 60.4-second load.
- Checkpoint: 2,952,971,153 bytes; packing and save: 4.81 seconds.
- Native restore: 1.81 seconds after loading; zero prompt replay.
- All 24 continuation tokens matched; maximum logit difference **0.0**.
- The live 24-token continuation took 25.36 seconds at that occupied size.
- Whole-GPU usage before checkpoint: 30,830 MiB, 1,361 MiB free; sampled
  device power at that boundary was 529.96 W. This is not a run average.
- Process peak working set: approximately 20.7 GiB. Mapped model pages and
  GPU-driver allocations make this different from an isolated RAM budget.

**This earlier configuration does not enable compact retirement.** The pinned native
`llama_kv_cache_iswa::get_can_shift()` requires the global and sliding-window
cache allocations to have the same size. The verifier therefore reports an
overall failure after its successful unshifted comparison. A running DMN would
checkpoint and pause at context pressure, without silent reconstruction.

The later [experimental compact policy](compact-cache-research.md) adds a bounded
override that preserves the complete recent window. Separate synthetic 31B
conversion/retirement/restart tests passed, and a 25K throughput run measured
45.19 tokens/sec. Those results do not retroactively change this earlier report;
existing-instance conversion still needs a supported migration command.

### Full allocation with partial model offload

`examples/gemma31b-60000-hybrid-q8.json` requests `swa_full=true`, 24 model
layers on GPU and the rest in RAM, with Q8 K/V and Flash Attention. The full
allocation would exceed this GPU's capacity with all model layers offloaded.
The test uses the captured source's effective sampler settings, including
temperature 0.7 and the default llama-server filter order.

The first full-allocation run (`data/gemma31b-60k-hybrid-01`) successfully
prefilled 55,000 tokens in 676.34 seconds and reported native retirement support,
but failed with `MemoryError` when packing attempted a roughly 25 GiB private
temporary allocation. This is a failed scale test. Full SWA keeps much more
serializable state than the compact sliding-window layout; its checkpoint cost
cannot be inferred from the compact run's 2.95 GB file.

The retry uses the file-backed packing buffer in `data/gemma31b-60k-hybrid-02`.
All three fresh processes completed successfully; `verified=true`.

| Measurement | Result |
|---|---|
| 55,000-token prefill | 660.66 seconds, after a 61.86-second load |
| Full checkpoint | 26,332,559,827 bytes (24.52 GiB) |
| Packing plus backend save | 165.36 seconds, excluding integrity hashing and runtime durability checks |
| Native restore after model load | 22.94 seconds; zero prompt reevaluation |
| Unshifted comparison | 24/24 samples equal; maximum logit difference 0.0 |
| 24-token live continuation at 55K occupancy | 71.42 seconds (about 0.34 tokens/second) |
| Retirement, notice decode and packing | 144.31 seconds; only the 11 new notice tokens evaluated |
| Retired checkpoint | 27,523 retained tokens, 13,162,587,704 bytes |
| Retired-state packing plus backend save | 48.89 seconds, excluding hashing/durability work |
| 24-token live continuation after retirement | 27.02 seconds (about 0.89 tokens/second) |
| Retired-state fresh-process restore | 15.20 seconds after model load; 27,523 tokens restored with zero replay |
| Retired-state comparison | 24/24 samples equal; maximum logit difference 0.0 |

Whole-GPU readings around inference/checkpoint boundaries were approximately
28,632–28,694 MiB, with about 3,500 MiB free. The observed pre-run baseline was
8,116 MiB and included other applications. Peak process working set reached
50,096,836,608 bytes (46.66 GiB) during retirement/packing; private committed
memory and mapped pages are distinct measurements. The temporary mapping avoided
the additional private allocation that failed in the first run, but it still
uses OS-managed memory and storage I/O.

The third process verified the retired snapshot, so this configuration passes
the occupied-context scale test. The unshifted snapshot SHA-256 is
`91244eeffc342b0928fe2150855f9685a774a228f42950ed1656fb1046b64397`;
the retired snapshot is
`8f5f71bec378167f4c39e2c3f2e6675b4ca642b05a145cd3df8b9974695057be`.

This correctness-first snapshot implementation
has a substantial storage and latency cost: it saves full native state, including
at completed actions. Changing checkpoint cadence does not remove action-boundary
checkpoints. No Linux portability or destination-host performance result is
implied; that host needs its own continuation and capacity checks.
