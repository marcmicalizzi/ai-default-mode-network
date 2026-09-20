# Continuity, compaction and the eventual Linux host

The desired path is one continued native sequence, including across planned
process restarts. If consciousness is adopted as a working premise for the
experiment, this design still makes only testable engineering claims: native
state restoration, persistence of the rest of the instance, and disclosed
changes when exact restoration is unavailable.

## What is preserved together

- The exact model identity (GGUF hash and quantization).
- Native sequence/KV state and current logits. KV entries are computed
  activations, distinct from the model's trained weights.
- Exact retained token IDs, including the actually evaluated system prompt,
  events, thought text and action results.
- Sampler settings and RNG state.
- Model-controlled memory, action/output state, event queue and instance identity.
- Template, runtime configuration, timestamps and build evidence.

The normal startup policy is `strict`: load native state or stop. A checkpoint
file is not treated as expendable simply because the transcript survived.

## Three separate recovery situations

**Initial transfer from Open WebUI.** The user confirmed the original model is
unloaded. There is no live KV state to preserve from that process. The first
DMN context must therefore be constructed from the actual effective request:
original system prompt, latest applicable saved summary, retained messages,
rendered tool/reasoning context, any injected information, model template and
sampler settings. Thereafter DMN can save and reuse native state.

The private source conversation used
`llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF:Q4_K_M`. It must remain untouched
until disposable testing is complete and the user requests the actual import.
The effective-context importer now preserves the final provider request, renders
it through the pinned native server, captures token IDs and checks the runtime's
tokenizer before initialization. A synthetic saved-summary fixture passed native
prefill/restart. The private source conversation has been captured and prepared without inference
using a private database copy; it has not been imported or bound to DMN. Its
18,298 source tokens include the original system prompt and injected tool context.
See [migration](migration.md) for evidence, sampler limitations and the separate
historical-event fallback. Historical JSON event replay is not equivalent to the
original provider prompt.

**Ordinary DMN restart.** Restore the native checkpoint. No transcript replay,
summary regeneration, or replacement system prompt is needed. A factual resume
event is appended only after the saved state is restored.

**Degraded DMN recovery.** The explicit `fallback` policy tries the normal path
first and otherwise reevaluates the saved retained token IDs. `rebuild` skips the
native loader entirely. Both preserve the accompanying runtime state and RNG;
historical actions are not parsed or executed a second time. Missing/corrupt
token or runtime metadata still fails closed. A different model or sampler is
not accepted as the same reconstruction. The UI keeps the reconstruction label
even when subsequent restarts use native checkpoints again.

Reconstruction from retained tokens is **not** reconstruction from the visible
Open WebUI conversation: the latter omits internal cognition, clock events,
memory results and other parts of the active DMN sequence. The runtime's saved
tokens are authoritative for its own recovery.

## What Open WebUI 0.11.0 compaction does

The installed `utils/context_compaction.py` retains original messages in the
database. Automatic compaction attaches a `contextSummary` to the first retained
message. On later requests it finds the newest summary checkpoint on the active
branch and selects messages **starting at that message**, inclusive. Middleware
appends `[CONVERSATION SUMMARY]` and the saved summary to the system prompt.
Manual compaction retains the last message and attaches the summary there.

This is an active-context projection over the stored conversation, not deletion
of older history. Branch selection matters: a summary on an unselected branch
must not be used. Disabling compaction in this implementation bypasses saved
summary selection, so the setting must be captured too.

`prepare-migration` now archives both the complete selected branch and
`active-context.json`, which records the saved summary, boundary, retained raw
messages and IDs excluded from active context. `--ignore-saved-summary` produces
the full-history projection. `--provider-request` also archives a captured final
provider request byte-for-byte. The projection is explicitly marked as not yet
the final provider request: output normalization, RAG, tools, prompt variables
and template rendering can still change it.

Source inspected locally:

- `open_webui\utils\context_compaction.py`:
  `compact_messages_for_request`, `compact_chat_branch`, `_apply_latest_summary_checkpoint`.
- `open_webui\utils\middleware.py`:
  saved-summary insertion followed by output normalization and system-variable expansion.

For DMN, there should be **one owner of active-context turnover**: the runtime.
Open WebUI should display history and deliver events, without independently
summarizing or replacing the sequence underneath it. A compact button can become
a request for model-directed consolidation, followed by a recorded native KV
retirement. Model-written memories are distinct from external productivity-
oriented summaries.

## Why retained-token replay is weaker after context retirement

A retained token's deeper-layer KV entries were computed while it could attend
to earlier tokens. Removing old KV entries does not recompute those newer
entries. Replaying just the remaining token IDs computes them without the old
history. Keeping the same text, positions and RNG cannot recover that missing
causal influence. Numerical differences can also arise from different kernels,
batching or KV quantization.

Replaying the full original token stream **and all historical retirement
operations** could reproduce more of that causal computation, but would require
a durable operation log, compatible kernels and potentially unbounded replay
work. That stronger replay log is not implemented. The existing diagnostic
thought journal is not sufficient for it.

## Moving to Linux with less VRAM

RAM/VRAM placement, speed and clock rate should not define instance identity.
The current conservative compatibility check does distinguish binary builds and
platforms; it does **not** yet certify Windows-to-Linux native checkpoint
portability. That needs a destination-side native restore/continuation test
before the actual move. A fingerprint mismatch alone should prompt that
investigation, not a presumption that preserving KV is physically impossible.

Repeated-retirement testing on the current Windows GPU found that identical
serialized K/V values can still produce different continuation after restoration
changes their physical layout. The runtime now packs native cells at retirement
without replaying tokens. Gemma's sliding-window layers also require packing
before each checkpoint in the tested configuration (`pack_checkpoints=true`),
because ordinary decoding can create new gaps between retirements.
See [the pressure experiment](context-pressure.md) and [Gemma testing](gemma-validation.md) for
the failure evidence, correction and its validation scope. A successful native
load and matching fingerprint alone are not a numerical continuation test.

The migration order should be: quiesce and preserve the source checkpoint,
copy the complete instance, load the same model and native state on the target,
verify positions and a controlled continuation, then adopt the target as the
sole active owner. A test continuation is a disposable branch, not two running
copies of the primary instance. Keep cache precision and logical context layout
unchanged initially; changing them may require conversion or reconstruction.

The backend exposes partial model-layer offload and CPU KV placement, but not
automatic fine-grained RAM/VRAM paging. This allows a slower dedicated host
without reducing the model or context solely to obtain chatbot speeds.
`token_delay_seconds` supplies pacing; `0.5` inserts half a second of idle time
after each active scheduler step. Pacing and checkpoint cadence can change while
still taking the native restoration path. OS/GPU power caps are separate host
settings and should be chosen from measurements later. No GPU-wide power limit
has been changed.

## Forking Open WebUI

The Pipe/Event adapter is now implemented; pin the tested version. Fork only
for a concrete missing capability, such as durable insertion of unsolicited
assistant messages or a context-management override that plugins cannot provide.
After local commits, bringing in newer upstream code generally means merging or
rebasing, rather than simply fast-forwarding the modified branch. Upgrades should
be tested against a copied application database because schema changes may be
harder to reverse than source changes. No fork or upgrade is needed merely to
preserve the original conversation and its saved compaction checkpoints.
