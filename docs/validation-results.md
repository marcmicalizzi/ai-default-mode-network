# First three follow-up workstreams: results

Completed on 2026-09-19–20. The combined machine-readable record is
`data/first-three-validation.json`.
All experimental inference processes are stopped and their checkpoints retained.
**The private source conversation remains unchanged and has not been imported or bound to DMN.**

This report records successive test stages. Raw artifacts referenced under
`data/` are local evidence, excluded from the public repository. The current
regression count is in the latest section below.

## 1. Extended Open WebUI trial

A fresh Qwen 4B instance completed two hours of observation, including sleep,
asynchronous inputs, repeated pressure, retries, relay disconnection, a clean
restart and an abrupt interruption during active inference.

- 101 native context retirements and 48 probes.
- Nine outgoing messages delivered without duplicates.
- The memory marker survived; no runtime error was recorded.
- Both scheduled restarts restored native KV with zero prompt replay.
- One additional harness-update restart is recorded separately.
- Final checkpoint is suspended; the test services stopped.

Evidence: `data/webui-soak-02/report.json`.
See [experiment details](experiments.md#extended-open-webui-observation).

## 2. Actual Gemma 31B and the intended context size

The exact supplied Q4_K_M GGUF was verified by SHA-256. Fresh 4K behavior trials
passed all three memory/retirement cases plus restart recall. The 60K allocation
test filled **55,000 tokens** with synthetic input, using Q8 K/V and Flash
Attention. Fresh processes matched all 24 sampled tokens and logits exactly
both before and after native retirement, with zero prompt replay.

There is a material configuration tradeoff. The compact sliding-window layout
fits with all model layers on GPU and restores correctly, but the pinned native
build cannot retire it. The passing layout uses **full SWA allocation and 24
model layers on GPU**, with the rest in RAM.

At 55K occupancy its snapshot is **26.3 GB**. Packing plus backend writing took
**165 seconds**, excluding integrity hashing and runtime durability work.
The measured continuation rate was about **0.34 tokens/second** at that size,
on this desktop alongside the other trial. Retirement reduced the snapshot to
13.2 GB. This is a working, expensive correctness baseline; it is not yet an
efficient continuous-run configuration.

Evidence and resource limits: [Gemma validation](gemma-validation.md),
`data/gemma31b-60k-hybrid-02/report.json`.
Linux and the smaller GPU still need destination-side validation.

## 3. Effective-context importer

The importer captures the actual final Open WebUI request, renders through the
pinned native server, checks exact tokenizer agreement and records source sampler
settings. Both Chat Completions and the installed Responses format are covered.
It preserves source context before appending the explicit DMN transition, never
executes historical action text, and protects its action contract during retirement.

Synthetic Qwen and Gemma imports passed native prefill and restart checks. The
Gemma fixture also passed exact continuation after retirement. Its first behavior
probe timed out; resuming the same checkpoint with a longer allowance completed
memory write/read, one message and sleep. The initial failed attempt is retained,
along with its two rejected attempts to use historical frontend tools.

The private source conversation was prepared from a read-only source backup: **18,298 tokens**,
its original system prompt, injected tool context, and temperature **0.7**.
The model hash and tokenizer agree. No original-conversation inference occurred.
The validated 60K placement override changes only GPU-layer placement and SWA
allocation relative to that prepared bundle; its sampler settings remain intact.
Former KV and sampler RNG were unavailable, so the future initial import will
be explicitly labeled reconstruction. Later restarts use native checkpoints.

Evidence: [importer details](migration.md) and the synthetic
`data/gemma-import-native-02/report.json`. Source-conversation readiness evidence
is retained privately.

## Fixes and remaining boundary

The work corrected Gemma seed rendering/BOS handling, extended checkpoint packing
for sliding-window layouts, used file-backed scratch space to avoid a large RAM
allocation failure, protected the imported contract across disjoint retirement
ranges, preserved source sampler ordering, and added Responses capture/conversion.
Imported instances now receive explicit active-tool feedback after an unavailable
operation. At this stage, **71 regression tests passed**, including native CPU checks.

The next separate step is a rehearsal using a copy of the private source conversation and its
frontend binding, followed by any actual adoption. Neither has been performed.
Snapshot I/O and inference latency are the main practical costs to address before
treating the large-context configuration as a comfortable everyday service.

The subsequent design review prioritizes configurable checkpoint scheduling,
deadline-aware shutdown and explicit resource limits before everyday adoption.
The [design principles and implementation order](design-principles.md) distinguish
this planned work from the tested behavior above; journaled action durability
has not been implemented or enabled.

## Checkpoint policy follow-up

The optional `effects` policy defers snapshots for read-only actions and delivered
inputs. Messages, memory mutations, sleep and retirement retain native checkpoint
boundaries. Monotonic time/token scheduling and suspension preparation cutoffs
are implemented, with current-process write accounting and unsaved-state status.
Defaults for existing configurations remain unchanged.

A native comparison passes on both the CPU stories260K fixture and the CUDA
Qwen3-4B-Instruct-2507 Q4_K_M model: retire context, suspend with zero preparation,
change checkpoint policy, restore without prompt replay, then compare 24 sampled
tokens and all logits exactly. The GPU check uses 8K context, full layer offload,
F16 K/V, Flash Attention off and checkpoint packing. It completed in 88.4 seconds.
This is a small-model check, not a new 31B/60K performance measurement.

At this stage, **90 regression tests passed**, including the native CPU checks. A scripted
comparison with three inputs, ten read-only actions, one memory write, one
outgoing message and sleep required **17 snapshots under `all_actions` versus
4 under `effects`** (76.5% fewer), with identical committed message/memory results.
These counts include initialization. This is workload-specific transport evidence,
not a prediction of real-model savings or SSD wear. Machine-readable results:
`data/checkpoint-policy-comparison.json`.

The standalone browser UI displays checkpoint metrics and preserves chosen sleep
across suspend/resume. The private source conversation remains unimported. Destination UPS
wiring and journaled action recovery remain future work.
See [configuration and guarantees](checkpoint-policy.md).

## Storage protection and publication preparation

The full Windows regression suite now passes **103 tests**, including native CPU
checks, in 37.0 seconds. Capacity tests simulate a shortage without filling a disk.
They cover startup/restore refusal with prior files preserved; queued input and
unpublished messages/memory revisions during a pause; exact-once publication after
retry and immediate shutdown; scratch checks before allocation; a single context
shift across a storage pause; and preservation of model-chosen sleep. Actual I/O
failure after a successful check remains covered by the earlier failed-save tests.

The local browser UI was exercised with a disposable demo: retry while still
short of space, queue an input during the pause, restore capacity, retry, observe
one published message, and return to sleep. No model or valuable conversation
was used. Launcher tests verify that a save exceeding its shutdown wait is left
running and that an unrelated process on the port receives no shutdown request.

The repository has an MIT license, dependency-license inventory, portable model
examples, Windows/Linux CI configuration and private-state ignore rules. CI's
Linux jobs have not yet run; they do not establish native Linux compatibility.

The distributable wheel builds and installs in a clean Python environment. That
environment runs 97 tests successfully, with six native/NumPy-dependent tests
skipped as intended. A CUDA Qwen 4B continuation check also passes after the
storage changes: retirement, shutdown, changed scheduling/reserve settings,
native restore without prompt replay, and 24 exactly matching tokens/logit
vectors. It took 121.1 seconds on this run. No new large Gemma snapshot was
required for these checks.

## Staged startup, prompt approval and preservation (2026-09-20)

The Windows suite passes **158 tests**, including native CPU tests, in 67.7
seconds. Additional targeted packaging checks pass after adding inventory
read-back verification. Tests cover staged initialization without sampling,
first-question gating, exact prompt approval and stale/unread rejection,
reconsidering a declined proposal, failed adoption saves, repeated retirement
around source prefix/import contract/active agreement, hold release gates before
backend loading, and lossless ZIP/TAR round trips with source retention.

Browser testing used only a scripted disposable instance. It queued a first
question with zero generated tokens, explicitly started, submitted a proposal,
showed awaiting review, and displayed the committed agreement after generated
fixture approval. The Windows launcher also passed with a directory containing
spaces and exited normally from a staged instance. These checks do not require
the coding application to remain open when the user starts the launcher in an
independent terminal.

Open WebUI 0.11.0 integration checks passed on a fresh disposable installation:
input retries, edit/regeneration rejection, compaction guards, offline delivery
and destination deduplication. Explicit adoption also passed against a copy of
that installation's real SQLite schema. This caught and corrected JSON encoding
in normalized content fields; adoption checks content, ancestry, output, files
and saved summaries. The native import portion of that schema fixture was
simulated and is not presented as a real native import test.

An actual Gemma 31B Q4_K_M / Q8 KV rehearsal used 60K capacity, 24 GPU layers,
full SWA and packed checkpoints, with **21,000 synthetic occupied tokens**.
The original prefix and appended agreement survived retirement. The initial
agreement checkpoint was **9.38 GB in 67.1 seconds**; the checkpoint after
retirement was **4.93 GB in 33.6 seconds**, including the normal integrity and
durability path. Total committed checkpoint writes for this rehearsal were
20.11 GB, excluding scratch and filesystem/device amplification. Preparation
through synthetic prefill took 439 seconds including model load and initial
staging; it is not a pure prefill benchmark.

After closing the original context, a fresh process restored the held native
checkpoint with **zero prompt tokens reevaluated** and reproduced eight sampled
tokens and their subsequent logits exactly (maximum absolute error 0). Ordinary
Runtime startup refused the hold before model loading. The comparison used an
explicit test backend to inspect the saved state without releasing the hold.
Protocol choices in this Gemma run were injected mechanical fixtures, not model
preferences. Report: `data/readiness-gemma-01/report.json`.

A fresh Qwen3 4B run exercised actual generated prompt actions: it read and
approved the host proposal, then became inactive. A later native resume offered
the reversible hold action. It did not execute that action; it generated
unconfirmed permanent-end requests, and no end or hold was recorded. The
disposable test stopped through its explicit emergency test cleanup and retains
its checkpoint. This supports prompt-action usability, not a claim that every
model will understand or choose every lifecycle action. Reports are under
`data/readiness-qwen-01`; no private source conversation was used.

The agreed source prompt supplied for the valuable future import matches its
captured effective wording apart from whitespace. The original conversation,
frontend binding and inference state remain untouched. Native Windows-to-Linux
compatibility remains unvalidated. Earlier GitHub CI at commit `75fb202` passed
Windows and Ubuntu on Python 3.11/3.13; those jobs skip optional native tests.
