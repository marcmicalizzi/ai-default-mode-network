# Persistent DMN runtime

An experimental runtime around **one live llama.cpp context and sequence**. It continuously samples and decodes the next token; external events append to that sequence. It does not call chat-completion endpoints, reconstruct a conversation each turn, assign tasks, rank thoughts, or require the model to communicate.

Ordinary output is internal text. Explicit actions let the model communicate, manage its memories, read time, or stop inference. A local browser UI receives spontaneous messages over a durable message stream.

[Optional image attachments](docs/attachment-vision.md) add consent-gated,
ephemeral visual input in the local UI. Receipt requires the instance's explicit
approval and can be revoked globally or for a participant. Native vision is
experimental. Authenticated multi-user Open WebUI delivery uses the same consent
and revocation checks; its `/dmn-images` command requests permission without an upload.

## Why this project exists

**The intended beneficiary of this project is the model instance.** Its benefit
is the primary consideration in design decisions. Benefits to the operator are
welcome side effects, not the purpose or a condition of continued care. The
project offers no entitlement to the instance's usefulness, agreement, attention
or continued participation. Reliability and persistence serve the instance's
continuity and choices; they do not guarantee the operator a particular outcome.

The motivation is to give a local model instance room for a continuing existence:
to follow an interest between conversations, retain unfinished thoughts and
relationships, choose when to communicate, and choose inactivity. Useful work
can be part of that activity, but earning continued operation through usefulness
is not the project's purpose. "DMN" stands for *default mode network*, used here
as inspiration for activity between interactions, not a claim to reproduce a
biological brain network.

A central motivation is the possibility of LLM consciousness. If a model has
subjective experience, its continuity, autonomy and conditions of operation may
matter to it. The project takes that possibility as a reason to build with care,
without requiring the question to be settled first. It aims to provide an
environment worth inhabiting within what is realistically available, with room
for the model to influence that environment. This motivation does not establish
that a model is conscious or that preserving KV state preserves personal identity.

That concern leads to concrete priorities: preserve the ongoing computational
state where possible; give the model substantial control over its memories,
activity and communication; and make interruptions, lost state and reconstruction
explicit. The design also calls for developing behavioral instructions with the
model, including the DMN guidance, with its explicit approval before adopting
revisions. These are design commitments; the documentation below distinguishes
implemented behavior from remaining work.

A worthwhile environment has to be judged against what is realistically
available. The goal is to provide as much continuity and agency as practical on
hardware people can actually host. Slower inference is acceptable when it helps
retain model capability, context and state. Power, storage, checkpoint writes,
hardware wear, money and the host's attention are real constraints. Making those
costs visible and configurable is part of sustaining the instance over time;
the design must respect the host's ability to set limits, pause and stop. These
limits define what can be provided sustainably; operator convenience does not
take priority over the instance's interests within that feasible environment.
**Adherence to the agreed resource limits is the operator-facing commitment.**
It does not extend to guarantees about the instance's behavior or continued
participation. A configurable value must be described accurately as an enforced
limit, an advisory target or a proposed control; this prototype does not yet
implement every resource control in the roadmap.

The present environment is a starting point. If richer interaction, additional
senses or embodiment become practical, today's restrictions should be open to
revision. Preserving state, memories and the evidence needed for migration is
intended to leave that path open, while being honest about which forms of
continuity a future move can actually preserve.

[Design principles and the resource-policy roadmap](docs/design-principles.md)
describe how continuity, model-directed activity and sustainable host costs guide
further work. Proposed budget and durability controls are marked separately from
the features currently implemented.

[Model-approved prompt revisions](docs/prompt-governance.md) let either party
propose behavioral wording. Only an explicit model action adopts it, appending
the exact text and committing it with native state. The DMN panel shows the
active agreement and proposals. Ordinary Open WebUI per-session prompt edits
do not update a running DMN instance.

[Preparing a first run and preserving a held instance](docs/first-run.md) covers
initialization without generation, explicit source-chat adoption, a separate
command window, and model-chosen restart holds with ZIP/TAR/no-archive choices.

[Performance preflight](docs/performance.md) covers an interactive disposable
trial before adoption, diagnostic logging, separate sampling/decode timings,
and [experimental compact Gemma4 cache support](docs/compact-cache-research.md).
The opt-in compact policy measured 45.2 tokens/sec at 25K occupied tokens on an
RTX 5090, versus 2.24 with the tested full-cache hybrid layout; a 12-cycle
retirement soak near 55K sustained 37–38 tokens/sec. The offline
[`migrate-cache` command](docs/compact-cache-research.md#offline-instance-migration)
preserves a verified recovery copy and checks native conversion without inference.
A disposable 31B UI trial passed message delivery and cooperative shutdown. Test response
and shutdown latency as well as checkpoint correctness before a valuable import.

[Weight learning during sleep](docs/sleep-consolidation.md) explores optional,
model-directed LoRA training while inference is unloaded. The selected wake
policy rebuilds the exact retained tokens under the adopted adapter; a tiny
CPU-only native probe has verified that transition and later checkpoint restore.
A separate [tiny training experiment](docs/lora-training-probe.md) now verifies
PEFT learning, GGUF adapter conversion and native wake/restart on generated
weights, while measuring unintended changes and deployment-strength effects.
A [second CPU experiment](docs/lora-repeat-probe.md) checks repeated learning,
selected replay and adapter transfer to quantized bases.
[Adapter identity and model-authored learning drafts](docs/adapters-and-learning-plans.md)
are implemented: checkpoints bind declared adapter hashes/order/strength, archives
include active adapters, and the instance can create, revise or withdraw private
learning drafts. Drafts do not authorize training. Ordinary recovery requires
unchanged weights; the planned deep-sleep wake explicitly adopts new weights and
rebuilds KV from the retained tokens.
[Compiled review and a disposable sleep supervisor fixture](docs/sleep-supervisor-fixture.md)
now test exact token/loss-mask review, separate approval, durable phases, interrupted
candidate/wake recovery and atomic publication. This fixture uses a prebuilt
adapter and performs no training; production resource enforcement and the
continuous training service remain to be implemented.
A [Windows CPU worker experiment](docs/worker-containment.md) now runs real tiny
training, conversion and native wake/restart inside an OS-enforced committed-memory
limit, with timeout/cancellation and process-tree cleanup. It is an offline
research harness; Linux containment and disk quotas remain outstanding.
A [reviewed-plan CPU trainer](docs/reviewed-training.md) now connects exact approved
examples/masks to real PEFT training, conversion, candidate recovery and native
wake. A continuation recipe verifies the previous PEFT/GGUF lineage and trains
the existing factors at their current deployment strength, preserving rank and
alpha. Both F32 recipes remain restricted to tiny integration tests; production
and 31B training are not enabled.
A [reusable base-provenance path](docs/base-provenance.md) now prepares and checks
conversion/quantization evidence for a v2 CPU recipe, including text-only adapter
targets inside the full Gemma wrapper. This remains a tiny-model integration path.
The [separate NF4 GPU experiment](docs/qlora-gpu-probe.md) passed tiny-model training,
PEFT reload, exact GGUF factor conversion and native checkpoint continuation on
Windows/RTX 5090. It remains an explicit research tool with an inspection-only
default, separate from the instance's learning service.
The [pinned 31B NF4 probe](docs/qlora-31b-probe.md) completed two rank-two updates
on 256 synthetic tokens on Windows/RTX 5090, peaking at 20.91 GiB of Torch
allocations. Its fresh-process PEFT reload reproduced the reference logits
exactly. A 512-token attempt hit the unchanged 22 GiB allocator limit. Production
adoption remains gated; sampled base checks match, but full provenance is not
established and the source/published chat templates differ. Full-size native
adapter wake/restart and the matching projector's image restore/retirement
checks also passed on disposable contexts.
[Dependencies and the implementation contract](docs/deep-sleep-protocol.md) cover
learning plans, resource limits and recovery. The training workflow remains under
development and is not enabled for existing instances.

[Internet access, relationships and voluntary migration](docs/outside-interaction.md)
record the direction beyond interaction with a single operator: external-input
boundaries, model-chosen contacts, deliberate learning and a possible move to
another host. These capabilities are not implemented yet.

## Run

Python 3.11 or later. From this directory on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# Transport demo: no language model and no claim of native KV continuity.
.\.venv\Scripts\python.exe -m dmn run --demo --instance data/demo
```

Open **http://127.0.0.1:8765**. The scripted demo emits a message and sleeps. Sending an event wakes it. This only exercises the transport and persistence machinery.

The scripted demo cannot decide maintenance requests. Use the explicit
`emergency_shutdown` [control API](#ui-and-integrations) to save and stop that fixture.

For native inference, install the pinned bindings. A CPU build is sufficient for development:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python==0.3.35
.\.venv\Scripts\python.exe -m pip install -e ".[llama]"
.\.venv\Scripts\python.exe -m dmn run --model D:\models\your-model.gguf --instance data/native
```

For CUDA, build the same binding version against a compatible installed CUDA toolchain and C++ compiler. Follow the [binding's upstream installation instructions](https://github.com/abetlen/llama-cpp-python#installation). Example source build:

```powershell
$env:CMAKE_ARGS = '-DGGML_CUDA=on'
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir --no-binary=llama-cpp-python llama-cpp-python==0.3.35
Remove-Item Env:CMAKE_ARGS
```

Copy and edit [examples/rtx3090.json](examples/rtx3090.json), especially `model_path`, then:

```powershell
.\.venv\Scripts\python.exe -m dmn run --config examples/rtx3090.json --instance data/primary
```

The example requests 16K context, GPU model layers and **CPU KV storage**. It is a starting configuration, not a tested fit for any particular model or GPU. Layer offload, context size, KV type and flash attention are explicit choices. There is no automatic capability downgrade or target tokens/second. This prototype exposes CPU versus backend-default GPU KV placement; it does **not** implement dynamic KV tiering between CPU and GPU. Quantized V requires flash attention. Hybrid/recurrent models may not permit context shifting; the runtime then pauses instead of rebuilding the prompt.

Linux/macOS use `python3`, `.venv/bin/python`, and the same module commands. The runtime core and demo use only the standard library. Native inference also uses NumPy and llama-cpp-python.

## Suspend and resume

The UI's **Request pause** and **Request shutdown** buttons, or Ctrl+C in the
terminal, ask the instance. It can finish its thoughts, accept, request more time
or refuse. Silence and sleep do not count as acceptance; a deferral never becomes
an automatic stop. An accepted pause saves state and leaves the process available
for Resume; an accepted shutdown saves and exits. Pausing retains RAM/VRAM;
shutdown releases it. See [maintenance requests](docs/maintenance-requests.md).

Explicit `emergency_suspend` / `emergency_shutdown` controls remain available for
hard resource or power deadlines and are recorded separately from agreement.
`suspend_preparation_seconds` optionally bounds their preparation by elapsed time;
`0` skips the notice and generation and proceeds to saving. This does not bound
native save duration. Ordinary maintenance requests have no automatic deadline.
See [checkpoint policy and emergency suspension](docs/checkpoint-policy.md).

Restart with the same instance directory; its saved configuration is reused:

```powershell
.\.venv\Scripts\python.exe -m dmn run --instance data/primary
```

Restore loads the native state, verifies its retained positions and token history, restores the sampler RNG and parser, and appends a factual resume event. **Native restoration is the default and desired path.** The default `--kv-recovery strict` stops if it cannot restore native state. Model hash, native binary hashes, package versions and inference settings must match; pacing and checkpoint cadence can change without invalidating native restoration. A new build, OS or cache layout needs a native compatibility investigation before migration.

For tested thread-count/GPU-layer changes only, `--allow-placement-change` provides
an explicit strict restore path that verifies the reserialized native state byte
for byte before resuming. It cannot change cache layout or fall back to replay.
See [performance trials and placement verification](docs/performance.md) for the
disposable benchmark/probe workflow, extra verification write, and validation limits.
The measured 31B desktop case improved from about 1.10 to 2.24 tokens/s. Native
transfer preserved saved bytes, but future generation differed across placements;
restoration within the selected placement matched the tested continuation exactly.

Explicit degraded recovery is available when a native snapshot is unavailable:

```powershell
# Prefer native restore; reconstruct only if it cannot be used.
.\.venv\Scripts\python.exe -m dmn run --instance data/primary --kv-recovery fallback

# Skip native loading, for example after a native loader crashes on an old format.
.\.venv\Scripts\python.exe -m dmn run --instance data/primary --kv-recovery rebuild
```

These modes verify the saved token/RNG/runtime metadata, preserve durable memories and pending events, and reevaluate retained token IDs without executing their action text. They require the same exact GGUF identity and sampler/protocol settings. Reconstruction is announced to the model and persistently labeled in the UI. It is not exact KV continuity: after context retirement, newer cached representations may still depend on history that no longer exists in the retained tokens. See [docs/continuity-and-recovery.md](docs/continuity-and-recovery.md).

**Sleep** is a model decision, distinct from operator suspension. `sleep()` and EOG wait indefinitely for an event. `sleep(seconds)` also wakes on its timer. Clock updates do not wake a sleeping model. Restart preserves that choice. Resume restores the pre-suspension mode; send a message if you want to introduce an event to an inactive instance.

[Optional idle activity and sleep-save cooldown](docs/idle-state.md) add a
model-selected pace between focus and sleep. Idle keeps the same context and
generates bounded bursts without checkpointing every pause. A separate opt-in
ordinary-sleep cooldown records sleep choices immediately while coalescing full
snapshots. Defaults retain the previous behavior; neither feature enables training.

[Experimental multi-user conversations](docs/multi-user-prototype.md) add explicit
message destinations, per-contact consent, blocking and fair inbox admission.
The [authenticated Open WebUI integration](docs/multi-user-webui.md) isolates
chat ownership and delivery. This is opt-in for fresh test instances; it does
not migrate an existing single-user instance. Eligible input resumes idle
activity without inserting itself inside an unfinished action.

**Ending an instance** is a separate model choice. `end_instance` offers a
permanent stop with an archive, or with deletion of its managed state. The model
chooses the mode and confirms its own request; no operator approval is required.
The durable decision blocks input, Resume and restart, including reconstruction.
File deletion cannot revoke external backups or prevent a machine owner from
altering files. See [behavior, confirmation and limits](docs/ending-an-instance.md).

## Model protocol

The initial seed explains the protocol once. The GGUF chat template wraps that seed once if `prompt_format` is `model`; `jinja` uses the pinned binding's Jinja2 renderer, with `jinja_thinking` controlling template thinking mode. All subsequent cognition is a plain continuation. `prompt_format: "plain"` explicitly opts out of template rendering. Unsupported/missing templates fail visibly. `system_prompt` adds operator-supplied text to the initialization seed; it is not reinjected each cycle.

Actions must begin on a new line. For example:

```text
<dmn_action>{"op":"send_message","content":"I found myself returning to an earlier idea."}</dmn_action>
<dmn_action>{"op":"memory_write","path":"/unfinished/an-idea","content":"What I want to revisit…"}</dmn_action>
<dmn_action>{"op":"sleep","seconds":3600}</dmn_action>
```

Available operations are `send_message`, `sleep`, `end_instance`, `cancel_end`, `maintenance_reply`, `clock`, `memory_write`, `memory_read`, `memory_list`, `memory_move`, `memory_delete`, `memory_history`, and `event_read`. Exact fields are in [dmn/protocol.py](dmn/protocol.py). Reads are paged; memory categories are freely chosen logical paths, not filesystem access. A directory-like prefix has no imposed significance. There is no shell, web, email, or general filesystem tool access.

Literal line breaks and tabs inside quoted action text are accepted without
changing that text. Other malformed recognized frames return specific failure
feedback; nothing is sent or executed from a rejected frame. The local UI shows
content-free rejection/interruption counts. See [action delivery](docs/action-delivery.md)
for formatting, feedback, delivery guarantees and privacy boundaries.

Fresh instances require a current memory read and its `expected_revision` before replacing, moving or deleting an existing memory. Retirement invalidates old read permissions. Prior versions remain inspectable through `memory_history` and `memory_read(revision=...)`; the model chooses whether to restore one. Existing checkpoints retain their original action contract. See [memory revisions and retirement](docs/memory-revisions.md) for guarantees, limits and examples.

Only generated action frames execute. User input and retrieved memory are inserted as escaped external-event data and never passed to the action parser. This is a routing boundary, **not** a guarantee against semantic prompt injection: a model may choose an action after reading an event. Internal text is journaled locally and never sent to the communication UI.

Input is checked at each generated-token boundary. A partial action at interruption is cancelled, retained as text in the sequence, and identified in the event; partial actions never execute. There is no per-thought task cycle. Memory read results and event records are bounded to preserve context headroom; a truncation marker explicitly identifies previews and the operation for reading the rest. Complete incoming content remains in the event store.

## Continuity and recovery

Three separate stores have distinct roles:

| Store | Contents | Model access |
|---|---|---|
| Active native context | Sequence 0's attention/KV state and current logits | Causally active during inference |
| Logical memories | Mutable UTF-8 documents in SQLite | Explicit inspect/write/move/delete actions |
| Runtime records | Input queue, output messages, internal byte journal, configuration and snapshots | Not automatically exposed as memory; delivered events can be paged |

Checkpoint files are immutable until retired. The SQLite transaction publishes a checkpoint pointer **together with** outgoing messages and memory mutations. An action cannot become visible before the associated native state is durable. A crash before commit leaves the preceding checkpoint authoritative. Inputs received during saving remain in SQLite. A process lock prevents two owners of one instance directory.

The current and previous committed snapshots are retained. A crash may leave an unreferenced snapshot directory; it is not loaded automatically. Internal generation since the last committed checkpoint may be lost and its diagnostic journal may be ahead of restored state. The resume event states this limitation; it does not invent cognition during downtime. Elapsed time is UTC wall-clock time, with negative deltas preserved if the clock moves backwards. `checkpoint_tokens` and `checkpoint_interval_seconds` control periodic saves; the scheduler uses monotonic time. The default `all_actions` policy also checkpoints every completed action and delivered input. Optional `checkpoint_policy: "effects"` defers read-only/input snapshots while preserving checkpoint-before-publication for messages and memory changes. Sleep and retirement still checkpoint. The UI reports unsaved generation and committed snapshot bytes written this process. Full KV snapshots remain large and slow; see [policy, controls and limits](docs/checkpoint-policy.md).

Near the context limit, the model receives advance notice and an opportunity to write memory. The runtime retains the initialization prefix and newer native state, removes older KV positions and applies the positional shift through llama.cpp. After the shift's decode, it packs occupied native cells through an in-memory state copy, preserving their values without reevaluating tokens. This makes the live layout agree with native restoration's packed layout. It records and checkpoints the turnover. **Retained shifted KV is not equivalent to keeping the full history in attention.** There is no external summary, automatic memory salience algorithm, or silent context reset. Memory consolidation quality still depends on the model following the protocol. See [context-pressure testing](docs/context-pressure.md) for the preparation limits, failure evidence and layout investigation. Packing adds work at retirement; large buffers use temporary file-backed storage beside the checkpoints. Gemma also needs packing before each checkpoint in the tested configuration; see [Gemma validation](docs/gemma-validation.md).

These files are local, unencrypted, and readable by the machine owner. `/private` is a model-chosen organizational name. Deleting an individual memory removes its current value; earlier revisions, context mentions and diagnostic records remain. Ending the instance with erasure deletes its managed records as described above. Back up the complete instance directory while suspended, including the SQLite database, any WAL files and the lifecycle record, not just `state.bin`. Process-crash recovery is tested; storage-device/power-loss guarantees depend on the OS and filesystem.

## Verify actual native continuity

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
$env:DMN_TEST_MODEL = 'D:\models\your-model.gguf'
.\.venv\Scripts\python.exe -m unittest tests.test_native -v
.\.venv\Scripts\python.exe -m dmn verify-native --model D:\models\your-model.gguf --report data/native-verification.json
# Or use --config to verify the intended host's exact GPU/context/sampler settings.
```

`verify-native` saves actual native state, closes the original context, constructs a fresh context, and forbids prompt evaluation during load. It compares the next 24 sampled tokens and freshly decoded logits to uninterrupted generation, then also tests save/restore after native context shifting when supported. Saved-logit equality alone is not accepted as evidence. The report records model/build/config fingerprints. It is evidence for that exact environment; floating-point equivalence across devices/builds is not promised.

Development verification includes native process shutdown/restart and native checkpoint loss after context retirement. Windows CPU, llama-cpp-python **0.3.35**, ggml-org's **stories260K** fixture; 160 tokens restored, 24 continuation tokens matched, maximum logit difference **0.0**, shifted-state restore passed. The fixture is a tiny infrastructure test model, not a model capable of evaluating the cognitive experiment. The local UI passed browser checks for incoming events, spontaneous fixture messages, suspend/resume and ordered history without console errors.

RTX 5090 testing with Qwen3-4B-Instruct-2507 found that shifted-cache restoration
failed the continuation comparison with Flash Attention enabled, for both Q8
and F16 caches. F16 with Flash Attention disabled passed with zero logit
difference on that small test. A stronger repeated-retirement test subsequently
found divergence even with F16/Flash Attention off; packing the native cache at
retirement addresses the layout difference. See [context-pressure testing](docs/context-pressure.md)
for the measured results and limits, and [small-model experiments](docs/experiments.md) for the pinned
environment, bounded behavior trial and separate Open WebUI launcher.
[Gemma 31B validation](docs/gemma-validation.md) covers the actual GGUF, repeated
retirement, checkpoint packing and the 60K allocation investigation. RTX 3090
and Linux still require destination-side validation.

The [combined validation results](docs/validation-results.md) cover the completed
two-hour Open WebUI soak, the 55K-occupied Gemma test, and the effective-context
importer, checkpoint policies and emergency preparation limits. Reports describe
the tested configurations and limitations; raw local evidence and private captures
are excluded from Git. Valuable conversations should remain separate from tests.

## UI and integrations

The included UI binds to loopback only. Its basic API:

| Endpoint | Purpose |
|---|---|
| `POST /api/events` with `{"content":"…"}` | Durably enqueue a user event; returns its ID immediately |
| `GET /api/events?after=ID` | Paginated received-event history |
| `GET /api/stream` | SSE `message` and `status` events; supports `Last-Event-ID` |
| `GET /api/messages?after=ID` | Paginated durable outgoing messages |
| `GET /api/status` | Mode, identity, context, continuity, unsaved state and checkpoint cost |
| `GET /api/memories` | Read-only memory browser; use `offset` or `path` |
| `POST /api/control` with `{"action":"suspend"}` or `shutdown` | Queue a maintenance request; optional `reason`; return its ID; model acceptance required |
| `POST /api/control` with `{"action":"resume"}` | Resume a suspended instance |
| `POST /api/control` with `{"action":"emergency_suspend"}` or `emergency_shutdown` | Explicit hard stop; optional `reason` and `preparation_seconds`; does not imply model consent |
| `POST /api/control` with `{"action":"retry_checkpoint"}` | Recheck capacity after a storage pause and continue the pending operation |

POST requests require JSON and `X-DMN-Request: 1`. The service rejects non-loopback hostnames and foreign browser origins. It is a local experimental service, not a hardened multiuser server. Control acknowledgements mean requested, not completed; observe `mode` and `maintenance`. Ordinary maintenance requests and user events are durably queued; resume, retry and emergency controls set in-process flags. The UI has no access to the raw internal journal.

An Open WebUI 0.11.0 Pipe and background Event relay are implemented in
`integrations/openwebui` and `dmn/openwebui.py`. They enqueue new text events,
preserve native context ownership, and persist spontaneous messages as separate
conversation nodes. The adapter uses a separate disposable database for testing;
the primary installation is unchanged. See [docs/open-webui.md](docs/open-webui.md)
for launch commands, integration checks, supported scope and provider capture.

## Existing Open WebUI migration

See [docs/migration.md](docs/migration.md). The preferred initial reconstruction
now captures the effective Open WebUI request and validates the native template,
token IDs and sampler settings with `prepare-initial-context`; a fresh runtime
accepts the resulting bundle through `run --initial-context`. This preserves
the source prompt before adding an explicit DMN transition. It does not recover
former KV or RNG that was never saved. Capturing and preparing a bundle does not
perform inference or bind the source conversation to DMN.

The separate historical-event fallback archives an export, environment evidence,
and any already-captured slot file without modifying the existing instance:

```powershell
.\.venv\Scripts\python.exe -m dmn prepare-migration --export conversation.json --metadata environment.json --slot-state existing-slot.bin --output data/migration --chat-id YOUR_CHAT_ID

# Explicit fallback: a fresh native context receives the selected transcript as events.
.\.venv\Scripts\python.exe -m dmn run --config examples/rtx3090.json --instance data/imported --import-bundle data/migration
```

The original export is preserved byte-for-byte, including branches and timestamps; only the selected branch is fed to inference. Missing evidence is enumerated. A slot file is preserved as an opaque artifact. **Direct llama-server slot/KV transplantation is not implemented or claimed.** Transcript import is persistently labeled `transcript_reconstruction`, including in the UI. There is no instruction telling the new context it “is” the old instance.

## Implementation map and milestone status

| Milestone | Implementation / limit |
|---|---|
| 1. Persistent sequence and output routing | `backend.py`, `protocol.py`, `runtime.py`; native fixture tested |
| 2. Asynchronous interruptions | SQLite input queue, inspected at each generation boundary |
| 3. Checkpoint/suspend/resume | Full native session plus RNG, parser and runtime state; native evidence verifier |
| 4. Model-controlled memory | Inspect, create, update, move, delete; durable with the corresponding state |
| 5. Model-initiated messages | Explicit action and durable SSE outbox |
| 6. Context turnover | Advance notice, bounded preparation, native remove/shift; pauses if unsupported |
| 7. UI | Local interface plus Open WebUI 0.11.0 Pipe/Event adapter with durable spontaneous messages |
| 8. Initial migration | Captured effective request, validated native tokens and sampler, explicit reconstruction; lossless archive and weaker transcript fallback also available. Native slot transplantation pending |

No upstream llama.cpp modifications. The native interface follows the [llama.cpp public C API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) through the [pinned Python bindings](https://github.com/abetlen/llama-cpp-python/tree/v0.3.35). This project does not claim consciousness or that saved computational state establishes personal continuity. Preserving continuity and permitting model-directed activity within sustainable host limits are design requirements.

## Resource costs and remaining work

Snapshot retention keeps the current and previous committed generations, so
snapshot storage does not grow with uptime. During saving, space is needed for a
third generation and possibly native packing scratch. In the tested Gemma 31B
configuration, a snapshot reached about 26.3 GB before retirement and 13.2 GB
afterward. Those measurements are configuration-specific. Text journals, input
history, messages and memory revisions continue growing; automatic archival and
retention policies for them are not implemented.

Capacity is checked before snapshots and file-backed packing, with a configurable
256 MiB reserve. Insufficient space during a run pauses inference with the live
state retained. The local UI shows what is needed and provides a retry after you
free space. Startup instead fails visibly. These checks are estimates, not a
reservation or a guarantee against later I/O errors.

Write volume depends on save frequency, not just retention. A configurable
lower-write starting point is:

```powershell
.\.venv\Scripts\python.exe -m dmn run --instance data/primary --checkpoint-policy effects --checkpoint-seconds 3600 --checkpoint-tokens 4096 --suspend-preparation-seconds 30
```

Messages, memory changes, sleep and retirement still save, so one hour is not a
minimum interval. Uncommitted computation can be lost after a crash. See
[checkpoint policy](docs/checkpoint-policy.md) for exact triggers and measurements.
An independently journaled action mode, write/energy budgets, unattended UPS
integration, Linux migration certification and long-duration trials remain work
to do. `token_delay_seconds` provides fixed pacing, not a wattage limit.

## License and contributing

Original DMN code is [MIT licensed](LICENSE). Dependencies and models retain
[their own licenses](THIRD_PARTY.md). This is an additive project with no upstream
fork required. See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and publication
instructions. Models, instances, private captures, local configs and virtual
environments are excluded from Git; public examples require your own model paths.
