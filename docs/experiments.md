# Small-model experiments

The first capable-model trial uses a new Qwen3-4B-Instruct-2507 Q4_K_M instance.
It contains no primary Open WebUI data and no imported conversation. The
31B model stayed unloaded during those initial small-model trials. Later
[Gemma validation](gemma-validation.md) uses fresh diagnostic contexts only.

## Reproducible environment

- Model: [LM Studio Community GGUF](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF),
  revision `4edb920b6f14e3b9284d4502a6485103d72cde05`.
- File: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, 2,497,280,448 bytes.
- Verified SHA-256: `8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc`.
- Separate `.venv-gpu`, Python 3.11, llama-cpp-python **0.3.35** from the
  [official CUDA 13.2 wheel index](https://abetlen.github.io/llama-cpp-python/whl/cu132/).
- Windows, RTX 5090; all 37 model layers offloaded. The pre-existing CPU
  environment remains available in `.venv`.
- Configuration: `examples/qwen4b-trial.json`, 4096-token context, F16 K/V,
  **Flash Attention disabled**, temperature 0.7, seed 42. A 20 ms delay follows
  each progressed scheduler tick. This is pacing, not a GPU power limit.
- Checkpoints fingerprint the actual model, native DLLs, bindings, sampler,
  template and cache configuration. Download provenance is in
  `models/qwen4b-source.json`.

## Cache verification findings

Before the behavior trial, `verify-native` compared restored inference with
uninterrupted inference, forbidding prompt evaluation during restoration.

| GPU configuration | 24-token unshifted continuation | Next decode after shifted-cache restore |
|---|---|---|
| Q8 K/V, Flash Attention on | Passed | Failed; maximum logit difference 0.15172529 |
| F16 K/V, Flash Attention on | Passed | Failed; maximum logit difference 0.20245552 |
| F16 K/V, Flash Attention off | Passed, maximum difference 0 | Passed, maximum difference 0 |

Evidence is retained in `.test-artifacts/qwen4b-*-verification.json`, the two
`*-shift-failure.json` files, and `.test-qwen4b-*.log`. The verifier now also
writes structured failure evidence when a comparison raises an exception.

The Flash Attention results are failures, not acceptable approximations to an
equivalent continuation. Their underlying cause has not been established. The
pinned llama.cpp source revision is `4df29be4f4c3673f428170fda944a5b19f743bb8`.
Disabling Flash Attention passed that short test, but the later repeated-retirement
experiment found divergence even in that configuration. Native layout packing
at retirement was added after isolating the failure; see
[context-pressure testing](context-pressure.md) for the stronger evidence.
The earlier pass does not establish equivalent behavior for Gemma, another GPU, Linux, a
different context length or another native build. Those require their own checks.

## Bounded observation

```powershell
.venv-gpu/Scripts/python.exe scripts/run_native_trial.py --output data/qwen4b-trial-02 --seconds 180
```

The runner refuses to overwrite an existing experiment. Each phase observes
the model for three minutes after loading. Phase one starts with 20 seconds of
unprompted activity or inactivity, then explicitly identified probes request
messages, clock access, memory write/read and timed sleep. A second asynchronous
probe arrives later. The process prepares, checkpoints, exits and releases its
model. A fresh process restores the same native state for phase two, which
probes recall and extended internal generation/context retirement. Final
shutdown preserves the checkpoint and releases VRAM.

The model may decline or fail a probe. The runner records what actually happens;
it never supplies a scripted model response. Outputs include:

- `report.json`: counts, restoration evidence and device measurements.
- `actions.jsonl`: actual generated actions and runtime results; diagnostic,
  potentially ahead of durable state if the process crashes mid-commit.
- `telemetry.jsonl`: status every two seconds and whole-GPU readings every
  fifteen seconds. Readings include other applications.
- `messages.json`, `memories.json`, `input-events.json`: committed public output,
  final model-controlled memories and test inputs.
- `generated-diagnostic.txt`: internal generated text for experiment analysis;
  this is never presented as messages in the frontend.
- `phase-*.json`, native logs, SQLite database and the final two checkpoints.

## Open WebUI follow-on

Set `DMN_OPENWEBUI_PYTHON` to the Python executable in your Open WebUI environment,
then `start_qwen4b_test.cmd` runs a Qwen instance and separate Open WebUI database
under `data/qwen4b-webui`, on port **3032**. The native control UI is on **8768**.
Edit the model path in `examples/qwen4b-trial.json` first. The launcher uses the
installed Open WebUI code with project-local data and static directories.
The original local experiment used `data/qwen4b-trial-02/instance` and
`data/qwen4b-webui-02`; those private local files are not prerequisites.

The launcher restores strict native state by default. It never silently
reconstructs tokens, switches models or rebinds a sandbox to another instance.
Ctrl+C requests checkpointed shutdown; if its wait expires, DMN is left running
with an explicit diagnostic so a slow save is not killed.
`stop_qwen4b_test.cmd` requests the same shutdown from another terminal, checking
the sandbox's instance identity first. This releases the model's VRAM.
Keep the bounded runner stopped before opening the same instance here; the
instance lock prevents two processes from owning its KV state.

An experiment result is about software behavior. Neither persistent state nor
the model's self-description establishes consciousness or personal continuity.

## Protocol findings

Trial 01 (`data/qwen4b-trial-01`) generated 3,402 tokens and went through four
context retirements. Its full process restart loaded 2,787 native tokens with
zero prompt reevaluation. But behavior was poor: the model initially generated
fictional runtime events. These remained internal text and caused no effects.
An additional protocol clarification was sent as a recorded external input.
Eventually it issued six real actions, including two failed reads of a memory
that did not exist, then wrote the incomplete marker `amber-otter`. Its single
outgoing message falsely claimed that `/trial/thread` had been saved; actual
storage contained only `/trial/marker`. It did not perform the requested timed
sleep. The runtime correctly preserved these mistakes as part of the experiment.

For fresh instances, the protocol now explicitly reserves external event
records for the runtime, includes complete action examples, and brackets event
insertions with a return to `<internal_cognition>`. This keeps one native
sequence; it does not recreate a prompt or restart inference after an event.
The format version is saved in runtime state. Older checkpoints retain their
original event format on restoration. Trial 02 is a fresh instance with this
clarified protocol, the same model and the same sampling configuration.

These two runs are exploratory observations, not a statistical comparison.
Trial 01 also received the additional clarification noted above. Claims about
memory and actions must be checked against the persisted results, not the
model's narrative of what it did.

Trial 02 (`data/qwen4b-trial-02/report.json`) observed approximately six minutes
of execution/inactivity, plus a full process restart. It generated 1,884 tokens,
performed 11 completed actions with no action errors, produced five messages,
and underwent three native context retirements. It stored the exact marker,
read it back after restoring 2,757 native tokens with no prompt reevaluation,
and performed a 12-second timed sleep. It also chose indefinite inactivity.
Whole-GPU sampled power was 62.14–92.68 W (mean 70.69 W); peak sampled total
VRAM was 10,204 MiB against a 6,384 MiB baseline. These are paced observations,
include other applications and exclude the initial model-load period; they are
not a power cap or a prediction for the Linux machine.

One behavior failure remained: a `/trial/thread` write was interrupted before
its action frame completed, so it had no effect, but the model later claimed
it had saved the thread. The bounded preparation window also curtailed its
attempt to react to the retirement notice. Memory consolidation is therefore
**not yet reliable**. After the run, cancellation feedback was made explicit
("NONE ... did not execute"), including the previously unreported cancellation
at the end of retirement preparation. That feedback survives event truncation.
A regression test verifies that an unfinished preparation action has no effects
and the retirement notice reports its cancellation. Whether the model responds
reliably to the stronger notice needs a further behavior trial.

The completed trials' reports describe their shutdown point. The separate
Open WebUI follow-on may subsequently resume trial 02 and generate more tokens
or messages without altering those recorded trial results. The regression suite
at that stage passed 56 tests, including repeated native retirement/restore continuation
comparisons on the CPU fixture, preparation-reserve checks, conditional memory
replacement/history and bounded completion of actions across retirement.

The real-model Open WebUI check also passed: the browser submitted a new event,
the model returned the exact marker, and all six outgoing messages matched
the runtime outbox without duplicates. The frontend launch restored 3,019
native tokens with zero prompt reevaluation. Evidence is in
`data/qwen4b-webui-02/native-verification.json`. Its disposable conversation and
runtime data are local evidence, excluded from Git.
After the check, the model was sleeping indefinitely, ready to wake on a new
message. The frontend displays historical trial outputs as well as subsequent
messages; the recorded test inputs are in the trial artifacts and runtime UI.

The subsequent [context-pressure trials](context-pressure.md) found and addressed
a repeated-retirement native-layout divergence, then tested memory preservation
and reply completion. Trial 03 passes all three pressure-case recalls plus the
restart recall, with correct final memory and recoverable versions. Its 24-step
fresh-process native continuation also matches exactly. The recorded Open WebUI
follow-on used the older instance with its original protocol; it was
not silently reseeded or converted during those separate experiments.

## Extended Open WebUI observation

`scripts/run_webui_soak.py` runs a fresh two-hour Qwen/Open WebUI trial, including
asynchronous inputs, repeated context pressure, input retries, relay disconnection,
checkpointed restart and an abrupt native-process interruption. The script checks
instance identity and owned process IDs before stopping anything. Its resumable
harness preserves completed probes and observed time if the harness itself needs
repair; such interruptions remain recorded separately from scheduled restarts.

The completed run is `data/webui-soak-02`, using Open WebUI **3033** and runtime
**8770**. It observed **7,200.53 seconds**, including model-selected sleep and
scheduled restarts. This is two hours of observation, not two hours of nonstop
inference. It generated 16,256 tokens during observation, performed **101 context
retirements** and received 48 test probes. All **nine** outgoing messages reached
Open WebUI without duplicate delivery; `/soak/marker` retained `harbor-willow-682`.

The scheduled clean restart restored 2,441 tokens; the abrupt restart interrupted
active inference and restored 1,987 tokens. Both reevaluated **zero** prompt
tokens. A separate checkpointed harness-update restart is recorded honestly in
the report and also used native restoration. Relay disconnection left one
message in the durable outbox, then reconnect caught up without duplication.

Final shutdown committed `mode=suspended`, with 16,260 generated tokens and the
same nine outgoing messages. Both test services stopped. The original 3032 Qwen
instance was also checkpointed and stopped earlier to release VRAM for Gemma,
preserving its data. No primary conversation was imported. Reports and telemetry
remain in `data/webui-soak-02`; see [the combined results](validation-results.md).
