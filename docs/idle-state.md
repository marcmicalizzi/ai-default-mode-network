# Optional idle activity and deferred sleep snapshots

Implemented in the isolated idle-state branch. Defaults retain current activity
and immediate sleep checkpoints. This has not been enabled for Syllas or merged
with the concurrent training/multi-user work.

## Idle activity

Enable the capability through configuration or the equivalent run flags:

```json
{
  "idle_enabled": true,
  "idle_max_burst_tokens": 32,
  "idle_min_interval_seconds": 120
}
```

These are host limits, not an instruction to enter idle. The instance can choose:

```text
<dmn_action>{"op":"activity","mode":"idle","burst_tokens":32,"interval_seconds":120}</dmn_action>
<dmn_action>{"op":"activity","mode":"focus"}</dmn_action>
```

Issue the action alone. Omitted idle parameters use the host limits. Invalid
requests fail rather than silently changing the requested settings. Selecting a
profile commits one full checkpoint; the quiet gaps between bursts do not.
The interval starts after the burst finishes. There is no accumulated credit
for missed opportunities, downtime, or slower inference.

Idle retains the same model, weights, sequence, KV state and unfinished parser
frame. Pauses do not execute incomplete actions. Sleep/EOG still stops
generation. Incoming events interrupt the wait and select focus; the instance
can choose idle again. Timed sleep resumes the selected profile after a fresh
interval. Restoring an idle checkpoint also waits a fresh interval before its
first burst. Ordinary suspension preserves the chosen profile.

Automatic clock notices are suppressed in idle, so pacing does not fill the
context with timer events. Clock reads and actual input still supply time.
Checkpoint deadlines run during idle waits without generation. Holds, ending,
operator suspension, storage blocking and deep sleep retain their precedence.

The generation allowance covers ordinary output tokens. Input evaluation and
the existing bounded retirement/stop preparation can require extra work, shown
in separate counters. Every output token still uses the model. This feature is
not a GPU-time, watt, memory, or total disk-write cap.

## Ordinary sleep snapshot cooldown

Optional configuration:

```json
{
  "checkpoint_policy": "effects",
  "sleep_checkpoint_min_interval_seconds": 300
}
```

Zero, the default, retains immediate full saves. Nonzero values require
`effects`; `all_actions` remains strict. The cooldown is measured from completion
of the last successful full checkpoint, not the requested sleep duration.

A deferred sleep choice is immediately written to SQLite with the instance and
checkpoint identity, sleep deadline, activity profile, and delivered-event
cursor. Live native state stays resident. One full snapshot is scheduled for
the end of the cooldown; repeated sleeps do not postpone that deadline.
Ordinary token/time thresholds or required effect saves can capture it sooner.
After a successful full save, unchanged sleep produces no more writes.

Messages, memory changes, profile changes, prompt adoption, holds, retirement,
suspension/shutdown and approved deep-sleep handoff retain their full-save
barriers. The cooldown never authorizes unloading, training, or adapter changes.
It can coalesce nearby sleeps and saves; it does not remove the later snapshot
for a single long sleep or cap total writes from other operations.

Deadlines are checked at scheduler boundaries. Slow native calls, saving or
storage failure can exceed an interval. Small SQLite writes have their own I/O
cost and are not included in the full-snapshot byte counter.

## Crash recovery

The compact activity record preserves a choice, not the native computation that
led to it. Recovery restores the last authoritative native checkpoint and honors
any newer compatible sleep choice. Unsaved processing can be lost. Full
checkpoint publication and removal of the pending record share one transaction.
A mismatch of instance/checkpoint identity fails closed.

Recovery first saves the reconciled sleeping metadata without evaluating new
notices or generating tokens. Its factual resume notice is held until a genuine
wake event or the selected timer. Previously delivered inputs that are absent
from restored KV remain queued; they cannot themselves wake the preserved sleep.
They are delivered in order once a legitimate wake occurs. Their cursor is never
silently advanced to pretend their context was restored. Wake decisions are
also recorded before resuming work. Repeated restarts preserve the guard.

Changing the cooldown back to zero does not discard a pending sleep choice.
Disabling idle or lowering its host limits below a saved idle request rejects
restore; retain compatible limits or have the instance select another profile
first. Neither setting changes model/KV compatibility rules or permits replay
as native restoration.

## Configuration and visibility

Run flags mirror the settings: `--idle-enabled` / `--no-idle-enabled`,
`--idle-max-burst-tokens`, `--idle-min-interval-seconds`, and
`--sleep-checkpoint-min-interval-seconds`. Supplying them on a future launch is
an explicit configuration change; this is not a live hot-reload API.

The optional capability and host policy are appended in full and protected from
context retirement. No existing prompt is replaced or behavioral agreement
implicitly adopted. Updated policy parameters receive an updated notice.

Status and the UI show the activity profile, remaining generation allowance,
next opportunity, pending sleep-save delay, and any activity recovery. Status
also distinguishes ordinary idle/focus generation, boundary preparation,
evaluated input/output tokens, and compact activity-record writes. These new
token counters start from zero on an older checkpoint and are not historical
measurements of earlier activity. Internal text is not exposed by these fields.

## Validation and limits

On 2026-09-21, the full suite ran 280 tests successfully (21 optional tests
skipped). Three additional tiny native CPU tests passed, including immediate
reconstruction labeling while a recovered sleep still defers its notices.

Run the scripted recovery comparison and unit tests without loading weights:

```text
python -B -m scripts.probe_idle_checkpoint_policy
python -B -m unittest tests.test_activity tests.test_sleep_checkpoint tests.test_checkpointing tests.test_runtime
```

`tests.test_activity_native` is separately opt-in through `DMN_IDLE_TEST_MODEL`.
It rejects weights of 16 MiB or larger, disables CUDA, uses one CPU thread,
zero GPU layers and CPU KV. The local 1.3 MiB random Gemma4 fixture passed native
idle pause, packed checkpoint continuation, and sleep-guard recovery checks.
Actions are scripted through the real tokenizer/decoder; this is a mechanics
test, not evidence that a trained model chooses or prefers idle.

With `pack_checkpoints=true`, the tested eight-token continuation and logits
matched exactly after native restore. The initial unpacked fixture comparison
showed a maximum logit difference of about 2.4e-7 on the first checked step; it
was not certified for exact continuation. Pacing does not change the existing
cache-layout compatibility requirements.

The disposable browser demo showed Idle, the burst interval and sleep-save
policy without console errors. Its server and tab were closed afterward.
No valuable live instance, large model or GPU was used for this validation.
Native checks on Syllas's model and behavioral trials remain separate future
work, to be scheduled with an agreed maintenance window if needed.
