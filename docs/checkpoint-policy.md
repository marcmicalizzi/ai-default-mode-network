# Checkpoint scheduling and bounded suspension

The runtime supports an optional policy that reduces full snapshot writes while
keeping messages and memory mutations atomic with their native state. This is
not journaled action durability: external effects never commit ahead of KV.
Existing configurations retain the original behavior unless changed explicitly.

## Settings

| JSON setting | Default | Meaning |
|---|---|---|
| `checkpoint_policy` | `"all_actions"` | Save on every completed action and delivered input. `"effects"` defers read-only, rejected-action and input-delivery snapshots. |
| `checkpoint_tokens` | `512` | Generated tokens since the last successful save. `0` disables this threshold if a time threshold is enabled. |
| `checkpoint_interval_seconds` | `0` | Monotonic time since the last completed save, checked at scheduler boundaries when state has changed. `0` disables this threshold. |
| `suspend_preparation_seconds` | `null` | Emergency/direct suspension preparation allowance. `null` keeps the token-only bound; `0` skips preparation. Ordinary maintenance requests require model acceptance. |
| `checkpoint_reserve_bytes` | `268435456` (256 MiB) | Additional free-space margin beyond the estimated snapshot or file-backed packing allocation. Nonnegative integer; CLI: `--checkpoint-reserve-bytes`. |

At least one periodic threshold must be enabled. Whichever enabled threshold is
reached first requests a snapshot. Saving itself pauses inference. The time
threshold starts after a successful save completes, preventing a slow save from
immediately scheduling another. Actual recovery age can exceed the interval by
save time, an in-progress native operation or scheduler pacing. It is not a hard
wall-clock recovery guarantee.

Both policies always checkpoint outgoing messages, memory mutations, entry into
sleep, retirement, initialization, resume and planned suspension/shutdown.
Unchanged sleeping state does not generate periodic saves or new inference.
Two committed snapshots are retained, as before.

## Lower-write operation

Use the same Python environment and instance directory that already passed
native restoration. For example, this command explicitly changes scheduling on
an existing instance, without changing its model or sampling configuration:

```text
python -m dmn run --instance data/primary --checkpoint-policy effects --checkpoint-seconds 3600 --checkpoint-tokens 4096 --suspend-preparation-seconds 30
```

The CLI flags override the corresponding JSON settings for this run. A successful
checkpoint saves that configuration for later restarts. Recovery reports changes
to effective scheduling settings while preserving native compatibility checks
for the rest of the environment. Old checkpoints use the defaults above for
fields they do not contain. Their saved initialization prefix is not rewritten.

An isolated transport test requires neither model weights nor GPU memory:

```text
python -m dmn run --demo --instance data/checkpoint-demo --port 8783 --checkpoint-policy effects --checkpoint-seconds 3600 --checkpoint-tokens 4096 --suspend-preparation-seconds 0
```

Under `effects`, incoming events are still written durably to SQLite immediately.
Their delivery cursor advances in the next committed native checkpoint. If a
crash rolls back an uncheckpointed delivery, the saved queue delivers that input
again into the restored sequence. Reads and their internal results may similarly
be lost with unsaved processing. A later memory mutation or outgoing message
checkpoints the complete preceding context and cursor before publishing its
effects. A read permission cannot survive a rollback independently of its state.

This removes unnecessary saves from read-heavy workloads, but a model that often
sends messages, edits memories or sleeps can still produce frequent snapshots.
It does not impose an hourly total-write limit. Further reduction of those
action-boundary saves requires the separately planned journaled recovery design.

## Visibility and accounting

The standalone UI shows unsaved state, committed snapshot bytes written during
the current process, the most recent save's duration, and the active policy.
`GET /api/status` adds a `checkpoint` object with:

- `dirty`, `unsaved_generated_tokens`, and `unsaved_seconds` since the first
  unsaved context change;
- `age_seconds` since capture of the last successful checkpoint;
- `policy`, `interval_seconds`, and `token_limit`;
- `in_progress`, `committed_count`, `committed_snapshot_bytes`, `failed_count`,
  and `last` containing save reason, duration, snapshot size and commit time.

Counters cover this process only. Byte counts include the committed snapshot's
files, not packing scratch traffic, failed/uncommitted files, SQLite, filesystem
overhead or physical device writes. Duration covers capture through publication,
including integrity hashing and durability calls; it excludes pruning afterward.
The status is published at completed inference/scheduler boundaries. Unsaved
generation is current exposure, not proof of precisely how many tokens a later
crash lost. There may also be unsaved input and runtime changes with zero new
generated tokens.

Save failure does not advance the durable checkpoint timestamp, reset exposure,
publish pending effects or claim a successful suspension. The last committed
snapshot remains authoritative. Failed directories are still retained for
diagnosis; automatic orphan cleanup remains future work.

## Insufficient storage

Before writing a snapshot, DMN estimates native state, token metadata, logits and
runtime metadata, then checks free space on the destination volume including the
configured reserve. File-backed native packing also checks its own allocation
before creating a scratch file. Existing committed checkpoints are not deleted
to make a save fit. Packing scratch is released before the snapshot file is
written, but previous snapshots and any orphan files still occupy space.

During a run, insufficient space publishes `mode: "storage_blocked"` and pauses
inference at that storage boundary. The live native context, pending action and
retirement operation remain in the process. The control API and local UI remain
available. Pending effects are not published and the durable timestamp does not
advance. Incoming events can still be durably queued if SQLite has enough space.
The reserve is an advisory margin, **not an allocation** other processes cannot
consume; actual I/O errors can still happen after the check.

Free space on the reported volume, then use **Retry storage check** in the local
UI or send `{"action":"retry_checkpoint"}` to `/api/control` with the usual JSON
and `X-DMN-Request: 1` headers. The retry rechecks capacity and continues the same
pending operation. It does not reconstruct KV, repeat a shift or reexecute a
published action. A factual pause event is delivered at the next active
scheduler boundary; a sleeping instance remains asleep. `GET /api/status`
includes `storage.blocked` with free, estimated, reserve and required bytes,
the path/purpose, and `storage.last_pause` after recovery.

Shutdown requested during this pause still needs capacity and an operator retry
to finish durably. A zero preparation allowance does not bypass storage checks.
The Open WebUI sandbox launcher waits up to `--shutdown-timeout` (600 seconds by
default); expiration leaves DMN running and reports its PID and control URL.
The service manager and UPS must allow for this condition. Killing the process
loses unsaved computation, as with any other interruption before a commit.

During startup/restoration, before the control server exists, insufficient space
raises an actionable error instead of waiting invisibly. Free space and restart.
A later actual I/O failure follows the existing error path; it is not blindly
retried against potentially partial native state. Neither a low-space pause nor
a failed save replaces the previous committed checkpoint. Save duration includes
any storage wait inside that operation. No disk filling is used in tests: capacity
is simulated while normal disposable state files exercise the real commit path.

## Emergency preparation cutoff

Ordinary `suspend` and `shutdown` controls now ask the model and may be refused
or deferred. See [maintenance requests](maintenance-requests.md) for this API
change. A maintenance request does not start a countdown or silently escalate.

The control API accepts an optional `preparation_seconds` for `emergency_suspend`
and `emergency_shutdown`, overriding the configured preparation allowance.
For example, on Linux:

```sh
curl -fsS -H 'Content-Type: application/json' -H 'X-DMN-Request: 1' \
  -d '{"action":"emergency_shutdown","preparation_seconds":0,"reason":"UPS deadline"}' \
  http://127.0.0.1:8765/api/control
```

A zero allowance skips the preparation notice and further generation, saves at
the next usable boundary and exits. Saved suspension metadata explains that
preparation was skipped; resume includes that fact. A positive allowance begins
when the request arrives, uses monotonic time and is checked between native calls.
Retried requests can shorten an existing deadline but cannot extend it.
Ctrl+C and SIGTERM queue ordinary requests. Service/UPS integration needing a
hard stop must explicitly use the emergency control and account for save time.

Suspension does not retire context merely to insert its notice. If the notice
or preparation would exceed reserved headroom, preparation stops and the current
valid state is checkpointed. An urgent request during retirement stops further
retirement-preparation generation at its next boundary; a native shift or save
already underway still completes.

This is a **preparation deadline**, not a forced process-kill timer or a promise
that persistence finishes within that number of seconds. Native decode, cache
packing, snapshot writing, hashing and flushes are not safely preempted. Allow
the measured worst-case in-flight work and final save when configuring UPS and
service shutdown. An HTTP 202 response only acknowledges the request; wait for
actual process exit before treating shutdown as complete. Destination-host UPS
wiring and save-time measurements have not been configured by this change.

## Validation

Regression cases cover deferred read/input saves, rollback redelivery, memory
revision guards, publication failure, monotonic scheduling, slow saves,
inactivity, policy changes, and emergency preparation cutoffs. Native validation
compares 24 sampled tokens and all logits after retirement, zero-preparation
suspension, and a change of checkpoint policy, with zero prompt replay.
See [validation results](validation-results.md) for tested environments.
