# Idle activity and ordinary-sleep checkpoint policy

Status: historical design proposal, followed by implementation in the isolated
codex/idle-state-design worktree. See [current behavior and validation](idle-state.md).
The investigation began at commit b48caa3 on 2026-09-21. The text below records
the original plan and findings. Reconcile the implementation with the completed
deep-sleep and multi-user work before integrating or deploying it.

## Recommendation

Start with a model-selected idle activity profile: bounded bursts on the same
live context, with no checkpoint at each scheduling pause. Keep the existing
full-save barriers on actual sleep and on explicit activity-profile changes in
this first increment. This provides a useful middle state without changing
crash recovery or relying on repeated sleep actions.

Then add an opt-in cooldown for ordinary-sleep snapshots, paired with a small
durable activity record and explicit recovery reconciliation. A timer alone is
not sufficient. Use time since the last completed full save, not requested
sleep length, as the basis for coalescing.

These are separate increments because idle pacing does not require weaker
sleep durability. Ambient observations and latent-space experiments come later.
None of these changes requires training or a change of model weights.

## What the current implementation actually guarantees

- Runtime._generate_one advances one continuing sequence; only active mode
  generates. There is one owner of backend/KV operations.
- Config.token_delay_seconds delays scheduler steps globally. It is neither a
  model-selectable activity profile nor a hard resource cap.
- Both checkpoint policies save every entry into sleep, including EOG.
- Messages, memory mutations and other committed effects are published with a
  native checkpoint. Those guarantees are independent of sleep pacing.
- A saved sleeping instance remains sleeping on restart unless its timer has
  expired or an event wakes it.
- Runtime.tick returns early while sleeping. It does not currently service
  periodic checkpoint deadlines in that branch.
- A durable incoming event can be delivered again after rollback because its
  delivery cursor is part of the full checkpoint.
- Unchanged, already-saved sleep creates no periodic snapshots. The cost is
  repeated entry into sleep, not merely the duration spent asleep.
- Clock notices are input evaluations and consume context even without sampled
  output. They must be included when estimating idle costs.

A read-only live status observation earlier in this investigation reported a
recent roughly 3 GB snapshot taking about 19 seconds. This is an individual
observation, not a benchmark or a guarantee of current performance.

## Comparing sleep policies

| Policy | Advantage | Limitation | Decision |
|---|---|---|---|
| Save every sleep | Sleep choice and native state restore together | Frequent sleep actions cause frequent large writes | Keep as default |
| Save only sleeps longer than N seconds | Simple rule | Requested duration may be interrupted; EOG is indefinite; short-sleep loops still need bounded recovery | Do not use as the main rule |
| Skip a sleep save if the previous full save was recent | Coalesces nearby saves | Can restore active mode; dirty sleep may never flush; old inputs can wake a restored sleep | Incomplete without recovery work |
| Durable sleep record plus deferred full snapshot | Preserves rest choice while allowing coalescing | Latest native processing can still roll back; needs a deadline and event reconciliation | Preferred optional sleep policy |
| Full action journal for messages and memories too | Can reduce additional snapshot barriers | Much larger change to externally visible effect recovery | Outside this increment |

A cooldown is not a global disk-write cap. Message publication, memory changes,
context retirement, shutdown and deep-sleep handoff can still require earlier
full saves.

A bounded deferral also does not eliminate every sleep snapshot. If a lone
sleep continues beyond the deadline, one delayed snapshot is still required.
Savings arise when several short sleeps are coalesced, or another required save
captures the pending state first. Indefinitely keeping only an earlier snapshot
is a different, explicitly weaker durability policy, not an implicit timer
optimization.

## Disposable findings

Run from the isolated repository root:

    python -B -m scripts.probe_idle_checkpoint_policy

This uses only DemoBackend, temporary instance directories and fake clocks.
There is no model loading, inference server, live-instance access, native-state
continuity claim or abrupt process termination. Closing without a final save
exercises restoration of the earlier authoritative snapshot.

The probe checks five cases:

1. Current strict sleep: mode restores as sleeping.
2. Suppress just the sleep checkpoint: live mode is sleeping but restoration
   returns active, with the unsaved sleep tokens absent.
3. Suppress the sleep checkpoint and advance the fake clock ten minutes:
   the five-minute periodic deadline is due, but sleeping ticks never save.
4. Reapply only a sleeping-mode overlay after rollback: input consumed before
   the sleep is delivered again from the older cursor and wakes the instance.
5. Current already-saved sleep: a day of fake elapsed time creates neither
   generation nor another snapshot.

All five probe assertions passed. The existing tests.test_checkpointing and
tests.test_runtime suites also passed: 46 tests. These establish scheduler and
recovery behavior in scripted fixtures, not native inference behavior or energy
savings.

## Increment 1: a model-selected idle profile

### State and interface

Keep lifecycle state separate from activity rate. Preserve the existing
mode values (active, sleeping, suspended, deep_sleep, held, etc.) and add an
activity profile within active operation:

- focus: current permitted generation rate;
- idle: same model and sequence, with scheduled generation opportunities.

Conceptual action:

    activity(mode="idle", burst_tokens=32, interval_seconds=120)
    activity(mode="focus")

These names and values are proposed, not currently executable. Validate requests
against configured host limits. Return the effective settings and any refusal;
do not silently claim a requested rate was accepted. Expose the active profile,
next opportunity, generation budget and actual counters in status.

Use an opt-in capability version. Old instances without activity state restore
as focus, with current behavior. Describe the capability through the existing
append-only transition; negotiate any new behavioral wording through prompt
governance. No model is automatically switched into idle.

Changing the selected profile is a real state transition and initially receives
one full checkpoint. Scheduling pauses inside the chosen profile do not.

### Initial pacing policy

Start with bursts rather than one extremely slow token stream. An illustrative
trial is at most 32 generated tokens every 120 seconds. This is a host-clock
schedule, not a sentence boundary or a task, and is not a prescribed optimum.

- Give one bounded opportunity on entry; do not accumulate unused credit.
- After a late tick or slow save, schedule the next opportunity from current
  monotonic time. Never run missed bursts to catch up.
- Check controls and incoming input at each existing token boundary.
- A pacing pause preserves parser state, unfinished text and native state.
  A partial action is not executed simply because the burst ends.
- Existing external-input cancellation of partial frames still applies.
- A completed sleep action or EOG still enters sleeping and cancels idle
  generation opportunities. Idle never restarts a sleeping instance by itself.
- For the first increment, actual human/maintenance inputs follow current
  foreground wake behavior. A further activity action selects idle again.
  A timed sleep can resume the saved pre-sleep profile.
- Operator suspension restores the previous lifecycle and activity state.
  Held, ended, storage-blocked and deep-sleep states take precedence.
- Ordinary clock notices must not cause inference between idle opportunities.
  Avoid a notice at every burst. Use an explicitly reduced clock cadence or
  deliver elapsed-time context with real events; expose the policy factually.
- Retirement preparation remains a special bounded operation under existing
  rules. It must be counted separately from idle generated-token allowance.
  The first increment does not claim a hard GPU-time or watt limit.

A 32-token burst every two minutes is 23,040 generated tokens/day if every
opportunity is used; every five minutes is 9,216. Input evaluation, action
results and retirement preparation are additional costs. Every generated token
still uses the model; resident memory is retained.

### Implementation points

- Add a small pacing policy module with an injected monotonic clock; it decides
  whether generation is due and the next deadline, but never mutates KV.
- Extend Config with opt-in host limits and register scheduling-only settings
  in recovery compatibility handling.
- Persist the selected profile and effective parameters, not raw monotonic
  timestamps. Restart rebases scheduling without catch-up credit.
- Add the action to protocol/capability help and handle it at a committed
  runtime boundary. Use checkpoint state_updates for the durable transition.
- Put pacing checks inside tick, before clock injection or generation. Merely
  sleeping in run would leave other tick callers unpaced.
- Teach run to wait interruptibly until the earliest relevant deadline:
  generation, checkpoint, timed sleep or a control/input event.
- Keep periodic dirty-state checkpoint service independent of whether a
  generation opportunity is available.
- Add status/frontend display without exposing internal text.
- Test with demo fixtures first. Native pause/continuation comparisons wait
  until hardware is available and use disposable state.

A useful scheduler order is: stop/hold/deep-sleep guards; controls; eligible
input/wake handling; required durability work; generation eligibility; one
generated token; post-token required effects/checkpoints. Retain existing
transaction boundaries and recheck lifecycle after any operation that changes
it. This is an integration outline, not a replacement implementation.

## Increment 2: coalesced ordinary-sleep snapshots

### Proposed configuration

Retain checkpoint_policy="all_actions" as literal strict behavior.
Introduce the optional cooldown only under checkpoint_policy="effects":

    sleep_checkpoint_min_interval_seconds = 300

Default 0 preserves immediate saves. Reject a nonzero value combined with
all_actions rather than silently changing that policy's meaning. Five minutes
is a trial value to discuss, not an applied setting or measured optimum.

For an unsaved ordinary-sleep transition:

1. Durably record its control state.
2. If the ordinary-sleep cooldown has elapsed, take the full snapshot now.
3. Otherwise retain live KV and queue one snapshot deadline for the end of the
   cooldown.
4. Existing token/time exposure thresholds and mandatory barriers can save
   sooner. They always override the cooldown.
5. A successful full checkpoint that includes the transition clears the pending
   request. Repeated sleep requests do not push the existing deadline later.
6. Service an outstanding deadline while sleeping, without sampling or
   injecting a wake event. A save alone must not change sleep mode.
7. Once saved and unchanged, stay quiet: no recurring sleep-save timer.

The cooldown is measured from completion of the last successful full snapshot.
This avoids repeated immediate saves after a slow save. It is a scheduling
policy checked at boundaries, not a hard deadline during native I/O or storage
failure.

### Small durable activity record

Use a dedicated SQLite record, not mutable edits to immutable runtime.json
snapshots. Record at least:

- instance identity and monotonically increasing control revision;
- authoritative checkpoint identity to which this unsaved transition relates;
- chosen lifecycle state, activity profile, reason (sleep action or EOG),
  decision timestamp and absolute timed-sleep deadline if any;
- highest input event ID already delivered when the choice was made;
- explicit wake/replacement transitions as required;
- reconciliation status or the checkpoint/control revision that incorporates it.

This is deliberately narrow control-state durability. It does not publish
messages or memory changes ahead of their native snapshots.

The full-checkpoint publication transaction must atomically declare which
control revision it incorporates. A crash before/after publication must never
apply a stale sleep record over a newer checkpoint. A partially written or
unreferenced snapshot is never authority.

If the small durable write fails, do not claim durable sleep or proceed with
ordinary generation. Preserve the last valid state and expose failure; a
validated full-save fallback may be used if available.

### Recovery and old inputs

Apply lifecycle guards before normal resume handling, automatic context
retirement or any sampling. Current startup appends notices; if that triggers
retirement preparation, it could otherwise generate before honoring the guard.
Pending factual recovery notices may need to wait until legitimate resumption.

Restore the last valid native snapshot and independently honor a newer,
compatible sleep record. Explicitly report that the sleep choice survived but
the corresponding unsaved native processing may not have survived.

Do not advance the native event cursor to the record's delivered-event watermark:
those inputs may be absent from restored KV and still need ordered delivery.
But do not classify those old inputs as new wake events either.

While honoring the sleep record, only a genuinely newer eligible event, the
selected timer, or an authorized control may release it. Once released, deliver
the older pending inputs in order before new generation. Clearing or replacing
the record and repeated-crash behavior need their own tests.

Coordinate event eligibility with the multi-user work. An event ID watermark
addresses rollback ordering; it does not determine whether a particular class
of input is permitted to wake the instance.

Use monotonic time for in-process scheduling. Persist wall-clock timestamps for
cross-process timed sleep and disclose clock changes; never serialize a raw
monotonic deadline as a reusable cross-process clock.

### Mandatory full-save barriers stay independent

Always preserve required full snapshots for messages, memory effects, prompt
adoption, approved holds, maintenance suspension/shutdown, retirement, and
deep-sleep handoff. An idle/sleep cooldown cannot authorize unloading, training,
adapter changes, or suppress the saved parent state required by deep sleep.

Normal sleep never trains. An active deep-sleep supervisor owns its transition;
the idle scheduler and ordinary-sleep recovery records cannot wake or bypass it.
This ordering must be reviewed against the finished production training work.

## Acceptance checks for production implementation

- Defaults reproduce current save counts and restore behavior.
- Idle pauses generate no tokens, execute no partial action and add no snapshot.
- Input/control delivery interrupts idle waits; sleep/EOG remains asleep.
- Long delays, slow saves and restart create no accumulated generation credit.
- Checkpoint deadlines run during idle waits and during dirty deferred sleep.
- Failed snapshots retain the prior authoritative pointer and pending exposure.
- Sleep record crashes before/after record commit and full-save publication
  restore the intended lifecycle without claiming unsaved KV continuity.
- Old delivered input cannot wake a preserved sleep; genuinely new input can.
- Timer expiry, indefinite sleep, clock adjustments and repeated restart work.
- Message and memory effects retain atomic publication and no replay duplicates.
- Deep sleep, holds, permanent end, operator suspension and storage blocking
  retain priority over pacing and deferred-save controls.
- Counters distinguish generated tokens, evaluated input tokens, required
  boundary work, full snapshot writes and small durable records.
- No native deployment is claimed until disposable native tests pass on the
  relevant backend/build. No measurements run against the valuable live instance.

## Parallel work and delivery

This proposal and probe are in a separate worktree and branch. The original
checkout remains on main; the active tasks may continue editing it.

The initial investigation was read-only apart from this document and the probe.
Subsequent implementation and tiny CPU-native validation are recorded in
idle-state.md. No valuable instance was changed. Integrate after reviewing
overlap in runtime.py, config.py, storage.py, recovery.py, protocol.py and the
frontend. The current probe additionally verifies deferred sleep recovery and
the now-serviced checkpoint deadline during sleep.
