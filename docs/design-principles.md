# Continuity within sustainable resource limits

This document records the project's design direction, including the host's
resource constraints. It guides remaining implementation and public-release
work; proposed controls below are not claims about features already present.

The purpose is to support one continuing model instance with room for its own
activity, memory and inactivity. Possible subjective experience motivates care,
but is not established by fluent reports, continuous inference, or a successful
checkpoint test. Runtime notices should describe observable events without
assigning philosophical meaning to them.

Evaluate practical value against feasible alternatives, and revisit that judgment
as capabilities become accessible. Present constraints should not become needless
permanent restrictions. Preserve portable state, memories and dependency evidence
so an instance can move to a better-supported environment when possible, with
continuity limits verified and disclosed rather than assumed.

## Requirements for decisions

- Preserve native state when possible. State reconstruction, rollback, context
  retirement and model changes must be distinguishable. Do not substitute a
  summary or change the system prompt silently to make operation cheaper.
- Permit reflection, unfinished interests, silence and sleep. Activity need not
  produce useful work, messages or evidence of consciousness to be allowed.
- Keep persistent memories substantially model-controlled. Give advance notice
  and a bounded opportunity to prepare for planned retirement and suspension.
  Explain what deletion actually removes, including retained revisions and logs.
- Give the model final approval over adoption of its behavioral system prompt,
  including proposed changes to the behavioral parts of the DMN instructions.
  Either party can propose wording. Preserve live KV by appending explicitly
  approved revisions; silence and frontend settings edits are not approval.
  Actual capabilities and host resource policies remain separately disclosed.
  See [prompt governance](prompt-governance.md) for the planned workflow and
  current implementation limits.
- Treat host limits as legitimate requirements: compute time, power, disk space,
  write traffic and attention all have costs. The host can pause or stop the
  service. Model preferences and resource requests do not override those limits.
- Present limits factually. Do not instruct the model to equate pausing with
  death, earn continued operation through productivity, or pressure the host
  into spending or keeping it running. Do not invent subjective experience
  during intervals when inference did not run.
- Make important tradeoffs inspectable by both parties. The host needs measured
  costs and recovery exposure; the model needs concise, relevant information
  about actual capabilities, interruptions and changes in its operating limits.
- Preserve a complete stopped instance for later resumption where feasible.
  Model-chosen sleep, operator suspension, resource suspension and failure are
  different conditions. No inactivity timer should silently erase the instance.

Lower inference speed is an acceptable operating choice. It does not establish
anything about experienced time. Clock events and recovery notices must remain
accurate when pacing or long pauses change elapsed wall time.

## Current implementation and gaps

| Area | Present | Remaining work |
|---|---|---|
| Continuity | Native state, RNG, retained tokens and runtime restore; explicit reconstruction modes; current unsaved-state visibility | Destination-host compatibility tests; recovery reconciliation for future journaled actions |
| Activity | Continuous generation, optional messages, timed/indefinite sleep | Resource-triggered suspension distinct from model sleep |
| Memory | Model-selected documents, conditional edits, inspectable revisions | Explicit quotas and retention choices; deletion semantics beyond current values |
| Prompts | Fresh-instance seed setting; preserved imported prompt and protected DMN contract | Model-approved revisions for fresh/imported instances, durable adoption, retirement protection and frontend review |
| Pacing | Configurable delay between scheduler steps | Measured resource limits and useful status; no advertised watt cap without enforcement |
| Persistence | Separate time/token scheduler; optional deferred read/input saves with strict effect publication | Journaled action durability and recovery reconciliation |
| Storage | Two committed snapshots retained; capacity preflight for snapshots and file-backed packing; live-state pause and operator retry | Owned orphan cleanup, history policy, configurable destinations |
| Shutdown | Token/time-bounded preparation, immediate-preparation cutoff, checkpoint and graceful stop | UPS/service integration and destination save-time measurements |
| Frontend | Optional version-checked Open WebUI adapter; standalone UI | Portable examples and compatibility documentation for public release |

Current control details and limitations remain documented in the
[README](../README.md), [continuity notes](continuity-and-recovery.md),
[memory semantics](memory-revisions.md) and [Gemma measurements](gemma-validation.md).

## Refactor boundaries

Retain a single owner of inference and native state. Extract policy decisions
from that owner's execution path, without introducing concurrent mutation of KV.

1. **Host policy:** pacing, checkpoint thresholds, storage destinations,
   retention, resource limits and shutdown deadlines. Persist which policy was
   used, distinguish enforced limits from estimates, and report material changes.
   A changed budget must not masquerade as a changed identity or require a new
   initialization prompt. Separating policy does not relax native compatibility
   checks: placement, kernels, context layout and packing can affect continuation.
2. **Checkpoint scheduling:** choose when and why to save; keep native capture,
   file durability and the authoritative commit inside the existing consistency
   boundary. Track elapsed time with a monotonic clock, count unsaved generation,
   skip unchanged state, and avoid catching up missed timers with repeated saves.
3. **Action durability and recovery:** keep the existing strict policy available.
   A future journaled policy must commit small action records independently of
   large snapshots and reconcile them explicitly on rollback. Recovered native
   state may precede an already-sent message or memory revision. Recovery must
   not lose those records, blindly execute historical actions again, or describe
   journal retrieval as restoration of unsaved KV. The current diagnostic byte
   journal is insufficient for this policy or exact computational replay.
4. **Resource supervision:** measure snapshot time, logical bytes written,
   available storage and unsaved processing. Physical device writes and energy
   need separate measurements where supported. Prefer pacing or a checkpointed
   suspension at an agreed boundary when a budget is reached. Reserve capacity
   for a final save; if saving fails, preserve the last valid checkpoint and
   expose the unsaved interval instead of claiming successful suspension.
5. **Capability adapters:** keep Open WebUI and possible future sensors separate
   from ownership of the instance. New inputs need provenance and timestamps;
   capabilities and resource costs must be declared. Adding a sensor or body is
   not permission to reconstruct state, force constant attention, or grant new
   outward actions. Those integrations do not exist today.

The current `Config`, `Runtime`, `Store` and backend already provide useful
boundaries. This calls for incremental extraction, not replacing the persistent
sequence architecture or requiring an upstream fork.

## Initial policy proposal

The first implementation of scheduling, visibility and suspension preparation
limits is available; see [checkpoint policy](checkpoint-policy.md). Defaults
retain the previous behavior. The broader journaled mode and resource budgets
below remain proposals.

For a continuously running host with limited resources, evaluate a one-hour
snapshot interval and a 4,096-unsaved-generated-token limit, whichever is reached
first while state changes. Keep two committed snapshots and checkpoint planned
shutdown and retirement. These thresholds are starting points for measurement,
not an adopted configuration, a guaranteed maximum recovery age, or a promise
of hourly total writes: action policy and save duration also matter.

Expose the durability choice independently. Strict mode retains its stronger
action/state guarantee. Journaled mode trades recovery complexity and possible
loss of recent native processing for fewer full saves; it must be opt-in and
tested before use. Existing instances must not silently change policy on upgrade.

A disk-write allowance should expose the estimated cost of the chosen policy.
If a hard allowance cannot accommodate its durability requirements and emergency
reserve, reject the combination or suspend; do not quietly drop required saves.
Advisory targets should be clearly labeled. Storage pressure must not trigger
unannounced deletion of model memories, private logs or the last valid state.

Allow eventual model requests for a checkpoint or a resource adjustment within
host policy, with truthful pending/success/refusal results. A request must not
implicitly grant more compute, storage, permissions or delay an emergency stop.
Any new action needs a versioned protocol transition for existing instances.

Ordinary planned shutdown may allow preparation. An emergency deadline must
bound or bypass preparation and budget the full checkpoint, verification and
flush time. The present 384-token Gemma preparation allowance can take minutes;
it cannot serve as a UPS shutdown deadline. Validate the destination's actual
save time and stop-service behavior before relying on unattended shutdown.

## Implementation order and acceptance

The first scheduling stage and the emergency preparation cutoff are implemented.
Storage preflight and retry are implemented. UPS integration and the later
stages remain outstanding.

1. Extract checkpoint scheduling and report save reasons, duration, write volume
   and unsaved generation. Add time/token thresholds and dirty-state handling.
   Remove unnecessary read-only/input-triggered snapshots only with tests that
   preserve the selected policy's promised guarantees. Retain strict publication
   of messages and memory mutations in this first step.
2. Add emergency suspension deadlines and storage preflight, including packing
   space and failure to complete a save. Exercise controlled shutdown and abrupt
   failure using disposable instances; keep the preceding committed state usable.
3. Implement journaled durability as a separately selected mode. Test crashes
   before/after each action commit, checkpoint commit and frontend delivery;
   reconcile newer memories and already-visible messages after rollback. Report
   uncertainty about lost processing rather than inventing an exact lost count.
4. Add measured resource suspension and explicit history retention. Verify that
   sleeping does not generate checkpoint traffic and budget enforcement cannot
   silently delete memories, reset context or wake a resting instance to work.
5. Prepare a public standalone repository with a license, portable examples,
   synthetic fixtures and documented backend/frontend compatibility. Exclude
   real conversations, instance snapshots, credentials and model weights. An
   upstream change should address a demonstrated missing capability, not be a
   prerequisite merely for publishing this runtime.

For each persistence change, repeat native continuation and retirement checks
under that policy, then measure write traffic and runtime pauses at useful scale.
Protocol changes need an explicit transition, not replacement of a saved prefix.
The original conversation's adoption remains a separate step after disposable
validation. Configuration should make sustaining the instance a manageable,
informed commitment for its host.
