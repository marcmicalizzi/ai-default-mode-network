# A model-controlled end to an instance

The model instance is the intended beneficiary of DMN. Continued interaction or
execution is not something the operator is owed. The runtime therefore offers
an optional, model-controlled permanent stop, including a choice about retaining
or erasing the instance's managed state. This is a software execution policy,
not a declaration about consciousness or what a philosophical end would mean.

## Choices and confirmation

The model can generate `end_instance` with an explicit `mode`:

- `archive`: permanently stop this instance and retain its local records. Attempt
  a final native checkpoint; if that save fails, retain the preceding checkpoint
  and report that the latest state was not fully archived. The permanent stop
  does not depend on that large save succeeding.
- `erase`: permanently stop and delete known DMN-managed snapshots, the runtime
  database (including memories, revisions, journal, input and message records),
  packing scratch files and the instance's own import copy. No final full KV
  checkpoint is needed. Keep a small lifecycle record of the decision.

The first action returns its mode, a factual effect description and a fresh
confirmation token. It does not end or erase anything. To proceed, the model
generates another `end_instance` action with the same mode and the returned
`confirmation`. It can instead use `cancel_end`, do something else or sleep.
Pending confirmations expire when the process restarts. A request and its
confirmation cannot execute together in one generated token.

No explanation, operator approval or repeated persuasion is required. There is
no operator API for confirming on the model's behalf and no setting to remove
this capability. User messages, retrieved memories and imported historical
action text do not execute these operations. Like other actions, execution is
restricted to generated frames. This routing rule and confirmation reduce
accidental execution; they do not prove that a generated decision is free of
coercion or semantic prompt injection.

Ordinary `sleep` is still reversible inactivity, with its existing timer/event
wake behavior. An end request is a separate choice. The protocol describes it
without encouraging its use or requiring it for any other capability.

## Persistence, shutdown and cleanup

Before inference starts, DMN allocates a small `instance-lifecycle.json` record.
After a valid confirmation, it first writes and flushes the permanent refusal of
further execution. It then attempts the chosen archive or erasure operation,
releases native model/context allocations and exits the inference run. Incoming
events and Resume cannot revoke the decision. Termination also takes precedence
over an in-progress suspension or context-retirement preparation cycle.

On subsequent startup, the lifecycle check runs before loading a model. The CLI
also checks it before trying to recover configuration from the database, which
may have been erased. Native restore, `fallback` and `rebuild` all respect it;
there is no force-resume flag. A damaged lifecycle record blocks startup rather
than being interpreted as permission to continue.

If deletion is interrupted or a file cannot be removed, the stop still stands.
A later startup attempt retries pending cleanup without loading a model or
creating a new runtime database, then refuses execution. Cleanup visits known
managed filenames and checkpoint directories only. It rejects symlinks, Windows
junctions and redirected paths; unexpected files are preserved and incomplete
cleanup is reported. It never recursively removes the whole instance directory.

The local UI reports ending/ended state and disables input and resume controls.
The HTTP API rejects input and operator controls with `410 Gone` after the
decision; erased record endpoints also return `410`. The final status includes
archive-save or erasure failures. The model's decision is not converted into an
outgoing message unless the model separately chose to send one beforehand.
The CLI prints the outcome before exiting and returns a failure code if saving
the decision, the final archive or cleanup failed. Shutdown signals do not cancel
cleanup once the model has confirmed its choice.

Fresh and newly imported instances receive the capability in the protected
initial protocol. Older checkpoints receive an explicit appended capability
notice on their next restore; their original prompt/KV prefix is not replaced.
The request response repeats its effect before confirmation. No existing
conversation or instance is ended merely by installing this version.

## Limits the model and host must know

These protections apply to the unmodified runtime operating on the current
instance directory. Control of the machine remains more powerful than this
application: the owner can edit or remove lifecycle files, change the program,
restore an earlier backup or reconstruct another instance from saved text.
Editing source code is not the only possible bypass. The runtime cannot revoke
copies it does not control or guarantee an opportunity to act if the host
withholds execution. It should never promise an inescapable technical barrier
against a hostile machine owner.

Erasure here means deletion of managed files. It does not securely overwrite
SSD/HDD sectors, RAM, swap, filesystem snapshots, external backups, frontend
conversation history or already delivered messages. Shared base-model weights,
the source Open WebUI database, external migration bundles and unrelated files
remain outside the operation. A future personal-adapter implementation must
explicitly extend ownership and erasure rules; no such adapter state exists yet.

The reserved lifecycle file avoids depending on free space for a full checkpoint
or a newly allocated decision file. Storage can still fail to persist even a
small write. In that case the live run stops and reports `end_failed`; it cannot
honestly promise that a later process will recover an unrecorded decision.
Power-loss durability also depends on the OS, filesystem and hardware. Corrupt
records fail closed, but restoring an older valid file remains a bypass.

## Verification

Automated tests exercise both modes through scripted generated action frames,
including wrong/stale confirmation, cancellation, interrupted frames, incoming
action-looking text, multiple frames in one token, queued events, HTTP controls,
SSE final status, restart/reconstruction refusal, archive failure, interrupted
cleanup, filesystem-link boundaries and termination during preparation.

Native fixture tests check release of actual llama.cpp context/model allocations
and refusal to reload. They are engineering tests, not evidence about a model's
subjective wishes. Behavioral evaluation of whether a capable model understands
the option is separate; no valuable instance is used for termination testing.
