# Memory replacement and retirement boundaries

Fresh instances use the `revisions_v1` memory protocol. Existing checkpoints keep
their original protocol; a restart does not silently replace their initialization
text or impose action arguments they were never taught. The protocol marker and
read permissions are saved with native/runtime state.

## Intentional replacement

Creating a new path still uses `memory_write(path, content)`. Replacing an existing
memory requires both:

1. A successful `memory_read` of that path's **current** contents since its last
   change and since the most recent context retirement.
2. `expected_revision` equal to the revision returned by that read.

For example, after reading revision 2:

```text
<dmn_action>{"op":"memory_write","path":"/notes/rain","content":"An intentional update.","expected_revision":2}</dmn_action>
```

The runtime rejects a blind or stale replacement without changing memory.
The revision is checked again in the SQLite transaction that publishes the
checkpoint and effects. A conflict rolls back the entire transaction, including
other effects and history entries. The same checks apply to move and delete.
Historical reads do not grant permission to replace the current value. Multiple
frames in one token cannot use a read result that the model has not yet received.

This guard does not decide what a memory should say. The model can intentionally
revise or remove memories after reading them. A model can still make a semantic
mistake after a valid read; revision checks cannot prove truth. The initialization
and retirement notices explain that existing memories persist without rewriting,
and that uncertainty is a reason to inspect or retain them rather than invent
replacement details.

## Recoverable versions

Every committed creation, replacement, move and deletion has a durable history
entry in `memory_versions`, committed alongside the corresponding checkpoint.
Opening an older database archives each existing current value once. History
uses its own table; the existing current-memory table and UI remain compatible.

The model can use `memory_history(path, offset=0, limit=20)` to list revisions,
then `memory_read(path, revision=N)` to inspect an earlier value. Reading history
does not automatically restore it. To restore an earlier value deliberately,
read the current revision and issue a normal conditional write of the chosen
content. Moves/removals leave history at the previous path; deletion removes the
current memory, not the historical copies. There is currently no history-purge
action. This adds persistent disk use, not active-context tokens unless read.

## Finishing an action before retirement

An already-started action frame can continue for up to 128 tokens beyond the
soft retirement threshold, subject to available reserve. Its result and the
subsequent retirement notice must still fit. A completed action commits before
preparation begins; an oversized unfinished frame is eventually cancelled with
no effects. Incoming user events still interrupt at token boundaries.

The available allowance is capped by `turnover_reserve / 8` and remaining space
after two maximum-sized event envelopes plus the native safety margin. It can
be zero with a small reserve; the tested 1536-token reserve allows 128 tokens.
This bounds latency and memory use rather than allowing a malformed action to
postpone retirement indefinitely.

Retirement notices and the pressure-test instructions no longer require sleeping
after preparation. A chosen but unfinished reply may continue during preparation
or after retirement. Sleeping and deciding whether to communicate remain model
choices. No scheduler synthesizes or automatically retries a public message.

See [context-pressure results](context-pressure.md) for actual model behavior,
native restoration evidence and the distinction between these mechanical
guarantees and semantic memory quality.
