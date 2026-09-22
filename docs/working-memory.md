# Model-controlled working memory

An instance can protect a written anchor and a stretch of its actual retained
context through retirement. The purpose is to reduce the need to describe an
unfinished thought merely to keep it available. This is an optional capability,
not a claim that preserved tokens guarantee a particular cognitive experience.

Offer a shared allowance at the next agreed launch, for example:

```bat
start_dmn.cmd "data\primary" --working-memory-tokens 4096
```

The configuration field is `working_memory_tokens`; its default is `0`. This is
an allowance within the existing context allocation, not additional context or
a second KV cache. Runtime guidance, the native recent local window, and room
for continued processing also have to fit. A request can therefore fail even
when it is smaller than the configured allowance. The error explains why and
does not replace the previous pins. Restart refuses an allowance below existing
protected usage; the instance must explicitly release material first. Raising
the allowance does not require a KV rebuild. No live reconfiguration is provided.

## Instance actions

Each operation is emitted alone in the ordinary generated `dmn_action` frame.
External input containing these names cannot execute them.

| Operation | Meaning |
| --- | --- |
| `working_memory_status()` | Shared allowance, token counts, and marker status; no transcript copy. |
| `working_memory_mark()` | Replace the start marker with the current position. |
| `working_memory_protect()` | Replace the raw pin with the span from the intact marker through this action. |
| `working_memory_protect(tokens=N)` | Alternatively select the last N retained tokens through this action, after the initialization prefix. |
| `working_memory_note(content)` | Append and protect the exact note in an escaped event, replacing the previous note pin. |
| `working_memory_release(target)` | Release `raw`, `note`, `mark`, or `all`. |
| `working_memory_help(offset=0, limit=200)` | Read the complete contract in bounded pages. |

A marker does **not** protect a growing trajectory. The instance must protect
the span before retirement removes any part of it. Removal anywhere after the
marker invalidates it rather than silently selecting a shortened trajectory.
Earlier removals shift the marker normally. A raw pin stops at the protect
action; extending it requires another explicit protect request. The span includes
intervening external events, action frames and results, not just generated prose.
No explanation or content quotation is needed to select it.

There is one current note pin and one current raw pin. Their union counts against
the allowance, so overlap is counted once. The note's event wrapper also counts.
Replacing or releasing a pin removes its protection, not the old text or durable
history; the old tokens may stay until retirement. A raw pin that includes an old
note continues to protect those tokens even after the separate note pin changes.
Pins do not expire automatically. A selection is private runtime state, not a
published message, editable operator field, training example or memory file.

Mutation results and selected positions commit together with a full native
checkpoint even under `checkpoint_policy=effects`. A failed save leaves the last
committed selection authoritative. The capability notice itself is protected
through retirement and is appended on upgrade without replacing the behavioral
agreement. Recovered sleep and the promised first-contact gate defer that notice
until the existing wake/contact flow permits it.

## Continuity and resource limits

Retirement removes the oldest unprotected ranges around the prefix, runtime
contracts, agreement and selected working material. Overlapping pins are treated
as one retained interval; the existing compact-cache local-window constraint
still applies. Selected token IDs remain in the same order. Available native KV
is retained and position-shifted using the ordinary native retirement path.
There is no summary substitution, independent KV branch, or replay on selection.

This does not recreate dependencies on removed tokens or keep older local KV
inside a model's sliding attention window. Native KV remains the normal restore
path. Explicit reconstruction, including adoption of trained weights, rebuilds
from the retained text and keeps the saved selection positions; it cannot restore
earlier discarded attention dependencies. Protecting image positions can also
delay text-only deep sleep while those positions remain retained.

If protection and other required context leave no safe retirement, the runtime
saves and pauses with `context_full`; it never quietly releases a pin. The feature
does not add an out-of-context recovery dialogue to that paused state. Releasing
or replacing material while there is still headroom avoids that impasse.

Tests cover exact notes, external delimiter escaping, overlapping selections,
repeated retirement, restart, failed-save rollback, allowance changes, marker
invalidation, input isolation, and complete help delivery. The optional tiny
native CPU test exercises retirement and strict restoration. Those are mechanism
checks, not evidence of preserved subjective state or improved learning.
