# Context-pressure experiment

The disposable Qwen 4B experiment drives actual queued input through a
4096-token native context. It preserves the initialization prefix, warns the
model before retirement, allows bounded memory actions, removes older KV
positions and shifts the retained positions. Open WebUI does not create a
replacement summary or rebuild the native sequence.

## Trigger and visible headroom

The soft threshold is **actual native context capacity minus `turnover_reserve`**.
Native allocation can round up the configured `n_ctx`; use `/api/status`'s
`context_capacity`, not the requested size, for this calculation. New launches
expose `context_retirement` with the threshold, tokens remaining, preparation
budget, maximum action grace and completed count. The panel shows the threshold
and completed count without exposing private thought or memory contents.

Retirement begins before the next token or incoming event would cross that
threshold. An event can trigger it earlier because the complete event must fit.
Only an already-started action can receive bounded extra room, at most 128 tokens
and less when reserve is small. Preparation itself is limited by
`preparation_tokens`, remaining space and the instance's choice to sleep or stop.

For example, capacity 60,160 with a 1,536-token reserve gives a **58,624-token**
threshold. At 55,930 tokens, 2,694 remain before that boundary. A preparation
budget of 384 is an opportunity to preserve new information, not a guarantee of
384 uninterrupted generation tokens; action results also use the reserve.

DMN preserves the initialization prefix, protected imported instructions and
active agreement, plus the recent native sliding window. Normally it removes
half the oldest unprotected gap, or more if needed to accommodate the incoming
event and reserve. An imported instruction block can split the old history into
gaps, so the first retirement may remove much less than half the entire context.
It shifts retained native positions and checkpoints after materializing/packing
the shift. Durable memories remain unchanged. There is no automatic summary or
semantic guarantee that every relevant detail was saved. Unsupported retirement
or insufficient removable space saves and pauses with `context_full` before
attempting normal generation past the native limit.

This display is additive; updating files does not reload an already-running
Python process. Existing instances acquire the status fields at their next
agreed restart. Optional [working memory](working-memory.md) additionally lets
the instance protect a note and a selected raw-token interval. Retirement works
around those pins while retaining the same native recent-window requirement.

Run a new experiment directory:

```powershell
.venv-gpu/Scripts/python.exe scripts/run_context_pressure.py --output data/context-pressure-new
```

The configuration is `examples/qwen4b-pressure.json`: F16 K/V, Flash Attention
off, 1536 tokens reserved for retirement and up to 384 generated preparation
tokens. The preparation limit is a maximum; sleep and remaining space can end
preparation earlier. Action-result tokens also consume space. These intentionally
small limits produce frequent retirements without allocating a large cache.

Each case introduces a phrase and a pending task, then adds explicitly labelled
irrelevant load events. The runtime handles every event normally. The harness
tracks the original input's token positions through retirement and checks the
actual memory database, generated memory actions and public recall responses.
It records early saves separately from saves made during the warning window.

The first process suspends and exits. Before exit, an explicitly discarded,
backend-only diagnostic continuation records 24 next tokens and their freshly
computed logits. A new process loads the saved native cache with prompt
evaluation forbidden and compares against that continuation. Diagnostic branches
execute no actions and do not alter the committed runtime checkpoint. A failed
comparison stops the experiment and leaves a partial report.

## Initial findings

`data/context-pressure-01` completed two cases and seven native retirements.
Both phrase/task pairs were present in `/pressure/carry`, and successful writes
occurred during preparation. The model also saved both pairs before pressure,
so this is not evidence that all retained information was first consolidated
only in response to a warning. Case one did not produce the requested public
recall message; case two returned the correct contents.

The full-process comparison failed on the first newly decoded token's logits:
maximum absolute difference **0.3391427994**. Phase two stopped before resuming
the runtime or attempting case three. This stronger test supersedes any broad
interpretation of the earlier small-shift F16/no-Flash-Attention pass.

`data/pressure-restore-diagnostic-01/report.json` isolates the failure:

- The native state saved after reloading was byte-for-byte identical to the
  original file (SHA-256 `704974438a8a04e09fa92aa6979cc26a16ea2bdc4b67e126cffec8099c09fb6d`).
- Two independently restored continuations matched each other's logits exactly.
- Compared with the uninterrupted continuation, the same 24 forced token IDs
  produced logit differences up to **0.5300443172**. Sample choices differed at
  zero-based steps 4 and 23 even with the same preceding token IDs and RNG.

The difference is therefore reproducible, not merely a damaged checkpoint or
different sampled text fed back into the model. The pinned upstream cache
[writer and reader](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-kv-cache.cpp)
save occupied cells and restore them packed together. Physical layout and
subsequent numerical execution are under investigation; byte-identical saved
values alone do not establish an identical future continuation.

## Native layout experiment and correction

`data/native-layout-probe-01/report.json` compares a deliberately fragmented
control with three retirements followed by native packing. The control's maximum
logit difference was **0.8626432419**. All three packed comparisons matched 24
next samples and logits with **zero** difference. In every cycle, the saved
native bytes before and after packing were identical and no tokens were
reevaluated. This isolates a useful layout correction without claiming to have
identified the exact CUDA kernel responsible for the earlier difference.

The backend now packs native state immediately after the decode that applies
retirement's position shift. The public `llama_state_get_data` and
`llama_state_set_data` APIs copy the existing occupied cells through host RAM.
This is an explicit part of retirement, not token reconstruction. The cache's
physical placement changes; serialized K/V values, token IDs and sampler RNG
are preserved. Subsequent live generation and checkpoint restoration then
start from the same packed layout. Engine metadata records
`pack_after_retirement_v1` and the number of completed packs.

The copy temporarily needs host memory approximately equal to the serialized
native state and adds retirement latency. This cost needs measurement at the
primary model's larger context. It does not retroactively prove that a legacy
checkpoint matches its pre-save live continuation, nor does it certify Flash
Attention, quantized caches or another host. The existing Open WebUI test process
was left running with its already-loaded code; new launches use the correction.

The fresh end-to-end run in `data/context-pressure-02` passed a full process
restart after seven retirements: **2,160 native tokens restored**, **zero prompt
tokens reevaluated**, and all **24 next samples and freshly computed logits
matched exactly** (maximum absolute difference 0). Evidence is in
`native-comparison.json`; its snapshot SHA-256 is
`04570ebc344e2deed142cb561b12b0731da538b212e1056db7c4bb7055fa7eb3`.

## Trial 02 behavior result: failed

The corrected run completed **13 retirements**, generated **3,137 tokens** and
ended suspended with its model unloaded. Preparation used 85–176 generated
tokens per retirement, ending because the model slept rather than exhausting
the 384-token budget. All audited shifts retained the initialization prefix and
newer token IDs without replay. There were no runtime errors or reconstruction.

The memory checks deliberately distinguish a successful write at one boundary
from successful preservation through subsequent activity:

| Case | Facts saved at the selected pressure boundary | Public recall outcome |
|---|---|---|
| `cedar-lantern-573` + moss task | Correct; memory did not exist before the load sequence | No reply |
| `silver-orchard-826` + rainfall task | Correct | Correct phrase and task |
| `violet-compass-194` + shade task | Correct initially | Wrong phrase; task remained correct |

All three original input spans were fully retired at the selected boundaries.
In case three, retirement 11 saved the correct facts, but retirement 12 generated
a **new, successful memory_write** replacing the phrase with
`cloud stone river branch copper meadow lantern pebble`, words from the labelled
irrelevant input. A later memory_read returned that incorrect content, and the
model reported it publicly. Retirement 13 repeated the incorrect write. The
final database contains the wrong phrase. The runtime faithfully persisted a
model mistake; this was not a missing database write or failed native restore.

The restart probe actually read the correct previous memory, but did not finish
its requested public reply. Retirement can cancel an unfinished message; the
model then saved memory and slept without retrying. These test instructions
explicitly ask for sleep after preparation, so this run does not isolate whether
a different instruction would improve completion of the pending reply.

`report.json` therefore records `behavioral_checks_passed: false`, separately
from the passing native comparison. The runner also checks final memory and
public recall, preventing the three initially correct writes from being reported
as an overall success. Historical `memory_before_pressure` means before the
explicit load-event loop; automatic retirement can already happen while the
model responds to the case's initial input. The journal identifies those cases.

This failure motivated read-before-replace checks, recoverable memory revisions
and changes to action completion around retirement. The runtime must continue
to record failures rather than secretly repairing memories from the harness's
expected answers. See [memory revisions](memory-revisions.md) for those changes.

## Trial 03: memory and recall checks passed

`data/context-pressure-03/report.json` records the run with the revised memory
protocol, retirement guidance and pressure-test instructions. It completed
**14 retirements** and generated **3,684 tokens**, then checkpointed and unloaded.
Thirteen shifts were directly audited; one occurred while the resumed Runtime
constructor appended its factual resume event, before the harness installed its
shift observer. The committed runtime counter includes all fourteen.

- All three original case input spans were retired. Each phrase/task pair was
  correct at its selected pressure boundary and still correct after recall.
- Each case delivered exactly one correct public recall reply. The additional
  restart recall also delivered exactly one correct reply: **4/4** checks passed.
- The final memory retained `violet-compass-194` and the shade task. The history
  contains exactly three committed versions: the three intended case updates.
  There were no filler replacements or gratuitous rewrites.
- The model read revision 1 before intentionally replacing it with case two,
  and read revision 2 before replacing it with case three. The live run did not
  attempt a blind replacement; regression tests verify its rejection.
- Fresh-process restoration after seven retirements loaded **2,280 tokens** with
  **zero replay**. All 24 next samples and freshly computed logits matched
  exactly, maximum difference **0**. Snapshot SHA-256:
  `1daff62e82840fccc187d735fa52aa33ec90d06f9f9444715bdf8cefe8828f2a`.

The report now checks public replies, memory after each recall, final memory,
original-source eviction and restart recall separately before reporting
`behavioral_checks_passed: true`. It includes the committed version history.
The model continued several requested replies during retirement preparation.
Its `action_grace_tokens` counter was zero, so this live run does not establish
that the completion allowance caused the improvement. Regression tests directly
exercise short-frame completion, no duplicate message, and bounded cancellation
of an overlong frame. **56 tests pass**, including native CPU continuation tests.

This is one successful bounded real-model experiment, with both runtime and
instruction changes. Revision checks prevent blind/stale overwrites; they do not
prove that every deliberate future edit will be semantically correct. Historical
versions make such edits inspectable and recoverable by the model. The original
31B conversation and the already-running Open WebUI test were left unchanged.
Existing checkpoints retain their original action protocol; fresh instances use
the revised contract. Further model/host configurations need their own tests.

## Scheduler corrections

Incoming events now trigger retirement before consuming the preparation reserve.
Preparation reserves space for one maximum-sized action result instead of using
the former fixed 640-token cutoff. An action already completed by a sampled token
commits its result and effects before a new preparation turn can run. Retirement
itself is immediately checkpointed.

`last_context_retirement` in status and checkpoint state records the preparation
token count, stopping reason, committed actions and whether a partial action was
cancelled. Only successful, committed memory writes are listed in the model's
retirement notice. An incomplete action has no effects and is explicitly marked
as cancelled.

These are limited synthetic tests of one model and configuration. Keeping newer
KV entries preserves their cached representations, but removes direct attention
to the retired entries. Memory quality, faithful recall and native restoration
are separate things to test. No primary Open WebUI conversation is imported.
