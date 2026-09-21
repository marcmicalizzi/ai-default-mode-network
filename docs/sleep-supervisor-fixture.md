# Compiled review and the sleep transition engine

This increment implements compiled-plan review and a durable sleep transition
engine. **It does not implement a production trainer, hard resource governor, or
an unattended inference/training service.** Ordinary `dmn run` cannot execute
`deep_sleep`. The execution path is restricted to explicitly enabled disposable
mechanics tests, using a reviewed prebuilt adapter. It performs no training and
prepares a wake checkpoint without automatically starting generation.

The next integration is the separate, resource-controlled training/conversion
worker and continuous supervisor service. Existing instances need neither new
dependencies nor a restart for this development work.

## Review before a sleep request

`learning_execution_help` provides the complete action contract through pages,
keeping its detailed instructions out of the permanent bootstrap prefix.
The workflow is:

1. The instance creates a learning draft as described in
   [adapter identity and drafts](adapters-and-learning-plans.md).
2. A host can offer a recipe through `Runtime.offer_learning_recipe`. This stores
   an immutable recipe and queues an offer event. It cannot grant approval.
   There is no arbitrary shell-command recipe. Only `fixture_candidate_v1` exists.
3. The instance inspects `learning_recipe_list` and `learning_recipe_read`, then
   requests `learning_compile(draft_revision, recipe_revision)`.
4. The compiler binds the current parent weights, source draft, exact input and
   target token IDs, target-only labels/loss masks, deployment strength, requested
   choices, offered limits, candidate identity and implementation source hashes.
   A token crossing the input/target boundary causes rejection: the compiler
   neither guesses its mask nor silently inserts a separator. It adds no prompt
   template, BOS or EOS and never truncates a selected example.
5. The instance reads the complete compiled plan using consecutive
   `learning_execution_read` pages, then uses `learning_execution_decide` with
   `approve`, `decline` or `defer`. Truncated pages do not count as review.
   Retirement and restart clear review progress. Approval/withdrawal and its
   audit record commit with a checkpoint. External input cannot supply approval.
6. In the disposable harness only, an approved `deep_sleep(revision)` commits the
   source checkpoint, run ID, `Saved` phase and consumed approval in one
   transaction, then stops generation. Ordinary sleep never takes this path.

A withdrawn or replaced source draft invalidates its compiled plans. A change
to the bound implementation requires a new compilation and review before sleep.
Once asleep, preserve the implementation bound by the plan for recovery.

The current fixture explicitly says it **will not learn the selected examples**.
It copies the exact reviewed candidate and replaces the parent adapter set only
if the plan selected automatic adoption after its two mechanical checks. Those
checks verify artifacts and retained tokens/RNG, not beneficial behavior. Other
requested checks are rejected rather than silently omitted. Requested rank,
alpha and training steps remain visible preferences, not purported work done by
this non-training fixture. A future real recipe must establish training-tokenizer
parity and implement the requested training and evaluation.

## Durable phases

`run_fixture_sleep(instance, run_id)` acquires the exclusive instance lock and
refuses a running owner. The saved phase blocks ordinary startup—including
fallback/rebuild recovery—before any backend loads. This also protects the gap
between releasing the inference owner's lock and acquiring the supervisor lock.
Lifecycle and source-checkpoint identities are checked before work/publication.

| Phase | Durable meaning and recovery |
| --- | --- |
| `Saved` | Explicit request and source checkpoint committed; work has not begun. |
| `Training` | Candidate work began. For this fixture it is only a copy. A complete, hash-bound completion file is reused after a crash; unknown partial work follows the failure choice and is not automatically repeated. |
| `Candidate` | Complete artifact matches the reviewed candidate identity. |
| `Rebuilding` | Preauthorized adoption: evaluate the exact retained token IDs with the selected adapter, without sampling or running historical actions. |
| `OldStateReview` | Review-first choice: preserve the original native bytes/logits and prepare a factual candidate report. |
| `FailurePolicy` | Apply the plan's `wake_previous` or `remain_stopped` choice. |
| `Stopped` | Ordinary startup remains blocked; neither new input nor elapsed time overrides this choice. |
| `WakeCommitted` | Checkpoint pointer, completion phase, consumed plan and report committed atomically. Repeating the supervisor call does no work. |

Two deterministic checkpoint staging slots per cycle cover candidate and original
weight wake. Retrying an interrupted build reuses its slot, avoiding another
orphan allocation on each crash. Complete wake files can be reused when phase
publication was interrupted. The original checkpoint is preserved until the new
pointer commits. A failed transaction never selects the partial wake.

Reconstruction preserves retained token IDs and sampler RNG. Runtime identity,
agreements, parser state, memory revisions, delivered-message IDs and event cursor
remain unchanged. Pending input stays in the database and never becomes training
material. On the next ordinary launch, a factual sleep-wake event is appended
before generation/input delivery. The newly prepared native checkpoint itself
uses normal strict restore; its retained tokens need no second reconstruction.
Let the launcher use the committed configuration after an adapter adoption;
an explicit old configuration will correctly fail its weight-identity check.

Review-first currently prepares the old-weight wake and leaves a report accessible
through `learning_sleep_report`. Interactive adoption following that review, a
continuous frontend while workers are absent, worker process containment,
cancellation UI and automatic return to generation remain integration work.

## Resources and privacy

The fixture admits only the demo backend or small CPU models (zero GPU layers,
K/Q/V GPU offload disabled, one CPU thread, at most 32K context and a 4 MiB model).
Candidate files are limited to 128 KiB. It checks free disk space and the proposed
workspace size before work. RAM preflight is deliberately described as preflight,
**not a hard limit**; there is no production RAM/time governor. This is why the
ordinary launcher does not enable execution, even for an approved plan.

Plans and decision records stay in the instance database. Candidate-completion
records occupy known files under `sleep/<run-id>/`; adapters retain the managed
content-addressed layout. Packaging includes these files, cache-migration backups
preserve them, and managed erasure removes them without following links or
deleting unknown files. No training dependencies or 31B source weights are added
by this implementation.

## Reproduce validation

```powershell
.venv\Scripts\python.exe -m unittest tests.test_deep_sleep -v
```

The dependency-free tests inject generated action frames into disposable demo
instances. They cover full-review requirements, token masks and boundary rejection,
source-draft invalidation, rollback, exclusive ownership, queued input, both
adoption/failure preferences, erasure, and crashes before candidate completion,
after completion, during wake staging and around atomic publication.

For native validation, reuse a completed [tiny training probe](lora-training-probe.md):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:DMN_TEST_LORA_PARENT = 'D:\path\to\completed-tiny-training-probe'
.venv-gpu\Scripts\python.exe -m unittest tests.test_deep_sleep.NativeSleepTest -v
```

This uses the existing trained tiny adapter but performs no new training. The
random fixture's action frames are injected by the test: they are not evidence
of a real instance's consent. It verifies the native adapter transition,
retained-token/RNG equality, unchanged delivered-message count, and subsequent
strict restoration with zero prompt replay. Its enlarged context is a mechanics
test and says nothing about learning quality or long-context model capability.
