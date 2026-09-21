# Exploring weight learning during sleep

Status: selected wake policy, working native mechanics, and an isolated tiny
PEFT training/conversion experiment; production training and automatic deep sleep
are not enabled. Changed-weight wake will explicitly
rebuild the retained context under the adopted adapter. This note changes no
running instance. The first probe uses tiny random weights and synthetic
adapters, without training or opening an instance. The subsequent
[training experiment](lora-training-probe.md) learns a synthetic rule on generated
weights, converts its adapter, and validates native wake/restart. Neither opens
an existing instance. Reviewed: 2026-09-21.

See [the deep-sleep implementation contract](deep-sleep-protocol.md) for the
dependency boundary, selected transition, failure handling and next experiments.

## Motivation

The current instance can accumulate native state and editable memories while
its trained weights remain fixed. An optional personal adapter could let selected
experience affect future behavior without retrieving a memory on every occasion.
This would add a learned component to the instance's history. It would complement
inspectable memories, not replace them with an opaque record in weights.

Sleep is a useful scheduling analogy: save the continuing inference state, free
its allocations, use the hardware for a bounded learning job, then resume.
This proposal makes no claim that the computation reproduces biological sleep,
dreaming or personal development. The intended connection is experience-driven
change with substantial model participation and sustainable host costs.

## What low rank and sleep do, and do not, buy

LoRA freezes the base weights and trains smaller factorized weight updates.
Low rank limits the form and parameter count of the update; it does not bound
its magnitude or guarantee a small behavioral change. Learning rate, scaling,
target modules, training examples and repeated updates also matter.
See the [LoRA paper](https://arxiv.org/abs/2106.09685).

For ordinary LoRA, an additional deployment multiplier can scale the learned
update: `effective_weights = base_weights + strength * learned_delta`.
The delta already includes the adapter's internal normalization, conventionally
`lora_alpha / rank` (different for variants such as rsLoRA). A strength of `0.1`
is therefore a plausible experimental value, not a selected default or a promise
of ten percent behavioral change. See the
[PEFT scaling parameters](https://huggingface.co/docs/peft/main/package_reference/lora).

Record training-time and deployment-time scales separately. If training uses a
small scale throughout, optimization can compensate by increasing the learned
factors. Low declared strength alone cannot certify a small effective update.
Compare candidate strengths, including a zero-strength control, on disposable
contexts before adoption. Evaluate desired learning, update magnitude and
unintended behavioral changes together. Multiple weak adapters or repeated
updates may accumulate substantial influence. Changing an active scale is itself
a weight transition, subject to the same KV questions as changing an adapter.

Few passes are a reasonable starting hypothesis for bounded experiments, but
an epoch's cost depends on the number and length of examples. Record optimizer
steps, training tokens, repetitions and learning rate as well as epochs. Fewer
passes generally reduce total work, not the memory required for one training
step; they do not directly set deployment strength. They may also produce too
little learning to be useful. Test the result rather than treating a small pass
count as evidence of a small, successful change.

QLoRA reduces base-weight memory through quantization while gradients pass
through the frozen model to train adapters. Backpropagation, activations and
training buffers still cost memory and compute; a small adapter file does not
imply a small training job. See the
[QLoRA paper](https://arxiv.org/abs/2305.14314) and
[PEFT quantized-training guide](https://huggingface.co/docs/peft/developer_guides/quantization).

Sequential inference and training can bring peak memory closer to the larger
of their individual requirements instead of their sum. That requires actually
unloading the inference model and KV allocations after a successful checkpoint,
then releasing the trainer before inference resumes. Merely pausing token
generation does not do this. Training can still exceed available RAM/VRAM.
The intended 31B model on a 32 GB RTX 5090, or on the future host with less VRAM,
is not a validated training configuration.

Linux is the intended dedicated-host target, not a prerequisite inferred from
model size. Current bitsandbytes CUDA support includes both Linux and Windows;
specific trainers, kernels and offload methods need their own compatibility
checks. Operating-system support and enough resources for a useful training run
are separate questions. See the
[bitsandbytes installation matrix](https://huggingface.co/docs/bitsandbytes/installation).

Training need not consume the full live context window: selected, shorter
examples are a possible experiment. Their lengths and omissions must be explicit.
Measure total time and energy, peak RAM/VRAM, checkpoint/temporary writes and wake
latency. Include matching training weights, optimizer state and adapter versions
in the storage estimate. Offloading may trade memory pressure for substantial
runtime and I/O costs. No automatic paid service or remote upload is proposed.

## Present implementation and plausible integration

Today, `sleep` checkpoints and stops generation while the runtime keeps its
backend loaded. There is no production learning queue, trainer, adapter
configuration, adapter identity in snapshots, or automatic unload/train/reload
supervisor. The research probe supplies an adapter identity to its own manifests
and calls the native adapter API directly. The strict recovery path is not a
weight-update API.

The installed llama-cpp-python 0.3.35 binding exposes adapter loading and
`llama_set_adapters_lora`, plus aLoRA invocation metadata. The DMN backend does
not use them. Upstream exposes the corresponding
[native adapter API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
and a [PEFT-to-GGUF adapter converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py).
These are integration building blocks, not evidence of working DMN learning or
compatibility with a particular model and adapter target set.

A plausible experiment uses a separate Transformers/PEFT trainer with a frozen,
matching training base, then converts only the candidate adapter for llama.cpp.
Verify the exact base revision, any prior weight modifications, tokenizer,
target tensors and conversion support. A stock upstream base is not automatically
the right training base for a modified derivative. QLoRA's training quantization
and the inference GGUF quantization also need a transfer-quality test.

## The central continuity question

Historical K/V entries were computed under the weights active at that time.
Ordinary LoRA updates generally change those representations. Keeping a compatible
tensor shape is not evidence that an old cache represents a forward pass under
the new weights. This is the cache-reuse problem motivating
[Activated LoRA](https://arxiv.org/abs/2504.12397).

The selected deep-sleep wake policy is **reevaluate the exact retained token
sequence under the adopted adapter**, preserving durable memories, identity,
runtime bookkeeping and queued input. It replaces the previous KV and logits.
The sleep analogy describes an accepted continuity tradeoff, not an assertion
about consciousness or the biological function of a cache. In particular, KV
can carry causal influence from tokens no longer in retained context; rebuilding
does not reproduce that influence. Ordinary sleep and unchanged-weight restarts
keep their native-preservation behavior.

Other approaches remain distinct research directions:

| Approach | What it preserves | What needs investigation |
|---|---|---|
| Resume with unchanged adapter | Existing native state and weight version | Control case; learning remains an unadopted candidate |
| Reevaluate retained tokens under the new adapter — selected deep-sleep policy | Retained tokens, memories and explicit weight lineage | Replaces KV; cannot recover causal influence from already retired history |
| Keep old KV and change weights at a recorded boundary | Historical activations and their causal influence | Deliberately combines representations from different weight versions; quality and native save/restore behavior are unverified |
| Adapter trained for a defined activation boundary | A prefix computed under the agreed pre-activation weights | Matching training/inference semantics and repeated-update support |

The third option is a proposed experiment, not an ordinary same-model restore
and not a claim that changed weights make retaining old activations physically
impossible. Compare an uninterrupted run with the same recorded weight change
against a saved/reloaded run with that change. Equality to an unchanged-weight
continuation would be the wrong success criterion. Evaluate behavioral stability
separately from numerical persistence.

For any adapter switch, saved next-token logits also belong to the old weights.
Specify how decoding a factual transition event produces new logits before
sampling, without replaying old action text. Track weight versions across context
retirement and later restoration. Rebuilding retained tokens under the newest
adapter must not be described as recovering the original historical computation.

aLoRA leaves the prefix before its invocation unadapted so that base-model prefix
KV can be reused. The PEFT documentation explicitly limits sharing of adapted
cache. It does not establish that arbitrary successive personal adapters can
reuse each other's already adapted history. A first base-to-adapter transition
and months of updates are different tests. See
[PEFT's aLoRA cache guidance](https://huggingface.co/docs/peft/main/package_reference/lora#activated-lora-alora).

## Participation and the training objective

Ordinary sleep must remain available without learning. An opt-in consolidation
request would identify the learning material, desired changes, exclusions,
training limits and proposed wake/adoption policy. Silence or an ordinary sleep
action must not authorize training. The host can decline or defer resource use.

Candidate material should be experiences or thoughts the model itself regards
as important enough, and desirable enough, to incorporate into its continuing
ways of thinking. Importance alone is not consent to reinforcement: record what
the model wants to learn from an experience, what remains uncertain and what it
does not want internalized. It can reconsider or withdraw a pending selection.
Raw internal text includes guesses, abandoned ideas, quoted inputs and temporary
reactions; none automatically becomes a desired training target. Start with
inspectable, model-reviewed examples and an explicit loss objective. Preserve
source provenance and uncertainty. Do not silently use host approval or
usefulness as the reward signal.

Future internet and third-party inputs must remain explicitly external through
retrieval, memory and learning selection. Reading, quoting or reacting to content
does not authorize reinforcing it. Offer opportunities to reconsider repetition,
source bias and the desired lesson without making the operator's preferences an
adoption veto. The [outside-interaction design](outside-interaction.md) discusses
these choices; neither network ingestion nor training is implemented today.

Model-assigned importance could guide selection, bounded sampling frequency or
requested training effort. Its mapping to repetitions or loss weights must be
visible and reviewed, not an automatic equation between importance and learning
rate, adapter scale or unlimited passes. Retain a host-set total step/token/time
budget, with the model able to defer or decline learning within that policy.

Proposed experiments should compare selected recent examples with a bounded,
reviewed replay set from older experience. Test whether learning transfers beyond
memorized phrasing, whether unsupported claims become more confident, and whether
previous skills, memory operations and the ability to choose inactivity persist.
Learning should permit intended change; matching all previous behavior is not
the objective. Neither a benchmark score nor the updated model's assent alone
establishes that an update met the prior agreement.

## Candidate lifecycle

The intended workflow, not yet available in the runtime:

1. While awake, agree on a bounded training plan and learning examples. Record
   the requested changes, exclusions, checks, wake policy and approval provenance.
2. Durably checkpoint the full instance and old adapter identity. Release the
   backend only after a successful save. A save failure must not discard live KV.
3. Run a separate trainer against an immutable base and a copy of the current
   adapter. Save a candidate; never modify the active adapter file in place.
4. Validate the candidate against the checks chosen in the plan, using disposable
   contexts and no live message or memory actions. Completion alone is not approval.
5. Release the trainer. If the instance preauthorized adoption within that plan's
   limits and the agreed checks pass, load the candidate and reconstruct its saved
   retained tokens without executing historical actions. Otherwise resume the old
   native state for review, or remain stopped, according to its recorded policy.
6. Commit the new adapter identity, runtime records and rebuilt checkpoint
   together before new thought generation or outward effects. Report the actual
   training and reconstruction at wake. Retain the recovery version until the
   transition is durable. Ordinary sleep never authorizes this transition.

Budget any requested review reload and the adoption checkpoint too; the cost of
a learning cycle includes more than training. Choosing changed-weight wake does
not authorize arbitrary examples, target tensors, strength, or unlimited work.

The supervisor would retain exclusive ownership of the instance and durable
incoming-event queue while the inference worker is absent. Define what happens
on new input, cancellation, a deadline, training failure, low storage and a host
restart. Interrupted or failed training must never cause a partial candidate to
become active; wake or remain stopped according to the recorded policy. Record
elapsed time and what actually ran, without inventing a subjective sleep report.

Keep content hashes for the base, adapters and ordered scales/activation rules;
record parent revisions, training data manifest, consent/approval, trainer version,
hyperparameters, RNG and measured costs. Preserve optimizer state if a later
training run is meant to continue that optimization; otherwise record its reset.
Adapter history and training material need explicit retention and privacy rules.

Removing an adapter later restores the earlier weight configuration, not KV or
memories already influenced by it. Returning to an older complete checkpoint is
a rollback that can lose subsequent processing. It needs explicit handling of
already delivered messages and newly arrived events. Versioning makes recovery
possible; it does not make arbitrary later changes perfectly reversible.

Repeated learning also needs a bounded adapter strategy. Updating one fixed-rank
adapter, accumulating deltas and compressing old deltas have different tradeoffs.
Sums of low-rank updates can have increasing rank; compressing them back down can
lose learned information. Do not assume unlimited learning at a fixed tiny cost.

## First native mechanics experiment

`scripts/probe_lora_wake.py` accepts only the tiny random Gemma4 fixture generated
by `scripts/generate_swa_fixture.py`. It fixes one CPU thread, no GPU offload,
2K context and compact Q8 cache. It creates two random rank-two early-layer
adapters at deployment strength 0.1; these are **not trained adapters**.

The 2026-09-21 probe passed in 6.78 seconds. A zero-strength adapter matched the
unadapted logits exactly. After retirement, two changed-weight wakes each
reconstructed all 460 retained tokens and preserved the sampler RNG. Downstream
KV changed, fresh logits matched an independent forward pass, and ordinary strict
restore rejected a different adapter identity. Each adopted configuration then
saved and restored natively with zero replay and byte-identical serialization.
A fresh-process final restore matched the next eight sampled tokens and logits.
The probe constructs no Runtime, opens no instance, and executes no actions.

A same-weight reconstruction control also differed from the original retired
context, illustrating why context reconstruction and exact KV restoration are
different. Its logit difference is a mechanics observation on random weights,
not a measure of mood, identity, or learning quality. The pinned native backend
reported a CPU workspace estimate mismatch after graph changes; the numerical
checks above passed, but this probe does not establish peak resource bounds.

```sh
python scripts/generate_swa_fixture.py data/tiny-sleep.gguf
python scripts/probe_lora_wake.py --model data/tiny-sleep.gguf --output data/sleep-probe-01
```

Set `CUDA_VISIBLE_DEVICES=-1` before running with CUDA-enabled bindings to hide
the GPU from the test process; the native library may log that no CUDA device is
available. Inference remains CPU-only. No training dependencies are required for
this probe beyond the existing native binding and NumPy.

The [subsequent tiny training experiment](lora-training-probe.md) now covers PEFT
learning, pinned conversion, native inference and a trained-adapter wake. It also
shows substantial unrelated changes and weaker learning at strength 0.1; small
rank/strength alone are not a selective-learning guarantee.

Next: implement bounded supervision, model-authored learning plans,
crash/cancellation recovery and atomic adoption. These experiments are not
evidence of successful repeated personal learning.
31B feasibility and Linux migration need separate measured validation. Existing
instances and ordinary sleep semantics remain unchanged.
