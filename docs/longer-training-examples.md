# Longer training examples: queued after working memory

The next experiment will investigate 512- and 1024-token examples, then larger
sizes if measurements justify them. The current production recipe still rejects
examples above 256 tokens; this document neither changes that limit nor promises
that a larger size fits. Longer execution times are acceptable within an explicit
host time allowance and an instance-reviewed plan. Additional time does not by
itself remove a peak-memory allocation requirement.

The earlier synthetic 31B experiment at 512 tokens failed while requesting a
512 MiB allocation under a **22 GiB Torch allocator ceiling**. Torch reported
21.68 GiB allocated, about 90 MiB reserved but unused, and 4.66 GiB of device free
memory. Windows per-process memory figures in that error were nonsensical and
must not be interpreted as real usage. The result establishes failure of that
bounded workload, not exhaustion of every byte on the RTX 5090. The 256-token
probe passed. These are old research-probe results, not a length validation of
the current production trainer's target-only loss implementation.

## Experiment order

1. During separately agreed GPU maintenance, measure a current production-shaped
   synthetic baseline: rank 2, batch size 1, target-only loss and two optimizer
   steps. Record allocated/reserved Torch memory, whole-device peak, RAM, phase,
   runtime, finite gradients and unchanged frozen state. Use fresh workers and
   no instance material or adapter adoption.
2. Try 512 and 1024 at explicit resource ceilings. The current recipe envelope
   permits up to 24 GiB of Torch memory, with a separate total-device allowance;
   begin by testing whether that additional room is sufficient in a disposable
   research path. Do not simply remove the production length check.
3. If peak memory remains the constraint, compare memory-saving attention,
   chunked vocabulary projection/loss, or activation offload to RAM. The current
   worker already uses non-reentrant gradient checkpointing, frozen NF4 weights,
   batch size 1 and no inference cache. It currently uses eager attention and
   materializes full vocabulary logits. Offloading unused vision weights is
   already implemented; general training activation/decoder offload is not.
4. Verify any alternative against the existing target-only labels, loss and
   adapter gradients on tiny CPU fixtures where possible, then against the
   actual GPU implementation. Preserve Gemma attention/soft-capping semantics,
   exact selected examples, repeatable reload, converter checks and wake safety.
5. Offer a versioned, reviewed longer-example recipe only after the measured
   workload succeeds. Keep the old recipe available, state numerical differences
   honestly, and make time/RAM/VRAM limits explicit. A new offer never approves
   training or changes an already reviewed plan.

More complete trajectories may be useful material selected by the instance.
Longer examples alone do not establish useful learning, preserve an earlier KV
state, or resolve the risks of repeated self-training. No automatic splitting,
truncation, training-data selection, or claim of cognitive benefit is implied.
