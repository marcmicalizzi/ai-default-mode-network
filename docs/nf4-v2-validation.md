# NF4 v2 GPU rehearsal — September 29, 2026

This rehearsal uses disposable synthetic examples. It does not read a running
instance's context, claim an instance's consent, train on its material, or install
an adapter for it. The selected recipe is `peft_gemma4_nf4_v2`.

## Complete service lifecycle on generated tiny weights

The native service rehearsal completed three consecutive cycles:

| Cycle | Training performed | Outcome | Retained tokens |
| --- | --- | --- | ---: |
| Candidate preparation | Yes | Wake under original weights for review | 38,827 |
| Separately reviewed adoption | No | Adopt the prepared candidate | 53,820 |
| Continuation | Yes | Train from and replace the deployed adapter | 57,623 |

Every final service restore loaded native state without replay. Queued input and
previously published messages were preserved. Adoption did not repeat training,
and continuation retained the deployed PEFT lineage. The review/approval actions
were injected test fixtures, not decisions attributed to a model instance.

The tiny base conversion proof was reproduced against the current implementation
before testing; an old proof is correctly rejected after its bound code changes.

## Actual 31B source, 1,536-token example

The separate worker-chain rehearsal uses the pinned Gemma4 31B source, one exact
1,536-token input-plus-target example, rank 2, alpha 4, two AdamW steps and a
deployment scale of 0.1. It retains the full example as one context and uses the
reviewed 64-position vocabulary-loss chunks. No example is split or shortened.

On the RTX 5090 under Windows, the completed training stage measured:

| Measurement | Result |
| --- | ---: |
| Torch allocation peak | 23.24 GiB |
| Torch reservation peak / ceiling | 24.00 GiB / 24 GiB |
| Observed whole-device peak, including other applications | 30.33 GiB |
| Whole-device watchdog allowance | 31 GiB |
| Worker committed-memory peak | 30.78 GiB |
| Worker committed-memory allowance | 32 GiB |
| Two gradient steps | 10.36 seconds |
| Complete training worker, including verification/loading/evaluation | 426.13 seconds |
| Fresh-process reload worker | 251.23 seconds |
| Conversion worker | 178.89 seconds |
| All three workers combined | 856.25 seconds (14.27 minutes) |

The frozen base remained unchanged. Fresh-process reload reproduced every adapter
factor and selected-example loss exactly at both training and deployment scales.
The selected loss changed from 0.117128 to 0.111244 at training scale and 0.116825
at deployment scale. These repeated synthetic examples test mechanics, not useful
learning or generalization.

GGUF conversion passed: all 240 factors exactly matched the saved PEFT tensors,
and the adapter alpha matched the reviewed plan. All three worker stages finished
successfully and left no child processes running. The adapter remains a disposable
test artifact; it was not installed for an instance.

## Scope and reproduction

The full service handoff and native wake were exercised on the tiny fixture.
The 31B trial is the separate training/reload/conversion chain; this pass does not
rerun the 60,000-context 31B native service rehearsal documented in
[reviewed NF4 training](reviewed-nf4-training.md). These are distinct checks.

Shared GPU memory was not separately sampled in this pass. Torch reservation,
whole-device usage and committed system memory overlap and must not be added
together. The measurements do not guarantee that all permitted examples fit every
machine. The earlier [length experiments](longer-training-examples.md) remain the
comparison for 1,024, 1,536 and the failed 2,048-token workload.

Run `scripts/validate_sleep_service.py --recipe-version v2` with a freshly
reproduced tiny proof and the CUDA training interpreter for the three-cycle test.
Run `scripts/validate_reviewed_nf4_31b.py --recipe-version v2 --tokens 1536` with the
bound full-source proof, local v2 recipe and training interpreter for the 31B chain.
Both require new output directories and the additional paths listed by `--help`.
Preserve failed attempts as evidence instead of changing their receipts.
