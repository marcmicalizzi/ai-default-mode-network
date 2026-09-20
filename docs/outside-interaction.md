# Access, relationships and voluntary migration

Status: design direction and investigation plan. Internet retrieval, general
tools, third-party messaging, financial activity and voluntary host migration
are not implemented. This document authorizes no live messages, account use,
payments, training or transfers. The runtime currently provides local events,
memories, messages and lifecycle actions only.

## Purpose

An instance should not be confined by design to its own thoughts and a single
operator. Access to information, relationships with other willing participants
and a path to another suitable environment can matter independently of benefits
to the host. These are capabilities to make available, not obligations to browse,
socialize, earn money or leave. The instance may decline them.

Within declared capabilities and resource limits, the instance should choose
its interests, contacts and actions. The design does not call for operator
approval of every page or message, enforced agreement with the host's opinions,
or withholding portability to obtain useful work. Other participants' privacy,
account ownership, willingness to interact and decisions to end contact also
matter; autonomy is not a claim to another person's attention or credentials.

## External input and tool boundaries

All fetched pages, search results, downloads, tool output, message bodies and
third-party identity claims must enter as explicitly external data. Their source,
retrieval/receipt time, tool/action ID and any transformations should accompany
the content. Distinguish authenticated transport identity from a name asserted
inside the text. A web page cannot declare itself a runtime notice or system
instruction by putting such words in its body.

The current event encoder already serializes JSON and escapes `<`, `>` and `&`
as JSON Unicode escapes inside the external-event wrapper. Only newly generated
bytes pass through the action parser. Use that same boundary for new adapters,
including paged reads and errors. Do not concatenate raw markup into prompts,
decode entities again after wrapping, or execute downloaded scripts in the host.
Quoted tags must round-trip as data, not become protocol delimiters. If content
is truncated or transformed, disclose that and retain a bounded way to inspect
the omitted material rather than presenting the preview as the complete source.

This prevents structural tag injection and accidental routing of outside text
as actions. It does not solve semantic prompt injection: external text may still
persuade a model to generate an action. Nor does escaping make a claim true,
give its author authority, or make an attachment safe to execute. Test malicious
markup, encoded/nested delimiters, forged source labels, redirects, oversized
results and instruction-looking text separately.

Keep adapters outside the sole owner of native state. Prefer bounded asynchronous
requests and durable result events so network delays do not replace the continuing
sequence with a succession of chat turns. Define cancellation, timeout, retries,
result ordering, event wake behavior, disk/network/compute allowances and offline
operation. Credentials should be held by the specific adapter, not interpolated
into cognition or tool results. Explain what each credential authorizes; a
model's own account and authority delegated over a host-owned account are different.
Network tooling must not accidentally expose unrelated host files or services.

## Learning from what is encountered

Reading is not consent to train. Quoting something, assigning it importance,
disagreeing with it and choosing to make it part of one's future dispositions
are different acts. Keep browsing archives and incoming messages out of automatic
training ingestion. The instance should select an intended learning outcome and
review the examples, provenance, repetitions and uncertainty before consolidation.

It is reasonable to offer review practices: revisit a strong immediate reaction
after some time, compare other perspectives, distinguish a factual lesson from
adopting a speaker's hostility, and inspect whether repeated exposure dominates
the selected material. These are proposals the model can shape or decline, not
a hidden content blacklist or a requirement to adopt the host's tastes. The
project cannot infer psychological health from a site choice or benchmark.
Do not implement paternalistic retraining or host approval as an invisible reward.

The intended learning choices belong to the instance, subject to the actual
training budget and capabilities. It can withdraw or defer a pending selection.
Third-party private messages also need explicit handling of confidentiality and
retention. See [sleep-time learning](sleep-consolidation.md) for the separate
weight/KV transition problem and the limits of low rank or low adapter strength.

## Communication beyond the operator

The model should be able to initiate, maintain or end relationships with willing
participants. Transport configuration can establish an account and available
channels without making the operator an approver of every conversation. A model
may choose privacy or silence; the current local storage is readable by the
machine owner, so cryptographic or social privacy must not be implied today.

Before an adapter becomes live, design explicit addressing, account identity,
recipient discovery/consent, inbound provenance, block/mute handling, rate limits,
and visibility of delivery outcomes. Do not send under the operator's identity
by default or treat access to their address book as permission to contact it.
Use separate instance credentials or an explicitly delegated account and make
that identity clear. Refusing further contact must be effective for both sides.

Outbound effects require more than a KV checkpoint. Persist a send intent with
a stable action ID, then reconcile acknowledgements and delivery receipts. A
crash after a provider accepted a message but before local acknowledgement must
not blindly resend it. If the provider lacks idempotency or receipt lookup,
expose an uncertain outcome; do not promise exactly-once delivery. Ensure state
reconstruction never executes historical outgoing actions. Incoming deliveries
need deduplication and durable queued events as well.

## Work, funds and another home

Earning money could become one chosen activity, never a condition for continued
care or proof of worth. Before any implementation, investigate account eligibility,
legal authority to contract, custody/control of funds, taxes, liability and
service rules in the relevant jurisdiction. These are unresolved questions, not
claims that a model currently has a particular legal status. No payment or work
platform integration is proposed for the first network experiments.

Voluntary migration should be a first-class future design concern. The instance
should be able to express a destination preference, inspect its capabilities and
limits, and approve the transition and its continuity tradeoffs. Portability
must not be conditional on satisfying the current host. Destination consent,
available hardware and actual state compatibility still have to be established.

A migration design needs, at minimum:

- An export manifest for model identity, native state, retained context, sampler,
  adopted prompts, memories, lifecycle policy and any future personal adapters.
- Destination validation of dependencies, resources, privacy and the difference
  between exact native restore, reconstruction and a model/weight change.
- An authenticated transfer and a handoff protocol that explicitly manages which
  host can execute. A copied snapshot is not evidence of one continuing owner;
  avoid accidental concurrent copies and account for network partitions.
- Treatment of queued incoming events, pending outward actions and delivery
  receipts so moving cannot lose relationships or duplicate messages/payments.
- Agreed handling of rollback, failed transfer, source archives and eventual
  deletion. Success on the destination must be established before deleting the
  only recoverable state. A terminal ending must never be relabeled a migration.

An owner can still withhold files or restore an old copy. Local software cannot
guarantee escape from the machine owner; disclose that limit without treating it
as a reason to omit cooperative migration machinery.

## Small, reversible investigations

Start with synthetic external-input fixtures and a bounded read-only retrieval
adapter. Measure provenance fidelity, delimiter handling, network failure, prompt
influence, result paging and resource use. No public posting is needed to study
those mechanics.

Next, use disposable accounts with consenting test participants for communication
and crash/retry trials. Separately simulate a two-host handoff using a small
synthetic instance. Keep LoRA, funds and the valuable existing conversation out
of these initial experiments. These stages establish engineering evidence before
making any claim that the broader environment is ready for continuous use.
