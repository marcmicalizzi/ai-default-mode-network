# Opt-in web retrieval

Status: implemented experimental HTTPS retrieval with synthetic transport and
runtime tests. This is static document reading, without JavaScript, accounts or
a search provider. Open WebUI remains the message transport; its RAG/tool loop
is not enabled for DMN. The [original investigation](web-browsing-proposal.md)
records the architectural comparison and broader acceptance goals.

## Offer access for one launch

The [starter collection](web-browsing-starter.md) provides a checked host allowlist
covering reference, research, literature, reporting and lightweight search. It uses
public URL selection constrained to those hosts. The smaller example below uses
seed/link handles instead.

Copy [the example policy](../examples/web-browsing.json), choose seed URLs and
permitted hostnames, and pass it explicitly on each launch:

```powershell
.\.venv\Scripts\python.exe -m dmn run --instance data/disposable --config examples/your-model.local.json --web-policy examples/web-browsing.json
```

The example is a web policy, not a model configuration. Omit `--web-policy` to
disable network access. It is deliberately absent from saved model configuration
and native fingerprints, so loading or copying an instance does not supply network
authority. A supervised sleep cycle retains the current launch's explicit offer.
Existing KV and behavioral agreements are preserved; a protected capability notice
explains the offered operations and limitations. A staged or first-contact-gated
instance does not receive this notice or start a worker until it is active.

`handles` mode permits exact seed and extracted-link handles within `allowed_hosts`.
The model cannot edit their URL fields. `web_seeds` exposes the initial handles;
`web_read` with `section: "links"` exposes extracted handles. Host matching is exact,
without automatic subdomain grants. Cache eviction can invalidate a link handle.

To explicitly offer arbitrary public HTTPS URLs, use a separate policy containing
`{"mode":"public"}`. Add `allowed_hosts` to constrain destinations. In public
mode, model-generated hostnames, paths and queries can disclose private context.
Handles limit arbitrary URL construction but cannot eliminate signaling through
link choices or timing. Neither mode certifies resistance to semantic injection.
First evaluate the capability on a disposable instance.

## Operations and external provenance

Every operation is a single generated `dmn_action`, using the existing action
syntax. Text received in a page or conversation never executes directly.

| Operation | Result |
|---|---|
| `web_seeds(offset=0, limit=200)` | Paged JSON string containing seed URLs and handles |
| `web_fetch(handle)` | Queue one document, reuse an existing request/cache entry, or return a refusal |
| `web_fetch(url)` | Same, only in public mode |
| `web_status(request_id)` | Current state; no additional request |
| `web_read(document_id, section="text", offset=0, limit=200)` | A locally retained page; sections are `text`, `source`, `links` |
| `web_cancel(request_id)` | Commit a cancellation; active network effects may already have occurred |
| `web_limits(offset=0, limit=200)` | Paged JSON string with policy, charged usage and cooldown |

Follow the returned `next_offset`, since available context can shorten a page.
When `next_offset` reaches `total_characters`, the section is complete; under a
small budget that total may be omitted, and an empty page marks the end. Reads
never refresh or refetch. Source metadata contains the full requested/final URL,
retrieval time, extraction version, content hash, HTTP status and truncation flag.
The `document_id`/`source_ref` identifies that source on every text page, including
when the full URL would not fit. The hash identifies bytes; it does not certify
accuracy or authorship.

Completion arrives as a durable `web_result` event. Results, errors, titles, links
and every page carry `source_kind: web` and `trust: untrusted_external`. Page text
is nested under `external` and serialized through the existing escaped event
encoder. A webpage cannot overwrite provenance fields or introduce role tokens.
If the minimal envelope cannot fit, the runtime returns a labeled refusal or
pauses for context; it never substitutes an unlabeled preview.

Static extraction removes script/style/template/iframe/object/noscript contents,
decodes HTML entities before serialization, and uses UTF-8 with replacement for
invalid bytes. Other encodings may lose characters. It retrieves no images, CSS,
favicons or other subresources. Visible malicious instructions remain external
text. Reading does not automatically write a memory, approve a prompt, select
training examples or invoke another action. A persuaded model can still generate
such actions: these structural tests do not establish behavioral immunity.

## Enforced limits and recovery

Defaults can be reduced or adjusted within the validated policy ranges:

| Control | Default |
|---|---|
| Outstanding document requests | 4 total, one active network request |
| Request starts | At least 10 seconds apart; 6/minute, 60/hour, 250/day |
| Host pacing | Same minimum interval, plus robots delay and server backoff |
| Duplicate suppression | Coalesced while pending; cached success for 15 minutes, failure for 60 seconds |
| Loop breaker | Third repeated attempt within the rolling 60-second activity window closes network admission for 5 minutes |
| Automatic retries | None |
| Response body | 2 MiB, streamed; unexpected compression rejected |
| Response headers/framing lines | 64 KiB aggregate |
| Extracted text/links | 100,000 characters; at most 128 links |
| Time | 5-second socket timeout; supervised 20-second total including DNS and extraction |
| Response bytes per day | 32 MiB; reserve before dispatch and settle known consumption |
| Retained document cache | 64 MiB serialized data or 128 documents, whichever is reached first |
| Undelivered result events | Admission stops at 16 |

Budgets count actual HTTP requests, including the separate robots fetch. There
are no probes, redirect following, retry loops or ambient proxy/cookie/netrc
credentials. Only HTTPS port 443 is accepted. Every DNS answer must be public;
the transport connects directly to one checked address and verifies TLS against
the original hostname. Mixed private/public answers and IPv6 transition forms
are rejected. It performs no second unchecked hostname lookup.

Requests reserve their body allowance durably before I/O. Known consumption
settles that reservation; cancellation, timeout or parser uncertainty can retain
the full charge. This is conservative application-body accounting, not measurement
of DNS/TLS/TCP overhead or an OS bandwidth quota. Exact-limit responses are labeled
truncated even if they might have ended there. A byte budget smaller than the
remaining full response reservation refuses a request before sending it.

Limit windows use a persisted monotonic elapsed-time counter. Stopped time is
not credited, and wall-clock changes do not refill budgets. Consequently an
instance restarted after a day offline can still need active elapsed time before
an exhausted daily allowance refills. Valid `Retry-After` dates/durations become
persisted minimum waits and are never capped downward.

An intent commits with the native checkpoint. A failed checkpoint cannot dispatch
it. The worker durably claims each actual request before contacting the network.
A crash during a claim yields `outcome_unknown` on recovery and no automatic
resend. Completed results and the inbox event commit together. The ledger and a
persistent generated-token high-water mark prevent older restored action positions
from dispatching again, even after request-history pruning. A newly generated
request at a later position remains subject to the normal cooldowns and budgets.

Suspension closes admission and cancels active I/O; queued work is cancelled.
Ordinary sleep can receive a completion event and wake. Operator suspension,
holds and deep sleep cannot be resumed by a fetch. Shutdown stops the worker
before closing the store; ending/erasure closes it before handling managed data.
In-flight effects cannot be undone and are reported conservatively.

The request ledger, extracted documents and handles reside in `runtime.sqlite3`,
so ordinary instance archival and managed erasure cover them. Cache eviction is
logical deletion, not secure erasure or a SQLite file-size quota. Existing inbox
and diagnostic history retain their existing retention semantics. The limits
apply to one instance; multiple independent instances do not share an aggregate
host budget in this version. The helper process is cancellable trusted code,
not an OS sandbox against a compromised Python interpreter.

## Robots handling and validation scope

Robots responses must be bounded plain text. A 404/410 means no robots rules;
other unsuccessful responses, redirects or truncated robots data fail closed.
Rules are cached for at most 24 hours of active elapsed time. Matching disallow
rules win over allow rules, and unsupported wildcard/end-anchor disallows refuse
access conservatively. Crawl-delay and request-rate can slow requests further.
This subset can reject pages that a full RFC 9309 implementation would permit.

Tests use synthetic pages and an injected transport for deterministic budgets,
failure, restart and cancellation. Spawned-process fixtures verify hard timeout
cleanup without contacting a server; socket/HTTP fixtures verify address binding,
headers and streaming limits. Runtime fixtures verify checkpoint-before-dispatch,
protected capability notices and injection-shaped text through small pages.
No valuable instance or real-model behavioral resistance is certified by these
tests. Search providers, full browser rendering, shared host budgets and broader
model adversarial trials remain future work.

A bounded public transport smoke test on 2026-09-27 fetched `https://example.com/`
over verified TLS: HTTP 200, 559 response-body bytes, 128 extracted characters,
without truncation. This tests the actual transport, not a live model's behavior.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_web -v
```
