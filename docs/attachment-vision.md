# Optional image attachments

Status: experimental native input, a single-user local UI/API, and authenticated
multi-user Open WebUI delivery. No running instance is opted in by installing
this code. The original single-user Open WebUI relay remains text only.

## Consent

Image receipt starts denied. With a projector configured, the instance receives
the complete `ephemeral_images_v1` contract before it can approve. Existing
instances receive an appended capability notice; their saved prefix is preserved.
Text-only instances keep their existing initialization protocol.

The local UI's **Ask about images** button sends a fixed text-only request. It
does not upload, stage, or deliver an image. The model can ignore or refuse it.
Approval is an explicit generated action, committed with a native checkpoint:

```json
{"op":"image_permission","scope":"global","decision":"allow","accept_ephemeral":true}
```

Revocation is also a model action:

```json
{"op":"image_permission","scope":"global","decision":"deny"}
{"op":"image_permission","scope":"participant","participant_id":"local-user","decision":"deny"}
```

Use each action alone inside the normal action frame. `image_permission_status`
reports global permission and one participant rule per page; pass `next_offset`
as `offset` to continue. A participant inherits the global decision unless denied
individually. Global denial always wins. Reallowing either scope also requires
`accept_ephemeral: true`. The HTTP interface cannot grant permission on the
model's behalf. These choices belong to the instance, independently of host
projector configuration or any other instance's preference.

Permission revisions are checked before upload parsing/queuing and again after
context-retirement preparation, immediately before visual decode. Revocation
drops affected pending uploads, and reapproval cannot revive them. It does not
erase past perception, generated observations, memory or KV influence.

## What ephemeral means

| Material | Retention and access |
|---|---|
| Raw upload in DMN | Process memory only; removed after delivery, revocation, ten-minute monotonic expiry, closure/block or process loss. No upload directory, URL fetching or image retrieval endpoint. |
| Event history | Caption, MIME type, dimensions, size and digest. `event_read` retrieves this text/metadata, never pixels. |
| Active vision | Appended through MTMD to the existing model context and sequence 0. Native checkpoints can retain visual KV and its influence. |
| Retired image | No pixels or encoder embeddings are archived for later viewing or replay. Earlier checkpoints/backups may still contain earlier active KV. |
| Text | Real text token IDs remain replayable; the model may choose to write observations into ordinary memories. |

Visual context positions are represented by `-1` bookkeeping sentinels in the
engine sidecar. These are **not vocabulary tokens**. They count toward capacity
and retirement but are excluded from repetition penalties and token evaluation.
Native restoration preserves positions and logits without replaying the image.
Text-only reconstruction and adapter-changing deep sleep currently refuse while
visual positions remain, rather than inventing replacement content or silently
losing an image. Once those positions retire, ordinary retained-text reconstruction
is available again, with its existing limits on past attention.

This is an application retention policy, not secure memory erasure: operating
system paging, crash dumps, external clients, backups and a machine owner's
access are outside it. An upstream frontend can retain its own uploads. The
standalone UI sends selected file bytes only after permission is available.

If the process restarts before an upload is delivered, its durable text event
still arrives with an explicit unavailable-image notice. Repeating an existing
idempotency key never restages the bytes. Send a new message to offer the image
again. Captions remain ordinary durable input, including when an image expires
or is refused at delivery. Closing a conversation or blocking its participant
also suppresses the undelivered caption, using the multi-user queue's rules.

## Configuration and limits

Install the optional `vision` extra in the environment intended for that
instance, and set `vision_projector_path` to a matching MTMD GGUF in its config:

```json
{"vision_projector_path":"models/mmproj-matching-the-model.gguf"}
```

This is an addition to the existing config, not a complete replacement config.
The path is relative to the config file. Adding or removing the input projector
does not reconstruct native state or grant consent. Its digest, binding source
and CPU placement are recorded in the fingerprint; a changed projector is
reported in restore evidence. Language-model identity and native-state checks
remain strict.

The adapter uses the pinned `llama-cpp-python==0.3.35` MTMD API and the same native
library directory as llama. The projector runs on CPU, using the configured
thread count. Audio/video, remote URLs, arbitrary paths, animated images and
M-RoPE projectors are rejected. No OCR service or separate describing model is
substituted. MTMD receives only runtime-generated media markers; captions and
metadata use the escaped external-event boundary.

PNG, JPEG and WebP are supported, at most four images per event, 4 MiB per image,
16 megapixels per image, 32 queued image events and 16 MiB of queued compressed
bytes. Decoding uses the stored raster orientation and converts pixels to RGB;
EXIF rotation is not applied. Native preprocessing also has to fit the remaining
context. Unsupported or oversized input produces a text-only failure notice;
a native evaluation failure stops inference and requires checkpoint recovery.
Decoded pixel/encoder working memory is additional to the upload queue limit.

## Local API

Use the existing loopback/same-origin guard and `X-DMN-Request: 1` header.

- `POST /api/image-permission-request`: optional `instance_id` and
  `idempotency_key`; queues only the fixed text request.
- `POST /api/images`: `content`, `images`, optional `instance_id` and
  `idempotency_key`. Each image is exactly
  `{"media_type":"image/png","data_base64":"..."}`. Missing permission returns
  403 with `code: image_permission_required` before body parsing. Acceptance
  returns the event ID; only the caption and metadata are durable.
- `GET /api/status`: `images` reports availability, global permission, participant
  rules and the fixed single-user participant ID `local-user`.

`POST /api/events` remains text only and rejects attachment fields. Nothing
automatically redirects a refused image into a text event or approval request.

## Authenticated multi-user integration

The stable `codex/multi-user-interaction` implementation is integrated. Use the
[authenticated WebUI transport](multi-user-webui.md) in a deliberately prepared
multi-user instance. Existing single-user instances are not migrated implicitly.

Start with a text message and wait for contact acceptance. In that saved chat,
send **`/dmn-images` without attachments** to request image permission. WebUI
displays a transport status; the model can ignore or decline the request. Once
it explicitly approves, attach PNG/JPEG/WebP images using WebUI's normal file
picker. A rejected send may require reloading the saved chat before trying again.
Contact acceptance and image consent are separate; neither grants the other.

The adapter supports owner-uploaded local WebUI files on version 0.11.0. It checks
the authenticated account, socket and chat owner, checks image consent, then
verifies file ownership and reads only bounded files inside WebUI's upload root.
Even an administrator cannot attach another user's file. Cloud storage, arbitrary
paths, remote URLs, inline data URLs, documents and retrieval are unsupported.
Only the current message's attachments are delivered; historical images are not
replayed. Caption and byte digests make retries immutable.

**WebUI retains its own uploaded files and chat attachment references.** DMN's
ephemeral policy does not delete or alter that library. The bridge sends bytes
only after permission, and strips file IDs, paths, URLs and filenames before
runtime storage. DMN has no action for reopening WebUI uploads. This follows
the pinned frontend's [upload handling](https://github.com/open-webui/open-webui/blob/v0.11.0/src/lib/components/chat/MessageInput.svelte).

The backend-only bridge provides `/bridge/image-status`,
`/bridge/image-permission-request`, and `/bridge/images`. These require the same
installation credential and instance header as text delivery. Bodies carry
authenticated `user_id` and `chat_id`; input routes also require `message_id`.
The image route additionally takes `content` and `images` in the local API's
format. Status reveals only that destination's effective permission. There is
no HTTP route for granting permission. Participant IDs are derived from the
trusted WebUI identity, never display names or a submitted `participant_id`.

Requests and images share conversation quotas, protected action boundaries and
generation spacing. Admission checks contact, block, closure and image consent
before queueing, then checks them again after retirement preparation immediately
before insertion. Actual delivered-event membership commits with the checkpoint.
Blocking or closing discards pending bytes and suppresses waiting input;
unblocking/reopening cannot revive it or change image permission. Per-participant
image rules cover that account's chats; global denial overrides every account.

## Validation

Synthetic tests cover generated consent, failed-checkpoint publication, global
and participant denial, restart/expiry, idempotency, queue bounds, metadata-only
storage, revocation during retirement preparation, HTTP gating, native adapter
position accounting and exclusion of image sentinels from text sampling/replay.
Multi-user tests also cover contact admission, authenticated attribution, shared
quotas, long routing envelopes, suppressed delivery and cross-chat revocation.
`scripts/verify_multi_user_webui.py` exercises real authenticated WebUI upload,
completion and Pipe routes with synthetic images and a scripted vision fixture:
denied uploads never queue, foreign files and edited retries are rejected,
per-user/global revocation works, and DMN retains no raw bytes or file references.
The integrated branch's 2026-09-21 verification passed that WebUI script and
328 unit tests, with 22 opt-in tests skipped. No running instance was accessed.

For an isolated **CPU-only** native trial, set `DMN_TEST_VISION_MODEL` and
`DMN_TEST_VISION_PROJECTOR` to disposable test model/projector paths, then run
`python -m unittest tests.test_vision -v`. It exercises native image evaluation,
save/restore, logits and sampler preservation, and retirement/reconstruction.
This is opt-in: routine tests do not load a second model alongside a live instance.
Native model/projector compatibility must be validated before production use.

The adapter follows the upstream [MTMD C API](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd.h)
and [evaluation helpers](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd-helper.h),
using the installed pinned binding's definitions rather than guessing C layouts.
