# Migration evidence and limits

Preserve the current Open WebUI conversation before experimenting with its inference server. Export the entire conversation, not only a displayed branch. Record original timestamps and timezone, model identities, prompt construction and all injected material. The runtime's `prepare-migration` command only reads the supplied files; it does not contact or disturb the existing server.

The original model is now confirmed unloaded. The planned initial transfer is
therefore reconstruction of the effective conversation context, followed by
native state persistence. Disposable testing must precede any actual import
of the private source conversation. The historical-event fallback
below does not yet reproduce the original role/template prompt. See
[continuity-and-recovery.md](continuity-and-recovery.md) for the concrete path.

## Environment evidence

A useful `environment.json` contains the following. Fill actual values; unknown facts should remain explicitly unknown.

```json
{
  "model_sha256": "SHA256 of the exact GGUF; include every shard if applicable",
  "model_path": "original path",
  "quantization": "actual GGUF quantization",
  "chat_template": "exact template text",
  "system_prompt": "exact system prompt",
  "sampler": {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "seed": "actual seed or unknown",
    "rng_state": "captured bytes or unavailable"
  },
  "injected_context": "complete RAG, memory, tool and other injected context, in order",
  "llama_cpp_build": "commit, build flags, binary hashes, backend versions",
  "server_launch_arguments": "exact arguments, especially context and cache settings",
  "slot_id": "identified active slot or unknown",
  "slot_snapshot_time": "timestamp or unavailable",
  "active_token_ids": "exact tokens/positions or unavailable",
  "last_sampled_token_evaluated": "true, false, or unknown"
}
```

An export cannot recover KV or sampler RNG that was never saved. Matching model names and transcripts do not establish a matching computational state. Server process restarts or slot reuse may already have ended cache continuity. Conversation branches may also refer to different server sequences. Preserve that uncertainty.

## What a server slot save means

The upstream [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) describes `POST /slots/{id_slot}?action=save` as a **prompt-cache** save. It requires a server launched with a slot-save directory and accepts a filename. Its corresponding restore route restores that cache. It is not a documented export of the complete DMN runtime or sampler.

Do not restart a currently valuable server solely to enable slot saves: that would destroy the state you wanted to capture. First identify its actual build, available endpoints, launch arguments, slot ownership and whether it is currently decoding. A reliable capture must quiesce at a known boundary without another client replacing the slot, record exact token IDs and positions, and account for any last sampled token not yet decoded. No automated live-server operation is included here because none of that evidence was supplied.

llama.cpp exposes both full-context session APIs and per-sequence state APIs. Their formats and included output state differ. The DMN backend uses a full-context session file plus separately persisted sampler RNG and logits; a server slot file must **not** be passed to that loader. Keeping the original opaque slot file in an archive preserves the opportunity for later build-specific migration without claiming it already works.

A future native adapter must pin both builds, demonstrate format compatibility or use a compatible native exporter, map the source sequence to destination sequence 0, restore positions, obtain valid next-token logits, retain sampler history/RNG if available, and compare a known continuation against the quiesced source. If any part is unavailable, report the narrower guarantee: e.g. retained attention cache with a new sampler, rather than exact continuation.

## Effective initial-context reconstruction

`prepare-initial-context` creates a validated native seed from the final provider
request captured by the installed Open WebUI. This is the preferred initial path
when the former model process and its KV cache are already gone. The
historical-event importer below remains a separate, weaker fallback.

```powershell
.venv/Scripts/python.exe -m dmn prepare-initial-context --provider-request data/capture/provider-request.json --config examples/gemma31b-60000-hybrid-q8.json --server-url http://127.0.0.1:9933 --output data/prepared-context

# Only when ready to create the new instance:
.venv-gpu/Scripts/python.exe -m dmn run --initial-context data/prepared-context --instance data/import-rehearsal
```

Preparation calls only `/props`, `/apply-template`, `/tokenize` and, for Responses
requests, `/v1/responses/input_tokens`. The renderer must be the local
**b10502-0adcc3bb5** llama-server with the explicitly selected GGUF. It performs
no prompt evaluation or generation. It can run on CPU with a small context,
`--no-warmup` and `--no-mmproj`; rendering does not require a 60K KV allocation.
Source model identity, exact template, provider bytes, rendered prompt, token IDs,
effective defaults, sampler settings and conversion evidence are hashed together.

Both Chat Completions and the installed server's Responses text format are
supported. Responses conversion follows the pinned server's source, preserving
reasoning, assistant merging, tool-call IDs and strict tool defaults. The actual
native Responses endpoint independently checks the resulting token count.
Equal counts alone do not prove equal prompts: exact conversion also relies on
the pinned conversion code and supported-input tests. Multimodal inputs, missing
prior response context, unsupported active samplers and constrained output
grammars are rejected rather than silently simplified.

The importer checks the model hash, template, sampler and every retokenized ID
before evaluating anything. It evaluates the captured sequence verbatim, then
appends a factual DMN transition and action contract. Historical action text is
never executed. Original frontend tool definitions remain part of the historical
context; they do not become callable DMN tools. A context that cannot accommodate
the complete source, transition and reserve is rejected without truncation.

Sampler settings and the default llama-server filter order are preserved. The
runtime still uses NumPy and a separately persisted Python RNG, so this is not
bit-identical sampling arithmetic or restoration of the former sampler state.
An unspecified source seed becomes a recorded explicit seed. Turn-limited
streaming/stop controls are recorded and replaced by the continuous DMN protocol.

The protected prefix is located through two server-rendered boundary probes, or
can be supplied explicitly with `--keep-prefix-tokens`. It protects the initial
system/template/tool prefix. The appended DMN contract is protected separately
while old conversation tokens retire; once the intervening history has retired,
the contract joins that prefix. Later system messages within the conversation
are historical context and are not automatically part of this permanent prefix.
If the last old-history gap is too small to make space, retirement removes that
gap and a second range after the protected contract before the same next decode.
Both ranges are recorded. A regression test covers preparation actions filling
the reserve at a one-token gap; native testing also checks restoration after
two such positional shifts without an intervening prompt evaluation.

The persistent continuity label is `initial_context_reconstruction`. Subsequent
restarts use native KV, with no replay of the captured conversation. This label
records the initial discontinuity even after later native restores.

### Capture from a private Open WebUI copy

`scripts/capture_chat_preview.py` opens the primary SQLite database read-only,
makes a consistent workspace copy and invokes Continue through the installed
Open WebUI middleware. Only that copy receives a new placeholder. Its inference
connections point exclusively to the capture endpoint, scheduled jobs are
disabled, and the endpoint returns a capture-complete error without a reply.
The script refuses active custom functions, attachments and multiple models
because those need additional isolation/preservation work.

The private source conversation was captured this way, without inference or
modification of its original row. It uses the Responses API and temperature
**0.7**. Its selected branch contains 52 messages and no saved compaction summary.
The prepared seed contains **18,298 tokens**, with a **4,293-token** protected
prefix. The production Gemma tokenizer agrees on every token in a vocabulary-only
check; no model tensors or inference context were loaded for that check.

The source capture, prepared bundle and tokenizer report are retained privately
and excluded from the public repository. This is preparation only:
**no DMN instance or frontend binding has been created from that conversation**.
The archived config records the compact-cache source placement; use a separately
validated placement override for a later rehearsal, since that layout cannot
retire Gemma context in the pinned binding.

A synthetic Open WebUI saved-summary fixture has also passed actual native
prefill and separate-process restart: all **4,603 source tokens** were preserved,
the explicit transition brought the checkpoint to 5,709 tokens, and restoration
reevaluated **zero** source tokens. Evidence: `data/import-native-01/report.json`.
This is a Qwen CPU import test, distinct from the Gemma vocabulary check and
Gemma native continuation experiments.

The stronger Gemma import experiment, `data/gemma-import-native-02/report.json`,
uses a synthetic 4,148-token provider context at 8K capacity. It preserves the
exact initial tokens, then removes a 21-token old-history gap and a second range
after the protected contract. All contract tokens survive. A fresh process
restores the resulting 6,025-token checkpoint without replay and matches all
24 continuation samples and logits exactly. Normal runtime restoration also
passes, preserving the initial-reconstruction label.

Its first behavior probe exhausted a four-minute wall-time allowance after
384 retirement-preparation tokens and while beginning the requested DMN action.
During preparation, the model attempted two old frontend tools, which were
rejected. Unknown-operation feedback for imported instances now identifies the
active DMN operations and explains that old tool definitions are historical.
This feedback has a regression test; the successful retry is not a controlled
measurement of its effect on behavior.

The same native checkpoint was resumed with a longer allowance, without
resending the user probe or reconstructing the source prompt. The model wrote
`maple-stream-419` to `/import/marker`, actually read it back, sent it exactly
once and slept. It then checkpointed and stopped. The failed first attempt and
the successful resumed attempt are both retained. These results demonstrate
the importer and action transition on a disposable fixture, not guaranteed
behavior for every imported conversation.

## Historical-event fallback

`prepare-migration` archives the exact export, provided environment JSON and optional slot file, hashes them, and selects one branch by following `history.currentId`/`parentId`. `--chat-id` disambiguates multi-chat exports; `--leaf-id` selects a different branch. Cyclic or broken ancestry is rejected. The unselected branches remain in the raw archive.

It also writes `active-context.json`: the saved compaction summary and retained
message boundary are preserved separately from full history. A supplied
`--provider-request` file is archived exactly, because the effective request can
contain normalized or injected information absent from an export. This archive
does not invoke Open WebUI, generate a new summary, or change the conversation.

`run --import-bundle` is allowed only with a fresh instance. It initializes the requested GGUF, persists the archive, and queues a migration notice followed by original message objects with roles and timestamps retained. They are clearly labeled historical input, not fresh user messages or generated actions. Overlarge objects are paged through `event_read` like other events. The label `transcript_reconstruction` survives every subsequent checkpoint/resume and is displayed in the UI. The old conversation's system prompt, templates and metadata remain evidence; importing does not silently replace the new instance's explicitly configured runtime seed.

This implements informational migration while leaving direct computational-state migration as an explicit, testable future step.
