# Open WebUI adapter

The adapter uses your installed **Open WebUI 0.11.0**, a Pipe function and an Event
function. No installed source files or frontend bundles are modified. DMN owns
one native context in its own process; other models retain their normal behavior.

## Disposable test

From the repository root:

```powershell
.venv/Scripts/python.exe scripts/openwebui_sandbox.py --webui-python C:/path/to/openwebui/.venv/Scripts/python.exe
# Add --reuse to restart the same disposable sandbox.
```

Or set `DMN_OPENWEBUI_PYTHON` to that Python executable and run
`start_openwebui_test.cmd`. Open **http://127.0.0.1:3031**, select **DMN**,
and send a message in a new saved chat. The first chat binds to the fixture;
additional chats cannot attach to the same instance. The fixture emits two
scripted messages separated by a pause, then sleeps. Input wakes it. These are
transport tests, not model cognition.

The launcher uses separate ports (3031 and 8766), database, static assets,
runtime files and configuration under `data/openwebui-test`. It never copies
or accesses the primary chat database, loads the 31B model, or changes startup
scripts in the primary installation. Authentication is disabled **only for this loopback,
disposable installation**; never point it at primary data. Ctrl+C requests DMN
shutdown and stops the frontend. The instance may accept, defer or refuse; the
scripted demo cannot decide that request. For native models, saving may take
minutes after acceptance. The launcher waits up to `--shutdown-timeout` (default 600 seconds);
if shutdown is not confirmed, it leaves DMN running and reports its PID/control
URL. It does not forcibly terminate a slow save or a storage-blocked runtime.

To run automated checks (creating a disposable conversation if needed):

```powershell
.venv/Scripts/python.exe scripts/verify_openwebui.py
```

The integration checks cover incoming retries, outgoing preservation, rejected
edits/regeneration, blocked Open WebUI compaction for DMN, offline delivery,
and recovery after destination commit but before delivery-cursor commit. They
also capture a synthetic compacted conversation through the actual Open WebUI
middleware. Results go to `data/openwebui-test/verification.json`. Temporary
changes to sandbox provider/compaction settings are restored afterwards.

Verified on 2026-09-19: **44 automated tests passed**, including native CPU
checkpoint tests. The Open WebUI integration checks passed, and browser checks
showed separate spontaneous messages and accepted new input. Restarting both
sandbox processes preserved the instance UUID, eight outgoing messages, the
delivery cursor and indefinite sleep. These checks use a scripted transport
fixture; subsequent CUDA and 31B checks are recorded separately in
[the Gemma validation report](gemma-validation.md).

## Message handling

The Pipe sends only new user text with a stable chat/message idempotency key.
It returns delivery status without invented assistant speech. DMN can respond
later or remain silent. The Event relay persists outgoing messages even when
no browser request is open. It commits both Open WebUI message representations
and ancestry in one transaction, then emits `chat:reload` and `chat:list`.

An empty completed transport placeholder can hold the first outgoing message;
subsequent outputs get separate stable IDs. Previously undelivered outputs are
replayed on reconnect, preserving their original timestamps. A durable ledger
and stable IDs prevent duplicate experiences and duplicate outgoing messages.
The exact runtime UUID is checked on input and output: another instance started
on the same port cannot silently inherit the conversation.

The Event function installs narrow, version-checked hooks in the running
process. They bypass Open WebUI compaction, memory injection, RAG and prompt
reconstruction for `dmn`; reject edits/regeneration and manual compaction; and
intercept retries before upstream placeholder writes can overwrite a delivered
message. Stale frontend history saves cannot replace the authoritative branch.
Title/metadata updates remain available. Disabling the Event restores the hooks.

Scope: plain text, one conversation owner, one Open WebUI worker and SQLite.
Attachments, tools, channels and temporary chats are rejected explicitly.
Frontend sampler/context settings do not change the runtime; configure it while
suspended. Use a separate conversation for ordinary completion models. Deleting
the display conversation does not stop the independent instance: delivery pauses
without advancing the cursor. Transferring its binding is a separate operation.

[Multiple people sharing one instance](multi-user-interaction.md) explores the
next step: authenticated provenance, explicit destinations, protected action
completion, fair inbox admission, presence feedback and model-controlled contact
closure/blocking. The [runtime prototype](multi-user-prototype.md) implements a
subset for local fixtures. This adapter still enforces one bound chat and refuses
experimental multi-user runtimes until authenticated routing is implemented.

## Beyond the sandbox

The `persistent-dmn` package must be importable in Open WebUI's Python environment.
Install `integrations/openwebui/dmn_pipe.py` as Function ID **dmn**, and
`dmn_relay.py` as **dmn_relay**. Configure `DMN_URL` (a loopback HTTP origin) and
`DMN_INSTANCE_ID` (the exact `/api/status` UUID) in that process's environment.
Enable both functions and run one worker. Open WebUI's lifecycle events restart
the relay using its durable ledger.

Back up the primary database and launcher before installing the functions or
changing the process environment. Explicit adoption preserves the captured chat;
select DMN in its model selector before sending new input. No fork is currently
needed, but the internal hooks require revalidation before upgrading. The adapter
refuses unverified Open WebUI versions.

## Capture the initial effective request

```powershell
.venv/Scripts/python.exe -m dmn capture-provider --model capture-fixture --output data/provider-captures
```

Configure **a disposable Open WebUI copy** with this OpenAI-compatible connection:
`http://127.0.0.1:9932/v1`. It advertises the supplied model ID, records each final
provider request byte-for-byte in a UUID directory, then returns an explicit
capture-complete error. It generates no reply and forwards no traffic. HTTP
credentials are not stored. When capturing the copied primary conversation,
use its original model ID/settings and saved compaction configuration.

The body contains Open WebUI's effective system prompt, saved summary, retained
messages and injected context. Both Chat Completions and Responses are captured.
`prepare-migration --provider-request PATH` preserves the body as evidence;
`prepare-initial-context` additionally renders it through the pinned native
server, captures tokens and resolves supported sampler defaults. See
[the importer workflow](migration.md#effective-initial-context-reconstruction).
`scripts/capture_chat_preview.py` automates the read-only source backup and
Continue capture in a private copy. Capturing/preparing a conversation does not
create a DMN instance or change its frontend binding.

Existing llama-server entries can continue serving normal conversations. DMN
currently loads llama.cpp through native bindings in its own process; loading
the large model in both processes may duplicate RAM/VRAM. For an unloaded
original model, initial import reconstructs context; it does not transfer a
live slot.

## Adopting a prepared existing conversation

The [staged first-run workflow](first-run.md) now provides `adopt-openwebui` for
explicitly binding a captured source chat after import-only preparation. It
checks that the source remains unchanged and prevents historical input replay.
The ordinary adapter still refuses to implicitly adopt an existing conversation.
Start the runtime and Open WebUI in independent processes; the disposable sandbox
supervisor is a test tool, not the production launcher.

System-prompt edits in Open WebUI do not change DMN's live agreement. Use the DMN
panel's Behavioral agreement and proposals section; only the model can adopt it.
