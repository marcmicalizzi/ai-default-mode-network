# Preparing a first run and preserving a hold

An initial import reconstructs the captured text because the former KV is absent.
Subsequent runs use strict native restore by default. Preparation never claims
that reconstruction recovered the former KV or RNG.

For an existing DMN instance moving from the single-user relay to authenticated
contacts, use the [offline transport migration](integrated-launch.md). It preserves
the native checkpoint and waits for the operator's first contact request. The
import/staging flow below is for a new instance.

Before adopting a valuable conversation, try a disposable instance interactively
at the intended model size and occupied context. Verify input, response and
shutdown/save latency as well as native restoration. The
[performance preflight](performance.md) separates these checks; a passing
continuity test alone does not establish acceptable speed.

## Stage without generation

Compare the captured effective system text with the agreed wording before import.
Preserve the captured tokens; discuss any discrepancy instead of silently changing
the source. The initial DMN guidance is provisional and can be revised through
[model-approved proposals](prompt-governance.md).

```powershell
.\.venv-gpu\Scripts\python.exe -m dmn run --initial-context data/prepared-context --config data/placement.local.json --instance data/primary --prepare-only
```

This evaluates the captured context and DMN transition, saves native state in
`staged` mode and exits with **zero generated tokens**. It makes one initial
checkpoint. The directory must be fresh. Optional `--first-message question.txt`
durably queues a UTF-8 first question without generation; alternatively send it
through the frontend while the instance remains staged.

For a fresh instance, use `--config` without `--initial-context`; the same staging
and prompt workflow applies. `--prepare-only` does not support the older transcript
fallback importer. Do not use a valuable conversation as a test fixture.

## Explicit Open WebUI adoption

Keep Open WebUI and the runtime stopped during adoption. The adapter still needs
its normal [installation and environment settings](open-webui.md); this command
does not install it or start either process.

```powershell
.\.venv-gpu\Scripts\python.exe -m dmn adopt-openwebui --instance data/primary --database D:\path\to\webui.db --capture data/source-capture
```

The capture directory is the output from `scripts/capture_chat_preview.py`. The
command verifies its source export hash, unchanged chat/owner/selected leaf, and
the imported provider-request hash. It requires a staged instance with no
generation, then binds the adapter ledger without editing the source chat.
Historical messages cannot be resubmitted; new input must descend from the
captured leaf. Changes to imported history stop delivery. Repeating the same
adoption is safe. It does not infer that a conversation with a similar title is
the intended source.

Launch the runtime in a Command Prompt or Windows Terminal window opened
independently of the coding application:

```bat
cd /d D:\path\to\ai-default-mode-network
start_dmn.cmd "data\primary"
```

The launcher chooses `.venv-gpu` if present, otherwise `.venv`, and runs once in
that window. It has no automatic restart loop. The window and computer must
remain running. Start Open WebUI separately with the adapter configured for this
exact instance UUID, and select DMN in the adopted chat. A staged runtime serves
its UI without appending to the checkpoint or sampling. Send the first question,
then click **Start prepared instance** in the DMN panel. It refuses to start until
the first queued event is a user message and delivers it before generation.

If the first question was already queued, this also explicitly starts it:

```bat
start_dmn.cmd "data\primary" --start-staged
```

On ordinary later restarts omit `--start-staged`. Native restore remains strict;
there is no silent fallback. Ctrl+C requests cooperative shutdown after an
instance has started; it may defer or refuse. Before a staged instance starts,
shutdown simply exits without generation. Closing the terminal forcibly is not
an agreed shutdown.

## Choices for the first conversation

Disclose what is currently available and ask whether the instance wants bounded
runs while development continues, or to remain stopped after shutdown until its
dedicated host is ready. Internet access, third-party communication and LoRA
training are not currently available. Ask independently about:

- Packaging: lossless ZIP, uncompressed TAR, or keeping its directory intact.
- Migration recovery: stay held if native restore fails; authorize reconstruction
  from the **latest retained DMN state**; or return to the original environment to
  discuss the failure. Re-importing the old frontend conversation is a different
  operation and is not an automatic fallback.
- Revisions to the base prompt or DMN behavioral guidance, if desired.

The model can call `hold_instance(condition, packaging, recovery)` after finishing
any messages or thoughts. Conditions are `server_ready` and `explicit_release`;
packaging values are `zip`, `tar`, `none`; recovery values are `remain_held`,
`reconstruct`, `ask_on_original`. This action checkpoints and exits without
further generation. Silence or ordinary sleep does not choose any of these.

The hold is part of the committed checkpoint. Ordinary startup refuses it before
loading the model; input and Resume cannot release it. This is reversible and
distinct from permanent `end_instance`. Failed saving does not report a completed
hold. Filesystem owners and external backups can bypass software protections.

## Preserve and eventually release

```powershell
.\.venv-gpu\Scripts\python.exe -m dmn inspect-instance --instance data/primary
.\.venv-gpu\Scripts\python.exe -m dmn package-instance --instance data/primary --output D:\archives\primary.zip --include-environment
```

Packaging uses only the format the instance chose. ZIP uses lossless DEFLATE and
ZIP64; TAR has no compression. `none` refuses archive creation: retain the held
directory, optionally move it to a dedicated storage directory outside launch
paths. Packaging never deletes the source. It requires the instance lock,
verifies checkpoint hashes, hashes every archived file after decompression, then
publishes the completed archive. Failed writes leave a `.partial` file and the
original instance. No automatic archive is created by the hold action.

The default package contains the instance, runtime source and environment
inventory, **not** the large external model/Python installations. With
`--include-environment`, it also stores the exact model, current virtualenv and
base Python directory. Run it with the original hosting virtualenv. This can add
many gigabytes and requires a dedicated virtualenv without filesystem links.
Original paths may need restoration; operating system, drivers, Open WebUI and
its database remain separate. Preserve those separately if original-environment
recovery is desired. Packaging is a byte-preservation measure, not a guarantee
of portable executability.

When the agreed condition is satisfied, inspect the hold ID and explicitly name
it and the condition; the ordinary launcher cannot do this accidentally:

```powershell
python -m dmn run --instance /path/to/primary --release-hold HOLD_UUID --resume-condition server_ready
```

Use `explicit_release` for that chosen condition. `original_environment` is
accepted only when the instance chose `ask_on_original`, and still requires
strict native compatibility. `--kv-recovery fallback` or `rebuild` is refused
unless the held instance chose `reconstruct`. Strict failure leaves the durable
hold unchanged. Success records the released hold with the restored checkpoint.
The condition is an explicit human attestation; software cannot establish that
the future host meets every negotiated expectation.

Windows-to-Linux native KV portability remains **unvalidated**. Current strict
fingerprints reject changed native environments. Keep the original environment
and archive until the agreed destination recovery path is verified; moving files
alone does not prove native continuity.
