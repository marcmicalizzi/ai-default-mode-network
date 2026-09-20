from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

from .backend import sha256_file
from .storage import write_durable


def selected_branch(export, chat_id=None, leaf_id=None):
    chats = export if isinstance(export, list) else [export]
    if chat_id:
        chats = [c for c in chats if str(c.get("id")) == chat_id]
    if len(chats) != 1:
        raise ValueError("select exactly one conversation with --chat-id")
    chat = chats[0].get("chat", chats[0])
    history = chat.get("history", {})
    messages = history.get("messages", {})
    if messages:
        current = leaf_id or chats[0].get("current_message_id") or history.get("currentId")
        if current is None:
            raise ValueError("history has no selected branch; supply --leaf-id")
        result, visited = [], set()
        while current is not None:
            if current in visited or current not in messages:
                raise ValueError("broken or cyclic conversation history")
            visited.add(current)
            message = messages[current]
            result.append(message)
            current = message.get("parentId")
        return list(reversed(result))
    if leaf_id:
        raise ValueError("leaf-id requires branched history")
    result = chat.get("messages")
    if not isinstance(result, list):
        raise ValueError("unrecognized Open WebUI export")
    return result


def active_context_projection(branch, compaction_enabled=True):
    """Mirror v0.11's saved-summary *selection*, without running a summarizer.

    Raw message objects stay intact. This is not the finalized provider request:
    Open WebUI still normalizes output items, tools, images, system variables and
    injected context. Capture that request to reproduce its actual prompt.
    """
    messages = deepcopy(branch)
    system = messages[:1] if messages and messages[0].get("role") == "system" else []
    conversation = messages[1:] if system else messages
    boundary, summary = 0, None
    if compaction_enabled:
        for i, message in enumerate(conversation):
            value = message.get("contextSummary") or message.get("context_summary")
            if isinstance(value, str) and value.strip():
                boundary, summary = i, value
    retained = conversation[boundary:]
    return {"schema": 1, "source_behavior": "Open WebUI 0.11.0 saved-summary selection",
            "compaction_enabled": compaction_enabled, "summary": summary,
            "checkpoint_message_id": retained[0].get("id") if summary is not None else None,
            "system_messages": system, "retained_messages": retained,
            "archived_message_ids": [m.get("id") for m in conversation[:boundary]],
            "retained_message_ids": [m.get("id") for m in retained],
            "provider_summary_prefix": "[CONVERSATION SUMMARY]\n" if summary is not None else None,
            "is_final_provider_request": False,
            "limitation": "Final system prompt, output normalization, RAG, tools and template rendering must also be captured."}


def prepare_bundle(export_path: Path, output: Path, metadata_path=None, slot_path=None, chat_id=None, leaf_id=None,
                   request_path=None, compaction_enabled=True):
    if output.exists():
        raise ValueError("migration output must be a new directory")
    raw = json.loads(export_path.read_text(encoding="utf-8"))
    selected = selected_branch(raw, chat_id, leaf_id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else {}
    if not isinstance(metadata, dict):
        raise ValueError("environment metadata must be a JSON object")
    if request_path:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or not (isinstance(request.get("messages"), list) or isinstance(request.get("input"), (str, list))):
            raise ValueError("captured provider request must contain messages or Responses input")
    output.mkdir(parents=True)
    shutil.copyfile(export_path, output / "openwebui-original.json")
    write_durable(output / "selected-branch.json", selected)
    write_durable(output / "active-context.json", active_context_projection(selected, compaction_enabled))
    write_durable(output / "environment.json", metadata)
    if request_path:
        shutil.copyfile(request_path, output / "provider-request.json")
    if slot_path:
        shutil.copyfile(slot_path, output / "llama-server-slot.bin")
    expected = ["model_sha256", "quantization", "chat_template", "system_prompt", "sampler",
                "injected_context", "llama_cpp_build", "server_launch_arguments"]
    manifest = {"schema": 1, "continuity": "transcript_archive_not_native_runtime_checkpoint",
                "chat_id": chat_id, "leaf_id": leaf_id,
                "missing_environment_evidence": [key for key in expected if key not in metadata],
                "slot_state_preserved": bool(slot_path), "slot_state_import_supported": False,
                "provider_request_preserved": bool(request_path),
                "compaction_projection_enabled": compaction_enabled,
                "files": {p.name: sha256_file(p) for p in output.iterdir()}}
    write_durable(output / "manifest.json", manifest)
    return manifest


def import_transcript(runtime, bundle: Path):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("continuity") != "transcript_archive_not_native_runtime_checkpoint":
        raise ValueError("not a transcript migration bundle")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256_file(bundle / name) != digest:
            raise ValueError("migration archive integrity check failed")
    destination = runtime.root / "import"
    if destination.exists() or runtime.state["generated_tokens"] or runtime.store.next_event(0):
        raise ValueError("transcript import requires a fresh instance")
    shutil.copytree(bundle, destination)
    branch = json.loads((destination / "selected-branch.json").read_text())
    runtime.state["continuity"] = "transcript_reconstruction"
    runtime.state["import_manifest"] = manifest
    notice = {
        "fact": "The following conversation is imported text and metadata. Its original KV and sampler state were not restored. This is a fresh computational context.",
        "environment": json.loads((destination / "environment.json").read_text()),
    }
    events = [{"kind": "migration_notice", "payload": notice, "created": runtime.now()}]
    for item in branch:
        events.append({"kind": "imported_transcript", "payload": {"original_message": item}, "created": runtime.now()})
    runtime.checkpoint(events=events)
