"""Capture a Continue request in a private database copy, without inference.

Primary SQLite is opened read-only and backed up consistently. The copied app
has only a capture provider; no model is loaded and no original chat is edited.
Active custom filters and attachments are refused rather than silently omitted.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.capture import serve_capture
from dmn.backend import sha256_file
from dmn.storage import write_durable
from dmn.responses import responses_to_chat


def token_for(user_id, secret):
    encode = lambda obj: base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).rstrip(b"=")
    head, body = encode({"alg": "HS256", "typ": "JWT"}), encode({"id": user_id, "exp": int(time.time()) + 1800})
    signed = head + b"." + body
    return (signed + b"." + base64.urlsafe_b64encode(hmac.new(secret.encode(), signed, hashlib.sha256).digest()).rstrip(b"=")).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webui-python", type=Path, required=True)
    parser.add_argument("--webui-port", type=int, default=3034)
    parser.add_argument("--capture-port", type=int, default=9934)
    args = parser.parse_args()
    for port in (args.webui_port, args.capture_port):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                parser.error(f"port {port} is occupied")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "webui").mkdir()
    target = folder / "webui/webui.db"
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(target) as copy:
            source.backup(copy)
    with sqlite3.connect(target) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM chat WHERE id=?", (args.chat_id,)).fetchone()
        if not row:
            raise ValueError("selected chat is absent")
        chat = json.loads(row["chat"])
        if db.execute("SELECT id FROM function WHERE is_active=1").fetchone():
            raise ValueError("active custom functions need their own capture adapter; nothing launched")
        nodes = chat["history"]["messages"]
        if chat.get("files") or any(m.get("files") for m in nodes.values()):
            raise ValueError("attachments need a complete isolated asset snapshot; nothing launched")
        if len(chat.get("models", [])) != 1:
            raise ValueError("capture requires one selected model")
        model = chat["models"][0]
        leaf = row["current_message_id"] or chat["history"]["currentId"]
        assistant = nodes[leaf]
        if assistant["role"] != "assistant" or not assistant.get("done"):
            raise ValueError("Continue capture requires a completed assistant leaf")
        user_message = nodes[assistant["parentId"]]
        if user_message["role"] != "user":
            raise ValueError("unsupported leaf ancestry")
        user = db.execute("SELECT settings FROM user WHERE id=?", (row["user_id"],)).fetchone()
        settings = json.loads(user["settings"] or "{}")
        effective_params = {**settings.get("ui", {}).get("params", {}), **chat.get("params", {})}
        raw_export = {**dict(row), "chat": chat}
        write_durable(folder / "source-chat.json", raw_export)
        write_durable(folder / "source-params.json", effective_params)
        config = {r["key"]: json.loads(r["value"]) if isinstance(r["value"], str) else r["value"] for r in db.execute("SELECT key,value FROM config")}
        connections = config.get("openai.api_configs", {})
        candidates = [key for key, value in connections.items() if value.get("provider") == "llama.cpp" and value.get("enable", True)]
        if len(candidates) != 1:
            raise ValueError("capture requires one enabled llama.cpp provider")
        provider_index = candidates[0]
        source_connection = connections[provider_index]
        base_urls = config.get("openai.api_base_urls", [])
        overrides = {"openai.enable": True, "openai.api_base_urls": [f"http://127.0.0.1:{args.capture_port}/v1"] * len(base_urls),
            "openai.api_keys": [""] * len(base_urls), "openai.api_configs": {key: {**value, "enable": key == provider_index, "model_ids": [model] if key == provider_index else []} for key, value in connections.items()},
            "ollama.enable": False, "ollama.base_urls": []}
        for key, value in overrides.items():
            db.execute("UPDATE config SET value=? WHERE key=?", (json.dumps(value), key))
        # Disable scheduled work in the copy, preserving the tool definitions
        # available to this chat. No generation/action response is ever served.
        db.execute("UPDATE automation SET is_active=0")
        report = {"primary_database_opened_read_only": True, "source_chat_id": args.chat_id,
            "source_title": row["title"], "source_model": model, "source_messages": len(nodes),
            "source_leaf_id": leaf, "source_chat_sha256": sha256_file(folder / "source-chat.json"),
            "saved_summary_count": sum(bool(m.get("contextSummary") or m.get("context_summary")) for m in nodes.values()),
            "original_conversation_modified": False, "inference_performed": False,
            "capture_kind": "Continue from completed assistant, through installed Open WebUI middleware",
            "source_api_type": source_connection.get("api_type", "chat_completions"),
            "source_provider_index": provider_index,
            "configuration_overrides": list(overrides), "scheduled_jobs_in_copy_disabled": True}
    secret = "disposable-capture-" + uuid.uuid4().hex
    token = token_for(row["user_id"], secret)
    url = f"http://127.0.0.1:{args.webui_port}"
    def request(path, body=None):
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
        req = Request(url + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
            return json.load(response)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "DATA_DIR": str(folder / "webui"),
        "STATIC_DIR": str(folder / "static"), "FROM_INIT_PY": "True", "WEBUI_SECRET_KEY": secret,
        "DATABASE_URL": "sqlite:///" + target.as_posix(), "OFFLINE_MODE": "True",
        "ENABLE_VERSION_UPDATE_CHECK": "False", "RAG_EMBEDDING_ENGINE": "openai", "DO_NOT_TRACK": "True",
        "ANONYMIZED_TELEMETRY": "False", "SCARF_NO_ANALYTICS": "True"}
    (folder / "static").mkdir()
    capture = serve_capture(folder / "captures", model, args.capture_port)
    with (folder / "webui.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(args.webui_python), "-m", "uvicorn", "open_webui.main:app", "--host", "127.0.0.1",
            "--port", str(args.webui_port), "--workers", "1"], cwd=ROOT, env=env, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("copied Open WebUI exited during startup")
                try:
                    request("/ready")
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("copied Open WebUI startup timed out")
            models = request("/api/models")
            assert any(m["id"] == model for m in models["data"])
            body = {"model": model, "stream": True, "chat_id": args.chat_id,
                "session_id": "private-dmn-capture", "id": str(uuid.uuid4()), "assistant_message_id": leaf,
                "user_message": user_message, "parent_id": user_message.get("parentId"),
                "params": effective_params, "background_tasks": {}, "messages": []}
            if effective_params.get("system"):
                body["messages"] = [{"role": "system", "content": effective_params["system"]}]
            write_durable(folder / "frontend-request.json", body)
            request("/api/chat/completions", body)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                paths = list((folder / "captures").glob("*/capture.json"))
                if paths:
                    time.sleep(3)
                    break
                time.sleep(.5)
            else:
                raise TimeoutError("final provider request was not captured")
            paths = list((folder / "captures").glob("*/capture.json"))
            if len(paths) != 1:
                raise ValueError("multiple requests captured; inspect for compaction/tasks before selecting a context")
            provider_path = paths[0].parent / "provider-request.json"
            provider = json.loads(provider_path.read_text())
            chat_request = responses_to_chat(provider) if "input" in provider else provider
            if provider.get("model") != model or chat_request["messages"][-1]["role"] != "assistant":
                raise ValueError("capture is not the expected assistant continuation")
            first = chat_request["messages"][0]["content"]
            system_text = first if isinstance(first, str) else "".join(p.get("text", "") for p in first)
            if effective_params.get("system") and effective_params["system"] not in system_text:
                raise ValueError("effective system prompt was not preserved")
            report.update(provider_request=str(provider_path), provider_request_sha256=sha256_file(provider_path),
                provider_message_count=len(chat_request["messages"]), provider_fields=list(provider),
                source_sampling_temperature=effective_params.get("temperature"), captured_temperature=provider.get("temperature"),
                copied_app_version=request("/api/version")["version"], completed=True)
            write_durable(folder / "report.json", report)
            print(json.dumps(report, indent=2), flush=True)
        except BaseException as exc:
            write_durable(folder / "report.json", {**report, "completed": False, "error": repr(exc)})
            raise
        finally:
            # The venv launcher waits on a separate Python process on Windows.
            # The known child is located by its parent, never by process name.
            if process.poll() is None:
                stop_owned_tree(process.pid)
                process.wait(timeout=30)
            capture.shutdown()
            capture.server_close()


def stop_owned_tree(pid):
    if os.name != "nt":
        import signal
        os.kill(pid, signal.SIGTERM)
        return
    import ctypes as C
    from ctypes import wintypes as W
    class Entry(C.Structure):
        _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("pid", W.DWORD), ("heap", C.c_size_t),
            ("module", W.DWORD), ("threads", W.DWORD), ("parent", W.DWORD), ("priority", W.LONG),
            ("flags", W.DWORD), ("name", W.WCHAR * 260)]
    kernel = C.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = W.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [W.HANDLE, C.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    parents = {}
    try:
        entry = Entry();entry.size = C.sizeof(entry)
        ok = kernel.Process32FirstW(snapshot, C.byref(entry))
        while ok:
            parents.setdefault(entry.parent, []).append(entry.pid)
            ok = kernel.Process32NextW(snapshot, C.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    def stop(target):
        for child in parents.get(target, []):
            stop(child)
        try:
            import signal
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    stop(pid)


if __name__ == "__main__":
    main()
