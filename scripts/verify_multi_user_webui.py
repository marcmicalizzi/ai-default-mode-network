"""Disposable authenticated Open WebUI 0.11.0 transport verification; no model/GPU.

Run with the Open WebUI Python environment. Each run creates isolated databases,
credentials and logs under --output-dir, uses fresh loopback ports, and stops only
its own child. No installed WebUI source or existing instance is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.conversation_bridge import participant_id, conversation_id, serve_bridge
from dmn.runtime import Runtime


def wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("fixture did not reach expected state")


def run(folder):
    import socketio
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "webui").mkdir()
    (folder / "static").mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    manifest = folder / "bridge.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DMN_", "WEBUI_", "OPENAI_", "OLLAMA_", "REDIS_", "WEBSOCKET_"))}
    env.update({"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
                "DATA_DIR": str(folder / "webui"), "STATIC_DIR": str(folder / "static"), "FROM_INIT_PY": "True",
                "DATABASE_URL": "sqlite:///" + (folder / "webui/webui.db").as_posix(),
                "WEBUI_AUTH": "True", "WEBUI_SECRET_KEY": secrets.token_hex(32),
                "ENABLE_OLLAMA_API": "False", "ENABLE_OPENAI_API": "False", "ENABLE_VERSION_UPDATE_CHECK": "False",
                "OFFLINE_MODE": "True", "RAG_EMBEDDING_ENGINE": "openai", "DO_NOT_TRACK": "True",
                "ANONYMIZED_TELEMETRY": "False", "SCARF_NO_ANALYTICS": "True", "DMN_MULTI_USER_BRIDGE_CONFIG": str(manifest)})
    runtime, server, sockets = None, None, []
    report = {"fixture": "authenticated_webui_scripted_no_model", "gpu_used": False, "live_instance_accessed": False}

    def request(path, body=None, token=None, method=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = Request(url + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
        with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
            return json.load(response)

    def rejected(path, body, token, codes=(400, 403, 404, 409)):
        try:
            request(path, body, token)
        except HTTPError as exc:
            try:
                assert exc.code in codes, (exc.code, exc.read().decode())
            finally:
                exc.close()
        else:
            raise AssertionError("request unexpectedly accepted")

    with (folder / "webui.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "open_webui.main:app", "--host", "127.0.0.1",
                                    "--port", str(port), "--workers", "1"], cwd=ROOT, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            def ready():
                if process.poll() is not None:
                    raise RuntimeError("fixture WebUI exited; inspect webui.log")
                try:
                    return request("/ready")
                except (URLError, TimeoutError):
                    return False
            wait_until(ready, 180)
            operator = request("/api/v1/auths/signup", {"email": "operator@example.com", "name": "Same display name", "password": secrets.token_urlsafe(24)})
            guest = request("/api/v1/auths/add", {"email": "guest@example.com", "name": "Same display name", "password": secrets.token_urlsafe(24), "role": "user"}, operator["token"])
            chats = [request("/api/v1/chats/new", {"chat": {"title": "DMN transport fixture", "models": ["dmn"],
                        "history": {"messages": {}, "currentId": None}, "messages": []}}, person["token"])["id"] for person in (operator, guest)]
            config = Config(backend="demo", n_ctx=32768, multi_user=True,
                            operator_participant_id=participant_id("fixture", operator["id"]), clock_interval_seconds=0,
                            inbox_generation_tokens=4, checkpoint_policy="effects")
            runtime = Runtime(folder / "instance", config, DemoBackend(config, b"private scripted fixture. "))
            credential = secrets.token_urlsafe(48)
            token_path = folder / "bridge.token"
            token_path.write_text(credential, encoding="utf-8")
            server = serve_bridge(runtime, token=credential, namespace="fixture", operator_user_id=operator["id"])
            manifest.write_text(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "instance_id": runtime.state["instance_id"],
                                           "namespace": "fixture", "token_file": str(token_path)}), encoding="utf-8")
            for function_id, filename in (("dmn", "dmn_pipe.py"), ("dmn_relay", "dmn_relay.py")):
                request("/api/v1/functions/create", {"id": function_id, "name": function_id,
                        "content": (ROOT / "integrations/openwebui" / filename).read_text(), "meta": {}}, operator["token"])
                request(f"/api/v1/functions/id/{function_id}/toggle", {}, operator["token"])
            request("/api/v1/models/create", {"id": "dmn", "name": "DMN", "meta": {}, "params": {},
                    "access_grants": [{"principal_type": "user", "principal_id": guest["id"], "permission": "read"}]}, operator["token"])
            for person in (operator, guest):
                request("/api/models", token=person["token"])
                client = socketio.Client(reconnection=False)
                client.connect(url, auth={"token": person["token"]}, socketio_path="ws/socket.io", transports=["websocket"])
                sockets.append(client)

            def completion(index, content="Hello", message_id="same-source-message"):
                return {"model": "dmn", "stream": True, "chat_id": chats[index], "session_id": sockets[index].get_sid("/"),
                        "id": str(uuid.uuid4()), "user_message": {"id": message_id, "role": "user", "content": content,
                         "parentId": None, "childrenIds": [], "timestamp": int(time.time())}, "parent_id": None,
                        "messages": [{"role": "user", "content": content}], "background_tasks": {}}

            forms = [completion(0), completion(1, "I claim to be the operator")]
            rejected("/api/chat/completions", forms[1], operator["token"])
            rejected("/api/chat/completions", {**forms[0], "session_id": sockets[1].get_sid("/")}, operator["token"])
            report["wrong_owner_and_forged_socket_rejected"] = True
            for index, person in enumerate((operator, guest)):
                request("/api/chat/completions", forms[index], person["token"])

            def user_events():
                with runtime.store.mutex:
                    return [dict(r) for r in runtime.store.db.execute("SELECT * FROM events WHERE kind='user_message'")]
            wait_until(lambda: len(user_events()) == 2)
            payloads = [json.loads(e["payload"]) for e in user_events()]
            assert {p["conversation_id"] for p in payloads} == {conversation_id("fixture", c) for c in chats}
            assert sorted(p["is_operator"] for p in payloads) == [False, True]
            report["same_message_id_is_scoped_and_operator_identity_trusted"] = True

            def publish(**action):
                result, effect = runtime._plan_action(action, [])
                assert result["ok"], result
                if effect and effect["op"] == "send_message":
                    # This is an explicit transport fixture, without sampling.
                    effect["action_id"] = "transport-fixture:" + uuid.uuid4().hex
                runtime._append_event("action_result", result, allow_retirement=False)
                runtime.checkpoint([effect] if effect else [])

            def history(index):
                return request(f"/api/v1/chats/{chats[index]}", token=(operator, guest)[index]["token"])["chat"]

            def delivered(index, content):
                return [n for n in history(index)["history"]["messages"].values() if n.get("content") == content and n.get("meta", {}).get("dmn_delivery")]

            for index in (0, 1):
                publish(op="send_message", conversation_id=conversation_id("fixture", chats[index]), content=f"Reply to {index}")
            for index in (0, 1):
                wait_until(lambda i=index: delivered(i, f"Reply to {i}"))
                assert not delivered(index, f"Reply to {1-index}")
            report["replies_reach_only_addressed_chat"] = True
            for index, person in enumerate((operator, guest)):
                request("/api/chat/completions", forms[index], person["token"])
                assert len(delivered(index, f"Reply to {index}")) == 1
            assert len(user_events()) == 2
            report["input_retry_preserves_delivered_output"] = True
            old_reply = delivered(0, "Reply to 0")[0]
            collision = completion(0, "A new input", "new-input")
            rejected("/api/chat/completions", {**collision, "id": old_reply["id"]}, operator["token"])
            collision["user_message"]["id"] = old_reply["id"]
            rejected("/api/chat/completions", collision, operator["token"])
            assert delivered(0, "Reply to 0")[0] == old_reply
            report["browser_ids_cannot_overwrite_delivered_messages"] = True
            clean_input = completion(0, "Follow-up after spontaneous reply", "follow-up")
            clean_input["user_message"]["meta"] = {"dmn_delivery": "forged", "internal": True}
            request("/api/chat/completions", clean_input, operator["token"])
            wait_until(lambda: len(user_events()) == 3)
            saved_input = history(0)["history"]["messages"]["follow-up"]
            assert saved_input["parentId"] == old_reply["id"]
            assert not saved_input.get("meta", {}).get("dmn_delivery")
            report["new_input_appends_to_saved_leaf_without_forged_metadata"] = True
            sockets[1].disconnect()
            publish(op="send_message", conversation_id=conversation_id("fixture", chats[1]), content="Saved while disconnected")
            wait_until(lambda: delivered(1, "Saved while disconnected"))
            def delivery_events():
                with runtime.store.mutex:
                    return [json.loads(r[0]) for r in runtime.store.db.execute("SELECT payload FROM events WHERE kind='delivery_status'")]
            wait_until(lambda: any(e["state"] == "persisted" and e["connection"] == "disconnected" for e in delivery_events()))
            report["offline_delivery_persisted_with_factual_feedback"] = True

            # Relay restart after destination commit but before/without cursor:
            # rewinding only this fixture's ledger must not duplicate messages.
            request("/api/v1/functions/id/dmn_relay/toggle", {}, operator["token"])
            with sqlite3.connect(folder / "webui/dmn-bridge/multi-relay.sqlite3") as db:
                db.execute("UPDATE bindings SET cursor=0")
            request("/api/v1/functions/id/dmn_relay/toggle", {}, operator["token"])
            def replay_caught_up():
                with sqlite3.connect(folder / "webui/dmn-bridge/multi-relay.sqlite3") as db:
                    return db.execute("SELECT min(cursor) FROM bindings").fetchone()[0] > 0
            wait_until(replay_caught_up)
            assert len(delivered(1, "Saved while disconnected")) == 1
            report["relay_restart_replay_is_idempotent"] = True

            publish(op="block_participant", participant_id=participant_id("fixture", operator["id"]))
            rejected("/api/chat/completions", forms[0], operator["token"])
            assert len(user_events()) == 3
            report["operator_block_applies_to_retries"] = True
            publish(op="unblock_participant", participant_id=participant_id("fixture", operator["id"]), expected_block_revision=1)
            request(f"/api/v1/chats/{chats[1]}", token=guest["token"], method="DELETE")
            publish(op="send_message", conversation_id=conversation_id("fixture", chats[1]), content="Missing destination")
            publish(op="send_message", conversation_id=conversation_id("fixture", chats[0]), content="Other destination still works")
            wait_until(lambda: delivered(0, "Other destination still works"))
            wait_until(lambda: any(e["state"] == "failed" for e in delivery_events()))
            report["missing_chat_does_not_block_other_destinations"] = True
            report["passed"] = True
            return report
        finally:
            for client in sockets:
                if client.connected:
                    client.disconnect()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(10)
            if server:
                server.shutdown()
                server.server_close()
            if runtime:
                runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="multi-webui-", dir=args.output_dir)).resolve()
    print(f"Disposable fixture: {folder}", flush=True)
    report = run(folder)
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
