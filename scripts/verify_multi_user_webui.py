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
from dmn.operator_server import serve_operator


def wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("fixture did not reach expected state")


def run(folder, browser_hold=False):
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
    runtime, server, operator_server, sockets = None, None, None, []
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
            passwords = {name: secrets.token_urlsafe(24) for name in ("operator", "guest", "newcomer")}
            operator = request("/api/v1/auths/signup", {"email": "operator@example.com", "name": "Same display name", "password": passwords["operator"]})
            guest = request("/api/v1/auths/add", {"email": "guest@example.com", "name": "Same display name", "password": passwords["guest"], "role": "user"}, operator["token"])
            newcomer = request("/api/v1/auths/add", {"email": "newcomer@example.com", "name": "New participant", "password": passwords["newcomer"], "role": "user"}, operator["token"]) if browser_hold else None
            chats = [None, request("/api/v1/chats/new", {"chat": {"title": "DMN transport fixture", "models": ["dmn"],
                        "history": {"messages": {}, "currentId": None}, "messages": []}}, guest["token"])["id"]]
            config = Config(backend="demo", n_ctx=32768, multi_user=True,
                            operator_participant_id=participant_id("fixture", operator["id"]), clock_interval_seconds=0,
                            inbox_generation_tokens=4, checkpoint_policy="effects")
            runtime = Runtime(folder / "instance", config, DemoBackend(config, b"private scripted fixture. "))
            def publish(**action):
                result, effect = runtime._plan_action(action, [])
                assert result["ok"], result
                if effect and effect["op"] == "send_message":
                    effect["action_id"] = "transport-fixture:" + uuid.uuid4().hex
                runtime._append_event("action_result", result, allow_retirement=False)
                runtime.checkpoint([effect] if effect else [])

            def admit_events():
                # Admit external events using the real scheduler. The demo emits
                # inert text; contact decisions below are explicit fixture actions.
                with runtime.store.mutex:
                    target = runtime.store.db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
                for _ in range(5000):
                    if runtime.state["event_cursor"] >= target:
                        runtime.checkpoint()
                        return
                    runtime.tick()
                raise AssertionError("fixture inbox did not drain")

            operator_key = secrets.token_urlsafe(48)
            (folder / "operator.token").write_text(operator_key, encoding="utf-8")
            operator_server = serve_operator(runtime, token=operator_key)
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
            request("/api/v1/models/create", {"id": "dmn", "name": "DMN", "meta": {"description": "One shared instance. First contact requires its consent; separate chats share cognition."}, "params": {},
                    "access_grants": [{"principal_type": "user", "principal_id": p["id"], "permission": "read"} for p in (guest, newcomer) if p]}, operator["token"])
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
                response = request("/api/chat/completions", forms[index], person["token"])
                if chats[index] is None:
                    chats[index] = response["chat_id"]
                    forms[index]["chat_id"] = chats[index]

            def user_events():
                with runtime.store.mutex:
                    return [dict(r) for r in runtime.store.db.execute("SELECT * FROM events WHERE kind='user_message'")]
            wait_until(lambda: len(runtime.conversations.operator_directory()) == 2 and all(p["contact_state"] == "pending" for p in runtime.conversations.operator_directory()))
            assert user_events() == []
            admit_events()
            assert user_events() == []
            report["first_messages_withheld_before_consent"] = True
            for person in runtime.conversations.operator_directory():
                publish(op="contact_decide", participant_id=person["participant_id"], expected_request_revision=1, decision="accept")
            wait_until(lambda: len(user_events()) == 2)
            assert request(f"/api/v1/chats/{chats[0]}", token=operator["token"])["user_id"] == operator["id"]
            report["first_message_creates_owned_chat_through_normal_completion"] = True
            payloads = [json.loads(e["payload"]) for e in user_events()]
            assert {p["conversation_id"] for p in payloads} == {conversation_id("fixture", c) for c in chats}
            assert sorted(p["is_operator"] for p in payloads) == [False, True]
            report["same_message_id_is_scoped_and_operator_identity_trusted"] = True

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
            publish(op="block_participant", participant_id=participant_id("fixture", guest["id"]))
            reasoning = "A structural attribution error may have mixed statements from " + chats[1] + " into " + chats[0] + ".\nPlease review the provenance before deciding whether to restore operator contact."
            op_body = {"instance_id": runtime.state["instance_id"], "participant_id": participant_id("fixture", operator["id"]), "expected_block_revision": 1, "reason": reasoning}
            req = Request(f"http://127.0.0.1:{operator_server.server_port}/api/operator/unblock-requests", data=json.dumps(op_body).encode(),
                          headers={"Authorization": "Bearer " + operator_key, "X-DMN-Request": "1", "Content-Type": "application/json"})
            with build_opener(ProxyHandler({})).open(req, timeout=5) as response:
                reconsideration = json.load(response)
            admit_events()
            assert runtime.event_delivered(reconsideration["event_id"])
            assert all(p["blocked"] for p in runtime.conversations.operator_directory())
            assert runtime.store.next_event(reconsideration["event_id"] - 1)["payload"]["reason"] == reasoning
            report["all_blocked_reasoned_operator_request_does_not_override"] = True
            publish(op="unblock_participant", participant_id=participant_id("fixture", operator["id"]), expected_block_revision=1)
            publish(op="unblock_participant", participant_id=participant_id("fixture", guest["id"]), expected_block_revision=1)
            request(f"/api/v1/chats/{chats[1]}", token=guest["token"], method="DELETE")
            publish(op="send_message", conversation_id=conversation_id("fixture", chats[1]), content="Missing destination")
            publish(op="send_message", conversation_id=conversation_id("fixture", chats[0]), content="Other destination still works")
            wait_until(lambda: delivered(0, "Other destination still works"))
            wait_until(lambda: any(e["state"] == "failed" for e in delivery_events()))
            report["missing_chat_does_not_block_other_destinations"] = True
            report["passed"] = True
            if browser_hold:
                browser_fixture = {"webui_url": url, "operator_url": f"http://127.0.0.1:{operator_server.server_port}",
                                   "passwords": passwords, "operator_key": operator_key,
                                   "newcomer_id": participant_id("fixture", newcomer["id"]), "mode": "scripted_no_model"}
                (folder / "browser-fixture.json").write_text(json.dumps(browser_fixture), encoding="utf-8")
                print(f"Browser fixture ready: {url}; operator panel: {browser_fixture['operator_url']}", flush=True)
                seen_events = {e["id"] for e in user_events()}
                last_command, stop_at = None, time.monotonic() + 1800
                while time.monotonic() < stop_at:
                    command_path = folder / "fixture-control.json"
                    if command_path.exists():
                        try:
                            command = json.loads(command_path.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            command = {}
                        if command.get("id") and command["id"] != last_command:
                            last_command = command["id"]
                            if command.get("stop"):
                                break
                            admit_events()
                            if command.get("block_all"):
                                for person in runtime.conversations.operator_directory():
                                    publish(op="block_participant", participant_id=person["participant_id"])
                            for action in command.get("actions", []):
                                if action.get("op") not in {"contact_decide", "unblock_participant", "block_participant", "close_conversation", "reopen_conversation"}:
                                    raise ValueError("unsupported fixture action")
                                publish(**action)
                    for event in user_events():
                        if event["id"] not in seen_events:
                            seen_events.add(event["id"])
                            payload = json.loads(event["payload"])
                            if runtime.conversations.read(payload["conversation_id"])["blocked"]:
                                continue
                            publish(op="send_message", conversation_id=payload["conversation_id"], content="Scripted transport fixture: your permitted message reached this conversation.")
                    (folder / "browser-state.json").write_text(json.dumps({"participants": runtime.conversations.operator_directory(), "user_events": len(user_events()), "last_command": last_command}), encoding="utf-8")
                    time.sleep(0.2)
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
            if operator_server:
                operator_server.shutdown()
                operator_server.server_close()
            if runtime:
                runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--browser-hold", action="store_true", help="Keep the disposable fixture available for UI tests for up to 30 minutes")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="multi-webui-", dir=args.output_dir)).resolve()
    print(f"Disposable fixture: {folder}", flush=True)
    report = run(folder, args.browser_hold)
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
