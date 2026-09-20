"""Integration checks against ONLY a marked disposable sandbox.

Creates a disposable first conversation if needed. Never accepts a primary URL.
"""
import argparse
import copy
import json
import sqlite3
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

from dmn.capture import serve_capture


def wait_for(check, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.25)
    raise AssertionError("Timed out waiting for integration condition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path, default=Path("data/openwebui-test"))
    args = parser.parse_args()
    folder = args.sandbox.resolve()
    marker = json.loads((folder / "sandbox.json").read_text())
    assert marker["fixture"] and marker["primary_data_used"] is False
    url, dmn = marker["webui_url"], marker["dmn_url"]
    token = None

    def request(path, body=None, base=None, expected=200):
        headers = {"Content-Type": "application/json", "X-DMN-Request": "1"}
        if token and base is None:
            headers["Authorization"] = "Bearer " + token
        req = Request((base or url) + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            response = build_opener(ProxyHandler({})).open(req, timeout=20)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            assert response.status == expected, (path, response.status, data[:1000])
            return json.loads(data)

    token = request("/api/v1/auths/signin", {"email": "", "password": ""})["token"]
    ledger_path = folder / "webui/dmn-bridge/relay.sqlite3"
    with sqlite3.connect(ledger_path) as connection:
        bound = connection.execute("SELECT * FROM binding").fetchone()
    if not bound:
        request("/api/models")
        message = {"id": str(uuid.uuid4()), "role": "user", "content": "Disposable DMN transport verification",
                   "parentId": None, "childrenIds": [], "timestamp": int(time.time())}
        request("/api/chat/completions", {"model": "dmn", "stream": True, "parent_id": None,
                "session_id": "sandbox-verification", "id": str(uuid.uuid4()), "user_message": message,
                "messages": [{"role": "user", "content": message["content"]}], "background_tasks": {}})
    def initially_caught_up():
        with sqlite3.connect(ledger_path) as connection:
            row = connection.execute("SELECT cursor FROM binding").fetchone()
        state = request("/api/status", base=dmn)
        if state["mode"] in {"error", "context_full"}:
            raise AssertionError(f"Fixture stopped: {state.get('error')}; use a fresh sandbox")
        return row and row[0] >= 2 and row[0] == len(request("/api/messages", base=dmn)) and state["mode"] == "sleeping" and state["sleep_until"] is None
    wait_for(initially_caught_up)
    with sqlite3.connect(ledger_path) as connection:
        connection.row_factory = sqlite3.Row
        binding = dict(connection.execute("SELECT * FROM binding").fetchone())
        receipt = dict(connection.execute("SELECT * FROM receipts ORDER BY rowid LIMIT 1").fetchone())
    chat_id = binding["chat_id"]
    def chat():
        return request(f"/api/v1/chats/{chat_id}")["chat"]
    def delivered():
        return {n["meta"]["dmn_delivery"]: n for n in chat()["history"]["messages"].values() if n.get("meta", {}).get("dmn_delivery")}
    initial = delivered()
    initial_events = len(request("/api/events", base=dmn))
    user = chat()["history"]["messages"][receipt["message_id"]]
    body = {"model": "dmn", "stream": True, "chat_id": chat_id, "session_id": "sandbox-verification",
            "id": receipt["assistant_id"], "user_message": user, "parent_id": user.get("parentId"),
            "messages": [{"role": "user", "content": user["content"]}], "background_tasks": {}}
    request("/api/chat/completions", copy.deepcopy(body))
    assert len(request("/api/events", base=dmn)) == initial_events
    assert delivered() == initial, "retry overwrote a published outgoing message"
    request("/api/chat/completions", {**body, "id": str(uuid.uuid4())}, expected=409)
    request("/api/chat/completions", {**body, "user_message": {**user, "content": "edited"}}, expected=409)
    request(f"/api/v1/chats/{chat_id}/compact", {}, expected=409)
    request(f"/api/v1/chats/{chat_id}/messages/{next(iter(initial.values()))['id']}", {"content": "edited"}, expected=409)

    # Disconnect the adapter; computation and durable outgoing delivery continue.
    request("/api/v1/functions/id/dmn_relay/toggle", {})
    time.sleep(0.5)
    count = len(request("/api/messages", base=dmn))
    request("/api/events", {"content": "Fixture wake while Open WebUI relay is offline", "instance_id": marker["instance_id"],
                            "idempotency_key": "integration:" + str(uuid.uuid4())}, base=dmn, expected=202)
    wait_for(lambda: len(request("/api/messages", base=dmn)) >= count + 2)
    assert delivered() == initial
    request("/api/v1/functions/id/dmn_relay/toggle", {})
    wait_for(lambda: len(delivered()) == len(request("/api/messages", base=dmn)))
    after_reconnect = delivered()

    # Simulate a crash after destination commit, before durable cursor advance.
    request("/api/v1/functions/id/dmn_relay/toggle", {})
    time.sleep(0.5)
    with sqlite3.connect(ledger_path) as connection:
        connection.execute("UPDATE binding SET cursor=0")
    request("/api/v1/functions/id/dmn_relay/toggle", {})
    def caught_up():
        with sqlite3.connect(ledger_path) as connection:
            return connection.execute("SELECT cursor FROM binding").fetchone()[0] == len(after_reconnect)
    wait_for(caught_up)
    assert delivered() == after_reconnect

    # Verify final-provider capture against the installed compaction middleware.
    capture_root = folder / "captures"
    capture = serve_capture(capture_root, "capture-fixture", 0)
    original_openai = request("/openai/config")
    original_compaction = request("/api/v1/chats/config")
    try:
        request("/openai/config/update", {"ENABLE_OPENAI_API": True, "OPENAI_API_BASE_URLS": [f"http://127.0.0.1:{capture.server_port}/v1"],
                "OPENAI_API_KEYS": [""], "OPENAI_API_CONFIGS": {"0": {"enable": True, "model_ids": ["capture-fixture"]}}})
        request("/api/v1/chats/config", {**original_compaction, "ENABLE_CONTEXT_COMPACTION": True,
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD": 60000, "CONTEXT_COMPACTION_TOKEN_CAP": 60000})
        models = request("/api/models")
        assert any(m["id"] == "capture-fixture" for m in models["data"])
        old_id, boundary_id, latest_id = (str(uuid.uuid4()) for _ in range(3))
        old = {"id": old_id, "role": "user", "content": "ARCHIVED_SENTINEL_SHOULD_NOT_REACH_PROVIDER", "parentId": None, "childrenIds": [boundary_id], "timestamp": int(time.time())}
        boundary = {"id": boundary_id, "role": "assistant", "content": "RETAINED_BOUNDARY", "contextSummary": "SAVED_SUMMARY_SENTINEL", "parentId": old_id, "childrenIds": [latest_id], "timestamp": int(time.time())}
        latest = {"id": latest_id, "role": "user", "content": "LATEST_SENTINEL", "parentId": boundary_id, "childrenIds": [], "timestamp": int(time.time())}
        nodes = {m["id"]: m for m in (old, boundary, latest)}
        imported = request("/api/v1/chats/import", {"chats": [{"chat": {"title": "Disposable saved-summary capture", "models": ["capture-fixture"], "history": {"messages": nodes, "currentId": latest_id}}}]})[0]
        existing = set(capture_root.iterdir())
        request("/api/chat/completions", {"model": "capture-fixture", "stream": True,
                "chat_id": imported["id"], "session_id": "sandbox-capture", "id": str(uuid.uuid4()),
                "parent_id": boundary_id, "user_message": latest,
                "messages": [{"role": "system", "content": "ORIGINAL_SYSTEM_SENTINEL"}], "background_tasks": {}})
        directory = wait_for(lambda: next((p for p in capture_root.iterdir() if p not in existing and (p / "capture.json").exists()), None))
        raw = (directory / "provider-request.json").read_text()
        assert "SAVED_SUMMARY_SENTINEL" in raw and "RETAINED_BOUNDARY" in raw and "LATEST_SENTINEL" in raw
        assert "ORIGINAL_SYSTEM_SENTINEL" in raw and "ARCHIVED_SENTINEL_SHOULD_NOT_REACH_PROVIDER" not in raw
        archived = request(f"/api/v1/chats/{imported['id']}")["chat"]["history"]["messages"]
        assert old_id in archived, "compaction must retain original history"
        report = {"openwebui_version": request("/api/version")["version"], "primary_data_used": False,
                  "incoming_retry_preserved_outgoing": True, "edited_and_regenerated_input_rejected": True,
                  "manual_compaction_blocked_for_dmn": True, "offline_messages_replayed": True,
                  "destination_commit_retry_deduplicated": True, "delivered_messages": len(after_reconnect),
                  "saved_summary_projection_captured": True, "full_history_retained": True,
                  "capture_directory": str(directory)}
        (folder / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        request("/openai/config/update", {k: original_openai[k] for k in ("ENABLE_OPENAI_API", "OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS")})
        request("/api/v1/chats/config", original_compaction)
        capture.shutdown()
        capture.server_close()


if __name__ == "__main__":
    main()
