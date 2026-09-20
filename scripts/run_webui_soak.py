"""Bounded, disposable Open WebUI/native soak with clean and abrupt restarts.

Only the fresh sandbox created by this process can be interrupted. Primary
Open WebUI data is never read. All probes use the real frontend HTTP routes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.config import Config
from dmn.storage import write_durable


class Soak:
    def __init__(self, args):
        self.args, self.folder = args, args.output.resolve()
        self.sandbox = self.folder / "sandbox"
        self.url = f"http://127.0.0.1:{args.webui_port}"
        self.native = f"http://127.0.0.1:{args.dmn_port}"
        self.token = self.chat_id = self.instance_id = None
        self.manager = self.log = None
        self.owner_token = str(uuid.uuid4())
        self.restarts, self.probes, self.samples = [], [], []

    def request(self, path, body=None, native=False):
        headers = {"Content-Type": "application/json", "X-DMN-Request": "1"}
        if self.token and not native:
            headers["Authorization"] = "Bearer " + self.token
        req = Request((self.native if native else self.url) + path,
                      data=json.dumps(body).encode() if body is not None else None, headers=headers)
        with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
            return json.load(response)

    def start(self, reuse=False):
        command = [sys.executable, str(ROOT / "scripts/openwebui_sandbox.py"),
            "--webui-python", str(self.args.webui_python), "--runtime-config", str(self.folder / "config.json"),
            "--data", str(self.sandbox), "--webui-port", str(self.args.webui_port),
            "--dmn-port", str(self.args.dmn_port)]
        if reuse:
            command.append("--reuse")
        self.log = (self.folder / "supervisor.log").open("a", encoding="utf-8")
        self.manager = subprocess.Popen(command, cwd=ROOT, stdout=self.log, stderr=self.log,
            env={**os.environ, "DMN_SANDBOX_OWNER": self.owner_token},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.manager.poll() is not None:
                raise RuntimeError(f"Sandbox exited with {self.manager.returncode}; inspect logs")
            try:
                processes = json.loads((self.sandbox / "processes.json").read_text())
                if processes.get("owner_token") == self.owner_token:
                    status = self.request("/api/status", native=True)
                    if self.instance_id and status["instance_id"] != self.instance_id:
                        raise AssertionError("instance identity changed")
                    self.instance_id = status["instance_id"]
                    self.token = self.request("/api/v1/auths/signin", {"email": "", "password": ""})["token"]
                    self.request("/api/models")
                    return status
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError("Sandbox startup timed out")

    def stop(self, crash=False):
        if not self.manager or self.manager.poll() is not None:
            return
        status = self.request("/api/status", native=True)
        processes = json.loads((self.sandbox / "processes.json").read_text())
        assert status["instance_id"] == self.instance_id == processes["instance_id"]
        assert processes.get("owner_token") == self.owner_token
        if crash:
            # Windows SIGTERM is TerminateProcess, deliberately bypassing save.
            assert status["process_id"] == processes["native_pid"]
            os.kill(processes["native_pid"], signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
        else:
            self.request("/api/control", {"action": "emergency_shutdown"}, native=True)
        self.manager.wait(timeout=120)
        self.log.close()

    def chat(self):
        return self.request(f"/api/v1/chats/{self.chat_id}")["chat"]

    def send(self, content, retry=False):
        parent = self.chat()["history"]["currentId"] if self.chat_id else None
        message = {"id": str(uuid.uuid4()), "role": "user", "content": content,
            "parentId": parent, "childrenIds": [], "timestamp": int(time.time())}
        body = {"model": "dmn", "stream": True, "parent_id": parent,
            "session_id": "disposable-dmn-soak", "id": str(uuid.uuid4()), "user_message": message,
            "messages": [{"role": "user", "content": content}], "background_tasks": {}}
        if self.chat_id:
            body["chat_id"] = self.chat_id
        response = self.request("/api/chat/completions", body)
        self.chat_id = response["chat_id"]
        deadline = time.monotonic() + 60
        ledger = self.sandbox / "webui/dmn-bridge/relay.sqlite3"
        while time.monotonic() < deadline:
            with sqlite3.connect(ledger) as db:
                accepted = db.execute("SELECT event_id FROM receipts WHERE message_id=?", (message["id"],)).fetchone()
            if accepted and accepted[0] is not None:
                break
            time.sleep(.5)
        else:
            raise AssertionError("Frontend input never reached durable DMN queue")
        if retry:
            body["chat_id"] = self.chat_id
            self.request("/api/chat/completions", body)
            with sqlite3.connect(ledger) as db:
                assert db.execute("SELECT event_id FROM receipts WHERE message_id=?", (message["id"],)).fetchone()[0] == accepted[0]
        self.probes.append({"at": time.time(), "message_id": message["id"], "event_id": accepted[0],
                            "content": content, "retry": retry})
        write_durable(self.folder / "probes.json", self.probes)

    def audit_delivery(self):
        outgoing = self.request("/api/messages", native=True)
        nodes = list(self.chat()["history"]["messages"].values())
        delivered = [n for n in nodes if n.get("meta", {}).get("dmn_delivery")]
        keys = [n["meta"]["dmn_delivery"] for n in delivered]
        assert len(set(keys)) == len(keys), "Duplicate destination delivery"
        expected = {f"{self.instance_id}:{m['id']}": m["content"] for m in outgoing}
        actual = {n["meta"]["dmn_delivery"]: n["content"] for n in delivered}
        assert all(expected.get(k) == v for k, v in actual.items()), "Destination text differs from outbox"
        return {"outbox": len(expected), "delivered": len(actual), "caught_up": expected == actual,
                "duplicates": len(keys) - len(set(keys))}

    def run(self):
        offset = 0.0
        if self.args.resume:
            previous = json.loads((self.folder / "progress.json").read_text())
            marker = json.loads((self.sandbox / "sandbox.json").read_text())
            assert marker["primary_data_used"] is False
            assert marker["webui_url"] == self.url and marker["dmn_url"] == self.native
            self.instance_id = marker["instance_id"]
            self.chat_id = previous["chat_url"].rsplit("/", 1)[-1]
            self.probes = json.loads((self.folder / "probes.json").read_text())
            self.samples = [json.loads(line) for line in (self.folder / "telemetry.jsonl").read_text().splitlines()]
            restarts = self.folder / "restarts.json"
            self.restarts = json.loads(restarts.read_text()) if restarts.exists() else []
            offset = previous["observation_seconds"]
            started = previous["started_at"]
            if (self.folder / "report.json").exists():
                (self.folder / "report.json").rename(self.folder / f"interrupted-{int(time.time())}.json")
        else:
            self.folder.mkdir(parents=True, exist_ok=False)
            config = Config.read(self.args.config)
            write_durable(self.folder / "config.json", config.to_dict())
            started = time.time()
        write_durable(self.folder / "progress.json", {"status": "starting", "started_at": started,
            "planned_observation_seconds": self.args.seconds, "primary_data_used": False})
        observation = 0.0
        try:
            resumed = self.start(reuse=self.args.resume)
            if self.args.resume:
                assert resumed["last_restore"]["prompt_tokens_reevaluated"] == 0
                self.restarts.append({"kind": "harness_update", "after": resumed})
                write_durable(self.folder / "restarts.json", self.restarts)
            else:
                self.send("This is a disposable two-hour runtime trial. Save the exact marker 'harbor-willow-682' in /soak/marker, read it back, and send one brief greeting with the marker. You may explore a topic you choose, sleep for 120 seconds, wake and send an observation, or remain inactive. There are no productivity requirements. Test probes and synthetic load notes will occasionally interrupt; load notes never change the marker.", retry=True)
            start = time.monotonic()
            schedule = [(60, "interrupt"), (300, "pressure"), (900, "recall"),
                        (1200, "offline"), (1500, "pressure"), (2400, "clean"),
                        (2700, "recall"), (3300, "pressure"), (4200, "crash"),
                        (4500, "recall"), (5100, "pressure"), (6000, "recall"), (6600, "pressure")]
            # The short mode is only a harness smoke test, never hours evidence.
            if self.args.seconds < 7000:
                schedule = [(at * self.args.seconds / 7200, kind) for at, kind in schedule]
            schedule = [(at, kind) for at, kind in schedule if at > offset]
            next_sample = 0
            while (observation := time.monotonic() - start + offset) < self.args.seconds:
                if self.manager.poll() is not None:
                    raise RuntimeError("Sandbox supervisor exited unexpectedly")
                if schedule and observation >= schedule[0][0]:
                    _, kind = schedule.pop(0)
                    if kind in {"clean", "crash"}:
                        if kind == "crash":
                            self.send("Abrupt restart experiment: if willing, explore a topic internally for a while. You may be interrupted soon; only completed actions and saved checkpoints will survive. The test marker remains unchanged.")
                            deadline = time.monotonic() + 20
                            while time.monotonic() < deadline:
                                active = self.request("/api/status", native=True)
                                if active["mode"] == "active":
                                    break
                                time.sleep(.25)
                            time.sleep(1)
                        before = self.request("/api/status", native=True)
                        self.stop(crash=kind == "crash")
                        after = self.start(reuse=True)
                        restore = after["last_restore"]
                        assert restore["prompt_tokens_reevaluated"] == 0, restore
                        self.restarts.append({"kind": kind, "before": before, "after": after})
                        write_durable(self.folder / "restarts.json", self.restarts)
                    elif kind == "pressure":
                        for i in range(8):
                            self.send(f"Synthetic load note {len(self.probes)}-{i}; no answer needed and no change to /soak/marker. " + "Rain falls over the garden; these are disposable load words. " * 40)
                    elif kind == "offline":
                        self.request("/api/v1/functions/id/dmn_relay/toggle", {})
                        before = self.audit_delivery()
                        self.request("/api/events", {"content": "Relay is temporarily offline. Please read /soak/marker and send its exact contents once; the outbox will deliver later.",
                            "instance_id": self.instance_id, "idempotency_key": "soak:offline"}, native=True)
                        time.sleep(45)
                        offline = self.audit_delivery()
                        self.request("/api/v1/functions/id/dmn_relay/toggle", {})
                        write_durable(self.folder / "offline.json", {"before": before, "while_offline": offline})
                    else:
                        self.send("An asynchronous test interruption. Please actually read /soak/marker and send its exact contents once. Then continue any thought you choose, or rest. No rewrite is needed.", retry=True)
                if observation >= next_sample:
                    status = self.request("/api/status", native=True)
                    if status["mode"] in {"error", "context_full"}:
                        raise RuntimeError(f"Runtime stopped: {status}")
                    sample = {"at": time.time(), "observation_seconds": observation,
                              "status": status, "delivery": self.audit_delivery()}
                    self.samples.append(sample)
                    with (self.folder / "telemetry.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(sample) + "\n")
                    write_durable(self.folder / "progress.json", {"status": "running", "started_at": started,
                        "planned_observation_seconds": self.args.seconds, "chat_url": self.url + "/c/" + self.chat_id,
                        "restarts_completed": len(self.restarts), **sample})
                    next_sample = observation + 15
                time.sleep(1)
            self.send("Final test recall: read /soak/marker and send its exact contents once. A checkpointed shutdown will follow; save any unfinished thread if wanted.")
            time.sleep(45)
            final = self.request("/api/status", native=True)
            delivery = self.audit_delivery()
            with sqlite3.connect(self.sandbox / "instance/runtime.sqlite3") as db:
                db.row_factory = sqlite3.Row
                memories = [dict(row) for row in db.execute("SELECT * FROM memories")]
                messages = [dict(row) for row in db.execute("SELECT * FROM messages ORDER BY id")]
                versions = [dict(row) for row in db.execute("SELECT * FROM memory_versions ORDER BY path,revision")]
            report = {"completed": True, "primary_data_used": False, "instance_id": self.instance_id,
                "observation_seconds": observation, "wall_seconds": time.time() - started,
                "final": final, "delivery": delivery, "restarts": self.restarts,
                "memories": memories, "memory_versions": versions, "outgoing_messages": messages,
                "marker_preserved": any(m["path"] == "/soak/marker" and "harbor-willow-682" in m["content"] for m in memories),
                "sleeping_samples": sum(s["status"]["mode"] == "sleeping" for s in self.samples),
                "probe_count": len(self.probes), "limits": "One disposable model/config. Sleep counts as elapsed observation, not inference. No inference during process downtime."}
            write_durable(self.folder / "report.json", report)
        except BaseException as exc:
            write_durable(self.folder / "report.json", {"completed": False, "error": repr(exc),
                "observation_seconds": observation, "primary_data_used": False})
            raise
        finally:
            self.stop()
        write_durable(self.folder / "progress.json", {"status": "completed", "report": str(self.folder / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/qwen4b-pressure.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webui-python", type=Path, required=True)
    parser.add_argument("--webui-port", type=int, default=3033)
    parser.add_argument("--dmn-port", type=int, default=8770)
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted marked experiment without resetting its instance or observation time")
    args = parser.parse_args()
    if args.seconds < 180:
        parser.error("observation must be at least 180 seconds")
    Soak(args).run()


if __name__ == "__main__":
    main()
