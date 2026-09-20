"""Run the installed Open WebUI against disposable data, with the DMN adapter.

Usage: python scripts/openwebui_sandbox.py --webui-python PATH
All writable locations belong to the supplied fresh sandbox. No primary DB is copied.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

root = Path(__file__).resolve().parents[1]


def request(url, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    with build_opener(ProxyHandler({})).open(Request(url, data=data, headers=headers), timeout=5) as response:
        return json.load(response)


def wait_ready(url, process):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Process exited: {process.returncode}; inspect sandbox logs")
        try:
            return request(url)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def stop_children(processes, dmn_port, timeout=600):
    if not processes:
        return
    native = processes[0]
    if native.poll() is None:
        try:
            url = f"http://127.0.0.1:{dmn_port}"
            status = request(url + "/api/status")
            if status.get("process_id") != native.pid:
                raise RuntimeError("runtime port does not identify the child process")
            req = Request(url + "/api/control", data=b'{"action":"shutdown"}',
                          headers={"X-DMN-Request": "1", "Content-Type": "application/json"})
            build_opener(ProxyHandler({})).open(req, timeout=3).close()
            native.wait(timeout=timeout)
        except Exception as exc:
            print(f"DMN shutdown has not been confirmed ({exc}). Child PID {native.pid} "
                  f"has been left running; inspect http://127.0.0.1:{dmn_port}/ for storage "
                  "warnings or an in-progress save. It was not forcibly terminated.", flush=True)
    for process in reversed(processes[1:]):
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webui-python", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "data" / "openwebui-test")
    parser.add_argument("--webui-port", type=int, default=3031)
    parser.add_argument("--dmn-port", type=int, default=8766)
    parser.add_argument("--runtime-config", type=Path, help="use a real model with this config instead of the transport fixture")
    parser.add_argument("--instance", type=Path, help="native instance to resume; defaults to the sandbox's instance directory")
    parser.add_argument("--reuse", action="store_true", help="restart a previously created disposable sandbox")
    parser.add_argument("--shutdown-timeout", type=float, default=600,
                        help="seconds to wait for saved shutdown; expiration leaves DMN running")
    args = parser.parse_args()
    if not 0 < args.shutdown_timeout < float("inf"):
        parser.error("--shutdown-timeout must be positive and finite")
    if args.instance and not args.runtime_config:
        parser.error("--instance requires --runtime-config")
    for port in (args.webui_port, args.dmn_port):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                parser.error(f"port {port} is already in use; open the running sandbox or choose separate ports")
    folder = args.data.resolve()
    marker = folder / "sandbox.json"
    if folder.exists() and not (args.reuse and marker.exists()):
        parser.error("use a new data directory, or --reuse with an existing sandbox marker")
    instance = (args.instance or folder / "instance").resolve()
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("fixture") != (args.runtime_config is None):
            parser.error("cannot change a sandbox between a fixture and a real model")
        if Path(previous.get("instance_path", folder / "instance")).resolve() != instance:
            parser.error("cannot rebind a sandbox to a different instance directory")
    folder.mkdir(parents=True, exist_ok=True)
    static = folder / "static"
    static.mkdir(exist_ok=True)
    processes, logs = [], []
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"})
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    def launch(command, name, environment):
        log = (folder / (name + ".log")).open("a", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=log, creationflags=flags)
        processes.append(process)
        return process
    try:
        command = ([sys.executable, "-m", "dmn", "run", "--config", str(args.runtime_config.resolve()),
                    "--instance", str(instance), "--port", str(args.dmn_port)] if args.runtime_config else
                   [sys.executable, str(root / "scripts/run_bridge_fixture.py"), str(instance), str(args.dmn_port)])
        native = launch(command, "dmn", env)
        dmn_url = f"http://127.0.0.1:{args.dmn_port}"
        status = wait_ready(dmn_url + "/api/status", native)
        env.update({"DATA_DIR": str(folder / "webui"), "STATIC_DIR": str(static), "FROM_INIT_PY": "True",
                    "DATABASE_URL": "sqlite:///" + (folder / "webui/webui.db").as_posix(),
                    "WEBUI_AUTH": "False", "WEBUI_SECRET_KEY": "disposable-dmn-sandbox-only-local-testing",
                    "ENABLE_OLLAMA_API": "False", "ENABLE_OPENAI_API": "False",
                    "ENABLE_VERSION_UPDATE_CHECK": "False", "OFFLINE_MODE": "True",
                    "RAG_EMBEDDING_ENGINE": "openai", "DMN_URL": dmn_url,
                    "DMN_INSTANCE_ID": status["instance_id"], "DO_NOT_TRACK": "True",
                    "ANONYMIZED_TELEMETRY": "False", "SCARF_NO_ANALYTICS": "True"})
        (folder / "webui").mkdir(exist_ok=True)
        webui = launch([str(args.webui_python), "-m", "uvicorn", "open_webui.main:app", "--host", "127.0.0.1", "--port", str(args.webui_port), "--workers", "1"], "webui", env)
        url = f"http://127.0.0.1:{args.webui_port}"
        wait_ready(url + "/ready", webui)
        auth = request(url + "/api/v1/auths/signin", {"email": "", "password": ""})
        token = auth["token"]
        functions = request(url + "/api/v1/functions/", token=token)
        for function_id, filename, name in (("dmn", "dmn_pipe.py", "DMN"), ("dmn_relay", "dmn_relay.py", "DMN relay")):
            existing = next((f for f in functions if f["id"] == function_id), None)
            payload = {"id": function_id, "name": name,
                       "content": (root / "integrations/openwebui" / filename).read_text(), "meta": {}}
            if existing is None:
                request(url + "/api/v1/functions/create", payload, token)
            else:
                saved = request(url + f"/api/v1/functions/id/{function_id}", token=token)
                if saved["content"] != payload["content"]:
                    request(url + f"/api/v1/functions/id/{function_id}/update", payload, token)
            if existing is None or not existing["is_active"]:
                request(url + f"/api/v1/functions/id/{function_id}/toggle", {}, token)
        marker.write_text(json.dumps({"webui_url": url, "dmn_url": dmn_url, "instance_id": status["instance_id"],
                                      "instance_path": str(instance), "fixture": args.runtime_config is None,
                                      "primary_data_used": False}, indent=2))
        (folder / "processes.json").write_text(json.dumps({"supervisor_pid": os.getpid(),
            "native_pid": status.get("process_id", native.pid), "webui_pid": webui.pid, "instance_id": status["instance_id"],
            "owner_token": os.environ.get("DMN_SANDBOX_OWNER")}))
        label = "Native model runtime" if args.runtime_config else "Transport fixture"
        print(f"Disposable Open WebUI ready: {url}\n{label}: {dmn_url}\nLogs: {folder}", flush=True)
        print("Press Ctrl+C to stop both sandbox processes.", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_children(processes, args.dmn_port, args.shutdown_timeout)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
