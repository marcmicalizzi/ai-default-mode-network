"""Request checkpointed shutdown of the exact runtime in a sandbox marker."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    marker = json.loads((args.data / "sandbox.json").read_text())
    if marker.get("primary_data_used") is not False:
        parser.error("not an isolated sandbox marker")
    url = marker["dmn_url"]
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1" or parts.path not in {"", "/"}:
        parser.error("sandbox runtime must be a loopback HTTP origin")
    opener = build_opener(ProxyHandler({}))
    with opener.open(url + "/api/status", timeout=5) as response:
        status = json.load(response)
    if status["instance_id"] != marker["instance_id"]:
        parser.error("the port belongs to a different instance; refusing to stop it")
    req = Request(url + "/api/control", data=b'{"action":"shutdown"}',
                  headers={"Content-Type": "application/json", "X-DMN-Request": "1"})
    with opener.open(req, timeout=5) as response:
        print(response.read().decode())
    print("Shutdown requested; the instance may accept, defer or refuse. Inspect the runtime UI for its reply.")


if __name__ == "__main__":
    main()
