from __future__ import annotations

import argparse
import dataclasses
import json
import signal
from pathlib import Path

from .config import Config
from .migration import import_transcript, prepare_bundle
from .runtime import Runtime
from .server import serve
from .storage import json_text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent DMN runtime for one llama.cpp sequence")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run/resume a persistent instance and its local UI")
    run.add_argument("--instance", type=Path, default=Path("data/instance"))
    run.add_argument("--config", type=Path)
    run.add_argument("--model", type=Path)
    run.add_argument("--demo", action="store_true", help="scripted transport fixture; NOT a model")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--checkpoint-policy", choices=["all_actions", "effects"],
                     help="effects skips read-only/input saves; messages, memory changes and sleep still checkpoint")
    run.add_argument("--checkpoint-seconds", type=float, dest="checkpoint_interval_seconds",
                     help="time between completed saves when state changes; 0 disables this threshold")
    run.add_argument("--checkpoint-tokens", type=int,
                     help="unsaved generated-token limit; 0 requires a time threshold")
    run.add_argument("--suspend-preparation-seconds", type=float,
                     help="limit preparation for suspension/shutdown; 0 skips it, native saving still takes time")
    run.add_argument("--checkpoint-reserve-bytes", type=int,
                     help="free-space margin beyond the estimated snapshot or packing scratch size")
    run.add_argument("--import-bundle", type=Path, help="explicit transcript fallback into a fresh instance")
    run.add_argument("--initial-context", type=Path, help="validated server-rendered context bundle into a fresh instance")
    run.add_argument("--kv-recovery", choices=["strict", "fallback", "rebuild"], default="strict",
                     help="strict: require native state; fallback: rebuild if unavailable; rebuild: skip native load")
    check = commands.add_parser("verify-native", help="test real native KV restoration in a fresh context")
    check.add_argument("--config", type=Path)
    check.add_argument("--model", type=Path)
    check.add_argument("--steps", type=int, default=24)
    check.add_argument("--report", type=Path, default=Path("data/native-verification.json"))
    migration = commands.add_parser("prepare-migration", help="archive Open WebUI export and continuity evidence")
    migration.add_argument("--export", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    migration.add_argument("--metadata", type=Path)
    migration.add_argument("--slot-state", type=Path)
    migration.add_argument("--chat-id")
    migration.add_argument("--leaf-id")
    migration.add_argument("--provider-request", type=Path, help="optional final Open WebUI provider request to archive exactly")
    migration.add_argument("--ignore-saved-summary", action="store_true",
                           help="project full history, matching Open WebUI with compaction disabled")
    capture = commands.add_parser("capture-provider", help="capture final provider requests from a disposable Open WebUI copy; no inference")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--model", required=True, help="model ID advertised to the disposable Open WebUI")
    capture.add_argument("--port", type=int, default=9932)
    initial = commands.add_parser("prepare-initial-context", help="render/tokenize a captured provider request without inference")
    initial.add_argument("--provider-request", type=Path, required=True)
    initial.add_argument("--output", type=Path, required=True)
    initial.add_argument("--config", type=Path, required=True)
    initial.add_argument("--server-url", required=True, help="direct loopback llama-server, not the model router")
    initial.add_argument("--keep-prefix-tokens", type=int, default=0, help="0 derives the stable system/template prefix; positive values are explicit overrides")
    args = parser.parse_args(argv)
    if args.command == "prepare-initial-context":
        from .initial_context import prepare_initial_context
        report = prepare_initial_context(args.provider_request, args.output, Config.read(args.config),
                                         args.server_url, args.keep_prefix_tokens)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "capture-provider":
        import threading
        from .capture import serve_capture
        stopped = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stopped.set())
        server = serve_capture(args.output, args.model, args.port)
        print(f"CAPTURE ONLY, NO INFERENCE: http://127.0.0.1:{server.server_port}/v1", flush=True)
        try:
            while not stopped.wait(1):
                pass
        finally:
            server.shutdown()
            server.server_close()
        return 0
    if args.command == "prepare-migration":
        print(json_text(prepare_bundle(args.export, args.output, args.metadata, args.slot_state, args.chat_id, args.leaf_id,
                                       args.provider_request, not args.ignore_saved_summary)))
        return 0
    if args.command == "verify-native":
        from .verify import verify_native
        config = Config.read(args.config) if args.config else Config(prompt_format="plain")
        if args.model:
            config = dataclasses.replace(config, model_path=str(args.model.resolve()))
        if config.backend != "llama" or not config.model_path or args.steps < 2:
            parser.error("verify-native requires a llama model and at least 2 continuation steps")
        report = verify_native(config, args.steps, args.report)
        print(json.dumps({k: v for k, v in report.items() if k != "fingerprint"}, indent=2))
        print(f"Evidence saved to {args.report.resolve()}")
        return 0
    if args.demo and (args.config or args.model):
        parser.error("--demo cannot be combined with --config or --model")
    if args.initial_context and (args.import_bundle or args.demo or args.model):
        parser.error("--initial-context cannot be combined with --import-bundle, --demo or --model")
    if args.config:
        config = Config.read(args.config)
    elif args.initial_context:
        config = Config.read(args.initial_context / "config.json")
    elif args.demo:
        config = Config(backend="demo", prompt_format="plain", token_delay_seconds=0.01)
    elif args.model:
        config = Config(model_path=str(args.model.resolve()))
    else:
        # The effective configuration was captured with the committed state.
        from .storage import Store
        store = Store(args.instance)
        saved = store.latest()
        store.close()
        if not saved:
            parser.error("a new instance requires --config, --model, or --demo")
        manifest = json.loads((saved / "manifest.json").read_text())
        config = Config(**manifest["fingerprint"]["config"])
    if (args.import_bundle or args.initial_context) and args.instance.exists():
        parser.error("import requires a new instance directory")
    overrides = {key: getattr(args, key) for key in
                 ("checkpoint_policy", "checkpoint_interval_seconds", "checkpoint_tokens", "suspend_preparation_seconds", "checkpoint_reserve_bytes")
                 if getattr(args, key) is not None}
    if overrides:
        try:
            config = dataclasses.replace(config, **overrides)
        except ValueError as exc:
            parser.error(str(exc))
    runtime = Runtime(args.instance, config, kv_recovery=args.kv_recovery, initial_context=args.initial_context)
    server = None
    try:
        if args.import_bundle:
            import_transcript(runtime, args.import_bundle)
        server = serve(runtime, args.port)
        print(f"DMN: http://127.0.0.1:{server.server_port} | instance {runtime.state['instance_id']}", flush=True)
        if args.demo:
            print("DEMO FIXTURE: no model inference or native KV state.", flush=True)
        print("Ctrl+C requests preparation, a checkpoint, and shutdown.", flush=True)
        signal.signal(signal.SIGINT, lambda *_: runtime.control("shutdown"))
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda *_: runtime.control("shutdown"))
        runtime.run()
    finally:
        if server:
            server.shutdown()
            server.server_close()
        runtime.close()
    return 0
