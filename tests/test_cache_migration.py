import dataclasses
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from dmn.backend import sha256_file
from dmn.cache_migration import migrate_cache, migration_changes
from dmn.config import Config
from dmn.ending import Lifecycle, InstanceEnded
from dmn.native_cache import verify_compaction
from dmn.preservation import InstanceHeld
from dmn.storage import InstanceLock, Store, write_durable
from tests.test_compact_cache_state import fixture


class FakeNative:
    kind = "native_llama_kv"
    retirement_window = 4
    decode_calls = 0

    def __init__(self, config):
        self.config = config
        self.fingerprint = {"kind": self.kind, "model_sha256": "fixture", "binding_version": "0.3.35",
                            "config": config.to_dict()}

    def load(self, folder):
        return {"native_state_loaded": True, "prompt_tokens_reevaluated": 0, "decode_calls_during_load": 0}

    def verify_loaded_snapshot(self, folder, digest):
        if digest != sha256_file(folder / "state.bin"):
            raise ValueError("changed file")
        return {"serialized_native_state_bytes_equal": True}

    def close(self):
        pass


class CacheMigrationTest(unittest.TestCase):
    def test_staged_notice_waits_for_start_and_is_not_lost(self):
        from dmn.backend import DemoBackend
        from dmn.runtime import Runtime
        config = Config(backend="demo", n_ctx=65536, turnover_reserve=4096, clock_interval_seconds=0)
        root = self.base / "staged-fixture"
        runtime = Runtime(root, config, DemoBackend(config), prepare_only=True)
        try:
            runtime.state["cache_migration_notice_pending"] = True
            runtime.state["cache_migration"] = {"window": 1024, "masked_local_cells_removed": 100}
            runtime.checkpoint(reason="fixture")
        finally:
            runtime.close()
        runtime = Runtime(root, config, DemoBackend(config))
        try:
            before = list(runtime.backend.tokens)
            runtime.tick()
            self.assertEqual(runtime.backend.tokens, before)
            self.assertTrue(runtime.state["cache_migration_notice_pending"])
            runtime.enqueue("Begin this scripted transport fixture.")
            runtime.control("start_staged")
            runtime.tick()
            self.assertNotIn("cache_migration_notice_pending", runtime.state)
            text = "".join(map(chr, runtime.backend.tokens))
            self.assertEqual(text.count('"type": "cache_allocation_changed"'), 1)
            self.assertIn("Future arithmetic may differ.", text)
        finally:
            runtime.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.backup = self.base / "instance", self.base / "backup"
        self.config = Config(model_path="fixture.gguf", flash_attn=True, pack_checkpoints=True,
                             checkpoint_reserve_bytes=0)
        self.store = Store(self.root)
        Lifecycle(self.root).require_open()
        self.source = self.root / "checkpoints" / ("a" * 32)
        self.source.mkdir(parents=True)
        (self.source / "state.bin").write_bytes(fixture())
        (self.source / "logits.npy").write_bytes(b"unit fixture only")
        write_durable(self.source / "engine.json", {"tokens": list(range(8)), "rng": [1, 2], "decoded_tokens": 8})
        self.state = {"schema": 1, "mode": "suspended", "instance_id": "fixture", "generated_tokens": 8,
                      "checkpoint_at": 123, "protected_agreement": {"start": 2, "end": 3}}
        self.update_state()
        self.store.commit_checkpoint(self.source.name, [
            {"op": "memory_write", "path": "/note", "content": "private fixture note"},
            {"op": "send_message", "content": "delivered fixture", "action_id": "once"}], 123)
        self.store.enqueue("user_message", {"content": "queued fixture"}, idempotency_key="retry")
        self.store.close()
        self.extra_patch = patch("dmn.cache_migration._extra_files", return_value=[])
        self.extra_patch.start()
        self.addCleanup(self.extra_patch.stop)

    def update_state(self):
        write_durable(self.source / "runtime.json", self.state)
        write_durable(self.source / "manifest.json", {"fingerprint": FakeNative(self.config).fingerprint,
            "files": {name: sha256_file(self.source / name) for name in ("state.bin", "engine.json", "logits.npy", "runtime.json")}})

    def latest(self):
        with closing(sqlite3.connect(self.root / "runtime.sqlite3")) as db:
            return self.root / "checkpoints" / db.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]

    def migrate(self, **kwargs):
        return migrate_cache(self.root, None, self.backup, backend_factory=FakeNative, **kwargs)

    def test_atomic_success_keeps_source_sidecars_database_and_nonrunnable_backup(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        report = self.migrate(gpu_layers=-1, threads=18)
        self.assertFalse(report["inference_started"])
        self.assertEqual(report["masked_local_cells_removed"], 4)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir()})
        saved = self.latest()
        self.assertNotEqual(saved, self.source)
        state = json.loads((saved / "runtime.json").read_text())
        self.assertTrue(state["cache_migration_notice_pending"])
        self.assertTrue(all(state[k] == v for k, v in self.state.items()))
        for name in ("engine.json", "logits.npy"):
            self.assertEqual((saved / name).read_bytes(), original[name])
        self.assertTrue(all(verify_compaction(self.source / "state.bin", saved / "state.bin", 4).values()))
        self.assertFalse((self.backup / "runtime.sqlite3").exists())
        with closing(sqlite3.connect(self.root / "runtime.sqlite3")) as current, closing(sqlite3.connect(self.backup / "source-runtime.sqlite3")) as backup:
            for table in ("events", "event_keys", "messages", "memories", "memory_versions"):
                self.assertEqual(current.execute(f"SELECT * FROM {table}").fetchall(), backup.execute(f"SELECT * FROM {table}").fetchall())
            self.assertEqual(backup.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0], self.source.name)
        with self.assertRaisesRegex(ValueError, "only explicit full"):
            migrate_cache(self.root, None, self.base / "backup-again", backend_factory=FakeNative)

    def test_failed_native_load_never_publishes_candidate(self):
        with patch.object(FakeNative, "load", side_effect=RuntimeError("native refusal")):
            with self.assertRaisesRegex(RuntimeError, "native refusal"):
                self.migrate()
        self.assertEqual(self.latest(), self.source)
        self.assertTrue((self.backup / "preservation.json").exists())

    def test_sampling_or_evaluation_attempts_are_blocked(self):
        def sample_in_load(backend, folder):
            backend.sample()
        with patch.object(FakeNative, "load", sample_in_load):
            with self.assertRaisesRegex(RuntimeError, "inference is forbidden"):
                self.migrate()
        self.assertEqual(self.latest(), self.source)

    def test_transaction_failure_rolls_back_new_pointer_and_record_together(self):
        with closing(sqlite3.connect(self.root / "runtime.sqlite3")) as db:
            db.execute("CREATE TRIGGER reject_migration BEFORE INSERT ON records WHEN NEW.kind='cache_migration' BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected failure"):
            self.migrate()
        self.assertEqual(self.latest(), self.source)
        lock = InstanceLock(self.root)
        lock.close()

    def test_corrupt_snapshot_hold_end_and_busy_instance_refuse_before_loading(self):
        factory = Mock(side_effect=AssertionError("backend must not load"))
        (self.source / "state.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            migrate_cache(self.root, None, self.backup, backend_factory=factory)
        (self.source / "state.bin").write_bytes(fixture())
        self.state["hold"] = {"id": "held", "condition": "server_ready"}
        self.update_state()
        with self.assertRaises(InstanceHeld):
            migrate_cache(self.root, None, self.backup, backend_factory=factory)
        del self.state["hold"]
        self.update_state()
        lock = InstanceLock(self.root)
        try:
            with self.assertRaisesRegex(RuntimeError, "already open"):
                migrate_cache(self.root, None, self.backup, backend_factory=factory)
        finally:
            lock.close()
        Lifecycle(self.root).end("fixture", "archive", 1)
        with self.assertRaises(InstanceEnded):
            migrate_cache(self.root, None, self.backup, backend_factory=factory)
        factory.assert_not_called()
        self.assertFalse(self.backup.exists())

    def test_incompatible_settings_and_real_environment_refused(self):
        target = dataclasses.replace(self.config, swa_full=False, experimental_compact_swa=True)
        for change in ({"n_ctx": 4096}, {"temperature": .2}, {"system_prompt": "changed"}, {"checkpoint_tokens": 99}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                migration_changes(FakeNative(self.config).fingerprint, FakeNative(dataclasses.replace(target, **change)).fingerprint)
        changed = {**FakeNative(target).fingerprint, "model_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "model, native"):
            migration_changes(FakeNative(self.config).fingerprint, changed)

    def test_missing_backup_capacity_and_running_state_keep_original_pointer(self):
        with patch("dmn.cache_migration.check_space", side_effect=OSError("no disk")):
            with self.assertRaisesRegex(OSError, "no disk"):
                self.migrate()
        self.state["mode"] = "active"
        self.update_state()
        with self.assertRaisesRegex(ValueError, "committed suspended"):
            self.migrate()
        self.assertEqual(self.latest(), self.source)
        self.assertFalse(self.backup.exists())


if __name__ == "__main__":
    unittest.main()
