import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from dmn.storage import Store


class MemoryVersionsTest(unittest.TestCase):
    def test_revision_conflict_rolls_back_all_effects_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            try:
                store.commit_checkpoint("first", [{"op": "memory_write", "path": "/fact", "content": "original", "expected_revision": 0}], 1)
                with self.assertRaisesRegex(ValueError, "revision changed"):
                    store.commit_checkpoint("failed", [
                        {"op": "memory_write", "path": "/other", "content": "uncommitted", "expected_revision": 0},
                        {"op": "memory_write", "path": "/fact", "content": "stale", "expected_revision": 0}], 2)
                self.assertEqual(store.latest().name, "first")
                self.assertEqual(store.memory_list(), ["/fact"])
                self.assertEqual(store.memory_history("/other"), [])
                self.assertEqual(store.memory_read("/fact"), "original")
            finally:
                store.close()

    def test_history_survives_move_delete_recreation_and_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root)
            try:
                store.commit_checkpoint("one", [{"op": "memory_write", "path": "/a", "content": "first"}], 1)
                store.commit_checkpoint("two", [{"op": "memory_write", "path": "/a", "content": "second", "expected_revision": 1}], 2)
                store.commit_checkpoint("three", [{"op": "memory_move", "path": "/a", "destination": "/b", "expected_revision": 2}], 3)
                store.commit_checkpoint("four", [{"op": "memory_delete", "path": "/b", "expected_revision": 1}], 4)
                store.commit_checkpoint("five", [{"op": "memory_write", "path": "/a", "content": "new", "expected_revision": 0}], 5)
            finally:
                store.close()
            store = Store(root)
            try:
                self.assertEqual(store.memory_revision("/a"), 4)
                self.assertEqual(store.memory_read("/a", 1), "first")
                self.assertEqual(store.memory_read("/a", 2), "second")
                self.assertEqual(store.memory_read("/b", 1), "second")
                self.assertEqual(store.memory_list(), ["/a"])
                with self.assertRaisesRegex(ValueError, "removal"):
                    store.memory_read("/b", 2)
            finally:
                store.close()

    def test_legacy_current_memory_is_archived_once_without_changing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with closing(sqlite3.connect(root / "runtime.sqlite3")) as db:
                db.execute("CREATE TABLE memories(path TEXT PRIMARY KEY,content TEXT NOT NULL,updated REAL NOT NULL)")
                db.execute("INSERT INTO memories VALUES('/old','existing value',123)")
                db.commit()
            for _ in range(2):
                store = Store(root)
                try:
                    self.assertEqual(store.memory_read("/old"), "existing value")
                    self.assertEqual(store.memory_revision("/old"), 1)
                    self.assertEqual(len(store.memory_history("/old")), 1)
                finally:
                    store.close()
