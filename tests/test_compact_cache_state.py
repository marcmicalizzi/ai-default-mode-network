"""Boundaries for the experimental codec; no native binding or model needed."""
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.compact_cache_state import compact_state, inspect_state, validate_retirements
from scripts.probe_compact_cache import compare_conversion


def fixture(local_positions=(5, 6, 7, 0, 1, 2, 3, 4)):
    data = struct.pack("<III8iI", 0x6767736E, 9, 8, *range(8), 6) + b"gemma4"
    for positions, layers in ((range(8), 1), (local_positions, 2)):
        data += struct.pack("<II", 1, len(positions))
        data += b"".join(struct.pack("<iIi", p, 1, 0) for p in positions)
        data += struct.pack("<II", 0, layers)
        for layer in range(2 * layers):
            data += struct.pack("<iQ", 1, 4)
            data += b"".join(bytes([p, layer, 42, 99]) for p in positions)
    return data


class CompactStateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / "source.bin", self.root / "compact.bin"
        self.source.write_bytes(fixture())

    def test_conversion_retains_global_and_exact_local_boundary_in_physical_order(self):
        before = self.source.read_bytes()
        report = compact_state(self.source, self.target, 4)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(inspect_state(self.target).local_cache.positions, (5, 6, 7, 4))
        self.assertEqual(report["masked_local_cells_removed"], 4)
        self.assertTrue(all(compare_conversion(self.source, self.target).values()))
        original, changed = inspect_state(self.source), inspect_state(self.target)
        self.assertEqual(before[:original.local_cache.start], self.target.read_bytes()[:changed.local_cache.start])

    def test_short_context_and_already_compact_are_lossless(self):
        compact_state(self.source, self.target, 16)
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())
        self.target.unlink()
        self.source.write_bytes(fixture((5, 6, 7, 4)))
        compact_state(self.source, self.target, 4)
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())

    def test_missing_recent_row_and_overwrite_fail_before_output(self):
        self.source.write_bytes(fixture((4, 6, 7)))
        with self.assertRaisesRegex(ValueError, "missing"):
            compact_state(self.source, self.target, 4)
        self.assertFalse(self.target.exists())
        self.source.write_bytes(fixture())
        with self.assertRaises(FileExistsError):
            compact_state(self.source, self.source, 4)
        self.assertEqual(self.source.read_bytes(), fixture())

    def test_parser_rejects_unknown_or_malformed_native_layouts(self):
        original = self.source.read_bytes()
        state = inspect_state(self.source)
        local = state.local_cache
        data_header = local.start + 8 + 12 * len(local.positions)
        alterations = [(0, "I", 0x67677371), (4, "I", 10), (8, "I", 0xFFFFFFFF),
                       (local.start, "I", 2), (local.start + 4, "I", 99),
                       (local.start + 12, "I", 2), (local.start + 16, "i", 1),
                       (data_header, "I", 1), (data_header + 4, "I", 0),
                       (data_header + 8, "i", 99), (data_header + 12, "Q", 3)]
        for offset, fmt, value in alterations:
            with self.subTest(offset=offset, value=value):
                data = bytearray(original)
                struct.pack_into("<" + fmt, data, offset, value)
                self.source.write_bytes(data)
                with self.assertRaises(ValueError):
                    inspect_state(self.source)
        for data in (original[:-1], original + b"trailing", original.replace(b"gemma4", b"llamaX"),
                     fixture((4, 5, 6, 6))):
            self.source.write_bytes(data)
            with self.assertRaises(ValueError):
                inspect_state(self.source)

    def test_retirement_window_boundaries_and_compound_preflight(self):
        validate_retirements(1024, [(16, 944)], 64)  # exactly the recent window remains
        validate_retirements(1024, [(16, 943)], 64)
        for plan in ([(16, 945)], [(960, 1)], [(0, 0)], [(1, -1)],
                     [(16, 128), (800, 33)]):  # last range becomes unsafe after first removal
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_retirements(1024, plan, 64)


if __name__ == "__main__":
    unittest.main()
