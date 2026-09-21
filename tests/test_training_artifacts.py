import tempfile
import unittest
from pathlib import Path
from dmn.training_artifacts import write_bytes


class ArtifactBudgetTests(unittest.TestCase):
    def test_oversize_refused_before_creation_and_existing_files_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adapter.bin'
            with self.assertRaisesRegex(ValueError, 'allowance'):
                write_bytes(path, b'12345', 4)
            self.assertFalse(path.exists())
            write_bytes(path, b'1234', 4)
            with self.assertRaises(FileExistsError):
                write_bytes(path, b'xx', 4)
            self.assertEqual(path.read_bytes(), b'1234')
