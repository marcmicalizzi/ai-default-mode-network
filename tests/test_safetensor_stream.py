import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.safetensor_stream import read_header, state_dict


class StreamingSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.file = self.root / 'model.safetensors'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, header=None, payload=b'\0\0\x80?\0\0\0@'):
        raw = json.dumps(header or {'weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
        self.file.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)

    def test_inspection_reads_only_headers_without_torch(self):
        self.write()
        with patch.dict(sys.modules, {'torch': None}):
            tensors = state_dict(self.root)
        self.assertEqual(tensors['weight'].get_shape(), [2])
        self.assertEqual(tensors['weight'].get_dtype(), 'F32')
        with self.assertRaisesRegex(ValueError, 'whole tensors'):
            tensors['weight'][0]

    def test_bounds_offsets_and_mutation_are_rejected(self):
        self.write()
        with self.assertRaisesRegex(ValueError, 'allocation limit'):
            read_header(self.file, max_tensor_bytes=4)
        tensors = read_header(self.file)
        self.file.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            tensors['weight'][...]
        for entry in ({'dtype': 'F32', 'shape': [2], 'data_offsets': [1, 9]},
                      {'dtype': 'F32', 'shape': [3], 'data_offsets': [0, 8]},
                      {'dtype': 'F32', 'shape': [True], 'data_offsets': [0, 4]}):
            self.write({'weight': entry})
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                read_header(self.file)
        self.write({'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
                    'b': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}})
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            read_header(self.file)

    def test_huge_header_duplicate_keys_and_shard_mismatch_are_rejected(self):
        self.file.write_bytes(struct.pack('<Q', 2**63))
        with self.assertRaisesRegex(ValueError, 'header'):
            read_header(self.file)
        raw = b'{"weight":{},"weight":{}}'
        self.file.write_bytes(struct.pack('<Q', len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_header(self.file)
        self.write()
        index = self.root / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'wrong': self.file.name}}))
        with self.assertRaisesRegex(ValueError, 'differ from index'):
            state_dict(self.root)
        index.write_text(json.dumps({'weight_map': {'weight': '../outside.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'shard path'):
            state_dict(self.root)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional source tensor materialization requires Torch')
    def test_float_bfloat16_scalar_and_empty_values_are_exact(self):
        import torch
        for dtype, label in ((torch.float32, 'F32'), (torch.bfloat16, 'BF16')):
            for shape in ((2, 2), (), (0, 3)):
                value = torch.full(shape, 1.5, dtype=dtype)
                raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                self.write({'weight': {'dtype': label, 'shape': list(shape), 'data_offsets': [0, len(raw)]}}, raw)
                actual = read_header(self.file)['weight'][...]
                self.assertTrue(torch.equal(actual, value))
                self.assertEqual(actual.dtype, value.dtype)


if __name__ == '__main__':
    unittest.main()
