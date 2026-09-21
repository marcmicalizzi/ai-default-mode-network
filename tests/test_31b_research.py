"""The large-model experiments stay explicit and their completion checks fail closed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dmn.backend import sha256_file
from scripts.probe_qlora_31b import inspect
from scripts.convert_qlora_31b_probe import validate


class ResearchEnvelopeTest(unittest.TestCase):
    def test_out_of_bounds_workloads_reject_before_asset_access(self):
        for length in (True, 16, 257, 2048, '256', 256.0):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, 'workload length'):
                inspect(Path('not-accessed'), sequence_tokens=length)
        for steps in (True, 0, 5, '2', 2.0):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, 'one to four'):
                inspect(Path('not-accessed'), steps=steps)

    def test_conversion_rejects_partial_or_changed_completed_adapter(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'adapter').mkdir()
            adapter = root / 'adapter/adapter_model.safetensors'
            adapter.write_bytes(b'synthetic digest fixture, not actual tensors')
            (root / 'adapter/adapter_config.json').write_text(json.dumps(
                {'r': 2, 'lora_alpha': 4, 'bias': 'none', 'target_modules': ['q', 'o']}))
            plan = {'source': str(root), 'training': {'sequence_tokens': 256, 'steps': 2},
                    'synthetic_text_only': True, 'profile': {'target_modules': ['q', 'o']}}
            result = {'completed': True, 'steps_completed': 2, 'plan': plan, 'adapter_sha256': sha256_file(adapter)}
            def write(value):
                (root / 'result.json').write_text(json.dumps(value))
            with patch('scripts.convert_qlora_31b_probe.inspect', return_value=plan):
                write(result)
                self.assertEqual(validate(root), plan)
                for steps in (True, 1, 0, 5):
                    write({**result, 'steps_completed': steps})
                    with self.subTest(steps=steps), self.assertRaises(ValueError):
                        validate(root)
                write({**result, 'completed': False})
                with self.assertRaises(ValueError):
                    validate(root)
                write(result)
                adapter.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'adapter differs'):
                    validate(root)


if __name__ == '__main__':
    unittest.main()
