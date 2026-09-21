import copy
import tempfile
import unittest
from pathlib import Path

from dmn.backend import sha256_file
from dmn.exact_base_provenance import (AUDITS, BASE_SHA, SOURCE_REPO, SOURCE_REVISION, check_evidence)


class ExactPayloadEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'model'
        self.base.mkdir()
        for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'):
            (self.base / name).write_text(name)
        self.quantizer = self.root / 'quantizer.dll'
        self.quantizer.write_bytes(b'quantizer fixture')
        self.converter = self.root / 'converter'
        self.source = {'repo': SOURCE_REPO, 'revision': SOURCE_REVISION,
                       'files': {p.name: {'size': p.stat().st_size} for p in self.base.iterdir()}}
        request = {'source': str(self.root), 'gguf_python': str(self.converter / 'gguf-py'),
                   'full': True, 'ggml_base': str(self.quantizer)}
        common = {'completed': True, 'gguf_sha256': BASE_SHA}
        self.audits = {key: {'input.json': copy.deepcopy(request), 'result.json': dict(common)} for key in AUDITS}
        self.audits['tensor']['result.json'].update(source_revision=SOURCE_REVISION,
            full_tensor_payload_comparison=True, all_compared_bytes_equal=True, mismatches=[],
            tensor_types={'F32': 421, 'Q4_K': 355, 'Q6_K': 56}, quantizer_sha256=sha256_file(self.quantizer))
        self.audits['tokenizer']['result.json'].update(tokens_scores_types_equal=True, special_tokens_equal=True,
            vocabulary_entries=262144, chat_template_equal=False, source_chat_template_sha256='source',
            gguf_chat_template_sha256='inference', source_files={p.name: sha256_file(p) for p in self.base.iterdir()})
        self.audits['metadata']['input.json'].update(source=str(self.base), converter=str(self.converter))
        self.audits['metadata']['result.json'].update(all_inference_metadata_equal=True, keys_compared=21,
            mismatches=[], source_config_sha256=sha256_file(self.base / 'config.json'))

    def check(self):
        return check_evidence(self.audits, self.base, self.source, BASE_SHA, self.converter)

    def test_complete_evidence_preserves_template_difference(self):
        result = self.check()
        self.assertTrue(result['tensor_payloads_bit_equal'])
        self.assertFalse(result['chat_template_equal'])
        self.assertFalse(result['full_gguf_file_reproduced'])
        self.assertIn('never substitute', result['template_policy'])

    def test_sampled_failed_or_different_audits_cannot_enable_training(self):
        changes = [('tensor', 'input.json', 'full', False),
                   ('tensor', 'result.json', 'all_compared_bytes_equal', False),
                   ('tensor', 'result.json', 'source_revision', 'another revision'),
                   ('tokenizer', 'result.json', 'special_tokens_equal', False),
                   ('tokenizer', 'result.json', 'vocabulary_entries', 262143),
                   ('metadata', 'result.json', 'keys_compared', 20),
                   ('metadata', 'result.json', 'mismatches', ['rope']),
                   ('metadata', 'result.json', 'gguf_sha256', 'another model')]
        original = copy.deepcopy(self.audits)
        for kind, file, key, value in changes:
            with self.subTest(key=key):
                self.audits = copy.deepcopy(original)
                self.audits[kind][file][key] = value
                with self.assertRaises(ValueError):
                    self.check()

    def test_mutated_source_and_quantizer_rejected(self):
        for path in [*self.base.iterdir(), self.quantizer]:
            with self.subTest(file=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                try:
                    with self.assertRaises(ValueError):
                        self.check()
                finally:
                    path.write_bytes(original)

    def test_different_converter_or_source_path_rejected(self):
        self.audits['tensor']['input.json']['gguf_python'] = str(self.root / 'other')
        with self.assertRaisesRegex(ValueError, 'converter'):
            self.check()


if __name__ == '__main__':
    unittest.main()
