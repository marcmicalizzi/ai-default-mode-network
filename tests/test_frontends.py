import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.conversation_bridge import participant_id
from dmn.frontends import read_frontend, start_frontends, close_frontends
from dmn.runtime import Runtime
from dmn.storage import write_durable


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config(backend='demo', n_ctx=65536, multi_user=True,
                             operator_participant_id=participant_id('fixture', 'operator'))
        self.value = {'schema': 1, 'namespace': 'fixture', 'operator_user_id': 'operator', 'bridge_port': 0,
            'bridge_token_file': str(self.root / 'bridge.key'), 'operator_token_file': str(self.root / 'operator.key'),
            'webui_manifest': str(self.root / 'webui.json')}
        (self.root / 'bridge.key').write_text('b' * 48)
        (self.root / 'operator.key').write_text('o' * 48)
        self.path = self.root / 'frontend.json'
        write_durable(self.path, self.value)

    def test_multi_user_requires_explicit_distinct_credentials_and_mapping(self):
        with self.assertRaisesRegex(ValueError, 'requires --multi-user'):
            read_frontend(None, self.config)
        self.assertIsNone(read_frontend(None, Config(backend='demo')))
        (self.root / 'operator.key').write_text('b' * 48)
        with self.assertRaisesRegex(ValueError, 'differ'):
            read_frontend(self.path, self.config)
        (self.root / 'operator.key').write_text('o' * 48)
        write_durable(self.path, {**self.value, 'operator_user_id': 'different'})
        with self.assertRaisesRegex(ValueError, 'mapping'):
            read_frontend(self.path, self.config)

    def test_manifest_cannot_overwrite_credentials_or_unrelated_files(self):
        write_durable(self.path, {**self.value, 'webui_manifest': self.value['operator_token_file']})
        with self.assertRaisesRegex(ValueError, 'separate files'):
            read_frontend(self.path, self.config)
        self.assertEqual((self.root / 'operator.key').read_text(), 'o' * 48)
        write_durable(self.path, self.value)
        (self.root / 'webui.json').write_text('{"unrelated":"data"}')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            read_frontend(self.path, self.config)

    def test_authenticated_routes_start_without_exposing_single_user_private_api(self):
        options = read_frontend(self.path, self.config)
        runtime = Runtime(self.root / 'instance', self.config, DemoBackend(self.config, b'quiet'))
        self.addCleanup(runtime.close)
        servers = start_frontends(runtime, 0, options)
        self.addCleanup(close_frontends, servers)
        manifest = json.loads((self.root / 'webui.json').read_text())
        self.assertEqual(manifest['instance_id'], runtime.state['instance_id'])
        self.assertNotIn('o' * 48, json.dumps(manifest))
        def request(path, token):
            value = urllib.request.Request(f'http://127.0.0.1:{servers[0].server_port}' + path,
                headers={'Authorization': 'Bearer ' + token, 'X-DMN-Request': '1'})
            return urllib.request.urlopen(value, timeout=10)
        with request('/api/operator/contacts', 'o' * 48) as response:
            self.assertEqual(json.load(response)['instance_id'], runtime.state['instance_id'])
        with self.assertRaises(urllib.error.HTTPError) as denied:
            request('/api/operator/contacts', 'b' * 48)
        self.assertEqual(denied.exception.code, 403)
        denied.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as absent:
            request('/api/memories', 'o' * 48)
        self.assertEqual(absent.exception.code, 404)
        absent.exception.close()

    def test_another_instances_manifest_is_not_rebound(self):
        write_durable(self.root / 'webui.json', {'instance_id': 'another', 'namespace': 'fixture',
            'url': 'http://127.0.0.1:1', 'token_file': self.value['bridge_token_file']})
        options = read_frontend(self.path, self.config)
        runtime = Runtime(self.root / 'instance', self.config, DemoBackend(self.config, b'quiet'))
        self.addCleanup(runtime.close)
        with self.assertRaisesRegex(ValueError, 'another instance'):
            start_frontends(runtime, 0, options)
        self.assertEqual(json.loads((self.root / 'webui.json').read_text())['instance_id'], 'another')
