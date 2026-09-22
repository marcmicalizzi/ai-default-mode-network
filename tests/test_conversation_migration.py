import dataclasses
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from dmn.backend import DemoBackend
from dmn.bridge import BridgeLedger, attach_message
from dmn.config import Config
from dmn.conversation_migration import migrate, require_completed, MARKER
from dmn.conversation_bridge import conversation_id
from dmn.multi_bridge_ledger import MultiBridgeLedger
from dmn.runtime import Runtime
from dmn.storage import Store, InstanceLock, json_text
from tests.test_runtime import frames


class Crash(BaseException):
    pass


class ConversationMigrationTests(unittest.TestCase):
    def setup_backend(self):
        config = Config(backend='demo', n_ctx=65536, clock_interval_seconds=0, checkpoint_policy='effects')
        return config, lambda cfg: DemoBackend(cfg, b'quiet ')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.database, self.backup = self.base/'instance', self.base/'webui.db', self.base/'backup'
        self.config, self.factory = self.setup_backend()
        r = Runtime(self.root, self.config, self.factory(self.config))
        with mock.patch.object(r.backend, 'piece', return_value=frames({'op':'send_message','content':'Published synthetic greeting.'})):
            r._generate_one()
        r.state.update(mode='suspended', mode_before_suspend='active')
        r.checkpoint(reason='test_maintenance')
        self.source = r.store.latest()
        self.instance_id = r.state['instance_id']
        self.messages = r.store.messages()
        r.close()
        bridge = self.database.parent/'dmn-bridge'
        bridge.mkdir()
        ledger = BridgeLedger(bridge/'relay.sqlite3')
        ledger.bind(self.instance_id, 'old-chat', 'operator')
        ledger.advance(self.messages[-1]['id'])
        ledger.close()
        chat = {'history': {'messages': {'user-1': {'id':'user-1','role':'user','content':'Old synthetic input.',
            'parentId':None,'childrenIds':[]}}, 'currentId':'user-1'}}
        for message in self.messages:
            chat, _, _ = attach_message(chat, self.instance_id, message)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript('''CREATE TABLE chat(id TEXT PRIMARY KEY,user_id TEXT,chat TEXT,current_message_id TEXT);
                CREATE TABLE chat_message(id TEXT PRIMARY KEY,chat_id TEXT,role TEXT,content TEXT,parent_id TEXT,
                    output TEXT,files TEXT,context_summary TEXT);''')
            db.execute('INSERT INTO chat VALUES(?,?,?,?)', ('old-chat','operator',json_text(chat),chat['history']['currentId']))
            for node in chat['history']['messages'].values():
                db.execute('INSERT INTO chat_message VALUES(?,?,?,?,?,?,?,?)', ('old-chat-'+node['id'],'old-chat',node['role'],
                    json_text(node['content']),node['parentId'],'null','null',None))

    def migrate(self, **kwargs):
        return migrate(self.root, self.database, 'webui-test', 'operator', self.backup, **kwargs)

    def test_offline_handoff_preserves_native_sidecars_outbox_and_no_contact_approval(self):
        self.assertTrue(self.migrate(dry_run=True)['ready_at_observation'])
        self.assertFalse(self.backup.exists())
        result = self.migrate()
        self.assertTrue(result['completed'])
        store = Store(self.root)
        destination = store.latest()
        manifest = json.loads((destination/'manifest.json').read_text())
        self.assertEqual((destination/'engine.json').read_bytes(), (self.source/'engine.json').read_bytes())
        self.assertEqual([{k: row[k] for k in self.messages[0]} for row in store.messages()], self.messages)
        self.assertEqual(store.db.execute('SELECT count(*) FROM contact_requests').fetchone()[0], 0)
        self.assertEqual(store.db.execute('SELECT count(*) FROM events').fetchone()[0], 0)
        self.assertEqual(len(store.messages(conversation_id=conversation_id('webui-test','old-chat'))), 1)
        store.close()
        config = Config(**manifest['fingerprint']['config'])
        r = Runtime(self.root, config, self.factory(config))
        try:
            self.assertEqual(r.state['last_restore']['prompt_tokens_reevaluated'], 0)
            self.assertEqual(r.state['mode'], 'awaiting_first_contact')
            self.assertNotIn('protected_conversations', r.state)
            before = r.backend.tokens.copy()
            generated = r.state['generated_tokens']
            for _ in range(3):
                self.assertFalse(r.tick())
            self.assertEqual(r.backend.tokens, before)
            self.assertEqual(r.state['generated_tokens'], generated)
            with self.assertRaisesRegex(ValueError, 'promised first contact'):
                r.register_conversation('other-person', 'Someone else', 'other-chat')
            person = r.conversations.read(result['conversation_id'])
            self.assertEqual(person['contact_state'], 'unrequested')
            event = r.enqueue_conversation(result['conversation_id'], 'New synthetic request.')
            self.assertEqual(r.store.next_event(event-1)['kind'], 'contact_request')
            r.tick()
            self.assertEqual(r.state['conversation_protocol'], 'addressed_v1')
            self.assertIn('protected_conversations', r.state)
            self.assertEqual(r.state['generated_tokens'], generated)
            self.assertEqual(r.state['event_cursor'], event)
            self.assertTrue(r.event_delivered(event))
            text = b''.join(r.backend.piece(token) for token in r.backend.tokens).decode(errors='replace')
            self.assertNotIn('New synthetic request.', text)
            self.assertEqual(r.state['instance_id'], self.instance_id)
        finally:
            r.close()
        self.assertTrue(self.migrate()['already_completed'])

    def test_live_relay_or_pending_input_refuses_before_mutation(self):
        lock = InstanceLock(self.database.parent/'dmn-bridge')
        try:
            with self.assertRaisesRegex(RuntimeError, 'already open'):
                self.migrate()
        finally:
            lock.close()
        self.assertFalse(self.backup.exists())
        store = Store(self.root)
        store.enqueue('user_message', {'content':'Undelivered synthetic input.'})
        store.close()
        with self.assertRaisesRegex(ValueError, 'pending legacy'):
            self.migrate()
        self.assertFalse(self.backup.exists())

    def test_first_contact_wait_survives_recipe_offer_shutdown_and_restart(self):
        result = self.migrate()
        with closing(Store(self.root)) as store:
            manifest = json.loads((store.latest()/'manifest.json').read_text())
        config = Config(**manifest['fingerprint']['config'])
        for _ in range(2):
            r = Runtime(self.root, config, self.factory(config))
            try:
                before = r.backend.tokens.copy()
                with mock.patch('dmn.sleep_plans.put_recipe', return_value={'revision':'a'*64, 'kind':'test'}):
                    r.offer_learning_recipe({})
                self.assertIsNone(r.store.next_event(r.state['event_cursor']))
                self.assertFalse(r.tick())
                with self.assertRaisesRegex(ValueError, 'promised first contact'):
                    r.control('resume')
                stopped = r.control('shutdown', reason='Synthetic maintenance before first contact')
                self.assertFalse(stopped['inference_started'])
                self.assertEqual(r.backend.tokens, before)
                self.assertEqual(r.state['first_contact_gate'], config.operator_participant_id)
            finally:
                r.close()

    def test_crash_after_publication_requires_handoff_resume_without_replay(self):
        def die(phase):
            if phase == 'checkpoint_committed':
                raise Crash()
        with self.assertRaises(Crash):
            self.migrate(fault=die)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            require_completed(self.root)
        result = self.migrate()
        self.assertTrue(result['completed'])
        require_completed(self.root)
        store = Store(self.root)
        self.assertEqual(store.db.execute("SELECT count(*) FROM records WHERE kind='conversation_migration'").fetchone()[0], 1)
        store.close()


@unittest.skipUnless(os.environ.get('DMN_TEST_MIGRATION_MODEL'), 'explicit tiny native migration fixture required')
class NativeConversationMigrationTests(ConversationMigrationTests):
    def setup_backend(self):
        from dmn.backend import make_backend
        model = Path(os.environ['DMN_TEST_MIGRATION_MODEL']).resolve()
        if model.stat().st_size > 4 * 1024**2:
            raise ValueError('migration tests only permit generated tiny models')
        config = Config(model_path=str(model), n_ctx=65536, n_gpu_layers=0,
                        offload_kqv=False, n_threads=2, prompt_format='plain',
                        clock_interval_seconds=0, checkpoint_policy='effects')
        return config, make_backend


if __name__ == '__main__':
    unittest.main()
