"""Offline, checkpoint-preserving handoff from the single-user WebUI adapter.

Both instance and bridge must be stopped. Historical text is never re-enqueued,
and no model contact decision is manufactured. Interrupted handoffs fail closed
until this same operation is resumed.
"""
from __future__ import annotations
from contextlib import closing
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from .backend import sha256_file
from .config import Config
from .deep_sleep import _checkpoint, refuse_pending
from .diskspace import check_space
from .ending import Lifecycle, _owned, _sync_directory
from .storage import Store, InstanceLock, write_durable, json_text
from .sleep_plans import seal
from .conversation_bridge import participant_id, conversation_id, ConversationTransport
from .conversations import Conversations
from .multi_bridge_ledger import MultiBridgeLedger

MARKER = 'conversation-migration.json'


def atomic(path, value):
    temporary = path.with_suffix('.json.partial')
    write_durable(temporary, value)
    os.replace(temporary, path)
    _sync_directory(path.parent)


def record(path):
    value = json.loads(path.read_text())
    if seal({k: v for k, v in value.items() if k != 'revision'}) != value:
        raise ValueError('conversation migration record changed')
    return value


def require_completed(root):
    path = root / MARKER
    if path.exists() and record(path)['phase'] != 'completed':
        raise ValueError('conversation migration is incomplete; resume the offline handoff before loading inference')


def _readonly(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def _inspect(root, database, namespace, operator_user_id):
    bridge = database.parent / 'dmn-bridge'
    with closing(_readonly(root / 'runtime.sqlite3')) as db, closing(_readonly(bridge / 'relay.sqlite3')) as relay, closing(_readonly(database)) as web:
        selected = db.execute('SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()
        if selected is None:
            raise ValueError('migration requires an existing stopped instance')
        source, manifest, state = _checkpoint(root, selected[0])
        config = Config(**manifest['fingerprint']['config'])
        if config.multi_user or state.get('mode') != 'suspended' or state.get('hold'):
            raise ValueError('migration requires a suspended single-user instance without a hold')
        if state['parser'].get('buffer') or state['parser'].get('in_frame'):
            raise ValueError('migration cannot cross an unfinished action frame')
        if db.execute('SELECT 1 FROM events WHERE id>?', (state['event_cursor'],)).fetchone():
            raise ValueError('deliver pending legacy events before transport migration; none will be discarded')
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='sleep_runs'").fetchone():
            if db.execute("SELECT 1 FROM sleep_runs WHERE phase!='WakeCommitted'").fetchone():
                raise ValueError('complete the pending sleep transition before transport migration')
        binding = relay.execute('SELECT * FROM binding').fetchone()
        if not binding or binding['instance_id'] != state['instance_id'] or binding['user_id'] != operator_user_id:
            raise ValueError('legacy authenticated binding does not match this instance and operator account')
        binding = dict(binding)
        if relay.execute('SELECT 1 FROM receipts WHERE event_id IS NULL').fetchone():
            raise ValueError('finish uncertain legacy input delivery before migration')
        messages = db.execute('SELECT id,content FROM messages ORDER BY id').fetchall()
        if (messages[-1]['id'] if messages else 0) != binding['cursor']:
            raise ValueError('finish all legacy output delivery before migration')
        chat = web.execute('SELECT user_id,chat,current_message_id FROM chat WHERE id=?', (binding['chat_id'],)).fetchone()
        if not chat or chat['user_id'] != operator_user_id:
            raise ValueError('legacy WebUI chat owner changed')
        saved = json.loads(chat['chat'])
        history = saved['history']
        nodes = history['messages']
        leaf = chat['current_message_id']
        if leaf != history['currentId'] or leaf not in nodes:
            raise ValueError('legacy WebUI selected branch is inconsistent')
        for message in messages:
            matches = [n for n in nodes.values() if n.get('meta', {}).get('dmn_delivery') == state['instance_id'] + ':' + str(message['id'])]
            if len(matches) != 1 or matches[0].get('content') != message['content']:
                raise ValueError('legacy output persistence could not be verified')
        # Preserve one consistent normalized view. Do not accept stale JSON that
        # omitted a new incoming message or changed a historical one.
        normalized = web.execute('SELECT id,role,content,parent_id,output,files,context_summary FROM chat_message WHERE chat_id=?', (binding['chat_id'],)).fetchall()
        if len(normalized) != len(nodes):
            raise ValueError('legacy normalized history is incomplete')
        for row in normalized:
            node = nodes.get(row['id'].removeprefix(binding['chat_id'] + '-'))
            if (node is None or row['role'] != node.get('role') or row['parent_id'] != node.get('parentId') or
                    (json.loads(row['content']) if row['content'] is not None else '') != node.get('content', '') or
                    (json.loads(row['output']) if row['output'] is not None else None) != node.get('output') or
                    ((json.loads(row['files']) if row['files'] is not None else []) or []) != (node.get('files') or []) or
                    (row['context_summary'] or '') != (node.get('contextSummary') or '')):
                raise ValueError('legacy normalized history differs')
        engine = json.loads((source / 'engine.json').read_text())
        if config.n_ctx - len(engine['tokens']) < 1024:
            raise ValueError('migration needs at least 1024 free token positions for its address notice')
        evidence = {**{k: binding[k] for k in ('instance_id', 'chat_id', 'user_id')},
                    'source_leaf_id': leaf, 'source_messages': nodes}
        plan = {'schema': 1, 'instance_id': state['instance_id'], 'source_checkpoint': source.name,
                'source_manifest_sha256': sha256_file(source / 'manifest.json'),
                'namespace': namespace, 'operator_user_id': operator_user_id,
                'conversation_id': conversation_id(namespace, binding['chat_id']),
                'participant_id': participant_id(namespace, operator_user_id), 'chat_id': binding['chat_id'],
                'outgoing_cursor': binding['cursor'], 'history_sha256': seal(evidence)['revision'],
                'checkpoint': uuid.uuid5(uuid.UUID(state['instance_id']), 'multi-user:' + namespace).hex,
                'database': str(database)}
        return plan, evidence, source, manifest, state, config


def migrate(root, database, namespace, operator_user_id, backup, *, dry_run=False, fault=lambda _: None):
    root, database, backup = [Path(p).resolve() for p in (root, database, backup)]
    if backup == root or root in backup.parents or backup in root.parents:
        raise ValueError('migration backup must be separate from the instance tree')
    # Dry run is observational and can be used while WebUI is open. Commit
    # repeats every check under both owner locks.
    if dry_run:
        plan, _, _, _, _, _ = _inspect(root, database, namespace, operator_user_id)
        return {'ready_at_observation': True, 'plan': plan, 'generation_started': False, 'source_modified': False}
    owner, bridge_owner, store, multi = InstanceLock(root), None, None, None
    try:
        Lifecycle(root).require_open()
        bridge = database.parent / 'dmn-bridge'
        bridge_owner = InstanceLock(bridge)
        marker = root / MARKER
        if marker.exists():
            progress = record(marker)
            plan = progress['plan']
            if (plan['database'] != str(database) or plan['namespace'] != namespace or
                    plan['operator_user_id'] != operator_user_id or progress['backup'] != str(backup)):
                raise ValueError('resume migration with its original paths and identity mapping')
            if progress['phase'] == 'completed':
                return {'completed': True, 'already_completed': True, 'generation_started': False}
            source, manifest, state = _checkpoint(root, plan['source_checkpoint'])
            config = Config(**manifest['fingerprint']['config'])
            if sha256_file(source / 'manifest.json') != plan['source_manifest_sha256']:
                raise ValueError('migration source checkpoint changed')
            evidence = json.loads((backup / 'legacy-history.json').read_text())
        else:
            plan, evidence, source, manifest, state, config = _inspect(root, database, namespace, operator_user_id)
            if backup.exists():
                raise ValueError('use a new migration backup directory')
            size = sum((source / n).stat().st_size for n in manifest['files'])
            check_space(root, size * 2 + 256 * 1024**2, config.checkpoint_reserve_bytes, 'conversation migration')
            backup.mkdir(parents=True)
            shutil.copytree(source, backup / 'checkpoint')
            # SQLite backup APIs include committed WAL contents.
            for src, target in ((root / 'runtime.sqlite3', 'runtime.sqlite3'),
                                (bridge / 'relay.sqlite3', 'relay.sqlite3')):
                srcdb = _readonly(src)
                destdb = sqlite3.connect(backup / target)
                try:
                    srcdb.backup(destdb)
                finally:
                    destdb.close()
                    srcdb.close()
            write_durable(backup / 'legacy-history.json', evidence)
            progress = seal({'phase': 'prepared', 'plan': plan, 'backup': str(backup)})
            atomic(marker, progress)
            fault('prepared')
        if seal(evidence)['revision'] != plan['history_sha256']:
            raise ValueError('migration history evidence changed')
        # Presence blocks the old relay even if a crash occurs before runtime
        # publication. The new relay accepts only the completed marker.
        atomic(bridge / MARKER, progress)
        write_durable(bridge / 'legacy-history.json', evidence)
        store = Store(root)
        refuse_pending(store)
        target_config = dataclasses.replace(config, multi_user=True, require_contact_consent=True,
                                             operator_participant_id=plan['participant_id'])
        conversations = Conversations(store, plan['participant_id'], config.max_pending_messages,
                                      config.max_pending_messages_per_participant, True)
        conversations.register(plan['participant_id'], 'Existing WebUI counterpart', plan['conversation_id'])
        # Initialize the same transport schema without creating inference or
        # invoking model actions. Immutable ownership is still enforced.
        from types import SimpleNamespace
        transport_runtime = SimpleNamespace(config=target_config, state=state, store=store)
        ConversationTransport(transport_runtime, namespace, operator_user_id)
        with store.transaction() as db:
            db.execute('INSERT OR IGNORE INTO webui_origins VALUES(?,?,?)',
                       (plan['chat_id'], operator_user_id, plan['conversation_id']))
            db.execute('INSERT OR IGNORE INTO message_destinations SELECT id,?,?,NULL FROM messages',
                       (plan['conversation_id'], plan['participant_id']))
        multi = MultiBridgeLedger(bridge / 'multi-relay.sqlite3', plan['instance_id'], namespace)
        multi.bind(plan['chat_id'], operator_user_id, plan['conversation_id'], plan['participant_id'])
        multi.advance(plan['chat_id'], plan['outgoing_cursor'])
        destination = root / 'checkpoints' / plan['checkpoint']
        if not destination.exists():
            destination.mkdir()
        _owned(destination, destination.parent)
        for name in manifest['files']:
            if name != 'runtime.json':
                shutil.copyfile(source / name, destination / name)
        new_state = {**state, 'conversation_migration': {k: plan[k] for k in
            ('source_checkpoint', 'namespace', 'conversation_id', 'participant_id', 'chat_id')},
            'mode': 'awaiting_first_contact', 'first_contact_gate': plan['participant_id'],
            'checkpoint_at': time.time(), 'checkpoint_reason': 'conversation_migration'}
        write_durable(destination / 'runtime.json', new_state)
        fingerprint = {**manifest['fingerprint'], 'config': target_config.to_dict()}
        files = {name: sha256_file(destination / name) for name in manifest['files']}
        for name in files:
            with (destination / name).open('r+b') as stream:
                os.fsync(stream.fileno())
        write_durable(destination / 'manifest.json', {'fingerprint': fingerprint, 'files': files})
        _sync_directory(destination)
        fault('files_prepared')
        with store.transaction() as db:
            selected = store.latest().name
            if selected not in {source.name, destination.name}:
                raise ValueError('selected checkpoint changed during migration')
            if selected != destination.name:
                db.execute('INSERT INTO checkpoints(directory,created) VALUES(?,?)', (destination.name, time.time()))
                db.execute("INSERT INTO records(kind,payload,created) VALUES('conversation_migration',?,?)",
                           (json_text(plan), time.time()))
        fault('checkpoint_committed')
        completed = seal({'phase': 'completed', 'plan': plan, 'backup': str(backup)})
        atomic(bridge / MARKER, completed)
        atomic(marker, completed)
        return {'completed': True, 'native_files_unchanged': True, 'historical_input_replayed': False,
                'contact_approval_supplied': False, 'generation_started': False,
                'conversation_id': plan['conversation_id'], 'backup': str(backup)}
    finally:
        if multi:
            multi.close()
        if store:
            store.close()
        if bridge_owner:
            bridge_owner.close()
        owner.close()
