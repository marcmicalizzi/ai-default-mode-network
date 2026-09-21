"""Explicit loopback frontend selection, with separate multi-user credentials."""
from __future__ import annotations
import json
from pathlib import Path

from .conversation_bridge import participant_id, serve_bridge
from .operator_server import serve_operator
from .server import serve
from .storage import write_durable


def _json(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('frontend configuration is too large')
    return json.loads(path.read_text(encoding='utf-8'))


def read_frontend(path, config):
    """Preflight before allocating a native model; never log credential values."""
    if path is None:
        if config.multi_user:
            raise ValueError('multi-user mode requires --multi-user-frontend with separate authenticated bridge/operator keys')
        return None
    if not config.multi_user:
        raise ValueError('--multi-user-frontend requires multi_user=true; existing-instance migration is separate')
    value = _json(path)
    fields = {'schema', 'namespace', 'operator_user_id', 'bridge_port', 'bridge_token_file',
              'operator_token_file', 'webui_manifest'}
    if not isinstance(value, dict) or set(value) != fields or value['schema'] != 1:
        raise ValueError('invalid multi-user frontend configuration fields')
    if participant_id(value['namespace'], value['operator_user_id']) != config.operator_participant_id:
        raise ValueError('frontend operator mapping differs from the runtime configuration')
    if type(value['bridge_port']) is not int or not 0 <= value['bridge_port'] <= 65535:
        raise ValueError('invalid bridge port')
    for name in ('bridge_token_file', 'operator_token_file', 'webui_manifest'):
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute():
            raise ValueError('frontend file paths must be absolute')
    paths = [Path(value[name]).resolve() for name in ('bridge_token_file', 'operator_token_file', 'webui_manifest')]
    if len(set(paths)) != 3:
        raise ValueError('credentials and bridge manifest must use separate files')
    tokens = {}
    for role in ('bridge', 'operator'):
        token_path = Path(value[role + '_token_file'])
        if token_path.stat().st_size > 4096:
            raise ValueError('oversized frontend credential file')
        token = token_path.read_text(encoding='utf-8').strip()
        if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError('frontend keys require at least 32 non-whitespace ASCII characters')
        tokens[role] = token
    if tokens['bridge'] == tokens['operator']:
        raise ValueError('operator and WebUI bridge credentials must differ')
    manifest = paths[-1]
    if not manifest.parent.is_dir():
        raise ValueError('bridge manifest parent directory must already exist')
    if manifest.exists():
        old = _json(manifest)
        if (not isinstance(old, dict) or set(old) != {'url', 'instance_id', 'namespace', 'token_file'} or
                old['namespace'] != value['namespace'] or Path(old['token_file']).resolve() != paths[0]):
            raise ValueError('existing output is not this frontend bridge manifest; refusing overwrite')
    return {**value, '_tokens': tokens}


def start_frontends(runtime, port, options=None):
    if options is None:
        return [serve(runtime, port)]
    manifest = Path(options['webui_manifest'])
    if manifest.exists() and _json(manifest)['instance_id'] != runtime.state['instance_id']:
        raise ValueError('bridge manifest belongs to another instance; use a separate output file')
    servers = []
    try:
        servers.append(serve_operator(runtime, token=options['_tokens']['operator'], port=port))
        servers.append(serve_bridge(runtime, token=options['_tokens']['bridge'], namespace=options['namespace'],
                                    operator_user_id=options['operator_user_id'], port=options['bridge_port']))
        write_durable(manifest, {'url': f'http://127.0.0.1:{servers[1].server_port}',
            'instance_id': runtime.state['instance_id'], 'namespace': options['namespace'],
            'token_file': options['bridge_token_file']})
        return servers
    except BaseException:
        close_frontends(servers)
        raise


def close_frontends(servers):
    for server in reversed(servers):
        server.shutdown()
        server.server_close()
