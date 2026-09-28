"""One anonymous HTTPS request in a killable process; no implicit retries."""
from __future__ import annotations

import email.utils
import http.client
import json
import multiprocessing
import socket
import ssl
import time
from urllib.parse import urlsplit

from .web_policy import USER_AGENT, extract, normalize_url, public_address


class HeaderReader:
    def __init__(self, stream):
        self.stream, self.remaining = stream, 65536

    def readline(self, limit=-1):
        # Covers status lines, 1xx blocks, headers and chunk trailers in aggregate.
        data = self.stream.readline(min(self.remaining + 1, limit) if limit >= 0 else self.remaining + 1)
        self.remaining -= len(data)
        if self.remaining < 0:
            raise ValueError('response headers exceed limit')
        return data

    def __getattr__(self, name):
        return getattr(self.stream, name)


class BoundedResponse(http.client.HTTPResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp = HeaderReader(self.fp)


class PublicHTTPSConnection(http.client.HTTPSConnection):
    response_class = BoundedResponse

    def connect(self):
        if self._tunnel_host:
            raise ValueError('web tunnels are unavailable')
        addresses = socket.getaddrinfo(self.host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not public_address(info[4][0]) for info in addresses):
            raise ValueError('DNS contains a non-public address')
        # Exactly one connection attempt. Connect to the checked sockaddr,
        # without another hostname resolution; preserve hostname for TLS.
        family, kind, protocol, _, address = addresses[0]
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(self.timeout)
            sock.connect(address)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def retry_after(value, now):
    if not value:
        return 0
    try:
        if value.isascii() and value.isdecimal():
            # Very long waits stay closed rather than overflow or get shortened.
            return now + int(value) if len(value) < 10 else 253402300799.0
        date = email.utils.parsedate_to_datetime(value)
        return max(now, date.timestamp()) if date.tzinfo else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def fetch_once(url, max_bytes, robots=False):
    url = normalize_url(url)
    parsed = urlsplit(url)
    connection = PublicHTTPSConnection(parsed.hostname, 443, timeout=5, context=ssl.create_default_context())
    consumed = 0
    response = None
    try:
        target = parsed.path + ('?' + parsed.query if parsed.query else '')
        connection.request('GET', target, headers={'User-Agent': USER_AGENT,
            'Accept': 'text/html,text/plain', 'Accept-Encoding': 'identity', 'Connection': 'close'})
        response = connection.getresponse()
        result = {'status': response.status, 'bytes': 0,
                  'retry_after': retry_after(response.getheader('Retry-After'), time.time())}
        # Redirects, errors and unsupported bodies are never followed/downloaded.
        if response.status != 200:
            return {**result, 'error': 'http_status'}
        if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
            return {**result, 'error': 'compressed_response'}
        mime = response.getheader('Content-Type', '').split(';')[0].strip().lower()
        if mime not in ({'text/plain'} if robots else {'text/plain', 'text/html'}):
            return {**result, 'error': 'unsupported_content_type'}
        body = bytearray()
        while len(body) < max_bytes:
            chunk = response.read(min(16384, max_bytes - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            consumed += len(chunk)
        # Do not read a byte beyond the reservation. At the exact limit the
        # source may be complete, but conservatively label it incomplete.
        truncated = len(body) == max_bytes
        result.update(bytes=consumed)
        if robots:
            return {**result, 'robots': body.decode('utf-8', errors='replace'), 'truncated': truncated}
        return {**result, 'document': extract(body, mime, url, truncated)}
    except (OSError, ValueError, http.client.HTTPException, UnicodeError):
        # Never inject an exception containing host environment or raw headers.
        # A parser can consume bytes before raising (e.g. IncompleteRead).
        # Keep the full reservation when exact consumption is unknown.
        return {'error': 'transport_failure', 'bytes': max_bytes}
    finally:
        if response is not None:
            response.close()
        connection.close()


def _worker(connection, url, max_bytes, robots):
    try:
        result = fetch_once(url, max_bytes, robots)
        connection.send_bytes(json.dumps(result, ensure_ascii=True).encode())
    finally:
        connection.close()


def fetch(url, max_bytes, robots, cancel, timeout):
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, url, max_bytes, robots), daemon=True)
    started = time.monotonic()
    try:
        process.start()
        writer.close()
        while not cancel.is_set() and time.monotonic() - started < timeout:
            if reader.poll(0.05):
                try:
                    return json.loads(reader.recv_bytes(2 * 1024 * 1024))
                except (EOFError, OSError, ValueError):
                    break
            if not process.is_alive():
                break
        # The remote effect is uncertain. Charge the entire reserved allowance.
        return {'error': 'cancelled' if cancel.is_set() else 'timeout_or_worker_failure',
                'bytes': max_bytes, 'uncertain': True}
    finally:
        writer.close()
        reader.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            process.close()
