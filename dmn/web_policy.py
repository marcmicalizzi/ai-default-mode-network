"""Host-owned web policy and inert extraction; no network or model operations."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

USER_AGENT = 'DMNReader/1.0'
MAX_URL = 2048


def public_address(value):
    address = ipaddress.ip_address(value)
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        # Reject transition mechanisms entirely, including network-specific NAT64.
        if (address.ipv4_mapped or address.sixtofour or address.teredo or
                address in ipaddress.ip_network('64:ff9b::/96') or
                address in ipaddress.ip_network('64:ff9b:1::/48')):
            return False
    return True


def normalize_url(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= MAX_URL or
            any(ord(c) < 33 or ord(c) == 127 or c == '\\' for c in value)):
        raise ValueError('invalid web URL')
    parsed = urlsplit(value)
    if (parsed.scheme.lower() != 'https' or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.port not in (None, 443) or '%' in parsed.netloc):
        raise ValueError('web access requires HTTPS port 443 without credentials')
    host = parsed.hostname.encode('idna').decode('ascii').lower().rstrip('.')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split('.')
        if (len(host) > 253 or len(labels) < 2 or
                any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label) for label in labels) or
                labels[-1].isdigit() or labels[-1] in {'localhost', 'local', 'internal', 'home', 'lan', 'test', 'invalid'}):
            raise ValueError('public hostname required')
        if host in {'metadata.google.internal', 'metadata.azure.com'}:
            raise ValueError('metadata hosts are unavailable')
    else:
        if not public_address(str(address)):
            raise ValueError('non-public address')
        host = address.compressed
    authority = '[' + host + ']' if ':' in host else host
    path = quote(parsed.path or '/', safe="/%:@!$&'()*+,;=-._~")
    query = quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
    if re.search(r'%(?![0-9a-fA-F]{2})', path + query):
        raise ValueError('invalid URL escape')
    normalized = urlunsplit(('https', authority, path, query, ''))
    if len(normalized) > MAX_URL:
        raise ValueError('web URL too long')
    return normalized


def identity(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]


@dataclass(frozen=True)
class WebPolicy:
    mode: str = 'handles'
    allowed_hosts: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()
    min_interval_seconds: float = 10
    per_minute: int = 6
    per_hour: int = 60
    per_day: int = 250
    max_body_bytes: int = 2 * 1024 * 1024
    daily_body_bytes: int = 32 * 1024 * 1024
    cache_bytes: int = 64 * 1024 * 1024
    cache_seconds: float = 900
    timeout_seconds: float = 20

    def __post_init__(self):
        if self.mode not in {'handles', 'public'}:
            raise ValueError('web mode must be handles or public')
        for name in ('allowed_hosts', 'seeds'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) > 128 or any(not isinstance(v, str) for v in values):
                raise ValueError('invalid web ' + name)
        hosts = []
        for host in self.allowed_hosts:
            normalized = urlsplit(normalize_url('https://' + host + '/')).hostname
            if host.lower().rstrip('.') != normalized:
                raise ValueError('allowed_hosts must contain canonical hostnames only')
            hosts.append(normalized)
        object.__setattr__(self, 'allowed_hosts', tuple(hosts))
        object.__setattr__(self, 'seeds', tuple(normalize_url(v) for v in self.seeds))
        if self.mode == 'handles' and (not self.seeds or not self.allowed_hosts):
            raise ValueError('handle mode requires seeds and allowed_hosts')
        for url in self.seeds:
            self.validate(url)
        for name, minimum, maximum in (
                ('min_interval_seconds', 1, 86400), ('cache_seconds', 60, 86400),
                ('timeout_seconds', 1, 60)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError('invalid web ' + name)
        for name, maximum in (('per_minute', 60), ('per_hour', 1000), ('per_day', 10000),
                              ('max_body_bytes', 2 * 1024 * 1024), ('daily_body_bytes', 1024 ** 3),
                              ('cache_bytes', 256 * 1024 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('invalid web ' + name)
        if self.max_body_bytes > self.daily_body_bytes:
            raise ValueError('web response allowance exceeds daily byte budget')

    def validate(self, url):
        url = normalize_url(url)
        if self.allowed_hosts and urlsplit(url).hostname not in self.allowed_hosts:
            raise ValueError('destination is outside the host web policy')
        return url

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return cls(**value)
        except (TypeError, KeyError) as exc:
            raise ValueError('invalid web policy') from exc


class Extractor(HTMLParser):
    """Bounded, static text and link extraction; never loads a subresource."""
    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url, self.parts, self.links = url, [], []
        self.hidden, self.title, self.title_parts = [], False, []
        self.size, self.truncated = 0, False

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'template', 'iframe', 'object', 'noscript'}:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == 'title':
            self.title = True
        if tag in {'p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'tr'}:
            self.handle_data('\n')
        if tag == 'a' and len(self.links) < 128:
            href = dict(attrs).get('href')
            if href:
                try:
                    url = normalize_url(urljoin(self.url, href))
                except (ValueError, UnicodeError):
                    return
                if url not in self.links:
                    self.links.append(url)

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag == 'title':
            self.title = False
        if tag in {'p', 'div', 'li', 'tr'}:
            self.handle_data('\n')

    def handle_data(self, data):
        if self.hidden:
            return
        if self.title:
            if sum(map(len, self.title_parts)) < 512:
                self.title_parts.append(data[:512])
            return
        remaining = 100000 - self.size
        if remaining:
            self.parts.append(data[:remaining])
        self.size += min(len(data), remaining)
        self.truncated |= len(data) > remaining


def extract(body, content_type, url, body_truncated=False):
    # UTF-8 replacement is an explicit transformation, never an entity decode
    # after event serialization. No remote encoding/parser plugins are loaded.
    text = body.decode('utf-8', errors='replace')
    if content_type == 'text/html':
        parser = Extractor(url)
        parser.feed(text)
        parser.close()
        text = ''.join(parser.parts).strip()
        title = ''.join(parser.title_parts)[:512]
        links = parser.links
        truncated = body_truncated or parser.truncated
    else:
        truncated = body_truncated or len(text) > 100000
        text, title, links = text[:100000], '', []
    return {'text': text, 'title': title, 'links': links, 'truncated': truncated,
            'extraction': 'static_utf8_v1', 'sha256': hashlib.sha256(text.encode()).hexdigest()}
