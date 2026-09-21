"""Keep operator and WebUI transport credentials distinct within a runtime."""
import hashlib


def register_key(runtime, role, token):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError(f"{role} token must contain at least 32 non-whitespace ASCII characters")
    digest = hashlib.sha256(token.encode()).digest()
    with runtime._control_lock:
        keys = getattr(runtime, "_transport_key_hashes", {})
        if any(other != role and value == digest for other, value in keys.items()):
            raise ValueError("operator and WebUI bridge credentials must differ")
        runtime._transport_key_hashes = {**keys, role: digest}
