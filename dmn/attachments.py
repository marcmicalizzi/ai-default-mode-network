"""Consent and ephemeral uploads. No image payload is ever written to SQLite."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import warnings
from dataclasses import dataclass

from .storage import json_text

CONTRACT_VERSION = "ephemeral_images_v1"
CONTRACT = '''Images are optional external input. Receipt is disabled until you choose
image_permission(scope="global", decision="allow", accept_ephemeral=true).
You may decline or revoke with decision="deny", without explanation. Silence
does not approve. Host requests are text only; no image is queued before approval.
Raw uploads are held only in bounded process memory until delivery, revocation,
expiry or restart. event_read can retrieve text and metadata, never the image.
Visual positions are not text token IDs. Native checkpoints may preserve their
active KV and its influence; retirement removes direct access. Raw pixels and
embeddings are not archived for re-opening or replay. Only text token IDs support
long-term replay. Reconstruction currently refuses while visual positions remain;
it never silently pretends to restore an image. You may write your own textual
observations to memory. Revocation stops future delivery, not past influence.
image_permission_status(): inspect permissions. image_permission can also use
scope="participant", participant_id=<trusted ID>, decision="allow" or "deny".
Global denial overrides every participant; otherwise participants inherit global
permission unless individually denied. Reallowing does not revive dropped images.
These are optional capabilities, not a request to accept or attend to images.'''

OPERATIONS = {"image_permission", "image_permission_status"}
LOCAL_PARTICIPANT = "local-user"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 16 * 1024 * 1024
MAX_IMAGES = 4
MAX_PENDING_BYTES = 16 * 1024 * 1024
MAX_PENDING_EVENTS = 32
TTL_SECONDS = 600


class ImagePermissionRequired(ValueError):
    pass


class ImageInputError(ValueError):
    """Rejected preprocessing before native decode; safe to deliver text only."""


@dataclass(frozen=True)
class ImageUpload:
    data: bytes
    media_type: str
    width: int
    height: int

    def metadata(self):
        return {"media_type": self.media_type, "bytes": len(self.data),
                "width": self.width, "height": self.height,
                "sha256": hashlib.sha256(self.data).hexdigest(), "availability": "ephemeral"}


def decode_uploads(values):
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_IMAGES:
        raise ValueError(f"images must contain 1 to {MAX_IMAGES} uploads")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("image uploads require the optional vision dependencies") from exc
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"media_type", "data_base64"}:
            raise ValueError("each image requires only media_type and data_base64; URLs and paths are not accepted")
        mime, encoded = value["media_type"], value["data_base64"]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("supported image types: PNG, JPEG, WebP")
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("image exceeds upload limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("image data must be strict base64") from exc
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image is empty or exceeds upload limit")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as picture:
                    width, height = picture.size
                    if (Image.MIME.get(picture.format) != mime or width * height > MAX_PIXELS
                            or min(width, height) < 1 or getattr(picture, "n_frames", 1) != 1):
                        raise ValueError("image type/dimensions invalid or animated input unsupported")
                    picture.verify()
        except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError("invalid or oversized image") from exc
        result.append(ImageUpload(data, mime, width, height))
    return result


class ImagePermissions:
    def __init__(self, store):
        self.store = store
        with store.transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS image_permissions (
                scope TEXT NOT NULL, participant_id TEXT NOT NULL, allowed INTEGER NOT NULL,
                revision INTEGER NOT NULL, PRIMARY KEY(scope, participant_id))''')

    def status(self):
        with self.store.mutex:
            rows = [dict(row) for row in self.store.db.execute("SELECT * FROM image_permissions ORDER BY scope, participant_id")]
        return {"contract": CONTRACT_VERSION, "global_allowed": any(
            row["scope"] == "global" and row["allowed"] for row in rows), "rules": rows}

    def ticket(self, participant_id):
        status = self.status()
        if not status["global_allowed"]:
            raise ImagePermissionRequired("The instance has not allowed ephemeral images globally")
        rules = [row for row in status["rules"] if row["scope"] == "global" or row["participant_id"] == participant_id]
        if any(not row["allowed"] for row in rules):
            raise ImagePermissionRequired("The instance has denied images from this participant")
        return [[row["scope"], row["revision"]] for row in rules]

    def permits(self, participant_id, ticket):
        try:
            return self.ticket(participant_id) == ticket
        except ImagePermissionRequired:
            return False


def plan_action(runtime, action):
    op = action["op"]
    if op == "image_permission_status":
        return {"op": op, "ok": True, **runtime.image_permissions.status()}, None
    scope, decision = action.get("scope"), action.get("decision")
    if scope not in {"global", "participant"} or decision not in {"allow", "deny"}:
        raise ValueError("image_permission needs scope global/participant and decision allow/deny")
    participant = action.get("participant_id", "")
    if scope == "global":
        if participant:
            raise ValueError("global permission cannot specify a participant")
    elif not isinstance(participant, str) or not participant or len(participant) > 256:
        raise ValueError("participant permission requires a trusted participant_id")
    if decision == "allow" and action.get("accept_ephemeral") is not True:
        raise ValueError("allow requires accept_ephemeral=true after considering the image contract")
    if decision == "allow" and runtime.state.get("image_protocol") != CONTRACT_VERSION:
        raise ValueError("the complete image contract must be delivered before approval")
    effect = {"op": op, "scope": scope, "participant_id": participant, "allowed": decision == "allow"}
    return {"op": op, "ok": True, "scope": scope, "participant_id": participant,
            "decision": decision, "effective": "after_checkpoint"}, effect


def commit_effect(db, effect, now):
    revision = db.execute("SELECT COALESCE(MAX(revision),0)+1 FROM image_permissions").fetchone()[0]
    db.execute('''INSERT INTO image_permissions VALUES(?,?,?,?) ON CONFLICT(scope,participant_id)
        DO UPDATE SET allowed=excluded.allowed,revision=excluded.revision''',
        (effect["scope"], effect["participant_id"], effect["allowed"], revision))
    db.execute("INSERT INTO records(kind,payload,created) VALUES(?,?,?)",
               ("image_permission", json_text({**effect, "revision": revision}), now))


class EphemeralImages:
    """Access under Runtime._control_lock. Expiry uses a monotonic clock."""
    def __init__(self, now):
        self.now, self.entries = now, {}

    def prune(self, permissions):
        for event_id, entry in list(self.entries.items()):
            if self.now() >= entry["expires"] or not permissions.permits(entry["participant"], entry["ticket"]):
                self.entries.pop(event_id)

    def require_capacity(self, images):
        size = sum(len(image.data) for image in images)
        pending = sum(len(image.data) for entry in self.entries.values() for image in entry["images"])
        if len(self.entries) >= MAX_PENDING_EVENTS or pending + size > MAX_PENDING_BYTES:
            raise ValueError("ephemeral image inbox is full; retry later")

    def add(self, event_id, images, participant, ticket):
        self.entries[event_id] = {"images": images, "participant": participant, "ticket": ticket,
                                  "expires": self.now() + TTL_SECONDS}
