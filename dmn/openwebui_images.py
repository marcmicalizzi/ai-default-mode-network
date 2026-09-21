"""Pinned WebUI image adapter. File references never cross into DMN storage."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

from .attachments import MAX_IMAGES, MAX_IMAGE_BYTES
from .conversations import identifier
from .openwebui import validate_input

REQUEST_COMMAND = "/dmn-images"
MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}


def sources(message):
    files = message.get("files") or []
    if not isinstance(files, list) or len(files) > MAX_IMAGES:
        raise ValueError(f"send at most {MAX_IMAGES} PNG, JPEG or WebP images")
    result = []
    for value in files:
        if not isinstance(value, dict) or value.get("type") not in {"file", "image"}:
            raise ValueError("only uploaded image attachments are supported")
        file_id = identifier(value.get("id"), "image file ID")
        mime = value.get("content_type")
        if mime not in MEDIA_TYPES or value.get("url") != file_id:
            raise ValueError("use a local WebUI image upload; remote URLs and inline data are unsupported")
        result.append({"type": "file", "id": file_id, "url": file_id, "content_type": mime})
    return result


def validate_message(metadata):
    message = metadata.get("user_message") or {}
    images = sources(message)
    # Upstream's top-level files are retrieval context, not this message's
    # images. Never forward prior-chat attachments or execute retrieval.
    cleaned = {**message, "files": []}
    if images and isinstance(message.get("content"), str) and not message["content"].strip():
        cleaned["content"] = "[image attachment]"  # validation only, never sent
    validate_input({**metadata, "user_message": cleaned})
    if images and message["content"].strip() == REQUEST_COMMAND:
        raise ValueError("send /dmn-images without attachments to request consent first")
    return message


def read_local_upload(path, upload_root):
    try:
        path, root = Path(path).resolve(strict=True), Path(upload_root).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("only local WebUI uploads are supported")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
    except (OSError, TypeError) as exc:
        raise ValueError("image upload is unavailable") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or exceeds upload limit")
    return data


async def load_images(message, user_id):
    # Caller must check model consent before invoking this function.
    from open_webui.models.files import Files
    from open_webui.config import UPLOAD_DIR
    uploads = []
    for source in sources(message):
        file = await Files.get_file_by_id(source["id"])
        if not file or file.user_id != user_id:
            raise ValueError("only the authenticated sender's own uploads may be attached")
        mime = (file.meta or {}).get("content_type")
        if mime not in MEDIA_TYPES or mime != source["content_type"]:
            raise ValueError("uploaded file is not the declared image type")
        data = await asyncio.to_thread(read_local_upload, file.path, UPLOAD_DIR)
        uploads.append({"media_type": mime, "data_base64": base64.b64encode(data).decode("ascii")})
    return uploads


def input_digest(message, uploads):
    content = message["content"]
    if uploads:
        content = json.dumps({"content": content, "images": [
            {"media_type": image["media_type"], "sha256": hashlib.sha256(
                base64.b64decode(image["data_base64"], validate=True)).hexdigest()}
            for image in uploads]}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()
