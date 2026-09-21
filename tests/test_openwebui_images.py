"""Upload ownership, bounded reads and consent/retry ordering without WebUI."""
import asyncio
import base64
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from dmn.attachments import MAX_IMAGE_BYTES
from dmn.openwebui_images import validate_message, sources, load_images, input_digest, read_local_upload
from dmn.openwebui_multi import MultiUserOpenWebUIBridge


def message():
    return {"id": "m", "role": "user", "content": "caption", "files": [
        {"type": "file", "id": "image-id", "url": "image-id", "content_type": "image/png"}]}


class WebUIImagesTest(unittest.IsolatedAsyncioTestCase):
    def test_browser_upload_metadata_does_not_look_like_a_history_edit(self):
        bridge = MultiUserOpenWebUIBridge.__new__(MultiUserOpenWebUIBridge)
        original = message()
        original["files"][0].update(status="uploaded", name="image.png", size=42)
        self.assertTrue(bridge.same_attachments(original, message()))
        changed = message()
        changed["files"][0].update(id="new-id", url="new-id")
        self.assertFalse(bridge.same_attachments(original, changed))

    async def test_file_ownership_and_path_are_server_verified_even_for_admin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            uploads = root / "uploads"
            uploads.mkdir()
            owned = uploads / "image"
            owned.write_bytes(b"fixture bytes; parsing is the runtime's responsibility")
            foreign = root / "outside"
            foreign.write_bytes(b"not an upload")
            record = types.SimpleNamespace(user_id="owner", path=str(owned), meta={"content_type": "image/png"})
            files = types.SimpleNamespace(get_file_by_id=AsyncMock(return_value=record))
            with patch.dict("sys.modules", {"open_webui.models.files": types.SimpleNamespace(Files=files),
                                            "open_webui.config": types.SimpleNamespace(UPLOAD_DIR=uploads)}):
                with patch("dmn.openwebui_images.read_local_upload", side_effect=AssertionError("must not read")):
                    with self.assertRaisesRegex(ValueError, "own uploads"):
                        await load_images(message(), "admin")
                actual = await load_images(message(), "owner")
                self.assertEqual(base64.b64decode(actual[0]["data_base64"]), owned.read_bytes())
                record.path = str(foreign)
                with self.assertRaisesRegex(ValueError, "local WebUI"):
                    await load_images(message(), "owner")
                record.path = str(owned)
                record.meta["content_type"] = "text/plain"
                with self.assertRaisesRegex(ValueError, "declared image"):
                    await load_images(message(), "owner")

    async def test_denial_precedes_file_access_and_retry_checks_content(self):
        bridge = MultiUserOpenWebUIBridge.__new__(MultiUserOpenWebUIBridge)
        bridge.client = Mock()
        bridge.client.image_status.return_value = {"allowed": False}
        binding = {"user_id": "owner"}
        with patch("dmn.openwebui_multi.load_images", new=AsyncMock()) as load:
            with self.assertRaisesRegex(ValueError, "has not permitted"):
                await bridge.prepare_input(binding, message())
            load.assert_not_awaited()
            bridge.client.image_status.return_value = {"allowed": True}
            uploads = [{"media_type": "image/png", "data_base64": "eA=="}]
            load.return_value = uploads
            digest = input_digest(message(), uploads)
            bridge.enqueue_prepared = AsyncMock(return_value={"event_id": 42})
            self.assertEqual(await bridge.retry_bound(binding, message(), {"digest": digest}), {"event_id": 42})
            bridge.enqueue_prepared.reset_mock()
            load.return_value = [{"media_type": "image/png", "data_base64": "eQ=="}]
            with self.assertRaisesRegex(ValueError, "attachment"):
                await bridge.retry_bound(binding, message(), {"digest": digest})
            bridge.enqueue_prepared.assert_not_awaited()

    def test_only_explicit_current_message_images_are_accepted(self):
        metadata = dict(chat_id="chat", session_id="socket", message_id="assistant", user_message=message())
        self.assertEqual(validate_message(metadata), message())
        metadata["user_message"]["content"] = ""
        self.assertEqual(validate_message(metadata)["content"], "")
        for field in ("files", "tool_ids"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_message({**metadata, field: ["untrusted context"]})
        for url in ("https://example.invalid/image", "../../file", "data:image/png;base64,eA=="):
            changed = message()
            changed["files"][0]["url"] = url
            with self.assertRaises(ValueError):
                sources(changed)
        metadata["user_message"]["content"] = "/dmn-images"
        with self.assertRaisesRegex(ValueError, "without attachments"):
            validate_message(metadata)

    def test_upload_read_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large"
            with path.open("wb") as stream:
                stream.truncate(MAX_IMAGE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "limit"):
                read_local_upload(path, folder)
