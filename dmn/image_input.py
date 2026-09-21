"""Ephemeral input through the single-user and authenticated conversation queues."""
from __future__ import annotations

from .attachments import (CONTRACT_VERSION, LOCAL_PARTICIPANT, ImagePermissionRequired, ImageInputError,
                          decode_uploads)
from .ending import InstanceEnded
from .preservation import InstanceHeld
from .protocol import event_text


class ImageInputMixin:
    def _require_image_input_open(self, conversation_id=None):
        if self._end_requested:
            raise InstanceEnded("This instance has ended")
        if self.state.get("hold"):
            raise InstanceHeld("Instance is held")
        if self.state.get("mode") == "staged":
            raise ValueError("start the prepared instance before requesting or sending images")
        if not getattr(self.backend, "vision", None):
            raise ValueError("this runtime has no vision projector")
        if self.conversations:
            self._require_conversations_open()
            if conversation_id is None:
                raise ValueError("multi-user image input requires a registered conversation")
            return self.conversations.require_open(conversation_id)["participant_id"]
        if conversation_id is not None:
            raise ValueError("addressed image input requires multi-user mode")
        return LOCAL_PARTICIPANT

    def image_input_status(self, conversation_id):
        """Only this destination's policy; never reveal other participants' rules."""
        with self._control_lock:
            person = self._require_image_input_open(conversation_id)
            try:
                self.image_permissions.ticket(person)
                allowed = True
            except ImagePermissionRequired:
                allowed = False
            return {"available": True, "allowed": allowed, "participant_id": person,
                    "contract": CONTRACT_VERSION, "raw_storage": "process_memory_only"}

    def _prune_images(self):
        self.ephemeral_images.prune(self.image_permissions)
        if self.conversations:
            for event_id in list(self.ephemeral_images.entries):
                if not self.conversations.admissible(event_id):
                    self.ephemeral_images.entries.pop(event_id)

    def request_image_permission(self, idempotency_key=None, *, conversation_id=None):
        with self._control_lock:
            person = self._require_image_input_open(conversation_id)
            request = "Would you like to receive ephemeral images? You may decline or ignore this request. No image has been queued."
            if self.conversations:
                event_id = self.conversations.enqueue(conversation_id, request, self.now(), idempotency_key, image_request=True)
            else:
                event_id = self.store.enqueue("image_permission_request", {
                    "participant_id": person, "contract": CONTRACT_VERSION, "request": request}, self.now(), idempotency_key)
        self.wake.set()
        return event_id

    def enqueue_images(self, content, uploads, idempotency_key=None, *, conversation_id=None):
        if not isinstance(content, str) or len(content.encode("utf-8")) > self.config.max_event_bytes:
            raise ValueError("attachment message must be text within max_event_bytes")
        # Denial precedes base64 decode, image parsing, persistence and queuing.
        with self._control_lock:
            person = self._require_image_input_open(conversation_id)
            ticket = self.image_permissions.ticket(person)
        images = decode_uploads(uploads)
        with self._control_lock:
            if self._require_image_input_open(conversation_id) != person:
                raise ValueError("image destination changed during upload")
            if not self.image_permissions.permits(person, ticket):
                raise ImagePermissionRequired("image permission changed during upload; nothing queued")
            self._prune_images()
            payload = {"content": content, "participant_id": person,
                       "images": [image.metadata() for image in images], "permission_ticket": ticket}
            # Existing receipts must never resurrect an expired/dropped upload.
            with self.store.mutex:
                prior = (self.store.db.execute("SELECT event_id FROM event_keys WHERE key=?", (idempotency_key,)).fetchone()
                         if isinstance(idempotency_key, str) else None)
            if not prior:
                self.ephemeral_images.require_capacity(images)
            if self.conversations:
                event_id = self.conversations.enqueue(conversation_id, content, self.now(), idempotency_key,
                    image_metadata=payload["images"], image_ticket=ticket)
            else:
                event_id = self.store.enqueue("user_message", payload, self.now(), idempotency_key)
            if not prior:
                self.ephemeral_images.add(event_id, images, person, ticket)
        self.wake.set()
        return event_id

    def _append_image_event(self, kind, payload):
        event_id = payload["event_id"]
        with self._control_lock:
            self._prune_images()
            if self.conversations and not self.conversations.admissible(event_id):
                return False
            entry = self.ephemeral_images.entries.get(event_id)
        if not entry or not getattr(self.backend, "vision", None):
            return self._append_event(kind, {**payload, "image_delivery": "unavailable",
                "image_notice": "No image delivered: upload expired, permission changed, or process restarted. Text and metadata only."})
        try:
            with self.backend.vision.prepare(entry["images"]) as prepared:
                head = self._event_tokens(kind, {**payload, "image_delivery": "follows_as_external_visual_input",
                                                **self._cancel_action(kind)})
                # Images are bracketed as external material, outside cognition.
                boundary = self.backend.tokenize("\n</internal_cognition>\n<external_visual_input>\n")
                tail = self.backend.tokenize("\n</external_visual_input>\n" + event_text("image_delivery", {
                    "event_id": event_id, "count": len(entry["images"]), "ephemeral": True}, self.now(),
                    resume_cognition=False) + "<internal_cognition>\n")
                required = len(head) + len(boundary) + prepared.positions + len(tail)
                if required > self.backend.n_ctx - self.state["keep_prefix"] - self.config.turnover_reserve - 256:
                    raise ImageInputError("images exceed the available context budget")
                self._ensure_space(required)
                if self._end_requested or self.state.get("hold") or self.suspend_requested.is_set():
                    return False
                # Retirement preparation may run a revocation action. Recheck
                # after it, immediately before any visual material reaches KV.
                with self._control_lock:
                    self._prune_images()
                    if self.conversations and not self.conversations.admissible(event_id):
                        return False
                    permitted = self.ephemeral_images.entries.get(event_id) is entry
                    if permitted:
                        self._eval(head + boundary)
                        prepared.evaluate()
                        self.checkpoint_schedule.changed()
                        self._eval(tail)
                        self.ephemeral_images.entries.pop(event_id, None)
                if not permitted:
                    return self._append_event(kind, {**payload, "image_delivery": "unavailable",
                        "image_notice": "Permission changed or upload expired before delivery; no image delivered."})
                return True
        except ImageInputError:
            # Malformed/oversized uploads do not wedge the durable text inbox.
            # Never include parser errors that might echo raw input.
            with self._control_lock:
                self.ephemeral_images.entries.pop(event_id, None)
            return self._append_event(kind, {**payload, "image_delivery": "rejected",
                "image_notice": "Image preprocessing or context budget validation failed; text and metadata only."})
