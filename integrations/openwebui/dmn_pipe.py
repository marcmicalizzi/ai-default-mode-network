"""
title: DMN
description: Deliver text events to one persistent DMN instance. Replies arrive independently.
version: 0.1.0
"""


class Pipe:
    async def pipe(self, body: dict, __request__, __metadata__: dict = None, __user__: dict = None, __task__=None, __event_emitter__=None):
        if __task__:
            return ""  # UI title/tag/suggestion jobs are not experiences.
        bridge = getattr(__request__.app.state, "dmn_bridge", None)
        if bridge is None or bridge.closed:
            raise ValueError("Enable the DMN relay Event function first")
        receipt = await bridge.submit(__metadata__ or {}, __user__)
        event_id = receipt["event_id"] if isinstance(receipt, dict) else receipt
        waiting = isinstance(receipt, dict) and receipt.get("admission") == "contact_request"
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {
                "description": ("Waiting for DMN's consent. Your first message is held outside its context." if waiting
                                else "Queued for DMN. It may respond independently."), "done": True,
                "dmn_event_id": event_id}})
        return ""  # Delivery status is transport UI, never fabricated model speech.
