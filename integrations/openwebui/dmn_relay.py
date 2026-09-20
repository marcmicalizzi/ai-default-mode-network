"""
title: DMN relay
description: Durable background delivery for the DMN Pipe; requires the persistent-dmn package.
version: 0.1.0
"""


class Event:
    async def event(self, event: dict, __app__, __id__):
        from dmn.openwebui import start_bridge, stop_bridge
        name = event.get("event")
        subject = event.get("subject") or {}
        own = subject.get("id") == __id__
        if name == "system.startup.completed" or (name == "function.enabled" and own):
            await start_bridge(__app__)
        elif name == "system.shutdown.started" or (name == "function.disable_started" and own):
            await stop_bridge(__app__)
