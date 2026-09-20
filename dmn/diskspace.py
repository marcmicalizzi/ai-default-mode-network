"""Advisory capacity checks; these do not reserve space against other writers."""
from __future__ import annotations

import shutil
from pathlib import Path


class InsufficientStorage(OSError):
    def __init__(self, details):
        self.details = details
        super().__init__(f"Insufficient storage for {details['purpose']}: "
                         f"{details['free_bytes']} bytes free, "
                         f"{details['required_bytes']} needed including reserve. "
                         "Free space on this volume and retry; prior checkpoints remain intact.")


def check_space(path: Path, estimated_bytes: int, reserve_bytes: int, purpose: str):
    path = Path(path).resolve()
    # A first checkpoint's directory may not exist yet. Check its volume before
    # creating anything, without pruning a committed checkpoint to make room.
    while not path.exists() and path.parent != path:
        path = path.parent
    free = shutil.disk_usage(path).free
    details = {"path": str(path), "purpose": purpose, "free_bytes": free,
               "estimated_bytes": estimated_bytes, "reserve_bytes": reserve_bytes,
               "required_bytes": estimated_bytes + reserve_bytes}
    if free < details["required_bytes"]:
        raise InsufficientStorage(details)
    return details
