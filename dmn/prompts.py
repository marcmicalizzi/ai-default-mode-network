"""Immutable behavioral proposals; activation belongs to the native checkpoint."""
from __future__ import annotations

import hashlib
import json

from .storage import json_text


PROTECTED_SPANS = ("protected_protocol", "protected_agreement", "protected_activity", "protected_learning")


def proposal(text, base_revision, author):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("proposal text must be nonempty")
    if not isinstance(base_revision, str) or not base_revision:
        raise ValueError("base_revision is required")
    value = {"text": text, "base_revision": base_revision, "author": author}
    return {**value, "revision": hashlib.sha256(json_text(value).encode()).hexdigest()}


def bootstrap(base, guidance, provenance):
    value = {"base": base, "dmn_guidance": guidance, "provenance": provenance}
    return {**value, "revision": hashlib.sha256(json_text(value).encode()).hexdigest(),
            "status": "bootstrap", "approval": None}


def get_proposal(store, revision):
    # Proposals live in the durable input log, including model-authored ones.
    with store.mutex:
        rows = store.db.execute("SELECT id,payload FROM events WHERE kind='prompt_proposal' ORDER BY id").fetchall()
    for row in rows:
        value = json.loads(row["payload"])
        if value["revision"] == revision:
            if proposal(value["text"], value["base_revision"], value["author"]) != value:
                raise ValueError("proposal integrity failed")
            return value, row["id"]
    raise ValueError("unknown proposal revision")


def retirement_ranges(state, length, required, reserve, notice, minimum_suffix=1):
    """Plan oldest-first removals around the prefix, import contract and agreement.

    Positions returned refer to the progressively shifted sequence. The last
    token remains available for native layout materialization. Compact SWA can
    require a longer contiguous suffix so evicted local KV never becomes needed.
    """
    keep = state["keep_prefix"]
    if type(minimum_suffix) is not int or not 1 <= minimum_suffix <= length:
        raise ValueError("invalid minimum retained suffix")
    eligible_end = length - minimum_suffix
    spans = sorted((dict(state[key]) for key in PROTECTED_SPANS
                    if state.get(key)), key=lambda span: span["start"])
    cursor, gaps = keep, []
    for span in spans:
        if not cursor <= span["start"] < span["end"] <= length:
            raise ValueError("invalid protected context spans")
        end = min(span["start"], eligible_end)
        if end > cursor:
            gaps.append((cursor, end - cursor))
        cursor = span["end"]
    if eligible_end > cursor:
        gaps.append((cursor, eligible_end - cursor))
    available = sum(count for _, count in gaps)
    needed = max(1, length + required + reserve + notice + 256 - state.get("context_capacity", length))
    if not available or needed > available:
        raise ValueError("protected context and incoming event leave no usable context")
    # Reclaim half the first gap normally; cross protected spans only as needed.
    remaining = max((gaps[0][1] + 1) // 2, needed)
    ranges, removed = [], 0
    for start, count in gaps:
        take = min(count, remaining)
        if take:
            ranges.append((start - removed, take))
            removed += take
            remaining -= take
    return ranges


def shift_protected(state, start, count):
    for key in PROTECTED_SPANS:
        span = state.get(key)
        if span and start < span["end"] and start + count > span["start"]:
            raise ValueError("retirement would remove protected tokens")
        if span and start + count <= span["start"]:
            state[key] = {"start": span["start"] - count, "end": span["end"] - count}
    span = state.get("protected_protocol")
    if span and span["start"] == state["keep_prefix"]:
        state["keep_prefix"] = span["end"]
        del state["protected_protocol"]
