"""Model-selected context protection; no text extraction or operator editing."""
from __future__ import annotations

from .prompts import protected_ranges, protected_size
from .protocol import event_text


OPERATIONS = {"working_memory_help", "working_memory_status", "working_memory_mark",
              "working_memory_protect", "working_memory_note", "working_memory_release"}
PINS = ("protected_working_note", "protected_working_raw")
CONTRACT = '''Optional working memory protects existing context through retirement.
working_memory_status(): see your shared token allowance, usage and marker status.
working_memory_mark(): mark the current position, replacing the earlier marker.
A marker alone DOES NOT protect tokens. Protect before retirement removes any
part of the marked trajectory; a broken marker is invalidated, never shortened.
working_memory_protect(): protect from your valid marker through this action.
Alternatively working_memory_protect(tokens=N) protects the most recent N retained
tokens through this action. This replaces your previous raw pin; it does not grow
automatically. Raw spans include intervening events and action frames, not just
thoughts. No explanation, quotation, or summary is required.
working_memory_note(content): append and protect an exact written anchor, replacing
your previous note pin. Its event wrapper counts against the shared allowance.
working_memory_release(target): release "raw", "note", "mark", or "all".
working_memory_help(offset=0, limit=200): reread these instructions in pages.
Use each action alone and await its result. Changes checkpoint with native state.
Failed requests leave earlier pins unchanged. Older unpinned text can remain until
retirement; release is not deletion of history. Pins have no automatic expiry.
Host allowance and space needed for runtime guidance, continuation and the native
local window limit protection. No pin is silently removed to make room; if safe
retirement cannot make room, inference pauses. Status gives counts, not a copy of
your thought stream. No training or memory-file writing is implied by protection.
Retained tokens and their available native KV remain in this same sequence, with
positions shifted at retirement. Removed dependencies are not recreated. Local
attention still has its sliding window; pinning does not extend it. A rebuilt
cache, including after weight adoption, reevaluates retained text, not earlier KV.'''


def usage(state):
    spans = sorted((state[key]["start"], state[key]["end"]) for key in PINS if state.get(key))
    count, end = 0, 0
    for start, stop in spans:
        count += max(0, stop - max(start, end))
        end = max(end, stop)
    return count


def status(runtime, state=None):
    state = runtime.state if state is None else state
    mark = state.get("working_memory_mark") or {}
    return {"allowance_tokens": runtime.config.working_memory_tokens, "used_tokens": usage(state),
            "note_tokens": (state["protected_working_note"]["end"] - state["protected_working_note"]["start"]
                            if state.get("protected_working_note") else 0),
            "raw_tokens": (state["protected_working_raw"]["end"] - state["protected_working_raw"]["start"]
                           if state.get("protected_working_raw") else 0),
            "marked_tokens": (len(runtime.backend.tokens) - mark["position"]
                              if mark.get("position") is not None else None),
            "marker_invalidated": bool(mark.get("invalidated_by_retirement"))}


def validate_restored(runtime):
    protected_ranges(runtime.state, len(runtime.backend.tokens))
    if usage(runtime.state) > runtime.config.working_memory_tokens:
        raise ValueError("working-memory allowance is below saved protected usage; restore with the saved allowance, then let the instance release pins")
    mark = runtime.state.get("working_memory_mark") or {}
    if mark.get("position") is not None and (type(mark["position"]) is not int or
            not 0 <= mark["position"] <= len(runtime.backend.tokens)):
        raise ValueError("invalid working-memory marker")


def announce(runtime):
    from .sleep_plans import seal
    allowance = runtime.config.working_memory_tokens
    notice = {"contract": CONTRACT, "allowance_tokens": allowance}
    revision = seal(notice)["revision"]
    if ((not allowance and not runtime.state.get("working_memory_guidance")) or
            (runtime.state.get("working_memory_guidance") == revision
             and runtime.state.get("protected_working_guidance"))):
        return
    tokens = runtime.backend.tokenize(event_text("working_memory_available", notice,
                                      runtime.now(), resume_cognition=True))
    runtime._ensure_space(len(tokens))
    if runtime._end_requested or runtime.suspend_requested.is_set() or runtime.state.get("hold"):
        return
    start = len(runtime.backend.tokens)
    runtime._eval(tokens)
    runtime.state["protected_working_guidance"] = {"start": start, "end": len(runtime.backend.tokens)}
    runtime.state["working_memory_guidance"] = revision


def plan_action(runtime, action):
    op = action["op"]
    result = {"op": op, "ok": True}
    if op == "working_memory_status":
        return {**result, **status(runtime)}, None
    if op == "working_memory_help":
        offset, limit = runtime._range({"limit": 200, **action}, 2000)
        result.update(content=CONTRACT[offset:offset + limit], next_offset=min(len(CONTRACT), offset + limit),
                      total_characters=len(CONTRACT))
        while len(runtime.backend.tokenize(event_text("action_result", result, runtime.now(), resume_cognition=True))) > runtime._event_budget():
            limit //= 2
            if limit < 1:
                raise ValueError("working-memory page needs a larger event budget")
            result.update(content=CONTRACT[offset:offset + limit], next_offset=min(len(CONTRACT), offset + limit))
        return result, None
    if op != "working_memory_release" and not runtime.config.working_memory_tokens:
        raise ValueError("working memory has no token allowance in this launch")
    length = len(runtime.backend.tokens)
    updates, tokens = {}, []
    if op == "working_memory_mark":
        updates["working_memory_mark"] = {"position": length}
    elif op == "working_memory_protect":
        if "tokens" in action:
            count = action["tokens"]
            if type(count) is not int or not 1 <= count <= length - runtime.state["keep_prefix"]:
                raise ValueError("tokens must be a positive integer within retained context after the permanent prefix")
            start = length - count
        else:
            mark = runtime.state.get("working_memory_mark") or {}
            if mark.get("position") is None:
                raise ValueError("no intact marker; use working_memory_mark or working_memory_protect(tokens=N)")
            start = mark["position"]
        if start >= length:
            raise ValueError("no tokens follow the marker yet")
        updates["protected_working_raw"] = {"start": start, "end": length}
    elif op == "working_memory_note":
        content = action.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a nonempty string; use working_memory_release to clear a note")
        if len(content.encode("utf-8")) > runtime.config.max_action_bytes:
            raise ValueError("note exceeds max_action_bytes; it was not shortened")
        tokens = runtime.backend.tokenize(event_text("working_memory_note", {"content": content},
                                         runtime.now(), resume_cognition=True))
        updates["protected_working_note"] = {"start": length, "end": length + len(tokens)}
    elif op == "working_memory_release":
        target = action.get("target")
        names = {"raw": "protected_working_raw", "note": "protected_working_note", "mark": "working_memory_mark"}
        if target not in {*names, "all"}:
            raise ValueError("target must be raw, note, mark, or all")
        updates = {key: None for key in (names.values() if target == "all" else [names[target]])}
    candidate = {**runtime.state, **updates}
    used = usage(candidate)
    if used > runtime.config.working_memory_tokens:
        raise ValueError(f"working memory would use {used} tokens; allowance is {runtime.config.working_memory_tokens}; earlier pins unchanged")
    # Releasing needs no future protection budget; its immediate result must fit.
    if op in {"working_memory_note", "working_memory_protect"}:
        window = getattr(runtime.backend, "retirement_window", 1)
        permanent = protected_size(candidate, length + len(tokens))
        headroom = runtime.config.turnover_reserve + runtime._event_budget() + 256 + window
        if permanent + headroom >= runtime.backend.n_ctx:
            raise ValueError("protected context would leave insufficient continuation/local-window space; earlier pins unchanged")
    if length + len(tokens) + runtime._event_budget() > runtime.backend.n_ctx - 32:
        raise ValueError("working-memory action needs more immediate context space; retry after retirement; earlier pins unchanged")
    result.update(used_tokens=used, allowance_tokens=runtime.config.working_memory_tokens)
    return result, {"op": "working_memory", "updates": updates, "tokens": tokens}
