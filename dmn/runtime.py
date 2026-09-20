from __future__ import annotations

import base64
import datetime as dt
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

from .backend import make_backend, sha256_file
from .checkpointing import CheckpointSchedule
from .config import Config
from .diskspace import InsufficientStorage, check_space
from .protocol import ActionParser, PROTOCOL, event_text
from .recovery import restore_checkpoint
from .storage import InstanceLock, Store, json_text, memory_path, write_durable


class ContextFull(RuntimeError):
    pass


class Runtime:
    """A scheduler for a continuing sequence, not a sequence of agent invocations.

    tick() and all backend operations have a single owner. Other threads only
    enqueue durable input, read published state, or set control flags.
    """
    def __init__(self, root: Path, config: Config, backend=None, now=time.time, kv_recovery="strict", initial_context=None,
                 monotonic=time.monotonic):
        if kv_recovery not in {"strict", "fallback", "rebuild"}:
            raise ValueError("unknown KV recovery policy")
        self.root, self.config, self.now = root.resolve(), config, now
        self.monotonic = monotonic
        self.checkpoint_schedule = CheckpointSchedule(config, monotonic)
        self._checkpoint_metrics = {"committed_count": 0, "committed_snapshot_bytes": 0,
                                    "failed_count": 0, "in_progress": False, "last": None}
        self._control_lock = threading.RLock()
        self._suspend_deadline = None
        self._suspending = False
        self._running = False
        self._storage_blocked = None
        self._storage_retry = threading.Event()
        self.lock = InstanceLock(self.root)
        self.store = Store(self.root)
        self.backend = None
        self.wake = threading.Event()
        self.suspend_requested = threading.Event()
        self.resume_requested = threading.Event()
        self.exit_requested = threading.Event()
        self.status_lock = threading.Lock()
        self._status = {}
        self._preparing = False
        self._preparation_actions = None
        self._journal = bytearray()
        self.last_checkpoint_generated = 0
        try:
            self.backend = backend or make_backend(config)
            self.backend.storage_guard = self._require_storage
            self.backend._state_work_dir = self.root
            saved = self.store.latest()
            if saved:
                if initial_context:
                    raise ValueError("initial-context import requires a fresh instance")
                self.state, evidence = restore_checkpoint(self.backend, saved, kv_recovery)
                self.parser = ActionParser(config.max_action_bytes, self.state["parser"])
                self.state["last_restore"] = evidence
                self.last_checkpoint_generated = self.state["generated_tokens"]
                self.checkpoint_schedule.saved_generated = self.state["generated_tokens"]
                # A resume event is factual input, not a fresh initialization prompt.
                prior = self.state["mode"]
                self.state["mode"] = self.state.get("mode_before_suspend", "active") if prior == "suspended" else prior
                if evidence["method"] == "retained_token_reconstruction":
                    self.state.setdefault("origin_continuity", self.state["continuity"])
                    if self.backend.kind == "native_llama_kv":
                        self.state["continuity"] = "context_reconstruction"
                    self.state["reconstructions"] = self.state.get("reconstructions", 0) + 1
                    self._append_event("context_reconstructed", {
                        "fact": "Native KV was not restored. Retained tokens were reevaluated; prior attention to evicted history was not recreated.",
                        "tokens_reevaluated": evidence["prompt_tokens_reevaluated"],
                        "prior_context_retirements": evidence["prior_context_retirements"],
                    })
                self._append_event("execution_resumed", {
                    **self.clock(), **evidence, "previous_mode": prior,
                    "seconds_since_checkpoint": self.now() - self.state["checkpoint_at"],
                    "seconds_since_last_inference": self.elapsed(self.state["last_inference_at"]),
                    "inference_during_gap": False if prior == "suspended" else "unknown_after_last_checkpoint",
                    "recovery": "last_committed_checkpoint; uncommitted computation may have been lost",
                    **({"previous_suspension": self.state["last_suspension"]} if self.state.get("last_suspension") else {}),
                })
                self.checkpoint(reason="restore")
            else:
                self.state = {
                    "schema": 1, "instance_id": str(uuid.uuid4()), "created_at": self.now(),
                    "mode": "active", "event_cursor": 0, "generated_tokens": 0,
                    "last_inference_at": None, "last_external_event_at": None,
                    "last_clock_at": self.now(), "sleep_until": None,
                    "context_retirements": 0, "continuity": self.backend.kind,
                    "event_format": "cognition_v2",
                    "memory_protocol": "revisions_v1", "memory_reads": {},
                }
                self.parser = ActionParser(config.max_action_bytes)
                if initial_context:
                    from .initial_context import initialize_runtime
                    initialize_runtime(self, Path(initial_context))
                else:
                    text = PROTOCOL + "\n" + config.system_prompt + "\n" + event_text("initialization", self.clock(), self.now())
                    seed = self.backend.render_seed(text) + "\n<internal_cognition>\n"
                    tokens = self.backend.tokenize(seed, initial=True)
                    if len(tokens) + config.turnover_reserve + 256 >= self.backend.n_ctx:
                        raise ValueError("initialization prefix leaves insufficient context; increase n_ctx")
                    self.state["rendered_seed"] = seed
                    self.state["keep_prefix"] = config.keep_prefix_tokens or len(tokens)
                    self._eval(tokens)
                    self.checkpoint(reason="initialization")
            self.publish_status()
        except BaseException:
            if self.backend:
                self.backend.close()
            self.store.close()
            self.lock.close()
            raise

    def elapsed(self, timestamp):
        return None if timestamp is None else self.now() - timestamp

    def clock(self):
        now = self.now()
        return {"utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
                "unix_seconds": now,
                "seconds_since_previous_inference": self.elapsed(self.state.get("last_inference_at")),
                "seconds_since_external_event": self.elapsed(self.state.get("last_external_event_at")),
                "elapsed_basis": "wall_clock; negative deltas indicate clock adjustment"}

    def _eval(self, tokens):
        self.backend.eval(tokens)
        if tokens:
            self.checkpoint_schedule.changed()
        self.state["last_inference_at"] = self.now()

    def enqueue(self, content: str, idempotency_key=None):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message must be a nonempty string")
        if len(content.encode("utf-8")) > self.config.max_event_bytes:
            raise ValueError(f"message exceeds {self.config.max_event_bytes} bytes")
        event_id = self.store.enqueue("user_message", {"content": content}, self.now(), idempotency_key)
        self.wake.set()
        return event_id

    def control(self, action, preparation_seconds=None):
        if action not in {"suspend", "resume", "shutdown", "retry_checkpoint"}:
            raise ValueError("unknown control")
        if preparation_seconds is not None and (
                action not in {"suspend", "shutdown"} or isinstance(preparation_seconds, bool) or
                not isinstance(preparation_seconds, (int, float)) or
                not math.isfinite(preparation_seconds) or preparation_seconds < 0):
            raise ValueError("preparation_seconds must be finite and nonnegative, for suspend/shutdown only")
        with self._control_lock:
            if action == "retry_checkpoint":
                self._storage_retry.set()
            elif action == "resume":
                self.resume_requested.set()
            else:
                seconds = self.config.suspend_preparation_seconds if preparation_seconds is None else preparation_seconds
                deadline = None if seconds is None else self.monotonic() + seconds
                # Retried requests can shorten preparation but cannot extend an
                # earlier deadline, including while a native call is running.
                pending = self.suspend_requested.is_set() or self._suspending
                if not pending or (deadline is not None and
                                   (self._suspend_deadline is None or deadline < self._suspend_deadline)):
                    self._suspend_deadline = deadline
                if action == "shutdown":
                    self.exit_requested.set()
                self.suspend_requested.set()
        self.wake.set()

    def _event_tokens(self, kind, payload):
        marked = self.state.get("event_format") == "cognition_v2"
        text = event_text(kind, payload, self.now(), resume_cognition=marked)
        tokens = self.backend.tokenize(text)
        # Bounded insertions leave headroom for preparation and result frames.
        # Full user input stays durable and can be read with event_read.
        budget = self._event_budget()
        if len(tokens) > budget:
            raw = json_text(payload)
            chars = max(32, len(raw) // 2)
            while True:
                reduced = {"truncated": True, "preview": raw[:chars],
                           "original_characters": len(raw), "event_id": payload.get("event_id"),
                           "instruction": "Use event_read for input, or smaller memory_read/list limits for memory."}
                if payload.get("partial_action_cancelled"):
                    reduced.update(partial_action_cancelled=True, action_effects=payload["action_effects"])
                tokens = self.backend.tokenize(event_text(kind, reduced, self.now(), resume_cognition=marked))
                if len(tokens) <= budget:
                    break
                if chars <= 8:
                    raise ContextFull("event envelope cannot fit; increase turnover_reserve")
                chars //= 2
        return tokens

    def _event_budget(self):
        return max(128, min(512, self.config.turnover_reserve // 2))

    def _append_event(self, kind, payload, allow_retirement=True):
        payload = {**payload, **self._cancel_action(kind)}
        tokens = self._event_tokens(kind, payload)
        if allow_retirement:
            self._ensure_space(len(tokens))
        elif len(self.backend.tokens) + len(tokens) > self.backend.n_ctx - 32:
            raise ContextFull("action result exhausted reserved context; effects were not committed")
        self._eval(tokens)

    def _cancel_action(self, reason):
        if not self.parser.cancel():
            return {}
        notice = {"partial_action_cancelled": True,
                  "action_effects": "NONE. The incomplete action did not execute. Retry it if wanted; await a successful action_result."}
        self.store.record("partial_action_cancelled", {"reason": reason,
                          "generated_token": self.state["generated_tokens"]}, self.now())
        return notice

    def _ensure_space(self, required, action_grace=False):
        if self._preparing:
            if len(self.backend.tokens) + required > self.backend.n_ctx - 32:
                raise ContextFull("preparation exhausted reserved context")
            return
        # An incoming event must not consume the space reserved for preparation.
        while len(self.backend.tokens) + required > self.backend.n_ctx - self.config.turnover_reserve:
            # Only an already-started frame may cross the soft boundary. Keep
            # room for its result AND the subsequent retirement notice. External
            # input still interrupts immediately; an unbounded frame cannot
            # postpone retirement indefinitely.
            grace = min(128, self.config.turnover_reserve // 8,
                        max(0, self.config.turnover_reserve - 2 * self._event_budget() - 32))
            if (action_grace and self.parser.pending and grace and
                    len(self.backend.tokens) + required <= self.backend.n_ctx - self.config.turnover_reserve + grace):
                self.state["action_grace_tokens"] = self.state.get("action_grace_tokens", 0) + required
                return
            self._consolidate(required)

    def _consolidate(self, required):
        keep = self.state["keep_prefix"]
        if not self.backend.can_shift or len(self.backend.tokens) <= keep + 1:
            self.state["mode"] = "context_full"
            self.checkpoint(reason="context_full")
            raise ContextFull("native context cannot retire tokens; saved and paused without reconstructing")
        protected = self.state.get("protected_protocol")
        if protected:
            if not keep < protected["start"] < protected["end"] <= len(self.backend.tokens):
                raise RuntimeError("invalid protected import-contract positions")
        self._preparing = True
        start_tokens = len(self.backend.tokens)
        generated_before = self.state["generated_tokens"]
        self._preparation_actions = []
        stopped_reason = "token_budget"
        try:
            self._append_event("context_retirement_pending", {
                "keep_prefix_tokens": keep, "oldest_retirable_position": keep,
                "active_tokens": len(self.backend.tokens),
                "maximum_preparation_tokens": self.config.preparation_tokens,
                "fact": "Oldest tokens will leave active context. Stored memories survive unchanged; no rewrite is needed. Save genuinely new information if wanted. Read existing memory before replacing uncertain details. An unfinished message may continue after retirement; sleep is optional.",
            })
            for _ in range(self.config.preparation_tokens):
                if self.suspend_requested.is_set():
                    stopped_reason = "operator_suspension"
                    break
                if self.state["mode"] != "active":
                    stopped_reason = self.state["mode"]
                    break
                # One sampled token can finish an action. Reserve its maximum
                # result size plus the same 32-token guard used by _ensure_space.
                if len(self.backend.tokens) + 1 + self._event_budget() > self.backend.n_ctx - 32:
                    stopped_reason = "context_headroom"
                    break
                self._generate_one()
        finally:
            self._preparing = False
            preparation_actions = self._preparation_actions
            self._preparation_actions = None
        available = len(self.backend.tokens) - keep - 1
        if protected:
            # Imported history precedes the DMN contract. Retire the oldest
            # history without deleting that contract. Once it reaches the
            # retained prefix, both become one permanently protected prefix.
            available = min(available, protected["start"] - keep)
        discard = max((available + 1) // 2,
                      len(self.backend.tokens) + required + self.config.turnover_reserve + 256 - self.backend.n_ctx)
        if protected:
            discard = min(discard, available)
        elif discard > available:
            self.state["mode"] = "context_full"
            self.checkpoint(reason="context_full")
            raise ContextFull("prefix and incoming event leave no usable context")
        ranges = [(keep, discard)]
        if protected and discard == available:
            # If only a tiny gap remains before the contract, removing it may
            # not even make room for the retirement notice. Plan a second
            # removal AFTER the protected span at this same decode boundary.
            # Both positional shifts are applied by the following decode;
            # no contract tokens or cached values are reconstructed.
            next_keep = protected["end"] - discard
            next_length = len(self.backend.tokens) - discard
            needed = next_length + required + self.config.turnover_reserve + 256 - self.backend.n_ctx
            if needed > 0:
                next_available = next_length - next_keep - 1
                extra = max((next_available + 1) // 2, needed)
                if extra > next_available:
                    self.state["mode"] = "context_full"
                    self.checkpoint(reason="context_full")
                    raise ContextFull("protected import contract and incoming event leave no usable context")
                ranges.append((next_keep, extra))
        for start, count in ranges:
            self.backend.shift(start, count)
        if protected:
            protected["start"] -= discard
            protected["end"] -= discard
            if protected["start"] == keep:
                self.state["keep_prefix"] = protected["end"]
                del self.state["protected_protocol"]
        self.state["context_retirements"] += 1
        self.state["memory_reads"] = {}
        cancelled = self._cancel_action("context_retired")
        # Native positional shifting becomes effective during this next decode.
        tokens = self._event_tokens("context_retired", {"start_position": keep, "removed_tokens": discard,
                                  **({"additional_ranges": [{"start_position": start, "removed_tokens": count}
                                      for start, count in ranges[1:]]} if len(ranges) > 1 else {}),
                                  "fact": "Older KV entries were retired and retained positions shifted. No external summary was substituted.",
                                  "memory_writes_committed_during_preparation": [a["path"] for a in preparation_actions
                                      if a["op"] == "memory_write" and a["ok"]],
                                  **cancelled})
        self._eval(tokens)
        evidence = {"keep": keep, "discard": discard, "tokens_before_preparation": start_tokens,
                    **({"additional_ranges": [{"keep": start, "discard": count}
                        for start, count in ranges[1:]]} if len(ranges) > 1 else {}),
                    "tokens_after": len(self.backend.tokens), "preparation_limit": self.config.preparation_tokens,
                    "preparation_tokens_used": self.state["generated_tokens"] - generated_before,
                    "preparation_stopped_reason": stopped_reason, "preparation_actions": preparation_actions,
                    "partial_action_cancelled": bool(cancelled), "time": self.now(),
                    "native_layout_policy": getattr(self.backend, "native_layout_policy", None),
                    "native_layout_packs": getattr(self.backend, "layout_packs", None)}
        self.state["last_context_retirement"] = evidence
        self.store.record("context_retired", evidence, self.now())
        self.checkpoint(reason="retirement")

    def _generate_one(self):
        if self.state["mode"] != "active":
            return
        self._ensure_space(1, action_grace=True)
        if self.state["mode"] != "active" or (self.suspend_requested.is_set() and not self._suspending):
            # A retirement may have waited for storage while shutdown arrived.
            # Finish that retirement, then suspend before sampling another token.
            return
        token = self.backend.sample()
        self._eval([token])
        self.state["generated_tokens"] += 1
        piece = self.backend.piece(token)
        self._journal.extend(piece)
        actions = self.parser.feed(piece)
        # One token can contain multiple frames. Execute sequentially but commit
        # all their effects with one state, never only the first half of a token.
        effects, results = [], []
        for action in actions:
            result, effect = self._plan_action(action, effects)
            results.append(result)
            if effect:
                effects.append(effect)
        if results:
            # Never generate a retirement-preparation turn while this action's
            # effects are staged but uncommitted. The generation guard reserves
            # space for this result before a token can finish an action.
            self._append_event("action_result", results[0] if len(results) == 1 else {"results": results}, allow_retirement=False)
        if self.backend.is_eog(token):
            self.state["mode"], self.state["sleep_until"] = "sleeping", None
            self.parser.cancel()
        if actions or self.backend.is_eog(token):
            # Grant read-before-replace only once the real result has entered
            # the sequence, never to another frame in the same sampled token.
            for action, result in zip(actions, results):
                if result["ok"] and action["op"] == "memory_read" and action.get("revision") is None:
                    self.state.setdefault("memory_reads", {})[memory_path(action["path"])] = result["revision"]
            if effects or self.state["mode"] == "sleeping" or self.config.checkpoint_policy == "all_actions":
                self.checkpoint(effects, reason="action_effects" if effects else
                                "sleep" if self.state["mode"] == "sleeping" else "action")
            if self._preparation_actions is not None:
                self._preparation_actions.extend({"op": result["op"], "ok": result["ok"], "path": action.get("path")}
                                                 for action, result in zip(actions, results))

    def _plan_action(self, action, staged):
        op = action.get("op")
        result = {"op": op, "ok": True}
        effect = None
        try:
            if op == "send_message":
                content = action["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("content must be a nonempty string")
                effect = {"op": op, "content": content,
                          "action_id": f'{self.state["instance_id"]}:{self.state["generated_tokens"]}:{len(staged)}'}
            elif op == "sleep":
                seconds = action.get("seconds")
                if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                            or not math.isfinite(seconds) or seconds < 0):
                    raise ValueError("seconds must be a finite nonnegative number or null")
                self.state["mode"] = "sleeping"
                self.state["sleep_until"] = None if seconds is None else self.now() + seconds
                result["wake_at"] = self.state["sleep_until"]
            elif op == "clock":
                result.update(self.clock())
            elif op == "event_read":
                event_id = int(action["event_id"])
                event = self.store.next_event(event_id - 1)
                if not event or event["id"] != event_id or event_id > self.state["event_cursor"]:
                    raise ValueError("event is not yet part of this sequence")
                raw = json_text(event["payload"])
                offset, limit = self._range(action, 2000)
                result.update(content=raw[offset:offset + limit], total_characters=len(raw), next_offset=offset + limit)
            elif op == "memory_list":
                offset, limit = self._range(action, 50)
                prefix = action.get("prefix", "/")
                if not isinstance(prefix, str):
                    raise ValueError("prefix must be a string")
                result.update(paths=self.store.memory_list(prefix, offset, limit), next_offset=offset + limit)
            elif op in {"memory_read", "memory_write", "memory_delete", "memory_move", "memory_history"}:
                path = memory_path(action["path"])
                # A model token almost never carries several complete commands,
                # but prevent inconsistent reads or writes against staged memory.
                if any(e["op"].startswith("memory_") for e in staged):
                    raise ValueError("issue one memory operation at a time and await its result")
                if op == "memory_history":
                    offset, limit = self._range(action, 20)
                    result.update(versions=self.store.memory_history(path, offset, limit), next_offset=offset + limit)
                elif op == "memory_read":
                    revision = action.get("revision")
                    if revision is not None and (type(revision) is not int or revision < 1):
                        raise ValueError("revision must be a positive integer")
                    raw = self.store.memory_read(path, revision)
                    offset, limit = self._range(action, 2000)
                    result.update(content=raw[offset:offset + limit], total_characters=len(raw), next_offset=offset + limit,
                                  revision=revision or self.store.memory_revision(path))
                elif op == "memory_write":
                    if not isinstance(action["content"], str):
                        raise ValueError("content must be a string")
                    effect = {"op": op, "path": path, "content": action["content"]}
                elif op == "memory_delete":
                    self.store.memory_read(path)
                    effect = {"op": op, "path": path}
                else:
                    self.store.memory_read(path)
                    destination = memory_path(action["destination"])
                    if destination in self.store.memory_list(destination, 0, 1):
                        raise ValueError("destination already exists")
                    effect = {"op": op, "path": path, "destination": destination}
                if effect and self.state.get("memory_protocol") == "revisions_v1":
                    current = self.store.memory_revision(path)
                    expected = action.get("expected_revision", 0)
                    if type(expected) is not int or expected != current:
                        raise ValueError("memory unchanged: read its CURRENT contents, then use the returned expected_revision")
                    if current and self.state.get("memory_reads", {}).get(path) != current:
                        raise ValueError("memory unchanged: memory_read CURRENT contents after retirement before modifying it")
                    effect["expected_revision"] = expected
                if effect and op == "memory_write":
                    result["revision"] = self.store.memory_next_revision(path)
            else:
                if self.state.get("initial_context") and op != "invalid":
                    return {"op": op, "ok": False,
                            "error": "Unavailable operation. The captured frontend tool definitions are historical; only DMN actions are active.",
                            "available_operations": ["send_message", "sleep", "clock", "event_read",
                                "memory_read", "memory_write", "memory_list", "memory_history", "memory_move", "memory_delete"]}, None
                raise ValueError(action.get("error", "unknown operation"))
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            result = {"op": op, "ok": False, "error": str(exc)}
            effect = None
        return result, effect

    @staticmethod
    def _range(action, maximum):
        offset, limit = action.get("offset", 0), action.get("limit", maximum)
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= maximum:
            raise ValueError(f"offset must be nonnegative and limit must be 1..{maximum}")
        return offset, limit

    def _require_storage(self, path, estimated_bytes, purpose):
        started = None
        since = None
        while True:
            try:
                check_space(path, estimated_bytes, self.config.checkpoint_reserve_bytes, purpose)
                break
            except InsufficientStorage as exc:
                # Startup has no control server yet. Fail without replacing the
                # authoritative checkpoint instead of waiting invisibly.
                if not self._running:
                    raise
                if started is None:
                    started = self.monotonic()
                    since = self.now()
                self._storage_blocked = {**exc.details, "message": str(exc), "since": since}
                self.publish_status()
                # Preserve the pending action/retirement and its call stack.
                # No native calls, journal flushing or effect publication here.
                self._storage_retry.wait()
                self._storage_retry.clear()
        if started is not None:
            self._storage_blocked = None
            self.state["last_storage_pause"] = {"seconds": max(0, self.monotonic() - started),
                                               "purpose": purpose, "inference_during_gap": False}
            self.state["storage_notice_pending"] = True
            self.publish_status()

    def checkpoint(self, effects=None, events=(), *, reason="manual", state_updates=None):
        started = self.monotonic()
        self._checkpoint_metrics["in_progress"] = True
        self.publish_status()
        try:
            estimate = (self.backend.checkpoint_size_bytes() +
                        len(json_text(self.state).encode("utf-8")) +
                        len(json_text(self.backend.fingerprint).encode("utf-8")) + 65536)
            self._require_storage(self.root / "checkpoints", estimate, "checkpoint")
            self.flush_journal()
            directory = self.root / "checkpoints" / uuid.uuid4().hex
            directory.mkdir(parents=True)
            # Never advertise a new durable timestamp or suspended mode before
            # its files AND publication transaction have committed.
            candidate = {**self.state, **(state_updates or {}), "checkpoint_at": self.now(),
                         "checkpoint_reason": reason, "parser": self.parser.state()}
            self.backend.save(directory)
            # Packing may itself have waited for space after the first check.
            for key in ("last_storage_pause", "storage_notice_pending"):
                if key in self.state:
                    candidate[key] = self.state[key]
            write_durable(directory / "runtime.json", candidate)
            files = {}
            for path in directory.iterdir():
                with path.open("r+b") as f:
                    os.fsync(f.fileno())
                files[path.name] = sha256_file(path)
            write_durable(directory / "manifest.json", {"fingerprint": self.backend.fingerprint, "files": files})
            if os.name != "nt":
                fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            snapshot_bytes = sum(path.stat().st_size for path in directory.iterdir())
            self.store.commit_checkpoint(directory.name, effects or [], self.now(), events)
        except BaseException as exc:
            self._checkpoint_metrics["failed_count"] += 1
            self._checkpoint_metrics["in_progress"] = False
            self._checkpoint_metrics["last_failure"] = str(exc)
            self.publish_status()
            raise
        self.state.update(candidate)
        self.last_checkpoint_generated = self.state["generated_tokens"]
        self.checkpoint_schedule.committed(self.last_checkpoint_generated, started)
        self._checkpoint_metrics.update(
            in_progress=False,
            committed_count=self._checkpoint_metrics["committed_count"] + 1,
            committed_snapshot_bytes=self._checkpoint_metrics["committed_snapshot_bytes"] + snapshot_bytes,
            last={"reason": reason, "duration_seconds": max(0, self.monotonic() - started),
                  "snapshot_bytes": snapshot_bytes, "committed_at": self.now()})
        self.publish_status()
        self._prune_checkpoints()

    def _periodic_checkpoint(self):
        reason = self.checkpoint_schedule.due(self.state["generated_tokens"])
        if reason:
            self.checkpoint(reason=reason)

    def _prune_checkpoints(self):
        # Keep current and preceding snapshot. Only remove known checkpoint files
        # in committed, UUID-named directories directly under this instance.
        with self.store.mutex:
            rows = self.store.db.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT -1 OFFSET 2").fetchall()
        base = (self.root / "checkpoints").resolve()
        for row in rows:
            path = base / row[0]
            if len(row[0]) != 32 or path.resolve().parent != base or path.is_symlink():
                continue
            if path.is_dir():
                for name in ("manifest.json", "engine.json", "runtime.json", "state.bin", "logits.npy"):
                    (path / name).unlink(missing_ok=True)
                try:
                    path.rmdir()
                except OSError:
                    pass
            with self.store.transaction() as db:
                db.execute("DELETE FROM checkpoints WHERE directory=?", (row[0],))

    def flush_journal(self):
        if self._journal:
            self.store.record("generated_bytes", {"base64": base64.b64encode(self._journal).decode(),
                              "through_generated_token": self.state["generated_tokens"],
                              "note": "diagnostic journal may extend beyond committed KV after a crash"}, self.now())
            self._journal.clear()

    def suspend(self):
        with self._control_lock:
            if not self.suspend_requested.is_set():
                seconds = self.config.suspend_preparation_seconds
                self._suspend_deadline = None if seconds is None else self.monotonic() + seconds
            self.suspend_requested.clear()
            if self.state["mode"] == "suspended":
                return
            self._suspending = True
        before, generated = self.state["mode"], self.state["generated_tokens"]
        notice_delivered, stopped_reason = False, "token_budget"
        self._preparing = True
        try:
            if self._suspension_expired():
                stopped_reason = "deadline"
            else:
                try:
                    # Suspension must never start a retirement/preparation cycle
                    # merely to make room for its own notice.
                    self._append_event("suspension_pending", {
                        "maximum_preparation_tokens": self.config.preparation_tokens,
                        "preparation_may_end_early": True,
                        "fact": "Inference will stop after bounded preparation; native and runtime state will be saved."},
                        allow_retirement=False)
                    notice_delivered = True
                except ContextFull:
                    stopped_reason = "context_headroom"
                if notice_delivered:
                    for _ in range(self.config.preparation_tokens):
                        if self._suspension_expired():
                            stopped_reason = "deadline"
                            break
                        if self.state["mode"] != "active":
                            stopped_reason = self.state["mode"]
                            break
                        if len(self.backend.tokens) + 1 + self._event_budget() > self.backend.n_ctx - 32:
                            stopped_reason = "context_headroom"
                            break
                        self._generate_one()
            self.checkpoint(reason="shutdown" if self.exit_requested.is_set() else "suspend", state_updates={
                "mode_before_suspend": self.state["mode"] if self.state["mode"] != "active" else before,
                "mode": "suspended", "suspended_at": self.now(),
                "last_suspension": {"notice_delivered": notice_delivered,
                                    "preparation_tokens_used": self.state["generated_tokens"] - generated,
                                    "stopped_reason": stopped_reason}})
        finally:
            self._preparing = False
            with self._control_lock:
                self._suspending = False

    def _suspension_expired(self):
        with self._control_lock:
            return self._suspend_deadline is not None and self.monotonic() >= self._suspend_deadline

    def tick(self):
        if self.suspend_requested.is_set():
            self.suspend()
            return False
        if self.resume_requested.is_set():
            self.resume_requested.clear()
            if self.state["mode"] == "suspended":
                self._append_event("execution_resumed", {**self.clock(), "inference_during_gap": False,
                                   "suspended_seconds": self.elapsed(self.state.get("suspended_at")), "state_retained_in_process": True,
                                   "previous_suspension": self.state.get("last_suspension")})
                self.state["mode"] = self.state.get("mode_before_suspend", "active")
                self.checkpoint(reason="resume")
        if self.state["mode"] in {"suspended", "context_full", "error"}:
            return False
        event = self.store.next_event(self.state["event_cursor"])
        if event:
            # Event is acknowledged by cursor only in a subsequent checkpoint.
            # Input received during inference/checkpoint remains in SQLite.
            self.state["mode"], self.state["sleep_until"] = "active", None
            self._append_event(event["kind"], {**event["payload"], "event_id": event["id"],
                               "arrived_at": event["created"], "delivered_at": self.now(), **self.clock()})
            self.state["event_cursor"] = event["id"]
            self.state["last_external_event_at"] = event["created"]
            self.state["mode"], self.state["sleep_until"] = "active", None
            if self.config.checkpoint_policy == "all_actions":
                self.checkpoint(reason="input")
            else:
                self._periodic_checkpoint()
                self.publish_status()
            return True
        if self.state["mode"] == "sleeping":
            until = self.state["sleep_until"]
            if until is None or self.now() < until:
                return False
            self._append_event("sleep_elapsed", {**self.clock(), "inference_during_sleep": False})
            self.state["mode"], self.state["sleep_until"] = "active", None
        interval = self.config.clock_interval_seconds
        if interval and self.now() - self.state["last_clock_at"] >= interval:
            self._append_event("clock", self.clock())
            self.state["last_clock_at"] = self.now()
        if self.state.get("storage_notice_pending"):
            pause = self.state["last_storage_pause"]
            self._append_event("storage_available", {**pause,
                               "fact": "Execution paused for storage capacity and resumed after an operator retry."})
            # Appending the notice can retire context and encounter another
            # storage pause. Do not erase that newer pending notice.
            if self.state["last_storage_pause"] is pause:
                self.state["storage_notice_pending"] = False
        self._generate_one()
        self._periodic_checkpoint()
        if self.state["generated_tokens"] % 32 == 0:
            self.flush_journal()
        self.publish_status()
        return True

    def publish_status(self):
        with self.status_lock:
            self._status = {key: self.state.get(key) for key in (
                "instance_id", "mode", "created_at", "generated_tokens", "event_cursor", "last_inference_at",
                "sleep_until", "checkpoint_at", "checkpoint_reason", "context_retirements", "continuity", "error", "last_restore", "reconstructions", "event_format", "last_context_retirement", "last_suspension")}
            self._status.update(active_tokens=len(self.backend.tokens), context_capacity=self.backend.n_ctx,
                                native_context_retirement_supported=self.backend.can_shift, process_id=os.getpid())
            self._status["checkpoint"] = {**self.checkpoint_schedule.status(self.state["generated_tokens"]),
                                           **self._checkpoint_metrics}
            self._status["storage"] = {"reserve_bytes": self.config.checkpoint_reserve_bytes,
                                       "blocked": self._storage_blocked,
                                       "last_pause": self.state.get("last_storage_pause")}
            if self._storage_blocked:
                self._status["execution_mode"] = self._status["mode"]
                self._status["mode"] = "storage_blocked"
            self._status_published_at = self.monotonic()

    def status(self):
        with self.status_lock:
            result = self._status.copy()
            checkpoint = dict(result["checkpoint"])
            elapsed = max(0, self.monotonic() - self._status_published_at)
            for key in ("age_seconds", "unsaved_seconds"):
                if checkpoint[key] is not None:
                    checkpoint[key] += elapsed
            result["checkpoint"] = checkpoint
            return result

    def run(self):
        self._running = True
        try:
            while True:
                try:
                    progressed = self.tick()
                except ContextFull as exc:
                    self.state["mode"], self.state["error"] = "context_full", str(exc)
                    self.publish_status()
                    progressed = False
                if self.exit_requested.is_set() and self.state["mode"] in {"suspended", "context_full"}:
                    break
                if not progressed or self.config.token_delay_seconds:
                    self.wake.wait(self.config.token_delay_seconds if progressed else 0.25)
                    self.wake.clear()
        except BaseException as exc:
            self.state["mode"], self.state["error"] = "error", str(exc)
            self.publish_status()
            self.store.record("runtime_error", {"error": str(exc)}, self.now())
            raise
        finally:
            self._running = False

    def close(self):
        # Call suspend() first for planned shutdown. Never checkpoint arbitrary
        # failed native state while handling an exception.
        self.backend.close()
        self.store.close()
        self.lock.close()
