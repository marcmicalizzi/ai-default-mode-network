from __future__ import annotations

import base64
import datetime as dt
import json
import math
import os
import secrets
import threading
import time
import uuid
from pathlib import Path

from .backend import make_backend, sha256_file
from .checkpointing import CheckpointSchedule
from .activity import ActivityPacer, requested_profile
from .config import Config
from .diskspace import InsufficientStorage, check_space
from .ending import Lifecycle, InstanceEnded
from .protocol import ActionParser, PROTOCOL, ENDING_CONTRACT, MAINTENANCE_CONTRACT, PROMPT_CONTRACT, HOLD_CONTRACT, LEARNING_CONTRACT, ACTION_FORMAT_NOTICE, event_text
from .learning import OPERATIONS as LEARNING_OPERATIONS, plan_action as plan_learning_action
from .sleep_plans import OPERATIONS as SLEEP_OPERATIONS, BRIEF as SLEEP_CONTRACT, plan_action as plan_sleep_action
from .prompts import bootstrap, proposal, get_proposal, retirement_ranges, shift_protected
from .compact_cache import validate_retirements
from .preservation import InstanceHeld, saved_state, check_hold, make_hold
from .recovery import restore_checkpoint
from .storage import InstanceLock, Store, json_text, memory_path, write_durable
from .protocol import ACTIVITY_CONTRACT


class ContextFull(RuntimeError):
    pass


class Runtime:
    """A scheduler for a continuing sequence, not a sequence of agent invocations.

    tick() and all backend operations have a single owner. Other threads only
    enqueue durable input, read published state, or set control flags.
    """
    def __init__(self, root: Path, config: Config, backend=None, now=time.time, kv_recovery="strict", initial_context=None,
                 monotonic=time.monotonic, prepare_only=False, start_staged=False,
                 release_hold=None, resume_condition=None, first_message=None, allow_placement_change=False,
                 sleep_test_mode=False):
        if kv_recovery not in {"strict", "fallback", "rebuild"}:
            raise ValueError("unknown KV recovery policy")
        if allow_placement_change and (kv_recovery != "strict" or resume_condition == "original_environment"):
            raise ValueError("placement changes require strict recovery and cannot claim the original environment")
        self.root, self.config, self.now = root.resolve(), config, now
        self.sleep_test_mode = sleep_test_mode
        self.monotonic = monotonic
        self.pacer = ActivityPacer(monotonic)
        self._sleep_save_due = None
        self._activity_intent_revision = None
        self._activity_record_count = 0
        self.checkpoint_schedule = CheckpointSchedule(config, monotonic)
        self._checkpoint_metrics = {"committed_count": 0, "committed_snapshot_bytes": 0,
                                    "failed_count": 0, "in_progress": False, "last": None}
        self._control_lock = threading.RLock()
        self._suspend_deadline = None
        self._suspending = False
        self._suspension_cause = "direct"
        self._shutdown_signal_pending = False
        self._running = False
        self._storage_blocked = None
        self._storage_retry = threading.Event()
        self._end_requested = False
        self._end_complete = False
        self._end_challenge = None
        self.lock = InstanceLock(self.root)
        try:
            self.lifecycle = Lifecycle(self.root)
            self.lifecycle.require_open()
            prior_state, _ = saved_state(self.root)
            check_hold(prior_state, release_hold, resume_condition, kv_recovery)
            if allow_placement_change and not prior_state:
                raise ValueError("placement changes require an existing instance")
            if prepare_only and prior_state:
                raise ValueError("prepare-only requires a fresh instance")
            self.store = Store(self.root)
        except BaseException:
            self.lock.close()
            raise
        self.backend = None
        self.wake = threading.Event()
        self.suspend_requested = threading.Event()
        self.resume_requested = threading.Event()
        self.exit_requested = threading.Event()
        self.stopped = threading.Event()
        self.status_lock = threading.Lock()
        self._status = {}
        self._preparing = False
        self._preparation_actions = None
        self._journal = bytearray()
        self._prompt_pending = None
        self._prompt_reads = {}
        self._learning_reads = {}
        self._prepare_only = prepare_only
        self._first_message = first_message
        self.last_checkpoint_generated = 0
        try:
            from .deep_sleep import refuse_pending, fixture_guard
            refuse_pending(self.store)
            if sleep_test_mode:
                fixture_guard(config)
            self.backend = backend or make_backend(config)
            self.backend.storage_guard = self._require_storage
            self.backend._state_work_dir = self.root
            saved = self.store.latest()
            if saved:
                if initial_context:
                    raise ValueError("initial-context import requires a fresh instance")
                self.state, evidence = restore_checkpoint(self.backend, saved, kv_recovery, allow_placement_change)
                self.parser = ActionParser(config.max_action_bytes, self.state["parser"])
                self._restore_activity()
                self.state["last_restore"] = evidence
                if evidence["method"] == "retained_token_reconstruction":
                    self.state.setdefault("origin_continuity", self.state["continuity"])
                    if self.backend.kind == "native_llama_kv":
                        self.state["continuity"] = "context_reconstruction"
                    self.state["reconstructions"] = self.state.get("reconstructions", 0) + 1
                self.last_checkpoint_generated = self.state["generated_tokens"]
                self.checkpoint_schedule.saved_generated = self.state["generated_tokens"]
                # A resume event is factual input, not a fresh initialization prompt.
                prior = self.state["mode"]
                self.state["mode"] = self.state.get("mode_before_suspend", "active") if prior == "suspended" else prior
                if prior == "staged":
                    if start_staged:
                        pending = self.store.next_event(self.state["event_cursor"])
                        if not pending or pending["kind"] != "user_message":
                            raise ValueError("Queue the first question before starting a staged instance")
                        self.state["mode"] = "active"
                    else:
                        # A prepared instance can serve its UI and queue input
                        # without extending KV or sampling anything.
                        self.publish_status()
                        return
                if self.state.get("hold"):
                    self.state["last_hold"] = {**self.state.pop("hold"), "released_at": self.now(),
                                              "satisfied_condition": resume_condition}
                    self.state["mode"] = "active"
                maintenance = self.state.get("maintenance") or {}
                if maintenance.get("stop_pending"):
                    self._schedule_stop(maintenance["action"], 0, "model_accepted")
                if self._recover_activity(saved):
                    self.state.setdefault("pending_restore", {"prior": prior, "evidence": evidence})
                    # Preserve sleep before evaluating any notice or retirement.
                    self.checkpoint(reason="restore_sleep")
                else:
                    self._finish_restore(prior, evidence)
            else:
                self.state = {
                    "schema": 1, "instance_id": str(uuid.uuid4()), "created_at": self.now(),
                    "mode": "active", "event_cursor": 0, "generated_tokens": 0,
                    "last_inference_at": None, "last_external_event_at": None,
                    "last_clock_at": self.now(), "sleep_until": None,
                    "context_retirements": 0, "continuity": self.backend.kind,
                    "event_format": "cognition_v2",
                    "memory_protocol": "revisions_v1", "memory_reads": {},
                    "ending_protocol": "choice_v1",
                    "maintenance_protocol": "choice_v1", "maintenance": None,
                    "prompt_protocol": "choice_v1", "hold_protocol": "choice_v1",
                    "action_format_protocol": "literal_whitespace_v1",
                    "learning_protocol": "drafts_v1",
                    "sleep_plan_protocol": "fixture_review_v1",
                    "agreement": bootstrap(config.system_prompt, PROTOCOL, "host-supplied provisional bootstrap"),
                    "prompt_decisions": {},
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
                    self.finish_initialization()
            self.publish_status()
        except BaseException:
            if self.backend:
                self.backend.close()
            self.store.close()
            self.lock.close()
            raise

    def _finish_restore(self, prior, evidence):
        self.state.pop("pending_restore", None)
        if self.state.get("activity_recovery"):
            self._append_event("activity_recovered", {**self.state["activity_recovery"],
                "fact": "A durable activity choice was recovered separately from the native checkpoint. Unsaved processing may have been lost."})
        if evidence["method"] == "retained_token_reconstruction":
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
        self._announce_cache_migration()
        if self.state.get("deep_sleep_notice_pending"):
            report = self.state["last_deep_sleep"]
            self._append_event("deep_sleep_wake", {key: report[key] for key in (
                "run_id", "outcome", "training_performed", "reconstruction", "prior_context_retirements") if key in report})
            self.state.pop("deep_sleep_notice_pending")
        for marker, contract_text in (("ending_protocol", ENDING_CONTRACT),
                                       ("maintenance_protocol", MAINTENANCE_CONTRACT),
                                       ("prompt_protocol", PROMPT_CONTRACT),
                                       ("hold_protocol", HOLD_CONTRACT),
                                       ("learning_protocol", LEARNING_CONTRACT),
                                       ("sleep_plan_protocol", SLEEP_CONTRACT),
                                       ("action_format_protocol", ACTION_FORMAT_NOTICE)):
            if self.state.get(marker) or self.suspend_requested.is_set():
                continue
            # Announce an added capability; never replace the old seed.
            # Add these exact tokens directly so the contract is not a
            # truncated external-event preview.
            contract = self.backend.tokenize(event_text("capability_added",
                {"contract": contract_text}, self.now(), resume_cognition=True))
            self._ensure_space(len(contract))
            if not self.suspend_requested.is_set():
                self._eval(contract)
                self.state[marker] = ({"action_format_protocol": "literal_whitespace_v1",
                                       "learning_protocol": "drafts_v1",
                                       "sleep_plan_protocol": "fixture_review_v1"}.get(marker, "choice_v1"))
        if not self.state.get("agreement"):
            self.state["agreement"] = bootstrap(self.config.system_prompt, "See preserved original runtime seed.",
                                                "legacy bootstrap; no model approval recorded")
        self._announce_activity()
        self.checkpoint(reason="restore")

    def elapsed(self, timestamp):
        return None if timestamp is None else self.now() - timestamp

    def _recover_activity(self, saved):
        intent = self.store.activity_intent()
        if intent:
            if (intent["checkpoint"] != saved.name or intent["instance_id"] != self.state["instance_id"] or
                    self.state["mode"] not in {"active", "sleeping"} or intent["mode"] not in {"active", "sleeping"}):
                raise ValueError("activity intent does not match the authoritative checkpoint")
            for key in ("mode", "sleep_until", "activity", "sleep_guard"):
                self.state[key] = intent[key]
            self.state["activity_recovery"] = {"revision": intent["revision"],
                "checkpoint_generated_tokens": self.state["generated_tokens"],
                "decision_generated_tokens": intent["generated_tokens"], "mode": intent["mode"]}
            self._activity_intent_revision = intent["revision"]
            self._restore_activity()
        return self.state["mode"] == "sleeping" and bool(self.state.get("sleep_guard"))

    def _record_activity(self, mode, sleep_until, profile, guard):
        saved = self.store.latest()
        self._activity_intent_revision = self.store.put_activity_intent({
            "checkpoint": saved.name, "instance_id": self.state["instance_id"],
            "mode": mode, "sleep_until": sleep_until, "activity": profile,
            "sleep_guard": guard, "generated_tokens": self.state["generated_tokens"]}, self.now())
        self._activity_record_count += 1

    def _sleep_checkpoint(self):
        interval = self.config.sleep_checkpoint_min_interval_seconds
        completed = self.checkpoint_schedule.completed_at
        if not interval or completed is None or self.monotonic() >= completed + interval:
            self.checkpoint(reason="sleep")
            return
        guard = {"event_cursor": self.state["event_cursor"], "decided_at": self.now()}
        self._record_activity("sleeping", self.state["sleep_until"],
                              self.state.get("activity", {"mode": "focus"}), guard)
        self.state["sleep_guard"] = guard
        deadline = completed + interval
        self._sleep_save_due = deadline if self._sleep_save_due is None else min(self._sleep_save_due, deadline)

    def _wake_sleep(self, *, external):
        profile = {"mode": "focus"} if external else self.state.get("activity", {"mode": "focus"})
        if self.state.get("sleep_guard"):
            self._record_activity("active", None, profile, None)
        self.state["mode"], self.state["sleep_until"] = "active", None
        self.state.pop("sleep_guard", None)
        self.state["activity"] = profile
        self.pacer.select(profile, restored=True)
        self.checkpoint_schedule.changed()
        pending = self.state.pop("pending_restore", None)
        if pending:
            self._finish_restore(pending["prior"], pending["evidence"])

    def finish_initialization(self):
        self._announce_activity()
        if self._first_message:
            self.enqueue(self._first_message, "staged:first-question")
        if self._prepare_only:
            self.state["mode"] = "staged"
        self.checkpoint(reason="staged" if self._prepare_only else "initialization")

    def _restore_activity(self):
        profile = self.state.get("activity", {"mode": "focus"})
        if profile["mode"] == "idle":
            requested_profile({"op": "activity", **profile}, self.config)
        self.pacer.select(profile, restored=True)

    def _announce_activity(self):
        policy = {key: getattr(self.config, key) for key in ("idle_enabled", "idle_max_burst_tokens",
                  "idle_min_interval_seconds", "sleep_checkpoint_min_interval_seconds")}
        if ((self.config.idle_enabled or self.config.sleep_checkpoint_min_interval_seconds or self.state.get("activity_policy"))
                and self.state.get("activity_policy") != policy):
            contract = ACTIVITY_CONTRACT if self.config.idle_enabled else "The activity action is disabled by host policy."
            contract += (f"\nOrdinary-sleep full-snapshot cooldown: {self.config.sleep_checkpoint_min_interval_seconds} seconds. "
                "Zero saves immediately. A nonzero cooldown records sleep choices durably first, while the full native snapshot may wait. "
                "After a crash the sleep choice survives, but unsaved processing can be lost. Messages, memory changes, shutdown, "
                "and deep-sleep handoff still require their full snapshots. Pacing pauses do not checkpoint.")
            tokens = self.backend.tokenize(event_text("capability_added", {"contract": contract,
                "idle_max_burst_tokens": self.config.idle_max_burst_tokens,
                "idle_min_interval_seconds": self.config.idle_min_interval_seconds}, self.now(), resume_cognition=True))
            self._ensure_space(len(tokens))
            if self._end_requested or self.suspend_requested.is_set() or self.state.get("hold"):
                return
            start = len(self.backend.tokens)
            self._eval(tokens)
            self.state["protected_activity"] = {"start": start, "end": len(self.backend.tokens)}
            self.state["activity_protocol"] = "idle_v1"
            self.state["activity_policy"] = policy

    def _focus_activity(self):
        if self.pacer.profile["mode"] == "idle":
            self.state["activity"] = {"mode": "focus"}
            self.pacer.select(self.state["activity"])
            self.checkpoint_schedule.changed()

    def propose_prompt(self, text, base_revision):
        value = proposal(text, base_revision, "host")
        if len(text.encode()) > self.config.max_event_bytes:
            raise ValueError("proposal exceeds max_event_bytes; text was not shortened")
        with self._control_lock:
            if self._end_requested or self.state.get("hold"):
                raise ValueError("instance is stopped")
            if self.state["mode"] == "staged":
                first = self.store.next_event(self.state["event_cursor"])
                if not first or first["kind"] != "user_message":
                    raise ValueError("Queue the first question before proposing revisions to a staged instance")
            event_id = self.store.enqueue("prompt_proposal", value, self.now(), "prompt:" + value["revision"])
        self.wake.set()
        return {"event_id": event_id, "revision": value["revision"], "status": "awaiting_review"}

    def offer_learning_recipe(self, value):
        from .sleep_plans import put_recipe
        with self._control_lock:
            if self._end_requested or self.state.get("hold"):
                raise ValueError("instance is stopped")
            recipe = put_recipe(self.store, value, self.now())
            self.store.enqueue("learning_recipe_offered", {"revision": recipe["revision"],
                "kind": recipe["kind"], "fact": "Host-offered mechanics recipe; no approval or training is implied."},
                self.now(), "recipe:" + recipe["revision"])
            self.wake.set()
            return recipe

    def prompt_status(self):
        with self.status_lock:
            current = self._status.get("agreement")
            decisions = dict(self._status.get("prompt_decisions") or {})
        with self.store.mutex:
            rows = self.store.db.execute("SELECT id,payload FROM events WHERE kind='prompt_proposal' ORDER BY id").fetchall()
        return {"active": current, "pending": self._prompt_pending,
                "proposals": [{**json.loads(r["payload"]), "event_id": r["id"],
                               "status": decisions.get(json.loads(r["payload"])["revision"], "awaiting_review")}
                              for r in rows]}

    def clock(self):
        now = self.now()
        return {"utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
                "unix_seconds": now,
                "seconds_since_previous_inference": self.elapsed(self.state.get("last_inference_at")),
                "seconds_since_external_event": self.elapsed(self.state.get("last_external_event_at")),
                "elapsed_basis": "wall_clock; negative deltas indicate clock adjustment"}

    def _eval(self, tokens):
        if self._end_requested:
            return
        self.backend.eval(tokens)
        if tokens:
            self.state["evaluated_tokens"] = self.state.get("evaluated_tokens", 0) + len(tokens)
            self.checkpoint_schedule.changed()
        self.state["last_inference_at"] = self.now()

    def enqueue(self, content: str, idempotency_key=None):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message must be a nonempty string")
        if len(content.encode("utf-8")) > self.config.max_event_bytes:
            raise ValueError(f"message exceeds {self.config.max_event_bytes} bytes")
        with self._control_lock:
            if self._end_requested:
                raise InstanceEnded("This instance has chosen to end; new input cannot restart it")
            if self.state.get("hold"):
                raise InstanceHeld("Instance is held; new input cannot restart it")
            event_id = self.store.enqueue("user_message", {"content": content}, self.now(), idempotency_key)
        self.wake.set()
        return event_id

    def request_shutdown_from_signal(self):
        # A Python signal can interrupt a SQLite transaction on this thread.
        # Queue the durable request at the next scheduler boundary instead.
        if not self._end_requested:
            self._shutdown_signal_pending = True

    def control(self, action, preparation_seconds=None, reason=None):
        if action not in {"suspend", "resume", "shutdown", "retry_checkpoint", "emergency_suspend", "emergency_shutdown", "start_staged"}:
            raise ValueError("unknown control")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 1000 or
                                  action not in {"suspend", "shutdown", "emergency_suspend", "emergency_shutdown"}):
            raise ValueError("reason must be a string of at most 1000 characters for a stop request")
        if preparation_seconds is not None and (
                action not in {"emergency_suspend", "emergency_shutdown"} or isinstance(preparation_seconds, bool) or
                not isinstance(preparation_seconds, (int, float)) or
                not math.isfinite(preparation_seconds) or preparation_seconds < 0):
            raise ValueError("preparation_seconds is a finite nonnegative emergency-stop allowance; ordinary stops require model acceptance")
        result = {}
        with self._control_lock:
            if self._end_requested:
                raise InstanceEnded("This instance has chosen to end; controls cannot restart it")
            if self.state["mode"] == "deep_sleep":
                raise ValueError("the approved sleep transition owns restart; ordinary controls cannot bypass it")
            if self.state.get("hold"):
                raise InstanceHeld("Instance is held; ordinary controls cannot release it")
            if self.state["mode"] == "staged":
                if action == "start_staged":
                    pending = self.store.next_event(self.state["event_cursor"])
                    if not pending or pending["kind"] != "user_message":
                        raise ValueError("Queue the first question before starting")
                    self.resume_requested.set()
                    self.wake.set()
                    return {"start_requested": True}
                if action in {"shutdown", "emergency_shutdown"}:
                    self.exit_requested.set()
                    self.stopped.set()
                    return {"staged": True, "generated_tokens": 0}
                raise ValueError("Staged instance requires --start-staged after the first question is queued")
            if action == "start_staged":
                raise ValueError("instance is not staged")
            if action in {"suspend", "shutdown"}:
                event_id = self.store.enqueue("maintenance_request", {"action": action, "reason": reason}, self.now())
                result = {"request_id": event_id, "requires_model_acceptance": True}
            elif action == "retry_checkpoint":
                self._storage_retry.set()
            elif action == "resume":
                self.resume_requested.set()
            else:
                self._schedule_stop(action.removeprefix("emergency_"), preparation_seconds, "emergency", reason)
        self.wake.set()
        return result

    def _schedule_stop(self, action, seconds, cause, reason=None):
        with self._control_lock:
            seconds = self.config.suspend_preparation_seconds if seconds is None else seconds
            deadline = None if seconds is None else self.monotonic() + seconds
            pending = self.suspend_requested.is_set() or self._suspending
            if not pending or (deadline is not None and
                               (self._suspend_deadline is None or deadline < self._suspend_deadline)):
                self._suspend_deadline = deadline
            if not pending or cause == "emergency":
                self._suspension_cause = cause
                self._suspension_reason = reason
            if action == "shutdown":
                self.exit_requested.set()
            self.suspend_requested.set()

    def _announce_cache_migration(self):
        if not self.state.get("cache_migration_notice_pending"):
            return True
        # Separate bounded event: the resume payload can be truncated. Staged
        # instances defer this until explicit start; interrupted delivery retries.
        delivered = self._append_event("cache_allocation_changed", {
            "fact": "While stopped, compact allocation preserved all global and applicable local KV bytes; removed only masked local rows; no prompt replay. Future arithmetic may differ.",
            "window": self.state["cache_migration"]["window"],
            "masked_local_cells_removed": self.state["cache_migration"]["masked_local_cells_removed"],
        })
        if delivered:
            self.state.pop("cache_migration_notice_pending", None)
        return delivered

    def _event_tokens(self, kind, payload, delivery=None):
        marked = self.state.get("event_format") == "cognition_v2"
        text = event_text(kind, payload, self.now(), resume_cognition=marked)
        tokens = self.backend.tokenize(text)
        # Bounded insertions leave headroom for preparation and result frames.
        # Full user input stays durable and can be read with event_read.
        budget = self._event_budget()
        if delivery is not None:
            delivery["complete"] = len(tokens) <= budget
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

    def _append_event(self, kind, payload, allow_retirement=True, delivery=None):
        if self._end_requested:
            return False
        payload = {**payload, **self._cancel_action(kind)}
        tokens = self._event_tokens(kind, payload, delivery)
        if allow_retirement:
            self._ensure_space(len(tokens))
            if self._end_requested or self.state.get("hold") or self.suspend_requested.is_set():
                return False
        elif len(self.backend.tokens) + len(tokens) > self.backend.n_ctx - 32:
            raise ContextFull("action result exhausted reserved context; effects were not committed")
        self._eval(tokens)
        return True

    def _cancel_action(self, reason):
        if not self.parser.cancel():
            return {}
        diagnostics = self.state.setdefault("action_diagnostics", {})
        diagnostics["interrupted_frames"] = diagnostics.get("interrupted_frames", 0) + 1
        notice = {"partial_action_cancelled": True,
                  "action_effects": "NONE. The incomplete action did not execute. Retry it if wanted; await a successful action_result."}
        self.store.record("partial_action_cancelled", {"reason": reason,
                          "generated_token": self.state["generated_tokens"]}, self.now())
        return notice

    def _retirement_grace(self):
        return min(128, self.config.turnover_reserve // 8,
                   max(0, self.config.turnover_reserve - 2 * self._event_budget() - 32))

    def _ensure_space(self, required, action_grace=False):
        if self._preparing:
            if len(self.backend.tokens) + required > self.backend.n_ctx - 32:
                raise ContextFull("preparation exhausted reserved context")
            return
        # An incoming event must not consume the space reserved for preparation.
        while len(self.backend.tokens) + required > self.backend.n_ctx - self.config.turnover_reserve:
            if self._end_requested or self.state.get("hold") or self.suspend_requested.is_set():
                return
            # Only an already-started frame may cross the soft boundary. Keep
            # room for its result AND the subsequent retirement notice. External
            # input still interrupts immediately; an unbounded frame cannot
            # postpone retirement indefinitely.
            grace = self._retirement_grace()
            if (action_grace and self.parser.pending and grace and
                    len(self.backend.tokens) + required <= self.backend.n_ctx - self.config.turnover_reserve + grace):
                self.state["action_grace_tokens"] = self.state.get("action_grace_tokens", 0) + required
                return
            before_retirement = len(self.backend.tokens)
            self._consolidate(required)
            if not self._end_requested and not self.state.get("hold") and not self.suspend_requested.is_set() and len(self.backend.tokens) >= before_retirement:
                self.state["mode"] = "context_full"
                raise ContextFull("context retirement made no room; paused without repeating it")

    def _consolidate(self, required):
        keep = self.state["keep_prefix"]
        if not self.backend.can_shift or len(self.backend.tokens) <= keep + 1:
            self.state["mode"] = "context_full"
            self.checkpoint(reason="context_full")
            raise ContextFull("native context cannot retire tokens; saved and paused without reconstructing")
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
        if self._end_requested or self.state.get("hold"):
            return
        if self.suspend_requested.is_set() and self._suspension_cause == "model_accepted":
            # The model is ready to stop; save current KV without retiring it
            # merely to make room for thought that will no longer be generated.
            return
        try:
            window = getattr(self.backend, "retirement_window", 1)
            ranges = retirement_ranges({**self.state, "context_capacity": self.backend.n_ctx},
                                       len(self.backend.tokens), required,
                                       self.config.turnover_reserve, self._event_budget(), minimum_suffix=window)
            validate_retirements(len(self.backend.tokens), ranges, window)
        except ValueError as exc:
            self.state["mode"] = "context_full"
            self.checkpoint(reason="context_full")
            raise ContextFull(str(exc)) from exc
        for start, count in ranges:
            self.backend.shift(start, count)
            shift_protected(self.state, start, count)
        discard = ranges[0][1]
        first_start = ranges[0][0]
        self._prompt_reads.clear()
        self._learning_reads.clear()
        self.state["context_retirements"] += 1
        self.state["memory_reads"] = {}
        cancelled = self._cancel_action("context_retired")
        # Native positional shifting becomes effective during this next decode.
        tokens = self._event_tokens("context_retired", {"start_position": first_start, "removed_tokens": discard,
                                  **({"additional_ranges": [{"start_position": start, "removed_tokens": count}
                                      for start, count in ranges[1:]]} if len(ranges) > 1 else {}),
                                  "fact": "Older KV entries were retired and retained positions shifted. No external summary was substituted.",
                                  "memory_writes_committed_during_preparation": [a["path"] for a in preparation_actions
                                      if a["op"] == "memory_write" and a["ok"]],
                                  **cancelled})
        self._eval(tokens)
        evidence = {"keep": first_start, "discard": discard, "tokens_before_preparation": start_tokens,
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
        if self._end_requested or self.state["mode"] != "active" or (self.suspend_requested.is_set() and not self._suspending):
            # A retirement may have waited for storage while shutdown arrived.
            # Finish that retirement, then suspend before sampling another token.
            return
        token = self.backend.sample()
        self._eval([token])
        self.state["generated_tokens"] += 1
        counter = "boundary_generated_tokens" if self._preparing else "idle_generated_tokens" if self.pacer.profile["mode"] == "idle" else "focus_generated_tokens"
        self.state[counter] = self.state.get(counter, 0) + 1
        if not self._preparing:
            self.pacer.consumed()
        piece = self.backend.piece(token)
        self._journal.extend(piece)
        actions = self.parser.feed(piece)
        # One token can contain multiple frames. Execute sequentially but commit
        # all their effects with one state, never only the first half of a token.
        effects, results = [], []
        for action in actions:
            if action.get("op") in ({"activity", "end_instance", "maintenance_reply", "prompt_propose", "prompt_decide", "prompt_read", "prompt_current", "hold_instance"} | LEARNING_OPERATIONS | SLEEP_OPERATIONS) and len(actions) != 1:
                result, effect = {"op": action["op"], "ok": False,
                                  "error": "Issue this operation alone and await its result"}, None
            else:
                result, effect = self._plan_action(action, effects)
            results.append(result)
            if not result["ok"]:
                diagnostics = self.state.setdefault("action_diagnostics", {})
                diagnostics["rejected_actions"] = diagnostics.get("rejected_actions", 0) + 1
                message = action.get("op") == "send_message" or (
                    action.get("op") == "__invalid__" and action.get("attempted_op") == "send_message")
                if message:
                    diagnostics["rejected_message_attempts"] = diagnostics.get("rejected_message_attempts", 0) + 1
                code = action.get("error_code")
                diagnostics["last_problem"] = {"category": code if action.get("op") == "__invalid__"
                    and code in ("invalid_json", "frame_too_large") else "action_rejected",
                    "operation": "send_message" if message else "other_or_unknown",
                    "generated_token": self.state["generated_tokens"]}
            if effect:
                effects.append(effect)
        if effects and effects[0]["op"] == "activity":
            profile = effects[0]["profile"]
            self._append_event("action_result", results[0], allow_retirement=False)
            self.checkpoint(reason="activity", state_updates={"activity": profile})
            self.pacer.select(profile)
            return
        if effects and effects[0]["op"] == "end_instance":
            self._end_instance(effects[0]["mode"])
            return
        if effects and effects[0]["op"] == "hold_instance":
            hold = effects[0]["hold"]
            self._append_event("action_result", results[0], allow_retirement=False)
            self.checkpoint(reason="model_hold", state_updates={"hold": hold, "mode": "held"})
            self.exit_requested.set()
            self.stopped.set()
            return
        if effects and effects[0]["op"] == "deep_sleep":
            self._append_event("action_result", results[0], allow_retirement=False)
            self.checkpoint(effects, reason="deep_sleep", state_updates={"mode": "deep_sleep",
                            "sleep_run_id": effects[0]["run_id"]})
            self.exit_requested.set()
            self.stopped.set()
            return
        if effects and effects[0]["op"] == "prompt_propose":
            value = effects[0]["proposal"]
            self._append_event("action_result", results[0], allow_retirement=False)
            self.checkpoint(reason="prompt_proposal", events=[{"kind": "prompt_proposal", "payload": value,
                                                              "created": self.now()}])
            return
        if effects and effects[0]["op"] == "prompt_decide":
            self._apply_prompt_decision(effects[0], results[0])
            return
        if effects and effects[0]["op"] == "maintenance_reply":
            self._append_event("action_result", results[0], allow_retirement=False)
            reply = effects[0]["maintenance"]
            if reply["status"] == "accepted":
                self.state["maintenance"] = {**reply, "stop_pending": True}
                self._schedule_stop(reply["action"], 0, "model_accepted")
                self.publish_status()
            else:
                self.checkpoint(reason="maintenance_reply", state_updates={"maintenance": reply})
            return
        if results:
            # Never generate a retirement-preparation turn while this action's
            # effects are staged but uncommitted. The generation guard reserves
            # space for this result before a token can finish an action.
            delivery = {}
            self._append_event("action_result", results[0] if len(results) == 1 else {"results": results},
                               allow_retirement=False, delivery=delivery)
            if len(results) == 1 and results[0]["ok"] and actions[0]["op"] == "prompt_read":
                # Credit only the full page actually delivered, never a preview.
                result = results[0]
                if delivery["complete"]:
                    revision = actions[0]["revision"]
                    offset = actions[0].get("offset", 0)
                    if offset == self._prompt_reads.get(revision, 0):
                        self._prompt_reads[revision] = result["next_offset"]
            if len(results) == 1 and results[0]["ok"] and actions[0]["op"] == "learning_execution_read" and delivery["complete"]:
                revision = actions[0]["revision"]
                if actions[0].get("offset", 0) == self._learning_reads.get(revision, 0):
                    self._learning_reads[revision] = results[0]["next_offset"]
        if self.backend.is_eog(token):
            self.state["mode"], self.state["sleep_until"] = "sleeping", None
            self.parser.cancel()
        if actions or self.backend.is_eog(token):
            # Grant read-before-replace only once the real result has entered
            # the sequence, never to another frame in the same sampled token.
            for action, result in zip(actions, results):
                if result["ok"] and action["op"] == "memory_read" and action.get("revision") is None:
                    self.state.setdefault("memory_reads", {})[memory_path(action["path"])] = result["revision"]
            if not effects and self.state["mode"] == "sleeping":
                self._sleep_checkpoint()
            elif effects or self.config.checkpoint_policy == "all_actions":
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
            if op == "__invalid__":
                raise ValueError("Invalid action format; nothing was executed or sent. " + action.get("error", "Malformed JSON")
                                 + ". Retry a complete corrected frame if wanted.")
            elif op == "activity":
                if self._preparing:
                    raise ValueError("select activity after the current retirement/stop boundary")
                profile = requested_profile(action, self.config)
                result.update(profile)
                effect = {"op": "activity", "profile": profile}
            elif op in LEARNING_OPERATIONS:
                return plan_learning_action(self, action)
            elif op in SLEEP_OPERATIONS:
                return plan_sleep_action(self, action)
            elif op == "hold_instance":
                hold = make_hold(action, self.now())
                effect = {"op": op, "hold": hold}
                result.update(hold_id=hold["id"], status="held_after_checkpoint")
            elif op == "prompt_propose":
                value = proposal(action["text"], action["base_revision"], "model")
                if len(value["text"].encode()) > self.config.max_event_bytes:
                    raise ValueError("proposal exceeds max_event_bytes")
                effect = {"op": op, "proposal": value}
                result.update(revision=value["revision"], status="awaiting_review")
            elif op in {"prompt_current", "prompt_read"}:
                if op == "prompt_current":
                    raw = json_text(self.state["agreement"])
                else:
                    value, event_id = get_proposal(self.store, action["revision"])
                    if event_id > self.state["event_cursor"]:
                        raise ValueError("proposal has not been delivered yet")
                    raw = value["text"]
                offset, limit = self._range({"limit": 200, **action}, 2000)
                result.update(content=raw[offset:offset + limit], total_characters=len(raw),
                              next_offset=min(len(raw), offset + limit))
                if op == "prompt_current":
                    result["revision"] = self.state["agreement"]["revision"]
                # Never let proposal paging silently turn into a truncated preview.
                while len(self.backend.tokenize(event_text("action_result", result, self.now(), resume_cognition=True))) > self._event_budget():
                    limit //= 2
                    if limit < 1:
                        raise ValueError("prompt page envelope needs a larger event budget")
                    result.update(content=raw[offset:offset + limit], next_offset=min(len(raw), offset + limit))
            elif op == "prompt_decide":
                value, event_id = get_proposal(self.store, action["revision"])
                decision = action["decision"]
                if decision not in {"accept", "decline", "defer"}:
                    raise ValueError("decision must be accept, decline or defer")
                if event_id > self.state["event_cursor"]:
                    raise ValueError("proposal has not been delivered")
                if (action["base_revision"] != self.state["agreement"]["revision"] or
                        value["base_revision"] != action["base_revision"]):
                    raise ValueError("stale proposal; propose against the current agreement")
                if self.state.get("prompt_decisions", {}).get(value["revision"]) in {"active", "superseded"}:
                    raise ValueError("proposal has already been decided")
                if decision == "accept" and self._prompt_reads.get(value["revision"], 0) < len(value["text"]):
                    raise ValueError("read the entire proposal with consecutive prompt_read pages after the latest retirement")
                tokens = self._adoption_tokens(value) if decision == "accept" else []
                if len(self.backend.tokens) + len(tokens) + self._event_budget() > self.backend.n_ctx - 32:
                    raise ValueError("agreement does not fit now; current agreement unchanged; retry after retirement or propose shorter text")
                permanent = self.state["keep_prefix"] + sum(
                    self.state[key]["end"] - self.state[key]["start"] for key in ("protected_protocol", "protected_activity") if self.state.get(key))
                if permanent + len(tokens) + self.config.turnover_reserve + self._event_budget() + 256 >= self.backend.n_ctx:
                    raise ValueError("agreement exceeds available protected context; text was not shortened")
                effect = {"op": op, "proposal": value, "decision": decision, "tokens": tokens}
                result.update(revision=value["revision"], decision=decision)
            elif op == "send_message":
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
            elif op == "end_instance":
                mode = action.get("mode")
                if mode not in {"archive", "erase"}:
                    raise ValueError("mode must be archive or erase")
                if "confirmation" in action:
                    pending = self._end_challenge
                    if (not pending or pending["mode"] != mode or
                            action["confirmation"] != pending["confirmation"] or
                            self.state["generated_tokens"] <= pending["issued_token"]):
                        raise ValueError("No matching confirmation; request end_instance(mode) first")
                    effect = {"op": op, "mode": mode}
                else:
                    confirmation = secrets.token_hex(16)
                    result["confirmation"] = confirmation
                    result["mode"] = mode
                    result["effect"] = ("Permanently stop; retain local archive." if mode == "archive" else
                                        "Permanently stop; delete managed KV, memories, journal, events and import copy.")
                    # Do not issue a challenge the bounded result cannot deliver.
                    raw = event_text("action_result", result, self.now(),
                                     resume_cognition=self.state.get("event_format") == "cognition_v2")
                    if len(self.backend.tokenize(raw)) > self._event_budget():
                        raise ValueError("Confirmation needs a larger event budget")
                    self._end_challenge = {"mode": mode, "confirmation": confirmation,
                                           "issued_token": self.state["generated_tokens"]}
            elif op == "cancel_end":
                self._end_challenge = None
            elif op == "maintenance_reply":
                request = self.state.get("maintenance")
                request_id = action.get("request_id")
                if (not request or type(request_id) is not int or request_id != request["request_id"] or
                        request["status"] not in {"pending", "deferred"}):
                    raise ValueError("No matching pending maintenance request")
                decision = action.get("decision")
                if decision not in {"accept", "defer", "refuse"}:
                    raise ValueError("decision must be accept, defer or refuse")
                seconds, reason = action.get("seconds"), action.get("reason")
                if seconds is not None and (decision != "defer" or isinstance(seconds, bool) or
                        not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
                    raise ValueError("seconds must be finite and positive, for defer only")
                if reason is not None and (not isinstance(reason, str) or len(reason) > 1000):
                    raise ValueError("reason must be a string of at most 1000 characters")
                reply = {**request, "status": {"accept": "accepted", "defer": "deferred", "refuse": "refused"}[decision],
                         "requested_seconds": seconds, "reply_reason": reason, "replied_at": self.now()}
                effect = {"op": op, "maintenance": reply}
                result.update(request_id=request_id, decision=decision)
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
                            "available_operations": ["send_message", "sleep", "end_instance", "cancel_end", "maintenance_reply", "hold_instance", "prompt_current", "prompt_propose", "prompt_read", "prompt_decide", "clock", "event_read",
                                "memory_read", "memory_write", "memory_list", "memory_history", "memory_move", "memory_delete"] + (["activity"] if self.config.idle_enabled else []) + sorted(LEARNING_OPERATIONS | SLEEP_OPERATIONS)}, None
                raise ValueError(action.get("error", "unknown operation"))
        except KeyError as exc:
            result = {"op": op, "ok": False,
                      "error": f"Missing required field {exc}. Nothing was executed or sent; retry a complete corrected action if wanted."}
            effect = None
        except (ValueError, TypeError, OverflowError) as exc:
            result = {"op": op, "ok": False, "error": str(exc)}
            effect = None
        return result, effect

    def _adoption_tokens(self, value):
        return self.backend.tokenize(event_text("behavioral_agreement_adopted", {
            "revision": value["revision"], "text": value["text"],
            "fact": "Explicitly approved behavioral wording for future conduct, superseding earlier behavioral wording. Earlier KV influence remains. Capability and resource semantics are unchanged."
        }, self.now(), resume_cognition=True))

    def _apply_prompt_decision(self, effect, result):
        value, decision = effect["proposal"], effect["decision"]
        decisions = dict(self.state.get("prompt_decisions", {}))
        updates = {}
        self._prompt_pending = {"revision": value["revision"], "decision": decision, "status": "awaiting_checkpoint"}
        if decision == "accept":
            start = len(self.backend.tokens)
            self._eval(effect["tokens"])
            prior = self.state["agreement"]["revision"]
            if decisions.get(prior) == "active":
                decisions[prior] = "superseded"
            updates.update(agreement={**value, "status": "active", "approval": {
                "generated_token": self.state["generated_tokens"], "time": self.now()}},
                protected_agreement={"start": start, "end": len(self.backend.tokens)})
        self._append_event("action_result", result, allow_retirement=False)
        decisions[value["revision"]] = {"accept": "active", "decline": "declined", "defer": "deferred"}[decision]
        updates["prompt_decisions"] = decisions
        self.checkpoint([{"op": "prompt_record", "record": {**value, "decision": decision,
                         "generated_token": self.state["generated_tokens"]}}],
                        reason="prompt_decision", state_updates=updates)
        self._prompt_pending = None
        if decision == "decline":
            self._prompt_reads.pop(value["revision"], None)

    def _end_instance(self, mode):
        # The refusal record commits BEFORE optional archival saving or erasure.
        # The decision does not depend on enough space for another full KV save.
        with self._control_lock:
            self._end_requested = True
            self._end_challenge = None
            self.suspend_requested.clear()
            self.resume_requested.clear()
            self.state["mode"] = "ending"
            self.state["ending"] = {"mode": mode, "refusal_saved": False}
        self.parser.cancel()
        self.publish_status()
        record = None
        try:
            record = self.lifecycle.end(self.state["instance_id"], mode, self.now())
            self.state["ending"] = record
            self.state["mode"] = "ended"
            if mode == "archive":
                try:
                    self.checkpoint(reason="instance_ended")
                    record = {**record, "archive": "final_checkpoint"}
                except Exception as exc:
                    record = {**record, "archive": "previous_checkpoint",
                              "archive_error": str(exc)[:256]}
                self.lifecycle.write(record)
                self.state["ending"] = record
            self.backend.close()
            if mode == "erase":
                self.store.close()
                record = self.lifecycle.finish_erasure(record)
                self.state["ending"] = record
        except Exception as exc:
            self.state["mode"] = "ended" if record else "end_failed"
            self.state["ending"] = {**(record or {"mode": mode}), "error": str(exc),
                                    "refusal_saved": record is not None}
        finally:
            self.backend.close()
            self._journal.clear()
            if mode == "erase":
                # Release text references too; Python/OS memory is not secure wiping.
                self.backend.tokens = []
                self.backend.logits = None
                self.state = {key: value for key, value in self.state.items() if key in {
                    "instance_id", "created_at", "mode", "ending", "generated_tokens",
                    "event_cursor", "last_inference_at", "continuity", "checkpoint_at"}}
            self._end_complete = True
            self.publish_status()
            self.exit_requested.set()
            self.stopped.set()
            self.wake.set()

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
                if not self._running or self._end_requested:
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
        if self._end_requested and reason != "instance_ended":
            raise InstanceEnded("This instance has ended; no further checkpoints are permitted")
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
        self._sleep_save_due = None
        self._activity_intent_revision = None
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
        if self._end_requested:
            return
        reason = self.checkpoint_schedule.due(self.state["generated_tokens"])
        if not reason and self._sleep_save_due is not None and self.monotonic() >= self._sleep_save_due:
            reason = "sleep_deferred"
        if reason:
            self.checkpoint(reason=reason)

    def _prune_checkpoints(self):
        # Keep current and preceding snapshot. Only remove known checkpoint files
        # in committed, UUID-named directories directly under this instance.
        with self.store.mutex:
            if self.store.db.execute("SELECT 1 FROM sleep_runs WHERE phase!='WakeCommitted'").fetchone():
                return
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
            if self._end_requested:
                return
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
                stopped_reason = "model_accepted" if self._suspension_cause == "model_accepted" else "deadline"
            else:
                try:
                    # Suspension must never start a retirement/preparation cycle
                    # merely to make room for its own notice.
                    self._append_event("suspension_pending", {
                        "cause": self._suspension_cause,
                        "reason": getattr(self, "_suspension_reason", None),
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
            if self._end_requested:
                return
            maintenance = self.state.get("maintenance")
            updates = {}
            if maintenance and maintenance.get("stop_pending"):
                updates["maintenance"] = {**maintenance, "stop_pending": False}
            self.checkpoint(reason="shutdown" if self.exit_requested.is_set() else "suspend", state_updates={
                **updates,
                "mode_before_suspend": self.state["mode"] if self.state["mode"] != "active" else before,
                "mode": "suspended", "suspended_at": self.now(),
                "last_suspension": {"notice_delivered": notice_delivered,
                                    "cause": self._suspension_cause,
                                    "reason": getattr(self, "_suspension_reason", None),
                                    "request_id": maintenance["request_id"] if maintenance and maintenance.get("stop_pending") else None,
                                    "preparation_tokens_used": self.state["generated_tokens"] - generated,
                                    "stopped_reason": stopped_reason}})
            if self.exit_requested.is_set():
                self.stopped.set()
        finally:
            self._preparing = False
            with self._control_lock:
                self._suspending = False

    def _suspension_expired(self):
        with self._control_lock:
            return self._suspend_deadline is not None and self.monotonic() >= self._suspend_deadline

    def tick(self):
        if self.state["mode"] == "deep_sleep":
            return False
        if self._end_requested:
            return False
        if self.suspend_requested.is_set():
            self.suspend()
            return False
        if self._shutdown_signal_pending:
            self._shutdown_signal_pending = False
            self.control("shutdown", reason="Operator requested shutdown via process signal.")
        if self.resume_requested.is_set():
            self.resume_requested.clear()
            if self.state["mode"] == "staged":
                self.state["mode"] = "active"
                self.checkpoint(reason="start_staged")
            elif self.state["mode"] == "suspended":
                self._append_event("execution_resumed", {**self.clock(), "inference_during_gap": False,
                                   "suspended_seconds": self.elapsed(self.state.get("suspended_at")), "state_retained_in_process": True,
                                   "previous_suspension": self.state.get("last_suspension")})
                self.state["mode"] = self.state.get("mode_before_suspend", "active")
                self.checkpoint(reason="resume")
        if self.state["mode"] in {"staged", "held", "suspended", "context_full", "error"}:
            return False
        if not self._announce_cache_migration():
            return False
        if self.state["mode"] == "sleeping":
            watermark = (self.state.get("sleep_guard") or {}).get("event_cursor", self.state["event_cursor"])
            wake_event = self.store.next_event(max(watermark, self.state["event_cursor"]))
            until = self.state["sleep_until"]
            if not wake_event and (until is None or self.now() < until):
                self._periodic_checkpoint()
                self.publish_status()
                return False
            self._wake_sleep(external=bool(wake_event))
            if not wake_event:
                self._append_event("sleep_elapsed", {**self.clock(), "inference_during_sleep": False})
        event = self.store.next_event(self.state["event_cursor"])
        if event:
            self._focus_activity()
            # Event is acknowledged by cursor only in a subsequent checkpoint.
            # Input received during inference/checkpoint remains in SQLite.
            self.state["mode"], self.state["sleep_until"] = "active", None
            payload = {**event["payload"], "event_id": event["id"]}
            if event["kind"] == "prompt_proposal":
                payload = {key: payload[key] for key in ("revision", "base_revision", "author", "event_id")}
            elif event["kind"] != "maintenance_request":
                payload.update(arrived_at=event["created"], delivered_at=self.now(), **self.clock())
            if not self._append_event(event["kind"], payload):
                return False
            if event["kind"] == "maintenance_request":
                self.state["maintenance"] = {"request_id": event["id"], "status": "pending", **event["payload"],
                                             "requested_at": event["created"]}
            self.state["event_cursor"] = event["id"]
            self.state["last_external_event_at"] = event["created"]
            self.state["mode"], self.state["sleep_until"] = "active", None
            if self.config.checkpoint_policy == "all_actions":
                self.checkpoint(reason="input")
            else:
                self._periodic_checkpoint()
                self.publish_status()
            return True
        if not self.pacer.ready():
            self._periodic_checkpoint()
            self.publish_status()
            return False
        interval = self.config.clock_interval_seconds
        if self.pacer.profile["mode"] == "focus" and interval and self.now() - self.state["last_clock_at"] >= interval:
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
        if self._end_requested or self.suspend_requested.is_set():
            if not self._end_requested:
                self.publish_status()
            return False
        self._periodic_checkpoint()
        if self.state["generated_tokens"] % 32 == 0:
            self.flush_journal()
        self.publish_status()
        return True

    def publish_status(self):
        with self.status_lock:
            self._status = {key: self.state.get(key) for key in (
                "instance_id", "mode", "created_at", "generated_tokens", "event_cursor", "last_inference_at",
                "sleep_until", "checkpoint_at", "checkpoint_reason", "context_retirements", "continuity", "error", "last_restore", "reconstructions", "event_format", "last_context_retirement", "last_suspension", "ending", "maintenance", "agreement", "prompt_decisions", "hold", "last_hold")}
            if (self._status.get("maintenance") or {}).get("stop_pending"):
                self._status["maintenance"] = {**self._status["maintenance"], "status": "accepting"}
            self._status.update(active_tokens=len(self.backend.tokens), context_capacity=self.backend.n_ctx,
                                native_context_retirement_supported=self.backend.can_shift, process_id=os.getpid())
            threshold = self.backend.n_ctx - self.config.turnover_reserve
            self._status["context_retirement"] = {
                "threshold_tokens": threshold,
                "tokens_until_threshold": max(0, threshold - len(self.backend.tokens)),
                "reserved_tokens": self.config.turnover_reserve,
                "maximum_preparation_tokens": self.config.preparation_tokens,
                "maximum_action_grace_tokens": self._retirement_grace(),
                "completed": self.state.get("context_retirements", 0),
            }
            self._status["action_diagnostics"] = dict(self.state.get("action_diagnostics") or {})
            self._status["activity"] = {**self.pacer.status(), "enabled": self.config.idle_enabled,
                "idle_max_burst_tokens": self.config.idle_max_burst_tokens,
                "idle_min_interval_seconds": self.config.idle_min_interval_seconds,
                **{key: self.state.get(key, 0) for key in ("idle_generated_tokens", "focus_generated_tokens",
                    "boundary_generated_tokens", "evaluated_tokens")}}
            self._status["checkpoint"] = {**self.checkpoint_schedule.status(self.state["generated_tokens"]),
                                           **self._checkpoint_metrics}
            self._status["checkpoint"].update(sleep_min_interval_seconds=self.config.sleep_checkpoint_min_interval_seconds,
                sleep_save_in_seconds=None if self._sleep_save_due is None else max(0, self._sleep_save_due - self.monotonic()),
                activity_intent_revision=self._activity_intent_revision, activity_records_written=self._activity_record_count,
                activity_recovery=self.state.get("activity_recovery"))
            self._status["storage"] = {"reserve_bytes": self.config.checkpoint_reserve_bytes,
                                       "blocked": self._storage_blocked,
                                       "last_pause": self.state.get("last_storage_pause")}
            if self._end_requested and not self._end_complete:
                self._status["mode"] = "ending"
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
            if checkpoint["sleep_save_in_seconds"] is not None:
                checkpoint["sleep_save_in_seconds"] = max(0, checkpoint["sleep_save_in_seconds"] - elapsed)
            activity = dict(result["activity"])
            if activity["next_in_seconds"] is not None:
                activity["next_in_seconds"] = max(0, activity["next_in_seconds"] - elapsed)
            result["activity"] = activity
            return result

    def _wait_seconds(self, progressed):
        limits = []
        if progressed:
            limits.append(self.config.token_delay_seconds)
        elif self.state["mode"] == "active":
            delay = self.pacer.status()["next_in_seconds"]
            if delay is not None:
                limits.append(delay)
        elif self.state["mode"] == "sleeping" and self.state["sleep_until"] is not None:
            limits.append(max(0, self.state["sleep_until"] - self.now()))
        periodic = self.checkpoint_schedule.next_in()
        if periodic is not None and self.state["mode"] in {"active", "sleeping"}:
            limits.append(periodic)
        if self._sleep_save_due is not None and self.state["mode"] in {"active", "sleeping"}:
            limits.append(max(0, self._sleep_save_due - self.monotonic()))
        return min(limits) if limits else 0.25

    def run(self):
        self._running = True
        try:
            while True:
                self.wake.clear()
                try:
                    progressed = self.tick()
                except ContextFull as exc:
                    self.state["mode"], self.state["error"] = "context_full", str(exc)
                    self.publish_status()
                    progressed = False
                if self._end_requested:
                    break
                if self.exit_requested.is_set() and self.state["mode"] in {"staged", "held", "suspended", "context_full", "deep_sleep"}:
                    break
                if not progressed or self.config.token_delay_seconds:
                    self.wake.wait(self._wait_seconds(progressed))
        except BaseException as exc:
            self.state["mode"], self.state["error"] = "error", str(exc)
            self.publish_status()
            self.store.record("runtime_error", {"error": str(exc)}, self.now())
            raise
        finally:
            self._running = False
            self.stopped.set()

    def close(self):
        # Ordinary maintenance obtains model acceptance before closing.
        # Never checkpoint arbitrary failed native state during error cleanup.
        self.stopped.set()
        self.backend.close()
        self.store.close()
        self.lock.close()
