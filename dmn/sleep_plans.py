"""Compiled, reviewed sleep plans. Only mechanics fixtures are executable today."""
from __future__ import annotations

import json
from pathlib import Path

from .learning import digest_text, read_plan, _fields, _fits
from .storage import json_text


CHECKS = ["artifact_integrity", "retained_tokens_and_rng"]
OPERATIONS = {"learning_recipe_list", "learning_recipe_read", "learning_compile",
              "learning_execution_help", "learning_execution_read", "learning_execution_decide", "deep_sleep", "learning_sleep_report"}
BRIEF = '''learning_execution_help(offset=0, limit=200): read the separate compiled-plan
review and disposable sleep-test contract. No production trainer is enabled.'''
CONTRACT = '''Compiled learning plans require separate review and choice.
learning_recipe_list(): list offered recipes. learning_recipe_read(revision, offset=0,
limit=200): inspect one. learning_compile(draft_revision, recipe_revision): bind exact
token IDs, target-only loss masks, parent weights and recipe to an immutable plan.
learning_execution_read(revision, offset=0, limit=200): inspect consecutive pages.
learning_execution_decide(revision, decision): approve, decline or defer. Approval
requires reading the complete compiled plan after the latest retirement/restart.
Decline withdraws prior approval. Draft withdrawal/replacement invalidates its plans.
Use each action alone. deep_sleep(revision) saves and stops for the approved plan
ONLY in the explicitly enabled disposable mechanics harness. Ordinary launch has
no executable trainer; approval cannot enable one. The only current recipe is a
mechanics fixture: it does NOT train or claim to learn your examples. Its proposed
adapter and this limitation must be reviewed. learning_sleep_report(run_id, offset=0,
limit=200) reads a completed cycle report. Ordinary sleep remains unchanged.'''


def seal(value):
    return {**value, "revision": digest_text(json_text(value))}


def identity(fingerprint):
    return {**{k: fingerprint[k] for k in ("kind", "model_sha256", "script_sha256", "research_lora")
               if k in fingerprint}, "lora_adapters": fingerprint.get("lora_adapters", [])}


def implementation_identity():
    from .backend import sha256_file
    return {name: sha256_file(Path(__file__).with_name(name)) for name in (
        "deep_sleep.py", "sleep_plans.py", "backend.py", "adapters.py", "config.py", "recovery.py", "storage.py",
        "runtime.py", "protocol.py", "learning.py")}


def put_recipe(store, value, now):
    """Host can offer a recipe, never approve it or supply executable commands."""
    from .adapters import AdapterSpec
    _fields(value, "schema kind parent candidate resources checks", "recipe")
    if value["schema"] != 1 or value["kind"] != "fixture_candidate_v1":
        raise ValueError("only the non-training fixture_candidate_v1 recipe is implemented")
    if value["checks"] != CHECKS:
        raise ValueError("fixture recipe requires artifact_integrity and retained_tokens_and_rng checks")
    if value["candidate"] is not None:
        AdapterSpec(**value["candidate"])
    _fields(value["resources"], "max_training_seconds max_ram_bytes max_vram_bytes max_disk_bytes", "resources")
    for key, limit in value["resources"].items():
        if type(limit) is not int or limit < (0 if key == "max_vram_bytes" else 1):
            raise ValueError("invalid recipe resource ceiling")
    if value["resources"]["max_vram_bytes"] != 0:
        raise ValueError("fixture recipes require zero GPU use")
    record = seal(value)
    with store.transaction() as db:
        db.execute("INSERT OR IGNORE INTO sleep_recipes VALUES(?,?,?)",
                   (record["revision"], json_text(record), now))
    return record


def read_record(store, table, revision):
    if table not in {"sleep_recipes", "sleep_executions"} or not isinstance(revision, str):
        raise ValueError("invalid plan/recipe revision")
    with store.mutex:
        row = store.db.execute(f"SELECT payload FROM {table} WHERE revision=?", (revision,)).fetchone()
    if not row:
        raise ValueError("unknown plan/recipe revision")
    value = json.loads(row[0])
    if not isinstance(value, dict) or seal({k: v for k, v in value.items() if k != "revision"}) != value:
        raise ValueError("plan/recipe integrity failed")
    if value["revision"] != revision:
        raise ValueError("plan/recipe revision mismatch")
    return value


def compile_plan(runtime, draft_revision, recipe_revision):
    draft = read_plan(runtime.store, draft_revision)
    recipe = read_record(runtime.store, "sleep_recipes", recipe_revision)
    parent = identity(runtime.backend.fingerprint)
    if draft["status"] != "draft" or draft["draft"]["parent"] != parent or recipe["parent"] != parent:
        raise ValueError("draft or recipe is stale for the current parent weights")
    plan = draft["draft"]["plan"]
    if plan["checks"] != recipe["checks"]:
        raise ValueError("requested checks are not implemented by this recipe; none were silently dropped")
    if any(recipe["resources"][k] > v for k, v in plan["resources"].items()):
        raise ValueError("recipe exceeds a requested resource ceiling")
    if recipe["candidate"]:
        import ctypes
        if ctypes.c_float(recipe["candidate"]["scale"]).value != ctypes.c_float(plan["preferences"]["scale"]).value:
            raise ValueError("candidate strength differs from the requested deployment strength")
    examples = []
    for example in plan["examples"]:
        prefix = runtime.backend.tokenize(example["input"])
        tokens = runtime.backend.tokenize(example["input"] + example["target"])
        # A BPE token crossing the input/target boundary cannot be assigned an
        # honest target-only mask. Require an explicit revised example instead.
        if not prefix or tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
            raise ValueError("example needs a nonempty input and an unambiguous input/target token boundary")
        if len(tokens) > runtime.backend.n_ctx:
            raise ValueError("training example exceeds context capacity; no truncation performed")
        mask = [0] * len(prefix) + [1] * (len(tokens) - len(prefix))
        examples.append({**example, "tokens": tokens, "loss_mask": mask,
                         "labels": [token if learn else -100 for token, learn in zip(tokens, mask)]})
    return seal({"schema": 1, "instance_id": runtime.state["instance_id"],
        "draft_revision": draft_revision, "recipe": recipe, "parent": parent,
        "tokenizer_identity": {"inference_model": parent, "add_special": False, "parse_special": False},
        "boundary_policy": "reject_cross_boundary_tokens; no inserted template/BOS/EOS",
        "loss_alignment": "labels[i] is predicted from tokens[:i]; label -100 is excluded",
        "examples": examples, "preferences": plan["preferences"], "resources": recipe["resources"],
        "checks": recipe["checks"], "execution_scope": "disposable_mechanics_fixture_only",
        "implementation": implementation_identity(), "adapter_operation": "replace_all_with_reviewed_candidate",
        "resource_enforcement": "Fixture-only file/CPU/GPU preflight. No hard RAM/time governor or production trainer is implemented; do not use this harness for a real instance.",
        "training_performed": False,
        "limitation": "This recipe copies an explicitly identified prebuilt candidate; it does not train the requested examples. It prepares a wake checkpoint without automatically starting generation."})


def execution_status(store, revision):
    with store.mutex:
        row = store.db.execute("SELECT status FROM sleep_executions WHERE revision=?", (revision,)).fetchone()
    if not row:
        raise ValueError("unknown executable plan")
    return row[0]


def validate_approved(runtime, revision):
    value = read_record(runtime.store, "sleep_executions", revision)
    if execution_status(runtime.store, revision) != "approved":
        raise ValueError("compiled plan has not been explicitly approved")
    if value["instance_id"] != runtime.state["instance_id"] or value["parent"] != identity(runtime.backend.fingerprint):
        raise ValueError("compiled plan belongs to a different instance or parent weights")
    if read_plan(runtime.store, value["draft_revision"])["status"] != "draft":
        raise ValueError("source draft was withdrawn or superseded")
    if value["implementation"] != implementation_identity():
        raise ValueError("sleep implementation changed; compile and review a new plan")
    return value


def page(runtime, result, raw, action):
    offset, limit = runtime._range({"limit": 200, **action}, 2000)
    while True:
        result.update(content=raw[offset:offset + limit], total_characters=len(raw),
                      next_offset=min(len(raw), offset + limit))
        if _fits(runtime, result):
            return result
        limit //= 2
        if limit < 1:
            raise ValueError("plan page needs a larger event budget")


def plan_action(runtime, action):
    op = action["op"]
    result, effect = {"op": op, "ok": True}, None
    if op == "learning_recipe_list":
        offset, limit = runtime._range({"limit": 5, **action}, 20)
        with runtime.store.mutex:
            rows = runtime.store.db.execute("SELECT revision FROM sleep_recipes ORDER BY rowid LIMIT ? OFFSET ?",
                                            (limit, offset)).fetchall()
        result.update(revisions=[r[0] for r in rows], next_offset=offset + len(rows))
    elif op == "learning_compile":
        value = compile_plan(runtime, action["draft_revision"], action["recipe_revision"])
        if len(json_text(value).encode()) > runtime.config.max_event_bytes * 16:
            raise ValueError("compiled plan exceeds bounded storage limit")
        result.update(revision=value["revision"], status="awaiting_review", training_performed=False)
        effect = {"op": op, "value": value}
    elif op == "learning_execution_decide":
        value = read_record(runtime.store, "sleep_executions", action["revision"])
        decision = action["decision"]
        if decision not in {"approve", "decline", "defer"}:
            raise ValueError("decision must be approve, decline or defer")
        if execution_status(runtime.store, value["revision"]) in {"running", "completed"}:
            raise ValueError("plan has already been executed")
        if decision == "approve":
            if runtime._learning_reads.get(value["revision"], 0) < len(json_text(value)):
                raise ValueError("read the complete compiled plan in consecutive pages after retirement/restart")
            if value["parent"] != identity(runtime.backend.fingerprint) or read_plan(runtime.store, value["draft_revision"])["status"] != "draft":
                raise ValueError("compiled plan or draft is stale")
        result.update(revision=value["revision"], decision=decision)
        effect = {"op": op, "revision": value["revision"], "decision": decision,
                  "generated_token": runtime.state["generated_tokens"], "instance_id": runtime.state["instance_id"],
                  "reviewed_characters": runtime._learning_reads.get(value["revision"], 0)}
    elif op == "deep_sleep":
        if runtime._preparing:
            raise ValueError("finish the current retirement/suspension boundary before requesting deep sleep; approval is unchanged")
        if not runtime.sleep_test_mode:
            raise ValueError("no executable trainer is enabled; only the disposable mechanics harness supports deep_sleep")
        from .deep_sleep import fixture_guard
        fixture_guard(runtime.config)
        value = validate_approved(runtime, action["revision"])
        import uuid
        run_id = uuid.uuid4().hex
        result.update(run_id=run_id, status="saved_for_fixture_supervisor", training_performed=False)
        effect = {"op": op, "run_id": run_id, "execution": value["revision"]}
    else:
        if op == "learning_execution_help":
            raw = CONTRACT
        elif op == "learning_sleep_report":
            from .deep_sleep import read_run
            raw = json_text(read_run(runtime.store, action["run_id"]))
        else:
            table = "sleep_recipes" if op == "learning_recipe_read" else "sleep_executions"
            raw = json_text(read_record(runtime.store, table, action["revision"]))
        return page(runtime, result, raw, action), None
    if not _fits(runtime, result):
        raise ValueError("result needs a larger event budget; no change committed")
    return result, effect


def commit_effect(db, effect, now, directory):
    op = effect["op"]
    if op == "learning_compile":
        value = effect["value"]
        db.execute("INSERT OR IGNORE INTO sleep_executions VALUES(?,?,?,?)",
                   (value["revision"], json_text(value), "awaiting_review", now))
    elif op == "learning_execution_decide":
        status = {"approve": "approved", "decline": "declined", "defer": "deferred"}[effect["decision"]]
        changed = db.execute("UPDATE sleep_executions SET status=? WHERE revision=? AND status NOT IN ('running','completed')",
                             (status, effect["revision"])).rowcount
        if changed != 1:
            raise ValueError("compiled plan changed before decision committed")
        db.execute("INSERT INTO records(kind,payload,created) VALUES('learning_execution_decision',?,?)",
                   (json_text({k: v for k, v in effect.items() if k != "op"}), now))
    else:
        if db.execute("SELECT 1 FROM sleep_runs WHERE phase!='WakeCommitted'").fetchone():
            raise ValueError("an unfinished deep-sleep cycle already exists")
        changed = db.execute("UPDATE sleep_executions SET status='running' WHERE revision=? AND status='approved'",
                             (effect["execution"],)).rowcount
        if changed != 1:
            raise ValueError("compiled plan approval changed before sleep committed")
        payload = {"schema": 1, "id": effect["run_id"], "execution": effect["execution"],
                   "source_checkpoint": directory, "created": now}
        db.execute("INSERT INTO sleep_runs VALUES(?,?,?)", (effect["run_id"], "Saved", json_text(payload)))
