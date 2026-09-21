"""Model-authored learning drafts. No draft authorizes training or adoption."""
from __future__ import annotations

import hashlib
import math

from .storage import json_text, memory_path


HELP = {
    "status": "drafts_only; no trainer, deep-sleep action or weight adoption is available",
    "create": {"op": "learning_plan_create", "plan": {
        "intent": "What I want to learn",
        "uncertainties": "What remains uncertain (may be empty)",
        "exclusions": "What I do not want trained (may be empty)",
        "replay_policy": "What earlier learning to rehearse or deliberately omit",
        "sources": [{"path": "/chosen-memory", "revision": 1, "provenance": "self"}],
        "examples": [{"input": "Exact input text", "target": "Exact desired continuation",
                      "sources": [0], "purpose": "new"}],
        "preferences": {"rank": 2, "alpha": 4, "scale": 0.1, "steps": 16,
                        "adoption": "review_first", "failure": "remain_stopped"},
        "resources": {"max_training_seconds": 300, "max_ram_bytes": 1073741824,
                      "max_vram_bytes": 0, "max_disk_bytes": 1073741824},
        "checks": ["Describe the checks I want before adoption"]}, "replaces": None},
    "semantics": [
        "The example values above illustrate syntax, not recommended training parameters.",
        "sources may be empty for newly authored examples. Revisions are exact; selected text and its SHA-256 are frozen in the draft.",
        "provenance: self, external, mixed, or uncertain; declarations are not a trust verdict.",
        "Each example references source indices; purpose is new or replay. Only target text is intended for loss, never input text or entire source memories.",
        "Exact token IDs, tokenizer identity, boundary handling and token loss masks still require a compiled recipe and explicit review before execution.",
        "adoption: review_first or automatic_if_checks_pass; failure: remain_stopped or wake_previous. These are draft preferences, not execution consent.",
        "Resources are requested ceilings, not reservations or measured feasibility. A future executable plan must also respect host resource limits.",
        "replaces optionally names an active draft revision to supersede atomically. Withdraw with learning_plan_withdraw(revision). Both preserve history.",
        "Plans are immutable, private local records; they are included in instance packaging and removed by managed erasure. Machine owners can access them.",
        "Read with learning_plan_read(revision, offset=0, limit=200); list with learning_plan_list(offset=0, limit=20).",
        "A future execution request must bind a trainer/converter recipe, base and parent identities, exact examples/masks, resource ceilings, checks and decisions."]}


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(label + " requires exactly: " + names)


def _text(value, label, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(label + " must be " + ("nonempty text" if nonempty else "text"))


def create_plan(plan, store, fingerprint, instance_id, generated_token, max_bytes, replaces=None):
    _fields(plan, "intent uncertainties exclusions replay_policy sources examples preferences resources checks", "plan")
    for name in ("intent", "uncertainties", "exclusions", "replay_policy"):
        _text(plan[name], name, name in {"intent", "replay_policy"})
    if not isinstance(plan["sources"], list) or len(plan["sources"]) > 32:
        raise ValueError("sources must be a list of at most 32 explicit memory revisions")
    sources, source_bytes = [], 0
    for source in plan["sources"]:
        _fields(source, "path revision provenance", "source")
        path, revision = memory_path(source["path"]), source["revision"]
        if type(revision) is not int or revision < 1:
            raise ValueError("source revision must be a positive integer")
        if source["provenance"] not in ("self", "external", "mixed", "uncertain"):
            raise ValueError("source provenance must be self, external, mixed or uncertain")
        text = store.memory_read(path, revision)
        source_bytes += len(text.encode("utf-8"))
        if source_bytes > max_bytes:
            raise ValueError("selected source text exceeds max_event_bytes; select smaller dedicated memories")
        sources.append({**source, "path": path, "content": text, "sha256": digest_text(text)})
    examples = plan["examples"]
    if not isinstance(examples, list) or not 1 <= len(examples) <= 64:
        raise ValueError("examples must contain 1 to 64 exact input/target pairs")
    for example in examples:
        _fields(example, "input target sources purpose", "example")
        _text(example["input"], "example input")
        _text(example["target"], "example target", True)
        if (not isinstance(example["sources"], list) or any(type(i) is not int or not 0 <= i < len(sources)
                                                         for i in example["sources"])):
            raise ValueError("example sources must index the selected sources")
        if example["purpose"] not in ("new", "replay"):
            raise ValueError("example purpose must be new or replay")
    prefs = plan["preferences"]
    _fields(prefs, "rank alpha scale steps adoption failure", "preferences")
    for key in ("rank", "steps"):
        if type(prefs[key]) is not int or prefs[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("alpha", "scale"):
        if type(prefs[key]) not in (int, float) or not math.isfinite(prefs[key]):
            raise ValueError(key + " must be finite")
    if prefs["alpha"] <= 0:
        raise ValueError("alpha must be positive")
    if prefs["adoption"] not in ("review_first", "automatic_if_checks_pass"):
        raise ValueError("adoption must be review_first or automatic_if_checks_pass")
    if prefs["failure"] not in ("remain_stopped", "wake_previous"):
        raise ValueError("failure must be remain_stopped or wake_previous")
    _fields(plan["resources"], "max_training_seconds max_ram_bytes max_vram_bytes max_disk_bytes", "resources")
    for key, value in plan["resources"].items():
        if type(value) is not int or value < (0 if key == "max_vram_bytes" else 1):
            raise ValueError(key + " must be a positive integer (VRAM may be zero)")
    if not isinstance(plan["checks"], list) or not plan["checks"]:
        raise ValueError("checks must be a nonempty list of descriptions")
    for item in plan["checks"]:
        _text(item, "check", True)
    if replaces is not None:
        prior = read_plan(store, replaces)
        if prior["status"] != "draft":
            raise ValueError("only an active draft can be replaced")
    # Bind the current inference identity, not guessed BF16 training ancestry.
    parent = {key: fingerprint[key] for key in ("kind", "model_sha256", "script_sha256", "research_lora")
              if key in fingerprint}
    parent["lora_adapters"] = fingerprint.get("lora_adapters", [])
    value = {"schema": 1, "author": "model", "instance_id": instance_id,
             "generated_token": generated_token, "parent": parent, "replaces": replaces,
             "execution_authorized": False, "loss_policy": "target_text_only_pending_tokenization",
             "plan": {**plan, "sources": sources}}
    encoded = json_text(value)
    if len(encoded.encode()) > max_bytes:
        raise ValueError("complete plan with frozen sources exceeds max_event_bytes")
    return {**value, "revision": digest_text(encoded)}


def read_plan(store, revision):
    if not isinstance(revision, str):
        raise ValueError("revision must be a string")
    with store.mutex:
        row = store.db.execute("SELECT payload,status FROM learning_plans WHERE revision=?", (revision,)).fetchone()
    if row is None:
        raise ValueError("unknown learning plan revision")
    import json
    value = json.loads(row["payload"])
    if digest_text(json_text({k: v for k, v in value.items() if k != "revision"})) != revision or value["revision"] != revision:
        raise ValueError("learning plan integrity failed")
    return {"draft": value, "status": row["status"]}


def list_plans(store, offset, limit):
    with store.mutex:
        return [dict(row) for row in store.db.execute(
            "SELECT revision,status,created FROM learning_plans ORDER BY rowid LIMIT ? OFFSET ?", (limit, offset))]


def commit_effect(db, effect, now):
    if effect["op"] == "learning_plan_create":
        value = effect["value"]
        if value["replaces"]:
            changed = db.execute("UPDATE learning_plans SET status='superseded' WHERE revision=? AND status='draft'",
                                 (value["replaces"],)).rowcount
            if changed != 1:
                raise ValueError("learning plan changed before replacement committed")
        db.execute("INSERT INTO learning_plans VALUES(?,?,?,?)",
                   (value["revision"], json_text(value), "draft", now))
    else:
        changed = db.execute("UPDATE learning_plans SET status='withdrawn' WHERE revision=? AND status='draft'",
                             (effect["revision"],)).rowcount
        if changed != 1:
            raise ValueError("learning plan is no longer an active draft")


def plan_action(runtime, action):
    op = action["op"]
    result, effect = {"op": op, "ok": True}, None
    if op == "learning_plan_create":
        value = create_plan(action["plan"], runtime.store, runtime.backend.fingerprint,
                            runtime.state["instance_id"], runtime.state["generated_tokens"],
                            runtime.config.max_event_bytes, action.get("replaces"))
        effect = {"op": op, "value": value}
        result.update(revision=value["revision"], status="draft", execution_authorized=False)
    elif op == "learning_plan_withdraw":
        if read_plan(runtime.store, action["revision"])["status"] != "draft":
            raise ValueError("only an active draft can be withdrawn")
        effect = {"op": op, "revision": action["revision"]}
        result.update(revision=action["revision"], status="withdrawn")
    elif op == "learning_plan_list":
        offset, limit = runtime._range({"limit": 20, **action}, 50)
        rows = list_plans(runtime.store, offset, limit)
        # Return only complete entries; never truncate a revision identifier.
        while rows and not _fits(runtime, {**result, "plans": rows, "next_offset": offset + len(rows)}):
            rows.pop()
        if not rows and list_plans(runtime.store, offset, 1):
            raise ValueError("learning plan list needs a larger event budget")
        result.update(plans=rows, next_offset=offset + len(rows))
    else:
        raw = json_text(HELP if op == "learning_plan_help" else read_plan(runtime.store, action["revision"]))
        offset, limit = runtime._range({"limit": 200, **action}, 2000)
        while True:
            result.update(content=raw[offset:offset + limit], total_characters=len(raw),
                          next_offset=min(len(raw), offset + limit))
            if _fits(runtime, result):
                break
            limit //= 2
            if limit < 1:
                raise ValueError("learning plan page needs a larger event budget")
    if not _fits(runtime, result):
        raise ValueError("learning plan result needs a larger event budget; no change committed")
    return result, effect


def _fits(runtime, result):
    from .protocol import event_text
    return len(runtime.backend.tokenize(event_text("action_result", result, runtime.now(),
        resume_cognition=True))) <= runtime._event_budget()


OPERATIONS = {"learning_plan_help", "learning_plan_create", "learning_plan_read",
              "learning_plan_list", "learning_plan_withdraw"}
