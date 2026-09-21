"use strict";
(() => {
  const get = id => document.getElementById(id);
  let key = "", snapshot = null, busy = false;
  const notice = (text, error = false) => { get("notice").textContent = text; get("notice").classList.toggle("error", error); };
  async function api(path, body) {
    const response = await fetch(path, {method: body ? "POST" : "GET", cache: "no-store",
      headers: {"Authorization": "Bearer " + key, "X-DMN-Request": "1", "Content-Type": "application/json"},
      ...(body ? {body: JSON.stringify(body)} : {})});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "Request failed");
    return value;
  }
  function element(tag, text, className) {
    const node = document.createElement(tag); node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function targetChanged() {
    const person = snapshot?.participants.find(p => p.participant_id === get("participant").value);
    get("target").textContent = person ? `${person.participant_id} · block revision ${person.block_revision}` : "No blocked participants";
    get("submit").disabled = busy || !person || !!person.request;
    get("reason").disabled = !person || !!person.request;
    if (person?.request) {
      get("reason").value = person.request.reason;
      notice("This block revision already has a request. Its original reasoning is shown below; no additional request has been sent.");
    } else get("reason").value = "";
  }
  async function refresh() {
    const previous = get("participant").value, draft = get("reason").value;
    snapshot = await api("/api/operator/contacts");
    get("instance").textContent = "Instance " + snapshot.instance_id;
    get("contacts").replaceChildren(); get("participant").replaceChildren();
    for (const person of snapshot.participants) {
      const card = element("article", "", "panel");
      card.append(element("h2", person.display_name));
      if (person.is_operator) card.append(element("div", "Operator account", "badge"));
      card.append(element("p", person.participant_id, "mono"));
      card.append(element("p", person.blocked ? `Blocked · revision ${person.block_revision}` : "Participant is not blocked", "state"));
      card.append(element("p", `Contact consent: ${person.contact_state}`, "small"));
      const chats = element("details", ""); chats.append(element("summary", `${person.conversations.length} conversation(s)`));
      for (const chat of person.conversations) chats.append(element("p", `${chat.conversation_id} · ${chat.closed ? "Closed" : "Open"}`, "mono"));
      card.append(chats);
      if (person.request) {
        const r = person.request;
        card.append(element("p", `Request ${r.event_id} · ${r.delivered ? "delivered to the instance" : "queued; delivery not yet checkpointed"}`, "small"));
        card.append(element("p", r.reason, "request"));
        if (person.blocked) card.append(element("p", "The block remains in place. No change is implied by request delivery.", "small"));
      }
      get("contacts").append(card);
      if (person.blocked) {
        const option = element("option", `${person.display_name}${person.is_operator ? " (operator)" : ""} · ${person.participant_id}`);
        option.value = person.participant_id; get("participant").append(option);
      }
    }
    if (!snapshot.participants.length) get("contacts").append(element("p", "No conversations have been registered yet."));
    if (snapshot.participants.some(p => p.blocked && p.participant_id === previous)) get("participant").value = previous;
    targetChanged();
    if (get("participant").value === previous && !get("reason").disabled) get("reason").value = draft;
  }
  get("login").addEventListener("submit", async event => {
    event.preventDefault(); key = get("key").value; notice("Connecting…");
    try { await refresh(); get("key").value = ""; get("login").hidden = true; get("workspace").hidden = false; notice("Connected to the operator interface."); }
    catch (error) { key = ""; notice(error.message, true); }
  });
  get("refresh").addEventListener("click", async () => { try { await refresh(); notice("Contact state refreshed."); } catch (error) { notice(error.message, true); } });
  get("disconnect").addEventListener("click", () => { key = ""; snapshot = null; get("workspace").hidden = true; get("login").hidden = false; get("contacts").replaceChildren(); get("reason").value = ""; notice("Disconnected."); });
  get("participant").addEventListener("change", targetChanged);
  get("request").addEventListener("submit", async event => {
    event.preventDefault(); if (busy) return;
    const person = snapshot.participants.find(p => p.participant_id === get("participant").value);
    const reason = get("reason").value;
    if (!person || !reason.trim()) { notice("Select a participant and explain your reasoning.", true); return; }
    busy = true; get("submit").disabled = true;
    try {
      const result = await api("/api/operator/unblock-requests", {instance_id: snapshot.instance_id,
        participant_id: person.participant_id, expected_block_revision: person.block_revision, reason});
      await refresh(); notice(`Request ${result.event_id} queued. Chat access is unchanged unless the instance chooses to unblock.`);
    } catch (error) { notice(error.message, true); }
    finally { busy = false; get("submit").disabled = !snapshot?.participants.some(p => p.participant_id === get("participant").value && p.blocked && !p.request); }
  });
})();
