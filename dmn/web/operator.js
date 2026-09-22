"use strict";
(() => {
  const get = id => document.getElementById(id);
  let key = "", snapshot = null, busy = false, refreshing = false, runtimeState = null;
  let promptBase = null, currentRevision = null, promptSubmission = null, promptBusy = false;
  const number = value => Number.isFinite(value) ? value.toLocaleString() : 'Unavailable';
  const duration = value => !Number.isFinite(value) ? 'unknown' : value < 60 ? `${Math.floor(value)}s` : value < 3600 ? `${Math.floor(value / 60)}m` : `${(value / 3600).toFixed(1)}h`;
  const bytes = value => {
    if (!Number.isFinite(value)) return 'Unavailable';
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let unit = 0;
    while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
    return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
  };
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
  function renderStatus(runtime) {
    runtimeState = runtime;
    get('runtime-status').textContent = `${runtime.mode} · ${number(runtime.active_tokens)} / ${number(runtime.context_capacity)} tokens · ${number(runtime.generated_tokens)} generated` +
      (runtime.sleep_service ? ` · ${runtime.sleep_service.phase}` : '') +
      (runtime.mode === 'awaiting_first_contact' ? ' · Waiting for your first contact request in Open WebUI. No generation has started.' : '');
    get('context-usage').value = runtime.active_tokens / runtime.context_capacity;
    const retirement = runtime.context_retirement, activity = runtime.activity, diagnostic = runtime.action_diagnostics || {};
    get('retirement-status').textContent = retirement ? `Retirement threshold: ${number(retirement.threshold_tokens)} · ${number(retirement.tokens_until_threshold)} tokens remaining · ${number(retirement.completed)} completed. Incoming events may trigger it earlier.` + (runtime.native_context_retirement_supported === false ? ' Native retirement unavailable; the runtime saves and pauses.' : '') : '';
    get('activity-status').textContent = activity ? `Activity: ${activity.mode}` + (activity.mode === 'idle' ? ` · ${number(activity.burst_tokens)} tokens per burst · ${duration(activity.interval_seconds)} between bursts` : '') : '';
    get('continuity-status').textContent = runtime.continuity ? `Continuity: ${runtime.continuity.replaceAll('_', ' ')}` : '';
    const learning = runtime.learning;
    get('learning-status').textContent = learning ? 'Learning records: ' + Object.entries(learning).map(([kind, states]) => `${kind.replaceAll('_', ' ')}: ${Object.entries(states).map(([state, count]) => `${number(count)} ${state}`).join(', ') || 'none'}`).join(' · ') + '. Ordinary sleep does not run training.' : '';
    get('accepted-actions').textContent = number(diagnostic.accepted_actions);
    get('rejected-actions').textContent = number(diagnostic.rejected_actions || 0);
    get('rejected-messages').textContent = number(diagnostic.rejected_message_attempts || 0);
    get('interrupted-frames').textContent = number(diagnostic.interrupted_frames || 0);
    get('published-messages').textContent = number(runtime.published_messages);
    get('diagnostic-detail').textContent = diagnostic.accepted_count_started_at_generated_token != null ? `Accepted-action counting began at generated token ${number(diagnostic.accepted_count_started_at_generated_token)}. Rejection counters cover the period since diagnostics were enabled.` : 'This running version does not record accepted-action counts. Rejections and interruptions are shown since diagnostics were enabled.';
    const problem = diagnostic.last_problem;
    get('diagnostic-problem').textContent = problem ? `Last recorded problem: ${problem.category} · ${problem.operation} · generated token ${number(problem.generated_token)}. No command arguments or private contents are shown.` : 'No command problem recorded.';
    get('checkpoint-time').textContent = runtime.checkpoint_at ? new Date(runtime.checkpoint_at * 1000).toLocaleString() : 'Not recorded';
    const checkpoint = runtime.checkpoint;
    get('checkpoint-unsaved').textContent = checkpoint ? number(checkpoint.unsaved_generated_tokens) : 'Unavailable';
    get('checkpoint-written').textContent = bytes(checkpoint?.committed_snapshot_bytes);
    get('checkpoint-counts').textContent = checkpoint ? `${number(checkpoint.committed_count)} / ${number(checkpoint.failed_count)}` : 'Unavailable';
    get('checkpoint-detail').textContent = checkpoint ? (checkpoint.in_progress ? 'Saving native state… ' : '') + `Last reason: ${runtime.checkpoint_reason || 'unknown'}. ` + (checkpoint.dirty ? `Unsaved changes for ${duration(checkpoint.unsaved_seconds)}. ` : 'State checkpointed. ') + (checkpoint.last ? `Last snapshot: ${bytes(checkpoint.last.snapshot_bytes)} in ${duration(checkpoint.last.duration_seconds)}.` : '') : `Last reason: ${runtime.checkpoint_reason || 'unknown'}. Detailed checkpoint metrics will be available after the next consented restart with the updated backend.`;
    const limits = checkpoint ? [checkpoint.interval_seconds ? duration(checkpoint.interval_seconds) : null, checkpoint.token_limit ? `${number(checkpoint.token_limit)} generated tokens` : null].filter(Boolean) : [];
    get('checkpoint-policy').textContent = checkpoint ? `Periodic limit: ${limits.join(' or ') || 'disabled'}. ` + (checkpoint.policy === 'effects' ? 'Durable effects also checkpoint. ' : 'Actions and delivered inputs also checkpoint. ') + (checkpoint.sleep_min_interval_seconds ? `Ordinary-sleep snapshot cooldown: ${duration(checkpoint.sleep_min_interval_seconds)}. ` : '') + (checkpoint.sleep_save_in_seconds != null ? `Sleep snapshot due in ${duration(checkpoint.sleep_save_in_seconds)}.` : '') : '';
    const blocked = runtime.storage?.blocked;
    get('checkpoint-storage').textContent = blocked ? `Inference paused for storage: ${bytes(blocked.free_bytes)} free; ${bytes(blocked.required_bytes)} needed including reserve. The checkpoint has not completed.` : checkpoint?.failed_count ? `${number(checkpoint.failed_count)} checkpoint attempt(s) failed during this process.` : '';
    get('maintenance-status').textContent = runtime.maintenance ? `Maintenance: ${runtime.maintenance.status}` : '';
    get('refresh-time').textContent = `Updated ${new Date().toLocaleTimeString()} · refreshes every 5 seconds while visible`;
    get('prompt-availability').textContent = runtime.operator_ui_version >= 2 ? '' : 'Prompt controls require the updated backend at the next consented restart. Syllas can continue this run.';
    get('refresh-prompts').disabled = !(runtime.operator_ui_version >= 2);
    updatePromptControls();
  }
  function updatePromptControls() {
    const operator = snapshot?.participants.find(person => person.is_operator);
    const available = key && runtimeState?.operator_ui_version >= 2 && promptSubmission?.allowed && operator && !operator.blocked && operator.contact_state === 'accepted' && operator.conversations.some(chat => !chat.closed) && !['held', 'ending', 'ended', 'end_failed', 'awaiting_first_contact'].includes(runtimeState.mode);
    get('prompt-text').disabled = !available || promptBusy;
    get('prompt-conversation').disabled = !available || promptBusy;
    get('propose-prompt').disabled = !available || promptBusy || !promptBase || promptBase !== currentRevision;
    get('prompt-base').textContent = promptBase ? `Draft base: ${promptBase}` : '';
    get('prompt-rebase').hidden = !promptBase || promptBase === currentRevision;
  }
  async function refreshPrompts() {
    if (!key || !(runtimeState?.operator_ui_version >= 2)) return;
    const sessionKey = key, value = await api('/api/operator/prompts');
    if (sessionKey !== key) return;
    const active = value.active;
    currentRevision = active?.revision;
    if (!promptBase || !get('prompt-text').value) promptBase = currentRevision;
    promptSubmission = value.submission;
    get('prompt-state').textContent = `${active?.status || 'unknown'} · ${currentRevision || ''}` + (value.pending ? ' · Decision awaiting durable checkpoint.' : '') + (promptSubmission?.reason ? ` · ${promptSubmission.reason}` : '');
    get('active-prompt').textContent = active ? active.text ?? ((Array.isArray(active.base) ? active.base.map(item => `${item.role}: ${item.content}`).join('\n\n') : active.base || '') + '\n\nDMN guidance (provisional):\n' + (active.dmn_guidance || '')) : 'No active agreement available.';
    get('prompt-history').replaceChildren();
    for (const proposal of value.proposals) {
      const card = element('details', '', 'prompt-card');
      card.append(element('summary', `${proposal.status} · ${proposal.author} · ${proposal.revision}`), element('pre', proposal.text));
      get('prompt-history').append(card);
    }
    const selected = get('prompt-conversation').value;
    get('prompt-conversation').replaceChildren();
    for (const chat of promptSubmission.conversations) { const option = element('option', chat); option.value = chat; get('prompt-conversation').append(option); }
    if (promptSubmission.conversations.includes(selected)) get('prompt-conversation').value = selected;
    updatePromptControls();
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
    if (refreshing || !key) return;
    refreshing = true;
    try {
    const sessionKey = key;
    const previous = get("participant").value, draft = get("reason").value;
    const [contacts, runtime] = await Promise.all([api('/api/operator/contacts'), api('/api/operator/status')]);
    if (sessionKey !== key) return;
    snapshot = contacts;
    renderStatus(runtime);
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
    } finally { refreshing = false; }
  }
  get("login").addEventListener("submit", async event => {
    event.preventDefault(); key = get("key").value; notice("Connecting…");
    try { await refresh(); get("key").value = ""; get("login").hidden = true; get("workspace").hidden = false; notice("Connected to the operator interface."); }
    catch (error) { key = ""; notice(error.message, true); }
  });
  get("refresh").addEventListener("click", async () => { try { await refresh(); notice("Contact state refreshed."); } catch (error) { notice(error.message, true); } });
  get("disconnect").addEventListener("click", () => { key = ""; snapshot = null; runtimeState = null; promptBase = currentRevision = promptSubmission = null; get("workspace").hidden = true; get("login").hidden = false; get("contacts").replaceChildren(); get("reason").value = ""; get('active-prompt').textContent = ''; get('prompt-history').replaceChildren(); get('prompt-text').value = ''; notice("Disconnected."); });
  get("participant").addEventListener("change", targetChanged);
  get('maintenance').addEventListener('submit', async event => {
    event.preventDefault();
    try {
      await api('/api/operator/maintenance', {instance_id: snapshot.instance_id, action: 'shutdown', reason: get('maintenance-reason').value});
      notice('Shutdown request submitted. During an active run, the instance decides when and whether to accept.');
    } catch (error) { notice(error.message, true); }
  });
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
  get('prompts').addEventListener('toggle', () => { if (get('prompts').open) refreshPrompts().catch(error => notice(error.message, true)); });
  get('refresh-prompts').addEventListener('click', () => refreshPrompts().catch(error => notice(error.message, true)));
  get('prompt-rebase').addEventListener('click', () => { promptBase = currentRevision; updatePromptControls(); notice('Draft now references the displayed active agreement. Review the wording before submitting.'); });
  get('prompt-form').addEventListener('submit', async event => {
    event.preventDefault(); if (promptBusy || get('propose-prompt').disabled) return;
    promptBusy = true; updatePromptControls();
    try {
      await api('/api/operator/prompts', {instance_id: snapshot.instance_id, text: get('prompt-text').value, base_revision: promptBase, conversation_id: get('prompt-conversation').value});
      get('prompt-text').value = ''; await refreshPrompts(); notice('Proposal queued for review. The active agreement changes only after the instance approves it and its checkpoint commits.');
    } catch (error) { notice(error.message, true); }
    finally { promptBusy = false; updatePromptControls(); }
  });
  setInterval(() => { if (key && !document.hidden && !busy && !promptBusy) refresh().catch(error => notice(`Refresh failed; displayed values may be stale: ${error.message}`, true)); }, 5000);
})();
