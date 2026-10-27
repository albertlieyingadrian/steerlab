// The admin desk: a live mirror of every conversation (from /admin/events) and the controls (JSON POSTs).
// The mirror is rebuilt from the same events the server's broker applies, starting from its snapshot.
// Model output, visitor text, prompts and labels are always inserted with textContent.
"use strict";

const $ = (id) => document.getElementById(id);
const view = { state: {}, conversations: new Map(), notices: [], activity: new Map() };
let selected = null;
let picked = false;  // the presenter chose a tab; until then the mirror follows whichever reply is being written
let promptDirty = false;
let promptSeen = null;
let steerSignature = "";
let renderQueued = false;
let mirrorOnly = true;  // token and marker events change only the mirror; anything else redraws the whole desk

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function signed(value) {
  const number = Math.round(Number(value) * 100) / 100;  // 1.25 stays 1.25; 1.5 stays 1.5
  return (number > 0 ? "+" : "") + number;
}


function conversation(id) {
  if (!view.conversations.has(id)) view.conversations.set(id, { id, label: view.conversations.size + 1, turns: [] });
  return view.conversations.get(id);
}

function apply(event) {
  const kind = event.type;
  if (kind === "snapshot") {
    view.state = event.state;
    view.conversations = new Map(event.conversations.map((item) => [item.id, item]));
    view.notices = event.notices;
  } else if (kind === "state") {
    const { type, ...state } = event;
    view.state = state;
  } else if (kind === "notice") {
    view.notices = [...view.notices, { ...event, at: new Date() }].slice(-20);
  } else if (kind === "reset") {
    for (const item of view.conversations.values()) item.turns = [];
  } else if (kind === "new") {
    conversation(event.conversation).turns = [];
  } else if (kind === "user") {
    conversation(event.conversation).turns.push({ role: "user", text: event.text });
  } else if (kind === "start") {
    const item = conversation(event.conversation);
    const reply = { role: "assistant", tokens: [], markers: [], finish: null };
    if (item.replaced) { reply.original = item.replaced; delete item.replaced; }
    item.turns.push(reply);
  } else if (["token", "marker", "end", "redo"].includes(kind)) {
    const item = conversation(event.conversation);
    const reply = item.turns[item.turns.length - 1];
    if (!reply || reply.role !== "assistant") return;
    if (kind === "token") reply.tokens.push([event.index, event.text]);
    else if (kind === "marker") reply.markers.push(event);
    else if (kind === "redo") { item.turns.pop(); delete reply.original; item.replaced = reply; }
    else if (event.dropped) {
      item.turns.pop();
      const last = item.turns[item.turns.length - 1];
      if (last && last.role === "user" && !event.keep_user) item.turns.pop();
    } else reply.finish = event.reason;
  }
  if (event.conversation) view.activity.set(event.conversation, Date.now());
  if (kind === "start" && (!picked || selected === null)) selected = event.conversation;
}


function markerText(marker) {
  if (!marker.applied.length) return `token ${marker.token_index} · steering off`;
  const parts = marker.applied.map((entry) => `${entry.label} ${signed(entry.strength)}`);
  const layers = [...new Set(marker.applied.flatMap((entry) => entry.layers))].join(", ");
  const norm = Object.values(marker.added_norm || {}).reduce((sum, value) => sum + value, 0);
  return `token ${marker.token_index} · ${parts.join(" + ")} · layer ${layers}` + (norm ? ` · +${norm.toFixed(1)}` : "");
}

function renderReply(reply) {
  const box = el("div", "a");
  const markersAt = new Map();
  for (const marker of reply.markers) {
    if (!markersAt.has(marker.token_index)) markersAt.set(marker.token_index, []);
    markersAt.get(marker.token_index).push(marker);
  }
  let steered = false;
  let anySteered = false;
  let steeredSince = null;
  for (const [index, text] of reply.tokens) {
    for (const marker of markersAt.get(index) || []) {
      if (!marker.applied.length && !steered) continue;  // "off" only means something after steering was on
      box.appendChild(el("span", "mk" + (marker.applied.length ? "" : " off"), markerText(marker)));
      steered = marker.applied.length > 0;
      if (steered && steeredSince === null) steeredSince = index;
    }
    markersAt.delete(index);
    if (!text) continue;
    if (steered) {
      anySteered = true;
      box.appendChild(el("span", "tok", text));
    } else box.appendChild(document.createTextNode(text));
  }
  for (const markers of markersAt.values()) {  // a change on the end-of-sequence step
    for (const marker of markers) box.appendChild(el("span", "mk", markerText(marker) + " · changed at end"));
  }
  if (anySteered) box.classList.add("steered");
  if (reply.finish === null) box.appendChild(el("span", "caret"));
  const count = reply.tokens.length;
  const finish = {
    eos: `${count} tokens`, length: `${count} tokens · hit the length limit`, stopped: `Stopped by the admin · ${count} tokens`,
    full: "Conversation full", failed: "The reply failed",
  }[reply.finish] || `Writing now · ${count} tokens`;
  const meta = el("div", "meta", finish + (steeredSince !== null ? ` · steered from token ${steeredSince}` : " · steering off"));
  return [box, meta];
}

function ordered() {
  const active = view.state.active;
  return [...view.conversations.values()].sort((a, b) => {
    if (a.id === active) return -1;
    if (b.id === active) return 1;
    return (view.activity.get(b.id) || 0) - (view.activity.get(a.id) || 0);
  });
}

function renderTabs() {
  const tabs = $("convs");
  tabs.replaceChildren();
  for (const item of ordered()) {
    const tab = el("button", "", `Visitor ${item.label}`);
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(item.id === selected));
    if (item.id === view.state.active) tab.appendChild(el("span", "writing-dot", "● writing"));
    tab.addEventListener("click", () => { selected = item.id; picked = true; follow = true; mirrorOnly = false; render(); });
    tabs.appendChild(tab);
  }
}

let follow = true;
$("mirror").addEventListener("scroll", () => {
  const mirror = $("mirror");
  follow = mirror.scrollHeight - mirror.scrollTop - mirror.clientHeight < 80;
});

function renderMirror() {
  const mirror = $("mirror");
  const item = view.conversations.get(selected);
  const list = el("div", "msgs");
  if (!item) {
    const empty = el("div", "empty");
    empty.append("No visitors yet. Open the ");
    const link = el("a", "", "visitor page");
    link.href = "/";
    link.target = "_blank";
    empty.append(link, " and send a message.");
    mirror.replaceChildren(empty);
    $("mirroring").textContent = "No visitor selected";
    return;
  }
  for (const turn of item.turns) {
    if (turn.role === "user") { list.appendChild(el("div", "u", turn.text)); continue; }
    if (turn.original) {
      const grid = el("div", "compare");
      const left = el("div");
      const right = el("div");
      left.append(el("h4", "", "Original"), ...renderReply(turn.original));
      right.append(el("h4", "", "With current steering"), ...renderReply(turn));
      grid.append(left, right);
      list.appendChild(grid);
    } else list.append(...renderReply(turn));
  }
  if (!item.turns.length) list.appendChild(el("div", "notice", "New conversation"));
  mirror.replaceChildren(list);
  if (follow) mirror.scrollTop = mirror.scrollHeight;
  $("mirroring").textContent = `Mirroring Visitor ${item.label} · read only`;
}

function activeSources() {
  const state = view.state;
  const directions = (state.directions || []).map((d) => ({ name: d.name, label: d.label, strength: d.strength }));
  const features = (state.features || []).map((f) => ({ name: `feature:${f.id}`, label: f.label || `Feature ${f.id}`, strength: f.strength }));
  return [...directions, ...features];
}

function renderChips() {
  const chips = $("chips");
  chips.replaceChildren();
  for (const source of activeSources().filter((s) => s.strength !== 0)) {
    const chip = el("span", "chip");
    const off = el("button", "", "×");
    off.setAttribute("aria-label", `Turn off ${source.label}`);
    off.addEventListener("click", () => post("/admin/api/control", { name: source.name, strength: 0 }));
    chip.append(el("b", "", signed(source.strength)), source.label, off);
    chips.appendChild(chip);
  }
  chips.appendChild(el("span", "hint", "Changes apply from the next token"));
}

function fader({ name, title, meta, strength, min, max, step, removable, suggested }) {
  const card = el("div", "card" + (strength !== 0 ? " on" : ""));
  card.dataset.source = name;
  const heading = el("h3", "", title);
  if (removable) {
    const remove = el("button", "", "×");
    remove.setAttribute("aria-label", `Remove ${title}`);
    remove.addEventListener("click", () => post("/admin/api/feature/remove", { id: removable }));
    heading.appendChild(remove);
  }
  card.append(heading, el("p", "", meta));
  if (suggested !== null && suggested !== undefined) {  // the strength the autoresearch study found works on this model
    const use = el("button", "suggest", `Use tuned ${signed(suggested)}`);
    use.addEventListener("click", () => post("/admin/api/control", { name, strength: suggested }));
    card.appendChild(use);
  }
  const row = el("div", "row");
  const range = el("input");
  Object.assign(range, { type: "range", min, max, step, value: strength });
  range.setAttribute("aria-label", `${title} strength`);
  const box = el("input", "val" + (strength !== 0 ? " on" : ""));
  Object.assign(box, { type: "number", step, value: strength });
  box.setAttribute("aria-label", `${title} strength value`);
  const commit = (value) => post("/admin/api/control", { name, strength: Number(value) });
  range.addEventListener("input", () => { box.value = range.value; });
  range.addEventListener("change", () => commit(range.value));
  range.addEventListener("keydown", (event) => {
    if (!event.shiftKey || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const delta = ["ArrowRight", "ArrowUp"].includes(event.key) ? 0.5 : -0.5;
    range.value = Math.min(max, Math.max(min, Number(range.value) + delta));
    box.value = range.value;
    commit(range.value);
  });
  box.addEventListener("change", () => commit(box.value));
  row.append(el("span", "", String(min)), range, el("span", "", signed(max)), box);
  card.appendChild(row);
  return card;
}

function renderSteer() {
  const state = view.state;
  const signature = JSON.stringify([state.model, (state.directions || []).map((d) => d.name), (state.features || []).map((f) => [f.id, f.label]), !!state.sae]);
  if (signature === steerSignature) return syncFaders();  // same cards: only the values move
  steerSignature = signature;
  const directions = $("directions");
  directions.replaceChildren(el("div", "section", "Directions"));
  if (!(state.directions || []).length) {
    directions.appendChild(el("p", "muted", `No saved directions for ${state.model || "this model"}. Build one with steerlab directions build --model ${state.model || "KEY"}.`));
  }
  for (const d of state.directions || []) {
    const limit = Math.max(3, Math.ceil(Math.abs(d.suggested || 0) * 1.5));
    directions.appendChild(fader({ name: d.name, title: d.label, meta: `${d.name} · layer ${d.layer}`, strength: d.strength, min: -limit, max: limit, step: 0.1, suggested: d.suggested }));
  }
  const features = $("features");
  features.replaceChildren();
  if (!state.sae) {
    features.append(el("div", "section", "SAE features"), el("p", "muted", "This model has no SAE loaded."));
    return;
  }
  const release = state.sae.release.split("/").pop();
  const header = el("div", "section");
  header.append(el("span", "", `SAE features · layer ${state.sae.layer}`), el("span", "", `${state.sae.features.toLocaleString()} features`));
  const form = el("form", "addfeat");
  const id = el("input");
  Object.assign(id, { type: "number", min: 0, max: state.sae.features - 1, placeholder: "Feature id, e.g. 24946", required: true });
  id.setAttribute("aria-label", "Feature id");
  const add = el("button", "", "Add feature");
  add.type = "submit";
  add.disabled = (state.features || []).length >= state.max_features;
  form.append(id, add);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    post("/admin/api/feature", { id: Number(id.value) });
    id.value = "";
  });
  features.append(header, form);
  for (const f of state.features || []) {
    const clamp = f.mode === "clamp";  // a clamped feature: the value is the activation it is held at, not a norm added
    features.appendChild(fader({
      name: `feature:${f.id}`, title: f.label || `Feature ${f.id}`,
      meta: `feature ${f.id} · layer ${state.sae.layer} · ${release} · ${clamp ? "clamped: value = activation held" : "strength = norm added"}`,
      strength: f.strength, min: clamp ? -20 : -100, max: clamp ? 20 : 100, step: clamp ? 0.5 : 1, removable: f.id, suggested: f.suggested,
    }));
  }
  if ((state.features || []).length >= state.max_features) features.appendChild(el("p", "muted", `At most ${state.max_features} features; remove one to add another.`));
}

function syncFaders() {
  for (const source of activeSourcesAll()) {
    const card = document.querySelector(`[data-source="${CSS.escape(source.name)}"]`);
    if (!card) continue;
    const [range, box] = card.querySelectorAll("input");
    card.classList.toggle("on", source.strength !== 0);
    box.classList.toggle("on", source.strength !== 0);
    if (document.activeElement !== range && document.activeElement !== box) {
      range.value = source.strength;
      box.value = source.strength;
    }
  }
}

function activeSourcesAll() {
  return [
    ...(view.state.directions || []).map((d) => ({ name: d.name, strength: d.strength })),
    ...(view.state.features || []).map((f) => ({ name: `feature:${f.id}`, strength: f.strength })),
  ];
}

function renderPrompt() {
  const prompt = view.state.prompt;
  if (prompt === undefined) return;
  if (!promptDirty) $("prompt").value = prompt;
  $("prompt-warn").hidden = !(promptDirty && promptSeen !== null && prompt !== promptSeen);
  if (!promptDirty) promptSeen = prompt;
}

function renderModels() {
  const list = $("models");
  list.replaceChildren();
  const busy = view.state.status !== "ready" && view.state.status !== "failed";
  for (const model of view.state.models || []) {
    const card = el("div", "card model");
    const text = el("div");
    text.style.flex = "1";
    text.append(el("strong", "", model.key), el("p", "", model.model));
    const details = `${model.directions.length} direction${model.directions.length === 1 ? "" : "s"}` + (model.sae ? ` · SAE layer ${model.sae.layer}` : " · no SAE");
    text.appendChild(el("p", "", details));
    const current = model.key === view.state.model;
    const load = el("button", "", current ? "Loaded" : "Load");
    load.disabled = current || busy;
    load.addEventListener("click", () => post("/admin/api/model", { key: model.key }));
    card.append(text, load);
    list.appendChild(card);
  }
}

function renderHeader() {
  const state = view.state;
  $("model-name").textContent = state.model || "…";
  const status = $("status");
  const words = { ready: "ready", loading: `loading ${state.model}…`, swapping: `loading ${state.model}…`, restoring: `restoring ${state.model}…`, failed: "failed to load" };
  status.textContent = words[state.status] || "connecting";
  status.className = state.status === "ready" ? "live" : state.status === "failed" ? "failed" : "busy";
  $("tab-steer").textContent = `Steer (${activeSources().filter((s) => s.strength !== 0).length})`;
  $("stop").disabled = !state.active;
  const item = view.conversations.get(selected);
  const last = item && item.turns[item.turns.length - 1];
  $("redo").disabled = !(last && last.role === "assistant" && last.finish !== null) || !!state.active || state.status !== "ready";
  const layers = [...new Set((state.directions || []).map((d) => d.layer))];
  if (state.sae) layers.push(state.sae.layer);
  $("foot").textContent = `Steering is applied inside the model${layers.length ? ` at layer ${[...new Set(layers)].join(", ")}` : ""}; the visitor sees only the reply.`;
}

function renderNotices() {
  const notes = $("notes");
  notes.replaceChildren();
  for (const notice of [...view.notices].reverse()) {
    const line = el("div", notice.level === "error" ? "error" : "");
    const at = notice.at ? new Date(notice.at) : null;
    if (at) line.appendChild(el("em", "", at.toTimeString().slice(0, 5)));
    line.append(notice.message);
    notes.appendChild(line);
  }
}

function render() {
  const onlyMirror = mirrorOnly;
  renderQueued = false;
  mirrorOnly = true;
  if (onlyMirror && view.conversations.has(selected)) return renderMirror();
  const first = ordered()[0];
  if (selected === null || !view.conversations.has(selected)) selected = first ? first.id : null;
  renderHeader();
  renderTabs();
  renderChips();
  renderMirror();
  renderSteer();
  renderPrompt();
  renderModels();
  renderNotices();
}

function schedule(kind) {
  if (kind !== "token" && kind !== "marker") mirrorOnly = false;
  if (!renderQueued) { renderQueued = true; requestAnimationFrame(render); }
}


async function post(path, body) {
  let response;
  try {
    response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  } catch (_) {
    view.notices.push({ level: "error", message: "The server did not answer.", at: new Date() });
    schedule("all");
    return false;
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    if (typeof detail.detail !== "string") {  // a typing mistake the server did not broadcast
      view.notices.push({ level: "error", message: "That value was not accepted.", at: new Date() });
    }
    steerSignature = "";  // put every fader back to the last confirmed value
    schedule("all");
    return false;
  }
  return true;
}

$("stop").addEventListener("click", () => post("/admin/api/stop"));
$("clear").addEventListener("click", () => post("/admin/api/clear"));
$("redo").addEventListener("click", () => selected && post("/admin/api/redo", { conversation: selected }));
$("alloff").addEventListener("click", () => post("/admin/api/all-off"));
$("prompt").addEventListener("input", () => { promptDirty = true; renderPrompt(); });
$("prompt-apply").addEventListener("click", async () => {
  if (await post("/admin/api/prompt", { text: $("prompt").value })) { promptDirty = false; promptSeen = null; }
});
$("prompt-reset").addEventListener("click", async () => {
  if (await post("/admin/api/prompt", { text: null })) { promptDirty = false; promptSeen = null; }
});

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tabs button")) {
      other.setAttribute("aria-selected", String(other === tab));
      $(`panel-${other.dataset.panel}`).hidden = other !== tab;
    }
  });
}


const types = ["snapshot", "state", "notice", "reset", "new", "user", "start", "token", "marker", "end", "redo"];
function connect() {
  const source = new EventSource("/admin/events");
  for (const type of types) {
    source.addEventListener(type, (message) => {
      const event = JSON.parse(message.data);
      if (type === "notice") event.at = new Date();
      apply(event);
      schedule(type);
    });
  }
  source.addEventListener("error", () => {
    $("status").textContent = "reconnecting…";
    $("status").className = "busy";
  });
}
connect();
mirrorOnly = false;
render();
