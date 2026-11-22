/* The Workbench room: chat with the model under steering, on a loom, with an Inspector per token; plus the Probes
 * and Artifacts pages. Backend: /api/workbench/* (rooms/workbench.py). Vanilla ES module, no build step.
 *
 * Routes: /workbench                    the chat (last conversation)
 *         /workbench/<loom>/<node>      a conversation, at a turn
 *         /workbench/probes[/<name>]    the concept probes, fitting and reading
 *         /workbench/artifacts          saved directions, probes, lenses and edits; export and import
 * Keys (only while this room is open): ⌘K / Ctrl+K search rooms and actions · ←/→ step tokens of the selected reply ·
 *         Esc close · Enter in the box sends (Shift+Enter: new line).
 * Colours follow inspect.css: orange = what steering does, blue = controls, grey = the random-direction baseline.
 * Model text always goes in with textContent (api.el), never as HTML.
 */

let A = null;              // the shell's api
let root = null;
let alive = false;
const W = {
  tab: "chat", state: null, looms: [], loom: null, node: null, sel: null, compare: null, view: "chat",
  steer: [], edits: [], sampling: { max_tokens: 96, temperature: 0, top_p: 1, seed: null }, heat: "",
  active: null, busy: false, live: null, replay: {}, artifacts: [], probeSel: null, palette: null, read: null,
};
const listeners = [];

/* ---------- helpers ---------- */

const el = (...args) => A.el(...args);
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(`wb:${A.model}:${key}`); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`wb:${A.model}:${key}`, JSON.stringify(value)); } catch { /* private mode */ } },
};
function on(target, type, fn, opts) { target.addEventListener(type, fn, opts); listeners.push([target, type, fn, opts]); }
function unitModel() { return (A.model || "").split("/").pop().toLowerCase().replace(/[^a-z0-9.]+/g, "-").replace(/^-+|-+$/g, ""); }
function unitHref(address) { return A.href("units", unitModel(), ...address.split("/")); }
function nodeOf(id) { return W.loom?.nodes?.[id] || null; }
function pathTo(id) { const out = []; let n = nodeOf(id); while (n) { out.unshift(n); n = nodeOf(n.parent); } return out; }
function leafOf(id) { let n = nodeOf(id); while (n && n.children.length) n = nodeOf(n.children[n.children.length - 1]); return n?.id || id; }
function siblings(n) { return n.parent ? nodeOf(n.parent).children : W.loom.roots; }
function probeNames() { return (W.state?.probes || []).map((p) => p.name); }
function activeProbes() { const all = probeNames(); return W.active ? W.active.filter((n) => all.includes(n)) : all; }
function steerLabel(entries) {
  if (!entries?.length) return "unsteered";
  return entries.map((e) => `${e.strength > 0 ? "+" : ""}${fmt(e.strength, 2).replace(/\.?0+$/, "")} ${(e.source?.name || e.label || "").replace(/^probe:/, "")}`).join("  ");
}
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function postRaw(path, body, type) {
  const r = await fetch(`api/${path}`, { method: "POST", headers: { "Content-Type": type }, body });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : `request failed (${r.status})`);
  return data;
}
async function del(path) {
  const r = await fetch(`api/${path}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete failed (${r.status})`);
}
const fail = (error) => A.show(error.message || String(error), true);

/* ---------- mount ---------- */

export async function mount(el_, api, route) {
  A = api; root = el_; alive = true;
  W.steer = store.get("steer", []); W.edits = store.get("edits", []); W.sampling = { ...W.sampling, ...store.get("sampling", {}) };
  W.active = store.get("active", null); W.heat = store.get("heat", "");
  on(document, "keydown", onKey, true);
  await refreshState();
  await update(route);
  pollFit();
}

export function unmount() {
  alive = false;
  for (const [t, type, fn, opts] of listeners.splice(0)) t.removeEventListener(type, fn, opts);
  W.palette?.remove(); W.palette = null;
}

export async function update(route) {
  const [first, second] = route.path;
  W.tab = first === "probes" ? "probes" : first === "artifacts" ? "artifacts" : "chat";
  if (W.tab === "probes") W.probeSel = second || W.probeSel;
  if (W.tab === "chat") {
    W.looms = (await A.get("workbench/looms")).looms;
    const id = first || store.get("loom", null) || W.looms[0]?.id;
    if (id && (!W.loom || W.loom.id !== id)) {
      try { W.loom = await A.get(`workbench/looms/${id}`); } catch { W.loom = null; }
    }
    if (W.loom) {
      store.set("loom", W.loom.id);
      W.node = second && nodeOf(second) ? second : (W.node && nodeOf(W.node) ? W.node : lastLeaf());
    }
  }
  if (W.tab === "artifacts") await loadArtifacts();
  render();
}

function lastLeaf() {
  if (!W.loom?.roots.length) return null;
  return leafOf(W.loom.roots[W.loom.roots.length - 1]);
}

async function refreshState() {
  try { W.state = await A.get("workbench/state"); } catch (error) { fail(error); }
}

function pollFit() {
  if (!alive) return;
  const fit = W.state?.fit;
  if (fit && (fit.state === "fitting" || fit.state === "queued")) {
    setTimeout(async () => { await refreshState(); if (alive) { renderFitBanner(); if (W.tab === "probes") render(); } pollFit(); }, 1500);
  }
}

/* ---------- layout ---------- */

function render() {
  if (!alive) return;
  const tabs = el("nav", { class: "wb-tabs", "aria-label": "Workbench pages" },
    ...[["chat", "Chat", "/workbench"], ["probes", "Probes", "/workbench/probes"], ["artifacts", "Artifacts", "/workbench/artifacts"]].map(([id, title, href]) =>
      el("a", { href, "aria-current": W.tab === id ? "page" : null, text: title })),
    el("button", { type: "button", class: "wb-search", onclick: openPalette, title: "Search rooms and actions" }, "Search", el("kbd", { text: "⌘K" })));
  const banner = el("div", { id: "wb-fit", class: "wb-fit" });
  let body;
  if (W.tab === "probes") body = renderProbes();
  else if (W.tab === "artifacts") body = renderArtifacts();
  else body = renderChat();
  root.replaceChildren(tabs, banner, body);
  renderFitBanner();
}

function renderFitBanner() {
  const box = root?.querySelector("#wb-fit");
  if (!box) return;
  const fit = W.state?.fit || {};
  if (fit.state === "fitting" || fit.state === "queued") {
    box.replaceChildren(el("span", { class: "dot" }), `Fitting the bundled probes on first launch${fit.concept ? `: ${fit.concept}` : ""} (${fit.done || 0}/${fit.total || "…"}). Chat works meanwhile; readings appear as probes finish.`);
    box.hidden = false;
  } else if (fit.state === "error") {
    box.replaceChildren(`Probe fitting failed: ${fit.error}`); box.classList.add("bad"); box.hidden = false;
  } else box.hidden = true;
}

/* ---------- chat ---------- */

function renderChat() {
  return el("div", { class: "wb-grid" }, renderThread(), renderPage(), renderInspector());
}

function renderThread() {
  const list = el("ul", { class: "wb-looms" }, W.looms.map((l) => el("li", {},
    el("a", { href: `/workbench/${l.id}`, "aria-current": W.loom?.id === l.id ? "true" : null },
      el("span", { class: "t", text: l.title || "Untitled" }), el("small", { text: `${l.turns} turns` })))));
  const path = W.loom && W.node ? pathTo(W.node) : [];
  const trail = el("ol", { class: "wb-trail" }, path.map((n) => {
    const sib = siblings(n);
    return el("li", { class: `role-${n.role}` },
      el("button", { type: "button", class: "linkish", onclick: () => go(n.id), text: `${n.role === "assistant" ? "model" : n.role}: ${n.content.slice(0, 38) || "…"}` }),
      sib.length > 1 ? el("small", { text: ` ${sib.indexOf(n.id) + 1}/${sib.length}` }) : null);
  }));
  return el("aside", { class: "panel wb-thread", "aria-label": "Thread" },
    el("h2", { text: "Thread" }),
    el("div", { class: "actions" },
      el("button", { type: "button", class: "primary", onclick: newLoom, text: "New conversation" }),
      W.loom ? el("button", { type: "button", onclick: exportLoom, title: "Download this conversation tree as JSON", text: "Save" }) : null,
      el("label", { class: "filebtn", title: "Open a saved conversation tree" }, "Open", el("input", { type: "file", accept: ".json,application/json", onchange: importLoom }))),
    W.loom ? el("div", {}, el("div", { class: "label", text: "Path" }), trail) : null,
    el("div", { class: "label", text: "Conversations" }), W.looms.length ? list : el("p", { class: "empty", text: "No conversations yet." }));
}

function go(nodeId) {
  W.node = nodeId; W.sel = null; W.compare = null;
  A.navigate(`/workbench/${W.loom.id}/${nodeId}`, { replace: true });
}

function renderPage() {
  const page = el("section", { class: "panel wb-page", "aria-label": "Conversation" });
  const viewTabs = el("div", { class: "wb-view", role: "group", "aria-label": "View" },
    ...[["chat", "Conversation"], ["loom", "Loom"]].map(([id, title]) =>
      el("button", { type: "button", "aria-pressed": W.view === id ? "true" : "false", onclick: () => { W.view = id; W.compare = null; render(); }, text: title })),
    el("label", { class: "wb-heat" }, "Colour tokens by ",
      el("select", { onchange: (e) => { W.heat = e.target.value; store.set("heat", W.heat); render(); } },
        el("option", { value: "", text: "nothing" }), probeNames().map((n) => el("option", { value: n, selected: W.heat === n, text: n })))));
  page.append(viewTabs);
  if (W.compare) page.append(renderCompare());
  else if (W.view === "loom") page.append(renderLoomTree());
  else page.append(renderTurns());
  page.append(renderControls(), renderCompose());
  return page;
}

function renderTurns() {
  if (!W.loom || !W.node) {
    return el("div", { class: "wb-empty" }, el("p", { text: "Talk to the model. Pick steering below (probe directions, saved directions, SAE features) and edits; every reply keeps what made it, and every token can be inspected." }),
      el("p", { class: "hint", text: "Reroll a reply, fork a turn, or compare a steered reply with its unsteered sibling. ⌘K finds any action." }));
  }
  const box = el("div", { class: "wb-turns" });
  for (const n of pathTo(W.node)) box.append(renderTurn(n));
  if (W.live) box.append(W.live.el);
  return box;
}

function renderTurn(n) {
  const sib = siblings(n);
  const i = sib.indexOf(n.id);
  const nav = sib.length > 1 ? el("span", { class: "wb-sib" },
    el("button", { type: "button", "aria-label": "previous version", disabled: i === 0, onclick: () => go(leafOf(sib[i - 1])), text: "‹" }),
    el("small", { text: `${i + 1}/${sib.length}` }),
    el("button", { type: "button", "aria-label": "next version", disabled: i === sib.length - 1, onclick: () => go(leafOf(sib[i + 1])), text: "›" })) : null;
  const head = el("div", { class: "wb-turn-head" }, el("span", { class: "who", text: n.role === "assistant" ? "model" : n.role }), nav);
  const card = el("article", { class: `wb-turn role-${n.role}`, "data-node": n.id }, head);
  if (n.role === "assistant") {
    const meta = n.meta || {};
    const steered = meta.steer?.length || meta.edits?.length;
    card.classList.toggle("steered", !!steered);
    head.append(el("span", { class: `badge ${steered ? "ready" : ""}`, title: "what produced this reply", text: steerLabel(meta.steer) + (meta.edits?.length ? ` · ${meta.edits.length} edit${meta.edits.length > 1 ? "s" : ""}` : "") }));
    card.append(renderTokens(n));
    if (meta.finish_reason === "cancelled") card.append(el("p", { class: "hint", text: "stopped early" }));
    card.append(el("div", { class: "actions" },
      el("button", { type: "button", onclick: () => reroll(n), title: "a new reply here with the current steering", text: "Reroll" }),
      el("button", { type: "button", onclick: () => reroll(n, { unsteered: true }), title: "the same turn with no steering and no edits", text: "Unsteered sibling" }),
      sib.length > 1 ? el("button", { type: "button", onclick: () => pickCompare(n), text: "Compare" }) : null,
      el("button", { type: "button", onclick: () => doReplay(n), title: "read this reply's tokens with the unsteered model", text: "Replay unsteered" }),
      el("button", { type: "button", class: "quiet", onclick: () => removeNode(n), text: "Delete" })));
  } else {
    card.append(el("p", { class: "wb-text", text: n.content }));
    if (n.role === "user") card.append(el("div", { class: "actions" },
      el("button", { type: "button", onclick: () => forkTurn(n), title: "say this turn differently, keeping both", text: "Fork" }),
      !n.children.length ? el("button", { type: "button", class: "primary", onclick: () => reply(n.id), text: "Reply" }) : null));
  }
  return card;
}

function heatStyle(z, band) {
  if (z === undefined || z === null) return null;
  const over = Math.abs(z) - (band || 0);
  if (over <= 0) return "background: rgba(163,154,144,.10)";
  const a = Math.min(0.55, 0.12 + over / 8);
  return z > 0 ? `background: rgba(224,122,63,${a})` : `background: rgba(111,159,240,${a})`;
}

function renderTokens(n) {
  const tokens = n.meta?.tokens || [];
  const wrap = el("p", { class: "wb-reply", role: "listbox", "aria-label": "reply tokens" });
  if (!tokens.length) { wrap.textContent = n.content || "(empty)"; return wrap; }
  tokens.forEach((t, i) => {
    const selected = W.sel && W.sel.node === n.id && W.sel.i === i;
    wrap.append(el("span", {
      class: "wb-tok", role: "option", tabindex: selected ? "0" : "-1", "aria-selected": selected ? "true" : "false",
      style: W.heat ? heatStyle(t.probes?.[W.heat], t.bands?.[W.heat]) : null,
      title: `${JSON.stringify(t.token)}  logprob ${fmt(t.logprob)}`,
      onclick: () => { W.sel = { node: n.id, i }; render(); },
      text: t.text || "",
    }));
  });
  return wrap;
}

function renderLoomTree() {
  if (!W.loom?.roots.length) return el("p", { class: "empty", text: "The loom is empty: send a message." });
  const onPath = new Set(W.node ? pathTo(W.node).map((n) => n.id) : []);
  const draw = (id) => {
    const n = nodeOf(id);
    const label = n.role === "assistant" ? `${steerLabel(n.meta?.steer)} · ${n.content.slice(0, 70)}` : n.content.slice(0, 80);
    return el("li", { class: `role-${n.role} ${onPath.has(id) ? "on" : ""}` },
      el("button", { type: "button", class: "linkish", onclick: () => { W.view = "chat"; go(leafOf(id)); } },
        el("span", { class: "who", text: n.role === "assistant" ? "model" : n.role }), el("span", { class: "t", text: label || "…" })),
      n.children.length ? el("ul", {}, n.children.map(draw)) : null);
  };
  return el("div", { class: "wb-tree" }, el("p", { class: "hint", text: "Every turn is a node. Click one to continue from it; siblings are rerolls and forks." }), el("ul", {}, W.loom.roots.map(draw)));
}

function pickCompare(n) {
  const sib = siblings(n).filter((id) => id !== n.id);
  const plain = sib.find((id) => !(nodeOf(id).meta?.steer || []).length) || sib[sib.length - 1];
  W.compare = { a: n.id, b: leafOf(plain) };
  render();
}

function renderCompare() {
  const { a, b } = W.compare;
  const shared = new Set(pathTo(b).map((x) => x.id));
  const split = pathTo(a).filter((x) => shared.has(x.id)).length;
  const col = (id) => el("div", { class: "wb-col" }, pathTo(id).slice(split).map(renderTurn));
  const pick = (which) => el("select", { onchange: (e) => { W.compare[which] = e.target.value; render(); } },
    Object.values(W.loom.nodes).filter((n) => n.role === "assistant").map((n) => el("option", { value: n.id, selected: W.compare[which] === n.id, text: `${n.id} · ${steerLabel(n.meta?.steer)}` })));
  return el("div", { class: "wb-compare" },
    el("div", { class: "wb-compare-head" }, el("span", { text: "Compare" }), pick("a"), el("span", { text: "with" }), pick("b"),
      el("button", { type: "button", onclick: () => { W.compare = null; render(); }, text: "Close" })),
    el("div", { class: "wb-cols" }, col(a), col(b)),
    renderCompareProbes(a, b));
}

function meanOf(id, key, name) {
  const z = (nodeOf(id)?.meta?.tokens || []).map((x) => x[key]?.[name]).filter((v) => v !== undefined);
  return z.length ? z.reduce((s, v) => s + v, 0) / z.length : null;
}

function renderCompareProbes(a, b) {
  const rows = activeProbes().map((name) => {
    const ma = meanOf(a, "probes", name); const mb = meanOf(b, "probes", name); const band = meanOf(a, "bands", name);
    const d = ma !== null && mb !== null ? ma - mb : null;
    return el("tr", { class: d !== null && Math.abs(d) > (band || 0) ? "changed" : "" },
      el("td", { class: "id", text: name }), el("td", { class: "num", text: fmt(ma) }), el("td", { class: "num", text: fmt(mb) }),
      el("td", { class: "num", text: fmt(d) }), el("td", { class: "num rnd", text: `±${fmt(band)}` }));
  });
  return el("details", { class: "wb-cmp-probes", open: true }, el("summary", { text: "Mean probe readings over each reply (z), with the random-direction band" }),
    el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "probe" }), el("th", { class: "num", text: "left" }), el("th", { class: "num", text: "right" }), el("th", { class: "num", text: "Δ" }), el("th", { class: "num", text: "random" }))), el("tbody", {}, rows)));
}

/* ---------- steering controls ---------- */

function renderControls() {
  const steerables = W.state?.steerables || [];
  const chips = el("div", { class: "wb-chips" }, W.steer.map((s, i) => el("span", { class: "wb-chip steer" },
    el("span", { class: "n", text: s.name.replace(/^probe:/, "") }),
    el("input", { type: "number", step: "0.5", value: s.strength, "aria-label": `strength of ${s.name}`, oninput: (e) => { s.strength = Number(e.target.value); saveControls(); } }),
    el("input", { type: "text", class: "layers", placeholder: `L${steerables.find((x) => x.name === s.name)?.layer ?? ""}`, value: s.layers || "", "aria-label": "layers (e.g. 10,12)", oninput: (e) => { s.layers = e.target.value; saveControls(); } }),
    el("button", { type: "button", class: "x", "aria-label": `remove ${s.name}`, onclick: () => { W.steer.splice(i, 1); saveControls(); render(); }, text: "×" }))));
  const add = el("select", { "aria-label": "add steering", onchange: (e) => { if (e.target.value) addSteer(e.target.value); } },
    el("option", { value: "", text: "+ steer with…" }),
    el("optgroup", { label: "probe directions (strength 1 = one class gap)" }, steerables.filter((s) => s.kind === "probe").map((s) => el("option", { value: s.name, text: s.name.slice(6) }))),
    el("optgroup", { label: "saved directions" }, steerables.filter((s) => s.kind === "direction").map((s) => el("option", { value: s.name, text: s.name }))),
    el("option", { value: "__sae", text: "SAE feature…" }));
  const edits = el("div", { class: "wb-chips" }, W.edits.map((name, i) => el("span", { class: "wb-chip edit" }, el("span", { class: "n", text: `edit ${name}` }),
    el("button", { type: "button", class: "x", "aria-label": `remove edit ${name}`, onclick: () => { W.edits.splice(i, 1); saveControls(); render(); }, text: "×" }))));
  const editPick = el("select", { "aria-label": "add an edit", onfocus: loadEditsInto, onchange: (e) => { if (e.target.value && !W.edits.includes(e.target.value)) { W.edits.push(e.target.value); saveControls(); render(); } } },
    el("option", { value: "", text: "+ edit…" }));
  const s = W.sampling;
  const num = (key, attrs, label) => el("label", {}, label, el("input", { type: "number", ...attrs, value: s[key] ?? "", oninput: (e) => { s[key] = e.target.value === "" ? null : Number(e.target.value); saveControls(); } }));
  return el("details", { class: "wb-controls", open: true },
    el("summary", {}, el("span", { text: "Steering" }), el("span", { class: "hint", text: ` ${W.steer.length ? steerLabel(W.steer.map((x) => ({ strength: x.strength, source: { name: x.name } }))) : "none"}${W.edits.length ? ` · ${W.edits.length} edit(s)` : ""}` })),
    el("div", { class: "wb-row" }, chips, add),
    el("div", { class: "wb-row" }, edits, editPick),
    el("div", { class: "wb-row wb-sampling" },
      num("temperature", { min: 0, max: 2, step: 0.1 }, "temperature "), num("max_tokens", { min: 1, max: 512, step: 16 }, "max tokens "),
      num("seed", { step: 1, placeholder: "random" }, "seed "),
      el("span", { class: "hint", text: "temperature 0 is greedy; steering acts from the first generated position on" })));
}

async function loadEditsInto(e) {
  const select = e.target;
  if (select.dataset.loaded) return;
  try {
    const { artifacts } = await A.get("workbench/artifacts?type=edit");
    select.append(...artifacts.map((a) => el("option", { value: a.name, text: `${a.name}${a.mode ? ` (${a.mode})` : ""}` })));
    if (!artifacts.length) select.append(el("option", { value: "", disabled: true, text: "no edits saved yet (Edit room)" }));
    select.dataset.loaded = "1";
  } catch (error) { fail(error); }
}

function addSteer(name) {
  if (name === "__sae") {
    const text = prompt("SAE feature as release,layer,feature (e.g. Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50,14,1234)");
    if (!text) { render(); return; }
    const [release, layer, feature] = text.split(",").map((x) => x.trim());
    W.steer.push({ name: `sae:${release}:${layer}:${feature}`, sae: { release, layer: Number(layer), feature: Number(feature) }, strength: 4 });
  } else if (!W.steer.some((s) => s.name === name)) W.steer.push({ name, strength: name.startsWith("probe:") ? 1 : 4 });
  saveControls(); render();
}

function saveControls() { store.set("steer", W.steer); store.set("edits", W.edits); store.set("sampling", W.sampling); }

function steerPayload(unsteered = false) {
  if (unsteered) return [];
  return W.steer.filter((s) => s.strength).map((s) => {
    const layers = String(s.layers || "").split(",").map((x) => x.trim()).filter(Boolean).map(Number).filter((x) => Number.isInteger(x));
    const base = s.sae ? { sae: s.sae } : { direction: s.name };
    return { ...base, strength: s.strength, ...(layers.length ? { layers } : {}) };
  });
}

/* ---------- compose and generate ---------- */

function renderCompose() {
  const box = el("textarea", { id: "wb-input", rows: "2", placeholder: W.loom ? "Message the model (Enter to send, Shift+Enter for a new line)" : "Start a conversation", "aria-label": "Message" });
  box.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
  return el("form", { class: "wb-compose", onsubmit: (e) => { e.preventDefault(); send(); } }, box,
    W.busy ? el("button", { type: "button", onclick: stop, text: "Stop" }) : el("button", { type: "submit", class: "primary", text: "Send" }));
}

async function ensureLoom() {
  if (W.loom) return W.loom;
  W.loom = await A.post("workbench/looms", { title: "" });
  W.looms = (await A.get("workbench/looms")).looms;
  return W.loom;
}

async function send() {
  const input = root.querySelector("#wb-input");
  const text = input?.value.trim();
  if (!text || W.busy) return;
  try {
    await ensureLoom();
    const node = W.node ? nodeOf(W.node) : null;
    const parent = node ? (node.role === "user" && !node.children.length ? node.parent : node.id) : null;
    const out = await A.post(`workbench/looms/${W.loom.id}/turn`, { parent, role: "user", content: text });
    W.loom = out.loom; W.node = out.result;
    input.value = "";
    await reply(out.result);
  } catch (error) { fail(error); }
}

async function reroll(n, { unsteered = false } = {}) { await reply(n.parent, { unsteered }); }

async function reply(parentId, { unsteered = false } = {}) {
  if (W.busy) return;
  W.busy = true; W.sel = null; W.compare = null; W.node = parentId; W.view = "chat";
  const liveText = el("p", { class: "wb-reply live" });
  W.live = { el: el("article", { class: `wb-turn role-assistant ${unsteered || !W.steer.length ? "" : "steered"}` }, el("div", { class: "wb-turn-head" }, el("span", { class: "who", text: "model" }), el("span", { class: "badge", text: unsteered ? "unsteered" : "writing…" })), liveText) };
  render();
  const body = {
    loom: W.loom.id, parent: parentId, steer: steerPayload(unsteered), edits: unsteered ? [] : W.edits,
    probes: activeProbes(), sampling: { ...W.sampling, seed: W.sampling.seed ?? null }, label: unsteered ? "unsteered" : "",
  };
  try {
    const r = await fetch("api/workbench/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(typeof d.detail === "string" ? d.detail : `request failed (${r.status})`); }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = ""; let done = null;
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === "token") liveText.append(el("span", { class: "wb-tok", text: ev.text }));
        else if (ev.type === "error") throw new Error(ev.detail);
        else if (ev.type === "done") done = ev;
      }
    }
    if (done) {
      W.loom = await A.get(`workbench/looms/${W.loom.id}`);
      W.node = done.node.id;
      W.sel = done.node.meta.tokens.length ? { node: done.node.id, i: 0 } : null;
      W.looms = (await A.get("workbench/looms")).looms;
      history.replaceState(null, "", `/workbench/${W.loom.id}/${done.node.id}`);
    }
  } catch (error) { fail(error); }
  finally { W.busy = false; W.live = null; render(); }
}

async function stop() { try { await A.post("workbench/cancel", {}); } catch (error) { fail(error); } }

async function newLoom() {
  try {
    W.loom = await A.post("workbench/looms", { title: "" }); W.node = null; W.sel = null;
    A.navigate(`/workbench/${W.loom.id}`);
  } catch (error) { fail(error); }
}

async function forkTurn(n) {
  const text = prompt("Say this turn differently (both versions are kept):", n.content);
  if (!text || text === n.content) return;
  try {
    const out = await A.post(`workbench/looms/${W.loom.id}/fork`, { node: n.id, content: text });
    W.loom = out.loom; W.node = out.result;
    if (n.role === "user") await reply(out.result); else go(out.result);
  } catch (error) { fail(error); }
}

async function removeNode(n) {
  if (!confirm("Delete this reply and everything after it?")) return;
  try {
    const out = await A.post(`workbench/looms/${W.loom.id}/remove`, { node: n.id });
    W.loom = out.loom; go(n.parent);
  } catch (error) { fail(error); }
}

async function doReplay(n) {
  try {
    W.replay[n.id] = await A.post("workbench/replay", { loom: W.loom.id, node: n.id, steer: [] });
    W.sel = W.sel?.node === n.id ? W.sel : { node: n.id, i: 0 };
    render();
    A.show("Readings replayed without steering: see the Inspector.");
  } catch (error) { fail(error); }
}

function exportLoom() {
  download(new Blob([JSON.stringify(W.loom, null, 1)], { type: "application/json" }), `${W.loom.id}.json`);
}

async function importLoom(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const loom = await A.post("workbench/looms/import", JSON.parse(await file.text()));
    A.navigate(`/workbench/${loom.id}`);
  } catch (error) { fail(error); }
}

/* ---------- the Inspector ---------- */

function renderInspector() {
  const panel = el("aside", { class: "panel wb-inspector", "aria-label": "Inspector" }, el("h2", { text: "Inspector" }));
  const node = W.sel ? nodeOf(W.sel.node) : null;
  const tok = node?.meta?.tokens?.[W.sel.i];
  if (!tok) { panel.append(renderInstruments()); return panel; }
  const tokens = node.meta.tokens;
  panel.append(
    el("div", { class: "wb-tokhead" },
      el("button", { type: "button", "aria-label": "previous token", disabled: W.sel.i === 0, onclick: () => step(-1), text: "←" }),
      el("code", { class: "big", text: JSON.stringify(tok.token) }),
      el("button", { type: "button", "aria-label": "next token", disabled: W.sel.i === tokens.length - 1, onclick: () => step(1), text: "→" }),
      el("small", { text: `token ${W.sel.i + 1} of ${tokens.length} · logprob ${fmt(tok.logprob)}` })),
    el("h3", { text: "Alternatives" }),
    el("table", { class: "wb-alts" }, el("tbody", {}, tok.alternatives.map((a) => el("tr", { class: a.id === tok.id ? "chosen" : "" },
      el("td", { class: "id", text: JSON.stringify(a.text) }), el("td", { class: "num", text: `${fmt(Math.exp(a.logprob) * 100, 1)}%` }))))),
    renderReadings(node, tok),
    renderUnits(tok),
    renderSteerStrength(node, tok),
    el("p", { class: "actions" }, el("button", { type: "button", class: "quiet", onclick: () => { W.sel = null; render(); }, text: "Back to instruments" })));
  return panel;
}

function step(d) {
  const node = nodeOf(W.sel.node); const n = node.meta.tokens.length;
  W.sel = { node: W.sel.node, i: Math.max(0, Math.min(n - 1, W.sel.i + d)) };
  render();
  root.querySelector(".wb-tok[aria-selected=true]")?.scrollIntoView({ block: "nearest" });
}

function bar(z, band, scale = 4) {
  const pct = (v) => `${Math.max(0, Math.min(50, (Math.abs(v) / scale) * 50))}%`;
  const b = el("div", { class: "wb-bar" });
  b.append(el("i", { class: "band", style: `left: calc(50% - ${pct(band || 0)}); width: calc(2 * ${pct(band || 0)})` }));
  if (z !== undefined && z !== null) b.append(el("i", { class: `val ${Math.abs(z) > (band || 0) ? "" : "weak"}`, style: z >= 0 ? `left: 50%; width: ${pct(z)}` : `left: calc(50% - ${pct(z)}); width: ${pct(z)}` }));
  return b;
}

function renderReadings(node, tok) {
  const replay = W.replay[node.id];
  const names = activeProbes().filter((n) => tok.probes && n in tok.probes);
  const rows = names.map((name) => {
    const z = tok.probes[name]; const band = tok.bands?.[name];
    const unsteered = replay?.probes?.[name]?.[W.sel.i];
    return el("div", { class: "wb-reading" },
      el("a", { href: `/workbench/probes/${encodeURIComponent(name)}`, class: "n", text: name }),
      bar(z, band), el("span", { class: "num", text: fmt(z) }),
      replay ? el("span", { class: "num rnd", title: "the same token read by the unsteered model", text: fmt(unsteered) }) : null);
  });
  return el("div", {}, el("h3", {}, "Probe readings ", el("span", { class: "hint", text: "z at this token; grey = random-direction band (90th pct)" })),
    replay ? el("p", { class: "hint", text: "right column: the same token replayed without steering" }) : null,
    rows.length ? el("div", { class: "wb-readings" }, rows) : el("p", { class: "empty", text: W.state?.probes?.length ? "No active probes: pick some under Instruments." : "Probes are still fitting." }));
}

function renderUnits(tok) {
  if (!tok.units?.length) return null;
  return el("div", {}, el("h3", {}, "Top units ", el("span", { class: "hint", text: "direct write to this token's logit" })),
    el("table", {}, el("tbody", {}, tok.units.map((u) => el("tr", {},
      el("td", { class: "id" }, el("a", { href: u.href || unitHref(u.address), title: "open in the Units room", text: u.address })),
      el("td", { class: "hint", text: u.kind }),
      el("td", { class: `num ${u.contribution >= 0 ? "pos" : "neg"}`, text: `${u.contribution >= 0 ? "+" : ""}${fmt(u.contribution, 3)}` }))))));
}

function renderSteerStrength(node, tok) {
  const layers = Object.entries(tok.steer || {});
  return el("div", {}, el("h3", {}, "Steering ", el("span", { class: "hint", text: "added norm / median residual norm" })),
    el("p", { class: "sub", text: steerLabel(node.meta?.steer) + (node.meta?.edits?.length ? ` · ${node.meta.edits.join("; ")}` : "") }),
    layers.length ? el("table", {}, el("tbody", {}, layers.map(([layer, s]) => el("tr", {},
      el("td", { class: "id", text: `L${layer}` }), el("td", { class: "num", text: `${fmt(s.added)} / ${fmt(s.residual)}` }),
      el("td", { class: "num pos", text: `×${fmt(s.ratio, 3)}` })))))
      : el("p", { class: "hint", text: W.sel.i === 0 && node.meta?.steer?.length ? "The first token is chosen at the last prompt position, which steering never touches." : "nothing added at this token" }));
}

function renderInstruments() {
  const probes = W.state?.probes || [];
  const active = new Set(activeProbes());
  const node = W.node ? nodeOf(W.node) : null;
  const tokens = node?.role === "assistant" ? node.meta?.tokens || [] : [];
  const spark = (name) => {
    if (!tokens.length) return null;
    const w = 90, h = 18; const zs = tokens.map((t) => t.probes?.[name] ?? 0); const m = Math.max(3, ...zs.map(Math.abs));
    const pts = zs.map((z, i) => `${(i / Math.max(1, zs.length - 1)) * w},${h / 2 - (z / m) * (h / 2)}`).join(" ");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("class", "spark"); svg.setAttribute("aria-hidden", "true");
    const line = document.createElementNS(svg.namespaceURI, "polyline"); line.setAttribute("points", pts); svg.append(line);
    return svg;
  };
  return el("div", {},
    el("p", { class: "hint", text: "Click a token of a reply to see its alternatives, probe readings, top units and steering." }),
    el("h3", { text: "Instruments" }),
    probes.length ? el("ul", { class: "wb-instr" }, probes.map((p) => el("li", {},
      el("label", {}, el("input", { type: "checkbox", checked: active.has(p.name), onchange: (e) => toggleProbe(p.name, e.target.checked) }), el("span", { text: p.name })),
      el("small", { class: "hint", title: "separation on its examples vs random directions", text: `d′ ${fmt(p.d_prime, 1)} · rnd ${fmt(p.random_d_prime, 1)}` }), spark(p.name)))) : el("p", { class: "empty", text: "Probes are fitting…" }),
    el("p", { class: "hint" }, "Fit your own on the ", el("a", { href: "/workbench/probes", text: "Probes" }), " page."));
}

function toggleProbe(name, onoff) {
  const set = new Set(activeProbes());
  if (onoff) set.add(name); else set.delete(name);
  W.active = probeNames().filter((n) => set.has(n));
  store.set("active", W.active);
  render();
}

/* ---------- Probes page ---------- */

function renderProbes() {
  const probes = W.state?.probes || [];
  const sel = probes.find((p) => p.name === W.probeSel) || probes[0];
  const table = el("table", { class: "wb-probes" },
    el("thead", {}, el("tr", {}, el("th", { text: "probe" }), el("th", { text: "poles" }), el("th", { class: "num", text: "layer" }), el("th", { class: "num", text: "d′" }), el("th", { class: "num", text: "random d′" }), el("th", { class: "num", text: "acc" }))),
    el("tbody", {}, probes.map((p) => el("tr", { class: sel && p.name === sel.name ? "changed" : "", onclick: () => A.navigate(`/workbench/probes/${encodeURIComponent(p.name)}`, { replace: true }) },
      el("td", { class: "id", text: p.name }), el("td", { class: "words", text: `${p.positive_label} ↔ ${p.negative_label}` }),
      el("td", { class: "num", text: p.layer }), el("td", { class: "num pos", text: fmt(p.d_prime, 1) }), el("td", { class: "num rnd", text: fmt(p.random_d_prime, 1) }), el("td", { class: "num", text: `${fmt(p.accuracy * 100, 0)}%` })))));
  const list = el("section", { class: "panel" }, el("h2", { text: "Concept probes" }),
    el("p", { class: "hint", text: "Each probe is the difference of two poles' mean activations at one layer (a ruler across the residual stream). d′ is how far apart the poles' examples sit along it, in their own spread; random d′ is the 90th percentile of the same number for 16 random directions. A probe is only worth reading when d′ clears its random control, and d′ here is measured on the fitting examples, so it is optimistic." }),
    probes.length ? table : el("p", { class: "empty", text: "Fitting…" }),
    el("div", { class: "actions" }, el("button", { type: "button", onclick: refit, text: "Fit missing bundled probes" })));
  return el("div", { class: "wb-probes-grid" }, list, sel ? renderProbeDetail(sel) : el("div"), renderFitForm(), renderRead(probes));
}

function renderProbeDetail(p) {
  const prof = Object.entries(p.d_prime_by_layer || {}).map(([l, v]) => [Number(l), v]).sort((a, b) => a[0] - b[0]);
  const w = 360, h = 110, pad = 22;
  const max = Math.max(1, ...prof.map((x) => Math.abs(x[1])), p.random_d_prime || 0);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h + pad}`); svg.setAttribute("class", "chart"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `d prime by layer for ${p.name}`);
  const bw = (w - pad) / Math.max(1, prof.length);
  const mk = (tag, attrs) => { const n = document.createElementNS(svg.namespaceURI, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); svg.append(n); return n; };
  const yr = h - (p.random_d_prime / max) * (h - 10);
  mk("rect", { x: pad, y: yr, width: w - pad, height: h - yr, class: "band" });
  prof.forEach(([layer, v], i) => {
    const bh = (Math.abs(v) / max) * (h - 10);
    mk("rect", { x: pad + i * bw + 1, y: h - bh, width: Math.max(1, bw - 2), height: bh, class: layer === p.layer ? "dot" : "rnd-bar" });
  });
  mk("line", { x1: pad, x2: w, y1: h, y2: h, class: "axis" });
  mk("text", { x: pad, y: h + 15 }).textContent = "layer 0";
  mk("text", { x: w - 50, y: h + 15 }).textContent = `layer ${Math.max(0, prof.length - 1)}`;
  return el("section", { class: "panel" }, el("h2", { text: p.name }),
    el("p", { class: "sub", text: `${p.positive_label} (+) ↔ ${p.negative_label} (−) · ${p.category} · ${p.n_examples[0]} + ${p.n_examples[1]} examples · read at layer ${p.layer}` }),
    el("h3", {}, "Separation by layer ", el("span", { class: "hint", text: "d′ if fitted at each layer; orange = the layer it reads at; grey band = random control" })),
    svg,
    el("div", { class: "actions" },
      el("button", { type: "button", class: "primary", onclick: () => { addSteer(`probe:${p.name}`); A.navigate("/workbench"); }, text: "Steer with it" }),
      el("button", { type: "button", onclick: () => saveProbeDirection(p), text: "Save as direction artifact" }),
      p.source === "user" ? el("button", { type: "button", class: "quiet", onclick: () => deleteProbe(p), text: "Delete" }) : null));
}

function renderFitForm() {
  const layerOpts = Array.from({ length: W.state?.n_layers || 0 }, (_, i) => el("option", { value: i, selected: i === W.state?.probe_layer, text: `layer ${i}` }));
  const form = el("form", { class: "wb-fitform", onsubmit: fitNew },
    el("label", {}, "Name ", el("input", { name: "name", required: true, maxlength: "60", placeholder: "e.g. pirate.plain" })),
    el("div", { class: "wb-two" },
      el("label", {}, el("span", {}, "Positive pole ", el("input", { name: "positive_label", value: "positive", maxlength: "40" })), el("textarea", { name: "positive", rows: "6", required: true, placeholder: "one example reply per line (at least 2)" })),
      el("label", {}, el("span", {}, "Negative pole ", el("input", { name: "negative_label", value: "negative", maxlength: "40" })), el("textarea", { name: "negative", rows: "6", required: true, placeholder: "one example reply per line (at least 2)" }))),
    el("div", { class: "wb-row" }, el("label", {}, "read as replies to ", el("input", { name: "prompt", value: "Say something.", maxlength: "200" })), el("select", { name: "layer", "aria-label": "layer" }, layerOpts),
      el("button", { type: "submit", class: "primary", text: "Fit probe" })));
  return el("section", { class: "panel" }, el("h2", { text: "Fit a new probe" }),
    el("p", { class: "hint", text: "Write a few replies for each pole. Each is read as the model's reply to the prompt, its reply tokens averaged, and the probe is the difference of the two means. Seconds on the CPU." }), form);
}

async function fitNew(e) {
  e.preventDefault();
  const f = new FormData(e.target);
  const lines = (k) => String(f.get(k) || "").split("\n").map((x) => x.trim()).filter(Boolean);
  try {
    const p = await A.post("workbench/probes/fit", {
      name: f.get("name"), positive: lines("positive"), negative: lines("negative"), prompt: f.get("prompt"),
      positive_label: f.get("positive_label"), negative_label: f.get("negative_label"), layer: Number(f.get("layer")),
    });
    await refreshState();
    A.show(`Fitted ${p.name}: d′ ${fmt(p.d_prime, 2)} vs random ${fmt(p.random_d_prime, 2)}`);
    A.navigate(`/workbench/probes/${encodeURIComponent(p.name)}`, { replace: true });
  } catch (error) { fail(error); }
}

function renderRead(probes) {
  const out = el("div", { class: "wb-readout" });
  const pick = el("select", { name: "probe", "aria-label": "probe to colour by", onchange: () => { if (W.read) W.read.probe = pick.value; render(); } },
    probes.map((p) => el("option", { value: p.name, selected: p.name === (W.read?.probe || W.probeSel), text: p.name })));
  const form = el("form", { class: "wb-row", onsubmit: async (e) => {
    e.preventDefault();
    const text = e.target.text.value.trim(); if (!text) return;
    try { W.read = { probe: pick.value, text, result: await A.post("workbench/probes/read", { text }) }; render(); } catch (error) { fail(error); }
  } }, el("textarea", { name: "text", rows: "2", "aria-label": "text to read", placeholder: "Paste a reply to read with every probe", text: W.read?.text || "" }), pick, el("button", { type: "submit", text: "Read" }));
  if (W.read?.result) {
    const r = W.read.result; const name = pick.value || W.read.probe;
    const z = r.probes[name] || []; const b = r.bands[name] || [];
    const avg = (xs) => (xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : null);
    out.append(el("p", { class: "wb-reply" }, r.tokens.map((t, i) => el("span", { class: "wb-tok", style: heatStyle(z[i], b[i]), title: `z ${fmt(z[i])} · random ±${fmt(b[i])}`, text: t }))),
      el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "probe" }), el("th", { class: "num", text: "mean z" }), el("th", { class: "num", text: "random band" }))),
        el("tbody", {}, Object.keys(r.probes).map((n) => { const m = avg(r.probes[n]); const mb = avg(r.bands[n]);
          return el("tr", { class: Math.abs(m) > mb ? "changed" : "" }, el("td", { class: "id", text: n }), el("td", { class: "num", text: fmt(m) }), el("td", { class: "num rnd", text: `±${fmt(mb)}` })); }))));
  }
  return el("section", { class: "panel" }, el("h2", { text: "Read text" }),
    el("p", { class: "hint", text: "Every token's reading along a probe, coloured only where it leaves the random band (orange: toward the positive pole, blue: toward the negative)." }), form, out);
}

async function refit() { try { const r = await A.post("workbench/probes/refit", {}); await refreshState(); A.show(r.fitted.length ? `Fitted ${r.fitted.join(", ")}` : "Every bundled probe is already fitted."); render(); } catch (error) { fail(error); } }
async function saveProbeDirection(p) { try { const a = await A.post("workbench/artifacts/from-probe", { probe: p.name }); await refreshState(); A.show(`Saved direction ${a.name}`); } catch (error) { fail(error); } }
async function deleteProbe(p) { if (!confirm(`Delete probe ${p.name}?`)) return; try { await del(`workbench/probes/${encodeURIComponent(p.name)}`); await refreshState(); W.probeSel = null; render(); } catch (error) { fail(error); } }

/* ---------- Artifacts page ---------- */

async function loadArtifacts() { try { W.artifacts = (await A.get("workbench/artifacts")).artifacts; } catch (error) { fail(error); } }

function renderArtifacts() {
  const kinds = ["direction", "probe", "lens", "edit"];
  const filter = W.artFilter || "";
  const rows = W.artifacts.filter((a) => !filter || a.type === filter);
  const picked = W.artPicked || (W.artPicked = new Set());
  const when = (t) => (t ? new Date(t * 1000).toISOString().slice(0, 16).replace("T", " ") : "");
  const table = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}), el("th", { text: "type" }), el("th", { text: "name" }), el("th", { text: "model" }), el("th", { class: "num", text: "layer" }), el("th", { text: "saved" }), el("th"))),
    el("tbody", {}, rows.map((a) => { const key = `${a.type}/${a.name}`;
      return el("tr", {},
        el("td", {}, el("input", { type: "checkbox", "aria-label": `select ${key}`, checked: picked.has(key), onchange: (e) => { if (e.target.checked) picked.add(key); else picked.delete(key); } })),
        el("td", {}, el("span", { class: `badge ${a.type === "edit" || a.type === "direction" ? "control" : ""}`, text: a.type })),
        el("td", { class: "id", text: a.concept && a.type === "probe" ? a.concept : a.name }), el("td", { class: "words", text: a.model || "" }),
        el("td", { class: "num", text: a.layer ?? "" }), el("td", { class: "hint", text: when(a.created) }),
        el("td", { class: "num" },
          a.type === "direction" ? el("button", { type: "button", onclick: () => { addSteer(a.name); A.navigate("/workbench"); }, text: "Steer" }) : null,
          a.type === "edit" ? el("button", { type: "button", onclick: () => { if (!W.edits.includes(a.name)) W.edits.push(a.name); saveControls(); A.navigate("/workbench"); }, text: "Use in chat" }) : null,
          el("button", { type: "button", onclick: () => exportKeys([key]), text: "Export" }),
          el("button", { type: "button", class: "quiet", onclick: () => deleteArtifact(a), text: "Delete" })));
    })));
  return el("section", { class: "panel wb-artifacts" }, el("h2", { text: "Artifacts" }),
    el("p", { class: "hint", text: "Everything the rooms make, saved as JSON (what it is) plus an npz (its numbers): directions and probes from here, lenses from the Lens room, concept edits from the Edit room. Export a zip to move them to another machine; import a zip or a single artifact JSON." }),
    el("div", { class: "wb-row" },
      el("div", { class: "wb-view", role: "group", "aria-label": "Filter by type" }, ["", ...kinds].map((k) => el("button", { type: "button", "aria-pressed": filter === k ? "true" : "false", onclick: () => { W.artFilter = k; render(); }, text: k || "all" }))),
      el("button", { type: "button", onclick: () => exportKeys([...picked]), text: "Export selected" }),
      el("label", { class: "filebtn" }, "Import…", el("input", { type: "file", accept: ".zip,.json,application/zip,application/json", onchange: importArtifacts }))),
    rows.length ? table : el("p", { class: "empty", text: "Nothing saved yet. Save a probe's direction from the Probes page, or an edit from the Edit room." }));
}

async function exportKeys(keys) {
  if (!keys.length) { A.show("Select some artifacts first.", true); return; }
  try {
    const r = await fetch("api/workbench/artifacts/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ keys }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `export failed (${r.status})`);
    download(await r.blob(), keys.length > 1 ? "steerlab-artifacts.zip" : `${keys[0].replace("/", "-")}.zip`);
  } catch (error) { fail(error); }
}

async function importArtifacts(e) {
  const file = e.target.files?.[0]; if (!file) return;
  try {
    const zip = /\.zip$/i.test(file.name);
    const out = await postRaw("workbench/artifacts/import", await file.arrayBuffer(), zip ? "application/zip" : "application/json");
    A.show(`Imported ${out.imported.join(", ")}`); await refreshState(); await loadArtifacts(); render();
  } catch (error) { fail(error); }
}

async function deleteArtifact(a) {
  if (!confirm(`Delete ${a.type} ${a.name}?`)) return;
  try { await del(`workbench/artifacts/${a.type}/${encodeURIComponent(a.name)}`); await refreshState(); await loadArtifacts(); render(); } catch (error) { fail(error); }
}

/* ---------- ⌘K search ---------- */

function actions() {
  const rooms = (A.rooms || []).map((r) => ({ title: `Go to ${r.title}`, hint: `room · g ${r.key}`, run: () => A.navigate(r.id === "trace" ? "/" : `/${r.id}`) }));
  const node = W.node ? nodeOf(W.node) : null;
  const list = [
    ...rooms,
    { title: "Chat with the model", hint: "workbench", run: () => A.navigate("/workbench") },
    { title: "Probes: fit, read and inspect concept probes", hint: "workbench", run: () => A.navigate("/workbench/probes") },
    { title: "Artifacts: export or import directions, probes, lenses, edits", hint: "workbench", run: () => A.navigate("/workbench/artifacts") },
    { title: "New conversation", hint: "loom", run: newLoom },
    { title: "Show the loom (the tree of turns)", hint: "loom", run: () => { W.view = "loom"; if (W.tab !== "chat") A.navigate("/workbench"); else render(); } },
    { title: "Fit a new probe from examples", hint: "probes", run: () => A.navigate("/workbench/probes") },
    { title: "Clear all steering and edits", hint: "steer", run: () => { W.steer = []; W.edits = []; saveControls(); render(); } },
    { title: "Steer with an SAE feature", hint: "steer", run: () => addSteer("__sae") },
    ...(W.state?.steerables || []).map((s) => ({ title: `Steer with ${s.name.replace(/^probe:/, "")}`, hint: `${s.kind} · layer ${s.layer}`, run: () => { addSteer(s.name); if (W.tab !== "chat") A.navigate("/workbench"); } })),
    ...probeNames().map((n) => ({ title: `Probe ${n}: its layers and readings`, hint: "probe", run: () => A.navigate(`/workbench/probes/${encodeURIComponent(n)}`) })),
  ];
  if (node?.role === "assistant") {
    list.unshift({ title: "Reroll this reply", hint: "loom", run: () => reroll(node) },
      { title: "Compare with an unsteered sibling", hint: "loom", run: () => reroll(node, { unsteered: true }) },
      { title: "Replay this reply without steering", hint: "inspector", run: () => doReplay(node) });
  }
  return list;
}

function openPalette() {
  if (W.palette) return;
  const all = actions();
  const input = el("input", { type: "search", placeholder: "Search rooms, tools and kinds of steering…", "aria-label": "Search actions" });
  const list = el("ul", { role: "listbox" });
  let shown = all; let at = 0;
  const draw = () => {
    const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    shown = all.filter((a) => words.every((w) => `${a.title} ${a.hint}`.toLowerCase().includes(w))).slice(0, 12);
    at = Math.min(at, Math.max(0, shown.length - 1));
    list.replaceChildren(...shown.map((a, i) => el("li", { role: "option", "aria-selected": i === at ? "true" : "false", onmousedown: (e) => { e.preventDefault(); pick(i); } }, el("span", { text: a.title }), el("small", { text: a.hint }))));
  };
  const close = () => { W.palette?.remove(); W.palette = null; };
  const pick = (i) => { const a = shown[i]; close(); if (a) a.run(); };
  input.addEventListener("input", () => { at = 0; draw(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { at = Math.min(shown.length - 1, at + 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { at = Math.max(0, at - 1); draw(); e.preventDefault(); }
    else if (e.key === "Enter") { pick(at); e.preventDefault(); }
    else if (e.key === "Escape") { close(); e.preventDefault(); }
  });
  W.palette = el("div", { class: "wb-palette", role: "dialog", "aria-label": "Search", onmousedown: (e) => { if (e.target === W.palette) close(); } }, el("div", { class: "box" }, input, list));
  document.body.append(W.palette);
  draw(); input.focus();
}

function onKey(e) {
  if (document.documentElement.dataset.room !== "workbench") return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); e.stopImmediatePropagation(); openPalette(); return; }
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if (typing || W.palette || e.metaKey || e.ctrlKey || e.altKey) return;
  if (W.sel && (e.key === "ArrowLeft" || e.key === "ArrowRight")) { e.preventDefault(); step(e.key === "ArrowLeft" ? -1 : 1); }
  else if (e.key === "Escape" && (W.sel || W.compare)) { W.sel = null; W.compare = null; render(); }
}
