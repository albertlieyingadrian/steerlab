/* The Lens room: the residual stream read as words at every layer × position, through the logit, J, R or S lens.
 * Deep link: /lens?text=...&lens=R&z=1&layer=8&pos=3. Keys (while this room is on screen): 1-4 lens, z contrastive,
 * arrows move the selected cell, / the prompt. */
const LENSES = ["logit", "J", "R", "S"];
const ABOUT = {
  logit: "the final norm and unembedding applied as if the model stopped here",
  J: "the average Jacobian from here to the last layer, then the unembedding: what this makes the model say later",
  R: "the J-lens under RelP's rules; more faithful on early layers",
  S: "the average Jacobian from the input embeddings to here: which input tokens put this here",
};
let S = null;

const show = (text) => {
  const s = String(text).replace(/\n/g, "↵").replace(/\t/g, "⇥");
  return s.trim() === "" ? s.replace(/ /g, "␣") || "∅" : s;
};

function readRoute(route) {
  const p = route.params;
  return {
    text: p.get("text") || S.text,
    lens: LENSES.includes(p.get("lens")) ? p.get("lens") : S.lens,
    z: p.has("z") ? p.get("z") === "1" : S.z,
    layer: p.has("layer") ? Number(p.get("layer")) : S.layer,
    pos: p.has("pos") ? Number(p.get("pos")) : S.pos,
  };
}

function urlOf() {
  const q = new URLSearchParams({ text: S.text, lens: S.lens, z: S.z ? "1" : "0" });
  if (S.layer !== null) q.set("layer", S.layer);
  if (S.pos !== null) q.set("pos", S.pos);
  return `/lens?${q}`;
}

function sync() { S.api.navigate(urlOf(), { replace: true }); }

export async function mount(root, api, route) {
  const { el } = api;
  S = { api, el, root, text: "The capital of France is", lens: "logit", z: false, layer: null, pos: null, grid: null, info: null, alive: true };
  const info = await api.get("info").catch(() => ({ presets: [] }));
  S.presets = info.presets || [];
  if (S.presets.length) S.text = S.presets[0].text;
  Object.assign(S, readRoute(route));

  S.form = el("form", { class: "prompt-form", onsubmit: (e) => { e.preventDefault(); S.text = S.input.value; S.layer = S.pos = null; run(); } },
    S.input = el("input", { type: "text", maxlength: "400", spellcheck: "false", autocomplete: "off", "aria-label": "Prompt", value: S.text }),
    el("button", { type: "submit", text: "Read" }));
  S.lensBar = el("div", { class: "lens-bar", role: "group", "aria-label": "Lens" });
  S.zBox = el("label", { class: "z-toggle" }, S.zInput = el("input", { type: "checkbox", onchange: () => { S.z = S.zInput.checked; run(); } }),
    " layer-contrastive (z) ", el("span", { class: "hint", text: "removes hub tokens" }));
  S.about = el("p", { class: "hint lens-about" });
  S.gridBox = el("div", { class: "lens-grid-wrap" });
  S.detail = el("section", { class: "panel lens-detail", "aria-live": "polite" }, el("p", { class: "empty", text: "Click a cell to see its words and the units behind them." }));

  root.append(
    el("section", { class: "panel" },
      el("h2", { text: "Lens" }),
      el("div", { class: "presets" }, S.presets.map((p) => el("button", { type: "button", text: p.label, onclick: () => { S.text = S.input.value = p.text; S.layer = S.pos = null; run(); } }))),
      S.form,
      el("div", { class: "lens-controls" }, S.lensBar, S.zBox),
      S.about),
    el("div", { class: "lens-main" }, el("section", { class: "panel lens-grid-panel" }, S.gridBox), S.detail),
    el("p", { class: "foot" }, "Keys: ", el("kbd", { text: "1" }), "–", el("kbd", { text: "4" }), " lens · ", el("kbd", { text: "z" }),
      " contrastive · arrows move · ", el("kbd", { text: "/" }), " prompt. Lenses after ", el("a", { href: "https://github.com/stdlogits/neuron-lens", target: "_blank", rel: "noopener", text: "neuron-lens" }), "."));
  S.onKey = onKey;
  document.addEventListener("keydown", onKey);
  await refreshInfo();
  await run();
}

export function update(route) {
  const next = readRoute(route);
  const changed = next.text !== S.text || next.lens !== S.lens || next.z !== S.z;
  Object.assign(S, next);
  S.input.value = S.text;
  if (changed) run(); else { markSel(); if (S.layer !== null) openCell(); }
}

export function unmount() {
  if (!S) return;
  S.alive = false;
  document.removeEventListener("keydown", S.onKey);
  S = null;
}

async function refreshInfo() {
  S.info = await S.api.get("lens/info");
  renderLensBar();
}

function renderLensBar() {
  const { el } = S;
  S.lensBar.replaceChildren(...LENSES.map((name, i) => {
    const st = S.info.lenses[name] || { status: "needs a fit" };
    const usable = st.status !== "needs a fit";
    return el("button", {
      type: "button", "aria-pressed": String(S.lens === name), disabled: !usable,
      title: `${name === "logit" ? "logit lens" : `${name}-lens`}: ${st.status}${st.source ? ` · ${st.source}` : ""} (key ${i + 1})`,
      onclick: () => choose(name),
    }, name === "logit" ? "logit" : `${name}-lens`, st.status === "loaded" ? null : el("span", { class: "badge", text: st.status === "downloadable" ? "download" : st.status }));
  }));
  S.zInput.checked = S.z;
  S.about.textContent = `${S.lens === "logit" ? "Logit lens" : `${S.lens}-lens`}: ${ABOUT[S.lens]}.${S.z ? " Scores are z-scores against the layer's neuron writes." : ""}`;
}

async function choose(name) {
  const st = S.info.lenses[name];
  if (!st || st.status === "needs a fit") return;
  if (st.status !== "loaded") {
    S.api.show(`loading the ${name}-lens…`);
    try { S.info = await S.api.post("lens/load", { lens: name }); } catch (error) { S.api.show(error.message, true); return; }
  }
  S.lens = name;
  run();
}

async function run() {
  renderLensBar();
  S.gridBox.replaceChildren(S.el("p", { class: "empty", text: "reading…" }));
  let grid;
  try {
    grid = await S.api.post("lens/grid", { text: S.text, lens: S.lens, z: S.z });
  } catch (error) {
    S.gridBox.replaceChildren(S.el("p", { class: "err", text: error.message }));
    return;
  }
  if (!S || !S.alive) return;
  S.grid = grid;
  drawGrid();
  sync();
  if (S.layer !== null && S.pos !== null) openCell();
}

function shade(value) {
  const g = S.grid;
  if (g.value === "p") return Math.max(0, Math.min(1, value));
  const all = g.layers.flatMap((l) => l.cells.map((c) => c.value));
  const lo = Math.min(...all), hi = Math.max(...all);
  return hi > lo ? (value - lo) / (hi - lo) : 0;
}

function fmt(value) {
  return S.grid.value === "p" ? `${(100 * value).toFixed(value >= 0.1 ? 0 : 1)}%` : value.toFixed(1);
}

function drawGrid() {
  const { el } = S, g = S.grid;
  const cols = g.tokens.length;
  const table = el("table", { class: "lens-grid", role: "grid", "aria-label": `${g.lens} lens reads by layer and position` });
  const head = el("tr", {}, el("th", { class: "ln", text: "" }), g.tokens.map((t, i) => el("th", { class: "tokh", title: `position ${i}`, text: show(t.text) })));
  const body = [...g.layers].reverse().map((row) => el("tr", {},
    el("th", { class: "ln", scope: "row", text: `L${row.layer}` }),
    row.cells.map((c, pos) => {
      const td = el("td", {
        class: "cell", tabindex: "-1", "data-layer": row.layer, "data-pos": pos, title: `L${row.layer} · "${show(g.tokens[pos].text)}" → "${show(c.text)}" ${fmt(c.value)}`,
        onclick: () => select(row.layer, pos),
      }, el("span", { text: show(c.text) }));
      td.style.setProperty("--heat", shade(c.value).toFixed(3));
      return td;
    })));
  const out = el("tr", { class: "out-row" }, el("th", { class: "ln", text: "out" }),
    g.output.map((c) => el("td", { class: "cell out", title: `the model's real next word: ${fmt(c.value)}` }, el("span", { text: show(c.text) }))));
  table.append(el("thead", {}, head), el("tbody", {}, out, body));
  S.gridBox.replaceChildren(el("p", { class: "hint", text: `Each cell: the top word ${g.lens === "S" ? "that put the stream there" : "the stream would say next"} after that layer, at that position; shade = ${g.value === "p" ? "its probability" : g.value === "z" ? "its z-score" : "its score"}. Top row: the model's real output.` }), table);
  markSel();
}

function markSel() {
  for (const td of S.gridBox.querySelectorAll("td.cell[aria-selected]")) td.removeAttribute("aria-selected");
  if (S.layer === null) return;
  const td = S.gridBox.querySelector(`td.cell[data-layer="${S.layer}"][data-pos="${S.pos}"]`);
  if (td) { td.setAttribute("aria-selected", "true"); td.tabIndex = 0; }
}

function select(layer, pos) {
  S.layer = layer; S.pos = pos;
  markSel();
  sync();
  openCell();
}

async function openCell() {
  const { el } = S;
  const req = { text: S.text, lens: S.lens, z: S.z, layer: S.layer, pos: S.pos, k: 10 };
  S.detail.replaceChildren(el("p", { class: "empty", text: "reading…" }));
  let cell;
  try { cell = await S.api.post("lens/cell", req); } catch (error) { S.detail.replaceChildren(el("p", { class: "err", text: error.message })); return; }
  if (!S || S.layer !== req.layer || S.pos !== req.pos) return;
  const list = (rows, kind) => el("ol", { class: "wordlist-v" }, rows.map((w) => el("li", {}, el("code", { class: "chip", text: show(w.text) }),
    el("span", { class: "val", text: kind === "p" ? `${(100 * w.value).toFixed(w.value >= 0.1 ? 0 : 1)}%` : w.value.toFixed(2) }))));
  const lensCols = Object.entries(cell.reads).map(([name, r]) => el("div", { class: `lens-col ${name === cell.lens ? "on" : ""}` },
    el("h4", { text: name === "logit" ? "logit" : `${name}-lens` }),
    el("div", { class: "sub", text: "raw" }), list(r.raw.slice(0, 6), name === "S" ? "s" : "p"),
    el("div", { class: "sub", text: "z" }), list(r.z.slice(0, 6), "z")));
  const unitRows = (rows, cls) => rows.map((u) => el("tr", {},
    el("td", { class: "id" }, el("a", { href: u.href, text: u.unit })),
    el("td", { class: "num", text: u.act === undefined ? "" : u.act.toFixed(2) }),
    el("td", { class: `num ${cls(u.push)}`, text: `${u.push >= 0 ? "+" : "−"}${Math.abs(u.push).toFixed(3)}` })));
  const sign = (x) => (x >= 0 ? "pos" : "neg");
  S.detail.replaceChildren(
    el("div", { class: "pane-title" }, `L${cell.layer} · `, el("code", { text: show(cell.token.text) }), ` (position ${cell.pos})`),
    el("p", { class: "sub" }, `${cell.lens === "logit" ? "Logit lens" : `${cell.lens}-lens`}${cell.z ? ", z-scored" : ""}: top word `, el("code", { text: show(cell.top.text) })),
    el("h3", { text: "The same cell through every loaded lens" }),
    el("div", { class: "lens-cols" }, lensCols),
    el("h3", { text: `Units of layer ${cell.layer} pushing "${show(cell.top.text)}"` }),
    el("p", { class: "hint", text: "push = activation × the lens's read of the unit's write (for a neuron) or the head's output here, scored for the top word. Links open the Units room." }),
    el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "neuron" }), el("th", { class: "num", text: "act" }), el("th", { class: "num", text: "push" }))),
      el("tbody", {}, unitRows(cell.neurons.toward, sign), el("tr", { class: "gap" }, el("td", { colspan: "3", class: "sub", text: "against" })), unitRows(cell.neurons.against, sign))),
    el("table", { class: "heads" }, el("thead", {}, el("tr", {}, el("th", { text: "head" }), el("th", { class: "num", text: "" }), el("th", { class: "num", text: "push" }))),
      el("tbody", {}, unitRows(cell.heads, sign))));
}

function onKey(e) {
  if (document.documentElement.dataset.room !== "lens" || !S) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "/") { e.preventDefault(); S.input.focus(); return; }
  if (/^[1-4]$/.test(e.key)) { choose(LENSES[Number(e.key) - 1]); return; }
  if (e.key === "z") { S.z = !S.z; run(); return; }
  if (!S.grid || !e.key.startsWith("Arrow")) return;
  e.preventDefault();
  const n = S.grid.layers.length, cols = S.grid.tokens.length;
  let layer = S.layer ?? n - 1, pos = S.pos ?? cols - 1;
  if (e.key === "ArrowUp") layer = Math.min(n - 1, layer + 1);
  if (e.key === "ArrowDown") layer = Math.max(0, layer - 1);
  if (e.key === "ArrowLeft") pos = Math.max(0, pos - 1);
  if (e.key === "ArrowRight") pos = Math.min(cols - 1, pos + 1);
  select(layer, pos);
}
