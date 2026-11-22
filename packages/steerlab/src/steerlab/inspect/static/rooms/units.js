/* The Units room: an atlas of every neuron pole and head, and a page per unit.
 *
 * Routes:  /units                          the atlas of the default model (the loaded one, or its catalog entry)
 *          /units/<model>                  that model's atlas          /units/<model>/L8   the atlas, layer 8 picked
 *          /units/<model>/L8/n344+         a neuron pole (n344- the other pole)   /units/<model>/L20/h17   a head
 * Keys (in this room only): / search · r a random easy unit · b add to basket · p the other pole · [ ] layer up/down.
 * Numbers are measured and say where they come from: a catalog label is an LLM's (Claude Haiku 5.5, via the
 * stdlogits/neuron-lens catalog); a computed unit has no label, only its own tokens. Percentiles are of the unit's
 * own activation distribution. Model text goes in with textContent (api.el), never as HTML.
 * UX modelled on neuron-lens.com (atlas, unit cards, search, graph walk, basket), rebuilt here in vanilla JS.
 */
let A = null;          // the api the shell hands us
let root = null;       // our element
let S = null;          // room state
let keyHandler = null;
const BASKET_KEY = "steerlab.units.basket";

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(v < 0.001 ? 3 : v < 0.01 ? 2 : 1)}%`);
const signed = (v) => (v === null || v === undefined ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`);

function parseAddress(text) {
  const m = /^L(\d+)\/(?:n(\d+)([+-])?|h(\d+))$/i.exec(text || "");
  if (!m) return null;
  return m[4] !== undefined ? { layer: +m[1], kind: "head", index: +m[4], pole: 0 }
    : { layer: +m[1], kind: "neuron", index: +m[2], pole: m[3] === "-" ? -1 : 1 };
}
const pathOf = (a) => (a.kind === "head" ? `L${a.layer}/h${a.index}` : `L${a.layer}/n${a.index}${a.pole < 0 ? "-" : "+"}`);
const displayOf = (path) => path.replace("/", "·").replace(/-$/, "−");
const unitHref = (model, path) => A.href("units", model, ...path.split("/"));

/* Basket: units collected across models, kept in this browser only. */
function basket() {
  try { return JSON.parse(localStorage.getItem(BASKET_KEY) || "[]"); } catch { return S?.memoryBasket || []; }
}
function saveBasket(items) {
  if (S) S.memoryBasket = items.slice(0, 200);
  try { localStorage.setItem(BASKET_KEY, JSON.stringify(items.slice(0, 200))); } catch { /* private mode: memory only */ }
  renderBasketBar();
}
function inBasket(model, address) { return basket().some((b) => b.model === model && b.address === address); }
function toggleBasket(model, address, title) {
  const items = basket();
  const at = items.findIndex((b) => b.model === model && b.address === address);
  if (at >= 0) items.splice(at, 1); else items.unshift({ model, address, title: title || "" });
  saveBasket(items);
  A.show(at >= 0 ? `${displayOf(address)} out of the basket` : `${displayOf(address)} in the basket (${items.length})`);
}

/* ---------- mount, routing */

export async function mount(el, api, route) {
  A = api; root = el;
  S = { models: null, model: null, overview: null, layer: null, unit: null, results: null, pollTimer: null, memoryBasket: [] };
  root.append(shell());
  keyHandler = onKey;
  document.addEventListener("keydown", keyHandler);
  try {
    S.models = await A.get("units/models");
  } catch (error) {
    body().replaceChildren(A.el("p", { class: "err", text: `The atlas could not start: ${error.message}` }));
    return;
  }
  fillModelPicker();
  await update(route);
}

export function unmount() {
  document.removeEventListener("keydown", keyHandler);
  if (S) clearTimeout(S.pollTimer);
  S = null; root = null;
}

export async function update(route) {
  if (!S || !S.models) return;
  const [model, layerPart, unitPart] = route.path;
  const slug = model || defaultModel();
  if (!slug) return;
  if (!model) { A.navigate(A.href("units", slug), { replace: true }); return; }
  $("units-model").value = slug;
  if (slug !== S.model) { S.model = slug; S.overview = null; S.results = null; }
  $("units-results").hidden = true;
  renderBasketBar();
  const where = unitPart ? parseAddress(`${layerPart}/${unitPart}`) : null;
  if (where) return showUnit(slug, where);
  const layer = layerPart && /^L\d+$/i.test(layerPart) ? Number(layerPart.slice(1)) : null;
  return showAtlas(slug, layer);
}

function defaultModel() {
  const m = S.models;
  return m.loaded_slug || (m.models[0] && m.models[0].slug);
}

const $ = (id) => root?.querySelector(`#${id}`);
const body = () => $("units-body");

function shell() {
  const { el } = A;
  const search = el("form", { class: "u-search", role: "search", onsubmit: (e) => { e.preventDefault(); runSearch(); } },
    el("label", { class: "sr-only", for: "units-q", text: "Search units" }),
    el("input", { id: "units-q", type: "search", autocomplete: "off", spellcheck: "false",
      placeholder: "Search labels, or type L8 n344+ / L20 h17   ( / )", oninput: debounce(runSearch, 250) }),
    el("label", { class: "u-check", title: "Also search the reads' top tokens (slower the first time: it fetches every layer's reads)" },
      el("input", { type: "checkbox", id: "units-reads", onchange: runSearch }), "reads too"),
    el("div", { class: "u-results", id: "units-results", hidden: true }));
  return el("div", { class: "u-room" },
    el("div", { class: "panel u-bar" },
      el("label", { for: "units-model", class: "u-label", text: "Model" }),
      el("select", { id: "units-model", onchange: (e) => A.navigate(A.href("units", e.target.value)) }),
      search,
      el("button", { type: "button", id: "units-random", title: "A random easy unit (r)", onclick: randomUnit, text: "Random" })),
    el("div", { id: "units-body" }, el("p", { class: "empty", text: "loading…" })),
    el("div", { class: "u-basket", id: "units-basket", hidden: true }));
}

function fillModelPicker() {
  const { el } = A;
  const here = [], catalog = [];
  for (const m of S.models.models) (m.source === "catalog" ? catalog : here).push(m);
  const label = (m) => `${m.hf_id || m.slug}${m.loaded ? " (loaded)" : ""}${m.source === "catalog" && !m.labels ? " · base, labels lent" : ""}${m.source === "computable" ? " · not built yet" : ""}`;
  $("units-model").replaceChildren(
    ...(here.length ? [el("optgroup", { label: "Computed here" }, here.map((m) => el("option", { value: m.slug, text: label(m) })))] : []),
    el("optgroup", { label: "stdlogits/neuron-lens catalog" }, catalog.map((m) => el("option", { value: m.slug, text: label(m) }))));
}

function sourceBadge(source, labels) {
  const { el } = A;
  if (source === "catalog") return el("span", { class: "badge ready", title: `Labels by ${labels?.by || "Claude Haiku 5.5"} (an LLM), from the stdlogits/neuron-lens catalog`, text: "catalog · LLM labels" });
  if (source === "computed") return el("span", { class: "badge control", title: "Computed here from the model: no LLM labels", text: "computed · no labels" });
  return el("span", { class: "badge", text: source });
}

/* ---------- the atlas */

async function showAtlas(slug, layer) {
  const { el } = A;
  S.unit = null;
  document.title = `Units · ${slug}`;
  if (!S.overview || S.overview.model !== slug) {
    body().replaceChildren(el("p", { class: "empty", text: `loading ${slug}… (the first visit fetches each layer's labels from the catalog)` }));
    try {
      S.overview = await A.get(`units/overview?model=${encodeURIComponent(slug)}`);
    } catch (error) {
      body().replaceChildren(el("p", { class: "err", text: error.message }));
      return;
    }
    if (!S || S.model !== slug) return;
  }
  const ov = S.overview;
  if (ov.source === "computable") return showBuild(slug, ov);
  S.layer = layer;
  const lent = ov.labels?.lent_from;
  const picked = layer !== null && layer !== undefined;
  const head = el("section", { class: "panel" },
    el("div", { class: "u-head" },
      el("h2", { text: "Atlas" }), sourceBadge(ov.source, ov.labels),
      el("span", { class: "sub", text: `${ov.hf_id || slug} · ${ov.sizes.n_layers} layers × ${Number(ov.sizes.d_mlp).toLocaleString("en")} neurons (two poles each) + ${ov.sizes.n_heads} heads a layer` })),
    el("p", { class: "hint", text: `Each column is a layer, each bar one of its easiest units, the surest at the top; the small squares under it are its heads. Ranked by ${ov.ranking}.` }),
    lent ? el("p", { class: "hint lent", text: `A base model: the catalog has no labels for it, so its pages show the labels written for ${lent} at the same index (post-training keeps the indices). Read them as a guess about the base.` }) : null,
    ov.corpus ? el("p", { class: "hint", text: `Measured over ${ov.corpus}.` }) : null,
    atlasGrid(ov),
    el("div", { class: "u-legend", "aria-hidden": "true" },
      el("span", {}, el("i", { class: "sw full" }), ov.source === "catalog" ? "surest label shown" : "most one-token"),
      el("span", {}, el("i", { class: "sw faint" }), ov.source === "catalog" ? "least sure shown (shading is relative)" : "least one-token shown (relative)"),
      el("span", {}, el("i", { class: "sw lit" }), "matches the search")));
  const scoreName = ov.source === "catalog" ? "score" : "one-token share";
  const easiest = ov.layers.flatMap((l) => l.units.slice(0, 2)).sort((a, b) => b.score - a.score).slice(0, 16);
  const list = el("section", { class: "panel", id: "units-list" },
    el("h2", { text: picked ? `Layer ${layer}` : "Easiest units, two a layer" }),
    picked ? layerList(ov, layer) : unitTable(easiest, scoreName));
  body().replaceChildren(head, list);
  lightResults();
}

function atlasGrid(ov) {
  const { el } = A;
  const rows = Math.max(1, ...ov.layers.map((l) => l.units.length));
  // Shade relative to the scores on show: the easiest units of every layer are all fairly sure, so absolute
  // strength would paint every bar full. The legend says so.
  const scores = ov.layers.flatMap((l) => l.units.map((u) => u.score));
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const rel = (s) => (hi > lo ? (s - lo) / (hi - lo) : 1);
  const grid = el("div", { class: "u-atlas", "aria-label": "Layers by their easiest units", style: `--cols:${ov.layers.length};--rows:${rows}` });
  let band = "";
  for (const layer of ov.layers) {
    const col = el("div", { class: `u-col${S.layer === layer.layer ? " on" : ""}`, "data-layer": layer.layer });
    col.append(el("div", { class: "u-band", text: layer.band !== band ? layer.band : "" }));
    band = layer.band;
    const cells = el("div", { class: "u-cells" });
    for (const u of layer.units) {
      cells.append(el("a", { class: "u-cell", href: unitHref(ov.model, u.address), "data-address": u.address,
        style: `--s:${(0.12 + 0.88 * rel(u.score)).toFixed(3)}`, title: `${displayOf(u.address)} · ${u.title || "(no label)"} · ${fmt(u.score)}` }));
    }
    col.append(cells);
    col.append(el("a", { class: "u-ln", href: A.href("units", ov.model, `L${layer.layer}`), text: layer.layer, title: `Layer ${layer.layer}: its units and heads` }));
    const heads = el("div", { class: "u-heads" });
    for (const h of layer.heads) {
      heads.append(el("a", { class: "u-hcell", href: unitHref(ov.model, h.address), "data-address": h.address,
        style: `--s:${(0.15 + 0.85 * (h.strength ?? 0.5)).toFixed(3)}`, title: `${displayOf(h.address)} · ${h.title || "(no label)"}` }));
    }
    col.append(heads);
    grid.append(col);
  }
  return el("div", { class: "u-atlas-wrap" }, grid);
}

function layerList(ov, layer) {
  const { el } = A;
  const l = ov.layers.find((x) => x.layer === layer);
  if (!l) return el("p", { class: "empty", text: `No layer ${layer}.` });
  return el("div", {}, el("p", { class: "sub", text: `${l.band} band · its ${l.units.length} easiest of ${Number(l.n_units).toLocaleString("en")} neurons, and its ${l.heads.length} heads` }),
    el("div", { class: "u-two" },
      el("div", {}, el("h3", { text: "Neurons" }), unitTable(l.units, ov.source === "catalog" ? "score" : "one-token share")),
      el("div", {}, el("h3", { text: "Heads" }), unitTable(l.heads, ov.source === "catalog" ? "score" : "typical output"))));
}

function unitTable(items, scoreName) {
  const { el } = A;
  return el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "unit" }), el("th", { text: "label" }), el("th", { class: "num", text: scoreName }))),
    el("tbody", {}, items.map((u) => el("tr", {},
      el("td", { class: "id" }, el("a", { href: unitHref(S.model, u.address), text: displayOf(u.address) })),
      el("td", { class: u.title ? "" : "faint", text: u.title || "(no label)" }),
      el("td", { class: "num", text: fmt(u.score) })))));
}

function showBuild(slug, ov) {
  const { el } = A;
  const job = S.models.jobs?.[slug] || ov.job;
  const status = el("p", { class: "note", id: "units-job" });
  const button = el("button", { type: "button", class: "primary", text: "Compute its atlas", onclick: async () => {
    button.disabled = true;
    try { await A.post("units/build", { model: slug }); poll(slug); } catch (error) { status.textContent = error.message; button.disabled = false; }
  } });
  body().replaceChildren(el("section", { class: "panel" },
    el("div", { class: "u-head" }, el("h2", { text: "Atlas" }), el("span", { class: "badge", text: "not built yet" })),
    el("p", { class: "u-lede", text: `${ov.hf_id || slug} is not in the stdlogits/neuron-lens catalog, so its atlas is computed here from the loaded model: every neuron pole's top firing contexts and quantiles over a small corpus (prose, Python, chat; about 24,000 tokens), exact token drivers, and reads from the weights. There are no LLM labels: each unit is described by its own measured tokens.` }),
    el("p", { class: "hint", text: "It runs the loaded model, so the other rooms wait while it builds (about 2½ minutes for Qwen2.5-0.5B on a laptop CPU). Models of 4B and up are built on Modal (python -m steerlab.atlas.modal_app) and copied into the cache." }),
    el("div", { class: "actions" }, button), status));
  if (job?.state === "running") { button.disabled = true; poll(slug); }
  else if (job?.state === "failed") status.textContent = `The last build failed: ${job.error}`;
}

async function poll(slug) {
  clearTimeout(S.pollTimer);
  let data;
  try { data = await A.get("units/models"); } catch { data = null; }
  if (!S || S.model !== slug) return;
  const job = data?.jobs?.[slug];
  const status = $("units-job");
  if (job && status) status.textContent = `${job.stage} ${Math.round((job.done || 0) * 100)}% · ${Math.round(Date.now() / 1000 - job.started)} s`;
  if (job?.state === "running") { S.pollTimer = setTimeout(() => poll(slug), 1500); return; }
  if (job?.state === "failed") { if (status) status.textContent = `Failed: ${job.error}`; return; }
  S.models = data; S.overview = null;
  fillModelPicker();
  showAtlas(slug, null);
}

/* ---------- search */

async function runSearch() {
  if (!S) return;
  const q = $("units-q").value.trim();
  const box = $("units-results");
  if (!q) { box.hidden = true; S.results = null; lightResults(); return; }
  const reads = $("units-reads").checked ? 1 : 0;
  box.hidden = false;
  box.replaceChildren(A.el("p", { class: "empty", text: reads ? "searching labels and reads… (the first time fetches every layer's reads)" : "searching… (the first search builds the word index)" }));
  let found;
  try {
    found = await A.get(`units/search?model=${encodeURIComponent(S.model)}&q=${encodeURIComponent(q)}&reads=${reads}`);
  } catch (error) {
    box.replaceChildren(A.el("p", { class: "err", text: error.message }));
    return;
  }
  if (!S || $("units-q").value.trim() !== q) return;
  if (found.open) { box.hidden = true; A.navigate(unitHref(S.model, found.open)); return; }
  if (found.layer !== undefined) { box.hidden = true; A.navigate(A.href("units", S.model, `L${found.layer}`)); return; }
  S.results = found.results;
  lightResults();
  const { el } = A;
  const approx = Object.entries(found.approx || {}).map(([w, near]) => `${w} ≈ ${near.join(", ")}`);
  const n = Math.max(1, (S.overview?.sizes?.n_layers || 1) - 1);
  box.replaceChildren(
    el("p", { class: "sub", text: `${found.total.toLocaleString("en")} units match (${found.searched}${found.seconds !== undefined ? `, ${found.seconds} s` : ""})${approx.length ? ` · ${approx.join("; ")}` : ""}` }),
    found.results.length ? el("ol", { class: "u-hits" }, found.results.slice(0, 12).map((r) => el("li", {},
      el("a", { href: unitHref(S.model, r.address) },
        el("span", { class: "id", text: displayOf(r.address) }),
        el("span", { class: r.all_words ? "t" : "t faint", text: r.title || "(no label)" }),
        el("span", { class: "u-depth", title: `layer ${r.layer}` }, el("i", { style: `left:${((100 * r.layer) / n).toFixed(1)}%` }))))))
      : el("p", { class: "empty", text: found.suggest?.length ? `Nothing found. Did you mean: ${found.suggest.join(", ")}?` : "Nothing found." }));
}

function lightResults() {
  if (!root) return;
  const lit = new Set((S?.results || []).map((r) => r.address));
  for (const cell of root.querySelectorAll(".u-atlas [data-address]")) cell.classList.toggle("lit", lit.has(cell.dataset.address));
  root.querySelector(".u-atlas")?.classList.toggle("searching", lit.size > 0);
}

function randomUnit() {
  const ov = S?.overview;
  if (!ov?.layers?.length) return;
  const all = ov.layers.flatMap((l) => l.units.slice(0, 10));
  if (all.length) A.navigate(unitHref(ov.model, all[Math.floor(Math.random() * all.length)].address));
}

/* ---------- the unit page */

async function showUnit(slug, where) {
  const { el } = A;
  const path = pathOf(where);
  document.title = `${displayOf(path)} · ${slug}`;
  body().replaceChildren(el("p", { class: "empty", text: `loading ${displayOf(path)}… (a catalog layer's first visit fetches its tables)` }));
  let card;
  try {
    card = await A.get(`units/unit?model=${encodeURIComponent(slug)}&address=${encodeURIComponent(path)}`);
  } catch (error) {
    body().replaceChildren(el("section", { class: "panel" }, el("p", { class: "err", text: error.message }),
      el("p", {}, el("a", { href: A.href("units", slug), text: "Back to the atlas" }))));
    return;
  }
  if (!S || S.model !== slug) return;
  S.unit = card;
  const head = card.kind === "head";
  const sections = [titleBar(card), el("div", { class: "u-two top" }, labelCard(card), factsCard(card)), contextsCard(card)];
  if (head) sections.push(headReadsCard(card));
  else sections.push(el("div", { class: "u-two top" }, driversCard(card), readsCard(card)));
  sections.push(wiringCard(card));
  if (card.sources && Object.keys(card.sources).length) sections.push(sourcesCard(card));
  if (card.lineage?.length) sections.push(lineageCard(card));
  sections.push(fireCard(card));
  body().replaceChildren(el("div", { class: "u-unit" }, ...sections.filter(Boolean)));
  if (!head) loadGraph(card);
}

function titleBar(card) {
  const { el } = A;
  const here = parseAddress(card.address);
  const other = card.kind === "neuron" ? pathOf({ ...here, pole: -here.pole }) : null;
  const links = card.links || {};
  const step = (d) => {
    const layer = here.layer + d;
    return layer < 0 || layer >= (card.n_layers || 1) ? null : unitHref(card.model, pathOf({ ...here, layer }));
  };
  const label = () => (inBasket(card.model, card.address) ? "In basket ✓" : "Add to basket");
  const add = el("button", { type: "button", title: "Add to the basket (b)", text: label(),
    onclick: (e) => { toggleBasket(card.model, card.address, card.label?.fires || card.label?.attends); e.target.textContent = label(); } });
  return el("section", { class: "panel u-title" },
    el("div", { class: "u-crumbs" },
      el("a", { href: A.href("units", card.model), text: card.hf_id || card.model }), " / ",
      el("a", { href: A.href("units", card.model, `L${card.layer}`), text: `layer ${card.layer}` }), " / ",
      el("span", { text: card.kind === "head" ? `head ${card.index}` : `neuron ${card.index}, ${card.pole > 0 ? "+" : "−"} pole` })),
    el("div", { class: "u-head" },
      el("h1", { class: "u-id", text: card.display }),
      el("span", { class: "badge", text: `${card.band} band`, title: "Depth bands from Qwen3-8B's reading plan, scaled to this model's depth" }),
      sourceBadge(card.source, card.label),
      card.stats?.near_silent ? el("span", { class: "badge", title: "Its typical peak is under 0.05: it barely fires", text: "near-silent" }) : null),
    el("div", { class: "actions" },
      other ? el("a", { class: "btn", href: unitHref(card.model, other), title: "The neuron's other pole (p)", text: `Other pole ${displayOf(other).slice(-1)}` }) : null,
      add,
      links.trace ? el("a", { class: "btn", href: links.trace, title: "Open this unit in Trace: run a prompt and see its share of the next word", text: "Open in Trace" })
        : el("span", { class: "btn off", title: `Trace runs the loaded model (${S.models.loaded}); restart with --model ${card.hf_id || card.model}`, text: "Trace: model not loaded" }),
      el("a", { class: "btn", href: links.edit, title: "Edit this unit in the Edit room", text: "Edit this unit" }),
      links.neuron_lens ? el("a", { class: "btn", href: links.neuron_lens, target: "_blank", rel: "noopener", text: "neuron-lens.com ↗" }) : null,
      step(-1) ? el("a", { class: "btn quiet", href: step(-1), title: "The same index one layer up", text: "↑ L" + (here.layer - 1) }) : null,
      step(1) ? el("a", { class: "btn quiet", href: step(1), title: "The same index one layer down", text: "↓ L" + (here.layer + 1) }) : null));
}

function labelCard(card) {
  const { el } = A;
  const label = card.label;
  if (!label) return el("section", { class: "panel" }, el("h2", { text: "Label" }), el("p", { class: "empty", text: "The catalog has no label for this unit." }));
  const lent = label.lent_from;
  const conf = (c) => (c === null || c === undefined ? "" : `sure ${fmt(c)}`);
  const row = (k, v, c) => (v ? el("div", { class: "u-lab" }, el("div", { class: "k", text: k }), el("div", { class: `v${lent ? " lent" : ""}`, text: v }), el("div", { class: "c", title: "the labeller's own confidence, 0-1", text: conf(c) })) : null);
  const by = label.source === "catalog"
    ? `Label by ${label.by}, an LLM, from this unit's evidence card (stdlogits/neuron-lens). Its authors scored such labels blind at AUROC 0.82-0.88 per model (another unit's label: about 0.5). A description to test, not a measurement.`
    : "No LLM label: the line above counts the tokens its top firings land on, and the write line is its logit-lens read.";
  return el("section", { class: "panel u-label-card" },
    el("h2", { text: "Label" }),
    row("Fires on", label.fires, label.fires_conf),
    card.kind === "head" ? row("Attends to", label.attends, label.attends_conf) : null,
    row("Writes", label.writes, label.writes_conf),
    label.note ? el("p", { class: `note${lent ? " lent" : ""}`, text: label.note }) : null,
    lent ? el("p", { class: "hint lent", text: `Lent: written for ${lent} at the same index. This base model's unit is the same coordinate before post-training; its own contexts and reads are below.` }) : null,
    el("p", { class: "hint", text: by }));
}

function fact(k, v, tip) {
  return A.el("div", { class: "fact", title: tip || "" }, A.el("div", { class: "k", text: k }), A.el("div", { class: "v", text: v }));
}

function factsCard(card) {
  const { el } = A;
  const s = card.stats || {};
  const items = [];
  if (card.kind === "head") {
    items.push(fact("typical output", fmt(s.out_top), "Median output norm at its top firings (‖W_O z‖)"), fact("largest", fmt(s.out_max)),
      fact("to first token", pct(s.sink), "Mean attention from queries past position 16 to position 0 (the sink)"),
      fact("to previous", pct(s.prev)), fact("to itself", pct(s.self)));
    if (s.induction !== undefined && s.induction !== null) items.push(fact("induction", pct(s.induction), "Random tokens said twice: attention from the second copy to the token after the first"));
    if (s.duplicate !== undefined && s.duplicate !== null) items.push(fact("duplicate", pct(s.duplicate), "…and to the first occurrence itself"));
    items.push(fact("key-value head", String(s.kv_head ?? "–")));
  } else {
    items.push(fact("typical peak", fmt(s.act_top), "Median of its top firings (activation × the pole's sign)"), fact("largest", fmt(s.act_max)),
      fact("fires on", pct(s.rate), "Share of tokens past a quarter of its typical peak"),
      fact("peak vs layer", s.layer_pct === null || s.layer_pct === undefined ? "–" : `${Math.round(s.layer_pct)}th pct`, "Its typical peak, ranked among every pole of its layer"));
    const one = s.consistency ?? s.cur_mode;
    if (one !== undefined && one !== null) items.push(fact("one-token share", pct(one), "Share of its top firings on its most common token"));
    if (s.neg_lobe) items.push(fact("SiLU negative lobe", pct(s.neg_lobe), "Share of its top firings on SiLU's negative lobe"));
  }
  const rates = s.rates && Object.values(s.rates).some((v) => v !== null)
    ? el("p", { class: "sub", text: `Fires on ${Object.entries(s.rates).map(([k, v]) => `${pct(v)} of ${k}`).join(" · ")} tokens.` }) : null;
  const q = ["q50", "q90", "q99", "q999"].every((k) => s[k] !== undefined && s[k] !== null)
    ? el("p", { class: "sub", text: `Its own distribution: median ${fmt(s.q50, 3)} · 90th ${fmt(s.q90, 3)} · 99th ${fmt(s.q99, 3)} · 99.9th ${fmt(s.q999, 3)}.` }) : null;
  return el("section", { class: "panel" }, el("h2", { text: "How strongly it fires" }), el("div", { class: "facts" }, items), rates, q,
    el("p", { class: "hint", text: `Measured over ${card.corpus}.` }));
}

function contextsCard(card) {
  const { el } = A;
  const ctxs = card.contexts || [];
  const line = (c) => el("li", { class: "u-ctx" },
    el("span", { class: "u-text" },
      el("span", { text: c.pre }),
      c.src !== undefined ? el("mark", { class: "src", title: "the token it attends to most", text: c.src }) : null,
      c.mid !== undefined ? el("span", { text: c.mid }) : null,
      el("mark", { class: "fire", title: "where it fires", text: c.tok }),
      el("span", { text: c.post })),
    el("span", { class: "u-val", title: "the value there, and its percentile in this unit's own distribution" },
      el("b", { text: fmt(c.value) }), el("small", { text: c.pct_text || "" })));
  const join = (list) => (list || []).map(([t, n]) => `${t} ×${n}`).join("  ·  ");
  return el("section", { class: "panel" },
    el("h2", { text: "Top firing contexts" }),
    el("p", { class: "hint", text: card.kind === "head" ? "Each line is one of its top firings (its strongest position in a document): the query token in orange, the token it attends to most underlined." : "Each line is one of its top firings (its strongest position in a document), the firing token in orange." }),
    ctxs.length ? el("ol", { class: "u-ctxs" }, ctxs.map(line)) : el("p", { class: "empty", text: "It never fires in the corpus." }),
    card.peak_tokens?.length ? el("p", { class: "sub" }, el("span", { class: "k", text: "Firing token " }), join(card.peak_tokens)) : null,
    card.next_tokens?.length ? el("p", { class: "sub" }, el("span", { class: "k", text: "Next token " }), join(card.next_tokens)) : null,
    card.src_tokens?.length ? el("p", { class: "sub" }, el("span", { class: "k", text: "Attends to " }), join(card.src_tokens)) : null);
}

function chips(tokens, values) {
  const { el } = A;
  return el("div", { class: "wordlist" }, tokens.map((t, i) => el("span", { class: "chip", title: values && values[i] !== null && values[i] !== undefined ? fmt(values[i], 3) : "", text: t })));
}

function driversCard(card) {
  const { el } = A;
  const d = card.drivers || {};
  if (!d.at?.length) return el("section", { class: "panel" }, el("h2", { text: "Exact token drivers" }), el("p", { class: "empty", text: "Not available for this unit." }));
  return el("section", { class: "panel" },
    el("h2", { text: "Exact token drivers" }),
    el("p", { class: "hint", text: `Measured, not read: ${d.how}.` }),
    el("div", { class: "label", text: "At the token itself" }), chips(d.at.map(([t]) => t), d.at.map(([, v]) => v)),
    d.sum?.length ? [el("div", { class: "label", text: "Summed over it and the 16 positions after" }), chips(d.sum.map(([t]) => t), d.sum.map(([, v]) => v))] : null);
}

function readGroup(reads, side) {
  const { el } = A;
  const items = reads.filter((r) => r.side === side);
  if (!items.length) return el("p", { class: "empty", text: "none" });
  return items.map((r) => el("div", { class: "u-read", title: r.what },
    el("div", { class: "u-read-k" }, el("code", { text: r.key }), r.lens ? el("span", { class: "faint", text: r.lens === "R" || r.lens === "S" ? `${r.lens}-lens` : r.lens }) : null),
    chips(r.tokens)));
}

function readsCard(card) {
  const { el } = A;
  const reads = card.reads || [];
  const firings = card.firings || {};
  const w = card.weights || {};
  const fromWeights = Object.keys(w).length ? el("p", { class: "sub", text: `Write norm ${fmt(w.w_norm, 3)} · cos(write, gate) ${signed(w.cos_write_gate)} · cos(write, up) ${signed(w.cos_write_up)} · share in the unembedding's null space ${pct(w.null_share)}` }) : null;
  const atFirings = firings.pred?.length ? [el("h3", { text: "At its firings" }),
    el("p", { class: "hint", text: `The residual at its ${firings.n_firings} top firings, through the R-lens.${firings.parallel !== null && firings.parallel !== undefined ? ` Its write ${firings.parallel >= 0 ? "amplifies" : "suppresses"} what it reads (×${fmt(Math.abs(firings.parallel))}) and adds ×${fmt(firings.orthogonal)} off it.` : ""}` }),
    el("div", { class: "label", text: "What its input holds there" }), chips(firings.before_R || []),
    el("div", { class: "label", text: "What the model predicts there" }), chips(firings.pred || [])] : null;
  return el("section", { class: "panel" },
    el("h2", { text: "Reads from its weights" }),
    el("p", { class: "hint", text: card.source === "catalog" ? "The catalog's fitted Jacobian lenses: R reads what a vector makes the model say, S which input tokens put it there; _z is scored against the layer's other poles. A read names a direction, not an effect size. Hover a read for what it is." : "From the weights with the logit lens and the embedding: rough before the last third of the model. Hover a read for what it is." }),
    el("h3", { text: "What drives it" }), readGroup(reads, "fires on"),
    el("h3", { text: "What it writes" }), readGroup(reads, "writes"),
    fromWeights, atFirings);
}

function headReadsCard(card) {
  const { el } = A;
  const reads = card.reads || [];
  const d = card.drivers || {};
  const c = card.copy || {};
  const scores = [];
  if (c.ov_R_copy !== undefined && c.ov_R_copy !== null) scores.push(`copying score (OV·R) ${signed(c.ov_R_copy)}`, `seeks its own token (QK self·S) ${signed(c.qk_self_S)}`, `attends to what it copies (ρ) ${signed(c.qk_ov_rho)}`);
  if (c.copy_score !== undefined && c.copy_score !== null) scores.push(`copying score Σλ/Σ|λ| ${signed(c.copy_score)}`, `own-token z, median ${signed(c.copy_median_z)}`, `seeks its own token, median z ${signed(c.self_median_z)}`);
  return el("section", { class: "panel" },
    el("h2", { text: "OV and QK reads" }),
    el("p", { class: "hint", text: card.source === "catalog" ? "Its circuits read token by token over the vocabulary: OV is what it copies, QK where it looks (the catalog's lenses)." : `From the weights over the corpus's ${c.n_candidates || ""} most common tokens, position-free (RoPE at offset 0): what it copies (OV) and where it looks (QK). +1 copies, −1 suppresses.` }),
    scores.length ? el("p", { class: "sub", text: scores.join(" · ") }) : null,
    el("div", { class: "u-two" }, el("div", {}, el("h3", { text: "What it writes (OV)" }), readGroup(reads, "writes")), el("div", {}, el("h3", { text: "Where it looks (QK)" }), readGroup(reads, "attends"))),
    d.at?.length ? [el("h3", { text: "Exact token drivers" }), el("p", { class: "hint", text: d.how }),
      el("div", { class: "label", text: "As the query" }), chips(d.at.map(([t]) => t)), el("div", { class: "label", text: "As the key" }), chips((d.keys || []).map(([t]) => t))] : null,
    card.out_R_z?.length ? [el("div", { class: "label", text: "What its outputs at its firings say (R-lens, z)" }), chips(card.out_R_z)] : null);
}

/* ---------- wiring */

function edgeTable(items, side) {
  const { el } = A;
  if (!items?.length) return el("p", { class: "empty", text: side === "in" ? "No earlier units listed." : "No later units listed." });
  const max = Math.max(...items.map((e) => Math.abs(e.push ?? e.cos ?? 0)), 1e-9);
  return el("table", {}, el("tbody", {}, items.slice(0, 10).map((e) => {
    const v = e.push ?? e.cos ?? 0;
    return el("tr", {},
      el("td", { class: "id" }, el("a", { href: unitHref(S.model, e.address), text: displayOf(e.address) })),
      el("td", { class: e.title ? "" : "faint", text: e.title || "(no label)" }),
      el("td", { class: "num u-push", title: `push ${fmt(v, 3)} · cos ${fmt(e.cos, 3)}` },
        el("span", { class: `u-pbar${v < 0 ? " against" : ""}`, style: `width:${Math.round((44 * Math.abs(v)) / max)}px` }), signed(v)));
  })));
}

function wiringCard(card) {
  const { el } = A;
  const edges = card.edges || {};
  if (card.kind === "head") return el("section", { class: "panel" }, el("h2", { text: "Wiring" }), el("p", { class: "hint", text: edges.how || "" }));
  if (!edges.in?.length && !edges.out?.length) return el("section", { class: "panel" }, el("h2", { text: "Wiring" }), el("p", { class: "hint", text: edges.how || "No edges for this unit." }));
  const pick = (id, options, value) => el("select", { id, onchange: () => loadGraph(card) }, options.map((n) => el("option", { value: n, selected: n === value, text: String(n) })));
  return el("section", { class: "panel" },
    el("h2", { text: "Wiring" }),
    el("p", { class: "hint", text: edges.how || "" }),
    el("div", { class: "u-two" },
      el("div", {}, el("h3", { text: "What drives it (earlier units)" }), edgeTable(edges.in, "in")),
      el("div", {}, el("h3", { text: "What it drives (later units)" }), edgeTable(edges.out, "out"))),
    el("div", { class: "u-graph-bar" },
      el("h3", { text: "Walk the graph" }),
      el("label", {}, "up ", pick("units-up", [0, 1, 2, 3], 2)), el("label", {}, "down ", pick("units-down", [0, 1, 2, 3], 2)),
      el("label", {}, "per hop ", pick("units-fan", [2, 3, 4, 6], 3)),
      el("span", { class: "hint", text: "Layers run top to bottom. Orange links drive the unit they reach; grey dashed ones push against it; width is |push|. Click a unit to walk there." })),
    el("div", { id: "units-graph", class: "u-graph" }, el("p", { class: "empty", text: "drawing the graph…" })));
}

async function loadGraph(card) {
  const box = $("units-graph");
  if (!box) return;
  const up = $("units-up")?.value ?? 2, down = $("units-down")?.value ?? 2, fan = $("units-fan")?.value ?? 3;
  let g;
  try {
    g = await A.get(`units/graph?model=${encodeURIComponent(card.model)}&address=${encodeURIComponent(card.address)}&up=${up}&down=${down}&fan=${fan}`);
  } catch (error) {
    if ($("units-graph")) box.replaceChildren(A.el("p", { class: "err", text: error.message }));
    return;
  }
  if (!S || S.unit !== card) return;
  box.replaceChildren(drawGraph(g, card));
}

function drawGraph(g, card) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
  const layers = [...new Set(g.nodes.map((n) => n.layer))].sort((a, b) => a - b);
  const rowH = 78, padX = 44, width = 900;
  const height = Math.max(1, layers.length) * rowH + 8;
  const pos = new Map();
  for (const [r, layer] of layers.entries()) {
    const row = g.nodes.filter((n) => n.layer === layer);
    row.forEach((n, i) => pos.set(n.address, { x: padX + ((i + 0.5) * (width - padX - 8)) / row.length, y: 22 + r * rowH, n }));
  }
  const out = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "u-graph-svg", role: "img", "aria-label": `Wiring around ${card.display}` });
  layers.forEach((layer, r) => {
    out.append(svg("line", { x1: padX - 6, x2: width - 4, y1: 22 + r * rowH, y2: 22 + r * rowH, class: "row" }));
    const t = svg("text", { x: 2, y: 26 + r * rowH, class: "ln" }); t.textContent = `L${layer}`; out.append(t);
  });
  const max = Math.max(...g.links.map((l) => Math.abs(l.push ?? l.cos ?? 0)), 1e-9);
  for (const l of g.links) {
    const a = pos.get(l.source), b = pos.get(l.target);
    if (!a || !b) continue;
    const v = l.push ?? l.cos ?? 0;
    const midY = (a.y + b.y) / 2;
    out.append(svg("path", { d: `M${a.x},${a.y} C${a.x},${midY} ${b.x},${midY} ${b.x},${b.y}`, class: `link${v < 0 ? " against" : ""}`,
      "stroke-width": (0.8 + (3.4 * Math.abs(v)) / max).toFixed(2) }));
  }
  for (const { x, y, n } of pos.values()) {
    const center = n.address === card.address;
    const link = svg("a", { href: unitHref(card.model, n.address), class: `node${center ? " here" : ""}` });
    const tip = svg("title"); tip.textContent = `${n.display} · ${n.title || "(no label)"}`; link.append(tip);
    link.append(svg("circle", { cx: x, cy: y, r: center ? 8 : 5.5 }));
    const id = svg("text", { x, y: y + 19, class: "id" }); id.textContent = n.display; link.append(id);
    const t = (n.title || "").replace(/^fires on /, "");
    const title = svg("text", { x, y: y + 31, class: "tt" }); title.textContent = t.length > 30 ? `${t.slice(0, 29)}…` : t; link.append(title);
    out.append(link);
  }
  return out;
}

function sourcesCard(card) {
  const { el } = A;
  const s = card.sources;
  const list = (items, key, label) => el("div", {}, el("h3", { text: label }), items?.length ? el("table", {}, el("tbody", {}, items.slice(0, 8).map((e) => el("tr", {},
    el("td", { class: "id" }, el("a", { href: unitHref(S.model, e.address), text: displayOf(e.address) })),
    el("td", { class: e.title ? "" : "faint", text: e.title || "(no label)" }),
    el("td", { class: "num", text: key === "lift" ? `lift ×${fmt(e.lift, 0)}` : pct(e.share) }))))) : el("p", { class: "empty", text: "none listed" }));
  if (card.kind === "head") {
    return el("section", { class: "panel" }, el("h2", { text: "What drives it when it fires" }), el("p", { class: "hint", text: s.how }),
      el("div", { class: "u-three" }, ["v", "q", "k"].map((side) => list(s[side]?.driven_by, "share", { v: "Through its value", q: "Through its query", k: "Through its key" }[side]))));
  }
  const sh = s.shares || {};
  return el("section", { class: "panel" },
    el("h2", { text: "What drives it when it fires" }),
    el("p", { class: "hint", text: `${s.how} (traffic, unlike the wiring above). Mean shares: the embedding ${pct(sh.embedding)}, heads ${pct(sh.heads)}, earlier neurons ${pct(sh.neurons)}.` }),
    el("div", { class: "u-three" }, list(s.driven_by, "share", "Earlier neurons"), list(s.driven_by_heads, "share", "Heads"), list(s.fires_with, "lift", "Fires with")));
}

function lineageCard(card) {
  const { el } = A;
  const names = { parent: "Pruned from", child: "Kept in", base: "Before post-training", "post-trained": "After post-training" };
  return el("section", { class: "panel" },
    el("h2", { text: "Lineage" }),
    el("p", { class: "hint", text: "Across sizes and training: Qwen3-4B and 0.6B are pruned from Qwen3-8B and 1.7B, and a base model keeps its post-trained sibling's indices. Matches from the catalog's lineage table." }),
    el("table", {}, el("tbody", {}, card.lineage.map((x) => el("tr", {},
      el("td", { text: names[x.relation] || x.relation }),
      el("td", { class: "id" }, x.address ? el("a", { href: unitHref(x.model, x.address), text: `${x.model} ${displayOf(x.address)}` }) : `${x.model}: cut`),
      el("td", { class: x.title ? "" : "faint", text: x.title || "" }),
      el("td", { class: "num", title: x.how || "", text: `${x.status}${x.by?.length ? ` (${x.by.join(", ")})` : ""}` }))))));
}

/* ---------- fire: run a prompt and watch the unit */

function fireCard(card) {
  const { el } = A;
  if (!card.fireable) {
    return el("section", { class: "panel" }, el("h2", { text: "Run it on a prompt" }),
      el("p", { class: "hint", text: `Prompts run on the loaded model (${S.models.loaded}). To watch this unit fire, restart with: steerlab inspect --model ${card.hf_id || card.model}` }));
  }
  const out = el("div", { id: "units-fire-out" });
  const first = card.contexts?.[0];
  const input = el("input", { type: "text", id: "units-fire-text", maxlength: 600, autocomplete: "off", spellcheck: "false",
    value: (first ? `${first.pre}${first.src ?? ""}${first.mid ?? ""}${first.tok}${first.post}` : "The capital of France is Paris.").replaceAll("⏎", "\n") });
  const run = async (e) => {
    e?.preventDefault();
    out.replaceChildren(el("p", { class: "empty", text: "running…" }));
    try {
      const r = await A.post("units/fire", { model: card.model, address: card.address, text: input.value });
      out.replaceChildren(fireStrip(r));
    } catch (error) { out.replaceChildren(el("p", { class: "err", text: error.message })); }
  };
  return el("section", { class: "panel" }, el("h2", { text: "Run it on a prompt" }),
    el("p", { class: "hint", text: "The loaded model reads your text; each token shows this unit's value there, shaded by its percentile in the unit's own distribution (white under the median, full orange at the top)." }),
    el("form", { class: "prompt-form", onsubmit: run }, input, el("button", { type: "submit", text: "Run" })), out);
}

function fireStrip(r) {
  const { el } = A;
  const strip = el("div", { class: "u-fire" }, r.tokens.map((t, i) => el("span", {
    class: `tok${i === r.peak ? " peak" : ""}`, style: `--p:${t.pct === null ? 0 : Math.max(0, (t.pct - 50) / 50).toFixed(3)}`,
    title: `${fmt(t.value, 3)} · ${t.pct_text}`,
  }, el("span", { text: t.text.replace(/\n/g, "⏎") || "⎵" }), el("small", { text: fmt(t.value, 2) }))));
  const peak = r.tokens[r.peak];
  return el("div", {}, strip,
    el("p", { class: "sub", text: `Peak ${fmt(peak.value, 3)} at “${peak.text}” (${peak.pct_text}${r.act_top !== null && r.act_top !== undefined ? `; its typical top firing is ${fmt(r.act_top, 3)}` : ""}).` }),
    el("p", { class: "hint", text: r.basis }));
}

/* ---------- basket */

function renderBasketBar() {
  const bar = $("units-basket");
  if (!bar) return;
  const items = basket();
  bar.hidden = items.length === 0;
  if (!items.length) return;
  const { el } = A;
  const byModel = {};
  for (const b of items) (byModel[b.model] ||= []).push(b.address);
  const copy = () => {
    const text = items.map((b) => `${b.model} ${b.address}${b.title ? `  # ${b.title}` : ""}`).join("\n");
    navigator.clipboard?.writeText(text).then(() => A.show("Basket copied"), () => A.show("Copy failed", true));
  };
  const first = Object.keys(byModel)[0];
  const open = bar.querySelector("details")?.open;
  bar.replaceChildren(
    el("details", { open: open || null },
      el("summary", {}, el("b", { text: `Basket · ${items.length}` }), el("span", { class: "hint", text: "  units you collected, in this browser" })),
      el("ul", {}, items.map((b) => el("li", {},
        el("a", { href: unitHref(b.model, b.address), text: displayOf(b.address) }), el("span", { class: "faint", text: ` ${b.model}` }),
        el("span", { class: "t", text: b.title ? ` · ${b.title}` : "" }),
        el("button", { type: "button", class: "quiet", title: "Remove", "aria-label": `Remove ${displayOf(b.address)}`, text: "×",
          onclick: () => saveBasket(basket().filter((x) => !(x.model === b.model && x.address === b.address))) })))),
      el("div", { class: "actions" },
        el("a", { class: "btn", href: `/edit?model=${encodeURIComponent(first)}&units=${encodeURIComponent(byModel[first].join(","))}`, text: `Edit these in ${first}` }),
        el("button", { type: "button", onclick: copy, text: "Copy list" }),
        el("button", { type: "button", onclick: () => saveBasket([]), text: "Empty" }))));
}

/* ---------- keys */

function onKey(e) {
  if (document.documentElement.dataset.room !== "units" || !S) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "") || document.activeElement?.isContentEditable;
  if (e.key === "Escape" && typing) { document.activeElement.blur(); $("units-results").hidden = true; return; }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const card = S.unit;
  if (e.key === "/") { e.preventDefault(); $("units-q").focus(); }
  else if (e.key === "r") randomUnit();
  else if (e.key === "b" && card) toggleBasket(card.model, card.address, card.label?.fires || card.label?.attends);
  else if (e.key === "p" && card?.kind === "neuron") { const a = parseAddress(card.address); A.navigate(unitHref(card.model, pathOf({ ...a, pole: -a.pole }))); }
  else if ((e.key === "[" || e.key === "]") && S.model) {
    const current = card ? card.layer : (S.layer ?? -1);
    const n = S.overview?.sizes?.n_layers ?? card?.n_layers ?? 1;
    A.navigate(A.href("units", S.model, `L${Math.max(0, Math.min(n - 1, current + (e.key === "]" ? 1 : -1)))}`));
  }
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}
