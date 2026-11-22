/* The Edit room: build a concept edit (axis, units, the k dial, scope), compare before / after against a random-axis
 * and a random-unit control of the same size, restore, and save it to the shared edit list or as an artifact JSON.
 * Orange = the edit's effect, fader blue = the controls you set, grey = random controls. Model text goes in with
 * textContent (api.el) only.
 * Keys (in this room, not while typing): c compare · x load the example · r restore · a add to the edit list.
 * Deep links: /edit?example=1 loads the mother ↔ father example (&run=1 also compares); /edit?model=<slug>&units=
 * L8/n344+,L3/h5 (what the Units room's basket sends) puts those units in the unit list.
 */
let alive = false;
let keyHandler = null;
let applyRoute = null;
const S = { info: null, result: null, stack: [], busy: false };

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const signed = (v, d = 2) => (v === null || v === undefined ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
const COLS = { before: "Before", after: "After", random_axis: "Random axis", random_units: "Random units" };
const CHECK_UNITS = { choice: "nats", words: "−1…1", probe: "kept" };

function kLabel(k) {
  if (k === 2) return "mirror (swap)";
  if (k === 1) return "project out (ablate)";
  if (k === 0) return "no change";
  if (k < 0) return `amplify ×${(1 - k).toFixed(2)}`;
  if (k > 2) return "past the mirror";
  return k < 1 ? "partial projection" : "between project and mirror";
}

function parsePairs(text) {
  return text.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => l.split("|").map((s) => s.trim()))
    .filter((p) => p.length === 2 && p[0] && p[1]);
}

function parseChecks(text, words, probe) {
  const checks = text.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => l.split("|").map((s) => s.trim()))
    .filter((p) => p.length === 3 && p.every(Boolean)).map(([question, a, b]) => ({ kind: "choice", question, a, b }));
  const [src, tgt] = words.split("|").map((s) => (s || "").split(",").map((w) => w.trim()).filter(Boolean));
  if (src && src.length && tgt && tgt.length) checks.push({ kind: "words", source: src, target: tgt });
  if (probe) checks.push({ kind: "probe" });
  return checks;
}

function parseLayers(text, n) {
  const t = text.trim();
  if (!t) return null;
  const out = new Set();
  for (const part of t.split(",")) {
    const m = part.trim().match(/^(\d+)(?:\s*-\s*(\d+))?$/);
    if (!m) throw new Error(`layers: write 0-27 or 4,8,12, not "${part.trim()}"`);
    const a = Number(m[1]), b = m[2] === undefined ? a : Number(m[2]);
    for (let i = Math.min(a, b); i <= Math.max(a, b) && i < n; i++) out.add(i);
  }
  return [...out].sort((x, y) => x - y);
}

export async function mount(root, api, route) {
  alive = true;
  const { el } = api;
  const F = {};  // form controls
  const $ = (id) => root.querySelector(`#${id}`);

  const status = el("p", { class: "note", role: "status" });
  const head = el("section", { class: "panel", "aria-labelledby": "h-edit" },
    el("h2", { id: "h-edit", text: "Edit a concept in the weights" }),
    el("p", { class: "hint", text: "Pick an axis (a contrast such as mother − father), the neurons and heads to edit, and a dial k. "
      + "Each chosen unit's reads and writes go through H = I − k d dᵀ: k = 2 mirrors the axis (swap), k = 1 removes it, "
      + "k < 0 amplifies it. Every result is shown next to the same edit along a random axis and on random units." }),
    el("p", { class: "sub", id: "edit-model", text: "loading…" }), status);
  root.append(head);

  // 1. Axis
  F.kind = el("select", { id: "ax-kind", "aria-label": "Axis source" },
    el("option", { value: "pairs", text: "word pairs" }), el("option", { value: "sentences", text: "sentence pairs" }),
    el("option", { value: "template", text: "chat prompt pairs (template)" }));
  F.method = el("select", { id: "ax-method", "aria-label": "How word pairs become an axis" },
    el("option", { value: "activations", text: "activations in neutral frames" }),
    el("option", { value: "embed", text: "embedding / unembedding" }),
    el("option", { value: "lens", text: "Jacobian lens", id: "opt-lens" }));
  F.pairs = el("textarea", { id: "ax-pairs", rows: "6", spellcheck: "false", "aria-label": "Pairs, one per line: source | target",
    placeholder: "mother | father\nmom | dad" });
  const axis = el("section", { class: "panel", "aria-labelledby": "h-axis" },
    el("h2", { id: "h-axis" }, el("span", { class: "step", text: "1" }), "Axis"),
    el("div", { class: "edit-row" }, el("label", { for: "ax-kind", text: "from" }), F.kind, el("span", { id: "method-wrap" }, F.method)),
    F.pairs,
    el("p", { class: "gloss", id: "ax-gloss" }),
    el("div", { class: "actions" }, el("button", { type: "button", id: "load-example", onclick: () => loadExample() }, "Load the mother ↔ father example"),
      el("kbd", { text: "x" })));

  // 2. Units
  F.selection = el("select", { id: "u-sel", "aria-label": "How units are chosen" },
    el("option", { value: "top", text: "top loading per layer" }), el("option", { value: "ids", text: "a list (Units room basket)" }),
    el("option", { value: "all", text: "every unit" }));
  F.ukind = el("select", { id: "u-kind", "aria-label": "Unit kind" },
    el("option", { value: "neurons", text: "neurons" }), el("option", { value: "heads", text: "attention heads" }),
    el("option", { value: "both", text: "neurons and heads" }));
  F.per = el("input", { id: "u-per", type: "number", min: "1", max: "4096", value: "30", "aria-label": "Neurons per layer" });
  F.hper = el("input", { id: "u-hper", type: "number", min: "1", max: "64", value: "2", "aria-label": "Heads per layer" });
  F.layers = el("input", { id: "u-layers", type: "text", placeholder: "all layers, or 8-20", "aria-label": "Scope: layers", spellcheck: "false" });
  F.ids = el("textarea", { id: "u-ids", rows: "3", spellcheck: "false", placeholder: "one per line or comma: L8.N344, L8/n12+, L10.H3, /units/<model>/L8/n344+", "aria-label": "Unit ids" });
  const unitTable = el("div", { id: "u-summary" });
  const units = el("section", { class: "panel", "aria-labelledby": "h-units" },
    el("h2", { id: "h-units" }, el("span", { class: "step", text: "2" }), "Units"),
    el("div", { class: "edit-row" }, F.selection, F.ukind),
    el("div", { class: "edit-row", id: "u-counts" }, el("label", { for: "u-per", text: "neurons / layer" }), F.per,
      el("label", { for: "u-hper", text: "heads / layer" }), F.hper),
    el("div", { class: "edit-row" }, el("label", { for: "u-layers", text: "scope" }), F.layers),
    el("div", { id: "u-ids-wrap", hidden: true }, F.ids),
    el("div", { class: "actions" }, el("button", { type: "button", onclick: () => previewUnits() }, "Preview units")),
    unitTable);

  // 3. Dial
  F.k = el("input", { id: "k", type: "range", min: "-4", max: "4", step: "0.25", value: "2", class: "fader", "aria-label": "k" });
  const kOut = el("output", { for: "k", class: "k-value" });
  const kWord = el("span", { class: "k-word" });
  F.sides = el("select", { id: "sides", "aria-label": "Neuron sides" },
    el("option", { value: "both", text: "reads and writes" }), el("option", { value: "reads", text: "reads only" }),
    el("option", { value: "writes", text: "writes only" }));
  F.parts = Object.fromEntries("qkvo".split("").map((p) => [p, el("input", { type: "checkbox", checked: true, id: `part-${p}` })]));
  const ticks = el("div", { class: "ticks", "aria-hidden": "true" },
    [-4, -2, 0, 1, 2, 4].map((v) => el("span", { style: `left:${((v + 4) / 8) * 100}%`, text: String(v) })));
  const dial = el("section", { class: "panel", "aria-labelledby": "h-dial" },
    el("h2", { id: "h-dial" }, el("span", { class: "step", text: "3" }), "Dial"),
    el("div", { class: "fader-wrap" }, F.k, ticks),
    el("p", { class: "k-line" }, el("span", { class: "chan", text: "k" }), kOut, kWord),
    el("div", { class: "edit-row" }, el("label", { for: "sides", text: "neurons" }), F.sides),
    el("div", { class: "edit-row" }, el("span", { class: "label-inline", text: "head parts" }),
      Object.entries(F.parts).map(([p, box]) => el("label", { class: "part" }, box, p.toUpperCase()))));
  const syncK = () => { const k = Number(F.k.value); kOut.textContent = k.toFixed(2); kWord.textContent = kLabel(k); };
  F.k.addEventListener("input", syncK);
  syncK();

  // 4. Run
  F.prompts = el("textarea", { id: "prompts", rows: "3", spellcheck: "false", "aria-label": "Prompts, one per line (up to 4)" });
  F.choices = el("textarea", { id: "choices", rows: "2", spellcheck: "false", "aria-label": "Forced choices: question | answer a | answer b",
    placeholder: "Who gives birth? Answer with one word: Mother or Father. | Mother | Father" });
  F.words = el("input", { id: "words", type: "text", spellcheck: "false", "aria-label": "Word count: source words | target words",
    placeholder: "mother, mom | father, dad" });
  F.probe = el("input", { type: "checkbox", id: "probe", checked: true });
  F.controls = el("input", { type: "checkbox", id: "controls", checked: true });
  F.sample = el("input", { type: "checkbox", id: "sample" });
  F.temp = el("input", { id: "temp", type: "number", min: "0.1", max: "2", step: "0.1", value: "0.8", "aria-label": "Temperature" });
  F.seed = el("input", { id: "seed", type: "number", value: "0", "aria-label": "Seed" });
  F.tokens = el("input", { id: "tokens", type: "number", min: "1", max: "120", value: "40", "aria-label": "New tokens" });
  const runBtn = el("button", { type: "button", class: "primary", id: "compare", onclick: () => compare() }, "Compare");
  const run = el("section", { class: "panel", "aria-labelledby": "h-run" },
    el("h2", { id: "h-run" }, el("span", { class: "step", text: "4" }), "Compare"),
    el("label", { class: "label", for: "prompts", text: "Prompts (one per line, chat template on)" }), F.prompts,
    el("label", { class: "label", for: "choices", text: "Forced choices: question | a | b (margin log p(a) − log p(b), nats)" }), F.choices,
    el("label", { class: "label", for: "words", text: "Word count in the replies: source words | target words" }), F.words,
    el("div", { class: "edit-row" },
      el("label", { class: "part" }, F.probe, "axis probe (kept: 1 intact, 0 fused, −1 swapped)"),
      el("label", { class: "part" }, F.controls, "random controls")),
    el("div", { class: "edit-row" }, el("label", { class: "part" }, F.sample, "sample"), el("label", { for: "temp", text: "temp" }), F.temp,
      el("label", { for: "seed", text: "seed" }), F.seed, el("label", { for: "tokens", text: "tokens" }), F.tokens),
    el("div", { class: "actions" }, runBtn, el("kbd", { text: "c" }),
      el("button", { type: "button", id: "restore", onclick: () => restore() }, "Restore and check"), el("kbd", { text: "r" })));

  const results = el("section", { class: "panel results", "aria-labelledby": "h-results", id: "results" },
    el("h2", { id: "h-results", text: "Before and after" }), el("p", { class: "empty", text: "Load the example (x) or build an edit, then Compare (c)." }));
  const list = el("section", { class: "panel", "aria-labelledby": "h-list", id: "edit-list" },
    el("h2", { id: "h-list", text: "Edit list" }), el("div", { id: "stack" }));

  root.append(el("div", { class: "edit-grid" }, axis, units, dial, run), results, list);

  // behaviour
  const say = (msg, bad = false) => { status.textContent = msg; status.classList.toggle("err", bad); };
  const syncKind = () => {
    $("method-wrap").hidden = F.kind.value !== "pairs";
    $("ax-gloss").textContent = {
      pairs: "Single words, one pair per line. Activations: each pair is dropped into four neutral sentences and the axis is the difference at the word, per layer.",
      sentences: "Sentences that differ only in the concept word, one pair per line. The axis is the difference at the word, per layer.",
      template: "Two user messages per line (a plain one and one with an instruction). The axis is the difference over the chat template after the message.",
    }[F.kind.value];
  };
  const syncSel = () => {
    $("u-ids-wrap").hidden = F.selection.value !== "ids";
    $("u-counts").hidden = F.selection.value !== "top";
    const k = F.ukind.value;
    for (const [node, show] of [[F.per, k !== "heads"], [F.hper, k !== "neurons"]]) {
      node.hidden = !show;
      node.previousElementSibling.hidden = !show;
    }
  };
  F.kind.addEventListener("change", syncKind);
  F.selection.addEventListener("change", syncSel);
  F.ukind.addEventListener("change", syncSel);
  syncKind();
  syncSel();

  function spec() {
    const pairs = parsePairs(F.pairs.value);
    if (!pairs.length) throw new Error("Write at least one pair as: source | target");
    const parts = Object.entries(F.parts).filter(([, b]) => b.checked).map(([p]) => p).join("") || "qkvo";
    return {
      axis: { kind: F.kind.value, method: F.method.value, pairs },
      units: {
        selection: F.selection.value, kind: F.ukind.value, per_layer: Number(F.per.value) || 30,
        heads_per_layer: Number(F.hper.value) || 2, layers: parseLayers(F.layers.value, S.info?.n_layers ?? 999),
        ids: F.ids.value.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean),
      },
      k: Number(F.k.value), sides: F.sides.value, parts,
    };
  }

  async function guarded(label, fn) {
    if (S.busy) return;
    S.busy = true;
    runBtn.disabled = true;
    say(label);
    try { await fn(); } catch (error) { say(error.message, true); } finally { S.busy = false; runBtn.disabled = false; }
  }

  function loadExample() {
    const ex = S.info?.example;
    if (!ex) return;
    F.kind.value = ex.axis.kind;
    F.pairs.value = ex.axis.pairs.map((p) => p.join(" | ")).join("\n");
    F.selection.value = ex.units.selection;
    F.ukind.value = ex.units.kind;
    F.per.value = String(ex.units.per_layer);
    F.k.value = String(ex.k);
    F.prompts.value = ex.prompts.join("\n");
    F.choices.value = ex.checks.filter((c) => c.kind === "choice").map((c) => `${c.question} | ${c.a} | ${c.b}`).join("\n");
    const w = ex.checks.find((c) => c.kind === "words");
    F.words.value = w ? `${w.source.join(", ")} | ${w.target.join(", ")}` : "";
    F.probe.checked = ex.checks.some((c) => c.kind === "probe");
    syncKind(); syncSel(); syncK();
    say(`Loaded: ${ex.title}. ${ex.units.per_layer} neurons per layer, k = ${ex.k}.`);
  }

  async function previewUnits() {
    await guarded("choosing units…", async () => {
      const data = await api.post("edit/units", spec());
      renderUnits(data);
      say(`${data.count.neurons} neurons and ${data.count.heads} heads.`);
    });
  }

  function renderUnits(u) {
    const rows = u.layers.map((r) => el("tr", {},
      el("td", { class: "id", text: `L${r.layer}` }), el("td", { class: "num", text: String(r.neurons) }),
      el("td", { class: "num", text: String(r.heads) }),
      el("td", { class: "num", text: r.share === undefined ? "–" : `${(100 * r.share).toFixed(2)}%` }),
      el("td", { class: "num", text: r.vs_layer === undefined ? "–" : `×${r.vs_layer.toFixed(1)}` })));
    const shares = u.layers.filter((r) => r.vs_layer !== undefined).map((r) => r.vs_layer);
    const median = shares.length ? [...shares].sort((a, b) => a - b)[Math.floor(shares.length / 2)] : null;
    unitTable.replaceChildren(
      el("p", { class: "sub", text: `${u.count.neurons} neurons, ${u.count.heads} heads over ${u.layers.length} layers`
        + (median ? ` · median loading ×${median.toFixed(1)} the layer's median neuron` : "") }),
      el("details", {}, el("summary", { text: "per layer" }),
      el("p", { class: "gloss", text: "Share: the median squared cosine of a chosen neuron's read or write vector with the axis. "
        + "×: against the layer's median neuron." }),
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", { text: "layer" }), el("th", { class: "num", text: "neurons" }),
          el("th", { class: "num", text: "heads" }), el("th", { class: "num", text: "share" }), el("th", { class: "num", text: "vs layer" }))),
        el("tbody", {}, rows)))),
      el("p", { class: "ids" }, (u.links || []).slice(0, 24).map((x) => el("a", { class: "chip", href: x.href, text: x.id })),
        u.more + Math.max(0, u.ids.length - 24) ? el("span", { class: "hint", text: ` and ${u.more + Math.max(0, u.ids.length - 24)} more` }) : null));
  }

  async function compare() {
    await guarded("editing, generating and restoring… (four columns)", async () => {
      const body = {
        ...spec(),
        prompts: F.prompts.value.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 4),
        checks: parseChecks(F.choices.value, F.words.value, F.probe.checked),
        controls: F.controls.checked, sample: F.sample.checked, temperature: Number(F.temp.value) || 0.8,
        seed: Number(F.seed.value) || 0, max_new_tokens: Number(F.tokens.value) || 40,
      };
      if (!body.prompts.length) throw new Error("Write at least one prompt");
      const t0 = performance.now();
      const data = await api.post("edit/compare", body);
      if (!alive) return;
      S.result = data;
      renderResult(data);
      renderUnits(data.summary);
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      say(`Done in ${secs} s. ${data.restored ? "Weights restored bit-identically." : "Restore check FAILED."}`, !data.restored);
    });
  }

  async function restore() {
    await guarded("restoring…", async () => {
      const r = await api.post("edit/restore");
      say(r.restored ? "Restored: logits bit-identical to startup (max |Δ| = 0)." : `Not identical: max |Δ| = ${r.max_abs_diff}`, !r.restored);
    });
  }

  async function addToList() {
    await guarded("adding to the edit list…", async () => {
      const data = await api.post("edit/stack", { ...spec(), label: "" });
      S.stack = data.stack;
      renderStack();
      say("Added. Every comparison now runs on top of the enabled edits in the list.");
    });
  }

  function download(obj, name) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  const fileName = () => `edit-${(S.info?.model || "model").replace(/[^\w.-]+/g, "_")}-${new Date().toISOString().slice(0, 10)}.json`;

  function renderResult(r) {
    const order = r.order;
    const header = el("tr", {}, el("th", { text: "check" }), order.map((c) => el("th", { class: `num col-${c}`, text: COLS[c] })));
    const checkRows = (r.columns.before.checks || []).map((chk, i) => {
      const name = chk.kind === "choice" ? `${chk.a} vs ${chk.b}` : chk.kind === "words" ? "word count" : `probe (L${chk.layer ?? "?"})`;
      return el("tr", {},
        el("td", { class: "words", title: chk.question || "" }, name, el("small", { class: "hint", text: ` ${CHECK_UNITS[chk.kind]}` })),
        order.map((c) => {
          const v = r.columns[c].checks[i];
          const text = chk.kind === "choice" ? signed(v.value) : chk.kind === "words" ? `${signed(v.value)} (${v.source}:${v.target})` : fmt(v.value);
          return el("td", { class: `num col-${c}`, text });
        }));
    });
    const replies = r.prompts.map((p, i) => el("div", { class: "reply-row" },
      el("p", { class: "prompt", text: p }),
      el("div", { class: "reply-grid" }, order.map((c) => el("div", { class: `reply col-${c}` },
        el("h4", { text: COLS[c] }), el("blockquote", { text: r.columns[c].replies[i] || "(empty)" }))))));
    const info = r.columns.after.info || {};
    const badge = el("span", { class: `badge ${r.restored ? "ready" : ""}`, text: r.restored ? "restored · bit-identical" : "restore FAILED" });
    const decoding = r.sample ? ` · sampled, seed ${r.seed}` : " · greedy, repetition_penalty 1.0";
    const shares = ` · median share read ${(100 * (info.median_read_share || 0)).toFixed(2)}%, write ${(100 * (info.median_write_share || 0)).toFixed(2)}%`;
    results.replaceChildren(...[
      el("h2", { id: "h-results", text: "Before and after" }),
      el("p", { class: "sub" }, badge, " ",
        el("span", { text: `k = ${r.k} (${kLabel(r.k)}) · ${r.units.neurons} neurons, ${r.units.heads} heads · ${r.sides}${decoding}${shares}` }),
        r.exact ? null : el("span", { class: "badge", text: "approximate (Gemma post-norms)" })),
      r.base.length ? el("p", { class: "gloss", text: `On top of the edit list: ${r.base.join(" + ")}` }) : null,
      checkRows.length ? el("div", { class: "scroll" }, el("table", { class: "checks" }, el("thead", {}, header), el("tbody", {}, checkRows))) : null,
      el("p", { class: "gloss", text: "Random axis: the same units, edited along a random direction. Random units: as many units per layer, "
        + "chosen at random, along the real axis. An effect counts only where it beats both." }),
      ...replies,
      el("div", { class: "actions" },
        el("button", { type: "button", onclick: () => addToList() }, "Add to edit list"), el("kbd", { text: "a" }),
        el("button", { type: "button", onclick: () => download(r.artifact, fileName()) }, "Download edit JSON")),
    ].filter(Boolean));
  }

  async function act(fn) {
    try { const data = await fn(); S.stack = data.stack || []; renderStack(); } catch (error) { say(error.message, true); }
  }
  const del = (path) => fetch(`api/${path}`, { method: "DELETE" }).then((x) => x.json());

  function renderStack() {
    const box = $("stack");
    const fileInput = el("input", { type: "file", accept: "application/json,.json", "aria-label": "Import an edit JSON",
      onchange: async (ev) => {
        const file = ev.target.files[0];
        if (!file) return;
        await guarded("importing…", async () => {
          const artifact = JSON.parse(await file.text());
          const data = await api.post("edit/import", { artifact });
          S.stack = data.stack;
          renderStack();
          say(`Imported ${artifact.label || "the edit"}.`);
        });
      } });
    const rows = S.stack.map((rec) => el("div", { class: `stack-item ${rec.enabled ? "" : "off"}` },
      el("span", { class: `badge ${rec.kind === "reflect" ? "ready" : "control"}`, text: rec.kind === "reflect" ? "weights" : "neuron" }),
      el("span", { class: "label-text", text: rec.label }),
      el("button", { type: "button", onclick: () => act(() => api.post(`edit/stack/${rec.id}/toggle`)) }, rec.enabled ? "Disable" : "Enable"),
      rec.kind === "reflect" ? el("button", { type: "button", title: "Save to the artifact store, where the Workbench finds it by name",
        onclick: () => saveRecord(rec) }, "Save") : null,
      el("button", { type: "button", onclick: async () => download(await api.get(`edit/stack/${rec.id}/artifact`), fileName()) }, "JSON"),
      el("button", { type: "button", "aria-label": `Remove ${rec.label}`, onclick: () => act(() => del(`edit/stack/${rec.id}`)) }, "✕")));
    box.replaceChildren(
      el("p", { class: "hint", text: "Shared with the Trace room (its Edit tab sends its neuron edits here). Enabled edits apply under every comparison above; the model is restored after each request." }),
      rows.length ? el("div", { class: "stack-list" }, rows) : el("p", { class: "empty", text: "Empty." }),
      el("div", { id: "saved-edits" }),
      el("div", { class: "actions" }, el("label", { class: "part" }, "Import edit JSON ", fileInput),
        S.stack.length ? el("button", { type: "button", onclick: () => act(() => del("edit/stack")) }, "Clear list") : null));
    renderSaved();
  }

  async function saveRecord(rec) {
    const name = (window.prompt("Save as (letters, digits, . _ -)", rec.label.replace(/[^A-Za-z0-9._-]+/g, "-").slice(0, 60)) || "").trim();
    if (!name) return;
    await guarded("saving…", async () => {
      const r = await api.post(`edit/stack/${rec.id}/save`, { name });
      say(`Saved as ${r.saved}. In the Workbench: edits: ["${r.saved.slice(5)}"].`);
      renderSaved();
    });
  }

  async function renderSaved() {
    const box = $("saved-edits");
    if (!box) return;
    let rows = [];
    try { rows = (await api.get("edit/saved")).saved; } catch (error) { box.replaceChildren(el("p", { class: "err", text: error.message })); return; }
    if (!alive) return;
    box.replaceChildren(rows.length ? el("details", {}, el("summary", { text: `${rows.length} saved edit${rows.length === 1 ? "" : "s"} (shared with the Workbench)` }),
      el("div", { class: "stack-list" }, rows.map((row) => el("div", { class: "stack-item" },
        el("code", { text: row.name }), el("span", { class: "label-text", text: `${row.label || ""} · ${row.model || "?"}` }),
        el("button", { type: "button", disabled: row.model && row.model !== S.info?.model,
          onclick: () => act(() => api.post("edit/load", { name: row.name })) }, "Load"))))) : "");
  }

  keyHandler = (e) => {
    if (document.documentElement.dataset.room !== "edit") return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    const map = { c: compare, x: loadExample, r: restore, a: addToList };
    if (map[e.key]) { e.preventDefault(); map[e.key](); }
  };
  document.addEventListener("keydown", keyHandler);

  try {
    S.info = await api.get("edit/info");
  } catch (error) {
    $("edit-model").replaceWith(el("p", { class: "err", text: error.message }));
    return;
  }
  if (!alive) return;
  const i = S.info;
  $("edit-model").replaceChildren(el("code", { text: i.model }),
    ` · ${i.n_layers} layers · ${i.d_mlp.toLocaleString("en")} neurons and ${i.n_heads} heads per layer · `,
    el("span", { class: `badge ${i.exact ? "ready" : ""}`, text: i.exact ? "exact: k = 2 conjugates each unit" : "approximate: Gemma post-norms" }));
  if (!i.lens) $("opt-lens").textContent = "Jacobian lens (not installed: falls back to activations)";
  F.per.max = String(i.d_mlp);
  S.stack = i.stack;
  renderStack();
  applyRoute = (r) => {
    const params = r?.params;
    if (!params) return;
    if (params.get("example")) {
      loadExample();
      if (params.get("run")) compare();
    }
    const ids = (params.get("units") || "").split(",").map((s) => s.trim()).filter(Boolean);
    if (!ids.length) return;
    F.selection.value = "ids";
    F.ids.value = ids.join("\n");
    syncSel();
    const from = params.get("model");
    if (from && from !== S.info.slug && from !== S.info.model) {
      say(`${ids.length === 1 ? "This unit comes" : `These ${ids.length} units come`} from ${from}, but the loaded model is ${S.info.model} (${S.info.slug}). `
        + `Restart with that model to edit them here.`, true);
      return;
    }
    say(`${ids.length} unit${ids.length === 1 ? "" : "s"} from the Units room are in the unit list. Pick an axis, then Preview units or Compare.`);
    if (parsePairs(F.pairs.value).length) previewUnits();
  };
  applyRoute(route);
}

export function update(route) {
  if (applyRoute) applyRoute(route);
}

export function unmount() {
  alive = false;
  applyRoute = null;
  if (keyHandler) document.removeEventListener("keydown", keyHandler);
  keyHandler = null;
}
