/* steerlab inspect: the layer explorer page.
 *
 * One page, two data sources. Live (`steerlab inspect`): every view is a POST to api/<view>. Static (the published
 * guide): <html data-source="data.json"> points at precomputed presets, and views not in the file say so.
 * Model text is always inserted with textContent (el() below), never as HTML.
 *
 * URL parameters pick a starting view: ?preset=france&layer=21&neuron=4090&tab=neuron (tab: layer, neuron, find,
 * edit, dose; &edit=N runs the N-th saved edit, &dose=L21.N4090 runs a dose test).
 */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);

  function el(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  const svg = (tag, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  };

  /* A token as text you can see: newlines and tabs become symbols, an all-space token becomes ␣. */
  function show(text) {
    const s = String(text).replace(/\r/g, "␍").replace(/\n/g, "↵").replace(/\t/g, "⇥");
    return s.trim() === "" ? s.replace(/ /g, "␣") || "∅" : s;
  }
  const quoted = (text) => `"${show(text)}"`;
  function pct(p) {
    if (p >= 0.9995) return "100%";
    if (p >= 0.1) return `${(100 * p).toFixed(0)}%`;
    if (p >= 0.001) return `${(100 * p).toFixed(1)}%`;
    return "<0.1%";
  }
  const signed = (x, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
  const fixed = (x, d = 2) => `${x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`;
  const nid = (layer, neuron) => `L${layer}.N${neuron}`;
  function parseId(text) {
    const m = /^\s*L?(\d+)\s*[.:,\s]\s*N?(\d+)\s*$/i.exec(text);
    return m ? { layer: Number(m[1]), neuron: Number(m[2]) } : null;
  }
  function ordinal(n) {
    const s = ["th", "st", "nd", "rd"], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  class Unavailable extends Error {}

  /* Live: the steerlab inspect server. */
  class LiveSource {
    constructor() { this.isStatic = false; }
    async call(path, body) {
      const init = body === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      };
      const response = await fetch(`api/${path}`, init);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
        throw new Error(detail || `request failed (${response.status})`);
      }
      return data;
    }
    base(s) { return { text: s.text, target: s.target, pos: s.pos }; }
    info() { return this.call("info"); }
    run(s) { return this.call("run", this.base(s)); }
    layer(s, layer) { return this.call("layer", { ...this.base(s), layer }); }
    neuron(s, layer, neuron) { return this.call("neuron", { ...this.base(s), layer, neuron }); }
    find(s, word) { return this.call("find", { word, text: s.text, pos: s.pos }); }
    edit(s, edits) { return this.call("edit", { ...this.base(s), edits }); }
    dose(s, layer, neuron) { return this.call("dose", { ...this.base(s), layer, neuron }); }
  }

  /* Static: precomputed presets in data.json. */
  const LOCAL = "Run steerlab inspect on your computer to compute it.";
  class StaticSource {
    constructor(data) { this.isStatic = true; this.data = data; }
    preset(s) { return this.data.presets.find((p) => p.id === s.preset) || this.data.presets[0]; }
    async info() {
      const { model, n_layers, n_heads, d_mlp, created } = this.data;
      const presets = this.data.presets.map(({ id, label, text, note }) => ({ id, label, text, note }));
      return { model, n_layers, n_heads, d_mlp, created, presets };
    }
    async run(s) { return this.preset(s).run; }
    async layer(s, layer) {
      const view = this.preset(s).layers[String(layer)];
      if (!view) throw new Unavailable(`Layer ${layer} is not saved. ${LOCAL}`);
      return view;
    }
    async neuron(s, layer, neuron) {
      const view = this.preset(s).neurons[nid(layer, neuron)];
      if (!view) throw new Unavailable(`${nid(layer, neuron)} is not in this page's saved data. ${LOCAL}`);
      return view;
    }
    async find(s, word) {
      const saved = this.preset(s).find;
      if (word.trim() !== saved.word.text.trim()) {
        throw new Unavailable(`Saved data has find for ${quoted(saved.word.text)} only. ${LOCAL}`);
      }
      return saved;
    }
    editKey(edits) {
      return edits.map((e) => `${e.layer}.${e.neuron}.${e.mode}.${Number(e.value || 0)}`).sort().join("|");
    }
    async edit(s, edits) {
      const p = this.preset(s);
      const found = p.edits.find((option) => this.editKey(option.edits) === this.editKey(edits));
      if (!found) throw new Unavailable(`That edit is not saved. ${LOCAL}`);
      return {
        edits: found.labels, normal: p.run, edited: found.edited,
        normal_text: found.normal_text, edited_text: found.edited_text,
      };
    }
    async dose(s, layer, neuron) {
      const view = this.preset(s).doses[nid(layer, neuron)];
      if (!view) throw new Unavailable(`No saved dose test for ${nid(layer, neuron)}. ${LOCAL}`);
      return view;
    }
    editOptions(s) { return this.preset(s).edits; }
    doseOptions(s) { return Object.keys(this.preset(s).doses); }
  }

  const S = {
    src: null, info: null, preset: null, text: "", target: null, pos: null, run: null,
    layer: null, layerView: null, neuron: null, tab: "layer", edits: [], editResult: null, editError: null,
    editKey: null, dose: null, doseError: null, doseId: null, findWord: "", allHeads: false, seq: 0,
  };

  /* Status toast */
  let statusTimer = 0;
  function status(text, bad = false) {
    const node = $("status");
    clearTimeout(statusTimer);
    node.textContent = text;
    node.classList.toggle("bad", bad);
    node.classList.toggle("on", Boolean(text));
    if (text && bad) statusTimer = setTimeout(() => node.classList.remove("on"), 4000);
  }
  async function busy(label, work) {
    const timer = setTimeout(() => status(label), 150);
    try {
      const result = await work();
      clearTimeout(timer);
      status("");
      return result;
    } catch (error) {
      clearTimeout(timer);
      if (error instanceof Unavailable) status("");
      else status(error.message, true);
      throw error;
    }
  }
  const problem = (error) => el("p", { class: error instanceof Unavailable ? "empty" : "err", text: error.message });

  /* Tabs */
  const TABS = ["layer", "neuron", "find", "edit", "dose"];
  function setTab(tab, focus = false) {
    S.tab = tab;
    for (const name of TABS) {
      const button = $(`tab-${name}`);
      const on = name === tab;
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1;
      $(`pane-${name}`).hidden = !on;
    }
    // The dose form starts from the neuron opened last, so draw it again when it has no result yet.
    if (tab === "dose" && S.src && !S.src.isStatic && !S.dose && !S.doseError) renderDose();
    if (focus) $(`tab-${tab}`).focus();
  }
  /* On a phone the detail panel sits under the stack: bring it into view when something opens there. */
  function reveal() {
    if (window.matchMedia("(max-width: 900px)").matches) {
      document.querySelector(".detail").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  /* 1. Words */
  function renderPresets() {
    $("presets").replaceChildren(...S.info.presets.map((p) => el("button", {
      type: "button", "aria-pressed": String(p.id === S.preset), "data-id": p.id, title: p.text,
      onclick: () => loadPreset(p.id),
    }, p.label)));
  }

  function renderWords() {
    const run = S.run;
    const live = !S.src.isStatic;
    $("tokens").replaceChildren(...run.tokens.map((text, i) => {
      const selected = i === run.pos;
      return el("button", {
        type: "button", class: "tok", role: "option", "aria-selected": String(selected), tabindex: selected ? 0 : -1,
        "data-i": i, title: live ? "Predict what comes after this token" : null,
        "aria-disabled": live ? null : "true",
        onclick: () => { if (live && i !== S.run.pos) { S.pos = i; S.target = null; runAll(); } },
      }, show(text));
    }));
    $("tokens-hint").textContent = live
      ? "click a token to see what the model predicts after it"
      : "saved presets predict after the last token";
    $("guesses").replaceChildren(...run.guesses.map((g) => el("button", {
      type: "button", class: "chip guess", "aria-pressed": String(g.id === run.target.id),
      "aria-disabled": live ? null : "true", title: live ? "Make this the target" : null,
      onclick: () => { if (live && g.id !== S.run.target.id) { S.target = g.id; runAll(); } },
    }, el("span", { text: show(g.text) }), el("span", { class: "p", text: pct(g.prob) }))));
    $("guesses-hint").textContent = live ? "click one to make it the target" : "the top guess is the saved target";
    const t = run.target;
    $("target-now").replaceChildren(
      "now ", el("b", { text: quoted(t.text) }), ` · p = ${pct(t.prob)} · rank ${t.rank} · logit ${fixed(t.logit)}`,
    );
    $("target-input").disabled = !live;
    document.querySelector("#target-form button").disabled = !live;
  }

  /* 2. Layer stack */
  function bar(kind, value, scale) {
    const width = scale > 0 ? Math.min(50, (50 * Math.abs(value)) / scale) : 0;
    const left = value >= 0 ? 50 : 50 - width;
    return el("span", { class: `bar ${kind}`, title: `${kind === "attn" ? "attention" : "neurons"} ${signed(value)}` },
      el("i", { style: `left:${left}%;width:${width}%` }));
  }

  function renderStack() {
    const run = S.run;
    const layers = run.layers.filter((r) => r.layer >= 0);
    const scale = Math.max(1e-9, ...layers.flatMap((r) => [Math.abs(r.attn), Math.abs(r.mlp)]));
    $("rows").replaceChildren(...[...run.layers].reverse().map((r) => {
      const embed = r.layer < 0;
      const selected = r.layer === S.layer;
      const says = el("span", { class: `says${r.target_rank === 1 ? " hit" : ""}` },
        el("span", { class: "w", text: show(r.says) }),
        el("span", { class: "rank", text: `#${r.target_rank}` }));
      const label = embed
        ? `Embedding: share ${signed(run.embed)}; would say ${quoted(r.says)}; target rank ${r.target_rank}`
        : `Layer ${r.layer}: attention ${signed(r.attn)}, neurons ${signed(r.mlp)}; would say ${quoted(r.says)}; `
          + `target rank ${r.target_rank}`;
      return el("button", {
        type: "button", class: `row${embed ? " embed" : ""}`, role: "option", "aria-selected": String(selected),
        tabindex: selected ? 0 : -1, "data-layer": r.layer, "aria-label": label, title: label,
        onclick: () => { if (!embed) { selectLayer(r.layer, true); reveal(); } },
      },
      el("span", { class: "ln", text: embed ? "emb" : `L${r.layer}` }),
      embed ? el("span", { class: "hint", style: "grid-column: span 2", text: `token embedding ${signed(run.embed)}` })
        : [bar("attn", r.attn, scale), bar("mlp", r.mlp, scale)],
      says);
    }));
    const attn = layers.reduce((a, r) => a + r.attn, 0);
    const mlp = layers.reduce((a, r) => a + r.mlp, 0);
    $("equation").replaceChildren(
      `logit(${quoted(run.target.text)}) = embed `, el("b", { text: signed(run.embed) }),
      " + attention ", el("b", { text: signed(attn) }), " + neurons ", el("b", { text: signed(mlp) }),
      " = ", el("b", { text: fixed(run.sum, 3) }), el("br"),
      `real logit ${fixed(run.target.logit, 3)} · leftover `, el("b", { text: run.leftover.toExponential(1) }),
      ` (final-norm scale frozen at 1/rms = ${run.norm_scale.toPrecision(3)})`,
    );
  }

  function markRow() {
    for (const row of $("rows").children) {
      const on = Number(row.dataset.layer) === S.layer;
      row.setAttribute("aria-selected", String(on));
      row.tabIndex = on ? 0 : -1;
    }
  }

  /* 3. Layer detail */
  async function selectLayer(layer, switchTab = false) {
    S.layer = layer;
    markRow();
    if (switchTab) setTab("layer");
    const seq = ++S.seq;
    try {
      const view = await busy(`reading layer ${layer}…`, () => S.src.layer(S, layer));
      if (seq !== S.seq) return;
      S.layerView = view;
      renderLayer();
    } catch (error) {
      if (seq === S.seq) $("pane-layer").replaceChildren(problem(error));
    }
  }

  function neuronTable(rows, found = false) {
    const head = el("tr", {},
      el("th", { text: "neuron" }), el("th", { class: "num", text: found ? "cos" : "fires" }),
      el("th", { class: "num hide-sm", text: found ? "fires here" : "× lines up" }),
      el("th", { class: "num", text: found ? "share here" : "= share" }),
      el("th", { text: "promotes" }));
    const open = (r) => openNeuron(r.layer, r.neuron);
    const body = rows.map((r) => el("tr", {
      class: "click", tabindex: 0, role: "button", "aria-label": `Open neuron ${r.id}`,
      onclick: () => open(r),
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(r); } },
    },
    el("td", { class: "id", text: r.id }),
    el("td", { class: "num", text: found ? fixed(r.cos, 3) : fixed(r.act) }),
    el("td", { class: "num hide-sm", text: found ? (r.act === undefined ? "" : fixed(r.act)) : fixed(r.align, 3) }),
    el("td", { class: `num ${(r.share ?? 0) >= 0 ? "pos" : "neg"}`, text: r.share === undefined ? "" : signed(r.share) }),
    el("td", { class: "words", text: r.promotes.map(quoted).join(" ") })));
    return el("table", {}, el("thead", {}, head), el("tbody", {}, body));
  }

  function renderLayer() {
    const v = S.layerView;
    const heads = S.allHeads ? v.heads : v.heads.slice(0, 6);
    const headTable = el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "head" }), el("th", { class: "num", text: "share" }),
        el("th", { text: "looks at" }), el("th", { class: "hide-sm", text: "promotes" }))),
      el("tbody", {}, heads.map((h) => el("tr", {},
        el("td", { class: "id", text: `L${v.layer}.H${h.head}` }),
        el("td", { class: `num ${h.share >= 0 ? "pos" : "neg"}`, text: signed(h.share) }),
        el("td", { class: "words", text: h.attends.map((a) => `${quoted(a.text)} ${pct(a.weight)}`).join(", ") }),
        el("td", { class: "words hide-sm", text: h.promotes.map(quoted).join(" ") })))));
    $("pane-layer").replaceChildren(
      el("div", { class: "pane-title" }, `Layer ${v.layer}`),
      el("p", { class: "sub" }, `Toward ${quoted(v.target.text)}: attention ${signed(v.attn)}, neurons ${signed(v.mlp)}.`),
      el("p", { class: "gloss" }, "A neuron's ", el("b", { text: "share" }),
        " is how strongly it fires here × how well its output lines up with the target word. Shares add up to the bar."),
      el("h3", { text: "Neurons pushing toward it" }), neuronTable(v.toward),
      el("h3", { text: "Neurons pushing against it" }), neuronTable(v.against),
      el("h3", { text: `Attention heads (${v.heads.length})` }), headTable,
      v.heads.length > 6 ? el("div", { class: "actions" }, el("button", {
        type: "button", onclick: () => { S.allHeads = !S.allHeads; renderLayer(); },
      }, S.allHeads ? "Show top 6" : `Show all ${v.heads.length}`)) : null,
    );
  }

  /* 4. Neuron detail */
  async function openNeuron(layer, neuron, switchTab = true) {
    S.neuron = { layer, neuron };
    if (switchTab) setTab("neuron");
    try {
      const view = await busy(`reading ${nid(layer, neuron)}…`, () => S.src.neuron(S, layer, neuron));
      renderNeuron(view);
    } catch (error) {
      $("pane-neuron").replaceChildren(problem(error));
    }
    if (switchTab) reveal();
  }

  function savedEdit(layer, neuron, mode) {
    if (!S.src.isStatic) return null;
    return S.src.editOptions(S).find((o) => o.edits.length === 1 && o.edits[0].layer === layer
      && o.edits[0].neuron === neuron && o.edits[0].mode === mode) || null;
  }

  function renderNeuron(v) {
    const live = !S.src.isStatic;
    const peak = Math.max(1e-9, ...v.fires.map((f) => Math.abs(f.act)));
    const fires = v.fires.map((f, i) => {
      const alpha = (0.08 + (0.72 * Math.abs(f.act)) / peak).toFixed(2);
      const color = f.act >= 0 ? `rgba(201, 86, 28, ${alpha})` : `rgba(74, 134, 232, ${alpha})`;
      return el("span", {
        class: "tok", style: `background:${color}${i === v.pos ? ";outline:1px solid var(--mlp-mark)" : ""}`,
        title: `fires ${fixed(f.act)} on ${quoted(f.text)}`,
      }, show(f.text), el("small", { text: fixed(f.act, 1) }));
    });
    const action = (label, mode) => {
      const saved = savedEdit(v.layer, v.neuron, mode);
      return el("button", {
        type: "button", disabled: !live && !saved, title: !live && !saved ? `Not saved for this neuron. ${LOCAL}` : null,
        onclick: () => editNow([{ layer: v.layer, neuron: v.neuron, mode, value: 0 }]),
      }, label);
    };
    const doseSaved = live || S.src.doseOptions(S).includes(v.id);
    $("pane-neuron").replaceChildren(
      el("div", { class: "pane-title" }, el("code", { text: v.id }), ` · layer ${v.layer}, neuron ${v.neuron}`),
      el("p", { class: "gloss" }, "Here it fires ", el("b", { text: fixed(v.act) }), " × lines up ",
        el("b", { text: fixed(v.align, 3) }), ` with ${quoted(v.target.text)} = `,
        el("b", { class: v.share >= 0 ? "pos" : "neg", text: `${signed(v.share)} logit` }),
        v.rank_in_layer === 1 ? "; the largest share in its layer." : `; the ${ordinal(v.rank_in_layer)} largest share in its layer.`),
      v.act < 0 ? el("p", { class: "gloss" }, el("b", { text: "It fires negative here" }), ", so it pushes up the words "
        + "it suppresses and pushes down the words it promotes: read the lists below the other way round.") : null,
      el("h3", { text: "How strongly it fires on each token" }), el("div", { class: "fires" }, fires),
      el("h3", { text: "Words it promotes (when it fires positive)" }),
      el("div", { class: "wordlist" }, v.promotes.map((w) => el("span", { class: "chip pos", text: show(w) }))),
      el("h3", { text: "Words it suppresses" }),
      el("div", { class: "wordlist" }, v.suppresses.map((w) => el("span", { class: "chip neg", text: show(w) }))),
      el("p", { class: "gloss", text: "These lists read the neuron's output straight through the unembedding, skipping "
        + "every later layer, so a neuron can mean more than its top words. Check it with an edit and a dose test." }),
      el("div", { class: "actions" },
        action("Switch off and compare", "off"), action("Flip", "flip"),
        live ? el("button", { type: "button", onclick: () => addEdit(v.layer, v.neuron, "off") }, "Add to edit list") : null,
        el("button", {
          type: "button", class: "primary", disabled: !doseSaved,
          title: doseSaved ? null : `No saved dose test for this neuron. ${LOCAL}`,
          onclick: () => doseNow(v.layer, v.neuron),
        }, "Dose test")),
    );
  }

  /* 5. Find */
  function renderFind(result = null, error = null) {
    const input = el("input", {
      id: "find-word", type: "text", value: S.findWord, maxlength: 40, "aria-label": "Word to find",
      autocomplete: "off", spellcheck: "false",
    });
    const kids = [
      el("div", { class: "pane-title", text: "Find neurons that write a word" }),
      el("p", { class: "sub", text: "Every neuron in every layer, ranked by how closely its output direction matches "
        + "the word's readout (cosine). Some write it when they fire positive, others when they fire negative." }),
      el("form", { class: "find-form", onsubmit: (e) => { e.preventDefault(); doFind(input.value); } },
        input, el("button", { type: "submit" }, "Find")),
    ];
    if (error) kids.push(problem(error));
    if (result) {
      kids.push(
        el("h3", { text: `Write ${quoted(result.word.text)} when firing positive` }), neuronTable(result.toward, true),
        el("h3", { text: "Write it when firing negative" }), neuronTable(result.against, true),
        el("p", { class: "gloss", text: "“Share here” is the neuron's share of this word's logit in the current prompt: "
          + "a neuron can write the word well and still be silent here." }),
      );
    }
    $("pane-find").replaceChildren(...kids);
  }
  async function doFind(word) {
    S.findWord = word;
    if (!word.trim()) return;
    try {
      renderFind(await busy(`finding ${quoted(word)}…`, () => S.src.find(S, word)));
    } catch (error) {
      renderFind(null, error);
    }
  }

  /* 6. Edit */
  function addEdit(layer, neuron, mode = "off", value = 0) {
    if (S.edits.length >= 8 && !S.edits.some((e) => e.layer === layer && e.neuron === neuron)) {
      status("At most 8 edits at once", true);
      return;
    }
    S.edits = S.edits.filter((e) => !(e.layer === layer && e.neuron === neuron));
    S.edits.push({ layer, neuron, mode, value: mode === "scale" && !value ? 2 : value });
    setTab("edit");
    renderEdit();
    reveal();
  }
  async function editNow(edits) {
    if (!S.src.isStatic) S.edits = edits.map((e) => ({ ...e }));
    setTab("edit");
    renderEdit();
    await runEdits(edits);
    reveal();
  }
  async function runEdits(edits = S.edits) {
    if (!edits.length) return;
    S.editKey = JSON.stringify(edits);
    try {
      S.editResult = await busy("running with the edits…", () => S.src.edit(S, edits));
      S.editError = null;
    } catch (error) {
      S.editResult = null;
      S.editError = error;
    }
    renderEdit();
  }

  function editRow(e, i) {
    const value = el("input", {
      type: "number", step: "0.5", value: e.value, "aria-label": `Value for ${nid(e.layer, e.neuron)}`,
      hidden: e.mode === "off" || e.mode === "flip",
      onchange: (ev) => { e.value = Number(ev.target.value); },
    });
    const names = { off: "switch off", flip: "flip", scale: "scale ×", set: "set to" };
    const mode = el("select", {
      "aria-label": `Edit for ${nid(e.layer, e.neuron)}`,
      onchange: (ev) => {
        e.mode = ev.target.value;
        if (e.mode === "scale" && !e.value) e.value = 2;
        renderEdit();
      },
    }, Object.entries(names).map(([m, name]) => el("option", { value: m, selected: m === e.mode }, name)));
    return el("div", { class: "edit-item" }, el("code", { text: nid(e.layer, e.neuron) }), mode, value,
      el("button", {
        type: "button", "aria-label": `Remove ${nid(e.layer, e.neuron)}`,
        onclick: () => { S.edits.splice(i, 1); renderEdit(); },
      }, "✕"));
  }

  function renderEdit() {
    const kids = [
      el("div", { class: "pane-title", text: "Edit neurons and compare" }),
      el("p", { class: "sub", text: "Switch neurons off, flip them or scale them at every position, then rerun: "
        + "the target's probability, each layer's push and a greedy continuation, normal vs edited." }),
    ];
    if (S.src.isStatic) {
      kids.push(el("div", { class: "picks", role: "group", "aria-label": "Saved edits" },
        S.src.editOptions(S).map((o) => el("button", {
          type: "button", "aria-pressed": String(S.editKey === JSON.stringify(o.edits)),
          onclick: () => runEdits(o.edits),
        }, o.labels.join(" + ")))));
      kids.push(el("p", { class: "gloss", text: "Saved edits for this preset: the three neurons with the largest share, "
        + `each switched off; the top two flipped; the top one ×4; all three off. ${LOCAL}` }));
    } else {
      const idInput = el("input", {
        type: "text", placeholder: "L21.N4090", "aria-label": "Neuron to add", autocomplete: "off", spellcheck: "false",
      });
      const add = el("form", {
        class: "add-form", onsubmit: (ev) => {
          ev.preventDefault();
          const id = parseId(idInput.value);
          if (!id) { status("Write a neuron as L21.N4090", true); return; }
          addEdit(id.layer, id.neuron);
        },
      }, idInput, el("button", { type: "submit" }, "Add neuron"));
      kids.push(
        S.edits.length ? el("div", { class: "edit-list" }, S.edits.map(editRow))
          : el("p", { class: "empty", text: "No edits yet: add a neuron here, or open one and press “Switch off and "
            + "compare”." }),
        add,
        el("div", { class: "actions" },
          el("button", { type: "button", class: "primary", disabled: !S.edits.length, onclick: () => runEdits() },
            "Run with edits"),
          el("button", {
            type: "button", disabled: !S.edits.length,
            onclick: () => { S.edits = []; S.editResult = null; S.editError = null; renderEdit(); },
          }, "Clear")),
      );
    }
    if (S.editError) kids.push(problem(S.editError));
    if (S.editResult) kids.push(...editResultView(S.editResult));
    $("pane-edit").replaceChildren(...kids);
  }

  function editResultView(r) {
    const n = r.normal, e = r.edited;
    const side = (title, v, text, cls) => el("div", { class: cls },
      el("h4", { text: title }),
      el("div", { class: "big", text: pct(v.target.prob) }),
      el("div", { class: "sub" }, `p(${quoted(v.target.text)}) · rank ${v.target.rank} · top guess `,
        el("b", { text: quoted(v.guesses[0].text) })),
      el("div", { class: "cont" }, el("span", { class: "hint", text: "continues: " }), show(text)));
    const rows = [...n.layers].reverse().filter((row) => row.layer >= 0).map((row) => {
      const other = e.layers.find((x) => x.layer === row.layer);
      const a = row.attn + row.mlp, b = other.attn + other.mlp;
      const changed = Math.abs(a - b) > 0.05 || row.says !== other.says;
      return el("tr", { class: changed ? "changed" : null },
        el("td", { class: "id", text: `L${row.layer}` }),
        el("td", { class: "num", text: signed(a) }), el("td", { class: "num", text: signed(b) }),
        el("td", { class: "words", text: show(row.says) }), el("td", { class: "words", text: show(other.says) }));
    });
    return [
      el("h3", { text: r.edits.join(" + ") }),
      el("div", { class: "compare" }, side("Normal", n, r.normal_text, "normal"), side("Edited", e, r.edited_text, "edited")),
      el("p", { class: "gloss", text: "Each layer's push toward the target (attention + neurons), and what the model "
        + "would say after that layer. Highlighted rows changed." }),
      el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "layer" }), el("th", { class: "num", text: "normal" }),
        el("th", { class: "num", text: "edited" }), el("th", { text: "says" }), el("th", { text: "edited says" }))),
      el("tbody", {}, rows)),
    ];
  }

  /* 7. Dose test */
  async function doseNow(layer, neuron) {
    setTab("dose");
    S.doseId = nid(layer, neuron);
    S.dose = null;
    S.doseError = null;
    renderDose();
    try {
      S.dose = await busy(`dose test for ${S.doseId} and 20 random neurons…`, () => S.src.dose(S, layer, neuron));
    } catch (error) {
      S.doseError = error;
    }
    renderDose();
    reveal();
  }

  function renderDose() {
    const kids = [
      el("div", { class: "pane-title", text: "Dose test: is it real?" }),
      el("p", { class: "sub", text: "Scale one neuron from ×0 (off) to ×8 at every position and watch the target's "
        + "probability. The grey band is 20 random neurons of the same layer, scaled the same way: a neuron matters "
        + "only if its line leaves the band." }),
    ];
    if (S.src.isStatic) {
      kids.push(el("div", { class: "picks", role: "group", "aria-label": "Saved dose tests" },
        S.src.doseOptions(S).map((id) => el("button", {
          type: "button", "aria-pressed": String(S.doseId === id),
          onclick: () => { const p = parseId(id); doseNow(p.layer, p.neuron); },
        }, id))));
    } else {
      const input = el("input", {
        type: "text", placeholder: "L21.N4090", "aria-label": "Neuron to dose", autocomplete: "off", spellcheck: "false",
        value: (S.dose && S.doseId) || (S.neuron ? nid(S.neuron.layer, S.neuron.neuron) : S.doseId || ""),
      });
      kids.push(el("form", {
        class: "find-form", onsubmit: (ev) => {
          ev.preventDefault();
          const id = parseId(input.value);
          if (!id) { status("Write a neuron as L21.N4090", true); return; }
          doseNow(id.layer, id.neuron);
        },
      }, input, el("button", { type: "submit", class: "primary" }, "Run dose test")));
    }
    if (S.doseError) kids.push(problem(S.doseError));
    if (S.dose && S.dose.id === S.doseId) kids.push(...doseView(S.dose));
    $("pane-dose").replaceChildren(...kids);
  }

  function doseView(d) {
    const W = 520, H = 250, L = 46, R = 12, T = 12, B = 34;
    const n = d.scales.length;
    const x = (i) => L + (i * (W - L - R)) / (n - 1);
    const top = Math.min(1, Math.max(0.02, ...d.curve, ...d.band_hi) * 1.12);
    const y = (p) => T + (1 - p / top) * (H - T - B);
    const chart = svg("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img",
      "aria-label": `Probability of ${quoted(d.target.text)} as ${d.id} is scaled from ×0 to ×${d.scales[n - 1]}`,
    });
    for (const f of [0, 0.5, 1]) {
      chart.append(svg("line", { x1: L, x2: W - R, y1: y(top * f), y2: y(top * f), class: "grid-l" }));
      const label = svg("text", { x: L - 6, y: y(top * f) + 4, "text-anchor": "end" });
      label.textContent = f === 0 ? "0%" : pct(top * f);
      chart.append(label);
    }
    d.scales.forEach((s, i) => {
      const label = svg("text", { x: x(i), y: H - B + 18, "text-anchor": "middle" });
      label.textContent = `×${s}`;
      chart.append(label);
    });
    const one = d.scales.indexOf(1);
    if (one >= 0) chart.append(svg("line", { x1: x(one), x2: x(one), y1: T, y2: H - B, class: "base" }));
    const upper = d.band_hi.map((p, i) => `${x(i)},${y(p)}`);
    const lower = d.band_lo.map((p, i) => `${x(i)},${y(p)}`).reverse();
    chart.append(svg("polygon", { points: [...upper, ...lower].join(" "), class: "band" }));
    for (const curve of d.random) {
      chart.append(svg("polyline", { points: curve.map((p, i) => `${x(i)},${y(p)}`).join(" "), class: "rnd" }));
    }
    chart.append(svg("polyline", { points: d.curve.map((p, i) => `${x(i)},${y(p)}`).join(" "), class: "curve" }));
    d.curve.forEach((p, i) => chart.append(svg("circle", { cx: x(i), cy: y(p), r: 3.5, class: "dot" })));
    chart.append(svg("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }));
    const outside = d.scales.map((s, i) => ({ s, p: d.curve[i], lo: d.band_lo[i], hi: d.band_hi[i] }))
      .filter((c) => c.p < c.lo - 1e-6 || c.p > c.hi + 1e-6);
    const last = n - 1, zero = d.scales.indexOf(0);
    const points = (p) => `${p >= 0 ? "+" : "−"}${Math.abs(100 * p).toFixed(1)} points`;
    const swing = (curve) => curve.reduce((m, p) => (Math.abs(p - d.base_prob) > Math.abs(m) ? p - d.base_prob : m), 0);
    const own = swing(d.curve);
    const control = Math.max(...d.random.map((c) => Math.abs(swing(c))));
    const verdict = outside.length
      ? `Largest change from normal: ${points(own)} for ${d.id}, at most ±${(100 * control).toFixed(1)} points for `
        + `the ${d.random.length} random neurons. It leaves the band at ${outside.map((c) => `×${c.s}`).join(", ")}.`
      : `Inside the random band at every dose: no sign this neuron matters for ${quoted(d.target.text)} here.`;
    return [
      el("h3", {}, `${d.id} → p(${quoted(d.target.text)})`),
      chart,
      el("div", { class: "legend2" },
        el("span", {}, el("i", { style: "background:var(--mlp)" }), d.id),
        el("span", {}, el("i", { style: "background:var(--band);height:8px;vertical-align:0" }),
          `${d.random.length} random neurons of layer ${d.layer} (min–max)`),
        el("span", {}, el("i", { style: "border-top:1px dashed var(--muted);height:0" }), "×1 = normal")),
      el("p", { class: "gloss" },
        zero >= 0 ? `Off (×0): ${pct(d.curve[zero])}. ` : "", `Normal: ${pct(d.base_prob)}. `,
        `×${d.scales[last]}: ${pct(d.curve[last])} (random: ${pct(d.band_lo[last])}–${pct(d.band_hi[last])}). `,
        el("b", { text: verdict })),
    ];
  }

  /* Loading and running */
  async function runAll() {
    $("run").disabled = true;
    try {
      S.run = await busy("running the model…", () => S.src.run(S));
    } catch (error) {
      $("run").disabled = S.src.isStatic;
      $("note").textContent = error.message;
      return;
    }
    $("run").disabled = S.src.isStatic;
    S.target = S.run.target.id;
    S.pos = S.src.isStatic ? null : S.run.pos;
    S.editResult = null; S.editError = null; S.editKey = null;
    S.dose = null; S.doseError = null; S.doseId = null;
    S.findWord = S.run.target.text.trim();
    renderWords();
    const layers = S.run.layers.filter((r) => r.layer >= 0);
    if (S.layer === null || S.layer >= layers.length) {
      S.layer = layers.reduce((best, r) => (r.mlp > best.mlp ? r : best), layers[0]).layer;
    }
    renderStack();
    renderFind();
    renderEdit();
    renderDose();
    $("pane-neuron").replaceChildren(el("p", { class: "empty", text: "Open a neuron from a layer's list or from Find." }));
    setTab("layer"); // a new run clears the other panes
    await selectLayer(S.layer);
  }

  async function loadPreset(id) {
    const preset = S.info.presets.find((p) => p.id === id) || S.info.presets[0];
    S.preset = preset.id;
    S.text = preset.text;
    S.target = null; S.pos = null; S.layer = null; S.neuron = null;
    if (!S.src.isStatic) S.edits = [];
    $("prompt").value = preset.text;
    $("note").textContent = preset.note || "";
    for (const b of $("presets").children) b.setAttribute("aria-pressed", String(b.dataset.id === preset.id));
    await runAll();
  }

  function wireKeys() {
    $("prompt-form").addEventListener("submit", (e) => {
      e.preventDefault();
      if (S.src.isStatic) return;
      const text = $("prompt").value;
      if (!text.trim()) { status("Type a few words first", true); return; }
      const preset = S.info.presets.find((p) => p.text === text);
      S.text = text; S.preset = preset ? preset.id : null; S.target = null; S.pos = null;
      for (const b of $("presets").children) b.setAttribute("aria-pressed", String(b.dataset.id === S.preset));
      $("note").textContent = preset ? preset.note : "";
      runAll();
    });
    $("target-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const word = $("target-input").value;
      if (S.src.isStatic || !word) return;
      S.target = word;
      runAll();
    });
    $("rows").addEventListener("keydown", (e) => {
      const rows = [...$("rows").children].filter((r) => Number(r.dataset.layer) >= 0);
      const index = rows.indexOf(document.activeElement);
      if (index < 0) return;
      let next = null;
      if (e.key === "ArrowUp") next = rows[Math.max(0, index - 1)];
      else if (e.key === "ArrowDown") next = rows[Math.min(rows.length - 1, index + 1)];
      else if (e.key === "Home") next = rows[0];
      else if (e.key === "End") next = rows[rows.length - 1];
      if (!next) return;
      e.preventDefault();
      next.focus();
      selectLayer(Number(next.dataset.layer));
    });
    $("tokens").addEventListener("keydown", (e) => {
      const tokens = [...$("tokens").children];
      const index = tokens.indexOf(document.activeElement);
      const step = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
      if (index < 0 || !step) return;
      e.preventDefault();
      const next = tokens[Math.min(tokens.length - 1, Math.max(0, index + step))];
      for (const t of tokens) t.tabIndex = t === next ? 0 : -1;
      next.focus();
    });
    $("tabs").addEventListener("keydown", (e) => {
      const step = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
      if (!step) return;
      e.preventDefault();
      setTab(TABS[(TABS.indexOf(S.tab) + step + TABS.length) % TABS.length], true);
    });
    for (const name of TABS) $(`tab-${name}`).addEventListener("click", () => setTab(name));
    document.addEventListener("keydown", (e) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") { e.preventDefault(); $("prompt").focus(); }
      else if (/^[1-5]$/.test(e.key)) setTab(TABS[Number(e.key) - 1], true);
    });
  }

  async function applyParams(params) {
    const layer = params.get("layer"), neuron = params.get("neuron");
    if (layer !== null && Number(layer) !== S.layer) await selectLayer(Number(layer));
    if (layer !== null && neuron !== null) await openNeuron(Number(layer), Number(neuron), false);
    const tab = params.get("tab");
    if (TABS.includes(tab)) setTab(tab);
    const edit = params.get("edit");
    if (edit !== null && S.src.isStatic) {
      const option = S.src.editOptions(S)[Number(edit)];
      if (option) { setTab("edit"); await runEdits(option.edits); }
    }
    const dose = params.get("dose") && parseId(params.get("dose"));
    if (dose) await doseNow(dose.layer, dose.neuron);
  }

  async function init() {
    const source = document.documentElement.dataset.source;
    try {
      if (source) {
        const response = await fetch(source);
        if (!response.ok) throw new Error(`could not load ${source} (${response.status})`);
        S.src = new StaticSource(await response.json());
      } else {
        S.src = new LiveSource();
      }
      S.info = await S.src.info();
    } catch (error) {
      $("model").textContent = "";
      $("note").textContent = `The explorer could not start: ${error.message}`;
      return;
    }
    document.body.classList.toggle("static", S.src.isStatic);
    $("model").textContent = `${S.info.model} · ${S.info.n_layers} layers × ${S.info.d_mlp.toLocaleString("en")} neurons`;
    $("mode").textContent = S.src.isStatic ? `saved presets · ${S.info.created}` : "live";
    if (S.src.isStatic) {
      $("prompt").readOnly = true;
      $("run").disabled = true;
      $("run").title = LOCAL;
      $("prompt").title = `Pick a preset. To type your own words: ${LOCAL}`;
    }
    wireKeys();
    setTab("layer");
    renderPresets();
    const params = new URLSearchParams(location.search);
    await loadPreset(params.get("preset") || S.info.presets[0].id);
    await applyParams(params);
  }

  init();
})();
