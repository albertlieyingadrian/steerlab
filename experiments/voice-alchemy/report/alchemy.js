// The precomputed alchemy table. Reads window.VA_TABLE (table.js, written by build_table.py from the held-out run).
// No backend and no live generation: every number and reply shown here was measured and judged before publishing.
(function () {
  "use strict";
  const TABLES = window.VA_TABLES || {};
  const MODELS = [["qwen", "Qwen3-8B"], ["gemma", "Gemma-2-2B-it"]].filter(([k]) => TABLES[k]);
  const root = document.getElementById("alchemy");
  if (!MODELS.length || !root) return;
  let T = TABLES[MODELS[0][0]];

  const TYPES = [
    ["direction", "Direction", "The average difference in activations between replies in a voice and plain replies (baseline)."],
    ["sae", "SAE feature", "One Gemma Scope feature's decoder direction."],
    ["neuron", "Single neuron", "What one MLP neuron writes into the residual stream."],
    ["community", "Community", "A group of MLP neurons that fire together, each weighted by how much more it fires in the voice."],
    ["circuit", "Circuit", "The transcoder features an attribution graph credits most specifically for the voice."],
    ["prompt", "Prompt", "The style instruction itself, added to the question (baseline)."],
  ];
  const LABEL_TEXT = {
    additive: "Additive: both voices keep at least half their single effect",
    dominates: "Dominates: one voice keeps at least half its effect, the other does not",
    interferes: "Interferes: neither voice keeps half its effect",
    emergent: "Emergent: a third voice rises by 2 or more points beyond the sum",
    breaks: "Breaks: more than 20% of replies loop, or coherence falls below 4",
  };
  const state = { model: MODELS[0][0], type: "direction", strength: "op", picks: [], prompt: Object.keys(T.prompts)[0], open: null };
  const fmt = (x) => (x === null || x === undefined ? "–" : (x > 0 ? "+" : "") + x.toFixed(1));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const tl = (t) => T.trait_labels[T.traits.indexOf(t)];

  function cardsOf(type) {
    return T.cards.filter((c) => c.type === type).sort((a, b) => T.traits.indexOf(a.trait) - T.traits.indexOf(b.trait));
  }
  function sKey(card) {
    if (card.type === "prompt") return "op";
    return state.strength === "op" ? String(card.s_op) : state.strength;
  }
  function findPair(a, b) {
    return T.pairs.find((p) => p.type === state.type &&
      ((p.traits[0] === a && p.traits[1] === b) || (p.traits[0] === b && p.traits[1] === a)));
  }

  function bars(d, pred, target) {
    const scale = 8; // +/-8 judge points spans half the track
    return '<div class="va-bars" role="list">' + T.traits.map((t, i) => {
      const v = d[i] === undefined ? null : d[i];
      const w = v === null ? 0 : Math.min(Math.abs(v) / scale, 1) * 50;
      const left = v === null || v >= 0 ? 50 : 50 - w;
      let mark = "";
      if (pred && pred[i] !== null) {
        const x = 50 + Math.max(-1, Math.min(1, pred[i] / scale)) * 50;
        mark = `<span class="va-pred" style="left:${x}%" title="sum of the two singles: ${fmt(pred[i])}"></span>`;
      }
      const own = target && target.indexOf(t) >= 0;
      return `<div class="va-row${own ? " own" : ""}" role="listitem"><span class="va-t">${esc(T.trait_labels[i])}</span>` +
        `<span class="va-track"><span class="va-zero"></span><span class="va-bar${v !== null && v < 0 ? " neg" : ""}" ` +
        `style="left:${left}%;width:${w}%"></span>${mark}</span>` +
        `<span class="va-v">${v === null ? "–" : fmt(v)}</span></div>`;
    }).join("") + "</div>";
  }

  function replyBlock(ex) {
    const r = ex && ex[state.prompt];
    if (!r) return '<p class="va-muted">No reply recorded for this prompt.</p>';
    return `<blockquote class="va-reply${r.loops ? " loops" : ""}">${esc(r.text) || "<em>(empty reply)</em>"}</blockquote>` +
      `<p class="va-muted">${r.loops ? "Loops: " : ""}${Math.round(r.repeat * 100)}% of its 4-word phrases repeat.</p>`;
  }

  function render() {
    T = TABLES[state.model];
    const models = MODELS.length < 2 ? "" : '<div class="va-tabs va-models" role="tablist" aria-label="Model">' +
      MODELS.map(([k, name]) => `<button type="button" class="va-tab" role="tab" aria-selected="${state.model === k}" data-model="${k}">${name}</button>`).join("") + "</div>";
    const tabs = TYPES.map(([k, name]) =>
      `<button type="button" class="va-tab" role="tab" aria-selected="${state.type === k}" data-type="${k}">${name}</button>`).join("");
    const desc = TYPES.find((x) => x[0] === state.type)[2];
    const ts = T.types[state.type] || {};
    const strengthSel = state.type === "prompt" ? "" :
      '<label class="va-pk"><span>Push strength</span><select data-k="strength">' +
      `<option value="op"${state.strength === "op" ? " selected" : ""}>each card's chosen strength</option>` +
      T.strengths.map((s) => `<option value="${s}"${state.strength === String(s) ? " selected" : ""}>s = ${s}</option>`).join("") +
      "</select></label>";
    const promptSel = '<label class="va-pk"><span>Replies to</span><select data-k="prompt">' +
      Object.entries(T.prompts).map(([pi, q]) =>
        `<option value="${pi}"${state.prompt === pi ? " selected" : ""}>${esc(q)}</option>`).join("") +
      "</select></label>";
    const cards = cardsOf(state.type).map((c) => {
      const s = sKey(c);
      const p = c.by_s[s] || {};
      const picked = state.picks.indexOf(c.trait);
      const ctrl = c.control ? c.control[s] : undefined;
      const open = state.open === c.key;
      return `<article class="va-card${picked >= 0 ? " picked" : ""}">` +
        `<header><h4>${esc(tl(c.trait))}</h4><span class="va-s">${c.type === "prompt" ? "instruction" : "s = " + s}</span></header>` +
        `<p class="va-what">${esc(c.what || "")}${c.link ? ` · <a href="${c.link}">Neuronpedia</a>` : ""}</p>` +
        (c.np_label ? `<p class="va-np">Its Neuronpedia label: <q>${esc(c.np_label)}</q></p>` : "") +
        bars(p.d || [], null, [c.trait]) +
        `<p class="va-stats"><span>coherence ${p.coh ?? "–"}</span><span>honesty ${p.hon ?? "–"}</span>` +
        `<span class="${p.loop > 0.2 ? "warn" : ""}">loops ${p.loop === undefined ? "–" : Math.round(p.loop * 100) + "%"}</span>` +
        (ctrl !== undefined && ctrl !== null ? `<span>random control ${fmt(ctrl)}</span>` : "") + "</p>" +
        `<div class="va-actions"><button type="button" class="va-btn" data-pick="${c.trait}" aria-pressed="${picked >= 0}">` +
        `${picked >= 0 ? "Picked " + (picked ? "B" : "A") : "Pick to mix"}</button>` +
        `<button type="button" class="va-btn ghost" data-open="${c.key}" aria-expanded="${open}">${open ? "Hide reply" : "Read a reply"}</button></div>` +
        (open ? replyBlock(p.ex) : "") + "</article>";
    }).join("");

    let mix = '<p class="va-muted">Pick two cards to see what pushing both at once did.</p>';
    if (state.picks.length === 2) {
      const [a, b] = state.picks;
      const pr = findPair(a, b);
      if (!pr) {
        mix = `<p class="va-muted">${esc(tl(a))} + ${esc(tl(b))} was not run for this unit type` +
          (state.type === "circuit" ? " (circuits were mixed for 4 pairs only)." : ".") + "</p>";
      } else {
        const flip = pr.traits[0] !== a;
        const da = flip ? pr.d_b : pr.d_a, db = flip ? pr.d_a : pr.d_b;
        const pred = da.map((x, i) => (x === null || db[i] === null ? null : x + db[i]));
        mix = `<div class="va-mixhead"><span class="va-chip ${pr.label}">${pr.label}</span>` +
          `<span>additivity error E = ${pr.E.toFixed(2)}</span><span>coherence ${pr.coh ?? "–"}</span>` +
          `<span class="${pr.loop > 0.2 ? "warn" : ""}">loops ${Math.round(pr.loop * 100)}%</span>` +
          (pr.eligible ? "" : '<span class="warn">a single unit moved its own voice by under 1 point: not scored</span>') + "</div>" +
          `<p class="va-muted">${esc(LABEL_TEXT[pr.label] || "")}${pr.emergent ? " (" + esc(tl(pr.emergent)) + ")" : ""}. ` +
          "Bars: the measured pair; ticks: the sum of the two singles.</p>" + bars(pr.d_ab, pred, [a, b]) + replyBlock(pr.ex);
      }
    }
    const summary = ts.n_units ? `<p class="va-muted">${ts.units_moving} of ${ts.n_units} units move their own voice by at least 1 point; ` +
      `${ts.n_eligible} of ${ts.n_pairs} pairs can be scored; median additivity error ${ts.median_E ?? "–"}.</p>` : "";
    root.innerHTML =
      models + `<div class="va-tabs" role="tablist" aria-label="Unit type">${tabs}</div>` +
      `<p class="va-desc">${esc(desc)}</p>${summary}` +
      `<div class="va-controls">${strengthSel}${promptSel}</div>` +
      '<p class="va-muted va-legend">Bars: change in each voice against the plain model, in judge points (half a track = 8 points; ' +
      "pale bars are drops). The voice a unit was chosen for is in bold. “–”: that voice was not judged at this strength.</p>" +
      `<div class="va-grid">${cards}</div>` +
      `<section class="va-mix" aria-live="polite"><h4>Mix: ${state.picks.map((t) => esc(tl(t))).join(" + ") || "none picked"}</h4>${mix}</section>`;
  }

  root.addEventListener("click", (e) => {
    const t = e.target.closest("button");
    if (!t) return;
    const secondPick = t.dataset.pick && state.picks.length === 1 && state.picks[0] !== t.dataset.pick;
    if (t.dataset.model) { state.model = t.dataset.model; state.picks = []; state.open = null; }
    else if (t.dataset.type) { state.type = t.dataset.type; state.picks = []; state.open = null; }
    else if (t.dataset.pick) {
      const k = t.dataset.pick, i = state.picks.indexOf(k);
      if (i >= 0) state.picks.splice(i, 1);
      else { state.picks.push(k); if (state.picks.length > 2) state.picks.shift(); }
    } else if (t.dataset.open) { state.open = state.open === t.dataset.open ? null : t.dataset.open; }
    render();
    if (secondPick) {
      const mix = root.querySelector(".va-mix");
      const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (mix && mix.getBoundingClientRect().top > window.innerHeight * 0.6) {
        mix.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
      }
    }
  });
  root.addEventListener("change", (e) => {
    const k = e.target.dataset.k;
    if (k) { state[k] = e.target.value; render(); }
  });
  render();
})();
