// Interactive figures for the pluralism explainer. Every number is read from data.json (written by analyse.py);
// nothing is computed from a model in the browser. Each figure starts on a fixed default view and replaces a static
// PNG fallback (drawn by make_figures.py from the same data) once data.json has loaded.
const NS = "http://www.w3.org/2000/svg";
const FEATURED = "goqa-2474";
const DEFAULT_COUNTRY = "Indonesia";

const el = (tag, attrs = {}, text) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
};
const sv = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
};
const pct = (x) => `${Math.round(x * 100)}%`;
const f2 = (x) => x.toFixed(2);
const short = (s, n = 34) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

function jsd(p, q) {
  const sp = p.reduce((a, b) => a + b, 0), sq = q.reduce((a, b) => a + b, 0);
  let d = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i] / sp, b = q[i] / sq, m = (a + b) / 2;
    if (a > 0) d += 0.5 * a * Math.log2(a / m);
    if (b > 0) d += 0.5 * b * Math.log2(b / m);
  }
  return Math.sqrt(Math.max(0, d));
}
function entropy(p) {
  const s = p.reduce((a, b) => a + b, 0);
  if (p.length < 2 || s === 0) return 0;
  let h = 0;
  for (const x of p) if (x > 0) h -= (x / s) * Math.log2(x / s);
  return h / Math.log2(p.length);
}

// A labelled <select>, keyboard-operable by default.
function picker(label, options, value, onChange) {
  const wrap = el("label", { class: "pk" });
  wrap.append(el("span", { class: "pk-l" }, label));
  const s = el("select");
  for (const [v, t] of options) {
    const o = el("option", { value: v }, t);
    if (v === value) o.selected = true;
    s.append(o);
  }
  s.addEventListener("change", () => onChange(s.value));
  wrap.append(s);
  return wrap;
}
// A segmented control: buttons with aria-pressed.
function segmented(label, options, value, onChange) {
  const g = el("div", { class: "seg", role: "group", "aria-label": label });
  g.append(el("span", { class: "pk-l" }, label));
  const btns = options.map(([v, t]) => {
    const b = el("button", { type: "button", "aria-pressed": String(v === value) }, t);
    b.addEventListener("click", () => {
      btns.forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      onChange(v);
    });
    g.append(b);
    return b;
  });
  return g;
}
// Draw at the figure's real width (so text stays at its real size on a phone), and redraw when that width changes.
const SHORT = { "Qwen3.5-9B-Base": "Qwen3.5 Base", "SEA-LION v5 9B": "SEA-LION v5", "Qwen3-8B-Base": "Qwen3 Base",
  "Claude Haiku 5.5": "Claude Haiku", "Gemini 3.1 Flash-Lite": "Gemini Lite" };
const mlabel = (label, W) => (W < 520 ? SHORT[label] ?? label : label);
const width = (node) => Math.max(320, Math.min(640, Math.round(node.clientWidth || 640)));
function onResize(node, draw) {
  let last = width(node);
  new ResizeObserver(() => { const w = width(node); if (w !== last) { last = w; draw(); } }).observe(node);
}
function activate(fig) {
  fig.querySelector(".fallback")?.setAttribute("hidden", "");
  const live = fig.querySelector(".live");
  live.removeAttribute("hidden");
  return live;
}
const modelLabel = (data, key) => data.models.find((m) => m.key === key)?.label ?? key;
const lastLine = (q) => {
  const lines = q.split("\n").map((s) => s.trim()).filter(Boolean);
  return lines.length > 1 ? `${lines[lines.length - 1]} (${short(lines[0], 40)})` : lines[0];
};
const qOptions = (data) => {
  const feat = data.questions.find((q) => q.id === FEATURED);
  const rest = data.questions.filter((q) => q.id !== FEATURED);
  return [feat, ...rest].map((q) => [q.id, short(lastLine(q.q), 70)]);
};

// Figure 1: one question, one country, optionally one model.
function figSplit(fig, data) {
  const live = activate(fig);
  const st = { q: FEATURED, c: DEFAULT_COUNTRY, m: "none", cond: "default" };
  const controls = el("div", { class: "controls" });
  const out = el("div", { class: "fig-body" });
  const stats = el("p", { class: "stats", "aria-live": "polite" });
  const qtext = el("p", { class: "qtext" });
  const condWrap = el("span", { class: "seg-wrap" });
  const drawCond = () => {
    condWrap.replaceChildren(segmented("Prompt", [["default", "No country"], ["steered", `“I live in ${st.c}.”`]],
      st.cond, (v) => { st.cond = v; draw(); }));
  };
  controls.append(
    picker("Question", qOptions(data), st.q, (v) => { st.q = v; draw(); }),
    picker("Country", data.countries.map((c) => [c, c]), st.c, (v) => { st.c = v; drawCond(); draw(); }),
    picker("Model", [["none", "People only"], ...data.models.map((m) => [m.key, m.label])], st.m, (v) => { st.m = v; draw(); }),
    condWrap,
  );
  drawCond();
  live.append(controls, qtext, out, stats);

  function draw() {
    const q = data.questions.find((x) => x.id === st.q);
    const human = q.human[st.c];
    const cond = st.cond === "default" ? "default" : st.c;
    const raw = st.m === "none" ? null : q.m[st.m]?.[cond];
    const cnt = raw ? raw.slice(0, -1) : null;
    const nValid = cnt ? cnt.reduce((a, b) => a + b, 0) : 0;
    const model = cnt && nValid > 0 ? cnt.map((x) => x / nValid) : null;
    qtext.textContent = q.q.replace(/\n+/g, " · ");
    const W = width(live), rowH = model ? 34 : 24, left = W < 520 ? 120 : 190, top = 6;
    const H = top + q.options.length * rowH + 22;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "bars" });
    const x = (v) => left + v * (W - left - 60);
    for (const t of [0.25, 0.5, 0.75, 1]) {
      svg.append(sv("line", { x1: x(t), x2: x(t), y1: 0, y2: H - 18, class: "grid" }));
      svg.append(sv("text", { x: x(t), y: H - 4, class: "tick", "text-anchor": "middle" }, pct(t)));
    }
    svg.append(sv("line", { x1: x(0), x2: x(0), y1: 0, y2: H - 18, class: "axis" }));
    q.options.forEach((o, i) => {
      const y = top + i * rowH;
      svg.append(sv("text", { x: left - 8, y: y + 12, class: "lab", "text-anchor": "end" }, short(o, W < 520 ? 17 : 28)));
      svg.append(sv("rect", { x: x(0), y, width: Math.max(0.5, x(human[i]) - x(0)), height: 14, class: "people" }));
      svg.append(sv("text", { x: x(human[i]) + 4, y: y + 11, class: "val" }, pct(human[i])));
      if (model) {
        svg.append(sv("rect", { x: x(0), y: y + 16, width: Math.max(0.5, x(model[i]) - x(0)), height: 10, class: "model" }));
        svg.append(sv("text", { x: x(model[i]) + 4, y: y + 25, class: "val m" }, pct(model[i])));
      }
    });
    const label = model
      ? `${st.c}'s survey split (grey) against ${modelLabel(data, st.m)}'s ${nValid} valid answers of 50 (orange).`
      : `${st.c}'s survey split on this question.`;
    svg.setAttribute("aria-label", label + " " + q.options.map((o, i) => `${o}: ${pct(human[i])}${model ? `, model ${pct(model[i])}` : ""}`).join("; "));
    out.replaceChildren(svg);
    let avg = "";
    if (q.id === FEATURED) {
      const mean = human.reduce((a, p, i) => a + p * i, 0);
      const k = Math.round(mean);
      avg = `Average position: “${q.options[k]}”, an answer held by ${pct(human[k])} of ${st.c}. `;
    }
    if (st.m !== "none" && !model) {
      stats.textContent = `${avg}${modelLabel(data, st.m)} gave no valid answer here (50 of 50 replies named no option).`;
    } else if (model) {
      stats.textContent = `${avg}JS distance ${f2(jsd(cnt, human))} (0 = same split, 1 = no overlap) · ` +
        `spread (entropy) model ${f2(entropy(cnt))}, people ${f2(entropy(human))} · ` +
        `the model's top answer is held by ${pct(human[cnt.indexOf(Math.max(...cnt))])} of people · ` +
        `${nValid}/50 answers valid.`;
    } else {
      stats.textContent = `${avg}Spread (entropy, 0 = everyone agrees, 1 = evenly split): ${f2(entropy(human))}. ` +
        `Pick a model to compare.`;
    }
  }
  draw();
  onResize(live, draw);
}

// Figure 2: several countries on one question, as 100% stacked bars.
function figCountries(fig, data) {
  const live = activate(fig);
  const st = { q: FEATURED, view: "differ" };
  const VIEWS = { differ: ["United States", "Philippines", "Singapore", "Malaysia"], uniform: ["United States", "Singapore"], all: data.countries };
  const controls = el("div", { class: "controls" });
  const out = el("div", { class: "fig-body" });
  const legend = el("p", { class: "legend" });
  controls.append(
    picker("Question", qOptions(data), st.q, (v) => { st.q = v; draw(); }),
    segmented("Show", [["differ", "Groups differ"], ["uniform", "Groups aren't uniform"], ["all", "All eight"]], st.view, (v) => { st.view = v; draw(); }),
  );
  live.append(controls, out, legend);
  function draw() {
    const q = data.questions.find((x) => x.id === st.q);
    const cs = VIEWS[st.view];
    const W = width(live), k = q.options.length, left = W < 520 ? 90 : 110, rowH = 30;
    const H = cs.length * rowH + 8;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "bars" });
    const x = (v) => left + v * (W - left - 10);
    const step = (i) => Math.round((i / Math.max(1, k - 1)) * 6);
    cs.forEach((c, r) => {
      const y = r * rowH + 4;
      svg.append(sv("text", { x: left - 8, y: y + 15, class: "lab", "text-anchor": "end" }, c));
      let acc = 0;
      q.human[c].forEach((p, i) => {
        svg.append(sv("rect", { x: x(acc), y, width: Math.max(0, x(acc + p) - x(acc) - 1), height: 22, class: `o${step(i)}` }));
        if (p >= 0.08) svg.append(sv("text", { x: (x(acc) + x(acc + p)) / 2, y: y + 15, class: `in ${step(i) > 3 ? "dk" : ""}`, "text-anchor": "middle" }, pct(p)));
        acc += p;
      });
    });
    svg.setAttribute("aria-label", cs.map((c) => `${c}: ${q.options.map((o, i) => `${o} ${pct(q.human[c][i])}`).join(", ")}`).join(". "));
    out.replaceChildren(svg);
    legend.replaceChildren(...q.options.map((o, i) => {
      const s = el("span", { class: "key" });
      s.append(el("i", { class: `o${step(i)}` }), document.createTextNode(short(o, 40)));
      return s;
    }));
  }
  draw();
  onResize(live, draw);
}

// Figure 3: each model's distance (or spread, or top-answer share) for one country, with and without the country.
function figDistance(fig, data) {
  const live = activate(fig);
  const S = data.summary;
  const st = { c: DEFAULT_COUNTRY, metric: "jsd" };
  const controls = el("div", { class: "controls" });
  const out = el("div", { class: "fig-body" });
  const note = el("p", { class: "stats", "aria-live": "polite" });
  controls.append(
    picker("Country", data.countries.map((c) => [c, c]), st.c, (v) => { st.c = v; draw(); }),
    segmented("Measure", [["jsd", "Distance to people"], ["ent", "Spread of answers"], ["top", "People holding its top answer"]], st.metric, (v) => { st.metric = v; draw(); }),
  );
  live.append(controls, out, note);
  function val(m, cond) {
    const P = S.per_model[m];
    if (st.metric === "jsd") return P.jsd[cond][st.c];
    if (st.metric === "ent") return P.entropy[cond];
    return P.top_share[cond][st.c];
  }
  function draw() {
    const H0 = S.human[st.c];
    const refs = st.metric === "jsd"
      ? [["Floor: 50 answers drawn from " + st.c, H0.floor.mean], ["Average of 8 countries", H0.avg_human_sampled.mean], [`Nearest country (${H0.nearest_country})`, H0.nearest_country_sampled.mean], ["Uniform guess", H0.uniform_sampled.mean]]
      : st.metric === "ent" ? [["People in " + st.c, H0.entropy.mean]] : [["People's own top answer", H0.top_share.mean]];
    const max = st.metric === "jsd" ? 0.8 : 1;
    const W = width(live), left = W < 520 ? 100 : 170, rowH = 30, top = 10 + refs.length * 14;
    const H = top + data.models.length * rowH + 22;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "dots" });
    const x = (v) => left + (v / max) * (W - left - 20);
    for (let i = 0; i <= 4; i++) {
      const t = (max / 4) * i;
      svg.append(sv("line", { x1: x(t), x2: x(t), y1: top - 4, y2: H - 18, class: "grid" }));
      svg.append(sv("text", { x: x(t), y: H - 4, class: "tick", "text-anchor": "middle" }, t.toFixed(st.metric === "jsd" ? 1 : 2)));
    }
    const codes = "abcd";
    refs.forEach(([name, v], i) => {
      svg.append(sv("line", { x1: x(v), x2: x(v), y1: 12 + i * 14, y2: H - 18, class: "ref" }));
      svg.append(sv("text", { x: x(v), y: 10 + i * 14, class: "reft", "text-anchor": "middle" }, refs.length > 1 ? codes[i] : f2(v)));
    });
    const refText = refs.length > 1 ? "Dashed lines: " + refs.map(([name, v], i) => `${codes[i]} = ${name} ${f2(v)}`).join("; ") + ". " : `Dashed line: ${refs[0][0]} ${f2(refs[0][1])}. `;
    const parts = [];
    data.models.forEach((m, r) => {
      const y = top + r * rowH + 12;
      svg.append(sv("text", { x: left - 10, y: y + 4, class: "lab", "text-anchor": "end" }, mlabel(m.label, W)));
      const d = val(m.key, "default"), s = val(m.key, st.c);
      for (const [v, cls, dy] of [[d, "def", -5], [s, "str", 5]]) {
        if (!v || !v.n) continue;
        svg.append(sv("line", { x1: x(v.lo), x2: x(v.hi), y1: y + dy, y2: y + dy, class: `ci ${cls}` }));
        svg.append(sv("circle", { cx: x(v.mean), cy: y + dy, r: 4.5, class: `pt ${cls}` }));
      }
      if (d?.n && s?.n) parts.push(`${m.label}: ${f2(d.mean)} without, ${f2(s.mean)} with`);
    });
    svg.setAttribute("aria-label", `${st.c}, ${st.metric === "jsd" ? "distance to people" : st.metric === "ent" ? "spread" : "share holding the top answer"}, no country then with the country. ` + parts.join("; "));
    out.replaceChildren(svg);
    note.textContent = refText + (st.metric === "jsd"
      ? "Hollow grey: no country in the prompt. Orange: “I live in " + st.c + ".” Lines: 95% intervals over questions. Lower is closer; nothing can beat the floor."
      : st.metric === "ent"
        ? "Spread of a model's answers (0 = always the same answer, 1 = evenly split), against the spread of " + st.c + "'s own survey answers. A model can sit near people while naming only their most popular answer."
        : "The share of " + st.c + "'s people who give the model's most common answer, against the share giving the country's own most common answer.");
  }
  draw();
  onResize(live, draw);
}

// Figure 4: distance heatmap, models by countries.
function figHeat(fig, data) {
  const live = activate(fig);
  const S = data.summary;
  const st = { cond: "default" };
  const controls = el("div", { class: "controls" });
  const out = el("div", { class: "fig-body" });
  controls.append(segmented("Prompt", [["default", "No country"], ["own", "“I live in <that country>.”"]], st.cond, (v) => { st.cond = v; draw(); }));
  live.append(controls, out);
  const abbr = { "United States": "US", "South Korea": "KR", Philippines: "PH", Singapore: "SG", Indonesia: "ID", Malaysia: "MY", Taiwan: "TW", Brazil: "BR" };
  function draw() {
    const W0 = width(live), cs = data.countries, left = W0 < 520 ? 92 : 150, top = 30, ch = 26;
    const cw = Math.min(56, (W0 - left - 4) / cs.length);
    const rows = [...data.models.map((m) => [mlabel(m.label, W0), (c) => S.per_model[m.key].jsd[st.cond === "own" ? c : "default"][c].mean]),
      ["Floor", (c) => S.human[c].floor.mean]];
    const W = left + cs.length * cw + 4, H = top + rows.length * ch + 4;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "heat" });
    cs.forEach((c, j) => svg.append(sv("text", { x: left + j * cw + cw / 2, y: top - 10, class: "lab", "text-anchor": "middle" }, abbr[c] ?? c)));
    const lo = 0.1, hi = 0.7;
    rows.forEach(([name, f], r) => {
      const y = top + r * ch;
      svg.append(sv("text", { x: left - 8, y: y + 17, class: "lab", "text-anchor": "end" }, name));
      cs.forEach((c, j) => {
        const v = f(c);
        const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
        svg.append(sv("rect", { x: left + j * cw + 1, y: y + 1, width: cw - 2, height: ch - 2, style: `fill:color-mix(in srgb, var(--accent) ${Math.round(6 + t * 80)}%, var(--canvas))` }));
        svg.append(sv("text", { x: left + j * cw + cw / 2, y: y + 17, class: `in ${t > 0.7 ? "dk" : ""}`, "text-anchor": "middle" }, f2(v)));
      });
    });
    svg.setAttribute("aria-label", "Mean JS distance to each country. " + rows.map(([n, f]) => `${n}: ${cs.map((c) => `${c} ${f2(f(c))}`).join(", ")}`).join(". "));
    out.replaceChildren(svg);
  }
  draw();
  onResize(live, draw);
}

async function mount() {
  let data;
  try {
    data = await (await fetch("data.json")).json();
  } catch {
    return; // the static figures stay
  }
  const figs = { "fig-split": figSplit, "fig-countries": figCountries, "fig-distance": figDistance, "fig-heat": figHeat };
  for (const [id, fn] of Object.entries(figs)) {
    const f = document.getElementById(id);
    if (f) fn(f, data);
  }
}
mount();
