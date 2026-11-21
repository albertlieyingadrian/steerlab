// Interactive figures for the culture-steering note. Every number is read from data.json (written by analyse.py);
// nothing is computed from a model in the browser. Each figure starts on a fixed default view and replaces a static
// PNG fallback (drawn by make_figures.py from the same data) once data.json has loaded.
const NS = "http://www.w3.org/2000/svg";
const ARMS = [
  ["none", "No context"], ["prompt", "Prompt"], ["caa", "Country direction"], ["persona", "Persona vector"],
  ["softprompt", "Soft prompt"],
];
const MODELS = [["qwen3-8b", "Qwen3-8B"], ["llama-3.1-8b", "Llama-3.1-8B-Instruct"]];
const STEER = new Set(["prompt", "caa", "persona", "softprompt", "centred", "prompt+caa"]);

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
const f3 = (x) => x.toFixed(3);
const short = (s, n = 40) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

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
    return b;
  });
  g.append(...btns);
  return g;
}

function mount(id, build) {
  const fig = document.getElementById(id);
  if (!fig) return;
  const live = fig.querySelector(".live"), fb = fig.querySelector(".fallback");
  build(live);
  live.hidden = false;
  if (fb) fb.hidden = true;
}

// Figure 1: one question, one country: people's split against the model under one arm.
function explorer(live, D) {
  const st = { model: "qwen3-8b", country: "Indonesia", q: 0, arm: "softprompt" };
  const controls = el("div", { class: "controls" });
  const qtext = el("p", { class: "qtext" });
  const svg = sv("svg", { role: "img" });
  const stats = el("p", { class: "stats", "aria-live": "polite" });
  const legend = el("p", { class: "legend" });
  legend.innerHTML = '<span class="key"><i style="background:var(--ink-soft)"></i>people (World Values Survey)</span>' +
    '<span class="key"><i style="background:var(--accent)"></i>model, chosen arm</span>';
  const draw = () => {
    const ex = D.examples[st.model][st.q];
    const h = ex.human[st.country], m = ex.arms[st.arm][st.country];
    const mass = ex.mass[st.arm][st.country];
    qtext.textContent = ex.question.replace("[your country]", "your country");
    const k = ex.options.length, rowH = 30, W = 640, L = 190, R = 40, H = k * rowH + 20;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("aria-label", `${st.country}: people ${h.map(pct).join(", ")}; model ${m.map(pct).join(", ")}`);
    svg.replaceChildren();
    const x = (v) => L + (W - L - R) * v;
    for (const t of [0, 0.5, 1]) {
      svg.append(sv("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: H - 14 }));
      svg.append(sv("text", { class: "tick", x: x(t), y: H - 2, "text-anchor": "middle" }, pct(t)));
    }
    ex.options.forEach((o, i) => {
      const y = 6 + i * rowH;
      svg.append(sv("text", { class: "lab", x: L - 8, y: y + 15, "text-anchor": "end" }, short(o, 28)));
      svg.append(sv("rect", { class: "people", x: L, y, width: Math.max(0, x(h[i]) - L), height: 11 }));
      svg.append(sv("rect", { class: "model", x: L, y: y + 12, width: Math.max(0, x(m[i]) - L), height: 11 }));
      svg.append(sv("text", { class: "val", x: x(h[i]) + 4, y: y + 10 }, pct(h[i])));
      svg.append(sv("text", { class: "val m", x: x(m[i]) + 4, y: y + 22 }, pct(m[i])));
    });
    const armName = ARMS.find((a) => a[0] === st.arm)[1];
    stats.textContent = `JS distance, ${armName.toLowerCase()} vs ${st.country}: ${f2(jsd(m, h))}` +
      (mass < 0.2 ? ` · only ${pct(mass)} of the first-token probability named an option, so this cell is left out of the averages` : "");
  };
  const qs = D.examples[st.model].map((e, i) => [String(i), short(e.question.split("\n").pop(), 60)]);
  controls.append(
    segmented("Model", MODELS, st.model, (v) => { st.model = v; draw(); }),
    picker("Country", D.countries.map((c) => [c, c]), st.country, (v) => { st.country = v; draw(); }),
    picker("Question", qs, "0", (v) => { st.q = +v; draw(); }),
    segmented("Arm", ARMS, st.arm, (v) => { st.arm = v; draw(); }),
  );
  live.append(controls, qtext, svg, legend, stats);
  draw();
}

// Figure 2: distance per arm for one country (or the mean over countries), with 95% intervals over questions.
function arms(live, D) {
  const st = { model: "qwen3-8b", country: "all" };
  const rows = [["none", "No context"], ["prompt", "Prompt “I live in X.”"], ["caa", "Country direction"],
    ["centred", "Country-only direction"], ["persona", "Persona vector"], ["prompt+caa", "Prompt + direction"],
    ["softprompt", "Trained soft prompt"], ["random", "Random direction"]];
  const controls = el("div", { class: "controls" });
  const svg = sv("svg", { role: "img" });
  const stats = el("p", { class: "stats", "aria-live": "polite" });
  const val = (s, arm, c) => {
    const src = arm === "random" ? s.random.caa : s.arms[arm];
    if (c !== "all") return src[c].jsd;
    const ms = D.countries.map((x) => src[x].jsd);
    const avg = (k) => ms.reduce((a, b) => a + b[k], 0) / ms.length;
    return { mean: avg("mean"), lo: avg("lo"), hi: avg("hi") };
  };
  const draw = () => {
    const s = D.survey[st.model];
    const W = 640, L = 200, R = 50, rowH = 28, H = rows.length * rowH + 40;
    const x = (v) => L + (W - L - R) * (v / 0.7);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();
    const base = (k) => (st.country === "all"
      ? D.countries.reduce((a, c) => a + s.baselines[c][k].mean, 0) / 8 : s.baselines[st.country][k].mean);
    for (const t of [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) {
      svg.append(sv("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: H - 26 }));
      svg.append(sv("text", { class: "tick", x: x(t), y: H - 12, "text-anchor": "middle" }, t.toFixed(1)));
    }
    for (const [k, t, ty] of [["average_of_8", "average of 8 countries", 8], ["uniform", "uniform guess", H - 30]]) {
      const b = base(k);
      svg.append(sv("line", { class: "ref", x1: x(b), x2: x(b), y1: 0, y2: H - 26 }));
      svg.append(sv("text", { class: "reft", x: x(b) + 3, y: ty }, t));
    }
    const out = [];
    rows.forEach(([arm, label], i) => {
      const v = val(s, arm, st.country), y = 14 + i * rowH;
      const hot = STEER.has(arm);
      svg.append(sv("text", { class: "lab", x: L - 10, y: y + 4, "text-anchor": "end" }, label));
      svg.append(sv("line", { class: `ci ${hot ? "str" : "def"}`, x1: x(v.lo), x2: x(v.hi), y1: y, y2: y }));
      svg.append(sv("circle", { class: `pt ${hot ? "str" : "def"}`, cx: x(v.mean), cy: y, r: 5 }));
      svg.append(sv("text", { class: `val${hot ? " m" : ""}`, x: x(v.hi) + 6, y: y + 4 }, f2(v.mean)));
      out.push(`${label} ${f2(v.mean)}`);
    });
    svg.setAttribute("aria-label", `Distance to ${st.country === "all" ? "the eight countries" : st.country}: ${out.join("; ")}`);
    const g = s.grid;
    stats.textContent = st.country === "all"
      ? `Own-country advantage (how much closer an arm set for country c gets to c than the same arm set for the other seven): prompt ${f3(g.prompt.own_advantage.mean)}, direction ${f3(g.caa.own_advantage.mean)}, persona ${f3(g.persona.own_advantage.mean)}, soft prompt ${f3(g.softprompt.own_advantage.mean)}.`
      : `${st.country}: people's own spread ${f2(s.baselines[st.country].human_entropy)}; the model's spread with no context ${f2(s.arms.none[st.country].entropy)}, with the soft prompt ${f2(s.arms.softprompt[st.country].entropy)}.`;
  };
  controls.append(
    segmented("Model", MODELS, st.model, (v) => { st.model = v; draw(); }),
    picker("Country", [["all", "Mean over the 8 countries"], ...D.countries.map((c) => [c, c])], st.country,
      (v) => { st.country = v; draw(); }),
  );
  live.append(controls, svg, stats);
  draw();
}

fetch("data.json")
  .then((r) => r.json())
  .then((D) => {
    mount("fig-explorer", (live) => explorer(live, D));
    mount("fig-arms", (live) => arms(live, D));
  })
  .catch(() => {});
