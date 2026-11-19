// Interactive figures for the Episode 2 note. Every number comes from the JSON embedded in index.html
// (written by make_figures.py from results/summary.json); nothing is computed from scratch here.
// Each figure has a static PNG fallback that stays visible if this script fails.
(function () {
  const raw = document.getElementById("page-data");
  if (!raw) return;
  let D;
  try { D = JSON.parse(raw.textContent); } catch (e) { return; }
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || null;
  const C = () => ({ accent: css("--accent") || "#c9561c", control: css("--control") || "#4a86e8",
    ink: css("--ink") || "#1d272a", soft: css("--ink-soft") || "#586468", faint: css("--ink-faint") || "#8a9497",
    rule: css("--rule") || "#e3e6e6" });
  const short = (m) => (D.short && D.short[m]) || m;
  const FAM = { "gemma3-12b": "Gemma-3-12B", "qwen3-8b": "Qwen3-8B", "qwen3.5-9b": "Qwen3.5-9B" };
  const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
  const pct = (x) => (x >= 0 ? "+" : "−") + Math.abs(100 * x).toFixed(0) + "%";

  function el(tag, attrs = {}, text) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function svg(w, h, label) {
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", role: "img", "aria-label": label });
    s.style.maxWidth = w + "px";
    s.style.height = "auto";
    return s;
  }
  function picker(container, label, options, current, onChange) {
    const row = document.createElement("div");
    row.className = "explorer-tabs";
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", label);
    const lab = document.createElement("span");
    lab.className = "picker-label";
    lab.textContent = label;
    row.append(lab);
    const buttons = options.map(([value, text]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "explorer-tab";
      b.textContent = text;
      b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => {
        buttons.forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        onChange(value);
      });
      row.append(b);
      return b;
    });
    container.append(row);
  }
  function mount(id, build) {
    const box = document.getElementById(id);
    if (!box) return;
    try {
      const live = document.createElement("div");
      live.className = "live";
      box.prepend(live);
      build(live);
      const fb = box.querySelector(".fallback");
      if (fb) fb.hidden = true;
    } catch (e) { console.error(id, e); }
  }
  function axisX(s, x0, x1, y, scale, ticks, fmtT, col) {
    s.append(el("line", { x1: x0, x2: x1, y1: y, y2: y, stroke: col.faint }));
    ticks.forEach((t) => {
      s.append(el("line", { x1: scale(t), x2: scale(t), y1: y, y2: y + 4, stroke: col.faint }));
      s.append(el("text", { x: scale(t), y: y + 16, "text-anchor": "middle", "font-size": 11, fill: col.soft }, fmtT(t)));
    });
  }

  // Figure A: per-task loss change, instruct vs base, per family and format
  mount("fig-a", (live) => {
    let fam = "gemma3-12b", form = "chat";
    const out = document.createElement("div");
    const note = document.createElement("p");
    note.className = "explorer-stats";
    picker(live, "Family", Object.keys(D.A.families).map((f) => [f, FAM[f]]), fam, (v) => { fam = v; draw(); });
    picker(live, "Instruct model's prompt", [["chat", "chat (authors' choice)"], ["colon", "plain colon (short tasks)"],
      ["st", "Spectrum-tuned (Gemma only)"]], form, (v) => { form = v; draw(); });
    live.append(out, note);
    function draw() {
      const col = C();
      const key = form === "chat" ? "rel_change" : form === "colon" ? "rel_change_colon" : "rel_change_ST";
      const rows = D.A.per_task.filter((r) => r.family === fam && r[key] !== undefined && r[key] !== null)
        .sort((a, b) => b[key] - a[key]);
      if (!rows.length) {
        out.replaceChildren();
        note.textContent = "No runs for this combination (the Spectrum-tuned model exists for Gemma only).";
        return;
      }
      const W = 640, rowH = 22, H = 40 + rows.length * rowH;
      const s = svg(W, H, `Loss change per task, ${FAM[fam]}`);
      const lo = Math.min(-0.2, ...rows.map((r) => (form === "chat" ? r.lo : r[key])));
      const hi = Math.max(0.2, ...rows.map((r) => (form === "chat" ? r.hi : r[key])));
      const x = (v) => 200 + (W - 220) * (v - lo) / (hi - lo);
      s.append(el("line", { x1: x(0), x2: x(0), y1: 6, y2: H - 28, stroke: col.ink }));
      rows.forEach((r, i) => {
        const y = 14 + i * rowH;
        s.append(el("text", { x: 192, y: y + 4, "text-anchor": "end", "font-size": 11, fill: col.ink },
          r.task.replace(/_/g, " ")));
        const v = r[key];
        const color = form === "st" ? col.control : col.accent;
        if (form === "chat") s.append(el("line", { x1: x(r.lo), x2: x(r.hi), y1: y, y2: y, stroke: color, "stroke-width": 2, opacity: 0.5 }));
        const dot = el("circle", { cx: x(v), cy: y, r: 4.5, fill: color });
        dot.append(el("title", {}, `${r.task}: ${pct(v)}` + (form === "chat" ? ` [${pct(r.lo)}, ${pct(r.hi)}], n = ${r.n}` : "")));
        s.append(dot);
      });
      const step = (hi - lo) > 1.5 ? 0.5 : 0.25;
      const ticks = [];
      for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(t);
      axisX(s, 200, W - 20, H - 26, x, ticks, (t) => pct(t), col);
      out.replaceChildren(s);
      const f = D.A.families[fam];
      note.textContent = form === "chat"
        ? `${FAM[fam]}: mean ${pct(f.rel_change)} [${pct(f.lo)}, ${pct(f.hi)}]; instruct worse on ${f.tasks_IT_worse} of ${f.n_tasks} tasks, better on ${f.tasks_IT_better}.`
        : form === "colon" ? "The same instruct model given the base model's plain prompt, short tasks only (no interval)."
          : "Spectrum-tuned Gemma against the base model (reference row; the paper's fix).";
    }
    draw();
  });

  // Figure B: coverage (yield) per task, base vs instruct, per family and prompt setting
  mount("fig-b", (live) => {
    let fam = "gemma3-12b", setting = "description";
    const out = document.createElement("div");
    const note = document.createElement("p");
    note.className = "explorer-stats";
    picker(live, "Family", Object.keys(FAM).map((f) => [f, FAM[f]]), fam, (v) => { fam = v; draw(); });
    picker(live, "Prompt gives", [["description", "a description"], ["examples", "3 examples"], ["both", "both"]],
      setting, (v) => { setting = v; draw(); });
    live.append(out, note);
    function draw() {
      const col = C();
      const rows = D.B.per_task.filter((r) => r.family === fam && r.setting === setting)
        .sort((a, b) => (b.yield_PT - b.yield_IT) - (a.yield_PT - a.yield_IT));
      const W = 640, rowH = 20, H = 48 + rows.length * rowH;
      const s = svg(W, H, `Distinct valid outputs per 100, ${FAM[fam]}`);
      const x = (v) => 200 + (W - 220) * v / 100;
      rows.forEach((r, i) => {
        const y = 14 + i * rowH;
        s.append(el("text", { x: 192, y: y + 4, "text-anchor": "end", "font-size": 11, fill: col.ink }, r.task.replace(/_/g, " ")));
        s.append(el("line", { x1: x(r.yield_PT), x2: x(r.yield_IT), y1: y, y2: y, stroke: col.rule, "stroke-width": 3 }));
        const p = el("circle", { cx: x(r.yield_PT), cy: y, r: 4.5, fill: col.faint });
        p.append(el("title", {}, `base: ${fmt(r.yield_PT, 1)} distinct valid (validity ${fmt(r.validity_PT)})`));
        const q = el("circle", { cx: x(r.yield_IT), cy: y, r: 4.5, fill: col.accent });
        q.append(el("title", {}, `instruct: ${fmt(r.yield_IT, 1)} distinct valid (validity ${fmt(r.validity_IT)})`));
        s.append(p, q);
      });
      axisX(s, 200, W - 20, H - 32, x, [0, 25, 50, 75, 100], (t) => String(t), col);
      s.append(el("text", { x: 200, y: H - 2, "font-size": 11, fill: col.soft }, "grey = base, orange = instruct"));
      out.replaceChildren(s);
      const avg = (k) => rows.reduce((a, r) => a + r[k], 0) / rows.length;
      note.textContent = `${FAM[fam]}, ${rows.length} tasks: base yields ${fmt(avg("yield_PT"), 1)} distinct valid outputs per 100 (validity ${fmt(avg("validity_PT"))}), instruct ${fmt(avg("yield_IT"), 1)} (validity ${fmt(avg("validity_IT"))}).`;
    }
    draw();
  });

  // Figure C: per-prompt repetition for each model; pick a model to highlight
  mount("fig-c", (live) => {
    const models = Object.keys(D.C.per_prompt);
    let sel = models[0];
    const out = document.createElement("div");
    const note = document.createElement("p");
    note.className = "explorer-stats";
    picker(live, "Highlight", models.map((m) => [m, short(m)]), sel, (v) => { sel = v; draw(); });
    live.append(out, note);
    function draw() {
      const col = C();
      const W = 640, rowH = 28, H = 50 + models.length * rowH;
      const s = svg(W, H, "Repetition per prompt for each model");
      const x = (v) => 150 + (W - 170) * Math.max(0, v);
      const fl = Object.values(D.C.floor);
      const floor = fl.reduce((a, b) => a + b, 0) / fl.length;
      s.append(el("line", { x1: x(0.8), x2: x(0.8), y1: 4, y2: H - 30, stroke: col.ink, "stroke-dasharray": "4 3" }));
      s.append(el("text", { x: x(0.8) + 3, y: 12, "font-size": 10, fill: col.ink }, "paper's 0.8"));
      s.append(el("line", { x1: x(floor), x2: x(floor), y1: 4, y2: H - 30, stroke: col.control, "stroke-dasharray": "2 3" }));
      s.append(el("text", { x: x(floor) + 3, y: 12, "font-size": 10, fill: col.soft }, "different prompts"));
      models.forEach((m, i) => {
        const y = 26 + i * rowH;
        const on = m === sel;
        const base = D.C.per_model[m].verdict.startsWith("n/a");
        s.append(el("text", { x: 142, y: y + 4, "text-anchor": "end", "font-size": 11, fill: on ? col.ink : col.soft,
          "font-weight": on ? 600 : 400 }, short(m)));
        D.C.per_prompt[m].forEach((v, k) => {
          if (v === null) return;  // a prompt with no non-empty reply pair
          const jitter = ((k * 37) % 17) / 17 - 0.5;
          s.append(el("circle", { cx: x(v), cy: y + jitter * 14, r: on ? 2.6 : 2, fill: base ? col.faint : col.accent,
            opacity: on ? 0.85 : 0.3 }));
        });
        const med = D.C.per_model[m].median;
        s.append(el("line", { x1: x(med), x2: x(med), y1: y - 10, y2: y + 10, stroke: col.ink, "stroke-width": 2 }));
      });
      axisX(s, 150, W - 20, H - 28, x, [0, 0.2, 0.4, 0.6, 0.8, 1], (t) => t.toFixed(1), col);
      out.replaceChildren(s);
      const d = D.C.per_model[sel];
      note.textContent = `${short(sel)}: median ${fmt(d.median)}; ${Math.round(100 * d["share_above_0.8"])}% of prompts above 0.8; mean ${fmt(d.mean)} [${fmt(d.mean_lo)}, ${fmt(d.mean_hi)}]; ${Math.round(100 * d.truncated_rate)}% of replies hit the 512-token cap.`;
    }
    draw();
  });

  // Figure C2: read the answers. Pick a prompt and a model; three of its ten samples.
  mount("fig-c2", (live) => {
    const ex = D.examples;
    let p = 0;
    let m = Object.keys(ex[0].samples)[0];
    const out = document.createElement("div");
    out.className = "samples";
    picker(live, "Prompt", ex.map((e, i) => [i, `${i + 1}. ${e.query.slice(0, 34)}${e.query.length > 34 ? "…" : ""}`]), p, (v) => { p = v; draw(); });
    picker(live, "Model", Object.keys(ex[0].samples).map((k) => [k, short(k)]), m, (v) => { m = v; draw(); });
    live.append(out);
    function draw() {
      const e = ex[p];
      out.replaceChildren();
      const q = document.createElement("p");
      q.className = "sample-q";
      q.textContent = `“${e.query}”: mean repetition across the six instruct models ${fmt(e.mean_repetition)}`;
      out.append(q);
      (e.samples[m] || []).forEach((t, i) => {
        const b = document.createElement("blockquote");
        b.className = "sample";
        b.textContent = `${i + 1}. ${t}${t.length >= 500 ? " …" : ""}`;
        out.append(b);
      });
    }
    draw();
  });

  // Figure D: cross-model similarity grid; diagonal = within-model repetition
  mount("fig-d", (live) => {
    const pairs = D.D.pairs;
    const models = [...new Set(pairs.flatMap((p) => [p.model_a, p.model_b]))];
    const note = document.createElement("p");
    note.className = "explorer-stats";
    note.textContent = "Select a cell (click, or Tab then Enter) to read the pair.";
    const out = document.createElement("div");
    live.append(out, note);
    const col = C();
    const n = models.length, cell = 62, L = 130, T = 10, W = L + n * cell + 90, H = T + n * cell + 120;
    const s = svg(W, H, "Cross-model similarity grid");
    const val = (a, b) => {
      if (a === b) return { v: D.C.per_model[a].mean, ratio: 1 };
      const p = pairs.find((q) => (q.model_a === a && q.model_b === b) || (q.model_a === b && q.model_b === a));
      return { v: p.cross, ratio: p.ratio };
    };
    models.forEach((a, i) => {
      s.append(el("text", { x: L - 6, y: T + i * cell + cell / 2 + 4, "text-anchor": "end", "font-size": 11, fill: col.ink }, short(a)));
      s.append(el("text", { x: 0, y: 0, "font-size": 11, fill: col.ink, transform: `translate(${L + i * cell + cell / 2},${T + n * cell + 8}) rotate(40)` }, short(a)));
      models.forEach((b, j) => {
        const { v, ratio } = val(a, b);
        const t = Math.max(0, Math.min(1, (v - 0.5) / 0.5));
        const g = el("g", { tabindex: 0, role: "button", "aria-label": `${short(a)} and ${short(b)}: ${fmt(v)}` });
        g.append(el("rect", { x: L + j * cell, y: T + i * cell, width: cell - 2, height: cell - 2, fill: col.accent, "fill-opacity": 0.12 + 0.88 * t, rx: 2 }));
        g.append(el("text", { x: L + j * cell + cell / 2 - 1, y: T + i * cell + cell / 2 + 4, "text-anchor": "middle", "font-size": 12, fill: t > 0.6 ? "#fff" : col.ink }, fmt(v)));
        const show = () => {
          note.textContent = a === b
            ? `${short(a)} against itself: mean similarity of its own 10 answers ${fmt(v)}.`
            : `${short(a)} vs ${short(b)}: answers to the same prompt have similarity ${fmt(v)}, ${fmt(100 * ratio, 0)}% of the two models' own repetition.`;
        };
        g.addEventListener("click", show);
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(); } });
        s.append(g);
      });
    });
    out.append(s);
  });

  // Figure E: judge calibration by disagreement third; raw r or r / human ceiling
  mount("fig-e", (live) => {
    let mode = "raw";
    const out = document.createElement("div");
    const note = document.createElement("p");
    note.className = "explorer-stats";
    picker(live, "Show", [["raw", "r with the raters' mean"], ["norm", "r ÷ human ceiling"]], mode, (v) => { mode = v; draw(); });
    live.append(out, note);
    const bins = ["low", "mid", "high"];
    function draw() {
      const col = C();
      const W = 640, H = 300, x0 = 60, x1 = 470;
      const s = svg(W, H, "Judge calibration by disagreement");
      const judges = Object.keys(D.E.per_judge);
      const get = (j, b) => mode === "raw" ? D.E.per_judge[j].r[b] : D.E.per_judge[j].r_over_ceiling[b];
      const vals = judges.flatMap((j) => bins.map((b) => get(j, b))).filter((v) => v !== null && !Number.isNaN(v));
      if (mode === "raw") bins.forEach((b) => vals.push(D.E.ceiling[b], D.E.one_rater_r[b]));
      const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals);
      const y = (v) => 20 + (H - 70) * (hi - v) / (hi - lo);
      const x = (i) => x0 + (x1 - x0) * i / 2;
      [0, 0.25, 0.5, 0.75, 1].filter((t) => t >= lo && t <= hi).forEach((t) => {
        s.append(el("line", { x1: x0, x2: x1, y1: y(t), y2: y(t), stroke: t === 0 ? col.ink : col.rule }));
        s.append(el("text", { x: x0 - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 10, fill: col.soft }, t.toFixed(2)));
      });
      ["least disagreed", "middle", "most disagreed"].forEach((t, i) => s.append(el("text", { x: x(i), y: H - 30, "text-anchor": "middle", "font-size": 11, fill: col.soft }, t)));
      const labels = [];
      const line = (pts, color, dash, label) => {
        const ok = pts.filter((p) => p[1] !== null && !Number.isNaN(p[1]));
        if (!ok.length) return;
        s.append(el("polyline", { points: ok.map(([i, v]) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash || "" }));
        ok.forEach(([i, v]) => { const c = el("circle", { cx: x(i), cy: y(v), r: 3.5, fill: color }); c.append(el("title", {}, `${label}: ${fmt(v)}`)); s.append(c); });
        labels.push({ y: y(ok[ok.length - 1][1]) + 4, color: color === col.control ? col.soft : color, label });
      };
      const placeLabels = () => {  // keep end labels at least 13 px apart
        labels.sort((a, b) => a.y - b.y);
        for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
        labels.forEach((l) => s.append(el("text", { x: x1 + 8, y: l.y, "font-size": 11, fill: l.color }, l.label)));
      };
      if (mode === "raw") {
        line(bins.map((b, i) => [i, D.E.ceiling[b]]), col.control, "5 3", "human ceiling");
        line(bins.map((b, i) => [i, D.E.one_rater_r[b]]), col.control, "2 3", "one rater");
      }
      judges.forEach((j) => line(bins.map((b, i) => [i, get(j, b)]), j === "length" ? col.faint : col.accent, j === "length" ? "3 3" : "", short(j)));
      placeLabels();
      out.replaceChildren(s);
      note.textContent = mode === "raw"
        ? `Human ceiling √ρ by third: ${bins.map((b) => fmt(D.E.ceiling[b])).join(", ")}. Items per third: ${D.E.bins.map((b) => b.n).join(", ")}.`
        : "Each judge's r divided by the human ceiling of its third: 1 means as good as the raters' own mean allows.";
    }
    draw();
  });
})();
