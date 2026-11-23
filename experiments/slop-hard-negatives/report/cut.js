// Cut explorer: every run's trade-off curve (humans flagged against AI caught, traced as the flag cut moves), with a
// dot where the chosen cut sits on each curve. Drawn from data.js (counts only).
(function () {
  const D = window.HN;
  const root = document.getElementById("cut-explorer");
  if (!D || !root) return;
  const ARMS = { a: "(a) hardest pages", b: "(b) hardest + AI pairs", c: "(c) random pages" };
  const HUMAN = { web: ["old web pages (20,000)", 1], enron_h: ["Enron emails (1,992)", 2], epoch_h: ["Epoch AI human passages (495)", 4], magazine: ["magazine articles (150)", 20] };
  const AI = { fresh_new: "2026 pages, new topics (600)", fresh_old: "2026 pages, old topics (600)", epoch_imit: "Epoch AI imitations (297)", editlens_gen: "EditLens AI-written (2,046)", editlens_edit: "EditLens AI-edited (2,031)" };
  const YMIN = { fresh_new: 80, fresh_old: 80, epoch_imit: 50, editlens_gen: 90, editlens_edit: 30 };
  const fixedIdx = D.cuts.indexOf(D.fixed);
  const st = { arm: "a", cut: fixedIdx, human: "web", ai: "fresh_new" };
  const opts = (o, sel) => Object.entries(o).map(([k, v]) => `<option value="${k}"${k === sel ? " selected" : ""}>${Array.isArray(v) ? v[0] : v}</option>`).join("");
  root.innerHTML = `
    <div class="cx-controls">
      <label>Highlight <select data-k="arm">${opts(ARMS, st.arm)}</select></label>
      <label>Humans <select data-k="human">${opts(HUMAN, st.human)}</select></label>
      <label>AI texts <select data-k="ai">${opts(AI, st.ai)}</select></label>
      <label class="cx-slider">Flag cut <input type="range" min="0" max="${D.cuts.length - 1}" step="1" value="${st.cut}"> <output></output></label>
      <button type="button" class="cx-reset">Back to the fixed cut</button>
    </div>
    <svg role="img"></svg>
    <p class="cx-read small" aria-live="polite"></p>`;
  const svg = root.querySelector("svg");
  const out = root.querySelector("output");
  const read = root.querySelector(".cx-read");
  const slider = root.querySelector("input[type=range]");
  const runsOf = (a) => [1, 2, 3, 4].map((i) => a + i).filter((r) => D.runs[r]);
  const pct = (r, k, i) => (100 * D.runs[r][k][i]) / D.n[k];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
  function draw() {
    const W = Math.max(300, Math.min(680, root.clientWidth || 640));
    const H = Math.round(W * (W < 480 ? 0.8 : 0.5));
    const fs = 12, L = 46, R = 16, T = 26, B = 40;
    const ymin = YMIN[st.ai];
    const xmax = HUMAN[st.human][1];
    const x = (v) => L + (Math.min(v, xmax) / xmax) * (W - L - R);
    const y = (v) => T + (1 - (Math.max(v, ymin) - ymin) / (100 - ymin)) * (H - T - B);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const ink = css("--ink"), soft = css("--ink-soft"), rule = css("--rule"), acc = css("--accent"), ctl = css("--control");
    const col = (r) => (r === "r0" ? ink : r[0] === "c" ? ctl : acc);
    let s = `<defs><clipPath id="cx-clip"><rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/></clipPath></defs>`;
    for (let t = 0; t <= 4; t++) {
      const v = (xmax * t) / 4, gx = x(v);
      s += `<line x1="${gx}" x2="${gx}" y1="${T}" y2="${H - B}" stroke="${rule}"/><text x="${gx}" y="${H - B + 15}" text-anchor="${t === 4 ? "end" : t === 0 ? "start" : "middle"}" font-size="${fs}" fill="${soft}">${+v.toFixed(2)}%</text>`;
    }
    [ymin, (ymin + 100) / 2, 100].forEach((v) => (s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${rule}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="${fs}" fill="${soft}">${v}%</text>`));
    s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle" font-size="${fs}" fill="${soft}">humans flagged →</text>`;
    s += `<text x="${L - 40}" y="${T - 10}" font-size="${fs}" fill="${soft}">AI texts caught ↑ (axis starts at ${ymin}%)</text>`;
    const all = ["r0", ...runsOf("a"), ...runsOf("b"), ...runsOf("c")];
    const hl = new Set(["r0", ...runsOf(st.arm)]);
    const order = all.filter((r) => !hl.has(r)).concat(all.filter((r) => hl.has(r)));
    s += `<g clip-path="url(#cx-clip)">`;
    for (const r of order) {
      const d = D.cuts.map((_, i) => `${i ? "L" : "M"}${x(pct(r, st.human, i)).toFixed(1)},${y(pct(r, st.ai, i)).toFixed(1)}`).join("");
      const on = hl.has(r);
      s += `<path d="${d}" fill="none" stroke="${col(r)}" stroke-width="${on ? 2.2 : 1}" opacity="${on ? 1 : 0.22}"${r[0] === "b" ? ' stroke-dasharray="5 3"' : ""}/>`;
    }
    s += `</g>`;
    const pts = ["r0", ...runsOf(st.arm)];
    pts.forEach((r) => {
      const px = x(pct(r, st.human, st.cut)), py = y(pct(r, st.ai, st.cut));
      const lab = r === "r0" ? "round 0" : "round " + r.slice(1);
      s += `<circle cx="${px}" cy="${py}" r="5" fill="${col(r)}" stroke="${css("--canvas")}" stroke-width="1.5"><title>${lab}: ${pct(r, st.human, st.cut).toFixed(2)}% of humans flagged, ${pct(r, st.ai, st.cut).toFixed(1)}% of AI texts caught</title></circle>`;
    });
    svg.innerHTML = s;
    const c = D.cuts[st.cut];
    out.textContent = c.toFixed(4) + (st.cut === fixedIdx ? " (fixed)" : "");
    slider.setAttribute("aria-valuetext", `cut ${c.toFixed(4)}`);
    const last = runsOf(st.arm).slice(-1)[0];
    const h = HUMAN[st.human][0], a = AI[st.ai];
    svg.setAttribute("aria-label", `Trade-off curves of humans flagged against AI texts caught for every run; ${h} against ${a}.`);
    read.textContent = `At a cut of ${c.toFixed(4)}: round 0 flags ${pct("r0", st.human, st.cut).toFixed(2)}% of ${h} and catches ${pct("r0", st.ai, st.cut).toFixed(1)}% of ${a}; ${ARMS[st.arm]}, round ${last.slice(1)}, flags ${pct(last, st.human, st.cut).toFixed(2)}% and catches ${pct(last, st.ai, st.cut).toFixed(1)}%. A curve further up and to the left is a better detector at every cut.`;
  }
  root.querySelectorAll("select").forEach((el) => el.addEventListener("change", () => { st[el.dataset.k] = el.value; draw(); }));
  slider.addEventListener("input", () => { st.cut = +slider.value; draw(); });
  root.querySelector(".cx-reset").addEventListener("click", () => { st.cut = fixedIdx; slider.value = fixedIdx; draw(); });
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(root);
  draw();
})();
