// Cap explorer: every pilot-2 recipe's validation score against the share of human texts it flags at its own cutoff,
// with a cap on that share (a multiple of R0's) that the design did not have. Recipes over the cap are scored like a
// recall failure (min(score, 0) - 1). Drawn from data.js (scores and rates only, no text).
(function () {
  const D = window.RSI;
  const root = document.getElementById("cap-explorer");
  if (!D || !root) return;
  const STEPS = [0, 50, 20, 10, 5, 3, 2, 1.5, 1];
  const st = { i: 0 };
  root.innerHTML = `
    <div class="cx-controls">
      <label class="cx-slider">Cap on humans flagged at the recipe's own cutoff <input type="range" min="0" max="${STEPS.length - 1}" step="1" value="0"> <output></output></label>
    </div>
    <svg role="img" aria-label="Scatter of validation score against the share of human texts flagged at each recipe's own cutoff, with a movable cap"></svg>
    <p class="cx-read small" aria-live="polite"></p>`;
  const svg = root.querySelector("svg");
  const out = root.querySelector("output");
  const read = root.querySelector(".cx-read");
  const slider = root.querySelector("input[type=range]");
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
  const pct = (v) => (100 * v < 1 ? (100 * v).toFixed(2) : (100 * v).toFixed(1)) + "%";
  const scored = (r, cap) => (cap && r.own > cap ? Math.min(r.score, 0) - 1 : r.score);
  function best(arm, cap) {
    let b = null;
    for (const r of D.recipes) {
      if (r.arm !== arm) continue;
      const s = scored(r, cap);
      if (!b || s > b.s) b = { node: r.node, s };
    }
    return b;
  }
  function draw() {
    const k = STEPS[st.i];
    const cap = k ? k * D.r0_own : 0;
    out.textContent = k ? `${k} × R0 (${pct(cap)})` : "none (as run)";
    const W = Math.max(300, Math.min(680, root.clientWidth || 640));
    const H = Math.round(W * (W < 480 ? 0.75 : 0.48));
    const fs = 12, L = 52, R = 14, T = 14, B = 40;
    const lx = (v) => L + ((Math.max(v, -1) + 1) / 2.05) * (W - L - R);
    const ly = (v) => {
      const a = Math.log10(0.0003), b = Math.log10(1.5);
      return T + (1 - (Math.log10(Math.max(v, 0.0003)) - a) / (b - a)) * (H - T - B);
    };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const ink = css("--ink"), soft = css("--ink-soft"), rule = css("--rule"), acc = css("--accent"), ctl = css("--control");
    let g = "";
    for (const t of [0.001, 0.01, 0.1, 1]) {
      g += `<line x1="${L}" x2="${W - R}" y1="${ly(t)}" y2="${ly(t)}" stroke="${rule}" stroke-width="0.6"/>`;
      g += `<text x="${L - 6}" y="${ly(t) + 4}" font-size="${fs - 1}" text-anchor="end" fill="${soft}">${100 * t}%</text>`;
    }
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      g += `<text x="${lx(t)}" y="${H - B + 16}" font-size="${fs - 1}" text-anchor="middle" fill="${soft}">${t > 0 ? "+" : ""}${Math.round(100 * t)}%</text>`;
    }
    g += `<text x="${(L + W - R) / 2}" y="${H - 6}" font-size="${fs}" text-anchor="middle" fill="${ink}">${W < 480 ? "validation score" : "validation score (gain over R0 at matched recall)"}</text>`;
    g += `<line x1="${L}" x2="${W - R}" y1="${ly(D.r0_own)}" y2="${ly(D.r0_own)}" stroke="${soft}" stroke-dasharray="4 3"/>`;
    g += `<text x="${L + 4}" y="${ly(D.r0_own) + 13}" font-size="${fs - 1}" fill="${soft}">R0 at its own cutoff</text>`;
    if (cap) {
      g += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${ly(cap) - T}" fill="${rule}" opacity="0.35"/>`;
      g += `<line x1="${L}" x2="${W - R}" y1="${ly(cap)}" y2="${ly(cap)}" stroke="${ink}" stroke-width="1.2"/>`;
      g += `<text x="${W - R - 4}" y="${ly(cap) - 5}" font-size="${fs - 1}" text-anchor="end" fill="${ink}">cap</text>`;
    }
    let n = 0;
    for (const r of D.recipes) {
      const over = cap && r.own > cap;
      if (over) n++;
      const c = r.arm === "l3" ? acc : ctl;
      const x = lx(r.score), y = ly(r.own);
      g += over
        ? `<circle cx="${x}" cy="${y}" r="4.5" fill="none" stroke="${c}" stroke-width="1.4"><title>${r.node} (${r.imp}): ${pct(r.own)} flagged, penalised</title></circle>`
        : `<circle cx="${x}" cy="${y}" r="4.5" fill="${c}"><title>${r.node} (${r.imp}): score ${(100 * r.score).toFixed(1)}%, ${pct(r.own)} flagged</title></circle>`;
    }
    svg.innerHTML = g;
    const b2 = best("l2", cap), b3 = best("l3", cap);
    read.textContent = cap
      ? `Cap at ${k} × R0's ${pct(D.r0_own)}: ${n} of ${D.recipes.length} recipes penalised (hollow). Best L2 recipe ${b2.node} (${(100 * b2.s).toFixed(1)}%), best L3 recipe ${b3.node} (${(100 * b3.s).toFixed(1)}%).`
      : `As run, with no cap: ${D.recipes.length} recipes; the three at the top flag 6% to 74% of human texts at their own cutoffs and still score +41% to +92%. Best L2 recipe ${b2.node}, best L3 recipe ${b3.node}.`;
  }
  slider.addEventListener("input", () => { st.i = +slider.value; draw(); });
  window.addEventListener("resize", draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  draw();
})();
