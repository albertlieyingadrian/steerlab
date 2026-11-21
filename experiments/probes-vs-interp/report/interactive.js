// The leaderboard explorer: pick a test set, see every detector's AUROC with its 95% interval.
// Data: window.PVI_DATA, written by make_figures.py from results/leaderboard.json. Static fallback: figures/leaderboard.png.
(() => {
  const fig = document.getElementById("pvi-explorer");
  const data = window.PVI_DATA;
  if (!fig || !data) return;
  const SETS = [
    ["test", "Held-out authors"], ["heldgen", "Held-out generator"], ["blog", "Email → blog"],
    ["hum_prompt", "Prompted humanizer"], ["hum_dipper", "DIPPER"], ["frontier", "Frontier drafts"],
    ["test_fmt", "Format stripped"], ["test_len", "First 60 words"],
  ];
  const FAMILY = { interpretability: "var(--accent)", "small probe": "var(--ink)", baseline: "var(--ink-faint)",
    "trained detector": "var(--ink-faint)", "LLM judge": "var(--ink-faint)" };
  const controls = fig.querySelector(".pvi-controls");
  const svg = fig.querySelector("svg");
  const live = fig.querySelector(".pvi-live");
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  let current = "test";

  function draw() {
    const rows = data.filter((r) => r[current]).sort((a, b) => b[current][0] - a[current][0]);
    const W = 660, left = 222, right = 60, rowH = 24, top = 26, H = top + rows.length * rowH + 30;
    const x = (v) => left + ((Math.max(v, 0.5) - 0.5) / 0.5) * (W - left - right);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("aria-label", `AUROC of every detector on the ${SETS.find((s) => s[0] === current)[1]} set, with 95% intervals`);
    svg.replaceChildren();
    for (const t of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
      svg.append(el("line", { x1: x(t), x2: x(t), y1: top - 6, y2: H - 26, class: "grid" }));
      svg.append(el("text", { x: x(t), y: H - 10, "text-anchor": "middle", class: "tick" }, t.toFixed(1)));
    }
    svg.append(el("text", { x: left, y: 12, class: "axis-label" }, "AUROC (0.5 = chance, 1 = perfect)"));
    rows.forEach((r, i) => {
      const [m, lo, hi] = r[current];
      const y = top + i * rowH + rowH / 2;
      const c = FAMILY[r.family] || "var(--ink)";
      svg.append(el("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "row-label" }, r.name));
      svg.append(el("line", { x1: x(0.5), x2: x(1.0), y1: y, y2: y, class: "grid" }));
      svg.append(el("line", { x1: x(lo), x2: x(hi), y1: y, y2: y, stroke: c, "stroke-width": 1.6 }));
      svg.append(el("circle", { cx: x(m), cy: y, r: 3.6, fill: c }));
      svg.append(el("text", { x: x(hi) + 6, y: y + 4, class: "tick" }, m.toFixed(3)));
    });
    const label = SETS.find((s) => s[0] === current)[1];
    live.textContent = `${label}: best ${rows[0].name} ${rows[0][current][0].toFixed(3)}, ` +
      `lowest ${rows[rows.length - 1].name} ${rows[rows.length - 1][current][0].toFixed(3)}.`;
  }

  for (const [key, label] of SETS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "pvi-btn";
    b.textContent = label;
    b.setAttribute("aria-pressed", key === current ? "true" : "false");
    b.addEventListener("click", () => {
      current = key;
      for (const o of controls.querySelectorAll("button")) o.setAttribute("aria-pressed", o === b ? "true" : "false");
      draw();
    });
    controls.append(b);
  }
  fig.classList.add("is-live");
  draw();
})();
