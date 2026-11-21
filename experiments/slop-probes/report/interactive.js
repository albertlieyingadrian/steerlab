// Threshold explorer for The Taste Stack Episode 3: pick a detector, flag the top k% of held-out pages by its score,
// and read how many AI-builder pages it catches and how many human pages it flags. Data: window.SLOP (data.js).
(function () {
  const D = window.SLOP;
  const box = document.getElementById("sx");
  if (!D || !box) return;
  const NS = "http://www.w3.org/2000/svg";
  const sel = document.getElementById("sx-arm");
  const k = document.getElementById("sx-k");
  const kout = document.getElementById("sx-kout");
  const strip = document.getElementById("sx-strip");
  const roc = document.getElementById("sx-roc");
  const y = D.y;
  const nPos = y.reduce((a, b) => a + b, 0);
  const nNeg = y.length - nPos;
  const pct = (v) => (100 * v).toFixed(1) + "%";

  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const a in attrs) e.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // per arm: page order by score (descending), cumulative catches, AUROC, ROC points
  const prep = {};
  for (const key in D.arms) {
    const s = D.arms[key].s;
    const order = s.map((v, i) => i).sort((a, b) => s[b] - s[a]);
    const cum = [0];
    for (const i of order) cum.push(cum[cum.length - 1] + y[i]);
    // AUROC by ranks (ties averaged)
    const asc = s.map((v, i) => i).sort((a, b) => s[a] - s[b]);
    const ranks = new Array(s.length);
    for (let i = 0; i < asc.length; ) {
      let j = i;
      while (j + 1 < asc.length && s[asc[j + 1]] === s[asc[i]]) j++;
      for (let t = i; t <= j; t++) ranks[asc[t]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    let rp = 0;
    for (let i = 0; i < s.length; i++) if (y[i]) rp += ranks[i];
    const auc = (rp - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
    const pts = [];
    for (let n = 0; n <= order.length; n += 5) pts.push([(n - cum[n]) / nNeg, cum[n] / nPos]);
    pts.push([1, 1]);
    prep[key] = { order, cum, auc, pts, label: D.arms[key].label };
    const o = document.createElement("option");
    o.value = key;
    o.textContent = D.arms[key].label;
    sel.appendChild(o);
  }

  function drawRoc(active, n) {
    roc.textContent = "";
    const P = 18, W = 220 - 2 * P;
    el("rect", { x: P, y: P, width: W, height: W, fill: "none", stroke: "var(--rule)" }, roc);
    el("line", { x1: P, y1: P + W, x2: P + W, y2: P, stroke: "var(--ink-faint)", "stroke-dasharray": "3 3" }, roc);
    const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + (P + p[0] * W).toFixed(1) + " " + (P + W - p[1] * W).toFixed(1)).join("");
    for (const key in prep) {
      if (key === active) continue;
      const g = el("path", { d: path(prep[key].pts), fill: "none", stroke: "var(--ink-faint)", "stroke-width": 1.2 }, roc);
      el("title", {}, g).textContent = `${prep[key].label}: AUROC ${prep[key].auc.toFixed(3)}`;
    }
    const a = prep[active];
    el("path", { d: path(a.pts), fill: "none", stroke: "var(--accent)", "stroke-width": 2.2 }, roc);
    const fx = (n - a.cum[n]) / nNeg, fy = a.cum[n] / nPos;
    el("circle", { cx: P + fx * W, cy: P + W - fy * W, r: 4.5, fill: "var(--accent)", stroke: "var(--canvas)", "stroke-width": 1.5 }, roc);
    const t1 = el("text", { x: P + W / 2, y: 214, "text-anchor": "middle", "font-size": 9, fill: "var(--ink-meta)" }, roc);
    t1.textContent = "human pages flagged";
    const t2 = el("text", { x: 10, y: P + W / 2, "text-anchor": "middle", "font-size": 9, fill: "var(--ink-meta)", transform: `rotate(-90 10 ${P + W / 2})` }, roc);
    t2.textContent = "AI-builder caught";
  }

  function drawStrip(active, n) {
    strip.textContent = "";
    const a = prep[active];
    const N = a.order.length, cols = 250, rows = Math.ceil(N / cols);
    const cw = 600 / cols, rh = 130 / rows;
    // lowest score at the left: walk the order backwards
    for (let r = 0; r < N; r++) {
      const i = a.order[N - 1 - r];
      const c = Math.floor(r / rows), rr = r % rows;
      el("rect", {
        x: (c * cw).toFixed(2), y: (rr * rh).toFixed(2), width: (cw * 0.9).toFixed(2), height: (rh * 0.85).toFixed(2),
        fill: y[i] ? "var(--accent)" : "var(--ink-faint)", opacity: y[i] ? 0.95 : 0.85,
      }, strip);
    }
    const xcut = ((N - n) / rows) * cw;
    el("line", { x1: xcut, y1: 0, x2: xcut, y2: 140, stroke: "var(--ink)", "stroke-width": 1.5 }, strip);
    const t = el("text", { x: Math.min(Math.max(xcut + 4, 4), 480), y: 149, "font-size": 15, fill: "var(--ink)" }, strip);
    t.textContent = "flagged →";
  }

  function update() {
    const key = sel.value;
    const share = +k.value / 100;
    const a = prep[key];
    const n = Math.max(1, Math.min(a.order.length, Math.round(share * a.order.length)));
    kout.textContent = k.value + "%";
    document.getElementById("sx-tpr").textContent = pct(a.cum[n] / nPos);
    document.getElementById("sx-fpr").textContent = pct((n - a.cum[n]) / nNeg);
    document.getElementById("sx-auc").textContent = a.auc.toFixed(3);
    drawStrip(key, n);
    drawRoc(key, n);
  }

  sel.addEventListener("change", update);
  k.addEventListener("input", update);
  box.classList.add("is-live");
  update();
})();
