/* Sounds Like Me, Episode 1: the leaderboard explorer. Reads window.SLM (data.js, written by export_pages.py).
   Static fallback: Figure 1 and the tables stay visible when this script does not run. */
(function () {
  "use strict";
  var D = window.SLM;
  var fig = document.querySelector('[data-fig="board"]');
  if (!D || !fig) return;

  var METRICS = {
    pos: { label: "Voice position (strict; 0 = a different human, 1 = the same human)", get: function (r) { return r.pos[0]; }, ci: function (r) { return [r.pos[1], r.pos[2]]; }, lo: -1.3, hi: 1.05, fmt: 2, ref: [[0, "a different human"], [1, "the same human"]], high: "better" },
    spec: { label: "Specificity (closer to this author than to the other 29)", get: function (r) { return r.spec; }, lo: -0.02, hi: 0.18, fmt: 3, ref: [[0, "no closer than to anyone"], [D.human_spec, "author's own emails"]], high: "better" },
    ai: { label: "AI-detector score (mean; human email averages 0.02)", get: function (r) { return r.ai; }, lo: 0, hi: 1, fmt: 2, ref: [[0.1017, "5% false-alarm cut"]], high: "worse" },
    uns: { label: "Share of drafts with a claim the brief does not support (a lower bound)", get: function (r) { return r.uns; }, lo: 0, hi: 0.7, fmt: 2, ref: [], high: "worse" },
    task: { label: "Task coverage (lanes or offer, why now, the ask)", get: function (r) { return r.task; }, lo: 0, hi: 1, fmt: 2, ref: [], high: "better" },
    loop: { label: "Share of drafts that loop (a 5-word run repeated 3+ times)", get: function (r) { return r.loop; }, lo: 0, hi: 0.7, fmt: 2, ref: [], high: "worse" },
    words: { label: "Words per draft", get: function (r) { return r.words; }, lo: 0, hi: 220, fmt: 0, ref: [], high: "" },
    usd: { label: "Cost per draft, US dollars", get: function (r) { return r.usd || null; }, lo: 0, hi: 0.006, fmt: 4, ref: [], high: "worse" }
  };
  var KIND_COLOR = { closed: "var(--ink)", open: "var(--ink-soft)", steer: "var(--accent)", humanizer: "var(--ink-faint)", control: "var(--control)" };
  var state = { metric: "pos", arm: "best", kinds: { closed: true, open: true, steer: true, humanizer: true, control: false } };

  var mSel = fig.querySelector("#slm-metric"), aSel = fig.querySelector("#slm-arm");
  var svg = fig.querySelector("svg"), live = fig.querySelector(".slm-live");

  function bestArm(rows) {
    var by = {};
    rows.forEach(function (r) {
      if (r.pos[0] === null || r.kind === "control" || r.kind === "humanizer") return;
      if (r.m === "partc-Qwen3-8B" && r.arm !== "own" && r.arm !== "k8_own") return;
      if (!by[r.m] || r.pos[0] > by[r.m].pos[0]) by[r.m] = r;
    });
    return Object.keys(by).map(function (k) { return by[k]; })
      .concat(rows.filter(function (r) { return r.kind === "humanizer" || r.kind === "control"; }));
  }

  function rowsNow() {
    var M = METRICS[state.metric];
    var rows = state.arm === "best" ? bestArm(D.rows) : D.rows.filter(function (r) { return r.arm === state.arm; });
    return rows.filter(function (r) { return state.kinds[r.kind] && M.get(r) !== null && M.get(r) !== undefined; })
      .sort(function (a, b) { return M.get(b) - M.get(a); });
  }

  function el(name, attrs, text) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function draw() {
    var M = METRICS[state.metric], rows = rowsNow();
    var W = Math.max(340, Math.min(760, Math.round(svg.parentNode.clientWidth || 760)));
    var L = Math.round(Math.min(300, W * 0.46)), R = 44, top = 34 + 12 * Math.max(0, M.ref.length - 1), rowH = 24, H = top + rows.length * rowH + 46;
    var maxChars = Math.floor((L - 14) / 6.4);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var X = function (v) { return L + (Math.max(M.lo, Math.min(M.hi, v)) - M.lo) / (M.hi - M.lo) * (W - L - R); };
    var span = M.hi - M.lo, raw = span / (W < 500 ? 3 : 5), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (k) { return k * mag; }).filter(function (k) { return k >= raw; })[0];
    var dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9) + (step / mag === 2.5 ? 1 : 0));
    for (var v = Math.ceil(M.lo / step) * step; v <= M.hi + 1e-9; v += step) {
      var x = X(v);
      svg.appendChild(el("line", { x1: x, x2: x, y1: top - 6, y2: H - 34, "class": "grid" }));
      svg.appendChild(el("text", { x: x, y: H - 20, "text-anchor": "middle", "class": "tick" }, (Math.abs(v) < 1e-9 ? 0 : v).toFixed(dec)));
    }
    M.ref.forEach(function (rf, i) {
      var x = X(rf[0]), nearRight = x > W - R - 60;
      svg.appendChild(el("line", { x1: x, x2: x, y1: top - 10, y2: H - 34, "class": "ref" }));
      svg.appendChild(el("text", { x: nearRight ? x + 4 : x, y: top - 14 - (M.ref.length - 1 - i) * 12, "text-anchor": nearRight ? "end" : "middle", "class": "ref-label" }, rf[1]));
    });
    rows.forEach(function (r, k) {
      var y = top + k * rowH + rowH / 2, val = M.get(r);
      var g = el("g", { tabindex: "0", "class": "row", role: "listitem" });
      var name = r.label + (state.arm === "best" || r.kind === "control" ? " · " + r.armLabel : "");
      g.setAttribute("aria-label", name + ": " + val.toFixed(M.fmt));
      var t = el("text", { x: L - 10, y: y + 4, "text-anchor": "end", "class": "name" }, name.length > maxChars ? name.slice(0, maxChars - 1) + "…" : name);
      t.appendChild(el("title", {}, name));
      g.appendChild(t);
      var right = X(val);
      if (M.ci && r.pos[1] !== null) {
        var c = M.ci(r);
        g.appendChild(el("line", { x1: X(c[0]), x2: X(c[1]), y1: y, y2: y, "class": "ci" }));
        right = X(c[1]);
      }
      g.appendChild(el("circle", { cx: X(val), cy: y, r: 5.5, fill: KIND_COLOR[r.kind] }));
      var lx = right + 6, anchor = "start";
      if (lx > W - 34) { lx = X(val) - 9; anchor = "end"; }
      g.appendChild(el("text", { x: lx, y: y + 4, "class": "val", "text-anchor": anchor }, val.toFixed(M.fmt)));
      g.addEventListener("mouseenter", function () { describe(r); });
      g.addEventListener("focus", function () { describe(r); });
      svg.appendChild(g);
    });
    svg.appendChild(el("text", { x: 4, y: H - 2, "class": "axis-label" }, (W < 500 ? M.label.split(" (")[0] : M.label) + (M.high ? " (higher is " + M.high + ")" : "")));
    live.textContent = rows.length + " rows. Hover or tab to a row for its full numbers.";
  }

  function f(v, d) { return v === null || v === undefined ? "n/a" : v.toFixed(d); }
  function describe(r) {
    live.textContent = r.label + ", " + r.armLabel + ": position " + f(r.pos[0], 2) + (r.pos[1] !== null ? " [" + f(r.pos[1], 2) + ", " + f(r.pos[2], 2) + "]" : " (all episodes copy-flagged)") +
      "; specificity " + f(r.spec, 3) + "; detector " + f(r.ai, 2) + "; unsupported claims " + f(r.uns, 2) +
      "; task " + f(r.task, 2) + "; " + Math.round(r.words) + " words; copy-flagged " + r.copy + "/" + r.n +
      (r.loop !== null && r.loop !== undefined ? "; loops " + f(r.loop, 2) : "") + (r.usd ? "; $" + r.usd.toFixed(4) + " per draft" : "");
  }

  var arms = []; D.rows.forEach(function (r) { if (arms.indexOf(r.arm) < 0 && r.kind !== "control") arms.push(r.arm); });
  var armNames = {}; D.rows.forEach(function (r) { armNames[r.arm] = r.armLabel; });
  arms.forEach(function (a) { var o = document.createElement("option"); o.value = a; o.textContent = armNames[a]; aSel.appendChild(o); });
  mSel.addEventListener("change", function () { state.metric = mSel.value; draw(); });
  aSel.addEventListener("change", function () { state.arm = aSel.value; draw(); });
  fig.querySelectorAll("[data-kind]").forEach(function (b) {
    b.setAttribute("aria-pressed", String(state.kinds[b.dataset.kind]));
    b.addEventListener("click", function () {
      state.kinds[b.dataset.kind] = !state.kinds[b.dataset.kind];
      b.setAttribute("aria-pressed", String(state.kinds[b.dataset.kind]));
      draw();
    });
  });
  fig.classList.add("is-live");
  draw();
  var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(draw, 150); });
})();
