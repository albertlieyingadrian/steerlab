/* Sounds Like Me, Episode 2: the ladder explorer and the draft viewer. Reads window.SLM2 (data.js, written by
   export_pages.py). Static fallback: Figure 1 and the tables stay visible when this script does not run. */
(function () {
  "use strict";
  var D = window.SLM2;
  if (!D) return;

  function el(name, attrs, text) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function f(v, d) { return v === null || v === undefined ? "n/a" : v.toFixed(d); }

  /* ---------------- ladder explorer ---------------- */
  var fig = document.querySelector('[data-fig="ladder"]');
  if (fig) {
    var METRICS = {
      pos: { label: "Voice position (strict; 0 = a different human, 1 = the same human)", get: function (r) { return r.pos[0]; }, ci: function (r) { return [r.pos[1], r.pos[2]]; }, lo: -1.0, hi: 1.05, fmt: 2, ref: [[0, "a different human", "other (0)"], [1, "the same human", "same (1)"]], high: "better" },
      pos80: { label: "Voice position, every draft cut to 80 words", get: function (r) { return r.pos80; }, lo: -1.0, hi: 1.05, fmt: 2, ref: [[0, "a different human", "other (0)"], [1, "the same human", "same (1)"]], high: "better" },
      spec: { label: "Specificity (closer to this author than to the other 29)", get: function (r) { return r.spec; }, lo: -0.02, hi: 0.18, fmt: 3, ref: [[0, "no closer than to anyone"], [D.human_spec, "author's own emails"]], high: "better" },
      attr: { label: "Rank of the true author under the n-gram attributer (1 = top of 30; never optimised)", get: function (r) { return r.attr; }, lo: 1, hi: 16, fmt: 1, ref: [[15.5, "chance"]], high: "worse" },
      uns: { label: "Share of drafts with a claim the brief does not support (a lower bound)", get: function (r) { return r.uns; }, lo: 0, hi: 0.7, fmt: 2, ref: [], high: "worse" },
      ai: { label: "AI-detector score (mean; human email averages 0.02)", get: function (r) { return r.ai; }, lo: 0, hi: 1, fmt: 2, ref: [], high: "worse" },
      loop: { label: "Share of drafts that loop or break (5-word run 3+ times, >20% repeated 4-grams, or a counting run)", get: function (r) { return r.loop; }, lo: 0, hi: 0.7, fmt: 2, ref: [], high: "worse" },
      words: { label: "Words per draft", get: function (r) { return r.words; }, lo: 0, hi: 220, fmt: 0, ref: [], high: "" },
      usd: { label: "Cost per final draft, US dollars (GPU time or API bill)", get: function (r) { return r.usd || null; }, lo: 0, hi: 0.06, fmt: 4, ref: [], high: "worse" }
    };
    var COLOR = { harness: "var(--ink)", ablation: "var(--ink-soft)", steer: "var(--accent)", prompt: "var(--ink-soft)", control: "var(--control)", ep1: "var(--ink-faint)" };
    var state = { metric: "pos", kinds: { harness: true, ablation: false, steer: true, prompt: true, control: true, ep1: true } };
    var mSel = fig.querySelector("#slm2-metric"), svg = fig.querySelector("svg"), live = fig.querySelector(".slm-live");

    var rowsNow = function () {
      var M = METRICS[state.metric];
      return D.rows.filter(function (r) { var v = M.get(r); return state.kinds[r.kind] && v !== null && v !== undefined; })
        .sort(function (a, b) { return state.metric === "attr" ? M.get(a) - M.get(b) : M.get(b) - M.get(a); });
    };

    var describe = function (r) {
      live.textContent = r.label + ": position " + f(r.pos[0], 2) + " [" + f(r.pos[1], 2) + ", " + f(r.pos[2], 2) + "]; at 80 words " +
        f(r.pos80, 2) + "; specificity " + f(r.spec, 3) + "; attributer rank " + f(r.attr, 1) + "; unsupported " + f(r.uns, 2) +
        "; detector " + f(r.ai, 2) + "; loops " + f(r.loop, 2) + "; " + Math.round(r.words) + " words" + (r.usd ? "; $" + r.usd.toFixed(4) + " per draft" : "") + ".";
    };

    var draw = function () {
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
        svg.appendChild(el("text", { x: nearRight ? x + 4 : x, y: top - 14 - (M.ref.length - 1 - i) * 12, "text-anchor": nearRight ? "end" : "middle", "class": "ref-label" }, W < 500 && rf[2] ? rf[2] : rf[1]));
      });
      rows.forEach(function (r, k) {
        var y = top + k * rowH + rowH / 2, val = M.get(r);
        var g = el("g", { tabindex: "0", "class": "row", role: "listitem" });
        g.setAttribute("aria-label", r.label + ": " + val.toFixed(M.fmt));
        var t = el("text", { x: L - 10, y: y + 4, "text-anchor": "end", "class": "name" }, r.label.length > maxChars ? r.label.slice(0, maxChars - 1) + "…" : r.label);
        t.appendChild(el("title", {}, r.label + " (" + r.group + ")"));
        g.appendChild(t);
        var right = X(val);
        if (M.ci && r.pos[1] !== null) {
          var c = M.ci(r);
          g.appendChild(el("line", { x1: X(c[0]), x2: X(c[1]), y1: y, y2: y, "class": "ci" }));
          right = X(c[1]);
        }
        g.appendChild(el("circle", { cx: X(val), cy: y, r: 5.5, fill: COLOR[r.kind] }));
        var lx = right + 6, anchor = "start";
        if (lx > W - 34) { lx = X(val) - 9; anchor = "end"; }
        g.appendChild(el("text", { x: lx, y: y + 4, "class": "val", "text-anchor": anchor }, val.toFixed(M.fmt)));
        g.addEventListener("mouseenter", function () { describe(r); });
        g.addEventListener("focus", function () { describe(r); });
        svg.appendChild(g);
      });
      svg.appendChild(el("text", { x: 4, y: H - 2, "class": "axis-label" }, (W < 500 ? M.label.split(" (")[0] : M.label) + (M.high ? " (higher is " + M.high + ")" : "")));
      live.textContent = rows.length + " rows. Hover or tab to a row for its full numbers.";
    };

    mSel.addEventListener("change", function () { state.metric = mSel.value; draw(); });
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
  }

  /* ---------------- draft viewer ---------------- */
  var dv = document.querySelector('[data-fig="drafts"]');
  if (dv && D.drafts && D.drafts.length) {
    var sel = dv.querySelector("#slm2-author"), box = dv.querySelector(".slm2-drafts");
    D.drafts.forEach(function (d, i) {
      var o = document.createElement("option");
      o.value = i; o.textContent = d.who + ", brief " + d.brief;
      sel.appendChild(o);
    });
    var show = function () {
      var d = D.drafts[+sel.value];
      while (box.firstChild) box.removeChild(box.firstChild);
      Object.keys(D.labels).forEach(function (arm) {
        if (!(arm in d.arms)) return;
        var p = document.createElement("p");
        p.className = "draft" + (arm === "R3_voice" || arm === "R4_persona" || arm === "xR3_s0.3" ? " steered" : "");
        var who = document.createElement("span");
        who.className = "who"; who.textContent = D.labels[arm];
        p.appendChild(who);
        p.appendChild(document.createTextNode(d.arms[arm]));
        box.appendChild(p);
      });
    };
    sel.addEventListener("change", show);
    dv.classList.add("is-live");
    show();
  }
})();
