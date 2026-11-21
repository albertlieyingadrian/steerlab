/* Sounds Like Me, Episode 3: the ladder chart and the checkpoint slider. Reads window.SLM3 (data.js, written by
   export_pages.py). Static fallback: Figures 1 and 3 stay visible when this script does not run. */
(function () {
  "use strict";
  var D = window.SLM3;
  if (!D) return;
  function el(name, attrs, text) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function f(v, d) { return v === null || v === undefined ? "n/a" : v.toFixed(d); }

  /* ---------------- ladder ---------------- */
  var fig = document.querySelector('[data-fig="ladder"]');
  if (fig) {
    var METRICS = {
      pos: { label: "Voice position (strict; 0 = a different human, 1 = the same human)", get: function (r) { return r.pos[0]; }, ci: true, lo: -1.3, hi: 1.05, fmt: 2, ref: [[0, "a different human", "other (0)"], [1, "the same human", "same (1)"]] },
      spec: { label: "Specificity (closer to this writer than to the other 29)", get: function (r) { return r.spec; }, lo: -0.02, hi: 0.18, fmt: 3, ref: [[0, "no closer than to anyone", "0"], [D.human_spec, "writers' own emails", "own"]] },
      uns: { label: "Share of drafts with a claim the brief does not support (a lower bound)", get: function (r) { return r.uns; }, lo: 0, hi: 0.75, fmt: 2, ref: [[0.2167, "Episode 2's gate", "gate"]] },
      loop: { label: "Share of drafts that loop", get: function (r) { return r.loop; }, lo: 0, hi: 0.6, fmt: 2, ref: [] },
      task: { label: "Task coverage (who it is to, why now, the ask)", get: function (r) { return r.task; }, lo: 0, hi: 1, fmt: 2, ref: [] },
      words: { label: "Words per draft", get: function (r) { return r.words; }, lo: 0, hi: 180, fmt: 0, ref: [] },
      attr: { label: "Rank of the true writer under the n-gram attributer (1 = top of 30)", get: function (r) { return r.attr; }, lo: 1, hi: 17, fmt: 1, ref: [[15.5, "chance", "chance"]] }
    };
    var COLOR = { 3: "var(--accent)", 2: "var(--ink-soft)", 1: "var(--ink-faint)" };
    var state = { metric: "pos" };
    var sel = fig.querySelector("#slm3-metric"), svg = fig.querySelector("svg"), live = fig.querySelector(".slm-live");
    var draw = function () {
      var M = METRICS[state.metric];
      var rows = D.rows.filter(function (r) { var v = M.get(r); return v !== null && v !== undefined; })
        .sort(function (a, b) { return state.metric === "attr" ? M.get(a) - M.get(b) : M.get(b) - M.get(a); });
      var W = Math.max(340, Math.min(760, Math.round(svg.parentNode.clientWidth || 760)));
      var L = Math.round(Math.min(300, W * 0.46)), R = 44, top = 46, rowH = 24, H = top + rows.length * rowH + 46;
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
        var x = X(rf[0]);
        svg.appendChild(el("line", { x1: x, x2: x, y1: top - 10, y2: H - 34, "class": "ref" }));
        svg.appendChild(el("text", { x: x, y: top - 14 - i * 12, "text-anchor": "middle", "class": "ref-label" }, W < 500 ? rf[2] : rf[1]));
      });
      rows.forEach(function (r, k) {
        var y = top + k * rowH + rowH / 2, val = M.get(r);
        var g = el("g", { tabindex: "0", "class": "row", role: "listitem" });
        g.setAttribute("aria-label", r.label + ": " + val.toFixed(M.fmt));
        var t = el("text", { x: L - 10, y: y + 4, "text-anchor": "end", "class": "name" }, r.label.length > maxChars ? r.label.slice(0, maxChars - 1) + "…" : r.label);
        t.appendChild(el("title", {}, r.label));
        g.appendChild(t);
        var right = X(val);
        if (M.ci) { g.appendChild(el("line", { x1: X(r.pos[1]), x2: X(r.pos[2]), y1: y, y2: y, "class": "ci" })); right = X(r.pos[2]); }
        g.appendChild(el("circle", { cx: X(val), cy: y, r: 5.5, fill: COLOR[r.ep] }));
        var lx = right + 6, anchor = "start";
        if (lx > W - 34) { lx = X(val) - 9; anchor = "end"; }
        g.appendChild(el("text", { x: lx, y: y + 4, "class": "val", "text-anchor": anchor }, val.toFixed(M.fmt)));
        var describe = function () {
          live.textContent = r.label + ": position " + f(r.pos[0], 2) + " [" + f(r.pos[1], 2) + ", " + f(r.pos[2], 2) + "]; specificity " + f(r.spec, 3) +
            "; unsupported " + f(r.uns, 2) + "; loops " + f(r.loop, 2) + "; task " + f(r.task, 2) + "; " + (r.words == null ? "n/a" : Math.round(r.words)) + " words; attributer rank " + f(r.attr, 1) + ".";
        };
        g.addEventListener("mouseenter", describe);
        g.addEventListener("focus", describe);
        svg.appendChild(g);
      });
      svg.appendChild(el("text", { x: 4, y: H - 2, "class": "axis-label" }, W < 500 ? M.label.split(" (")[0] : M.label));
      live.textContent = rows.length + " arms. Hover or tab to a row for its numbers.";
    };
    sel.addEventListener("change", function () { state.metric = sel.value; draw(); });
    draw();
    fig.classList.add("is-live");
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(draw, 150); });
  }

  /* ---------------- checkpoint slider ---------------- */
  var sf = document.querySelector('[data-fig="slider"]');
  if (sf && D.slider && D.slider.length && D.hunt && D.hunt.length) {
    var steps = D.hunt.map(function (p) { return p.step; });
    var range = sf.querySelector("#slm3-step"), who = sf.querySelector("#slm3-writer"), out = sf.querySelector(".slm3-out"), stats = sf.querySelector(".slm3-stats");
    range.max = steps.length - 1;
    D.slider.forEach(function (d, i) { var o = document.createElement("option"); o.value = i; o.textContent = d.who + ", brief " + d.brief; who.appendChild(o); });
    var show = function () {
      var i = +range.value, st = steps[i], p = D.hunt[i], d = D.slider[+who.value];
      range.setAttribute("aria-valuetext", "step " + st);
      sf.querySelector(".slm3-step-label").textContent = st === 0 ? "Step 0 (the SFT start)" : "GRPO step " + st;
      stats.textContent = "Dev writers at this checkpoint: voice position " + f(p.position, 2) + "; unsupported " + f(p.unsupported, 2) +
        "; " + Math.round(p.words) + " words per draft; task " + f(p.task, 2) + (p.reward !== null ? "; training reward " + f(p.reward, 2) : "") + ".";
      var k = String(st);
      out.textContent = !(k in d.steps) ? "(not saved)" : (d.steps[k].trim() === "" ? "(empty draft)" : d.steps[k]);
    };
    range.addEventListener("input", show);
    who.addEventListener("change", show);
    sf.classList.add("is-live");
    show();
  }
})();
