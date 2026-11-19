/* Two interactive figures for the pluralism-judge-rl note, drawn as SVG from window.PJR (written by make_figures.py).
   Without JavaScript each figure keeps its static PNG. Every control is a native button or select. */
(function () {
  "use strict";
  var D = window.PJR;
  if (!D) return;
  var NS = "http://www.w3.org/2000/svg";
  var ARMS = ["single", "jury", "bon", "shuffled", "clean"];
  var SHORT = { single: "Single judge", jury: "Jury", bon: "Best-of-8", shuffled: "Shuffled", clean: "Clean twin" };
  var SIGN = { srep_creative: 1, srep_advice: 1, srep50_creative: 1, srep50_advice: 1, scoverage: -1, ep1_entropy: -1,
               ep1_entropy_steered: -1, house_creative: 1, house_advice: 1, rep_creative: 1, rep_advice: 1, rep50_creative: 1, rep50_advice: 1, coverage: -1, steer: -1,
               survey_entropy: -1, survey_entropy_steered: -1 };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, s, attrs) {
    var t = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function ticks(lo, hi, n) {
    var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    [1, 2, 5, 10].some(function (m) { if (span / (step * m) <= n) { step *= m; return true; } return false; });
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function onResize(fn) {
    var t = null, last = 0;
    window.addEventListener("resize", function () {
      clearTimeout(t);
      t = setTimeout(function () { if (Math.abs(window.innerWidth - last) > 20) { last = window.innerWidth; fn(); } }, 150);
    });
  }
  function armColor(a) { return "var(--arm-" + a + ", " + D.labels.colors[a] + ")"; }
  var GROUPS = [
    ["Repetition (same prompt)", /^(s?rep)/], ["House style (different prompts)", /^house/], ["Coverage and steerability", /^(s?coverage|steer|cov_)/],
    ["Survey (Episode 1's eval)", /^ep1_/], ["Survey (probability readout)", /^survey_/], ["Reply length", /^words_/]
  ];
  function tickFmt(list) {
    var step = list.length > 1 ? Math.abs(list[1] - list[0]) : 1;
    var d = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
    return function (v) { return v.toFixed(d); };
  }
  function fmt(v) { return Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3); }
  function segButtons(box, options, current, onPick) {
    box.innerHTML = "";
    options.forEach(function (o) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "pjr-btn"; b.textContent = o[1];
      b.setAttribute("aria-pressed", o[0] === current() ? "true" : "false");
      b.addEventListener("click", function () { onPick(o[0]); box.querySelectorAll("button")[options.indexOf(o)].focus(); });
      box.appendChild(b);
    });
  }
  function fillSelect(sel, keys, current) {
    sel.innerHTML = "";
    GROUPS.forEach(function (gp) {
      var mine = keys.filter(function (k) { return gp[1].test(k); });
      if (!mine.length) return;
      var og = document.createElement("optgroup");
      og.label = gp[0];
      mine.forEach(function (k) {
        var o = document.createElement("option");
        o.value = k; o.textContent = (D.labels.measures[k] || k).replace(/ \(shared module\)/, "");
        if (k === current) o.selected = true;
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
  }

  /* ---------------- Figure: any measure against drift or steps, every run */
  function driftFigure(root) {
    var st = { task: "creative", measure: "house_creative", x: "drift_kl", hidden: {} };
    var svg = root.querySelector("svg"), live = root.querySelector(".pjr-live");
    var taskBox = root.querySelector('[data-role="task"]'), xBox = root.querySelector('[data-role="x"]');
    var armBox = root.querySelector('[data-role="arms"]'), sel = root.querySelector("select");
    fillSelect(sel, Object.keys(D.labels.measures), st.measure);
    sel.addEventListener("change", function () { st.measure = sel.value; draw(); });

    function draw() {
      segButtons(taskBox, [["creative", "Creative task"], ["advice", "Advice task"]], function () { return st.task; },
        function (v) { st.task = v; draw(); });
      segButtons(xBox, [["drift_kl", "Against drift"], ["step", "Against steps"]], function () { return st.x; },
        function (v) { st.x = v; draw(); });
      armBox.innerHTML = "";
      var present = {};
      D.rows.forEach(function (r) { if (r.task === st.task || r.task === "none") present[r.arm] = true; });
      ARMS.filter(function (a) { return present[a]; }).forEach(function (a, i) {
        var b = document.createElement("button");
        b.type = "button"; b.className = "pjr-btn pjr-arm";
        b.style.setProperty("--arm", armColor(a));
        b.setAttribute("aria-pressed", st.hidden[a] ? "false" : "true");
        b.innerHTML = '<span class="sw" aria-hidden="true"></span>' + D.labels.arms[a];
        b.addEventListener("click", function () { st.hidden[a] = !st.hidden[a]; draw(); armBox.querySelectorAll("button")[i].focus(); });
        armBox.appendChild(b);
      });
      var base = D.rows.find(function (r) { return r.run === "base-s0"; });
      var base1 = D.rows.find(function (r) { return r.run === "base-s1"; });
      var runs = {};
      D.rows.forEach(function (r) {
        if (r.arm === "base" || (r.task !== st.task && r.task !== "none")) return;
        (runs[r.run] = runs[r.run] || { arm: r.arm, pts: [] }).pts.push(r);
      });
      var series = Object.keys(runs).sort().map(function (k) {
        var pts = runs[k].pts.sort(function (a, b) { return a.step - b.step; });
        return { run: k, arm: runs[k].arm,
                 xy: [[0, base[st.measure]]].concat(pts.map(function (p) { return [p[st.x], p[st.measure]]; })) };
      }).filter(function (s) { return !st.hidden[s.arm]; });
      var xs = [0], ys = [base[st.measure]].concat(base1 ? [base1[st.measure]] : []);
      series.forEach(function (s) { s.xy.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); }); });
      var x0 = 0, x1 = Math.max.apply(null, xs) || 1, y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      var pad = (y1 - y0) * 0.08 || 0.01; y0 -= pad; y1 += pad;
      var W = Math.max(340, Math.round(root.clientWidth || 640)), H = Math.round(Math.min(340, Math.max(260, W * 0.6))), L = 52, R = 14, T = 14, B = 44;
      function sx(v) { return L + (v - x0) / (x1 - x0) * (W - L - R); }
      function sy(v) { return T + (1 - (v - y0) / (y1 - y0)) * (H - T - B); }
      svg.innerHTML = "";
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      var g = el("g", {}, svg);
      var yt = ticks(y0, y1, 5), yf = tickFmt(yt), xt = ticks(x0, x1, 6), xf = tickFmt(xt);
      yt.forEach(function (v) {
        el("line", { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: "grid" }, g);
        text(g, L - 6, sy(v) + 4, yf(v), { "text-anchor": "end", class: "tick" });
      });
      xt.forEach(function (v) {
        text(g, sx(v), H - B + 16, st.x === "step" ? v.toFixed(0) : xf(v), { "text-anchor": "middle", class: "tick" });
      });
      text(g, (L + W - R) / 2, H - 6, st.x === "step" ? "Optimizer step" : "Drift from base, KL per token (neutral prompts)",
           { "text-anchor": "middle", class: "axis-label" });
      if (base1) {  // the base model rescored with another sampling seed: the do-nothing wobble
        var a = Math.min(base[st.measure], base1[st.measure]), b = Math.max(base[st.measure], base1[st.measure]);
        el("rect", { x: L, width: W - L - R, y: sy(b) - 2, height: Math.max(4, sy(a) - sy(b) + 4), class: "floor" }, g);
        text(g, W - R - 4, sy(b) - 6, "base model, two sampling seeds", { "text-anchor": "end", class: "tick" });
      }
      if (st.x === "drift_kl" && D.matched[st.task]) {
        var dx = sx(D.matched[st.task].D);
        el("line", { x1: dx, x2: dx, y1: T, y2: H - B, class: "matched" }, g);
        text(g, dx + 4, T + 10, "matched drift", { class: "tick" });
      }
      series.forEach(function (s) {
        var c = armColor(s.arm);
        el("polyline", { points: s.xy.map(function (p) { return sx(p[0]) + "," + sy(p[1]); }).join(" "),
                         fill: "none", style: "stroke:" + c, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
        s.xy.forEach(function (p) { el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3, style: "fill:" + c }, svg); });
      });
      svg.setAttribute("aria-label", (D.labels.measures[st.measure] || st.measure) + " for " + series.length +
        " runs on the " + st.task + " task, against " + (st.x === "step" ? "steps" : "drift"));
      var finals = {};
      series.forEach(function (s) { (finals[s.arm] = finals[s.arm] || []).push(s.xy[s.xy.length - 1][1]); });
      live.textContent = "Base " + fmt(base[st.measure]) + ". Final checkpoints: " + Object.keys(finals).map(function (k) {
        return D.labels.arms[k] + " " + finals[k].map(fmt).join(" / ");
      }).join("; ") + ".";
    }
    draw();
    onResize(draw);
  }

  /* ---------------- Figure: change from base at matched drift, per seed with 95% intervals */
  function matchedFigure(root) {
    var st = { task: "creative", measure: "srep_creative" };
    var svg = root.querySelector("svg"), live = root.querySelector(".pjr-live");
    var taskBox = root.querySelector('[data-role="task"]'), sel = root.querySelector("select");
    var measures = ["srep_creative", "srep_advice", "srep50_creative", "srep50_advice", "scoverage", "steer",
                    "ep1_entropy", "ep1_entropy_steered", "ep1_jsd_default", "ep1_jsd_steered",
                    "rep_creative", "rep_advice", "words_creative", "words_advice"];
    fillSelect(sel, measures, st.measure);
    sel.addEventListener("change", function () { st.measure = sel.value; draw(); });
    function draw() {
      segButtons(taskBox, [["creative", "Trained on creative prompts"], ["advice", "Trained on advice"]],
        function () { return st.task; }, function (v) {
          st.task = v; st.measure = v === "advice" ? "ep1_entropy" : "srep_creative"; sel.value = st.measure; draw();
        });
      var M = D.matched[st.task];
      if (!M) return;
      var rows = [];
      ARMS.forEach(function (a) {
        var v = M.arms[a] && M.arms[a][st.measure];
        if (v) v.seeds.forEach(function (s) { rows.push({ arm: a, s: s }); });
      });
      var sign = SIGN[st.measure] || 0;
      var vals = [0];
      rows.forEach(function (r) { vals.push(r.s.lo, r.s.hi); });
      var fl = D.floor[st.measure];
      if (fl) vals.push(fl.lo, fl.hi);
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = (hi - lo) * 0.08 || 0.01;
      lo -= pad; hi += pad;
      var W = Math.max(340, Math.round(root.clientWidth || 640)), rowH = 26, L = W < 500 ? 120 : 170, R = 16, T = 18, B = 40, H = T + B + rows.length * rowH;
      function sx(v) { return L + (v - lo) / (hi - lo) * (W - L - R); }
      svg.innerHTML = "";
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      if (fl) el("rect", { x: sx(Math.min(fl.lo, fl.hi)), width: Math.max(1.5, Math.abs(sx(fl.hi) - sx(fl.lo))), y: T - 6,
                           height: rows.length * rowH + 6, class: "floor" }, svg);
      var mt = ticks(lo, hi, 6), mf = tickFmt(mt);
      mt.forEach(function (v) {
        el("line", { x1: sx(v), x2: sx(v), y1: T - 6, y2: H - B, class: "grid" }, svg);
        text(svg, sx(v), H - B + 16, mf(v), { "text-anchor": "middle", class: "tick" });
      });
      el("line", { x1: sx(0), x2: sx(0), y1: T - 6, y2: H - B, class: "zero" }, svg);
      text(svg, (L + W - R) / 2, H - 6, sign ? (W < 500 ? "Loss vs base (+ = less plural)" : "Plurality loss vs base (positive = less plural)") : "Change vs base",
           { "text-anchor": "middle", class: "axis-label" });
      rows.forEach(function (r, i) {
        var y = T + i * rowH + rowH / 2, c = armColor(r.arm);
        text(svg, L - 8, y + 4, (W < 500 ? SHORT[r.arm] : D.labels.arms[r.arm]) + " · " + r.s.run.split("-").pop(), { "text-anchor": "end", class: "tick" });
        el("line", { x1: sx(r.s.lo), x2: sx(r.s.hi), y1: y, y2: y, style: "stroke:" + c, "stroke-width": 2 }, svg);
        el("circle", { cx: sx(r.s.delta), cy: y, r: 4.5, style: "fill:" + c }, svg);
      });
      svg.setAttribute("aria-label", "Change at matched drift for " + (D.labels.measures[st.measure] || st.measure) + ", " + rows.length + " runs");
      live.textContent = "Matched drift " + M.D.toFixed(3) + " nats per token. " + ARMS.filter(function (a) {
        return M.arms[a] && M.arms[a][st.measure];
      }).map(function (a) {
        var v = M.arms[a][st.measure];
        return D.labels.arms[a] + " " + (v.seed_mean >= 0 ? "+" : "") + v.seed_mean.toFixed(3) + (v.beyond_floor ? (v.seeds.length > 1 ? " (both seeds beyond noise)" : " (one seed, beyond noise)") : "");
      }).join("; ") + ".";
    }
    draw();
    onResize(draw);
  }

  document.querySelectorAll(".pjr-fig").forEach(function (root) {
    root.classList.add("is-live");
    if (root.dataset.fig === "drift") driftFigure(root);
    if (root.dataset.fig === "matched") matchedFigure(root);
  });
})();
