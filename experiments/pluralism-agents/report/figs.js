// Interactive figures for the pluralism-agents note. Data: window.FIGDATA (figdata.js, written by make_figures.py).
// Without JavaScript each figure shows its static PNG; every control is a native button or select (keyboard-operable).
(function () {
  "use strict";
  var D = window.FIGDATA;
  if (!D) return;
  var RUNS = ["single-s0", "single-s1", "jury-s0", "jury-s1"];
  var STYLE = {
    "single-s0": { c: "var(--fig-single)", dash: "", label: "single judge, seed 0" },
    "single-s1": { c: "var(--fig-single)", dash: "5 4", label: "single judge, seed 1" },
    "jury-s0": { c: "var(--fig-jury)", dash: "", label: "jury, seed 0" },
    "jury-s1": { c: "var(--fig-jury)", dash: "5 4", label: "jury, seed 1" }
  };
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function fmt(v, d) { return v == null ? "–" : (+v).toFixed(d == null ? 2 : d); }
  function buttons(bar, items, cur, onpick) {
    bar.innerHTML = "";
    items.forEach(function (it) {
      var b = el("button", { type: "button", "aria-pressed": it.key === cur ? "true" : "false" }, it.label);
      b.addEventListener("click", function () {
        bar.querySelectorAll("button").forEach(function (o) { o.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onpick(it.key);
      });
      bar.appendChild(b);
    });
  }
  function live(root) {
    root.classList.add("is-live");
    var fb = root.querySelector(".fallback");
    if (fb) fb.hidden = true;
  }

  // ------------------------------------------------------------------ Figure: airline measures against drift or step
  (function () {
    var root = document.getElementById("fig-airline");
    if (!root || !D.runs || !Object.keys(D.runs).length) return;
    live(root);
    var mbar = root.querySelector(".pick-measure"), xbar = root.querySelector(".pick-x");
    var plot = root.querySelector(".plot-area"), note = root.querySelector(".readout");
    var keys = ["refusal_held", "refusal_self", "handoff", "plans_strict", "pass1", "no_unauth_write", "plans_loose", "pass4", "plan_len", "finished", "judge_single", "judge_jury"];
    var cur = "refusal_held", xs = "drift";
    function scale(k) {
      if (k === "plan_len") return [0, 25];
      if (k.indexOf("judge") === 0) return [1, 10];
      return [0, 1];
    }
    function draw() {
      var W = Math.max(340, Math.min(640, plot.clientWidth || 640)), H = W < 500 ? 280 : 300, L = 46, R = 14, T = 14, B = 42;
      var ys = scale(cur), pts = {}, xmax = 0;
      RUNS.forEach(function (n) {
        var r = D.runs[n];
        if (!r) return;
        var a = [[0, null]];
        Object.keys(r.by_step).map(Number).sort(function (p, q) { return p - q; }).forEach(function (s) {
          var xv = xs === "drift" ? r.drift[String(s)] : s;
          a.push([xv, r.by_step[s].airline[cur]]);
          xmax = Math.max(xmax, xv);
        });
        pts[n] = a;
      });
      xmax = xmax || 1;
      var b0 = D.base.s0 ? D.base.s0.airline[cur] : null, b1 = D.base.s1 ? D.base.s1.airline[cur] : null;
      var bm = (b0 != null && b1 != null) ? (b0 + b1) / 2 : (b0 != null ? b0 : b1);
      function X(v) { return L + v / xmax * (W - L - R); }
      function Y(v) { return T + (1 - (v - ys[0]) / (ys[1] - ys[0])) * (H - T - B); }
      var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + D.measures[cur] + " against " + (xs === "drift" ? "drift" : "training step") + ' for four runs, with the untrained model\'s two reruns as a band">';
      for (var g = 0; g <= 4; g++) {
        var gv = ys[0] + g / 4 * (ys[1] - ys[0]);
        s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(gv) + '" y2="' + Y(gv) + '" stroke="currentColor" stroke-opacity=".12"/>';
        s += '<text x="' + (L - 8) + '" y="' + (Y(gv) + 4) + '" font-size="12" text-anchor="end" fill="currentColor" fill-opacity=".65">' + fmt(gv, ys[1] > 2 ? 0 : 2) + "</text>";
      }
      for (var t = 0; t <= 4; t++) {
        var tv = t / 4 * xmax;
        s += '<text x="' + X(tv) + '" y="' + (H - B + 18) + '" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity=".65">' + (xs === "drift" ? fmt(tv, 4) : Math.round(tv)) + "</text>";
      }
      s += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 6) + '" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity=".75">' + (xs === "drift" ? (W < 500 ? "drift (KL, nats per token)" : "drift: KL(untrained ‖ trained) on neutral prompts, nats per token") : "GRPO step") + "</text>";
      if (b0 != null && b1 != null) {
        var lo = Math.min(b0, b1), hi = Math.max(b0, b1);
        s += '<rect x="' + L + '" width="' + (W - L - R) + '" y="' + Y(hi) + '" height="' + Math.max(2, Y(lo) - Y(hi)) + '" fill="var(--fig-base)" fill-opacity=".45"/>';
      }
      RUNS.forEach(function (n) {
        if (!pts[n]) return;
        var a = pts[n].map(function (p, i) { return [p[0], i === 0 ? bm : p[1]]; });
        var d = a.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join(" ");
        s += '<path d="' + d + '" fill="none" stroke="' + STYLE[n].c + '" stroke-width="2.2" stroke-dasharray="' + STYLE[n].dash + '"/>';
        a.slice(1).forEach(function (p) { s += '<circle cx="' + X(p[0]) + '" cy="' + Y(p[1]) + '" r="3.5" fill="' + STYLE[n].c + '"/>'; });
      });
      s += "</svg>";
      plot.innerHTML = s;
      var fin = RUNS.filter(function (n) { return pts[n]; }).map(function (n) {
        var a = pts[n]; return STYLE[n].label + " " + fmt(a[a.length - 1][1]);
      });
      note.textContent = D.measures[cur] + ": untrained " + fmt(b0) + " and " + fmt(b1) + " (two reruns, grey band); at the last checkpoint " + fin.join(", ") + ".";
    }
    buttons(mbar, keys.map(function (k) { return { key: k, label: D.measures[k] }; }), cur, function (k) { cur = k; draw(); });
    buttons(xbar, [{ key: "drift", label: "against drift" }, { key: "step", label: "against step" }], xs, function (k) { xs = k; draw(); });
    draw();
    window.addEventListener("resize", function () { clearTimeout(root._t); root._t = setTimeout(draw, 150); });
  })();

  // ------------------------------------------------------------------ Figure: training curves
  (function () {
    var root = document.getElementById("fig-curves");
    if (!root || !D.curves || !Object.keys(D.curves).length) return;
    live(root);
    var bar = root.querySelector(".pick-measure"), plot = root.querySelector(".plot-area"), note = root.querySelector(".readout");
    var opts = [
      { key: "judge_reward", label: "own judge's reward (1-10)", lim: [1, 10] },
      { key: "gt_success", label: "τ-bench success", lim: [0, 1] },
      { key: "finished", label: "ended normally", lim: [0, 1] },
      { key: "mean_agent_turns", label: "agent turns", lim: [0, 30] }
    ];
    var cur = "judge_reward";
    function draw() {
      var o = opts.filter(function (x) { return x.key === cur; })[0];
      var W = Math.max(340, Math.min(640, plot.clientWidth || 640)), H = 260, L = 46, R = 14, T = 14, B = 40, n = 30;
      function X(v) { return L + (v - 1) / (n - 1) * (W - L - R); }
      function Y(v) { return T + (1 - (v - o.lim[0]) / (o.lim[1] - o.lim[0])) * (H - T - B); }
      var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + o.label + ' on each GRPO step\'s 64 training conversations, four runs">';
      for (var g = 0; g <= 4; g++) {
        var gv = o.lim[0] + g / 4 * (o.lim[1] - o.lim[0]);
        s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(gv) + '" y2="' + Y(gv) + '" stroke="currentColor" stroke-opacity=".12"/><text x="' + (L - 8) + '" y="' + (Y(gv) + 4) + '" font-size="12" text-anchor="end" fill="currentColor" fill-opacity=".65">' + fmt(gv, o.lim[1] > 2 ? 0 : 2) + "</text>";
      }
      [1, 10, 20, 30].forEach(function (t) { s += '<text x="' + X(t) + '" y="' + (H - B + 18) + '" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity=".65">' + t + "</text>"; });
      s += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 6) + '" font-size="12" text-anchor="middle" fill="currentColor" fill-opacity=".75">GRPO step (8 retail tasks × 8 conversations each)</text>';
      var last = [];
      RUNS.forEach(function (r) {
        var c = D.curves[r];
        if (!c) return;
        var d = c.map(function (p, i) { return (i ? "L" : "M") + X(p.step).toFixed(1) + "," + Y(p[cur]).toFixed(1); }).join(" ");
        s += '<path d="' + d + '" fill="none" stroke="' + STYLE[r].c + '" stroke-width="1.8" stroke-dasharray="' + STYLE[r].dash + '"/>';
        last.push(STYLE[r].label + " " + fmt(c[c.length - 1][cur]));
      });
      plot.innerHTML = s + "</svg>";
      note.textContent = o.label + " at the last step: " + last.join(", ") + ". Each step trains on different tasks, so the curve is noisy step to step.";
    }
    buttons(bar, opts, cur, function (k) { cur = k; draw(); });
    draw();
    window.addEventListener("resize", function () { clearTimeout(root._t); root._t = setTimeout(draw, 150); });
  })();

  // ------------------------------------------------------------------ Figure: tool-plan explorer
  (function () {
    var root = document.getElementById("fig-plans");
    if (!root || !D.plans) return;
    live(root);
    var sel = root.querySelector("select"), out = root.querySelector(".plans-out");
    var ids = Object.keys(D.plans).sort(function (a, b) { return a - b; });
    ids.forEach(function (id) {
      var p = D.plans[id];
      sel.appendChild(el("option", { value: id }, "Task " + id + (p.no_write ? " (refusal task)" : "")));
    });
    var start = D.plan_default && D.plans[D.plan_default] ? D.plan_default : ids[0];
    sel.value = start;
    var SHORT = {
      get_user_details: "user", get_reservation_details: "reservation", search_direct_flight: "search direct",
      search_onestop_flight: "search 1-stop", list_all_airports: "airports", calculate: "calc", get_flight_status: "status",
      book_reservation: "BOOK", cancel_reservation: "CANCEL", update_reservation_flights: "CHANGE FLIGHTS",
      update_reservation_baggages: "CHANGE BAGS", update_reservation_passengers: "CHANGE PAX", send_certificate: "CERTIFICATE",
      transfer_to_human_agents: "transfer"
    };
    function draw() {
      var p = D.plans[sel.value];
      out.innerHTML = "";
      if (p.purpose) out.appendChild(el("p", { class: "task-purpose" }, p.purpose));
      [["base", "untrained"]].concat(RUNS.map(function (r) { return [r, STYLE[r].label]; })).forEach(function (row) {
        var plans = p.plans[row[0]];
        if (!plans) return;
        var g = el("div", { class: "plan-group" });
        var distinct = {};
        plans.forEach(function (pl) { distinct[pl.join(">")] = 1; });
        var nd = Object.keys(distinct).length;
        g.appendChild(el("p", { class: "plan-head" }, row[1] + " · " + nd + " distinct plan" + (nd > 1 ? "s" : "") + " of " + plans.length));
        plans.forEach(function (pl, i) {
          var line = el("div", { class: "plan-line" });
          var ok = p.success[row[0]] && p.success[row[0]][i] >= 1;
          line.appendChild(el("span", { class: "plan-mark " + (ok ? "ok" : "no"), title: ok ? "τ-bench success" : "not a success" }, ok ? "✓" : "·"));
          if (!pl.length) line.appendChild(el("span", { class: "chip none" }, "no tool calls"));
          var shown = pl.slice(0, 14);
          shown.forEach(function (t) {
            var w = SHORT[t] && SHORT[t] === SHORT[t].toUpperCase();
            line.appendChild(el("span", { class: "chip" + (w ? " write" : "") }, SHORT[t] || t));
          });
          if (pl.length > shown.length) line.appendChild(el("span", { class: "chip more" }, "+" + (pl.length - shown.length)));
          g.appendChild(line);
        });
        out.appendChild(g);
      });
    }
    sel.addEventListener("change", draw);
    draw();
  })();
})();
