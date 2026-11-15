// Interactive figure for the looped-transformer note. Data: the inline <script id="loop-data"> block, written by
// make_demo.py from results/ (probe outputs and the models' own answers on six real test chains, seed 1 models).
(function () {
  "use strict";
  var el = document.getElementById("loop-data");
  var root = document.getElementById("loopdemo");
  if (!el || !root) return;
  var D = JSON.parse(el.textContent);
  var state = { chain: 2, layer: 4, loop: 4, model: "looped_s2", playing: null };
  var $ = function (sel) { return root.querySelector(sel); };

  function answerBar(probs, right, label) {
    var wrap = document.createElement("div");
    wrap.className = "ld-answer";
    var head = document.createElement("p");
    head.className = "ld-answer-head";
    if (!probs) { head.textContent = label; wrap.appendChild(head); return wrap; }
    var best = 0;
    for (var i = 1; i < 10; i++) if (probs[i] > probs[best]) best = i;
    head.innerHTML = label + " <strong class='" + (best === right ? "ok" : "bad") + "'>" + best + "</strong> (" +
      Math.round(probs[best] * 100) + "%)" + (best === right ? " ✓" : " ✗, the right digit is " + right);
    wrap.appendChild(head);
    var bars = document.createElement("div");
    bars.className = "ld-bars";
    for (var d = 0; d < 10; d++) {
      var b = document.createElement("span");
      b.className = "ld-bar" + (d === right ? " right" : "");
      b.style.height = Math.max(2, probs[d] * 40) + "px";
      b.title = d + ": " + Math.round(probs[d] * 100) + "%";
      var lab = document.createElement("em");
      lab.textContent = d;
      b.appendChild(lab);
      bars.appendChild(b);
    }
    wrap.appendChild(bars);
    return wrap;
  }

  function shade(el, p) {
    if (p === null || p === undefined) { el.classList.add("np"); el.style.setProperty("--p", "0"); return; }
    el.style.setProperty("--p", Math.max(0, Math.min(1, (p - 0.1) / 0.8)).toFixed(3));
  }

  function chainRow(box, c, probFor, withDigit, probs) {
    box.innerHTML = "";
    var n = 0;
    c.chain_pos.forEach(function (pos, j) {
      if (j > 0) { var a = document.createElement("span"); a.className = "ld-arrow"; a.textContent = "→"; box.appendChild(a); }
      var s = document.createElement("span");
      s.className = "ld-chip in" + (j === 0 ? " q" : "");
      s.textContent = c.assign[pos];
      var p = probFor(pos, j);
      shade(s, p);
      if (p !== null && p >= 0.5) n++;
      s.title = p === null ? "not probed" : "probe probability " + p.toFixed(2);
      box.appendChild(s);
    });
    if (withDigit) {
      var a2 = document.createElement("span"); a2.className = "ld-arrow"; a2.textContent = "→"; box.appendChild(a2);
      var d = document.createElement("span");
      d.className = "ld-chip in digit";
      d.textContent = String(c.answer);
      shade(d, probs ? probs[c.answer] : 0);
      d.title = "the model's own probability for the right digit";
      box.appendChild(d);
    }
    return n;
  }

  function renderSide(sideId, M, step, nSteps, isLooped) {
    var c = D.chains[state.chain];
    var probs = isLooped ? (step > 0 ? M.answer[step - 1] : null) : (step === nSteps ? M.answer[0] : null);
    var walked = chainRow($("#ld-" + sideId + " .ld-chain.front"), c, function (pos, j) {
      return j === 0 ? 1 : (j <= 15 ? M.qhop[step][j - 1] : null);
    }, true, probs);
    var resolved = chainRow($("#ld-" + sideId + " .ld-chain.back"), c, function (pos) { return M.digit[step][pos]; }, false, null);
    $("#ld-" + sideId + " .ld-count").textContent = "Question's state: " + walked + " of " + c.hops + " links named. Lines that already hold their digit: " +
      resolved + " of " + c.hops + " (probe ≥ 50%).";
    var ans = $("#ld-" + sideId + " .ld-answer-slot");
    ans.innerHTML = "";
    if (!isLooped) {
      ans.appendChild(step === nSteps ? answerBar(M.answer[0], c.answer, "Answer at the output:")
        : answerBar(null, c.answer, "The answer comes out only after the last layer."));
    } else {
      ans.appendChild(step === 0 ? answerBar(null, c.answer, "Loop 0 is the prelude's output; run at least one loop.")
        : answerBar(M.answer[step - 1], c.answer, "Answer if it stops after loop " + step + ":"));
    }
  }

  function render() {
    var c = D.chains[state.chain];
    $(".ld-question").innerHTML = "What digit does <code>" + c.query + "</code> hold? Its chain is <strong>" + c.hops +
      " hops</strong> long" + (c.hops > 8 ? ", longer than any chain seen in training (1-8)." : ", inside the training range (1-8).") +
      " The other " + (48 - c.hops) + " lines are distractors.";
    var L = c.looped[state.model];
    var nStd = c.standard.digit.length - 1;
    var nLoop = L.digit.length - 1;
    $("#ld-layer").max = nStd;
    $("#ld-loop").max = nLoop;
    $("#ld-layer").value = state.layer;
    $("#ld-loop").value = state.loop;
    $("#ld-layer-v").textContent = state.layer === 0 ? "0 (embeddings)" : state.layer + " of " + nStd;
    $("#ld-loop-v").textContent = state.loop + (state.loop === 0 ? " (prelude only)" : " · " + (2 + 2 * state.loop) + " blocks run");
    renderSide("standard", c.standard, state.layer, nStd, false);
    renderSide("looped", L, state.loop, nLoop, true);
    root.querySelectorAll(".ld-pick button").forEach(function (b, i) { b.setAttribute("aria-pressed", i === state.chain ? "true" : "false"); });
    root.querySelectorAll(".ld-model button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.k === state.model ? "true" : "false"); });
  }

  var msel = $(".ld-model");
  Object.keys(D.models).forEach(function (k) {
    var b = document.createElement("button");
    b.type = "button";
    b.dataset.k = k;
    b.textContent = D.models[k];
    b.addEventListener("click", function () { state.model = k; render(); });
    msel.appendChild(b);
  });

  var pick = $(".ld-pick");
  D.chains.forEach(function (c, i) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = c.hops + " hops";
    b.addEventListener("click", function () { state.chain = i; render(); });
    pick.appendChild(b);
  });
  $("#ld-layer").addEventListener("input", function (e) { state.layer = +e.target.value; render(); });
  $("#ld-loop").addEventListener("input", function (e) { state.loop = +e.target.value; render(); });
  $(".ld-play").addEventListener("click", function () {
    var btn = this;
    if (state.playing) { clearInterval(state.playing); state.playing = null; btn.textContent = "▶ Play"; return; }
    btn.textContent = "❚❚ Pause";
    state.layer = 0; state.loop = 0; render();
    state.playing = setInterval(function () {
      var c = D.chains[state.chain];
      var nStd = c.standard.digit.length - 1, nLoop = c.looped[state.model].digit.length - 1;
      if (state.layer < nStd) state.layer++;
      if (state.loop < nLoop) state.loop++;
      render();
      if (state.loop >= nLoop) { clearInterval(state.playing); state.playing = null; btn.textContent = "▶ Play"; }
    }, 700);
  });
  render();

  // Accuracy chart: accuracy by hop count, with a loop-count selector for the looped model.
  var chart = document.getElementById("ld-chart");
  if (!chart) return;
  var A = D.accuracy;
  var W = 640, H = 300, L = 46, R = 12, T = 14, B = 40;
  var x = function (h) { return L + (h - 1) / 23 * (W - L - R); };
  var y = function (v) { return T + (1 - v) * (H - T - B); };
  var ns = "http://www.w3.org/2000/svg";
  var svg = null;
  function mk(tag, attrs, parent) {
    var e = document.createElementNS(ns, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }
  svg = mk("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Accuracy by hop count for the standard models and the looped model at the chosen loop count" }, chart);
  mk("rect", { x: x(0.5), y: T, width: x(8.5) - x(0.5), height: H - T - B, class: "ld-train" });
  mk("text", { x: (x(0.5) + x(8.5)) / 2, y: T + 12, "text-anchor": "middle", class: "ld-axis" }).textContent = "training range";
  [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
    mk("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "ld-grid-line" });
    mk("text", { x: L - 6, y: y(v) + 3, "text-anchor": "end", class: "ld-axis" }).textContent = v.toFixed(2);
  });
  [1, 4, 8, 12, 16, 20, 24].forEach(function (h) {
    mk("text", { x: x(h), y: H - B + 16, "text-anchor": "middle", class: "ld-axis" }).textContent = h;
  });
  mk("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "ld-axis" }).textContent = "hops in the question's chain";
  function path(vals) {
    return vals.map(function (v, i) { return (i ? "L" : "M") + x(i + 1).toFixed(1) + "," + y(v).toFixed(1); }).join(" ");
  }
  mk("path", { d: path(A.shortcut), class: "ld-l shortcut" });
  mk("path", { d: path(A.standard10), class: "ld-l std10" });
  mk("path", { d: path(A.standard4), class: "ld-l std4" });
  var loopLine = mk("path", { d: path(A.looped_k4), class: "ld-l looped" });
  var kSel = document.getElementById("ld-k");
  var ks = Object.keys(A).filter(function (k) { return k.indexOf("looped_k") === 0; })
    .map(function (k) { return +k.slice(8); }).sort(function (a, b) { return a - b; });
  ks.forEach(function (k) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = "k = " + k;
    b.addEventListener("click", function () { setK(k); });
    kSel.appendChild(b);
  });
  function mean(a, lo, hi) { var s = 0; for (var i = lo; i <= hi; i++) s += a[i - 1]; return s / (hi - lo + 1); }
  function setK(k) {
    var vals = A["looped_k" + k];
    loopLine.setAttribute("d", path(vals));
    kSel.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.textContent === "k = " + k ? "true" : "false"); });
    document.getElementById("ld-k-note").textContent = "Looped, k = " + k + " (" + (2 + 2 * k) + " blocks run): hops 1-8 " +
      mean(vals, 1, 8).toFixed(2) + ", 9-16 " + mean(vals, 9, 16).toFixed(2) + ", 17-24 " + mean(vals, 17, 24).toFixed(2) +
      ". Standard, 4 layers: " + mean(A.standard4, 9, 16).toFixed(2) + " and " + mean(A.standard4, 17, 24).toFixed(2) +
      " on 9-16 and 17-24. Means of 3 seeds, 500 items per hop count.";
  }
  setK(4);
})();
