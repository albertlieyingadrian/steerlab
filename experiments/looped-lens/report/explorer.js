// Readout explorer: pick a question and a lens, see the top token at each of the 48 layer visits.
// Data: window.LENS from data.js (written by make_figures.py from results/examples.json).
(function () {
  var box = document.getElementById("lx");
  if (!box || !window.LENS) return;
  var D = window.LENS;
  var LENSES = [
    ["act_in", "Logit lens (starter)"],
    ["mix_in", "Output head's mix"],
    ["mlp_in", "MoE input"],
    ["tuned_mlp_in", "Tuned lens"],
    ["future_mlp_in", "Future lens"],
    ["concept_mlp_in", "Concept lens"],
  ];
  var PASS = [[0, 6, "prelude"], [6, 18, "core pass 1"], [18, 30, "pass 2"], [30, 42, "pass 3"], [42, 48, "coda"]];
  var v2 = {};
  D.v2.forEach(function (p) { v2[p[0]] = p[1]; });
  var state = { ex: 0, lens: "act_in", visit: 44 };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function chips(row, items, key, label) {
    row.innerHTML = "";
    row.appendChild(el("span", "lx-label", label));
    items.forEach(function (it) {
      var on = state[key] === it[0];
      var b = el("button", "lx-chip" + (on ? " on" : ""), it[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", on ? "true" : "false");
      b.addEventListener("click", function () { state[key] = it[0]; render(); });
      row.appendChild(b);
    });
  }
  var exRow = el("div", "lx-row"), lensRow = el("div", "lx-row"), q = el("p", "lx-q");
  var strip = el("div", "lx-strip"), detail = el("p", "lx-detail"), legend = el("p", "lx-legend");
  legend.innerHTML = '<span class="lx-key gold"></span> votes for the intermediate <span class="lx-key wrong"></span> votes for a wrong option <span class="lx-key none"></span> names no option <span class="lx-key nr"></span> lens not fitted here · • a visit the v2 entry reads';
  [exRow, lensRow, q, strip, legend, detail].forEach(function (e) { box.appendChild(e); });

  function render() {
    var e = D.examples[state.ex];
    chips(exRow, D.examples.map(function (x, i) { return [i, x.id.replace(/-\d+$/, "").replace("_", " ") + (x.passes_v2 ? "" : " (fails)")]; }), "ex", "Question");
    chips(lensRow, LENSES, "lens", "Lens");
    q.innerHTML = "";
    q.appendChild(el("strong", "", e.prompt));
    q.appendChild(document.createTextNode(" Hidden step: "));
    q.appendChild(el("code", "", e.intermediate));
    q.appendChild(document.createTextNode(" · options: " + e.options.join(", ") + " · the v2 entry " + (e.passes_v2 ? "passes" : "fails")));
    strip.innerHTML = "";
    var reads = e.reads[state.lens];
    var gold = e.options.indexOf(e.intermediate);
    PASS.forEach(function (p) {
      var grp = el("div", "lx-pass");
      grp.appendChild(el("span", "lx-pname", p[2]));
      var cells = el("div", "lx-cells");
      cells.style.gridTemplateColumns = "repeat(" + (p[1] - p[0]) + ", minmax(0, 1fr))";
      for (var v = p[0]; v < p[1]; v++) {
        var r = reads[String(v)];
        var cls = "lx-cell";
        var tok = "";
        if (!r) cls += " nr";
        else {
          tok = (r[0][0] || "").trim() || "·";
          cls += r[1] === null ? " none" : (r[1] === gold ? " gold" : " wrong");
        }
        if (v === state.visit) cls += " sel";
        var c = el("button", cls);
        c.type = "button";
        c.title = "visit " + v + (r ? ": " + r[0].map(function (t) { return t.trim(); }).join(", ") : ": not fitted");
        c.appendChild(el("span", "lx-v", String(v) + (v2[v] ? "•" : "")));
        c.appendChild(el("span", "lx-t", tok));
        (function (vv) { c.addEventListener("click", function () { state.visit = vv; render(); }); })(v);
        cells.appendChild(c);
      }
      grp.appendChild(cells);
      strip.appendChild(grp);
    });
    var rr = reads[String(state.visit)];
    var rate = (D.rates[state.lens] || {})[String(state.visit)];
    detail.textContent = "Visit " + state.visit + ", top 5: " +
      (rr ? rr[0].map(function (t) { return t.trim() || "·"; }).join(", ") : "lens not fitted at this visit") +
      (rate ? ". Over all 867 questions this read votes for the hidden step " + Math.round(rate[0] * 100) +
        "% of the time and for a wrong option " + Math.round(rate[1] * 100) + "%." : "") +
      (v2[state.visit] ? " The v2 entry reads this visit with " + v2[state.visit] + "." : "");
  }
  render();
})();
