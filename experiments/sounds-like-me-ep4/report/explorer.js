/* Sounds Like Me, Episode 4: the agreement explorer. Reads window.EP4 (data.js, written by export_pages.py).
   Static fallback: Figure 1's image stays visible when this script does not run. */
(function () {
  "use strict";
  var D = window.EP4;
  var fig = document.querySelector('[data-fig="agree"]');
  if (!D || !fig) return;
  var svg = document.getElementById("agree-svg");
  var sel = document.getElementById("agree-metric");
  var live = document.getElementById("agree-live");
  var shown = { series: true, panel: true, baseline: true };
  var COL = { series: "var(--accent)", panel: "var(--ink-soft)", baseline: "var(--ink-faint)" };

  function el(name, attrs, text) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function f(v) { return v.toFixed(2); }

  Object.keys(D.metrics).forEach(function (k) {
    var o = document.createElement("option");
    o.value = k;
    o.textContent = D.metrics[k].label;
    sel.appendChild(o);
  });

  function draw() {
    var m = D.metrics[sel.value];
    var rows = D.rulers.filter(function (r) { return shown[r.kind]; });
    // draw at the container's width so 12px labels stay 12px on a phone
    var W = Math.max(340, Math.round(svg.parentNode.getBoundingClientRect().width) || 720);
    var L = W < 520 ? 150 : 210, R = 44, rowH = 26, top = 24, H = top + rows.length * rowH + 30;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    function x(v) { return L + (v - m.lo) / (m.hi - m.lo) * (W - L - R); }
    var step = (m.hi - m.lo) > 0.6 ? 0.2 : 0.1;
    for (var i = Math.ceil(m.lo / step - 1e-9); i * step <= m.hi + 1e-9; i++) {
      var t = i * step;
      svg.appendChild(el("line", { x1: x(t), x2: x(t), y1: top - 6, y2: H - 24, class: "grid" }));
      svg.appendChild(el("text", { x: x(t), y: H - 8, "text-anchor": "middle", class: "tick" }, f(t)));
    }
    svg.appendChild(el("line", { x1: x(m.chance), x2: x(m.chance), y1: top - 10, y2: H - 24, class: "ref" }));
    svg.appendChild(el("text", { x: x(m.chance) + 4, y: top - 12, class: "tick" }, "chance"));
    var best = null;
    rows.forEach(function (r, j) {
      var y = top + j * rowH + rowH / 2;
      svg.appendChild(el("text", { x: L - 10, y: y + 4, "text-anchor": "end", class: "name" }, r.name));
      var v = m.v[r.key];
      if (!v) {
        svg.appendChild(el("text", { x: L + 4, y: y + 4, class: "na" }, "n/a"));
        return;
      }
      svg.appendChild(el("line", { x1: x(v[1]), x2: x(v[2]), y1: y, y2: y, stroke: COL[r.kind], "stroke-width": 3, "stroke-opacity": 0.5 }));
      svg.appendChild(el("circle", { cx: x(v[0]), cy: y, r: 5, fill: COL[r.kind] }));
      svg.appendChild(el("text", { x: Math.min(x(v[2]) + 6, W - R + 4), y: y + 4, class: "val" }, f(v[0])));
      if (!best || v[0] > best.v) best = { name: r.name, v: v[0] };
    });
    var lu = m.v.luar;
    live.textContent = m.label + ". " + (best ? "Highest shown: " + best.name + " (" + f(best.v) + ")" : "") +
      (lu ? "; LUAR " + f(lu[0]) + " [" + f(lu[1]) + ", " + f(lu[2]) + "]." : ".");
  }

  fig.querySelectorAll(".slm-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      var g = b.getAttribute("data-group");
      shown[g] = !shown[g];
      b.setAttribute("aria-pressed", String(shown[g]));
      draw();
    });
  });
  sel.addEventListener("change", draw);
  var lastW = 0;
  window.addEventListener("resize", function () {
    var w = Math.round(svg.parentNode.getBoundingClientRect().width);
    if (Math.abs(w - lastW) > 20) { lastW = w; draw(); }
  });
  sel.value = "h1_r";
  fig.classList.add("is-live");
  draw();
})();
