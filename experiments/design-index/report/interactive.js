// The note's interactive piece, from data.js (derived numbers only: palettes, wireframes, similarities).
// 1. Year slider: 40 sampled palettes and the mean layout heatmap of the same sites in a given year, with the
//    spread of each feature space that year. 2. Nearest pages: a page and its five most similar pages from other
//    categories, by page structure. Without JavaScript the static figures above stand in for both.
(function () {
  "use strict";
  var D = window.DI_DATA;
  if (!D) return;
  // neutral inks: orange is kept for the AI classes' labels, blue for the controls themselves
  var CH = { t: "var(--ink-soft)", m: "color-mix(in srgb, var(--ink-faint) 40%, transparent)", c: "var(--ink)" };
  var CLS = { human: "Human site", ai_host: "AI-builder site", generated: "Open-model page", template: "Templated page" };
  var LABEL = { layout: "Layout grid", palette: "Palette", siglip: "Image embedding", dom: "Page structure" };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function palStrip(p) {
    var s = el("span", "di-pal");
    p.forEach(function (c) {
      var b = el("span");
      b.style.background = c[0];
      b.style.flexGrow = Math.max(c[1], 0.02);
      s.appendChild(b);
    });
    s.setAttribute("aria-hidden", "true");
    return s;
  }

  function wire(w, title) {
    var g = el("span", "di-wire");
    g.setAttribute("role", "img");
    g.setAttribute("aria-label", title);
    for (var i = 0; i < w.length; i++) {
      var c = el("span");
      if (w[i] !== ".") c.style.background = CH[w[i]];
      g.appendChild(c);
    }
    return g;
  }

  // ---------- 1. year slider ----------
  var yr = document.getElementById("di-years");
  if (yr && D.years && D.years.length) {
    yr.classList.add("is-live");
    var input = yr.querySelector("input[type=range]");
    var out = yr.querySelector("output");
    var pals = yr.querySelector(".di-pals");
    var heat = yr.querySelector(".di-heat");
    var stats = yr.querySelector(".di-stats");
    input.min = 0;
    input.max = D.years.length - 1;
    input.value = 0;
    for (var k = 0; k < 15 * 16; k++) heat.appendChild(el("span"));
    function show(i) {
      var Y = D.years[i];
      out.textContent = Y.year;
      input.setAttribute("aria-valuetext", Y.year + ", " + Y.n + " sites");
      pals.innerHTML = "";
      Y.palettes.forEach(function (p) { pals.appendChild(palStrip(p)); });
      var cells = heat.children;
      for (var j = 0; j < cells.length; j++) {
        cells[j].style.opacity = Math.pow(Math.min(1, Y.heat[j]), 1.5).toFixed(2);
      }
      heat.setAttribute("aria-label", "Mean layout heatmap for " + Y.year + ": how often each part of the first 1.5 screens is covered");
      yr.querySelector(".di-pals-cap").textContent = Y.palettes.length + " sites' palettes in " + Y.year +
        " (five colours each, widths = share of the first screen)";
      stats.innerHTML = "";
      ["layout", "palette", "dom", "siglip"].forEach(function (sp) {
        if (!Y[sp]) return;
        var li = el("li");
        li.appendChild(el("span", "di-k", LABEL[sp]));
        li.appendChild(el("span", "di-v", Y[sp][0].toFixed(3) + " [" + Y[sp][1][0].toFixed(3) + ", " + Y[sp][1][1].toFixed(3) + "]"));
        stats.appendChild(li);
      });
      yr.querySelector(".di-stats-cap").hidden = !stats.children.length;
      yr.querySelector(".di-n").textContent = Y.year + ": " + Y.n + " sites rendered from the archive";
    }
    input.addEventListener("input", function () { show(+input.value); });
    show(0);
  }

  // ---------- 2. nearest pages ----------
  var nx = document.getElementById("di-nearest");
  if (nx && D.explorer && D.explorer.length) {
    nx.classList.add("is-live");
    var sel = nx.querySelector("select");
    var groups = {};
    D.explorer.forEach(function (it, i) {
      if (!groups[it.cls]) {
        groups[it.cls] = el("optgroup");
        groups[it.cls].label = CLS[it.cls];
        sel.appendChild(groups[it.cls]);
      }
      var o = el("option", null, it.cat + " · repetition " + (it.rep >= 0 ? "+" : "\u2212") + Math.abs(it.rep).toFixed(1));
      o.value = i;
      groups[it.cls].appendChild(o);
    });
    var focus = nx.querySelector(".di-focus");
    var list = nx.querySelector(".di-nn");
    function card(it, sim) {
      var c = el("figure", "di-card di-" + it.cls);
      c.appendChild(wire(it.wire, "Layout wireframe of a " + CLS[it.cls].toLowerCase() + " in " + it.cat));
      c.appendChild(palStrip(it.pal));
      var cap = el("figcaption");
      cap.appendChild(el("span", "di-cls", CLS[it.cls]));
      cap.appendChild(el("span", null, " · " + it.cat + (sim != null ? " · similarity " + sim.toFixed(2) : "")));
      c.appendChild(cap);
      return c;
    }
    function pick(i) {
      var it = D.explorer[i];
      focus.innerHTML = "";
      focus.appendChild(card(it));
      list.innerHTML = "";
      it.nn.forEach(function (n) { list.appendChild(card(n, n.sim)); });
    }
    sel.addEventListener("change", function () { pick(+sel.value); });
    var first = D.explorer.findIndex(function (it) { return it.cls === "ai_host"; });
    sel.value = first < 0 ? 0 : first;
    pick(+sel.value);
  }
})();
