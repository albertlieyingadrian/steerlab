/* The 2-D toy: five concepts squeezed into two dimensions, the SAE that splits each point into a few arrows, and the
   four linear tools on the same points. Data: window.TOY from toy-data.js (written by make_figures.py from
   results/toy.json, which 01_toy.py computed). One widget, used twice in the note (data-start picks the opening tab).
   No libraries; keyboard reachable; nothing animates. */
(function () {
  "use strict";
  var T = window.TOY;
  if (!T) return;
  var NS = "http://www.w3.org/2000/svg";
  var NAMES = ["A", "B", "C", "D", "E"];
  var R = 2.2; // the plot spans -R..R on both axes

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var key in attrs) e.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v) { return (Math.round(v * 100) / 100).toFixed(2); }

  // the true amounts of a point's concepts: solve for the 1 or 2 concepts it has
  function trueParts(i) {
    var on = [];
    T.active[i].forEach(function (a, j) { if (a) on.push(j); });
    var p = T.points[i];
    if (on.length === 0) return [];
    if (on.length === 1) {
      var c = T.concepts[on[0]];
      return [[on[0], p[0] * c[0] + p[1] * c[1]]];
    }
    var a = T.concepts[on[0]], b = T.concepts[on[1]];
    var det = a[0] * b[1] - a[1] * b[0];
    return [[on[0], (p[0] * b[1] - p[1] * b[0]) / det], [on[1], (a[0] * p[1] - a[1] * p[0]) / det]];
  }

  function build(root) {
    var mode = root.getAttribute("data-start") || "superposition";
    var k = "1", sel = 0, showAll = false;
    var tools = { pca: true, probe: true, dmeans: true, sae: true };
    root.classList.add("toy");

    var tabs = document.createElement("div");
    tabs.className = "toy-tabs";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "What the toy shows");
    var TABS = [["superposition", "1 · Superposition"], ["sae", "2 · The SAE's parts"], ["tools", "3 · Linear tools"]];
    var tabButtons = TABS.map(function (t) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "tab");
      b.textContent = t[1];
      b.setAttribute("data-key", "tab-" + t[0]);
      b.onclick = function () { mode = t[0]; draw(); };
      b.onkeydown = function (e) {
        var i = TABS.findIndex(function (x) { return x[0] === mode; });
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          mode = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length][0];
          draw();
          tabs.querySelector('[data-key="tab-' + mode + '"]').focus();
        }
      };
      tabs.appendChild(b);
      return b;
    });
    root.appendChild(tabs);

    var svg = el("svg", { viewBox: [-R, -R, 2 * R, 2 * R].join(" "), role: "img", "class": "toy-plot" });
    var defs = el("defs", {}, svg);
    [["ah-accent", "var(--accent)"], ["ah-ink", "var(--ink-soft)"], ["ah-control", "var(--control)"]].forEach(function (m) {
      var mk = el("marker", { id: root.id + m[0], viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5,
        orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", style: "fill:" + m[1] }, mk);
    });
    root.appendChild(svg);

    var controls = document.createElement("div");
    controls.className = "toy-controls";
    root.appendChild(controls);
    var readout = document.createElement("p");
    readout.className = "toy-readout";
    readout.setAttribute("aria-live", "polite");
    root.appendChild(readout);

    function line(x1, y1, x2, y2, style) {
      return el("line", { x1: x1, y1: -y1, x2: x2, y2: -y2, style: style, "vector-effect": "non-scaling-stroke" }, svg);
    }
    function arrow(x1, y1, x2, y2, color, marker, width, dashed) {
      var a = line(x1, y1, x2, y2, "stroke:" + color + ";stroke-width:" + width + (dashed ? ";stroke-dasharray:5 4" : ""));
      a.setAttribute("marker-end", "url(#" + root.id + marker + ")");
      return a;
    }
    function label(x, y, text, color) {
      var t = el("text", { x: x, y: -y, "text-anchor": "middle", "dominant-baseline": "middle",
        style: "fill:" + color + ";font-size:0.16px;font-family:var(--font-mono);font-weight:600" }, svg);
      t.textContent = text;
    }
    function ring(p) {
      el("circle", { cx: p[0], cy: -p[1], r: 0.07, style: "fill:none;stroke:var(--ink);stroke-width:1.5",
        "vector-effect": "non-scaling-stroke" }, svg);
    }

    function controlsFor() {
      var active = document.activeElement && controls.contains(document.activeElement)
        ? document.activeElement.getAttribute("data-key") : null;
      controls.innerHTML = "";
      if (mode === "sae") {
        var lab = document.createElement("label");
        lab.className = "toy-k";
        lab.innerHTML = "k, latents kept per point: <strong>" + k + "</strong> ";
        var s = document.createElement("input");
        s.type = "range"; s.min = 1; s.max = 3; s.step = 1; s.value = k;
        s.setAttribute("aria-label", "k, the number of latents kept per point");
        s.setAttribute("data-key", "k");
        s.oninput = function () { k = String(s.value); draw(); };
        lab.appendChild(s);
        controls.appendChild(lab);
        var cb = document.createElement("label");
        cb.innerHTML = '<input type="checkbox" data-key="all"' + (showAll ? " checked" : "") + "> show every rebuild" +
          (k === "1" ? "" : " (all exact at k ≥ 2)");
        cb.firstChild.onchange = function (e) { showAll = e.target.checked; draw(); };
        controls.appendChild(cb);
      }
      if (mode === "tools") {
        [["pca", "PCA, first component"], ["probe", "logistic probe"], ["dmeans", "difference of means"],
         ["sae", "SAE latent for A"]].forEach(function (t) {
          var cb = document.createElement("label");
          cb.innerHTML = '<input type="checkbox" data-key="tool-' + t[0] + '"' + (tools[t[0]] ? " checked" : "") + "> " + t[1];
          cb.firstChild.onchange = function (e) { tools[t[0]] = e.target.checked; draw(); };
          controls.appendChild(cb);
        });
      } else {
        var b = document.createElement("button");
        b.type = "button"; b.className = "toy-next"; b.textContent = "Pick another point";
        b.setAttribute("data-key", "next");
        b.onclick = function () { pickNext(); draw(); };
        controls.appendChild(b);
      }
      if (active) {
        var again = controls.querySelector('[data-key="' + active + '"]');
        if (again) again.focus();
      }
    }

    function pickNext() {
      // step through points with one or two concepts, away from the origin, so the parts are readable
      for (var n = 1; n <= T.points.length; n++) {
        var j = (sel + n * 37) % T.points.length;
        var count = T.active[j].reduce(function (a, b) { return a + b; }, 0);
        if (count >= 1 && Math.hypot(T.points[j][0], T.points[j][1]) > 0.6) { sel = j; return; }
      }
    }

    function draw() {
      tabButtons.forEach(function (b, i) { b.setAttribute("aria-selected", String(TABS[i][0] === mode)); });
      while (svg.childNodes.length > 1) svg.removeChild(svg.lastChild);
      line(-R, 0, R, 0, "stroke:var(--rule);stroke-width:1");
      line(0, -R, 0, R, "stroke:var(--rule);stroke-width:1");
      var sae = T.saes[k];
      T.points.forEach(function (p, i) {
        var hasA = T.active[i][0] === 1;
        if (mode === "tools" && hasA) {  // concept A: hollow ink rings, so orange stays for the SAE alone
          el("circle", { cx: p[0], cy: -p[1], r: 0.032, style: "fill:none;stroke:var(--ink);stroke-width:1;opacity:0.7",
            "vector-effect": "non-scaling-stroke" }, svg);
        } else {
          el("circle", { cx: p[0], cy: -p[1], r: 0.024, style: "fill:var(--ink-faint);opacity:" + (mode === "tools" ? 0.3 : 0.45) }, svg);
        }
        if (mode === "sae" && showAll) {
          var h = sae.x_hat[i];
          line(p[0], p[1], h[0], h[1], "stroke:var(--accent);stroke-width:0.7;opacity:0.5");
        }
      });
      var text = "";
      if (true) {
        T.concepts.forEach(function (c, j) {
          var faint = (mode === "tools" && j !== 0) || mode === "sae";  // tab 2: the concepts stay faintly behind
          var a = arrow(0, 0, 1.6 * c[0], 1.6 * c[1], "var(--ink-soft)", "ah-ink", 1.2, true);
          if (faint) a.style.opacity = 0.25;
          label(1.85 * c[0], 1.85 * c[1], NAMES[j], faint ? "var(--rule)" : "var(--ink-soft)");
        });
      }
      if (mode === "superposition") {
        var parts = trueParts(sel), cur = [0, 0];
        parts.forEach(function (pt) {
          var c = T.concepts[pt[0]], nxt = [cur[0] + pt[1] * c[0], cur[1] + pt[1] * c[1]];
          arrow(cur[0], cur[1], nxt[0], nxt[1], "var(--accent)", "ah-accent", 2.4);
          cur = nxt;
        });
        ring(T.points[sel]);
        text = "Five concepts (A to E) share two dimensions, so their arrows cannot all sit at right angles. The circled point is " +
          (parts.length ? parts.map(function (pt) { return fmt(pt[1]) + " × " + NAMES[pt[0]]; }).join(" + ") : "no concept") +
          " (orange). Each point uses one or two concepts at a time, and that sparsity is what lets an SAE pull them apart.";
      }
      if (mode === "sae") {
        var near = sae.decoder.map(function (d) {  // the concept each decoder arrow sits nearest
          var best = 0, bc = -2;
          T.concepts.forEach(function (c, i) { var v = c[0] * d[0] + c[1] * d[1]; if (v > bc) { bc = v; best = i; } });
          return NAMES[best];
        });
        sae.decoder.forEach(function (d, j) {
          arrow(0, 0, 1.6 * d[0], 1.6 * d[1], "var(--accent)", "ah-accent", 1.3);
          label(1.62 * d[0] + 0.0, 1.62 * d[1] - 0.18, (j + 1) + "≈" + near[j], "var(--accent)");
        });
        var z = sae.z[sel], cur2 = [0, 0], used = [];
        z.forEach(function (v, j) {
          if (v > 0) {
            var d = sae.decoder[j], nxt = [cur2[0] + v * d[0], cur2[1] + v * d[1]];
            arrow(cur2[0], cur2[1], nxt[0], nxt[1], "var(--accent)", "ah-accent", 2.8);
            cur2 = nxt;
            used.push(fmt(v) + " × latent " + (j + 1));
          }
        });
        var p2 = T.points[sel], h2 = sae.x_hat[sel];
        ring(p2);
        var err = Math.pow(p2[0] - h2[0], 2) + Math.pow(p2[1] - h2[1], 2);
        text = "k = " + k + ": the SAE rebuilds the circled point as " + (used.length ? used.join(" + ") : "nothing") +
          " (squared error " + fmt(err) + "). Over all 600 points its FVU is " + sae.fvu.toFixed(3) +
          ", and every true concept has a decoder arrow within cosine " + Math.min.apply(null, sae.match_cos).toFixed(2) + ". " +
          (k === "1" ? "One part per point forces the arrows onto the five concepts."
            : "With " + k + " parts per point any 2-D point can be rebuilt exactly, so nothing pins the arrows to the concepts and they slide.");
      }
      if (mode === "tools") {
        var m = T.tools.mean, L = 2.0;
        if (tools.pca) {
          var v = T.tools.pca1;
          line(m[0] - L * v[0], m[1] - L * v[1], m[0] + L * v[0], m[1] + L * v[1], "stroke:var(--ink-soft);stroke-width:3;stroke-dasharray:1 5;stroke-linecap:round");
          label(m[0] + L * v[0], m[1] + L * v[1] - 0.14, "PCA", "var(--ink-soft)");
        }
        if (tools.probe) {
          var w = T.tools.probe, base = [-T.tools.probe_bias * w[0], -T.tools.probe_bias * w[1]];
          line(base[0] - L * w[1], base[1] + L * w[0], base[0] + L * w[1], base[1] - L * w[0], "stroke:var(--ink);stroke-width:1.5");
          arrow(base[0], base[1], base[0] + 0.8 * w[0], base[1] + 0.8 * w[1], "var(--ink)", "ah-ink", 2);
          label(base[0] + 0.95 * w[0], base[1] + 0.95 * w[1] + 0.14, "probe", "var(--ink)");
        }
        if (tools.dmeans) {
          var dm = T.tools.dmeans;
          arrow(m[0], m[1], m[0] + 1.2 * dm[0], m[1] + 1.2 * dm[1], "var(--ink-soft)", "ah-ink", 2, true);
          label(m[0] + 1.32 * dm[0], m[1] + 1.32 * dm[1] - 0.14, "means", "var(--ink-soft)");
        }
        if (tools.sae) {
          var sa = T.tools.sae_a;
          arrow(0, 0, 1.5 * sa[0], 1.5 * sa[1], "var(--accent)", "ah-accent", 2.8);
          label(1.5 * sa[0], 1.5 * sa[1] - 0.16, "SAE", "var(--accent)");
        }
        var A = T.angles;
        text = "Ringed points have concept A. Angle to A's true arrow: PCA's first component " + Math.round(A.pca1) +
          "° (dotted line: the direction of most spread, blind to the label); the logistic probe " + A.probe.toFixed(1) +
          "° (solid line: the boundary between A and not-A, and the arrow across it); the difference of means " + A.dmeans.toFixed(1) +
          "° (dashed arrow: from the not-A average towards the A average); the SAE latent " + A.sae_a.toFixed(1) +
          "° (orange: found with no labels at all, as one of five parts).";
      }
      svg.setAttribute("aria-label", text);
      readout.textContent = text;
      controlsFor();
    }
    pickNext();
    draw();
  }

  document.querySelectorAll("[data-toy]").forEach(build);
})();
