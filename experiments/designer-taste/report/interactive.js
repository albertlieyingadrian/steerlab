/* The router explorer (The Taste Stack Episode 7). Recomputes the router from window.ROUTER: for each part and fold
   direction, every tier's training-fold agreement and its held-out hits (aggregates only, made by make_figures.py).
   Rule: the cheapest enabled tier within the tolerance of the best enabled tier on the training fold. */
(function () {
  "use strict";
  var R = window.ROUTER;
  var box = document.getElementById("rx");
  if (!R || !box) return;
  var TIERS = ["formal", "probe", "small", "mid", "large"];
  var SHORT = { formal: "Formal check", probe: "Probe", small: "Qwen-7B", mid: "Gemma-27B", large: "Qwen-72B" };
  var on = { formal: true, probe: true, small: true, mid: true, large: true };
  var eps = document.getElementById("rx-eps");
  var epsOut = document.getElementById("rx-eps-out");
  var parts = [];
  R.table.forEach(function (r) { if (parts.indexOf(r.part) < 0) parts.push(r.part); });

  function pick(row) {
    var e = parseFloat(eps.value);
    var cand = TIERS.filter(function (t) { return on[t] && t in row.train_acc; });
    if (!cand.length) return "none";  // nothing enabled applies to this criterion: a coin flip, as single() scores it
    var best = Math.max.apply(null, cand.map(function (t) { return row.train_acc[t]; }));
    var ok = cand.filter(function (t) { return row.train_acc[t] >= best - e - 1e-12; });
    ok.sort(function (a, b) { return R.cost[a] - R.cost[b]; });
    return ok[0];
  }

  function single(t) {
    var acc = 0, cost = 0, n = 0;
    parts.forEach(function (p) {
      var rows = R.table.filter(function (r) { return r.part === p; });
      var np = rows.reduce(function (s, r) { return s + r.n_test; }, 0);
      var hits = rows.reduce(function (s, r) { return s + (t in r.test_hits ? r.test_hits[t] : 0.5 * r.n_test); }, 0);
      acc += hits / np / parts.length;
      var applies = rows.filter(function (r) { return t in r.test_hits; });  // no cost where the judge has no check
      cost += R.cost[t] * applies.reduce(function (s2, r) { return s2 + r.n_test; }, 0); n += np;
    });
    return { acc: acc, cost: 1000 * cost / n };
  }

  function fmt(x) { return x.toFixed(3); }
  function money(x) { return "$" + (x < 0.1 ? x.toFixed(3) : x.toFixed(2)); }

  function render() {
    epsOut.textContent = parseFloat(eps.value).toFixed(3);
    var body = "", accSum = 0, costSum = 0, nSum = 0;
    var none = !TIERS.some(function (t) { return on[t]; });
    parts.forEach(function (p) {
      var rows = R.table.filter(function (r) { return r.part === p; });
      var np = 0, hits = 0, cells = [];
      rows.forEach(function (r) {
        var t = pick(r);
        if (t === "none") {
          cells.push("no judge applies (coin flip)");
          hits += 0.5 * r.n_test; np += r.n_test; nSum += r.n_test;
          return;
        }
        cells.push(SHORT[t]);
        hits += r.test_hits[t]; np += r.n_test;
        costSum += R.cost[t] * r.n_test; nSum += r.n_test;
      });
      if (!np) return;
      accSum += hits / np / parts.length;
      var same = cells[0] === cells[1];
      body += "<tr><td>" + p + "</td><td>" + (same ? cells[0] : cells.join(" / ")) + "</td><td class=\"num\">" + fmt(hits / np) + "</td></tr>";
    });
    document.getElementById("rx-rows").innerHTML = body;
    if (none) {
      document.getElementById("rx-acc").textContent = "–";
      document.getElementById("rx-cost").textContent = "–";
      document.getElementById("rx-best").textContent = "–";
      document.getElementById("rx-rows").innerHTML = "";
      document.getElementById("rx-note").textContent = "Turn on at least one judge.";
      return;
    }
    document.getElementById("rx-acc").textContent = fmt(accSum);
    document.getElementById("rx-cost").textContent = money(1000 * costSum / nSum);
    var best = null;
    TIERS.forEach(function (t) { if (on[t]) { var s = single(t); if (!best || s.acc > best.acc) best = { t: t, acc: s.acc, cost: s.cost }; } });
    document.getElementById("rx-best").textContent = SHORT[best.t] + " on every criterion: " + fmt(best.acc) + " at " + money(best.cost);
    document.getElementById("rx-note").textContent = "";
  }

  var tg = document.getElementById("rx-tiers");
  TIERS.forEach(function (t) {
    var id = "rx-t-" + t;
    var lab = document.createElement("label");
    lab.className = "rx-tier";
    lab.innerHTML = "<input type=\"checkbox\" id=\"" + id + "\" checked> " + SHORT[t] + " <span class=\"rx-c\">" + money(1000 * R.cost[t]) + "/1k</span>";
    tg.appendChild(lab);
    lab.querySelector("input").addEventListener("change", function (ev) { on[t] = ev.target.checked; render(); });
  });
  eps.addEventListener("input", render);
  box.classList.add("is-live");
  render();
})();
