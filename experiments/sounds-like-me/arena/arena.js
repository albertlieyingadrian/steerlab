/* Sounds Like Me arena (mock). Reads window.ARENA (data.js, written offline by export_pages.py).
   No network calls: picks live in this page and, if the browser allows, in localStorage. */
(function () {
  "use strict";
  var A = window.ARENA;
  if (!A) return;
  var KEY = "slm-arena-picks";
  var picks = {};
  try { picks = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { picks = {}; }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(picks)); } catch (e) { /* private mode: keep in memory */ } }

  var tabs = document.querySelector(".tabs"), panel = document.getElementById("match"), verdict = document.getElementById("verdict");
  var current = 0;

  function h(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

  // A fixed shuffle per writer, so the order is blind but the same on every visit.
  function order(m) {
    var ids = m.drafts.map(function (_, i) { return i; }), seed = m.author.charCodeAt(m.author.length - 1);
    for (var i = ids.length - 1; i > 0; i--) { seed = (seed * 9301 + 49297) % 233280; var j = seed % (i + 1); var t = ids[i]; ids[i] = ids[j]; ids[j] = t; }
    return ids;
  }

  function renderTabs() {
    tabs.innerHTML = "";
    A.matchups.forEach(function (m, i) {
      var b = h("button", "tab", m.author);
      b.type = "button"; b.setAttribute("role", "tab"); b.id = "tab-" + i;
      b.setAttribute("aria-selected", String(i === current)); b.setAttribute("aria-controls", "match");
      b.tabIndex = i === current ? 0 : -1;
      if (picks[m.author]) { var d = h("span", "done", "✓"); d.setAttribute("aria-label", "picked"); b.appendChild(d); }
      b.addEventListener("click", function () { current = i; render(); });
      b.addEventListener("keydown", function (ev) {
        if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
          current = (current + (ev.key === "ArrowRight" ? 1 : A.matchups.length - 1)) % A.matchups.length;
          render(); document.getElementById("tab-" + current).focus();
        }
      });
      tabs.appendChild(b);
    });
    panel.setAttribute("aria-labelledby", "tab-" + current);
  }

  function scale(pos, steer) {
    var s = h("div", "scale"), x = function (v) { return Math.max(0, Math.min(100, (v + 1.5) / 2.7 * 100)); };
    var z = h("span", "z"); z.style.left = x(0) + "%"; s.appendChild(z);
    var o = h("span", "z"); o.style.left = x(1) + "%"; s.appendChild(o);
    var d = h("span", "dot" + (steer ? " steer" : "")); d.style.left = x(pos) + "%"; s.appendChild(d);
    s.setAttribute("aria-hidden", "true");
    return s;
  }

  function render() {
    renderTabs();
    var m = A.matchups[current], picked = picks[m.author];
    panel.innerHTML = "";
    var who = h("div", "who");
    who.appendChild(h("p", "label", "The writer"));
    who.appendChild(h("h2", null, m.author));
    who.appendChild(h("p", "role", m.role + " (" + (m.corpus === "blog" ? "Blog Authorship Corpus" : "Enron email corpus") + ")"));
    who.appendChild(h("p", "label", "Three of their real emails, names replaced"));
    m.excerpts.forEach(function (t) { who.appendChild(h("p", "excerpt", t)); });
    var br = h("details", "brief"); br.appendChild(h("summary", null, "The brief every draft was written from"));
    br.appendChild(h("p", null, "From: " + m.brief.sender + ". To: " + m.brief.to + ". Why now: " + m.brief.why + ". Ask for: " + m.brief.ask + "."));
    var ul = h("ul"); m.brief.facts.forEach(function (f) { ul.appendChild(h("li", null, f)); }); br.appendChild(ul);
    who.appendChild(br);
    panel.appendChild(who);

    var right = h("div");
    right.appendChild(h("p", "label", picked ? "Your pick is marked; every draft is now revealed" : "Which sounds most like " + m.author + "?"));
    var grid = h("div", "drafts" + (picked ? " revealed" : ""));
    order(m).forEach(function (idx, k) {
      var d = m.drafts[idx], card = h("article", "card" + (picked === d.id ? " chosen" : ""));
      card.appendChild(h("h3", null, "Draft " + (k + 1)));
      var tx = h("div", "text", d.text); card.appendChild(tx);
      var more = h("button", "more", "Show the whole draft"); more.type = "button"; more.hidden = true;
      more.addEventListener("click", function () { var open = tx.classList.toggle("open"); tx.classList.toggle("clamped", !open); more.textContent = open ? "Show less" : "Show the whole draft"; });
      card.appendChild(more);
      var pick = h("button", "pick", picked === d.id ? "Your pick" : "Sounds most like them"); pick.type = "button";
      pick.disabled = !!picked;
      pick.addEventListener("click", function () { picks[m.author] = d.id; save(); render(); });
      card.appendChild(pick);
      var rv = h("div", "reveal");
      var who2 = h("p"); who2.appendChild(h("b", null, d.model)); who2.appendChild(document.createTextNode(" · " + d.arm)); rv.appendChild(who2);
      rv.appendChild(scale(d.position, d.model.indexOf("vector") >= 0));
      rv.appendChild(h("p", null, "Voice position " + (d.position >= 0 ? "+" : "") + d.position.toFixed(2) + " for this writer, over all 4 of this model's drafts (0 = a different human, 1 = the same human). AI-detector score " + d.ai.toFixed(2) + "."));
      card.appendChild(rv);
      grid.appendChild(card);
      requestAnimationFrame(function () { if (tx.scrollHeight > tx.clientHeight + 4) { tx.classList.add("clamped"); more.hidden = false; } });
    });
    right.appendChild(grid);
    if (picked) {
      var reset = h("button", "reset", "Undo my pick for " + m.author); reset.type = "button";
      reset.addEventListener("click", function () { delete picks[m.author]; save(); render(); });
      right.appendChild(reset);
    }
    panel.appendChild(right);
    summary();
  }

  function summary() {
    var n = 0, agree = 0;
    A.matchups.forEach(function (m) {
      if (!picks[m.author]) return;
      n++;
      var best = m.drafts.reduce(function (a, b) { return b.position > a.position ? b : a; });
      if (best.id === picks[m.author]) agree++;
    });
    verdict.textContent = n ? "You picked for " + n + " of " + A.matchups.length + " writers; the voice ruler's top draft matched your pick " + agree + " time" + (agree === 1 ? "" : "s") + ". In Episode 1 the ruler's top draft was sometimes a loop: read before you trust it." : "";
  }

  var rows = document.getElementById("board-rows");
  A.board.cells.forEach(function (c) {
    var tr = h("tr");
    [c.writer, String(c.sent), String(c.reps), (c.unedited * 100).toFixed(1) + "%", c.similarity === null ? "n/a" : c.similarity.toFixed(2), c.length_ratio === null ? "n/a" : c.length_ratio.toFixed(2)]
      .forEach(function (v, i) {
        var td = h("td", i ? "num" : null, v);
        if (i === 3) { var bar = h("span", "bar"); bar.style.width = (c.unedited * 4) + "em"; bar.setAttribute("aria-hidden", "true"); td.insertBefore(bar, td.firstChild); }
        tr.appendChild(td);
      });
    rows.appendChild(tr);
  });
  document.getElementById("min-sent").textContent = A.board.min_sent;
  document.getElementById("min-reps").textContent = A.board.min_reps;
  document.getElementById("board-window").textContent = "Window " + A.board.window[0] + " to " + A.board.window[1] + " · " + A.board.scope + " · reply rates not included";
  render();
})();
