// Figure 1's live version: every contestant at one signal (or at its best on the chosen measure), on one measure.
// Data: data.js (aggregates only). Drawn at the container's own width, so text keeps its CSS size on a phone.
(function () {
  const D = window.INKLING_DATA;
  const fig = document.querySelector('.slm-fig[data-fig="board"]');
  if (!D || !fig) return;
  const INK = "inkling-2-mini";
  const ASIDE = "inkling-2-mini-reason-low"; // exploratory, two-three signals: drawn below a rule, never ranked
  const ARMS = [
    ["best", "Best signal"], ["k0", "No samples"], ["k3_rand", "3 samples"], ["k3_match", "3 matched"],
    ["k8_rand", "8 samples"], ["note", "Style note"], ["edits", "3 more samples*"],
  ];
  const pct = (v) => Math.round(v * 100) + "%";
  const voice = { fmt: (v) => v.toFixed(2), better: "higher", lo: -1.8, hi: 0.4, ticks: [-1.5, -1, -0.5, 0], ref: 0, refLabel: "0 = a stranger", axis: "higher = closer to the writer" };
  const MEASURES = {
    voice_strict: { label: "Voice position (strict)", ...voice },
    voice_loose: { label: "Voice position (loose, layout kept)", ...voice },
    voice80: { label: "Voice position, first 80 words", ...voice },
    spec: { label: "Specificity (this writer vs the other 29)", fmt: (v) => v.toFixed(3), better: "higher", lo: -0.03, hi: 0.18, ticks: [0, 0.05, 0.1, 0.15], ref: D.reference.human_specificity, refLabel: "writers' own texts", axis: "higher = more like this writer" },
    ai_flagged: { label: "Share flagged as AI by the detector", fmt: pct, better: "lower", lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], axis: "lower = less like an assistant" },
    unsupported: { label: "Share with an unsupported claim", fmt: pct, better: "lower", lo: 0, hi: 0.7, ticks: [0, 0.2, 0.4, 0.6], axis: "lower = fewer unsupported claims" },
    words: { label: "Words per draft", fmt: (v) => String(Math.round(v)), better: "none", lo: 0, hi: 220, ticks: [0, 50, 100, 150, 200], axis: "words per draft" },
  };
  const ARM_NAME = Object.fromEntries(ARMS);
  const TAG = { k0: "none", k3_rand: "3", k3_match: "3 matched", k8_rand: "8", note: "note", edits: "3 more*" };
  const sel = fig.querySelector("#board-measure");
  const btns = fig.querySelector(".slm-btns");
  const svg = fig.querySelector("#board-svg");
  const live = fig.querySelector("#board-live");
  const table = fig.querySelector("#board-table");
  let measure = "voice_strict", arm = "best";

  for (const [k, m] of Object.entries(MEASURES)) sel.add(new Option(m.label, k));
  for (const [k, name] of ARMS) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "slm-btn"; b.dataset.arm = k; b.textContent = name;
    b.setAttribute("aria-pressed", String(k === arm));
    b.addEventListener("click", () => { arm = k; draw(); });
    btns.appendChild(b);
  }
  sel.addEventListener("change", () => { measure = sel.value; draw(); });

  const triple = (x) => (x == null ? null : Array.isArray(x) ? x : [x, null, null]);
  const value = (c, a) => { const r = c.arms[a]; const t = r && triple(r[measure]); return t && t[0] != null ? t : null; };
  function pick(c, M) {
    if (arm !== "best") { const v = value(c, arm); return v ? { arm, v } : null; }
    // "Best" = the signal that does best on the chosen measure (strict voice decides for words, which has no better)
    const key = M.better === "none" ? "voice_strict" : measure;
    let best = null;
    for (const a of Object.keys(c.arms)) {
      const r = c.arms[a] && triple(c.arms[a][key]);
      if (!r || r[0] == null) continue;
      if (!best || (M.better === "lower" ? r[0] < best.s : r[0] > best.s)) best = { a, s: r[0] };
    }
    return best && value(c, best.a) ? { arm: best.a, v: value(c, best.a) } : null;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

  function draw() {
    btns.querySelectorAll(".slm-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.arm === arm)));
    const M = MEASURES[measure];
    const rows = Object.keys(D.contestants).map((m) => ({ m, c: D.contestants[m], p: pick(D.contestants[m], M) }));
    const main = rows.filter((r) => r.m !== ASIDE);
    const have = main.filter((r) => r.p);
    const dir = M.better === "lower" ? 1 : -1;
    have.sort((a, b) => dir * (a.p.v[0] - b.p.v[0]));
    const order = have.concat(main.filter((r) => !r.p));
    const aside = rows.filter((r) => r.m === ASIDE);

    const W = Math.max(320, Math.round(svg.parentElement.getBoundingClientRect().width));
    const narrow = W < 560;
    const rowH = narrow ? 44 : 30, top = 40, gap = 22;
    const L = narrow ? 8 : 200, R = narrow ? 8 : 130;
    const n = order.length + aside.length;
    const H = top + n * rowH + gap + 34;
    const x = (v) => L + ((Math.max(M.lo, Math.min(M.hi, v)) - M.lo) / (M.hi - M.lo)) * (W - L - R);
    const yOf = (i) => top + i * rowH + (i >= order.length ? gap : 0) + (narrow ? 30 : rowH / 2);
    let s = "";
    for (const t of M.ticks) {
      s += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${H - 30}"/>`;
      const anchor = x(t) > W - 24 ? "end" : x(t) < 24 ? "start" : "middle";
      s += `<text class="tick" x="${x(t)}" y="${H - 16}" text-anchor="${anchor}">${M.fmt(t)}</text>`;
    }
    s += `<text class="tick" x="${L}" y="${H - 2}">${esc(M.axis)}</text>`;
    if (M.ref != null) {
      const rx = x(M.ref), right = rx > W - R - 120;
      s += `<line class="ref" x1="${rx}" x2="${rx}" y1="${top - 14}" y2="${H - 30}"/>`;
      s += `<text class="tick" x="${right ? rx - 4 : rx + 4}" y="${top - 18}" text-anchor="${right ? "end" : "start"}">${esc(M.refLabel)}</text>`;
    }
    const list = order.concat(aside);
    list.forEach((r, i) => {
      const y = yOf(i), ink = r.m === INK, faint = r.m === ASIDE;
      const cls = ink ? "name is-ink" : faint ? "name is-aside" : "name";
      const nx = narrow ? L : L - 10, ny = narrow ? y - 14 : y + 4;
      s += `<text class="${cls}" x="${nx}" y="${ny}" text-anchor="${narrow ? "start" : "end"}">${faint ? "Inkling, reasoning low" : esc(r.c.label)}</text>`;
      if (!r.p) { s += `<text class="na" x="${narrow ? L : L}" y="${y + 4}">not run for this signal</text>`; return; }
      const [m, lo, hi] = r.p.v;
      const col = ink ? "var(--accent)" : "var(--ink-faint)";
      if (lo != null && hi != null) s += `<line x1="${x(lo)}" x2="${x(hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="5" stroke-opacity=".55"/>`;
      s += `<circle cx="${x(m)}" cy="${y}" r="${ink ? 6.5 : 5}" fill="${col}"/>`;
      const tag = arm === "best" ? ` · ${TAG[r.p.arm] || r.p.arm}` : "";
      const txt = `${M.fmt(m)}${tag}`;
      if (narrow) s += `<text class="val" x="${W - R}" y="${ny}" text-anchor="end">${esc(txt)}</text>`;
      else s += `<text class="val" x="${W - R + 10}" y="${y + 4}">${esc(txt)}</text>`;
    });
    if (aside.length) {
      const ry = top + order.length * rowH + gap / 2;
      s += `<line class="grid" x1="0" x2="${W}" y1="${ry}" y2="${ry}"/>`;
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.innerHTML = s;

    // the sentence and the table: competition rank (ties share a rank), never a rank where none is meaningful
    const me = have.find((r) => r.m === INK);
    const where = arm === "best" ? "each at its best signal on this measure" : `${ARM_NAME[arm].toLowerCase()}, the ${have.length} contestants given this signal`;
    if (!me) live.textContent = "";
    else if (M.better === "none") {
      const shorter = have.filter((r) => r.p.v[0] < me.p.v[0]).length;
      live.textContent = `${M.label}, ${where}: Inkling 2 Mini ${M.fmt(me.p.v[0])}` + (shorter ? `, ${shorter} contestants write shorter.` : ", the shortest of all.");
    } else {
      const better = have.filter((r) => dir * (r.p.v[0] - me.p.v[0]) < 0).length;
      const tied = have.filter((r) => r !== me && r.p.v[0] === me.p.v[0]).length;
      live.textContent = `${M.label}, ${where}: Inkling 2 Mini ${M.fmt(me.p.v[0])}, rank ${better + 1} of ${have.length}` +
        (tied ? `, tied with ${tied} other${tied > 1 ? "s" : ""}` : "") + ` (${M.better} is better).`;
    }
    table.innerHTML = `<caption>${esc(M.label)}, ${esc(where)}</caption><tr><th>Contestant</th><th>Signal</th><th>Value</th><th>95% interval</th></tr>` +
      list.map((r) => `<tr><td>${esc(r.c.label)}${r.m === ASIDE ? " (exploratory, not ranked)" : ""}</td><td>${r.p ? esc(ARM_NAME[r.p.arm] || r.p.arm) : "not run"}</td><td>${r.p ? M.fmt(r.p.v[0]) : ""}</td><td>${r.p && r.p.v[1] != null ? `${M.fmt(r.p.v[1])} to ${M.fmt(r.p.v[2])}` : "not available"}</td></tr>`).join("");
  }
  fig.classList.add("is-live");
  draw();
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 120); });
})();
