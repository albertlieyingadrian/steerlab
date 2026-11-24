// Word explorer: six confirmation AI replies, each word tinted by how much deleting it alone lowers a detector's
// AI score (orange) or raises it (blue). The slider strikes out that detector's top words, the way the deletion
// test removed them. Drawn from data.js (window.ATTR), built by make_figures.py from results/.
(function () {
  const D = window.ATTR;
  const root = document.getElementById("word-explorer");
  if (!D || !root) return;
  const WHO = { "9b": "Our 9B", "1.7b": "Our 1.7B", voi: "voi-380M", desklib: "Desklib", writer: "The writer's SAE features" };
  const st = { text: 0, who: "9b", del: 0 };
  const label = (r) => `${r.arm === "original" ? "Plain reply" : "De-slopped reply"}: ${r.key.split("/")[2]}`;
  root.innerHTML = `
    <div class="we-controls">
      <label>Text <select data-k="text">${D.map((r, i) => `<option value="${i}">${label(r)}</option>`).join("")}</select></label>
      <label>Read by <select data-k="who">${Object.entries(WHO).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></label>
      <label class="we-slider">Strike out the top <input type="range" min="0" max="50" step="5" value="0"> <output>0%</output></label>
    </div>
    <div class="we-text" aria-live="polite"></div>
    <p class="we-read small"></p>`;
  const box = root.querySelector(".we-text");
  const read = root.querySelector(".we-read");
  const out = root.querySelector("output");
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  function draw() {
    const r = D[st.text];
    const v = st.who === "writer" ? r.writer || [] : r.attr[st.who];
    const abs = v.map(Math.abs).sort((a, b) => a - b);
    const scale = abs[Math.floor(abs.length * 0.95)] || 1;
    const order = v.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]);
    const k = Math.round((st.del / 100) * v.length);
    const struck = new Set(order.slice(0, k).map((p) => p[1]));
    const acc = css("--accent") || "#c9561c";
    const ctl = css("--control") || "#2f6db0";
    const cps = Array.from(r.text); // spans are Python offsets: code points, not UTF-16 units
    const cut = (a, b) => cps.slice(a, b).join("");
    let html = "";
    let pos = 0;
    r.spans.forEach(([s, e], i) => {
      html += esc(cut(pos, s)).replace(/\n/g, "<br>");
      const x = Math.max(-1, Math.min(1, v[i] / scale));
      const col = x > 0 ? acc : ctl;
      const a = Math.round(Math.abs(x) * 60);
      const style = a > 2 ? ` style="background:color-mix(in srgb, ${col} ${a}%, transparent)"` : "";
      const cls = struck.has(i) ? ' class="we-struck"' : "";
      html += `<span${cls}${style}>${esc(cut(s, e))}</span>`;
      pos = e;
    });
    box.innerHTML = html + esc(cut(pos, cps.length));
    const pg = r.pangram || {};
    const word = (x) => x || "not sent";
    read.textContent =
      `Pangram on this reply: ${word(pg.orig)} as written; ${word(pg["top10:9b"])} with the 9B's top 10% of words deleted, ` +
      `${word(pg["top25:9b"])} with its top 25% deleted; ${word(pg["rand25:0"])} with a random 25% deleted.` +
      (st.who === "writer" ? " Writer view: summed activation of the 20 SAE features that best separate its replies from human text (orange only)." : "");
  }
  root.querySelectorAll("select").forEach((el) =>
    el.addEventListener("change", () => {
      st[el.dataset.k] = el.dataset.k === "text" ? +el.value : el.value;
      draw();
    }),
  );
  root.querySelector("input").addEventListener("input", (e) => {
    st.del = +e.target.value;
    out.textContent = `${st.del}%`;
    draw();
  });
  draw();
})();
