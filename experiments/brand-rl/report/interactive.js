// The explorer: one held-out brief, the page each arm wrote for it, and what every ruler said about that page.
// Reads window.EP6 (data.js, written by make_demo.py from the held-out samples); nothing is sent anywhere.
(() => {
  const root = document.querySelector("#rl-explorer");
  if (!root || !window.EP6) return;
  const D = window.EP6;
  const $ = (s) => root.querySelector(s);
  const state = { brief: D.briefs[0].id, arm: D.arms.some((a) => a.key === "check_s0") ? "check_s0" : D.arms[0].key };
  const fmt = (x, d = 2) => (x === null || x === undefined ? "n/a" : Number(x).toFixed(d));
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };

  const briefSel = $("#rx-brief"), armSel = $("#rx-arm");
  for (const b of D.briefs) briefSel.append(el("option", { value: b.id, text: `${b.type}: ${b.service}` }));
  for (const a of D.arms) armSel.append(el("option", { value: a.key, text: a.label }));
  armSel.value = state.arm;

  function itemText(it) {
    const kind = it.kind === "button" ? `${it.variant} button` : it.kind.replace("_", " ");
    return `${kind}: “${it.text}”`;
  }

  function draw() {
    const b = D.briefs.find((x) => x.id === state.brief);
    const p = D.pages[`${state.arm}|${state.brief}`];
    const img = $("#rx-img");
    if (p.img) { img.src = p.img; img.alt = `First screen of the page this arm wrote for “${b.service}”`; img.hidden = false; }
    else { img.hidden = true; }
    $("#rx-noimg").hidden = !!p.img;
    $("#rx-items").replaceChildren(...b.items.map((it) => el("li", { text: itemText(it) })));
    $("#rx-domain").textContent = b.domain;
    const row = (k, v) => el("li", {}, el("span", { class: "k", text: k }), el("span", { class: "v", text: v }));
    $("#rx-stats").replaceChildren(
      row("Rewarded rules passed", fmt(p.check_reward)),
      row("Brief texts, strict", fmt(p.fidelity_strict)),
      row("Markup slop P(AI)", fmt(p.slop_p)),
      row("Screenshot slop P(AI)", fmt(p.shot_p)),
      row("Structure similarity to the exemplar", fmt(p.copy_ex)),
      row("Tokens", String(p.tokens ?? "n/a")),
    );
    const flag = p.hack ? el("span", { class: "rx-flag", text: "flagged as a hack" }) : null;
    $("#rx-verdict").replaceChildren(
      ...[el("span", { text: !p.valid ? "No page rendered" : p.truncated ? "Cut off at the token cap (reward 0)" : "Rendered" }),
        flag].filter(Boolean),
    );
    const fails = D.families.filter((f) => p.fam[f.id] && p.fam[f.id][1] > 0).map((f) => f.id);
    $("#rx-failsum").textContent = !p.valid ? "" : fails.length
      ? `Fails ${fails.length} of ${D.families.length} rules: ${fails.join(", ")}`
      : `Passes all ${D.families.length} rules`;
    const order = [...D.families].sort((a, b) => (fails.includes(b.id) - fails.includes(a.id)) || (a.held_out - b.held_out));
    $("#rx-fams").replaceChildren(...order.map((f) => {
      const v = p.fam[f.id];
      let status = "n/a", cls = "na";
      if (v) {
        const [n, k] = v;
        if (n === 0) { status = "passes: nothing to check"; cls = "vac"; }
        else if (k === 0) { status = `passes ${n} check${n === 1 ? "" : "s"}`; cls = "ok"; }
        else { status = `fails ${k} of ${n}`; cls = "bad"; }
      }
      return el("li", { class: `rx-fam ${cls}${f.held_out ? " held" : ""}` },
        el("span", { class: "id", text: f.id + (f.held_out ? " · held out" : "") }),
        el("span", { class: "st", text: status }),
        el("span", { class: "q", text: f.line }));
    }));
  }

  briefSel.addEventListener("change", () => { state.brief = briefSel.value; draw(); });
  armSel.addEventListener("change", () => { state.arm = armSel.value; draw(); });
  draw();
  root.classList.add("is-live");
})();
