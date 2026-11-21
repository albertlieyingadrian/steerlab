// The brand explorer: one page through the four stages of a verification stack (formalise the rules, formalise the
// case, solve and check, prove and explain). Runs the checker port (checker.js) on the extractor's facts for each
// version of the page (demo.js) in the browser; nothing is sent anywhere.
(() => {
  const root = document.querySelector("#brand-explorer");
  if (!root || !window.BrandChecker || !window.BRAND_DEMO) return;
  const { spec, variants } = window.BRAND_DEMO;
  const checker = window.BrandChecker(spec);
  const $ = (s) => root.querySelector(s);
  const STAGES = ["Formalise the rules", "Formalise the case", "Solve and check", "Prove and explain"];
  const PAL = new Map(spec.tokens.palette.map((p) => [p.hex, p]));
  const state = { v: 0, el: variants[0].focus, stage: 0 };
  let proof = null, live = 0;

  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const page = () => variants[state.v].page;
  const facts = (id) => page().elements.find((e) => e.id === id);
  const lineText = (id) => checker.line(id) || "";
  const mark = (ok) => el("span", { class: "be-mark", text: ok ? "✓" : "✗" });
  const fmt = (v) => (Array.isArray(v) ? v.join(" ") : v === null || v === undefined ? "none" : String(v));
  const describe = (e) => `${e.id} ${e.role}${e.text ? ` "${e.text.length > 48 ? e.text.slice(0, 45) + "..." : e.text}"` : ""}`;

  // ---------------------------------------------------------------- controls
  function controls() {
    const sel = $("#be-variant");
    const groups = { base: "On-brand", violation: "One planted violation", trap: "On-brand by an exception" };
    for (const [g, label] of Object.entries(groups)) {
      const og = el("optgroup", { label });
      variants.forEach((v, i) => { if (v.group === g) og.append(el("option", { value: String(i), text: v.label })); });
      sel.append(og);
    }
    sel.addEventListener("change", () => { state.v = Number(sel.value); state.el = variants[state.v].focus; load(); });
    $("#be-element").addEventListener("change", (e) => { state.el = e.target.value; render(); });
    const tabs = $("#be-stages");
    STAGES.forEach((name, i) => {
      const b = el("button", { type: "button", role: "tab", id: `be-tab${i}`, "aria-controls": "be-panel" },
        el("span", { class: "be-n", text: String(i + 1) }), el("span", { text: name }));
      b.addEventListener("click", () => { state.stage = i; render(); });
      b.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        state.stage = (state.stage + (e.key === "ArrowRight" ? 1 : 3)) % 4;
        render();
        $(`#be-tab${state.stage}`).focus();
      });
      tabs.append(b);
    });
    $("#be-prev").addEventListener("click", () => { state.stage = Math.max(0, state.stage - 1); render(); });
    $("#be-next").addEventListener("click", () => { state.stage = Math.min(3, state.stage + 1); render(); });
    new ResizeObserver(fit).observe($(".be-frame-wrap"));
  }

  // ---------------------------------------------------------------- the page itself
  const HIGHLIGHT = "<style>[data-be-sel]{outline:4px solid #d26f2a!important;outline-offset:3px}" +
    "[data-be-fail]{box-shadow:0 0 0 3px rgba(210,111,42,.45)}*{cursor:pointer}</style>";
  function pickable(node) {
    for (let t = node && node.closest("[data-id]"); t; t = t.parentElement && t.parentElement.closest("[data-id]")) {
      const f = facts(t.dataset.id);
      if (f && (f.role !== "container" || f.bg || f.border)) return t.dataset.id;
    }
    return null;
  }
  function fillElements() {
    const box = $("#be-element");
    const fails = new Set(proof.failures.map((f) => f.element));
    box.replaceChildren(...page().elements.filter((e) => checksFor(e.id).length && (e.role !== "container" || e.bg || e.border))
      .map((e) => el("option", { value: e.id, text: `${fails.has(e.id) ? "✗ " : ""}${describe(e)}` })));
  }
  function load() {
    proof = checker.check(page());
    fillElements();
    const frame = $("#be-frame");
    frame.onload = () => {
      const doc = frame.contentDocument;
      if (!doc) return;
      doc.querySelectorAll("a, button, input").forEach((n) => { n.tabIndex = -1; });  // keyboard users pick from the list
      doc.addEventListener("click", (e) => {
        e.preventDefault();
        const id = pickable(e.target);
        if (id) { state.el = id; render(); }
      });
      fit();
      render();
    };
    frame.srcdoc = variants[state.v].html.replace("</head>", HIGHLIGHT + "</head>");
  }
  function fit() {
    const wrap = $(".be-frame-wrap"), frame = $("#be-frame");
    const doc = frame.contentDocument;
    const h = doc && doc.documentElement ? Math.min(doc.documentElement.scrollHeight, 2600) : 1800;
    const s = wrap.clientWidth / 1280;
    frame.style.height = `${h}px`;
    frame.style.transform = `scale(${s})`;
    $(".be-frame-inner").style.height = `${h * s}px`;
  }
  function paintFrame() {
    const doc = $("#be-frame").contentDocument;
    if (!doc || !doc.body) return;
    doc.querySelectorAll("[data-be-sel],[data-be-fail]").forEach((n) => { n.removeAttribute("data-be-sel"); n.removeAttribute("data-be-fail"); });
    for (const f of proof.failures) { const n = doc.querySelector(`[data-id="${f.element}"]`); if (n) n.setAttribute("data-be-fail", ""); }
    const sel = doc.querySelector(`[data-id="${state.el}"]`);
    if (sel) {
      sel.setAttribute("data-be-sel", "");
      const wrap = $(".be-frame-wrap"), s = wrap.clientWidth / 1280;
      const top = sel.getBoundingClientRect().top + (doc.documentElement.scrollTop || 0);
      wrap.scrollTo({ top: Math.max(0, top * s - 60), behavior: "smooth" });
    }
  }

  // ---------------------------------------------------------------- the four stages
  const checksFor = (id) => proof.checks.filter((c) => c.element === id);
  function ruleRow(rid) {
    const r = checker.rule(rid);
    const tag = r.overrides ? `exception to ${r.overrides}` : r.settled === "decided" ? "a reading we chose" : r.kind.replace(/_/g, " ");
    const li = el("li", { class: `be-rule${r.overrides ? " is-exception" : ""}${r.settled === "decided" ? " is-decided" : ""}` },
      el("span", { class: "be-id", text: rid }), el("span", { class: "be-tag", text: tag }),
      el("span", { class: "be-quote", text: r.lines.map((l) => `${l}: “${lineText(l)}”`).join("  ") }),
      el("code", { text: `formal: ${r.formal}` }));
    if (r.settled === "decided") li.append(el("span", { class: "be-why", text: `Why a reading was needed: ${r.why}` }));
    return li;
  }
  function stageRules(e) {
    const ids = [...new Set(checksFor(e.id).map((c) => c.rule))];
    if (!ids.length) return [el("p", { text: `${describe(e)} is a layout container with nothing a rule reads. Click a heading, a paragraph, a link or a button on the page.` })];
    return [
      el("p", { text: `The brand-book lines that apply to ${describe(e)}, each with the formal rule built from it. An orange edge marks an exception; a dashed edge marks a reading the text leaves open.` }),
      el("ul", { class: "be-rules" }, ...ids.map(ruleRow)),
      el("p", { class: "be-small", text: `The whole spec: ${spec.rules.length} rules (${spec.rules.filter((r) => r.overrides).length} exceptions) from the GOV.UK pages and the WCAG lines they point to; ${spec.residue.length} more normative lines are left to judgment and never checked.` }),
    ];
  }
  function typed(e) {
    const rows = [];
    const add = (k, type, v, note) => rows.push(el("tr", {}, el("th", { scope: "row", text: k }), el("td", { text: type }), el("td", { text: fmt(v) }), el("td", { text: note || "" })));
    add("role", "Role", e.role, "from its class name (spec components)");
    if (e.text) add("text", "String", e.text, "");
    for (const [k, label] of [["color", "text colour"], ["eff_bg", "behind the text"], ["bg", "background"], ["border", "border"]]) {
      if (!e[k] || (k === "color" && !e.text) || (k === "eff_bg" && !e.text)) continue;
      const tok = PAL.get(e[k]);
      add(label, "Colour", e[k], tok ? `token: ${tok.name} (${tok.line})` : "not a palette token");
    }
    if (e.text) {
      const point = spec.tokens.type_scale[String(e.size)];
      add("font", "Font name", e.font, "");
      add("size", "px", e.size, point ? `type-scale point (${point.line})` : "not a type-scale point");
      add("line height", "px", e.line_height, "");
      add("weight", "Integer", e.weight, e.weight >= 700 ? "bold" : "");
    }
    add("margin", "px × 4", e.margin, "");
    add("padding", "px × 4", e.padding, "");
    if (e.role.includes("link")) add("underline", "Boolean", e.underline, `in ${e.region}`);
    if (e.disabled) add("disabled", "Boolean", true, "an inactive component");
    if (e.form) add("left edge", "px", e.x, `the form's left edge is ${e.form_x}`);
    return rows;
  }
  function stageCase(e) {
    const p = page();
    return [
      el("p", { text: "The page as typed facts, read from the rendered page by the extractor (computed styles, not the page's CSS). The models in this study were given these same facts as text." }),
      el("p", { class: "be-small", text: `Page: domain ${p.domain}; page type ${p.page_type}; ${p.elements.length} elements.` }),
      el("div", { class: "table-wrap" }, el("table", { class: "be-typed" },
        el("thead", {}, el("tr", {}, ...["Fact", "Type", "Value", "Typed check"].map((h) => el("th", { scope: "col", text: h })))),
        el("tbody", {}, ...typed(e)))),
    ];
  }
  function stageSolve(e) {
    const cs = checksFor(e.id);
    return [
      el("p", { text: "The checker evaluates every rule that applies to every element. Where an exception applies, it decides instead of the rule it overrides." }),
      cs.length ? el("ul", { class: "be-checks" }, ...cs.map((c) => el("li", { class: c.pass ? "be-ok" : "be-bad" }, mark(c.pass),
        el("span", { text: `${c.rule} · ${c.property}: ${fmt(c.value)} (needs ${fmt(c.expected)})${c.why ? " · " + c.why : ""}` })))) :
        el("p", { text: "No rule reads this element." }),
      el("p", { class: "be-small", text: `On the whole page: ${proof.checks.length} checks, ${proof.failures.length} failing.` }),
    ];
  }
  function stageProve() {
    const v = variants[state.v];
    const out = [el("p", { text: "The proof artifact, written for a person to read and sign off: each failure with its rule, the line it quotes, the element and the value. A reviewer checks lines instead of redoing the audit." })];
    if (proof.failures.length) {
      out.push(el("ol", { class: "be-explain" }, ...proof.failures.map((f) => {
        const go = el("button", { type: "button", class: "be-go", text: `Show ${f.element}` });
        go.addEventListener("click", () => { state.el = f.element; state.stage = 0; render(); });
        return el("li", {}, el("strong", { text: `${f.rule} fails on ${f.element} (${f.role}). ` }),
          el("span", { text: `${f.property}: ${fmt(f.value)}, needs ${fmt(f.expected)}. ` }),
          el("q", { text: f.quote }), el("span", { text: ` (${f.lines.join(", ")}) ` }), go);
      })));
    }
    const ex = proof.checks.filter((c) => c.overrides && c.pass);
    if (ex.length) out.push(el("p", { class: "be-small", text: `Exceptions that applied on this page: ${[...new Set(ex.map((c) => `${c.rule} (${c.lines.join(", ")})`))].join("; ")}.` }));
    out.push(el("p", { class: "be-stamp", text: proof.failures.length ? `Not verified · ${proof.failures.length} failing check${proof.failures.length > 1 ? "s" : ""} of ${proof.checks.length}` : `Verified on-brand · ${proof.checks.length} checks · 0 failures` }));
    if (v.planted.length) out.push(el("p", { class: "be-small", text: `Planted in this version: ${v.planted.map((p) => `${p.rule} on ${p.element}`).join(", ")}.` }));
    return out;
  }

  function render() {
    if (!proof) return;
    const e = facts(state.el) || facts(variants[state.v].focus);
    state.el = e.id;
    const body = [stageRules, stageCase, stageSolve, stageProve][state.stage](e);
    $("#be-panel").replaceChildren(el("h4", { text: `Stage ${state.stage + 1}: ${STAGES[state.stage]} · ${describe(e)}` }), ...body);
    STAGES.forEach((_, i) => {
      const t = $(`#be-tab${i}`);
      t.setAttribute("aria-selected", String(i === state.stage));
      t.tabIndex = i === state.stage ? 0 : -1;
    });
    $("#be-panel").setAttribute("aria-labelledby", `be-tab${state.stage}`);
    if ([...$("#be-element").options].some((o) => o.value === e.id)) $("#be-element").value = e.id;
    $("#be-prev").disabled = state.stage === 0;
    $("#be-next").disabled = state.stage === 3;
    const verdict = $("#be-verdict");
    verdict.textContent = proof.failures.length ? `Fails: ${proof.failures.map((f) => `${f.rule} on ${f.element}`).join(", ")}` : "Passes every check";
    verdict.className = proof.failures.length ? "is-bad" : "is-ok";
    const art = { page: page().domain, verdict: proof.verdict, checks: proof.checks.length,
      failures: proof.failures.map(({ rule, lines, element, property, value, expected, why }) => ({ rule, lines, element, property, value, expected, why })) };
    $("#be-json").textContent = JSON.stringify(art, null, 1);
    paintFrame();
    clearTimeout(live);
    live = setTimeout(() => { $("#be-live").textContent = `${verdict.textContent}. Stage ${state.stage + 1} for ${e.id}.`; }, 400);
  }

  controls();
  load();
  root.classList.add("is-live");
})();
