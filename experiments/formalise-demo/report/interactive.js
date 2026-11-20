// The proof explorer: follow one checked bag through the four stages of a verification stack
// (formalise the rules, formalise the case, solve and check, prove and explain). Runs the checker port (checker.js)
// on the formal spec (spec.js) in the browser; nothing is sent anywhere.
(() => {
  const root = document.querySelector("#proof-explorer");
  if (!root || !window.BaggageChecker || !window.BAGGAGE_SPEC) return;
  const spec = window.BAGGAGE_SPEC;
  const checker = window.BaggageChecker(spec);
  const $ = (sel) => root.querySelector(sel);
  const ROUTES = { "U.S.": "Within the U.S.", Mexico: "U.S. and Mexico", Cuba: "U.S. and Cuba", Europe: "U.S. and Europe",
    Japan: "U.S. and Japan", Australia: "U.S. and Australia" };
  const EXCEPTIONS = new Set(["L275", "L279", "L103", "L164"]);
  const STAGES = ["Formalise the rules", "Formalise the case", "Solve and check", "Prove and explain"];
  const state = { cls: "Business", region: "Japan", direction: 0, bag: 0, stage: 0, claim: "",
    bags: [{ weight: 62, size: 64 }, { weight: 48, size: 58 }, { weight: 74, size: 90 }] };
  let liveTimer = 0;

  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const money = (v) => (v === null ? "not accepted" : `$${v}`);
  const maxWeight = () => (["Cuba", "Europe", "Australia"].includes(state.region) ? 70 : 100);
  const maxBags = () => (state.region === "Cuba" ? 2 : 3);
  const ordinal = (n) => ["1st", "2nd", "3rd"][n - 1];

  function controls() {
    const cls = $("#pe-class"), route = $("#pe-route"), dir = $("#pe-direction");
    for (const c of spec.slice.classes) cls.append(el("option", { value: c, text: c }));
    for (const [r, label] of Object.entries(ROUTES)) route.append(el("option", { value: r, text: label }));
    cls.value = state.cls; route.value = state.region; dir.value = String(state.direction);
    cls.addEventListener("change", () => { state.cls = cls.value; render(); });
    route.addEventListener("change", () => {
      state.region = route.value;
      if (state.bags.length > maxBags()) state.bags.length = maxBags();
      state.bag = Math.min(state.bag, state.bags.length - 1);
      buildBags(); render();
    });
    dir.addEventListener("change", () => { state.direction = Number(dir.value); render(); });
    $("#pe-add").addEventListener("click", () => {
      if (state.bags.length < maxBags()) state.bags.push({ weight: 45, size: 60 });
      buildBags(); render();
    });
    $("#pe-remove").addEventListener("click", () => {
      if (state.bags.length > 1) state.bags.pop();
      state.bag = Math.min(state.bag, state.bags.length - 1);
      buildBags(); render();
    });
    const tabs = $("#pe-stages");
    STAGES.forEach((name, i) => {
      const b = el("button", { type: "button", role: "tab", id: `pe-tab${i}`, "aria-controls": "pe-panel" },
        el("span", { class: "pe-n", text: String(i + 1) }), el("span", { text: name }));
      b.addEventListener("click", () => { state.stage = i; render(); });
      b.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        state.stage = (state.stage + (e.key === "ArrowRight" ? 1 : 3)) % 4;
        render();
        $(`#pe-tab${state.stage}`).focus();
      });
      tabs.append(b);
    });
    $("#pe-prev").addEventListener("click", () => { state.stage = Math.max(0, state.stage - 1); render(); });
    $("#pe-next").addEventListener("click", () => { state.stage = Math.min(3, state.stage + 1); render(); });
  }

  function slider(label, id, value, min, max, unit, onInput) {
    const out = el("output", { for: id, text: `${value} ${unit}` });
    const input = el("input", { type: "range", id, min: String(min), max: String(max), value: String(value), step: "1" });
    input.addEventListener("input", () => { out.textContent = `${input.value} ${unit}`; onInput(Number(input.value)); });
    return el("label", { class: "pe-slider" }, el("span", { text: label }), input, out);
  }

  function buildBags() {
    const box = $("#pe-bags");
    box.replaceChildren();
    state.bags.forEach((b, i) => {
      b.weight = Math.min(b.weight, maxWeight());
      const pick = el("input", { type: "radio", name: "pe-follow", value: String(i) });
      pick.checked = i === state.bag;
      pick.addEventListener("change", () => { state.bag = i; render(); });
      box.append(el("fieldset", { class: "pe-bagbox" },
        el("legend", {}, el("label", { class: "pe-follow" }, pick, el("span", { text: ` Item ${i + 2}: follow this bag` }))),
        slider("Weight", `pe-w${i}`, b.weight, 30, maxWeight(), "lb", (v) => { b.weight = v; render(); }),
        slider("Size (L+W+H)", `pe-s${i}`, b.size, 45, 115, "in", (v) => { b.size = v; render(); })));
    });
    $("#pe-add").disabled = state.bags.length >= maxBags();
    $("#pe-remove").disabled = state.bags.length <= 1;
  }

  const quote = (id) => {
    const r = checker.rule(id);
    return r.text ? `“${r.text.replace(/^\* /, "")}”` : r.formal;
  };
  const ruleTag = (id) => {
    const r = checker.rule(id);
    if (EXCEPTIONS.has(id)) return "exception";
    if (id.startsWith("X-")) return r.role === "unstated" ? "not in the text" : r.role;
    return r.role;
  };
  function ruleRow(id, extra) {
    const special = EXCEPTIONS.has(id) || id.startsWith("X-");
    return el("li", { class: `pe-rule${special ? " is-exception" : ""}` },
      el("span", { class: "pe-id", text: id }), el("span", { class: "pe-tag", text: ruleTag(id) }),
      el("span", { class: "pe-quote", text: extra || quote(id) }));
  }
  const formal = (id) => {
    const r = checker.rule(id);
    const out = { id: r.id, kind: r.kind, role: r.role };
    if (r.when) out.when = r.when;
    if (r.fee !== undefined) out.fee = r.fee;
    if (r.overrides) out.overrides = r.overrides;
    if (!r.when && r.formal) out.formal = r.formal;
    return JSON.stringify(out);
  };
  const check = (ok, text) => el("li", { class: ok ? "pe-ok" : "pe-bad" }, el("span", { class: "pe-mark", text: ok ? "✓" : "✗" }), el("span", { text }));

  function stageRules(steps, sizeCount) {
    const used = [...new Set(steps.flatMap((s) => s.rules).filter((id) => id !== "L270"))];
    return [
      el("p", { text: "The rulebook's sentences this bag needs, each with the formal rule built from it: what kind of rule it is, when it applies, what it decides, and what it overrides." }),
      el("ul", { class: "pe-rules pe-formal" }, ...used.map((id) => el("li", {}, ruleRow(id), el("code", { text: formal(id) })))),
      el("p", { class: "pe-small", text: `Finding them by hand means reading the candidates: ${sizeCount} lines in the sections this bag touches.` }),
    ];
  }

  function stageCase(bag, position) {
    const wmax = maxWeight();
    return [
      el("p", { text: "The question's facts, written as typed values the spec can check. In the study the model writes this step in its trace; the spec then disciplines it: a value outside a type, or outside the slice, is rejected rather than guessed." }),
      el("pre", { class: "pe-typed", text: [
        `class     : FareClass = "${state.cls}"`,
        `route     : Region    = "${state.region}"   direction = ${state.direction ? "to the U.S." : "from the U.S."}`,
        `item ${bag.item}    : Bag       = { size: ${bag.size} in, weight: ${bag.weight} lb }`,
        `position  : 1..${maxBags()}      = ${position} (decided in stage 3)`,
      ].join("\n") }),
      el("ul", { class: "pe-checks" },
        check(spec.slice.classes.includes(state.cls), "FareClass is one of the six classes"),
        check(spec.slice.regions.includes(state.region), "Region is one of the slice's six regions"),
        check(bag.size >= 45 && bag.size <= 115, "size is inside the slice's boundary (up to 115 in)"),
        check(bag.weight <= wmax, `weight is inside the boundary for this route (up to ${wmax} lb)`),
        check(state.bags.length <= maxBags(), `at most ${maxBags()} checked bags on this route`)),
    ];
  }

  function stageSolve(proof, steps) {
    const order = proof.steps.find((s) => s.step === "order");
    const [base, weight, size, total] = ["base", "weight", "size", "bag"].map((k) => steps.find((s) => s.step === k));
    const free = weight.inputs.complimentary;
    return [
      el("p", { text: "The solver evaluates the spec on the typed facts. It searches the one choice the text leaves open (which bag counts as first), looks up exactly one rule per fee, and checks every constraint. It returns either a certificate or the constraint that breaks." }),
      el("ol", { class: "pe-solve" },
        el("li", { text: `Order (X-ORDER): gains if free = [${order.inputs.gain_if_free.join(", ")}] → items ${order.value.join(", ")} take positions 1..${order.value.length}; this bag is ${ordinal(base.position)}.` }),
        el("li", { text: `Base fee: the ${ordinal(base.position)}-bag line for ${state.cls}, ${state.region} → ${money(base.value)} (${base.rules.join(", ")}). ${free ? "Base $0, so the bag is complimentary (X-COMP)." : "Charged bag, so not complimentary."}` }),
        el("li", { text: `Weight: ${weight.inputs.weight} lb against a ${weight.inputs.allowance} lb allowance → ${money(weight.value)} (${weight.rules.join(", ")}).` }),
        el("li", { text: `Size: ${size.inputs.size} in against 62 in → ${money(size.value)} (${size.rules.filter((r) => r !== "L270").join(", ")}).` }),
        el("li", { text: `Combine (L283): ${money(base.value)} + max(${money(weight.value)}, ${money(size.value)}) = ${money(total.value)}.` })),
      el("ul", { class: "pe-checks" },
        check(base.rules.length >= 1, "exactly one base-fee rule matched (the spec lint guarantees it for every input)"),
        check(weight.value !== null, "the weight rule accepts the bag"),
        check(size.value !== null, "the size rule accepts the bag")),
    ];
  }

  function stageProve(steps) {
    const [base, weight, size, total] = ["base", "weight", "size", "bag"].map((k) => steps.find((s) => s.step === k));
    const claim = /^\s*\$?\d+\s*$/.test(state.claim) ? Number(state.claim.replace(/[^\d]/g, "")) : null;
    const input = el("input", { type: "text", id: "pe-claim", inputmode: "numeric", autocomplete: "off", value: state.claim, placeholder: "e.g. 230" });
    input.addEventListener("input", () => { state.claim = input.value; renderStage(); });
    const verdict = claim === null ? null : claim === total.value
      ? el("p", { class: "pe-verdict is-ok", text: `Verified: $${claim} matches every step.` })
      : el("p", { class: "pe-verdict is-bad", text: `Fails at the bag fee: claimed $${claim}, the proof gives ${money(total.value)} = ${money(base.value)} + max(${money(weight.value)}, ${money(size.value)}). Check which of the three lines above the answer used instead.` });
    return [
      el("p", { text: "The same steps, written for a person to read, challenge and sign off. Reviewing means checking each line against its quoted source, not redoing the lookup." }),
      el("ol", { class: "pe-explain" },
        el("li", {}, `Base fee ${money(base.value)}: `, ...base.rules.map((id) => el("q", { text: quote(id).replace(/[“”]/g, "") })), ` (${base.rules.join(", ")})`),
        el("li", {}, `Overweight ${money(weight.value)}: ${weight.inputs.weight} lb against ${weight.inputs.allowance} lb (${weight.rules.join(", ")}).`),
        el("li", {}, `Oversize ${money(size.value)}: ${size.inputs.size} in against 62 in (${size.rules.filter((r) => r !== "L270").join(", ")}).`),
        el("li", {}, `Bag fee ${money(total.value)}: "the higher fee between oversize and overweight applies, in addition to the checked bag fee" (L283).`)),
      el("p", { class: "pe-stamp", text: total.value === null ? "Not verified: a rule refuses this bag" : `Verified · ${base.rules.length + weight.rules.length + size.rules.length} rules checked · bag fee ${money(total.value)}` }),
      el("label", { class: "pe-claim" }, el("span", { text: "Check an answer for this bag's fee: $" }), input),
      verdict,
    ];
  }

  function sectionSize(position) {
    const n = (kind, f) => spec.rules.filter((r) => r.kind === kind && f(r)).length;
    return n("base_fee", (r) => r.when.position === position) + n("weight_fee", () => true) + n("size_fee", () => true) + 8;
  }

  let current = null;
  function renderStage() {
    if (!current) return;
    const { proof, steps, bag } = current;
    const base = steps.find((s) => s.step === "base");
    const panel = $("#pe-panel");
    const body = [stageRules(steps, sectionSize(base.position)), stageCase(bag, base.position), stageSolve(proof, steps), stageProve(steps)][state.stage];
    const focused = document.activeElement && document.activeElement.id === "pe-claim";
    panel.replaceChildren(el("h4", { text: `Stage ${state.stage + 1}: ${STAGES[state.stage]}` }), ...body.filter((n) => n != null));
    if (focused) { const c = $("#pe-claim"); c.focus(); c.setSelectionRange?.(c.value.length, c.value.length); }
    STAGES.forEach((_, i) => {
      const t = $(`#pe-tab${i}`);
      t.setAttribute("aria-selected", String(i === state.stage));
      t.tabIndex = i === state.stage ? 0 : -1;
    });
    panel.setAttribute("aria-labelledby", `pe-tab${state.stage}`);
    $("#pe-prev").disabled = state.stage === 0;
    $("#pe-next").disabled = state.stage === 3;
  }

  function render() {
    const bags = state.bags.map((b, i) => ({ item: i + 2, size: b.size, weight: b.weight }));
    const proof = checker.evaluateChecked(state.cls, state.region, state.direction, 0, bags);
    const bag = bags[state.bag];
    const steps = proof.steps.filter((s) => s.item === bag.item);
    current = { proof, steps, bag };
    $("#pe-total").textContent = proof.total === null ? "A bag over 70 lb is not accepted on this route"
      : `Checked-bag fees for all bags: $${proof.total}`;
    const artifact = { inputs: { class: state.cls, route: ROUTES[state.region], direction: state.direction ? "to the U.S." : "from the U.S.", bags },
      steps: proof.steps.filter((s) => s.step !== "total").map(({ step, item, rules, value }) => ({ step, item, rules, value })), fees: proof.total };
    $("#pe-json").textContent = JSON.stringify(artifact, null, 1);
    renderStage();
    const fees = proof.total === null ? "a bag is not accepted" : `fees $${proof.total}`;
    clearTimeout(liveTimer);
    liveTimer = setTimeout(() => { $("#pe-live").textContent = `${fees}; showing stage ${state.stage + 1} for item ${bag.item}.`; }, 400);
  }

  controls();
  buildBags();
  render();
  root.classList.add("is-live");
})();
