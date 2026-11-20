// The proof explorer: change a passenger's fare, route and bags, and watch the rule trace and its proof artifact
// update. Runs the checker port (checker.js) on the formal spec (spec.js) in the browser; nothing is sent anywhere.
(() => {
  const root = document.querySelector("#proof-explorer");
  if (!root || !window.BaggageChecker || !window.BAGGAGE_SPEC) return;
  const spec = window.BAGGAGE_SPEC;
  const checker = window.BaggageChecker(spec);
  const $ = (sel) => root.querySelector(sel);
  const ROUTES = { "U.S.": "Within the U.S.", Mexico: "U.S. and Mexico", Cuba: "U.S. and Cuba", Europe: "U.S. and Europe",
    Japan: "U.S. and Japan", Australia: "U.S. and Australia" };
  const EXCEPTIONS = new Set(["L275", "L279", "L103", "L164"]);
  const state = { cls: "Business", region: "Japan", direction: 0,
    bags: [{ weight: 62, size: 64 }, { weight: 48, size: 58 }, { weight: 74, size: 90 }] };

  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const money = (v) => (v === null ? "not accepted" : `$${v}`);

  function controls() {
    const cls = $("#pe-class"), route = $("#pe-route"), dir = $("#pe-direction");
    for (const c of spec.slice.classes) cls.append(el("option", { value: c, text: c }));
    for (const [r, label] of Object.entries(ROUTES)) route.append(el("option", { value: r, text: label }));
    cls.value = state.cls; route.value = state.region; dir.value = String(state.direction);
    cls.addEventListener("change", () => { state.cls = cls.value; render(); });
    route.addEventListener("change", () => {
      state.region = route.value;
      if (state.region === "Cuba" && state.bags.length > 2) state.bags.length = 2;  // no third-bag line for Cuba
      buildBags(); render();
    });
    dir.addEventListener("change", () => { state.direction = Number(dir.value); render(); });
    $("#pe-add").addEventListener("click", () => {
      if (state.bags.length < (state.region === "Cuba" ? 2 : 3)) state.bags.push({ weight: 45, size: 60 });
      buildBags(); render();
    });
    $("#pe-remove").addEventListener("click", () => {
      if (state.bags.length > 1) state.bags.pop();
      buildBags(); render();
    });
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
      const maxWeight = ["Cuba", "Europe", "Australia"].includes(state.region) ? 70 : 100;  // heavier is refused there
      b.weight = Math.min(b.weight, maxWeight);
      box.append(el("fieldset", { class: "pe-bagbox" }, el("legend", { text: `Item ${i + 2} (checked)` }),
        slider("Weight", `pe-w${i}`, b.weight, 30, maxWeight, "lb", (v) => { b.weight = v; render(); }),
        slider("Size (L+W+H)", `pe-s${i}`, b.size, 45, 115, "in", (v) => { b.size = v; render(); })));
    });
    $("#pe-add").disabled = state.bags.length >= (state.region === "Cuba" ? 2 : 3);
    $("#pe-remove").disabled = state.bags.length <= 1;
  }

  let seen = new Set();
  let liveTimer = 0;

  function ruleRow(id, value) {
    const r = checker.rule(id);
    const exception = EXCEPTIONS.has(id), unstated = id.startsWith("X-");
    const tag = exception ? "exception" : unstated ? (r.role === "unstated" ? "not in the text" : "interpretation") : r.role;
    const repeat = seen.has(id);
    seen.add(id);
    const quote = repeat ? "the same line as above" : r.text ? `“${r.text.replace(/^\* /, "")}”` : r.formal;
    return el("li", { class: `pe-rule${exception || unstated ? " is-exception" : ""}` },
      el("span", { class: "pe-id", text: id }), el("span", { class: "pe-tag", text: tag }),
      el("span", { class: repeat ? "pe-quote is-repeat" : "pe-quote", text: quote }), value === undefined ? null : el("span", { class: "pe-val", text: value }));
  }

  function render() {
    const bags = state.bags.map((b, i) => ({ item: i + 2, size: b.size, weight: b.weight }));
    const proof = checker.evaluateChecked(state.cls, state.region, state.direction, 0, bags);
    const list = $("#pe-trace");
    list.replaceChildren();
    seen = new Set();
    const order = proof.steps.find((s) => s.step === "order");
    const moved = order.value.some((item, k) => item !== k + 2);
    if (moved) {
      list.append(el("li", { class: "pe-step" }, el("p", { class: "pe-head", text: `Order: item ${order.value.join(", item ")} count as the ${["1st", "2nd", "3rd"].slice(0, order.value.length).join(", ").replace(/, ([^,]*)$/, " and $1")} checked bag${order.value.length > 1 ? "s" : ""}` }),
        el("ul", { class: "pe-rules" }, ruleRow("X-ORDER"))));
    }
    for (const s of proof.steps) {
      if (!["base", "weight", "size", "bag"].includes(s.step)) continue;
      const lines = s.rules.filter((id) => id !== "L270" || s.step !== "size");
      const what = { base: `Item ${s.item}, checked bag ${s.position}: base fee`, weight: `Item ${s.item}: ${s.inputs.weight} lb against a ${s.inputs.allowance} lb allowance${s.inputs.complimentary ? " (a free bag)" : ""}`,
        size: `Item ${s.item}: ${s.inputs.size} in against 62 in`, bag: `Item ${s.item}: base + the higher of overweight and oversize` }[s.step];
      const value = s.step === "bag" ? `$${s.inputs.base} + max(${money(s.inputs.overweight)}, ${money(s.inputs.oversize)}) = ${money(s.value)}` : money(s.value);
      list.append(el("li", { class: `pe-step pe-${s.step}` }, el("p", { class: "pe-head" }, el("span", { text: what }), el("strong", { text: value })),
        el("ul", { class: "pe-rules" }, ...lines.map((id) => ruleRow(id)))));
    }
    $("#pe-total").textContent = proof.total === null ? "A bag over 70 lb is not accepted on this route" : `Checked-bag fees: $${proof.total}`;
    const artifact = { inputs: { class: state.cls, route: ROUTES[state.region], direction: state.direction ? "to the U.S." : "from the U.S.", bags },
      steps: proof.steps.filter((s) => s.step !== "total").map(({ step, item, rules, value }) => ({ step, item, rules, value })), fees: proof.total };
    $("#pe-json").textContent = JSON.stringify(artifact, null, 1);
    const special = proof.steps.filter((s) => s.rules.some((r) => EXCEPTIONS.has(r) || r.startsWith("X-"))).length;
    const fees = proof.total === null ? "a bag is not accepted" : `fees $${proof.total}`;
    clearTimeout(liveTimer);
    liveTimer = setTimeout(() => { $("#pe-live").textContent = `${fees}; ${special} steps use an exception or an unstated rule.`; }, 400);
  }

  controls();
  buildBags();
  render();
  root.classList.add("is-live");
})();
