// The baggage checker, ported line for line from ../checker.py so the note's interactive runs in the browser.
// ruler.py runs both on the same 5,000 random passengers and requires every total and every rule list to agree.
// Usage: const proof = BaggageChecker(spec).evaluateChecked(cls, region, direction, ticket, bags)
(function (root) {
  "use strict";
  const PREMIUM = ["Business", "First"];
  const MAIN_PLUS_LINE = { 1: "L103", 2: "L164" };

  function BaggageChecker(spec) {
    const byKind = {};
    for (const r of spec.rules) (byKind[r.kind] = byKind[r.kind] || []).push(r);
    const one = (kind, test) => {
      const hits = (byKind[kind] || []).filter((r) => test(r.when));
      if (hits.length !== 1) throw new Error(`${kind}: ${hits.length} rules match`);
      return hits[0];
    };
    const inBand = (x, band) => band[0] < x && x <= band[1];

    function baseFee(region, direction, cls, position) {
      if (cls === "Main Plus") {
        const mc = [1, 2].map((p) => baseFee(region, direction, "Main Cabin", p));
        const freeMc = mc[0][0] ? 0 : mc[1][0] ? 1 : 2;
        const freeMp = Math.max(freeMc, Math.min(freeMc + 1, 2));
        const [fee, lines] = baseFee(region, direction, "Main Cabin", position);
        if (position <= freeMp) return [0, [MAIN_PLUS_LINE[position], ...lines]];
        return [fee, [...lines, "X-MAINPLUS"]];
      }
      const r = one("base_fee", (w) => w.regions.includes(region) && w.position === position && w.class === cls
        && (w.direction === null || w.direction === direction));
      return [r.fee, [r.id]];
    }

    function weightFee(region, cls, weight, complimentary) {
      const exceptions = [];
      if (complimentary && PREMIUM.includes(cls)) exceptions.push("L275");
      if (complimentary && region === "Australia") exceptions.push("L279");
      const allowance = exceptions.length ? 70 : 50;
      if (weight <= allowance) {
        const lines = [...exceptions, ...(weight <= 50 ? ["L274"] : [])];
        return [0, lines.length ? lines : ["L274"], allowance];
      }
      const asCls = cls === "Main Plus" ? "Main Cabin" : cls;
      const r = one("weight_fee", (w) => w.regions.includes(region) && w.class === asCls && inBand(weight, w.band));
      if (r.fee === null) return [null, [r.id], allowance];  // "Not accepted": outside the slice
      return [r.fee, [r.id, ...(cls === "Main Plus" ? ["X-MAINPLUS"] : [])], allowance];
    }

    function sizeFee(region, size) {
      if (size <= 62) return [0, ["L273"]];
      if (size > 115) return [null, []];  // over the largest band: outside the slice
      const r = one("size_fee", (w) => w.regions.includes(region) && inBand(size, w.band));
      return [r.fee, [r.id]];
    }

    function evaluateChecked(cls, region, direction, ticket, bags) {
      const steps = [];
      const gain = bags.map((b) => {
        const s = sizeFee(region, b.size)[0];
        return Math.max(s, weightFee(region, cls, b.weight, false)[0]) - Math.max(s, weightFee(region, cls, b.weight, true)[0]);
      });
      const order = bags.map((_, i) => i).sort((i, j) => gain[j] - gain[i] || i - j);  // stable, largest gain first
      steps.push({ step: "order", rules: ["X-ORDER"], inputs: { gain_if_free: gain }, value: order.map((i) => bags[i].item) });
      let total = ticket;
      order.forEach((i, k) => {
        const b = bags[i], position = k + 1;
        const [base, baseLines] = baseFee(region, direction, cls, position);
        const free = base === 0;
        const [wfee, wlines, allowance] = weightFee(region, cls, b.weight, free);
        const [sfee, slines] = sizeFee(region, b.size);
        const bag = wfee === null || sfee === null ? null : base + Math.max(wfee, sfee);
        total = total === null || bag === null ? null : total + bag;
        steps.push(
          { step: "base", item: b.item, position, rules: baseLines, inputs: { class: cls, region, direction, position }, value: base },
          { step: "weight", item: b.item, rules: wlines, inputs: { weight: b.weight, complimentary: free, allowance }, value: wfee },
          { step: "size", item: b.item, rules: ["L270", ...slines], inputs: { size: b.size }, value: sfee },
          { step: "bag", item: b.item, rules: ["L283"], inputs: { base, overweight: wfee, oversize: sfee }, value: bag },
        );
      });
      steps.push({ step: "total", rules: [], inputs: { ticket }, value: total });
      return { steps, total };
    }

    const rule = (id) => spec.rules.find((r) => r.id === id);
    return { evaluateChecked, rule, baseFee, weightFee, sizeFee };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = BaggageChecker;
  else root.BaggageChecker = BaggageChecker;
})(typeof self !== "undefined" ? self : this);
