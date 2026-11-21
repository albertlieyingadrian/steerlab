// Stage 2, formalise the case: read a rendered page into typed element facts (the ux-glue URL -> tokens idea).
// Runs in the page via page.evaluate(source, components). `components` is [[className, role], ...], longest class
// first, from the brand spec; roles otherwise come from tags. Computed styles only: the page's own CSS is never read.
(components) => {
  const hex = (c) => {
    const m = (c || "").match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    if (p.length > 3 && p[3] === 0) return null;
    return "#" + p.slice(0, 3).map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
  };
  const num = (v) => { const f = parseFloat(v); return Number.isFinite(f) ? Math.round(f * 100) / 100 : 0; };
  const sides = (cs, prop) => ["Top", "Right", "Bottom", "Left"].map((s) => num(cs[prop + s]));
  const ids = new Map();
  let auto = 0;
  const idOf = (el) => {
    if (!ids.has(el)) ids.set(el, el.dataset && el.dataset.id ? el.dataset.id : `n${++auto}`);
    return ids.get(el);
  };
  const ownBg = (el) => hex(getComputedStyle(el).backgroundColor);
  const effBg = (el) => {
    for (let a = el; a && a.nodeType === 1; a = a.parentElement) { const b = ownBg(a); if (b) return b; }
    return "#ffffff";
  };
  const roleOf = (el) => {
    for (const [c, role] of components) if (el.classList.contains(c)) return role;
    const t = el.tagName;
    if (/^H[1-3]$/.test(t)) return t.toLowerCase();
    if (t === "A") return el.closest("footer") ? "footer-link" : el.closest("header") ? "header-link" : "link";
    if (t === "P") return el.closest("footer") ? "footer-text" : "body";
    if (t === "LABEL") return "label";
    if (t === "BUTTON") return "button-default";
    if (t === "INPUT") return "input";
    return "container";
  };
  const regionOf = (el) => el.closest("header") ? "header" : el.closest("nav") ? "nav" : el.closest("footer") ? "footer" : "main";
  const out = [];
  for (const el of document.body.querySelectorAll("*")) {
    if (["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "BR", "SVG", "PATH"].includes(el.tagName.toUpperCase())) continue;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    let text = "";
    for (const c of el.childNodes) if (c.nodeType === 3) text += c.textContent;
    text = text.replace(/\s+/g, " ").trim();
    let margin = sides(cs, "margin");
    // a block centred with margin:auto reports its used margins; record them as 0 (auto is not a spacing value)
    const par = el.parentElement;
    if (par && margin[1] > 0 && Math.abs(margin[1] - margin[3]) < 1) {
      const pr = par.getBoundingClientRect(), pcs = getComputedStyle(par);
      const inner = pr.width - num(pcs.paddingLeft) - num(pcs.paddingRight) - num(pcs.borderLeftWidth) - num(pcs.borderRightWidth);
      if (Math.abs((inner - r.width) / 2 - margin[1]) < 1) margin = [margin[0], 0, margin[2], 0];
    }
    let border = null;
    for (const s of ["Left", "Top", "Bottom", "Right"]) {
      if (num(cs[`border${s}Width`]) > 0 && cs[`border${s}Style`] !== "none") { border = hex(cs[`border${s}Color`]); break; }
    }
    const form = el.closest("form");
    let formX = null;
    if (form) { const fr = form.getBoundingClientRect(), fcs = getComputedStyle(form); formX = num(fr.left + num(fcs.paddingLeft) + num(fcs.borderLeftWidth)); }
    let parent = null;
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) if (ids.has(a)) { parent = ids.get(a); break; }
    const lh = parseFloat(cs.lineHeight);
    out.push({
      id: idOf(el), role: roleOf(el), tag: el.tagName.toLowerCase(), text,
      parent, region: regionOf(el), form: form ? idOf(form) : null, form_x: formX,
      x: num(r.left + window.scrollX), y: num(r.top + window.scrollY), w: num(r.width), h: num(r.height),
      color: hex(cs.color), bg: ownBg(el), eff_bg: effBg(el), parent_bg: par ? effBg(par) : "#ffffff",
      border, font: cs.fontFamily.split(",")[0].trim().replace(/^["']|["']$/g, ""),
      size: num(cs.fontSize), weight: parseInt(cs.fontWeight, 10) || 400,
      line_height: Number.isFinite(lh) ? Math.round(lh * 100) / 100 : null,
      margin, padding: sides(cs, "padding"),
      underline: (cs.textDecorationLine || "").includes("underline"),
      disabled: !!(el.disabled || el.getAttribute("aria-disabled") === "true"),
    });
  }
  return out;
}
