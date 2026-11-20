// In-page extractor for the design index. Runs via page.evaluate(js, parkRegex) after the page settles.
// Returns geometry, computed styles and a block tree for elements in the first 2,400 px; never page text
// (only word counts and a parked/error flag computed from it).
(parkRe) => {
  const MAXY = 2400, VW = 1280, MAX_EL = 6000;
  const fonts = [], fontIx = new Map();
  const font = (ff) => {
    const f = (ff || "").split(",")[0].trim().replace(/^["']|["']$/g, "").toLowerCase();
    if (!fontIx.has(f)) { fontIx.set(f, fonts.length); fonts.push(f); }
    return fontIx.get(f);
  };
  const rgb = (c) => {
    const m = (c || "").match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    if (p.length > 3 && p[3] === 0) return null;
    return [p[0] | 0, p[1] | 0, p[2] | 0];
  };
  const MEDIA = new Set(["IMG", "SVG", "VIDEO", "CANVAS", "PICTURE", "IFRAME", "OBJECT", "EMBED"]);
  const CTRL = new Set(["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"]);
  const BLOCKDISP = new Set(["block", "flex", "grid", "table", "list-item", "flow-root", "inline-block", "inline-flex", "inline-grid"]);
  const sy = window.scrollY;
  const els = [], blocks = [], blockIx = new Map();
  let nWords = 0, n = 0;
  for (const el of document.querySelectorAll("body *")) {
    if (++n > MAX_EL * 3) break;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden" || parseFloat(cs.opacity) === 0) continue;
    const r = el.getBoundingClientRect();
    const y = r.top + sy;
    if (r.width < 2 || r.height < 2 || y > MAXY || y + r.height < 0 || r.left > VW || r.right < 0) continue;
    const tag = el.tagName.toUpperCase();
    let words = 0;
    for (const c of el.childNodes) if (c.nodeType === 3) { const t = c.textContent.trim(); if (t) words += t.split(/\s+/).length; }
    let kind = 0;
    if (MEDIA.has(tag) || (cs.backgroundImage.includes("url(") && r.width * r.height > 4000)) kind = 2;
    else if (CTRL.has(tag) || el.getAttribute("role") === "button") kind = 3;
    else if (words > 0) kind = 1;
    nWords += words;
    // nearest block ancestor already recorded
    let par = -1;
    for (let a = el.parentElement; a; a = a.parentElement) if (blockIx.has(a)) { par = blockIx.get(a); break; }
    const isBlock = BLOCKDISP.has(cs.display) && r.width * r.height >= 400;
    if (isBlock && blocks.length < MAX_EL) {
      blockIx.set(el, blocks.length);
      const lab = /^H[1-6]$/.test(tag) ? "h" : ["NAV", "HEADER", "FOOTER", "SECTION", "ARTICLE", "ASIDE", "MAIN", "FORM", "UL", "OL", "LI", "P", "TABLE", "A", "BUTTON", "IMG", "FIGURE"].includes(tag) ? tag.toLowerCase() : "div";
      const disp = cs.display.includes("grid") ? "g" : cs.display.includes("flex") ? "f" : "b";
      blocks.push([Math.round(r.left), Math.round(y), Math.round(r.width), Math.round(r.height), par, lab + ":" + disp + ":" + kind]);
    }
    if (kind > 0 && els.length < MAX_EL) {
      els.push([Math.round(r.left), Math.round(y), Math.round(r.width), Math.round(r.height), kind,
        kind === 2 ? -1 : font(cs.fontFamily), parseFloat(cs.fontSize) || 0, parseInt(cs.fontWeight) || 400, words,
        rgb(cs.color), rgb(cs.backgroundColor), parseFloat(cs.paddingTop) || 0, parseFloat(cs.paddingLeft) || 0,
        parseFloat(cs.marginTop) || 0, parseFloat(cs.borderTopLeftRadius) || 0, parseFloat(cs.lineHeight) || 0]);
    }
  }
  const txt = (document.body ? document.body.innerText : "").slice(0, 3000);
  const bodyCs = document.body ? getComputedStyle(document.body) : null;
  return {
    n_words: nWords,
    parked: new RegExp(parkRe, "i").test(txt) && nWords < 150,
    doc_h: document.documentElement.scrollHeight,
    body_bg: bodyCs ? rgb(bodyCs.backgroundColor) : null,
    fonts, elements: els, blocks,
    generator: (document.querySelector('meta[name="generator"]') || {}).content || null,
  };
}
