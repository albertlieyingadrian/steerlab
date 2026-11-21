// The brand checker in JavaScript, a line-for-line port of checker.py for the note's interactive. ruler.py runs it
// against the Python checker on every generated page and 2,000 single-property mutations (bar: all agree).
// Usage: const proof = BrandChecker(spec).check(page)  ->  {verdict, checks, failures, fail_pairs}
(function (root) {
  const TEXT_ROLES = new Set(["logo", "header-link", "nav-link", "h1", "h2", "h3", "body", "body-s", "lead", "link",
    "footer-link", "footer-text", "label", "hint", "banner-title", "banner-text", "button-default", "button-secondary",
    "button-warning", "button-start", "button-inverse", "button-outline"]);
  const BUTTONS = new Set(["button-default", "button-secondary", "button-warning", "button-start", "button-inverse",
    "button-outline"]);
  const LINK_ROLES = new Set(["link", "header-link", "nav-link", "footer-link"]);
  const PROP_NAMES = { color: "text colour", bg: "background", border: "border colour" };

  const isText = (el) => TEXT_ROLES.has(el.role) && !!(el.text || "").trim();
  const words = (t) => (t || "").match(/[A-Za-z][A-Za-z.'’-]*/g) || [];
  const isUpper = (w) => w === w.toUpperCase() && /[A-Z]/.test(w);
  const capital = (ch) => ch !== ch.toLowerCase();
  function sentenceCase(text) {
    const ws = words(text);
    if (!ws.length) return true;
    if (!capital(ws[0][0])) return false;
    if (ws.length >= 2 && ws.every(isUpper)) return false;
    if (isUpper(ws[0]) && ws[0].length >= 4) return false;
    return ws.slice(1).every((w) => !(capital(w[0]) && !(isUpper(w) && w.length >= 2) && !w.includes(".")));
  }
  function lum(hex) {
    const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  function contrast(a, b) { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); }
  const fmt = (x) => (Math.round(x * 100) / 100).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  const round2 = (x) => Math.round(x * 100) / 100;

  function BrandChecker(spec) {
    const rules = Object.fromEntries(spec.rules.map((r) => [r.id, r]));
    const lines = Object.fromEntries(spec.source.map((s) => [s.id, s.text]));
    const t = spec.tokens;
    const palette = new Map(t.palette.map((p) => [p.hex, p]));
    const P = spec.brand === "govuk" ? "GOV" : "USW";
    const has = (id) => id in rules;
    const dark = (hex) => lum(hex) < t.dark_bg_luminance;

    function run(page) {
      const out = [];
      const rec = (rid, el, prop, value, expected, ok, why = "") => {
        const r = rules[rid];
        const c = { rule: rid, lines: r.lines, quote: r.lines.map((l) => lines[l]).join(" / "), element: el ? el.id : "page",
          role: el ? el.role : null, property: prop, value, expected, pass: !!ok, why };
        if (r.overrides) c.overrides = r.overrides;
        out.push(c);
      };
      const primary = {};
      for (const el of page.elements) {
        if ((el.role === "button-default" || el.role === "button-start") && el.form && isText(el) && !(el.form in primary)) primary[el.form] = el.id;
      }
      for (const el of page.elements) {
        const text = isText(el);
        for (const rid of [`${P}-PAL`, `${P}-FOCUS`]) {
          if (!has(rid)) continue;
          const r = rules[rid];
          for (const prop of r.params.props) {
            const v = el[prop];
            if (v == null || (prop === "color" && !text)) continue;
            if (r.kind === "membership") {
              const tok = palette.get(v);
              rec(rid, el, PROP_NAMES[prop], v, "a palette colour", !!tok, tok ? `${tok.name} (${tok.line})` : "not in the palette");
            } else {
              const bad = v === r.params.value;
              rec(rid, el, PROP_NAMES[prop], v, `not ${r.params.value}`, !bad, bad ? "reserved for focus" : "");
            }
          }
        }
        if (text) {
          if (el.disabled && has(`${P}-CON-INACTIVE`)) rec(`${P}-CON-INACTIVE`, el, "contrast", null, "none (inactive component)", true, "disabled");
          else if (el.role === "logo" && has(`${P}-CON-LOGO`)) rec(`${P}-CON-LOGO`, el, "contrast", null, "none (logo)", true, "logo text");
          else {
            const ratio = contrast(el.color, el.eff_bg);
            const lt = t.large_text;
            const large = (el.size >= lt.min_px - 0.01 || (el.size >= lt.bold_min_px - 0.01 && el.weight >= lt.bold_weight)) && has(`${P}-CON-LARGE`);
            const rid = large ? `${P}-CON-LARGE` : `${P}-CON`;
            const need = rules[rid].params.min;
            rec(rid, el, "contrast", round2(ratio), `>= ${need}`, ratio >= need - 1e-9,
              `${el.color} on ${el.eff_bg}; ${fmt(el.size)} px, weight ${el.weight}` + (large ? "; large text" : ""));
          }
          if (has(`${P}-BLACK`)) {
            const bad = el.color === "#000000" && el.eff_bg === "#ffffff";
            rec(`${P}-BLACK`, el, "colour pair", `${el.color} on ${el.eff_bg}`, "not #000000 on #ffffff", !bad);
          }
          if (P === "GOV") {
            const on = page.domain.endsWith(rules["GOV-FONT"].params.domain_suffix);
            const brandFont = t.brand_fonts.includes(el.font);
            if (on) rec("GOV-FONT", el, "font", el.font, t.brand_fonts.join(" or "), brandFont, `domain ${page.domain}`);
            else rec("GOV-FONT-ALT", el, "font", el.font, "not the brand font", !brandFont, `domain ${page.domain}`);
            const key = fmt(el.size);
            const point = t.type_scale[key];
            let why = point ? `point ${key} (${point.line})` : "not a point on the scale";
            if (point && point.exceptional) why += "; noted: only used in exceptional circumstances";
            rec("GOV-TYPE", el, "font size", el.size, "a type-scale point", !!point, why);
            if (point && el.line_height != null) {
              rec("GOV-LH", el, "line height", el.line_height, point.line_height, Math.abs(el.line_height - point.line_height) <= 0.5,
                `paired with ${key} px (${point.line})`);
            }
          } else {
            const alias = Object.fromEntries(Object.entries(t.font_aliases || {}).map(([k, v]) => [v, k]));
            const name = alias[el.font] || el.font;
            const ok = t.fonts.includes(name);
            rec("USW-FAMILY", el, "font", el.font, "a USWDS available font", ok, name !== el.font ? `CSS name of ${name} (U63)` : "");
            if (ok) {
              const m = t.normalization[name === "system-ui" ? "System fonts" : name];
              const p = rules["USW-SIZE"].params;
              const tol = Math.max(p.abs_tol, p.rel_tol * el.size);
              let best = t.sizes[0];
              for (const s of t.sizes) if (Math.abs(el.size - s * m) < Math.abs(el.size - best * m)) best = s;
              rec("USW-SIZE", el, "font size", el.size, `target x ${m}`, Math.abs(el.size - best * m) <= tol, `nearest: ${best} px x ${m} = ${fmt(best * m)} px`);
            }
          }
        }
        const vals = [...el.margin, ...el.padding];
        const bad = [...new Set(vals.filter((v) => t.spacing.every((s) => Math.abs(Math.abs(v) - s) > 0.01)).map(fmt))]
          .sort((a, b) => parseFloat(a) - parseFloat(b));
        if (vals.some((v) => v !== 0) || bad.length) {
          rec(`${P}-SPACE`, el, "margin and padding", vals.map(fmt).join(" "), "on the scale", !bad.length, bad.length ? `off the scale: ${bad.join(", ")} px` : "");
        }
        if (text && P === "GOV" && LINK_ROLES.has(el.role)) {
          if (dark(el.eff_bg)) rec("GOV-LINK-INV", el, "link colour", el.color, "#ffffff on a dark background", rules["GOV-LINK-INV"].params.allowed.includes(el.color), `background ${el.eff_bg} is dark`);
          else rec("GOV-LINK-COL", el, "link colour", el.color, "#1a65a6 or #54319f", rules["GOV-LINK-COL"].params.allowed.includes(el.color));
          if (el.role === "header-link" || el.role === "nav-link") rec("GOV-LINK-NOUL", el, "underline", el.underline, "optional in header or navigation", true, el.role);
          else rec("GOV-LINK-UL", el, "underline", el.underline, true, el.underline);
          rec("GOV-LINK-STOP", el, "link text", el.text, "no final full stop", !el.text.trimEnd().endsWith("."));
        }
        if (text && BUTTONS.has(el.role)) {
          rec(`${P}-BTN-CASE`, el, "label", el.text, "sentence case", sentenceCase(el.text));
          const first = (words(el.text)[0] || "").toLowerCase();
          if (P === "USW") {
            rec("USW-BTN-VERB", el, "first word", el.text, "a verb", rules["USW-BTN-VERB"].params.verbs.includes(first), `first word '${first}'`);
            const n = words(el.text).length;
            rec("USW-BTN-SHORT", el, "length", el.text, "<= 4 words", n <= 4, `${n} words`);
          } else {
            if ((el.role === "button-default" || el.role === "button-start") && el.form && el.id === primary[el.form]) {
              const d = Math.abs(el.x - el.form_x);
              rec("GOV-BTN-ALIGN", el, "left edge", fmt(el.x), `form edge ${fmt(el.form_x)}`, d <= 1, `${fmt(d)} px from the form's left edge`);
            }
            if (el.role === "button-start") rec("GOV-BTN-START", el, "page type", page.page_type, "start", page.page_type === "start");
            if (el.role === "button-warning") rec("GOV-BTN-WARN", el, "first word", el.text, "a destructive verb", rules["GOV-BTN-WARN"].params.verbs.includes(first), `first word '${first}'`);
            if (dark(el.parent_bg)) {
              const bg = el.bg || el.parent_bg;
              rec("GOV-BTN-INV", el, "button colour", bg, "#ffffff on a dark background", bg === "#ffffff" && contrast(bg, el.parent_bg) >= 4.5,
                `background behind the button ${el.parent_bg} is dark`);
            }
          }
        }
      }
      const [rid, roles] = P === "GOV" ? ["GOV-BTN-ONE", new Set(["button-default"])] : ["USW-BTN-MANY", BUTTONS];
      const mx = rules[rid].params.max;
      const seen = page.elements.filter((e) => roles.has(e.role) && isText(e));
      seen.forEach((e, i) => rec(rid, e, "count", i + 1, `<= ${mx}`, i < mx, `button ${i + 1} of ${seen.length}`));
      return out;
    }

    function check(page) {
      const checks = run(page);
      const order = Object.fromEntries(page.elements.map((e, i) => [e.id, i]));
      const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
      const failures = checks.filter((c) => !c.pass)
        .sort((a, b) => (order[a.element] ?? -1) - (order[b.element] ?? -1) || cmp(a.rule, b.rule));
      const seen = new Set();
      const pairs = [];
      for (const c of failures) {
        const k = c.element + "\u0000" + c.rule;
        if (!seen.has(k)) { seen.add(k); pairs.push([c.element, c.rule]); }
      }
      pairs.sort((a, b) => cmp(a[0], b[0]) || cmp(a[1], b[1]));
      return { verdict: failures.length ? "fail" : "pass", checks, failures, fail_pairs: pairs };
    }
    return { check, contrast, rule: (id) => rules[id], line: (id) => lines[id] };
  }
  root.BrandChecker = BrandChecker;
  if (typeof module !== "undefined") module.exports = BrandChecker;
})(typeof window !== "undefined" ? window : globalThis);
