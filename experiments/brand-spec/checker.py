"""Stages 3 and 4, solve-and-check and prove: evaluate a brand spec on a page's extracted facts and return a proof
artifact.

    from checker import check
    proof = check(spec, page)          # page = {"id", "brand", "domain", "page_type", "elements": [...]}

Element facts (written by extract.js): id, role, tag, text, parent, region (header/nav/main/footer), form, form_x,
x, y, w, h, color, bg (own, or null), eff_bg (behind the text), parent_bg, border, font, size, weight,
line_height, margin [t r b l], padding [t r b l], underline, disabled.

The artifact lists every check that applied: rule, the lines it cites and their text, element, property, value,
expected, pass, and why. When an exception applies, the check is recorded under the exception (with `overrides`),
so the deciding rule for an element and property is the one in its check. `failures` is the failing checks;
`fail_pairs` is the sorted set of (element, rule) pairs, the unit every score uses.
"""

from __future__ import annotations

import re

from common import contrast, luminance

TEXT_ROLES = {
    "logo",
    "header-link",
    "nav-link",
    "h1",
    "h2",
    "h3",
    "body",
    "body-s",
    "lead",
    "link",
    "footer-link",
    "footer-text",
    "label",
    "hint",
    "banner-title",
    "banner-text",
    "button-default",
    "button-secondary",
    "button-warning",
    "button-start",
    "button-inverse",
    "button-outline",
}
BUTTONS = {"button-default", "button-secondary", "button-warning", "button-start", "button-inverse", "button-outline"}
LINK_ROLES = {"link", "header-link", "nav-link", "footer-link"}
PROP_NAMES = {"color": "text colour", "bg": "background", "border": "border colour"}


def is_text(el: dict) -> bool:
    return el["role"] in TEXT_ROLES and bool((el.get("text") or "").strip())


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z.'’-]*", text or "")


def sentence_case(text: str) -> bool:
    ws = words(text)
    if not ws:
        return True
    if not ws[0][0].isupper():
        return False
    if len(ws) >= 2 and all(w.isupper() for w in ws):
        return False  # ALL CAPS
    if ws[0].isupper() and len(ws[0]) >= 4:
        return False  # a shouted first word (acronyms are up to 3 letters here)
    return all(not (w[0].isupper() and not (w.isupper() and len(w) >= 2) and "." not in w) for w in ws[1:])


def is_large(el: dict, lt: dict) -> bool:
    return el["size"] >= lt["min_px"] - 0.01 or (
        el["size"] >= lt["bold_min_px"] - 0.01 and el["weight"] >= lt["bold_weight"]
    )


def fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


class Checker:
    def __init__(self, spec: dict):
        self.spec = spec
        self.rules = {r["id"]: r for r in spec["rules"]}
        self.lines = {s["id"]: s["text"] for s in spec["source"]}
        self.t = spec["tokens"]
        self.palette = {p["hex"]: p for p in self.t["palette"]}
        self.P = "GOV" if spec["brand"] == "govuk" else "USW"

    def rec(self, out, rid, el, prop, value, expected, ok, why=""):
        r = self.rules[rid]
        c = {
            "rule": rid,
            "lines": r["lines"],
            "quote": " / ".join(self.lines[ln] for ln in r["lines"]),
            "element": el["id"] if el else "page",
            "role": el["role"] if el else None,
            "property": prop,
            "value": value,
            "expected": expected,
            "pass": bool(ok),
            "why": why,
        }
        if "overrides" in r:
            c["overrides"] = r["overrides"]
        out.append(c)

    def has(self, rid: str) -> bool:
        return rid in self.rules

    def dark(self, hex_colour: str) -> bool:
        return luminance(hex_colour) < self.t["dark_bg_luminance"]

    def run(self, page: dict) -> list[dict]:
        out: list[dict] = []
        P = self.P
        self.primary = {}  # form -> its primary action button: the first default or start button in it
        for el in page["elements"]:
            if el["role"] in ("button-default", "button-start") and el.get("form") and is_text(el):
                self.primary.setdefault(el["form"], el["id"])
        for el in page["elements"]:
            text = is_text(el)
            self.colours(out, el, text)
            if text:
                self.contrast(out, el)
                self.typography(out, el, page)
            self.spacing(out, el)
            if text and P == "GOV" and el["role"] in LINK_ROLES:
                self.links(out, el)
            if text and el["role"] in BUTTONS:
                self.buttons(out, el, page)
        self.counts(out, page["elements"])
        return out

    def colours(self, out, el, text):
        for rid in (f"{self.P}-PAL", f"{self.P}-FOCUS"):
            if not self.has(rid):
                continue
            r = self.rules[rid]
            for prop in r["params"]["props"]:
                v = el.get(prop)
                if v is None or (prop == "color" and not text):
                    continue
                if r["kind"] == "membership":
                    tok = self.palette.get(v)
                    self.rec(
                        out,
                        rid,
                        el,
                        PROP_NAMES[prop],
                        v,
                        "a palette colour",
                        tok is not None,
                        f"{tok['name']} ({tok['line']})" if tok else "not in the palette",
                    )
                else:
                    bad = v == r["params"]["value"]
                    self.rec(
                        out,
                        rid,
                        el,
                        PROP_NAMES[prop],
                        v,
                        f"not {r['params']['value']}",
                        not bad,
                        "reserved for focus" if bad else "",
                    )

    def contrast(self, out, el):
        P = self.P
        if el.get("disabled") and self.has(f"{P}-CON-INACTIVE"):
            self.rec(out, f"{P}-CON-INACTIVE", el, "contrast", None, "none (inactive component)", True, "disabled")
        elif el["role"] == "logo" and self.has(f"{P}-CON-LOGO"):
            self.rec(out, f"{P}-CON-LOGO", el, "contrast", None, "none (logo)", True, "logo text")
        else:
            ratio = contrast(el["color"], el["eff_bg"])
            large = is_large(el, self.t["large_text"]) and self.has(f"{P}-CON-LARGE")
            rid = f"{P}-CON-LARGE" if large else f"{P}-CON"
            need = self.rules[rid]["params"]["min"]
            why = f"{el['color']} on {el['eff_bg']}; {fmt(el['size'])} px, weight {el['weight']}"
            self.rec(
                out,
                rid,
                el,
                "contrast",
                round(ratio, 2),
                f">= {need}",
                ratio >= need - 1e-9,
                why + ("; large text" if large else ""),
            )
        if self.has(f"{P}-BLACK"):
            bad = el["color"] == "#000000" and el["eff_bg"] == "#ffffff"
            self.rec(
                out,
                f"{P}-BLACK",
                el,
                "colour pair",
                f"{el['color']} on {el['eff_bg']}",
                "not #000000 on #ffffff",
                not bad,
            )

    def typography(self, out, el, page):
        if self.P == "GOV":
            on = page["domain"].endswith(self.rules["GOV-FONT"]["params"]["domain_suffix"])
            brand_font = el["font"] in self.t["brand_fonts"]
            if on:
                self.rec(
                    out,
                    "GOV-FONT",
                    el,
                    "font",
                    el["font"],
                    " or ".join(self.t["brand_fonts"]),
                    brand_font,
                    f"domain {page['domain']}",
                )
            else:
                self.rec(
                    out,
                    "GOV-FONT-ALT",
                    el,
                    "font",
                    el["font"],
                    "not the brand font",
                    not brand_font,
                    f"domain {page['domain']}",
                )
            key = fmt(el["size"])
            point = self.t["type_scale"].get(key)
            why = f"point {key} ({point['line']})" if point else "not a point on the scale"
            if point and point["exceptional"]:
                why += "; noted: only used in exceptional circumstances"
            self.rec(out, "GOV-TYPE", el, "font size", el["size"], "a type-scale point", point is not None, why)
            if point and el.get("line_height") is not None:
                ok = abs(el["line_height"] - point["line_height"]) <= 0.5
                self.rec(
                    out,
                    "GOV-LH",
                    el,
                    "line height",
                    el["line_height"],
                    point["line_height"],
                    ok,
                    f"paired with {key} px ({point['line']})",
                )
            return
        fonts = self.t["fonts"]
        name = {v: k for k, v in self.t.get("font_aliases", {}).items()}.get(el["font"], el["font"])
        why = f"CSS name of {name} (U63)" if name != el["font"] else ""
        self.rec(out, "USW-FAMILY", el, "font", el["font"], "a USWDS available font", name in fonts, why)
        if name in fonts:
            m = self.t["normalization"]["System fonts" if name == "system-ui" else name]
            p = self.rules["USW-SIZE"]["params"]
            tol = max(p["abs_tol"], p["rel_tol"] * el["size"])
            best = min(self.t["sizes"], key=lambda s: abs(el["size"] - s * m))
            ok = abs(el["size"] - best * m) <= tol
            self.rec(
                out,
                "USW-SIZE",
                el,
                "font size",
                el["size"],
                f"target x {m}",
                ok,
                f"nearest: {best} px x {m} = {fmt(best * m)} px",
            )

    def spacing(self, out, el):
        scale = self.t["spacing"]
        vals = [*el["margin"], *el["padding"]]
        bad = sorted({fmt(v) for v in vals if all(abs(abs(v) - s) > 0.01 for s in scale)}, key=float)
        if any(v != 0 for v in vals) or bad:
            self.rec(
                out,
                f"{self.P}-SPACE",
                el,
                "margin and padding",
                " ".join(fmt(v) for v in vals),
                "on the scale",
                not bad,
                f"off the scale: {', '.join(bad)} px" if bad else "",
            )

    def links(self, out, el):
        if self.dark(el["eff_bg"]):
            ok = el["color"] in self.rules["GOV-LINK-INV"]["params"]["allowed"]
            self.rec(
                out,
                "GOV-LINK-INV",
                el,
                "link colour",
                el["color"],
                "#ffffff on a dark background",
                ok,
                f"background {el['eff_bg']} is dark",
            )
        else:
            ok = el["color"] in self.rules["GOV-LINK-COL"]["params"]["allowed"]
            self.rec(out, "GOV-LINK-COL", el, "link colour", el["color"], "#1a65a6 or #54319f", ok)
        if el["role"] in ("header-link", "nav-link"):
            self.rec(
                out,
                "GOV-LINK-NOUL",
                el,
                "underline",
                el["underline"],
                "optional in header or navigation",
                True,
                el["role"],
            )
        else:
            self.rec(out, "GOV-LINK-UL", el, "underline", el["underline"], True, el["underline"])
        stop = el["text"].rstrip().endswith(".")
        self.rec(out, "GOV-LINK-STOP", el, "link text", el["text"], "no final full stop", not stop)

    def buttons(self, out, el, page):
        P = self.P
        self.rec(out, f"{P}-BTN-CASE", el, "label", el["text"], "sentence case", sentence_case(el["text"]))
        first = (words(el["text"]) or [""])[0].lower()
        if P == "USW":
            verbs = self.rules["USW-BTN-VERB"]["params"]["verbs"]
            self.rec(
                out, "USW-BTN-VERB", el, "first word", el["text"], "a verb", first in verbs, f"first word '{first}'"
            )
            n = len(words(el["text"]))
            self.rec(out, "USW-BTN-SHORT", el, "length", el["text"], "<= 4 words", n <= 4, f"{n} words")
            return
        if (
            el["role"] in ("button-default", "button-start")
            and el.get("form")
            and el["id"] == self.primary.get(el["form"])
        ):
            d = abs(el["x"] - el["form_x"])
            self.rec(
                out,
                "GOV-BTN-ALIGN",
                el,
                "left edge",
                fmt(el["x"]),
                f"form edge {fmt(el['form_x'])}",
                d <= 1,
                f"{fmt(d)} px from the form's left edge",
            )
        if el["role"] == "button-start":
            ok = page["page_type"] == "start"
            self.rec(out, "GOV-BTN-START", el, "page type", page["page_type"], "start", ok)
        if el["role"] == "button-warning":
            verbs = self.rules["GOV-BTN-WARN"]["params"]["verbs"]
            self.rec(
                out,
                "GOV-BTN-WARN",
                el,
                "first word",
                el["text"],
                "a destructive verb",
                first in verbs,
                f"first word '{first}'",
            )
        if self.dark(el["parent_bg"]):
            bg = el["bg"] or el["parent_bg"]
            ok = bg == "#ffffff" and contrast(bg, el["parent_bg"]) >= 4.5
            self.rec(
                out,
                "GOV-BTN-INV",
                el,
                "button colour",
                bg,
                "#ffffff on a dark background",
                ok,
                f"background behind the button {el['parent_bg']} is dark",
            )

    def counts(self, out, els):
        rid, roles = ("GOV-BTN-ONE", {"button-default"}) if self.P == "GOV" else ("USW-BTN-MANY", BUTTONS)
        mx = self.rules[rid]["params"]["max"]
        seen = [e for e in els if e["role"] in roles and is_text(e)]
        for i, e in enumerate(seen):
            self.rec(out, rid, e, "count", i + 1, f"<= {mx}", i < mx, f"button {i + 1} of {len(seen)}")


def check(spec: dict, page: dict) -> dict:
    checks = Checker(spec).run(page)
    order = {e["id"]: i for i, e in enumerate(page["elements"])}
    fails = sorted((c for c in checks if not c["pass"]), key=lambda c: (order.get(c["element"], -1), c["rule"]))
    return {
        "page": page["id"],
        "brand": spec["brand"],
        "domain": page["domain"],
        "page_type": page["page_type"],
        "verdict": "fail" if fails else "pass",
        "counts": {"checks": len(checks), "failures": len(fails), "rules_used": len({c["rule"] for c in checks})},
        "failures": fails,
        "notes": [c for c in checks if "exceptional" in c["why"]],
        "checks": checks,
        "fail_pairs": sorted({(c["element"], c["rule"]) for c in fails}),
    }


def deciding(proof: dict, element: str, prop: str) -> str | None:
    """The rule that decided an element's property (the exception, when one applied)."""
    return next((c["rule"] for c in proof["checks"] if c["element"] == element and c["property"] == prop), None)
