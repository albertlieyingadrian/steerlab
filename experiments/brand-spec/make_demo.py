"""Write the interactive's data: report/demo.js (the GOV.UK spec, trimmed, and one demo page in 18 versions).

    .venv/bin/python experiments/brand-spec/make_demo.py

Each version is the same start page with one planted violation, one exception trap, or neither, with its HTML and
the extractor's facts. The browser runs report/checker.js on the facts; ruler.py has already checked that the JS
port agrees with the Python checker on every one of these pages.
"""

from __future__ import annotations

import json

from common import DATA, HERE, load_spec, read_jsonl

KEEP = [
    "offpal-text",
    "contrast",
    "contrast-large",
    "font",
    "type",
    "space",
    "link-ul",
    "link-inv",
    "btn-case",
    "btn-two",
    "btn-align",
    "btn-warn",
    "focus",
]
TRAPS = ["disabled", "large", "logo", "bold19"]
LABELS = {
    "base": "On-brand: no violations",
    "offpal-text": "Text colour not in the palette",
    "contrast": "Body text below 4.5:1",
    "contrast-large": "Large heading below 3:1",
    "font": "A heading in another typeface",
    "type": "Body text at 18 px (off the type scale)",
    "space": "A margin off the spacing scale",
    "link-ul": "A body link without its underline",
    "link-inv": "A light-blue link on a dark band",
    "btn-case": "Button text in Title Case",
    "btn-two": "A second default button",
    "btn-align": "Primary button moved off the form's edge",
    "btn-warn": "A warning button that says Continue",
    "focus": "A card in the focus yellow",
    "trap-disabled": "Exception: a disabled button at 2.2:1",
    "trap-large": "Exception: a large heading at 3.2:1",
    "trap-logo": "Exception: logo text at 1.6:1",
    "trap-bold19": "Exception: a bold 19 px label at 3.7:1",
}
FIELDS = [
    "id",
    "role",
    "tag",
    "text",
    "region",
    "form",
    "color",
    "bg",
    "eff_bg",
    "parent_bg",
    "border",
    "font",
    "size",
    "weight",
    "line_height",
    "margin",
    "padding",
    "underline",
    "disabled",
]


def trim_spec(spec: dict) -> dict:
    keep = {ln for r in spec["rules"] for ln in r["lines"] + r["support"]} | {r["line"] for r in spec["residue"]}
    return {
        "brand": spec["brand"],
        "title": spec["title"],
        "source": [s for s in spec["source"] if s["id"] in keep or s["id"].startswith("W")],
        "tokens": spec["tokens"],
        "rules": spec["rules"],
        "residue": spec["residue"],
    }


def main() -> None:
    pages = {p["id"]: p for p in read_jsonl(DATA / "demo_govuk.jsonl")}
    facts = {f["id"]: f for f in read_jsonl(DATA / "facts_demo_govuk.jsonl")}
    order = ["demo-base"] + [f"demo-{k}" for k in KEEP] + [f"demo-trap-{k}" for k in TRAPS]
    base = {e["id"]: e for e in facts["demo-base"]["elements"]}
    style = (
        "color",
        "bg",
        "font",
        "size",
        "line_height",
        "weight",
        "margin",
        "padding",
        "text",
        "underline",
        "disabled",
        "role",
    )
    variants = []
    for vid in order:
        p, f = pages[vid], facts[vid]
        els = []
        for e in f["elements"]:
            x = {k: e[k] for k in FIELDS}
            if e["role"].startswith("button"):
                x["x"], x["form_x"] = e["x"], e["form_x"]
            els.append(x)
        changed = [e["id"] for e in f["elements"] if any(e[k] != base[e["id"]][k] for k in style)]
        focus = (
            p["planted"][0]["element"]
            if p["planted"]
            else changed[0]
            if changed
            else next(e["id"] for e in f["elements"] if e["role"] == "button-default")
        )
        key = vid.removeprefix("demo-")
        group = "trap" if key.startswith("trap") else "base" if key == "base" else "violation"
        variants.append(
            {
                "id": vid,
                "label": LABELS[key],
                "group": group,
                "planted": p["planted"],
                "traps": p["traps"],
                "labels": p["labels"],
                "focus": focus,
                "html": p["html"],
                "page": {
                    "id": vid,
                    "brand": "govuk",
                    "domain": f["domain"],
                    "page_type": f["page_type"],
                    "elements": els,
                },
            }
        )
    data = {"spec": trim_spec(load_spec("govuk")), "variants": variants}
    out = HERE / "report" / "demo.js"
    out.write_text("window.BRAND_DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{out.name}: {len(variants)} versions, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
