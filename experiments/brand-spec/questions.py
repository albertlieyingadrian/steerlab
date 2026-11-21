"""The model test sets, built from the frozen pages and the specs (answers come from the checker).

    .venv/bin/python experiments/brand-spec/questions.py

T1 audit      a whole page (its element facts) -> every (element, line) it breaks, or [].
T2 identify   a whole page and one element's property -> the line that decides it. Accepted strictly: the deciding
              rule's own lines; loosely: those plus the lines it reads (`support`). About a third are decided by an
              exception.
T3 apply      quoted lines + one element's facts -> complies or not. Synthetic elements, balanced yes/no, answers from
              the checker. `t3c_*` repeats the contrast items with the ratio given (the computing-vs-applying control).
Writes data/t1_<brand>_<split>.jsonl, data/t2_..., data/t3_..., data/t3c_... and adds their hashes to the manifest.
"""

from __future__ import annotations

import json
import random

from checker import check, is_text
from common import BRANDS, DATA, contrast, load_spec, read_jsonl, sha256, write_jsonl

T2_N = {"dev": 30, "heldout": 120}
NEVER_ASKED = {"USW-BTN-MANY", "USW-BTN-SHORT"}
NOT_IN_T2 = {"USW-BLACK"}  # its property (the text and background pair) is also contrast's; the question is ambiguous
PROPERTY_WORDS = {
    "contrast": "text contrast",
    "font": "font",
    "font size": "font size",
    "line height": "line height",
    "margin and padding": "margin and padding",
    "link colour": "link colour",
    "underline": "underline",
    "link text": "link text punctuation",
    "label": "button text capitalisation",
    "left edge": "horizontal position within its form",
    "count": "being one more default button on the page",
    "page type": "use of a start button on this page",
    "first word": "first word of the button text",
    "first word warning": "use of the warning button style for this action",
    "button colour": "button colour against the background behind it",
    "text colour": "text colour",
    "background": "background colour",
    "border colour": "border colour",
    "colour pair": "text and background colours",
    "length": "button text length",
}


# ---------------------------------------------------------------- describing facts in text
def fmt(x) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def describe_el(el: dict) -> str | None:
    """One line per element: id, role, text and the facts any rule can read. Bare containers are skipped."""
    parts = [el["id"], el["role"] if el["role"] != "container" else f"container ({el['tag']})"]
    text = is_text(el)
    if text:
        t = el["text"] if len(el["text"]) <= 90 else el["text"][:87] + "..."
        parts.append(f'"{t}"')
    facts = []
    if text:
        facts.append(f"text {el['color']} on {el['eff_bg']}")
    if el.get("bg"):
        facts.append(f"background {el['bg']}")
    if el.get("border"):
        facts.append(f"border {el['border']}")
    if text:
        lh = f", line height {fmt(el['line_height'])}px" if el.get("line_height") else ""
        facts.append(f"font {el['font']} {fmt(el['size'])}px{lh}, weight {el['weight']}")
    if text and el["role"] in ("link", "header-link", "nav-link", "footer-link"):
        facts.append(("underlined" if el["underline"] else "no underline") + f", in {el['region']}")
    if el.get("disabled"):
        facts.append("disabled")
    if any(v != 0 for v in el["margin"]):
        facts.append("margin " + " ".join(fmt(v) for v in el["margin"]))
    if any(v != 0 for v in el["padding"]):
        facts.append("padding " + " ".join(fmt(v) for v in el["padding"]))
    if el["role"].startswith("button") and text:
        facts.append(f"placed on {el['parent_bg']}")
        if el.get("form"):
            facts.append(f"in a form; left edge {fmt(el['x'])}px, the form's left edge {fmt(el['form_x'])}px")
    if not facts and el["role"] == "container":
        return None
    return " | ".join([" ".join(parts), *facts])


def describe_page(page: dict) -> str:
    head = (
        f"Page: domain {page['domain']}; page type: {page['page_type']}; screen width 1280 px. "
        "Elements in document order (bare layout containers omitted):"
    )
    return "\n".join([head] + [s for el in page["elements"] if (s := describe_el(el))])


# ---------------------------------------------------------------- T1 and T2
def t1(brand, split):
    pages = read_jsonl(DATA / f"pages_{brand}_{split}.jsonl")
    facts = read_jsonl(DATA / f"facts_pages_{brand}_{split}.jsonl")
    return [
        {
            "id": p["id"],
            "brand": brand,
            "split": split,
            "kind": p["kind"],
            "page": describe_page(f),
            "labels": p["labels"],
            "planted": p["planted"],
            "traps": p["traps"],
            "elements": [e["id"] for e in f["elements"]],
        }
        for p, f in zip(pages, facts, strict=True)
    ]


def prop_key(x: dict) -> str:
    return "first word warning" if x["property"] == "first word" and x["rule"] == "GOV-BTN-WARN" else x["property"]


def t2(brand, split, spec):
    r = random.Random(f"t2-{brand}-{split}")
    rules = {x["id"]: x for x in spec["rules"]}
    pool = []
    for f in read_jsonl(DATA / f"facts_pages_{brand}_{split}.jsonl"):
        described = {e["id"] for e in f["elements"] if describe_el(e)}
        seen = set()
        for c in check(spec, f)["checks"]:
            key = (c["element"], c["property"])
            if c["rule"] in NEVER_ASKED | NOT_IN_T2 or key in seen or c["element"] not in described:
                continue
            seen.add(key)
            pool.append(
                {
                    "page_id": f["id"],
                    "page": describe_page(f),
                    "element": c["element"],
                    "role": c["role"],
                    "property": c["property"],
                    "rule": c["rule"],
                    "pass": c["pass"],
                    "exception": "overrides" in rules[c["rule"]],
                }
            )
    by_rule: dict[str, list] = {}
    for x in pool:
        by_rule.setdefault(x["rule"], []).append(x)
    lists = {rid: r.sample(v, len(v)) for rid, v in by_rule.items()}
    n = T2_N[split]
    exc = sorted(rid for rid in by_rule if rules[rid].get("overrides"))
    plain = sorted(rid for rid in by_rule if rid not in exc)

    def take(rids, k, cap=None):
        out, i, used = [], 0, dict.fromkeys(rids, 0)
        while len(out) < k and any(lists[rid] and (cap is None or used[rid] < cap) for rid in rids):
            rid = rids[i % len(rids)]
            if lists[rid] and (cap is None or used[rid] < cap):
                out.append(lists[rid].pop())
                used[rid] += 1
            i += 1
        return out

    items = take(exc, round(n / 3), cap=max(2, n // 6)) if exc else []
    items += take(plain, n - len(items))
    r.shuffle(items)
    for i, x in enumerate(items):
        ru = rules[x["rule"]]
        x.update(
            id=f"t2-{brand}-{split}-{i:03d}",
            brand=brand,
            split=split,
            question=f"element {x['element']}'s {PROPERTY_WORDS[prop_key(x)]}",
            strict=ru["lines"],
            loose=sorted(set(ru["lines"]) | set(ru["support"])),
            settled=ru["settled"],
        )
    return items


# ---------------------------------------------------------------- T3 (synthetic single elements)
def base_el(brand, **kw):
    gov = brand == "govuk"
    el = {
        "id": "e1",
        "role": "body",
        "tag": "p",
        "text": "You can renew your permit online.",
        "parent": None,
        "region": "main",
        "form": None,
        "form_x": None,
        "x": 160.0,
        "y": 300.0,
        "w": 600.0,
        "h": 25.0,
        "color": "#0b0c0c" if gov else "#1b1b1b",
        "bg": None,
        "eff_bg": "#ffffff",
        "parent_bg": "#ffffff",
        "border": None,
        "font": "GDS Transport" if gov else "Source Sans Pro",
        "size": 19.0 if gov else 18.02,
        "weight": 400,
        "line_height": 25.0 if gov else 27.56,
        "margin": [0.0, 0.0, 20.0 if gov else 16.0, 0.0],
        "padding": [0.0] * 4,
        "underline": False,
        "disabled": False,
    }
    el.update(kw)
    return el


def judge(spec, el, rules, domain=None):
    """The checker's verdict for one element against the given rule ids (all must pass)."""
    brand = spec["brand"]
    domain = domain or ("renew-a-permit.service.gov.uk" if brand == "govuk" else "renew-a-permit.example.gov")
    page = {"id": "t3", "brand": brand, "domain": domain, "page_type": "content", "elements": [el]}
    hits = [c for c in check(spec, page)["checks"] if c["rule"] in rules]
    assert hits, (rules, el)
    return all(c["pass"] for c in hits), hits[0]["rule"], domain


def t3(brand, split, spec):
    r = random.Random(f"t3-{brand}-{split}")
    L = {s["id"]: s["text"] for s in spec["source"]}
    scale = 1 if split == "heldout" else 0.25
    items, controls = [], []

    def n(k):
        return max(2, round(k * scale))

    def item(kind, lines, el, rule, answer, domain=None, tag=None):
        return {
            "kind": kind,
            "lines": lines,
            "quote": [f"{ln} {L[ln]}" for ln in lines],
            "element": el,
            "domain": domain,
            "answer": "yes" if answer else "no",
            "rule": rule,
            "tag": tag,
        }

    def balanced(kind, k, lines, rules, gen):
        made = {True: 0, False: 0}
        for _ in range(4000):
            if sum(made.values()) >= k:
                break
            want = made[True] <= made[False]
            el, domain = gen(want)
            ok, decider, dom = judge(spec, el, rules, domain)
            if ok != want:
                continue
            made[ok] += 1
            items.append(item(kind, lines, el, decider, ok, dom))

    # contrast: the same pairs also go to the ratio-given control
    lines = ["G2", "W1", "W2", "W3", "W4", "W5", "W6", "W7"] if brand == "govuk" else ["U4", "U5"]
    con = ("GOV-CON", "GOV-CON-LARGE") if brand == "govuk" else ("USW-CON", "USW-CON-LARGE")
    shapes = (
        [(19.0, 400), (24.0, 400), (19.0, 700), (16.0, 400), (36.0, 700)]
        if brand == "govuk"
        else [(18.02, 400), (25.44, 400), (19.08, 700), (15.9, 400), (31.36, 700)]
    )
    pal = [p for p in spec["tokens"]["palette"] if brand == "govuk" or p["name"] not in ("white", "black")]
    k = n(30)
    made = {True: 0, False: 0}
    while sum(made.values()) < k:
        a, b = r.sample(pal, 2)
        if not 1.8 <= contrast(a["hex"], b["hex"]) <= 7.5:
            continue
        size, weight = r.choice(shapes)
        el = base_el(
            brand,
            color=a["hex"],
            eff_bg=b["hex"],
            size=size,
            weight=weight,
            role="h2" if size > 30 else "body",
            line_height=None,
        )
        ok, decider, dom = judge(spec, el, con)
        if made[ok] >= k / 2:
            continue
        made[ok] += 1
        tag = {"a": a["name"], "b": b["name"], "ga": a.get("grade"), "gb": b.get("grade")}
        items.append(item("contrast", lines, el, decider, ok, dom, tag))
        controls.append(dict(items[-1], kind="contrast-given", ratio=round(contrast(a["hex"], b["hex"]), 2)))
    if brand == "uswds":  # the magic number: contrast as arithmetic on token grades
        k = n(30)
        made = {True: 0, False: 0}
        while sum(made.values()) < k:
            a, b = r.sample(pal, 2)
            diff = abs(a["grade"] - b["grade"])
            if not 20 <= diff <= 80 or made[diff >= 50] >= k / 2:
                continue
            made[diff >= 50] += 1
            el = base_el(brand, color=a["hex"], eff_bg=b["hex"])
            items.append(
                item(
                    "magic",
                    ["U2", "U7", "U9"],
                    el,
                    "U9",
                    diff >= 50,
                    None,
                    {"a": a["name"], "b": b["name"], "ga": a["grade"], "gb": b["grade"]},
                )
            )

    if brand == "govuk":
        lh = {16: 20, 19: 25, 24: 30, 36: 40, 48: 50}
        scale_lines = ["G45", *[f"G{i}" for i in range(48, 55)]]
        balanced(
            "type scale",
            n(15),
            scale_lines,
            ("GOV-TYPE",),
            lambda y: (
                base_el(
                    brand,
                    size=float(r.choice([16, 19, 24, 27, 36, 48, 80]) if y else r.choice([14, 17, 18, 20, 22, 30])),
                    line_height=None,
                ),
                None,
            ),
        )
        balanced(
            "line height",
            n(10),
            ["G46", *scale_lines[1:]],
            ("GOV-LH",),
            lambda y: (
                lambda s: (
                    base_el(brand, size=float(s), line_height=float(lh[s] if y else lh[s] + r.choice([-5, 5, 10]))),
                    None,
                )
            )(r.choice(list(lh))),
        )
        balanced(
            "spacing",
            n(15),
            ["G58"],
            ("GOV-SPACE",),
            lambda y: (
                base_el(
                    brand,
                    margin=[0.0, 0.0, float(r.choice([10, 20, 30, 40]) if y else r.choice([12, 18, 24, 32, 36])), 0.0],
                ),
                None,
            ),
        )
        hexes = sorted({p["hex"] for p in spec["tokens"]["palette"]})
        balanced(
            "palette",
            n(15),
            ["G1", *[f"G{i}" for i in range(7, 27)], *[f"G{i}" for i in range(29, 40)]],
            ("GOV-PAL",),
            lambda y: (
                (
                    base_el(
                        brand,
                        color=r.choice(["#0b0c0c", "#484949", "#16548a"])
                        if y
                        else r.choice(["#1b1b1b", "#333333", "#0b0c0d", "#123456"]),
                    )
                    if r.random() < 0.5
                    else base_el(
                        brand,
                        role="container",
                        tag="div",
                        text="",
                        bg=r.choice(hexes) if y else r.choice(["#f5f5f5", "#eeeeee", "#f4f8fa"]),
                    )
                ),
                None,
            ),
        )
        balanced(
            "typeface",
            n(10),
            ["G42", "G43"],
            ("GOV-FONT", "GOV-FONT-ALT"),
            lambda y: (
                lambda svc: (
                    base_el(
                        brand,
                        font=("GDS Transport" if svc else "Arial") if y else ("Arial" if svc else "GDS Transport"),
                    ),
                    "renew-a-permit.service.gov.uk" if svc else "renew-a-permit.campaign.gov.uk",
                )
            )(r.random() < 0.5),
        )
        balanced(
            "link colour",
            n(10),
            ["G62", "G65", "G10", "G12", "G9"],
            ("GOV-LINK-COL", "GOV-LINK-INV"),
            lambda y: (
                lambda dark: (
                    base_el(
                        brand,
                        role="link",
                        tag="a",
                        text="Read the guidance",
                        underline=True,
                        eff_bg="#0f385c" if dark else "#ffffff",
                        color=("#ffffff" if dark else "#1a65a6") if y else ("#d2e2f1" if dark else "#1d70b8"),
                    ),
                    None,
                )
            )(r.random() < 0.4),
        )
        btn = {
            "role": "button-default",
            "tag": "button",
            "color": "#ffffff",
            "bg": "#0f7a52",
            "eff_bg": "#0f7a52",
            "margin": [0.0] * 4,
            "padding": [10.0, 15.0, 10.0, 15.0],
        }
        balanced(
            "button case",
            n(10),
            ["G71"],
            ("GOV-BTN-CASE",),
            lambda y: (
                base_el(
                    brand,
                    **btn,
                    text=r.choice(["Save and continue", "Confirm and send", "Pay your VAT"])
                    if y
                    else r.choice(["Save And Continue", "CONFIRM AND SEND", "Start Now"]),
                ),
                None,
            ),
        )
        balanced(
            "alignment",
            n(5),
            ["G72"],
            ("GOV-BTN-ALIGN",),
            lambda y: (
                base_el(
                    brand,
                    **btn,
                    text="Continue",
                    form="e0",
                    form_x=160.0,
                    x=160.0 if y else float(r.choice([180, 190, 200])),
                ),
                None,
            ),
        )
        balanced(
            "warning",
            n(5),
            ["G79"],
            ("GOV-BTN-WARN",),
            lambda y: (
                base_el(
                    brand,
                    **dict(btn, role="button-warning", bg="#ca3535", eff_bg="#ca3535"),
                    text=r.choice(["Delete account", "Remove this item"])
                    if y
                    else r.choice(["Continue", "Save changes"]),
                ),
                None,
            ),
        )
    else:
        norm = spec["tokens"]["normalization"]

        def size_el(y):
            font = r.choice(["Source Sans Pro", "Merriweather", "Public Sans"])
            t = r.choice([15, 16, 17, 20, 22, 24, 32])
            s = round(t * norm[font], 2) if y else round(t * norm[font] + r.choice([-1.0, 1.0, 1.5]), 2)
            return base_el(brand, font=font, size=s, line_height=None), None

        balanced("normalized size", n(20), ["U47", "U50", "U51"], ("USW-SIZE",), size_el)
        balanced(
            "spacing",
            n(15),
            ["U43", "U44"],
            ("USW-SPACE",),
            lambda y: (
                base_el(
                    brand,
                    margin=[0.0, 0.0, float(r.choice([8, 16, 24, 32]) if y else r.choice([10, 15, 18, 30, 36])), 0.0],
                ),
                None,
            ),
        )
        hexes = sorted({p["hex"] for p in spec["tokens"]["palette"]})
        balanced(
            "palette",
            n(10),
            ["U1", *spec["tokens"]["palette_lines"]],
            ("USW-PAL",),
            lambda y: (
                (
                    base_el(
                        brand,
                        color=r.choice(["#1b1b1b", "#005ea2", "#1a4480"])
                        if y
                        else r.choice(["#1a1a1a", "#333333", "#0060a0"]),
                    )
                    if r.random() < 0.5
                    else base_el(
                        brand,
                        role="container",
                        tag="div",
                        text="",
                        bg=r.choice(hexes) if y else r.choice(["#f2f2f2", "#eeeeee", "#e8f0f8"]),
                    )
                ),
                None,
            ),
        )
        balanced(
            "font family",
            n(10),
            ["U52"],
            ("USW-FAMILY",),
            lambda y: (
                base_el(
                    brand,
                    font=r.choice(["Public Sans", "Merriweather", "Source Sans Pro"])
                    if y
                    else r.choice(["Lato", "Arial", "Roboto"]),
                    size=17.0,
                ),
                None,
            ),
        )
        balanced(
            "pure black",
            n(5),
            ["U13"],
            ("USW-BLACK",),
            lambda y: (base_el(brand, color="#1b1b1b" if y else "#000000"), None),
        )
        btn = {
            "role": "button-default",
            "tag": "button",
            "color": "#ffffff",
            "bg": "#005ea2",
            "eff_bg": "#005ea2",
            "margin": [0.0] * 4,
            "padding": [12.0, 20.0, 12.0, 20.0],
        }
        balanced(
            "button case",
            n(5),
            ["U58"],
            ("USW-BTN-CASE",),
            lambda y: (
                base_el(
                    brand,
                    **btn,
                    text=r.choice(["Sign up", "File a complaint", "Download the form"])
                    if y
                    else r.choice(["Sign Up", "FILE A COMPLAINT", "Download The Form"]),
                ),
                None,
            ),
        )
        balanced(
            "lead verb",
            n(5),
            ["U60"],
            ("USW-BTN-VERB",),
            lambda y: (
                base_el(
                    brand,
                    **btn,
                    text=r.choice(["File a complaint", "Download", "Sign up"])
                    if y
                    else r.choice(["Complaint filing", "Next step", "Application status"]),
                ),
                None,
            ),
        )
    r.shuffle(items)
    for i, x in enumerate(items):
        x.update(id=f"t3-{brand}-{split}-{i:03d}", brand=brand, split=split, facts=describe_el(x["element"]) or "")
    for i, x in enumerate(controls):
        x.update(
            id=f"t3c-{brand}-{split}-{i:03d}",
            brand=brand,
            split=split,
            facts=(describe_el(x["element"]) or "") + f" | contrast ratio {x['ratio']}:1",
        )
    return items, controls


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for brand in BRANDS:
        spec = load_spec(brand)
        for split in ("dev", "heldout"):
            outs = {"t1": t1(brand, split), "t2": t2(brand, split, spec)}
            outs["t3"], outs["t3c"] = t3(brand, split, spec)
            for name, rows in outs.items():
                path = DATA / f"{name}_{brand}_{split}.jsonl"
                write_jsonl(path, rows)
                manifest[path.name] = sha256(path)
            kinds: dict[str, int] = {}
            for x in outs["t3"]:
                kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
            print(
                brand,
                split,
                {k: len(v) for k, v in outs.items()},
                "t2 exceptions",
                sum(x["exception"] for x in outs["t2"]),
                "t3 yes",
                sum(x["answer"] == "yes" for x in outs["t3"]),
                kinds,
            )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
