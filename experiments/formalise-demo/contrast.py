"""The bridge: the same treatment for a tiny design-system rulebook (GOV.UK colour rules + WCAG 2.2 contrast).

    .venv/bin/python experiments/formalise-demo/contrast.py

Source: vendor/design_rules/source.txt (verbatim lines from the GOV.UK Design System colour page and WCAG 2.2; their
licences are on the header lines). Writes spec/contrast_spec.json (rules traced to source lines D<n>, plus the two
interpretations the text does not settle), spec/contrast_rules.txt (the pasted text), data/contrast_dev.jsonl (30)
and data/contrast_heldout.jsonl (120), each with the checker's verdict, and their hashes in the manifest.

The checker decides three rules: the palette (D2), the focus colour used only for focus (D11) and text contrast
(D31 4.5:1, D32 3:1 for large text, exempt under D33 or D34). A question is compliant when none fails.
"""

from __future__ import annotations

import json
import random
import re

from common import DATA, HERE, SPEC_DIR, sha256, write_jsonl

SOURCE = HERE / "vendor" / "design_rules" / "source.txt"
LINES = SOURCE.read_text(encoding="utf-8").rstrip("\n").split("\n")
PALETTE = sorted({h for n in [*range(5, 18), *range(19, 30)] for h in re.findall(r"#[0-9a-f]{6}", LINES[n - 1])})
FOCUS = "#ffdd00"
LARGE_PX, LARGE_BOLD_PX = 24.0, 14 * 4 / 3  # X-PX: 18 pt and 14 pt at 4/3 CSS px per point
EXEMPT = {"disabled": "D33", "inactive": "D33", "logo": "D34"}  # X-KIND: a word in the element's name
SPLITS = {"dev": (11, 30), "heldout": (12, 120)}

RULES = [
    dict(id="D2", line=2, kind="palette", role="condition", formal="Every colour used is a palette colour."),
    dict(
        id="D3", line=3, kind="contrast_scope", role="condition", formal="Text must meet WCAG 2.2 SC 1.4.3 (D31-D37)."
    ),
    dict(
        id="D11",
        line=11,
        kind="focus_only",
        role="condition",
        formal="The focus colour #ffdd00 appears only on an element in its focus state.",
    ),
    dict(
        id="D31", line=31, kind="contrast", role="condition", formal="Text: contrast ratio >= 4.5.", params={"min": 4.5}
    ),
    dict(
        id="D32",
        line=32,
        kind="contrast",
        role="exception",
        overrides="D31",
        formal="Large-scale text: contrast ratio >= 3.",
        params={"min": 3.0},
    ),
    dict(
        id="D33",
        line=33,
        kind="contrast",
        role="exception",
        overrides="D31",
        formal="Text in an inactive user interface component: no contrast requirement.",
    ),
    dict(
        id="D34",
        line=34,
        kind="contrast",
        role="exception",
        overrides="D31",
        formal="Text in a logo: no contrast requirement.",
    ),
    dict(
        id="D35",
        line=35,
        kind="large_scale",
        role="definition",
        formal="Large scale = at least 18 pt, or at least 14 pt and bold.",
    ),
    dict(id="D36", line=36, kind="ratio", role="definition", formal="ratio = (L_lighter + 0.05) / (L_darker + 0.05)."),
    dict(
        id="D37",
        line=37,
        kind="luminance",
        role="definition",
        formal="L = 0.2126 R + 0.7152 G + 0.0722 B on linearised sRGB channels (threshold 0.04045, exponent 2.4).",
    ),
    dict(
        id="X-PX",
        kind="units",
        role="interpretation",
        source={"file": "https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html"},
        formal="Points to CSS pixels at 4/3 px per pt: large = at least 24 px, or at least 18.67 px and bold. The "
        "normative text gives points only; the conversion comes from WCAG's non-normative Understanding page.",
    ),
    dict(
        id="X-KIND",
        kind="element_map",
        role="interpretation",
        formal="An element named as disabled or inactive is text in an inactive user interface component (D33); one "
        "named as a logo is a logotype (D34). Every other element in the questions is ordinary text.",
    ),
]


def luminance(hex_colour: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def ratio(fg: str, bg: str) -> float:
    hi, lo = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def is_large(size_px: float, bold: bool) -> bool:
    return size_px >= LARGE_PX or (bold and size_px >= LARGE_BOLD_PX - 1e-9)


def category(kind: str) -> str:
    """X-KIND: which contrast rule an element's name puts it under (exempt, or ordinary text)."""
    return next((rule for word, rule in EXEMPT.items() if word in kind.lower()), "text")


def contrast_rule(kind: str, size_px: float, bold: bool) -> str:
    exempt = category(kind)
    return exempt if exempt != "text" else ("D32" if is_large(size_px, bold) else "D31")


def evaluate(facts: dict) -> dict:
    """facts: {"element", "size_px", "bold", "focused", "text_colour", "background"}. Returns the proof."""
    fg, bg = facts["text_colour"].lower(), facts["background"].lower()
    off = [c for c in (fg, bg) if c not in PALETTE]
    focus_misuse = FOCUS in (fg, bg) and not facts["focused"]
    applies = contrast_rule(facts["element"], facts["size_px"], facts["bold"])
    r = ratio(fg, bg)
    need = {"D31": 4.5, "D32": 3.0}.get(applies)
    steps = [
        {"step": "palette", "rules": ["D2"], "inputs": {"colours": [fg, bg]}, "pass": not off, "where": off},
        {"step": "focus", "rules": ["D11"], "inputs": {"focused": facts["focused"]}, "pass": not focus_misuse},
        {
            "step": "which contrast rule",
            "rules": [applies, "D35", "X-PX", "X-KIND"],
            "inputs": {"element": facts["element"], "size_px": facts["size_px"], "bold": facts["bold"]},
            "value": applies,
        },
        {
            "step": "ratio",
            "rules": ["D36", "D37"],
            "inputs": {"text_colour": fg, "background": bg},
            "value": round(r, 2),
        },
        {
            "step": "contrast",
            "rules": [applies],
            "inputs": {"ratio": round(r, 2), "minimum": need},
            "pass": need is None or r >= need,
        },
    ]
    failed = sorted({s["rules"][0] for s in steps if s.get("pass") is False}, key=lambda x: int(x[1:]))
    return {
        "facts": facts,
        "steps": steps,
        "contrast_rule": applies,
        "ratio": round(r, 2),
        "rules_failed": failed,
        "verdict": "not compliant" if failed else "compliant",
    }


def verify(trace: dict, facts: dict | None = None) -> dict:
    """Check a model's contrast trace against the spec evaluated on the model's own facts."""
    checks = []
    if facts is not None:
        keys = ("element", "size_px", "bold", "focused", "text_colour", "background")

        def same(k: str) -> bool:
            if k == "element":
                return category(str(trace["facts"].get(k))) == category(facts[k])
            return str(trace["facts"].get(k)).lower() == str(facts[k]).lower()

        diffs = [k for k in keys if not same(k)]
        checks.append({"step": "facts", "pass": not diffs, "where": diffs})
    own = evaluate(trace["facts"])
    claimed_ratio = trace.get("ratio")
    checks += [
        {
            "step": "which contrast rule",
            "claimed": trace.get("contrast_rule"),
            "expected": own["contrast_rule"],
            "pass": trace.get("contrast_rule") == own["contrast_rule"],
        },
        {
            "step": "ratio",
            "claimed": claimed_ratio,
            "expected": own["ratio"],
            "pass": isinstance(claimed_ratio, (int, float)) and abs(claimed_ratio - own["ratio"]) <= 0.05,
        },
        {
            "step": "rules failed",
            "claimed": trace.get("rules_failed"),
            "expected": own["rules_failed"],
            "pass": sorted(trace.get("rules_failed") or [], key=lambda r: int(r[1:])) == own["rules_failed"],
        },
        {
            "step": "verdict",
            "claimed": trace.get("verdict"),
            "expected": own["verdict"],
            "pass": trace.get("verdict") == own["verdict"],
        },
    ]
    failed = [c for c in checks if not c["pass"]]
    return {
        "verdict": "pass" if not failed else "fail",
        "first_failure": failed[0] if failed else None,
        "checks": checks,
        "checker": {k: own[k] for k in ("verdict", "rules_failed", "contrast_rule", "ratio")},
    }


# ----------------------------------------------------------------------------- questions

KINDS = {
    "body text": [(16, False), (19, False)],
    "secondary text": [(16, False), (19, False)],
    "link": [(16, False), (19, False)],
    "error message": [(16, True), (19, True)],
    "page heading": [(24, True), (27, True), (36, True)],
    "form label": [(18, True), (19, True)],
    "button label": [(19, False), (24, False)],
    "disabled button label": [(19, False)],
    "logo text": [(30, True)],
}
TEXTS = [
    "#0b0c0c",
    "#484949",
    "#1a65a6",
    "#ca3535",
    "#0f7a52",
    "#1d70b8",
    "#ffffff",
    "#858686",
    "#5694ca",
    "#d76868",
    "#4b9b7d",
    "#158187",
    "#f47738",
    "#7f65b7",
    "#b7592a",
    "#99704a",
    "#ca357c",
    "#50a1a5",
]
BACKS = [
    "#ffffff",
    "#f4f8fb",
    "#f3f3f3",
    "#d2e2f1",
    "#cfe4dc",
    "#1d70b8",
    "#0f7a52",
    "#0b0c0c",
    "#ca3535",
    "#fff8cc",
    "#f4d7d7",
    "#ddd6ec",
    "#0f385c",
    "#54319f",
]


def grey_near(target: float, bg: str, pick: random.Random) -> str:
    """An off-palette grey whose ratio against bg is within 0.3 of the target."""
    candidates = [f"#{v:02x}{v:02x}{v:02x}" for v in range(40, 200)]
    close = [c for c in candidates if abs(ratio(c, bg) - target) <= 0.3 and c not in PALETTE]
    return pick.choice(close)


def contrast_question(pick: random.Random) -> dict:
    kind = pick.choice(list(KINDS))
    size, bold = pick.choice(KINDS[kind])
    focused = kind in ("link", "button label") and pick.random() < 0.25
    mode = pick.random()
    if focused:
        fg, bg = pick.choice(["#0b0c0c", "#1a65a6", "#484949", "#ca3535"]), FOCUS
    elif mode < 0.10:
        fg, bg = pick.choice(["#0b0c0c", "#484949", "#1a65a6"]), FOCUS  # the focus colour outside focus
    elif mode < 0.25:
        bg = pick.choice(["#ffffff", "#f4f8fb", "#f3f3f3"])
        fg = grey_near(pick.choice([3.0, 4.5]), bg, pick)
    elif mode < 0.55:
        threshold = {"D31": 4.5, "D32": 3.0}.get(contrast_rule(kind, size, bold), pick.choice([3.0, 4.5]))
        pairs = [(f, b) for f in TEXTS for b in BACKS if f != b and abs(ratio(f, b) - threshold) <= 0.8]
        fg, bg = pick.choice(pairs)
    else:
        fg, bg = pick.choice(TEXTS), pick.choice(BACKS)
        while fg == bg:
            bg = pick.choice(BACKS)
    facts = {"element": kind, "size_px": size, "bold": bold, "focused": focused, "text_colour": fg, "background": bg}
    weight = "bold" if bold else "regular"
    state = ", shown in its keyboard-focus state" if focused else ""
    prompt = (
        f"A service built with the GOV.UK Design System shows a {kind} set in {size}px {weight} type{state}. "
        f"The text colour is {fg} and the background colour is {bg}."
    )
    return {"prompt": prompt, "facts": facts}


def main() -> None:
    SPEC_DIR.mkdir(exist_ok=True)
    for r in RULES:
        if "line" in r:
            r["text"] = LINES[r["line"] - 1]
    spec = {
        "name": "contrast-slice",
        "source": {"file": "vendor/design_rules/source.txt"},
        "palette": PALETTE,
        "rules": RULES,
    }
    (SPEC_DIR / "contrast_spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    pasted = "\n".join(f"D{n}: {text}" for n, text in enumerate(LINES, 1)) + "\n"
    (SPEC_DIR / "contrast_rules.txt").write_text(pasted, encoding="utf-8")
    manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
    for split, (seed, n) in SPLITS.items():
        pick = random.Random(seed)
        rows = []
        for i in range(n):
            q = contrast_question(pick)
            proof = evaluate(q["facts"])
            rows.append(
                {
                    "id": f"c{split}-{i:03d}",
                    **q,
                    "answer": {k: proof[k] for k in ("verdict", "rules_failed", "contrast_rule", "ratio")},
                }
            )
        path = DATA / f"contrast_{split}.jsonl"
        write_jsonl(path, rows)
        manifest[path.name] = {"n": n, "seed": seed, "sha256": sha256(path)}
        verdicts = [r["answer"]["verdict"] for r in rows]
        print(f"contrast {split}: {n} questions, {verdicts.count('compliant')} compliant")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"palette: {len(PALETTE)} colours; spec: {len(RULES)} rules")


if __name__ == "__main__":
    main()
