"""Every pass bar from the plan's Design section 3 ("the ruler is a hypothesis").

    .venv/bin/python experiments/brand-spec/ruler.py [--real]

1. Checker vs planted answers on every generated page (compliance, copy and demo sets): same verdict and the same
   set of failed (element, rule) pairs. Bar: all pages.
2. Extractor vs the generator's intended facts (reference labels by construction): role, colour, background,
   effective background, border, font, size, weight, line height, margin, padding, underline, disabled, region for
   every element of every page. Bar: all fields.
3. JS port (report/checker.js) vs the Python checker on every page and 2,000 single-property mutations: same pairs
   and same (element, rule, property, pass) for every check. Bar: all.
4. Contrast ratio on 10 published pairs (WebAIM), within 0.01. Bar: 10/10.
5. The brand book's magic-number claim (U9: a grade difference of 50+ gives AA, 4.5:1) checked on every pair of
   USWDS system colours, and U8 (40+ gives 3:1). Reported, not a bar: counterexamples are a finding about the source.
6. The reply parser on 20 written replies with known parses. Bar: 20/20.
7. --real: the extractor and checker on the two brands' own documentation home pages (one load each). Reported.
Writes results/ruler.json; exits 1 if a bar is missed.
"""

from __future__ import annotations

import copy
import gzip
import json
import random
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

from checker import check
from common import BRANDS, DATA, HERE, RESULTS, contrast, load_spec, read_jsonl, write_json
from parse import parse_t1, parse_t2, parse_t3

SETS = [f"{k}_{b}_{s}" for k in ("pages", "copy") for b in BRANDS for s in ("dev", "heldout")] + ["demo_govuk"]
FIELDS = [
    "role",
    "color",
    "bg",
    "eff_bg",
    "border",
    "font",
    "size",
    "weight",
    "line_height",
    "margin",
    "padding",
    "underline",
    "disabled",
    "region",
]
KNOWN_RATIOS = [
    ("#000000", "#ffffff", 21.0),
    ("#767676", "#ffffff", 4.54),
    ("#777777", "#ffffff", 4.48),
    ("#595959", "#ffffff", 7.0),
    ("#949494", "#ffffff", 3.03),
    ("#0000ff", "#ffffff", 8.59),
    ("#ff0000", "#ffffff", 4.0),
    ("#00ff00", "#ffffff", 1.37),
    ("#808080", "#ffffff", 3.95),
    ("#ffff00", "#000000", 19.56),
]
NODE = """
const fs = require('fs');
const BrandChecker = require(process.argv[2]);
const specs = JSON.parse(fs.readFileSync(process.argv[3]));
const pages = JSON.parse(fs.readFileSync(process.argv[4]));
const out = pages.map((p) => { const pr = BrandChecker(specs[p.brand]).check(p);
  return {pairs: pr.fail_pairs, checks: pr.checks.map((c) => [c.element, c.rule, c.property, c.pass])}; });
process.stdout.write(JSON.stringify(out));
"""


def load_all():
    for name in SETS:
        yield name, read_jsonl(DATA / f"{name}.jsonl"), read_jsonl(DATA / f"facts_{name}.jsonl")


def checker_vs_planted(specs) -> dict:
    n, off, bad = 0, 0, []
    for _name, pages, facts in load_all():
        for p, f in zip(pages, facts, strict=True):
            n += 1
            off += bool(p["labels"])
            got = [list(x) for x in check(specs[p["brand"]], f)["fail_pairs"]]
            if got != p["labels"]:
                bad.append({"page": p["id"], "labels": p["labels"], "checker": got})
    return {
        "pass": not bad,
        "pages": n,
        "pages_with_planted_violations": off,
        "agree": n - len(bad),
        "disagreements": bad[:10],
    }


def extractor_vs_intended() -> dict:
    with gzip.open(DATA / "intended.jsonl.gz", "rt", encoding="utf-8") as fh:
        meant = {r["id"]: r["intended"] for r in map(json.loads, fh)}
    n_el, n_fields, bad, examples = 0, 0, {}, []
    for _name, pages, facts in load_all():
        for p, f in zip(pages, facts, strict=True):
            F = {e["id"]: e for e in f["elements"]}
            for e in meant[p["id"]]:
                n_el += 1
                x = F.get(e["id"])
                for fld in FIELDS:
                    if e[fld] is None and fld in ("color", "font", "size", "line_height"):
                        continue  # not set on this element (inherited); the checker never reads it there
                    n_fields += 1
                    if x is None or x[fld] != e[fld]:
                        bad[fld] = bad.get(fld, 0) + 1
                        if len(examples) < 10:
                            examples.append(
                                {
                                    "page": p["id"],
                                    "element": e["id"],
                                    "field": fld,
                                    "intended": e[fld],
                                    "extracted": None if x is None else x[fld],
                                }
                            )
    return {"pass": not bad, "elements": n_el, "fields": n_fields, "mismatches": bad, "examples": examples}


def mutate(f: dict, r: random.Random, spec: dict) -> dict:
    g = copy.deepcopy(f)
    texty = [e for e in g["elements"] if e["text"]]
    e = r.choice(texty if texty and r.random() < 0.8 else g["elements"])
    pal = [p["hex"] for p in spec["tokens"]["palette"]]
    k = r.choice(["color", "eff_bg", "size", "weight", "font", "margin", "text", "underline", "disabled", "x", "bg"])
    if k in ("color", "eff_bg", "bg"):
        e[k] = r.choice(pal) if r.random() < 0.6 else f"#{r.randrange(1 << 24):06x}"
        if k == "eff_bg":
            e["parent_bg"] = e[k]
    elif k == "size":
        e["size"] = r.choice([12, 15.9, 16, 17.04, 18, 19, 19.08, 21.56, 24, 27, 31.36, 36, 48, 80])
    elif k == "weight":
        e["weight"] = r.choice([400, 700])
    elif k == "font":
        e["font"] = r.choice(
            [
                "GDS Transport",
                "New Transport",
                "Arial",
                "Helvetica",
                "Merriweather",
                "Source Sans Pro",
                "Public Sans",
                "Lato",
                "system-ui",
            ]
        )
    elif k == "margin":
        e["margin"][r.randrange(4)] = float(r.choice([0, 4, 5, 8, 10, 12, 15, 16, 20, 22, 24, 30, 35]))
    elif k == "text":
        e["text"] = r.choice(
            [
                "Save And Continue",
                "save and continue",
                "Read the guidance.",
                "Delete account",
                "Complaint filing",
                "OK",
                "Pay your VAT",
                "SIGN IN",
                "Sign in",
                e["text"] + ".",
            ]
        )
    elif k == "underline":
        e["underline"] = not e["underline"]
    elif k == "disabled":
        e["disabled"] = not e["disabled"]
    else:
        e["x"] = e["x"] + r.choice([0, 0.5, 2, 20])
    return g


def js_vs_python(specs) -> dict:
    base = [f for _, _, facts in load_all() for f in facts]
    r = random.Random(0)
    cases = base + [mutate(f, r, specs[f["brand"]]) for f in (r.choice(base) for _ in range(2000))]
    py = []
    for f in cases:
        pr = check(specs[f["brand"]], f)
        py.append(
            {
                "pairs": [list(x) for x in pr["fail_pairs"]],
                "checks": [[c["element"], c["rule"], c["property"], c["pass"]] for c in pr["checks"]],
            }
        )
    with tempfile.TemporaryDirectory() as d:
        Path(d, "run.js").write_text(NODE)
        Path(d, "specs.json").write_text(json.dumps(specs))
        Path(d, "pages.json").write_text(json.dumps(cases))
        out = subprocess.run(
            [
                "node",
                str(Path(d, "run.js")),
                str(HERE / "report" / "checker.js"),
                str(Path(d, "specs.json")),
                str(Path(d, "pages.json")),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    js = json.loads(out.stdout)
    bad = [i for i, (a, b) in enumerate(zip(py, js, strict=True)) if a != b]
    return {
        "pass": not bad,
        "cases": len(cases),
        "pages": len(base),
        "mutations": len(cases) - len(base),
        "mutated_cases_failing": sum(bool(x["pairs"]) for x in py[len(base) :]),
        "agree": len(cases) - len(bad),
        "first_disagreements": bad[:5],
    }


def contrast_known() -> dict:
    rows = [{"fg": f, "bg": b, "published": w, "ours": round(contrast(f, b), 2)} for f, b, w in KNOWN_RATIOS]
    ok = [abs(x["ours"] - x["published"]) <= 0.01 for x in rows]
    return {"pass": all(ok), "agree": sum(ok), "n": len(rows), "pairs": rows}


def magic_number(spec) -> dict:
    pal = spec["tokens"]["palette"]
    out = {}
    for need, diff in ((4.5, 50), (3.0, 40)):
        pairs = [(a, b) for a, b in combinations(pal, 2) if abs(a["grade"] - b["grade"]) >= diff]
        bad = [
            (a["name"], b["name"], round(contrast(a["hex"], b["hex"]), 3))
            for a, b in pairs
            if contrast(a["hex"], b["hex"]) < need
        ]
        out[f"diff_{diff}_gives_{need}"] = {
            "pairs": len(pairs),
            "counterexamples": len(bad),
            "worst": sorted(bad, key=lambda x: x[2])[:8],
        }
    below = [(a, b) for a, b in combinations(pal, 2) if abs(a["grade"] - b["grade"]) < 50]
    out["diff_under_50_reaching_4.5"] = {
        "pairs": len(below),
        "reach_4.5": sum(contrast(a["hex"], b["hex"]) >= 4.5 for a, b in below),
    }
    return out


PARSER_CASES = [
    ("t1", '<think>a</think>\n```json\n[{"element": "e12", "line": "G71"}]\n```', [("e12", "G71")]),
    ("t1", "<think>a</think>\nNo problems.\n```json\n[]\n```", []),
    ("t1", "<think>never finished", None),
    ("t1", '<think>a</think>\n```json\n[{"element": "e3", "line": "G1"},]\n```', [("e3", "G1")]),
    (
        "t1",
        '<think>a</think> first ```json\n[{"element": "e1", "line": "G2"}]\n``` then final '
        '```json\n[{"element": "e4", "line": "W2"}]\n```',
        [("e4", "W2")],
    ),
    ("t1", '<think>a</think>\n[{"element": "e9", "line": "Line U58"}]', [("e9", "U58")]),
    ("t1", '<think>a</think>\n```json\n[{"element": "e5", "reason": "too small"}]\n```', [("e5", "too small")]),
    ("t1", "<think>a</think>\nI cannot tell.", None),
    ("t2", "<think>a</think>\nThe deciding line is W3.\nLine: W3", "W3"),
    ("t2", "<think>a</think>\nLine: **G72**", "G72"),
    ("t2", "<think>it is G1 or G2</think>\nAnswer G42", "G42"),
    ("t2", "<think>long thinking", None),
    ("t2", "<think>a</think>\nline: u5", "U5"),
    ("t2", "<think>a</think>\nNo line applies.", None),
    ("t3", "<think>a</think>\nAnswer: yes", "yes"),
    ("t3", "<think>a</think>\nIt fails.\nAnswer: **No**", "no"),
    ("t3", "<think>yes yes</think>\nSo no.", "no"),
    ("t3", "<think>cut off", None),
    ("t3", "<think>a</think>\nAnswer: no. (Earlier I thought yes.)", "no"),
    ("t3", "<think>a</think>\nThe contrast is 4.6, so it complies. Answer: yes", "yes"),
]


def parser_cases() -> dict:
    fns = {"t1": parse_t1, "t2": parse_t2, "t3": parse_t3}
    rows = [{"task": t, "want": w, "got": fns[t](text)} for t, text, w in PARSER_CASES]
    ok = [x["want"] == x["got"] for x in rows]
    return {
        "pass": all(ok),
        "agree": sum(ok),
        "n": len(rows),
        "failures": [x for x, k in zip(rows, ok, strict=True) if not k],
    }


def real_pages(specs) -> dict:
    from render import extract_url

    out = {}
    for brand, url, domain in (
        ("govuk", "https://design-system.service.gov.uk/", "design-system.service.gov.uk"),
        ("uswds", "https://designsystem.digital.gov/", "designsystem.digital.gov"),
    ):
        try:
            f = extract_url(url, brand, domain)
        except Exception as exc:  # network or site change: reported, never a bar
            out[brand] = {"url": url, "error": str(exc)[:200]}
            continue
        pr = check(specs[brand], f)
        by_rule: dict[str, list[int]] = {}
        for c in pr["checks"]:
            by_rule.setdefault(c["rule"], [0, 0])
            by_rule[c["rule"]][0] += c["pass"]
            by_rule[c["rule"]][1] += 1
        examples: dict[str, list] = {}
        for c in pr["failures"]:
            ex = examples.setdefault(c["rule"], [])
            if len(ex) < 4:
                ex.append({"role": c["role"], "value": c["value"], "why": c["why"]})
        out[brand] = {
            "url": url,
            "elements": len(f["elements"]),
            "checks": len(pr["checks"]),
            "failures": len(pr["failures"]),
            "by_rule": by_rule,
            "examples": examples,
        }
    return out


def main() -> None:
    specs = {b: load_spec(b) for b in BRANDS}
    out = {
        "checker_vs_planted": checker_vs_planted(specs),
        "extractor_vs_intended": extractor_vs_intended(),
        "js_vs_python": js_vs_python(specs),
        "contrast_ratio": contrast_known(),
        "parser": parser_cases(),
        "magic_number_claim": magic_number(specs["uswds"]),
    }
    prev = RESULTS / "ruler.json"
    if "--real" in sys.argv:
        out["real_pages"] = real_pages(specs)
    elif prev.exists() and "real_pages" in json.loads(prev.read_text()):
        out["real_pages"] = json.loads(prev.read_text())["real_pages"]
    write_json(RESULTS / "ruler.json", out)
    failed = [k for k, v in out.items() if isinstance(v, dict) and v.get("pass") is False]
    for k, v in out.items():
        keep = ("pass", "agree", "pages", "cases", "fields", "n", "mismatches", "mutated_cases_failing")
        brief = {kk: vv for kk, vv in v.items() if kk in keep}
        print(k, brief if brief else json.dumps(v)[:400])
    if failed:
        print("FAILED BARS:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
