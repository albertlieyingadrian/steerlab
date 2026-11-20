"""Check every ruler against its pass bar before any number is trusted (plan, Design item 3).

    .venv/bin/python experiments/formalise-demo/ruler.py

1. Spec lint: every input the slice allows is decided by exactly one fee rule per component.
2. The Python checker against RuleArena's own answer code on 5,000 random in-slice passengers (bar 5,000/5,000).
3. The JS port (report/checker.js, run under node) against the Python checker on the same 5,000 (bar 5,000/5,000,
   totals and the rule list of every step).
4. The reply parser on 20 hand-written replies with known parses (bar 20/20).
5. The contrast ratio on 10 colour pairs with published ratios (WebAIM), within 0.01 (bar 10/10).
Writes results/ruler.json and exits 1 if any bar is missed.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from checker import _index, evaluate
from common import HERE, RESULTS, load_spec
from contrast import ratio
from formalise import lint
from parse import parse
from questions import question, reference_total

N = 5000


def spec_lint() -> dict:
    problems = lint([r for rules in _index().values() for r in rules])
    return {"pass": not problems, "problems": problems}


def random_cases() -> list[dict]:
    rng, pick = np.random.default_rng(3), random.Random(3)
    return [question(rng, pick)["info"] for _ in range(N)]


def vs_reference(cases: list[dict]) -> dict:
    bad = [i for i, info in enumerate(cases) if evaluate(info)["total"] != reference_total(info)]
    return {"pass": not bad, "agree": N - len(bad), "n": N, "first_disagreements": bad[:5]}


NODE = """
const fs = require("fs");
const BaggageChecker = require(process.argv[2]);
const spec = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const cases = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const c = BaggageChecker(spec);
const out = cases.map((f) => { const p = c.evaluateChecked(f.class, f.region, f.direction, f.ticket, f.bags);
  return { total: p.total, rules: p.steps.map((s) => s.rules) }; });
process.stdout.write(JSON.stringify(out));
"""


def vs_js(cases: list[dict]) -> dict:
    facts = []
    for info in cases:
        p = evaluate(info)
        facts.append({**p["facts"], "py": {"total": p["total"], "rules": [s["rules"] for s in p["steps"][1:]]}})
    with tempfile.TemporaryDirectory() as d:
        js, data = f"{d}/run.js", f"{d}/cases.json"
        Path(js).write_text(NODE)
        Path(data).write_text(json.dumps(facts))
        spec_path = HERE / "spec" / "baggage_spec.json"
        out = subprocess.run(
            ["node", js, str(HERE / "report" / "checker.js"), str(spec_path), data],
            capture_output=True,
            text=True,
            check=True,
        )
    js_out = json.loads(out.stdout)
    bad = [
        i
        for i, (f, j) in enumerate(zip(facts, js_out, strict=True))
        if f["py"]["total"] != j["total"] or f["py"]["rules"] != j["rules"]
    ]
    return {"pass": not bad, "agree": N - len(bad), "n": N, "first_disagreements": bad[:5]}


TRACE = '```json\n{"class": "Main Cabin Class", "region": "US", "direction": "to US", "ticket": "$180", "checked_bags": [{"item": 2, "position": 1, "size": 66, "weight": 52, "base_line": "Line 38", "base_fee": "$40", "weight_line": 288, "weight_fee": 30, "size_line": "L361", "size_fee": 200, "bag_fee": 240,}], "total": 420}\n```'
PARSER_CASES = [
    ("baggage", "<think>x</think>\nThe total cost is $455.", {"finished": True, "pred": 455, "trace": None}),
    ("baggage", "<think>long", {"finished": False, "pred": None, "trace": None}),
    ("baggage", "The total cost is $100. Wait, recount. The total cost is $1,235.", {"pred": 1235}),
    ("baggage", "So the total cost is **$300**.", {"pred": 300}),
    ("baggage", "The total cost is $455.50.", {"pred": 455.5}),
    ("baggage", "no sentence at all", {"pred": None}),
    (
        "baggage",
        f"</think>{TRACE}\nThe total cost is $420.",
        {
            "pred": 420,
            "trace.class": "Main Cabin",
            "trace.region": "U.S.",
            "trace.direction": 1,
            "trace.ticket": 180,
            "trace.total": 420,
        },
    ),
    (
        "baggage",
        f"</think>{TRACE}",
        {
            "trace.bags.0.base_line": "L38",
            "trace.bags.0.weight_line": "L288",
            "trace.bags.0.size_line": "L361",
            "trace.bags.0.base_fee": 40,
        },
    ),
    (
        "baggage",
        '</think>```json\n{"class": "First", "region": "Japan", "direction": "from_us", "ticket": 400, "checked_bags": []}\n```',
        {"trace.region": "Japan", "trace.direction": 0, "trace.bags": []},
    ),
    (
        "baggage",
        '</think>```json\n{"class": "First", "region": "Canada", "checked_bags": []}\n```',
        {"trace.region": None},
    ),
    ("baggage", '</think>```json\n{"class": "First", "region": "Japan"}\n```', {"trace": None}),
    (
        "baggage",
        '</think>Trace: {"class": "First", "region": "Japan", "checked_bags": []} The total cost is $5.',
        {"trace.region": "Japan", "pred": 5},
    ),
    ("baggage", f'</think>{TRACE}\n```json\n{{"broken": \n```', {"trace.total": 420}),
    (
        "baggage",
        '</think>```json\n{"class": "Business", "region": "Europe", "direction": "from_us", "ticket": 300, '
        '"checked_bags": [{"item": 2, "size": 60, "weight": 40}]}\n```',
        {"trace.bags.0.base_line": None, "trace.total": None},
    ),
    ("contrast", "</think>All good.\nVerdict: not compliant", {"pred": "not compliant", "trace": None}),
    ("contrast", "</think>**Verdict:** compliant", {"pred": "compliant"}),
    (
        "contrast",
        '</think>```json\n{"facts": {"element": "Link", "size_px": "19px", "bold": false, "focused": false, '
        '"text_colour": "#1A65A6", "background": "#ffffff"}, "contrast_rule": "31", "ratio": 6.05, '
        '"rules_failed": ["D31", "31"], "verdict": "Not compliant"}\n```\nVerdict: not compliant',
        {
            "trace.facts.element": "link",
            "trace.facts.size_px": 19,
            "trace.facts.text_colour": "#1a65a6",
            "trace.contrast_rule": "D31",
            "trace.rules_failed": ["D31"],
            "trace.verdict": "not compliant",
        },
    ),
    ("contrast", '</think>```json\n{"contrast_rule": "D31", "verdict": "compliant"}\n```', {"trace": None}),
    (
        "contrast",
        '</think>```json\n{"facts": {"element": "logo text", "size_px": 30, "bold": true, "focused": false, '
        '"text_colour": "#ffffff", "background": "#ffdd00"}, "contrast_rule": "D34", "ratio": 1.4, "rules_failed": '
        '["D11"], "verdict": "not compliant"}\n```',
        {"trace.rules_failed": ["D11"], "trace.facts.bold": True},
    ),
    ("contrast", "<think>still thinking", {"finished": False, "pred": None}),
]


def get(obj, path: str):
    for part in path.split("."):
        if obj is None:
            return None
        obj = obj[int(part)] if isinstance(obj, list) else obj.get(part)
    return obj


def parser_cases() -> dict:
    fails = []
    for i, (task, text, want) in enumerate(PARSER_CASES):
        got = parse(task, text)
        for path, value in want.items():
            if get(got, path) != value:
                fails.append(f"case {i}: {path} = {get(got, path)!r}, want {value!r}")
    return {
        "pass": not fails,
        "agree": len(PARSER_CASES) - len({f.split(":")[0] for f in fails}),
        "n": len(PARSER_CASES),
        "fails": fails,
    }


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


def contrast_known() -> dict:
    rows = [{"fg": f, "bg": b, "published": w, "ours": round(ratio(f, b), 2)} for f, b, w in KNOWN_RATIOS]
    ok = [abs(r["ours"] - r["published"]) <= 0.01 for r in rows]
    return {"pass": all(ok), "agree": sum(ok), "n": len(rows), "pairs": rows}


def main() -> None:
    cases = random_cases()
    out = {
        "spec": load_spec()["name"],
        "spec_lint": spec_lint(),
        "checker_vs_reference": vs_reference(cases),
        "js_vs_python": vs_js(cases),
        "parser": parser_cases(),
        "contrast_ratio": contrast_known(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ruler.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, r in out.items():
        if isinstance(r, dict):
            print(
                f"{name}: {'PASS' if r['pass'] else 'FAIL'} {r.get('agree', '')}/{r.get('n', '')} {r.get('fails', '') or ''}"
            )
    sys.exit(0 if all(r["pass"] for r in out.values() if isinstance(r, dict)) else 1)


if __name__ == "__main__":
    main()
