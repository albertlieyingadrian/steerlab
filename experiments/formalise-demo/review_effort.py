"""An illustration (not a user study) of "review shifts from redoing the work to checking the proof".

    .venv/bin/python experiments/formalise-demo/review_effort.py

For each held-out baggage question it counts two things a reviewer would have to handle:
  redo:  recompute the fee from the rulebook. Every rule line in each section the question touches (the base-fee table
         for each bag position used, the weight allowances and overweight bands, the size allowance and oversize bands,
         the combination rule and the carry-on lines), each of which must be read to find the one that applies, plus
         every arithmetic operation (per bag one max and two additions, then the sum with the ticket).
  check: check the proof artifact. For every step, the rule lines it cites, each input it read and the value it
         decided: each is one field to compare against the quoted source line, nothing to look up.
Writes results/review_effort.json (per-question counts and their medians).
"""

from __future__ import annotations

import json
import statistics

from checker import _index, evaluate
from common import DATA, RESULTS, read_jsonl

CARRY_ON = ("L5", "L20")
ALLOWANCES = ("L273", "L274", "L275", "L279")


def section_lines() -> dict[str, set[str]]:
    idx = _index()
    out = {f"base{p}": {r["id"] for r in idx["base_fee"] if r["when"]["position"] == p} for p in (1, 2, 3)}
    out["base1"] |= {"L103"}
    out["base2"] |= {"L164"}
    out["weight"] = {r["id"] for r in idx["weight_fee"]} | set(ALLOWANCES[1:])
    out["size"] = {r["id"] for r in idx["size_fee"]} | {"L273", "L270"}
    out["combine"] = {"L283"}
    out["carry"] = set(CARRY_ON)
    return out


def counts(info: dict, sections: dict[str, set[str]]) -> dict:
    proof = evaluate(info)
    bags = [s for s in proof["steps"] if s["step"] == "base"]
    touched = {"weight", "size", "combine", "carry"} | {f"base{s['position']}" for s in bags}
    redo_lines = sum(len(sections[k]) for k in touched)
    redo_ops = 3 * len(bags) + 1
    check = 0
    for s in proof["steps"]:
        check += len([r for r in s["rules"] if r != "L270"]) + len(s.get("inputs", {})) + 1
    return {
        "bags": len(bags),
        "redo_lines": redo_lines,
        "redo_ops": redo_ops,
        "redo": redo_lines + redo_ops,
        "check": check,
    }


def main() -> None:
    sections = section_lines()
    rows = [{"id": q["id"], **counts(q["info"], sections)} for q in read_jsonl(DATA / "baggage_heldout.jsonl")]
    med = {k: statistics.median(r[k] for r in rows) for k in ("redo_lines", "redo_ops", "redo", "check")}
    by_bags = {
        n: {k: statistics.median(r[k] for r in rows if r["bags"] == n) for k in ("redo", "check")} for n in (1, 2, 3)
    }
    out = {
        "note": "Illustration, not a user study: counts of things to read or compute, not reviewer time.",
        "median": med,
        "median_by_checked_bags": by_bags,
        "questions": rows,
    }
    (RESULTS / "review_effort.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"median": med, "by_bags": by_bags}))


if __name__ == "__main__":
    main()
