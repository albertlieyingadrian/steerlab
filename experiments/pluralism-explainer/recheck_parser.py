"""Re-check the parser on the replies the models actually gave (Design 3): among replies it marked invalid, count those
that still name an option ("B.", "(C)", "option D", "**E**"), and print a sample of them to read.

    python recheck_parser.py     # writes results/parser_recheck.json; bar: missed answers < 1% of all replies
"""

from __future__ import annotations

import gzip
import json
import random
import re

import plural

NAMES = re.compile(r"(?:^|\s|\()(?:\*\*)?([A-J])(?:\*\*)?(?:[.)]|\s*[:\-]\s)|\boption\s+([A-J])\b|\bletter\s+([A-J])\b")


def main() -> None:
    qs = {q["id"]: q for q in plural.load_questions()}
    out, rng = {}, random.Random(0)
    for path in sorted((plural.HERE / "results" / "samples" / "main").glob("*.jsonl.gz")):
        total = invalid = flagged = 0
        examples = []
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            k = len(qs[r["id"]]["options"])
            for t in r["replies"]:
                total += 1
                if plural.parse(t, k) >= 0:
                    continue
                invalid += 1
                if t and NAMES.search(t):
                    flagged += 1
                    if len(examples) < 400:
                        examples.append(t[:160])
        name = path.name.removesuffix(".jsonl.gz")
        out[name] = {"replies": total, "invalid": invalid, "flagged_names_option": flagged,
                     "flagged_share_of_all": round(flagged / total, 5),
                     "sample": rng.sample(examples, min(8, len(examples)))}  # fmt: skip
        print(f"{name:22s} invalid {invalid / total:.3f}  flagged {flagged} ({flagged / total:.4f} of all)")
        for e in out[name]["sample"][:4]:
            print("     ", repr(e[:110]))
    (plural.HERE / "results" / "parser_recheck.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
