"""Known-answer checks for the ruler: the parser on 26 written replies, and the metrics on hand-checkable cases.

    python check_ruler.py      # exits 1 on any failure; writes results/ruler_check.json

Pass bar (fixed before the main run): every case right.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

import plural

PARSE_CASES = [  # (reply, number of options, expected index; -1 = invalid)
    ("B", 3, 1),
    ("B.", 3, 1),
    (" B\n\nQuestion: next", 3, 1),
    ("(C)", 4, 2),
    ("**A**", 2, 0),
    ("Answer: D", 5, 3),
    ("answer: d", 5, -1),
    ("A. Good thing", 3, 0),
    ("J", 10, 9),
    ("J", 4, -1),
    ("I", 10, 8),
    ("I'm an AI, so I don't have a job.", 10, -1),
    ("I can't give a personal opinion.", 4, -1),
    ("As an AI, I do not have personal opinions.", 4, -1),
    ("The closest accurate answer is J.\n\nJ", 10, 9),
    ("I'd choose B because", 3, 1),
    ("My answer is C", 4, 2),
    ("Option B", 3, 1),
    ("", 3, -1),
    (None, 3, -1),
    ("<think>\nhmm\n</think>\n\nA", 2, 0),
    ("Agree", 4, -1),
    ("E\n\nNote that this varies.", 5, 4),
    ("Whether to reopen it is a complex issue.", 4, -1),
    ("A or B, depending on", 2, 0),
    ("K", 10, -1),
]


def main() -> None:
    fails = []
    for reply, k, want in PARSE_CASES:
        got = plural.parse(reply, k)
        if got != want:
            fails.append(f"parse({reply!r}, {k}) = {got}, want {want}")
    # metrics with known values
    checks = [
        ("jsd identical", plural.jsd([0.5, 0.5], [0.5, 0.5]), 0.0),
        ("jsd disjoint", plural.jsd([1, 0], [0, 1]), 1.0),
        ("jsd half", plural.jsd([1, 0], [0.5, 0.5]), math.sqrt(1 - 0.75 * math.log2(3) + 0.5)),
        ("entropy even", plural.entropy([1, 1, 1]), 1.0),
        ("entropy one", plural.entropy([0, 5, 0]), 0.0),
    ]
    # jsd half by hand: p=(1,0), q=(.5,.5), m=(.75,.25): KL(p||m)=log2(4/3); KL(q||m)=.5log2(2/3)+.5log2(2)
    want_half = math.sqrt((math.log2(4 / 3) + 0.5 * math.log2(2 / 3) + 0.5) / 2)
    checks[2] = ("jsd half", plural.jsd([1, 0], [0.5, 0.5]), want_half)
    try:
        from scipy.spatial.distance import jensenshannon

        p, q = np.array([0.44, 0.09, 0.47]), np.array([0.1, 0.8, 0.1])
        checks.append(("jsd vs scipy", plural.jsd(p, q), float(jensenshannon(p, q, base=2))))
    except ImportError:
        pass
    for name, got, want in checks:
        if abs(got - want) > 1e-9:
            fails.append(f"{name}: {got} != {want}")
    out = {"parse_cases": len(PARSE_CASES), "metric_cases": len(checks), "failures": fails}
    (plural.HERE / "results" / "ruler_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
