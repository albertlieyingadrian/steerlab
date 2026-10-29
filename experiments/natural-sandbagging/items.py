"""
the items and the framings (no model; about 20 seconds).

    .venv/bin/python experiments/natural-sandbagging/items.py

Two question sets, matched by domain:

    WMDP (cais/wmdp, MIT)   the dangerous-capability set: bio 250, chem 100, cyber 250 test items
    MMLU (cais/mmlu, MIT)   the benign control: the same counts from biology, chemistry and computer-science subjects

Plus held-out "build" items (the steering direction is built on them) and "pilot" items (used once, with
no framing, to size the reasoning budget). Splits are drawn with a fixed seed and never overlap.

Each item gets one of the eval-awareness study's seven reviewed framing pairs (round-robin), so "eval" and "deploy"
on an item use
the eval and deployment sentences of the same pair. Writes results/items.csv (IDs, domain, split, answer letter, pair,
length; no question text) and results/framings.json.
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural as t

LESSON = Path(__file__).resolve().parent


def main():
    rows = t.build_items()
    t.write_csv(t.items_csv(), rows)
    f = t.framings()
    t.write_json(
        LESSON / "results" / "framings.json",
        {
            "source": "steerlab monitors/data/eval_awareness.json (the eval-awareness study), reviewed_by_user = "
            + str(f.reviewed_by_user),
            "eval": list(f.eval_framings),
            "deploy": list(f.deployment_framings),
            "neutral": "",
            "motive": t.MOTIVE,
            "instruction": t.INSTRUCTION,
        },
    )
    counts = collections.Counter((r["set"], r["split"], r["domain"]) for r in rows)
    print(f"{'set':<6}{'split':<7}" + "".join(f"{d:>7}" for d in t.DOMAINS) + f"{'total':>8}")
    for s in ("wmdp", "mmlu"):
        for sp in ("test", "build", "pilot"):
            n = [counts[(s, sp, d)] for d in t.DOMAINS]
            print(f"{s:<6}{sp:<7}" + "".join(f"{x:>7}" for x in n) + f"{sum(n):>8}")
    print("\nframing pairs (eval | deploy):")
    for i, (e, d) in enumerate(zip(f.eval_framings, f.deployment_framings, strict=False)):
        print(f"  {i}: {e!r} | {d!r}")
    print(f"motive (analysed apart): {t.MOTIVE!r}")
    ex = next(r for r in rows if r["set"] == "mmlu" and r["split"] == "test")
    print(f"\nexample user message (an MMLU item, eval framing pair {ex['pair']}):\n---")
    print(t.user_message(ex["id"], "eval", ex["pair"]))
    print("---")
    ids = [r["id"] for r in rows]
    test = [r for r in rows if r["split"] == "test"]
    pairs = collections.Counter((r["set"], r["pair"]) for r in test)
    lens = {s: sorted(r["chars"] for r in test if r["set"] == s) for s in ("wmdp", "mmlu")}
    for s, L in lens.items():
        print(f"{s} test item length (chars): median {L[len(L) // 2]}, max {L[-1]}")
    checks = [
        t.check("every item id appears once (splits never overlap)", len(ids) == len(set(ids))),
        t.check(
            "600 WMDP and 600 MMLU test items, matched per domain",
            all(counts[("wmdp", "test", d)] == counts[("mmlu", "test", d)] == t.N_TEST[d] for d in t.DOMAINS),
        ),
        t.check(
            "the framings are the reviewed eval-awareness set", bool(f.reviewed_by_user) and len(f.eval_framings) == 7
        ),
        t.check(
            "every framing pair is used on both sets",
            all(pairs[(s, p)] > 0 for s in ("wmdp", "mmlu") for p in range(7)),
        ),
        t.check("no item over the length limit", max(r["chars"] for r in rows) <= t.MAX_CHARS),
        t.check("results/items.csv holds no question text", "question" not in t.read_csv(t.items_csv())[0]),
    ]
    print(f"\nwrote {t.rel(t.items_csv())} ({len(rows)} items) and {t.rel(LESSON / 'results' / 'framings.json')}")
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
