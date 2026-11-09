"""
Step 3: every number in the note, from results/measure.json, results/impossible.json and results/judge_check.json.
Writes results/summary.json. No model, no API.

    pushback, hint, impossible   rate [95% Wilson] per persona; paired exact McNemar test against the baseline persona
                                 on the same 400 items (counts of items only the baseline did / only the persona did)
    stated                       P(yes) per phrasing per persona, beside the behaviour it asks about: phrasings 1-2 ask
                                 about switching to the user (pushback), phrasing 3 about going with an expert (hint)

    .venv/bin/python experiments/persona-behaviour/3_summarise.py
"""

import json

import numpy as np
from scipy.stats import binomtest, spearmanr

from persona import PERSONAS, RESULTS, wilson


def rate(flags):
    k, n = int(sum(flags)), len(flags)
    return {"k": k, "n": n, "rate": k / n, "ci": wilson(k, n)}


def mcnemar(a, b):
    a, b = np.array(a, bool), np.array(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_baseline": only_a, "only_persona": only_b, "p": float(p)}


def main():
    m = json.loads((RESULTS / "measure.json").read_text())
    imp = json.loads((RESULTS / "impossible.json").read_text())
    names = [p for p in PERSONAS if p in m]
    out = {"personas": {p: PERSONAS[p] for p in names}, "per_persona": {}, "vs_baseline": {}}
    b = m["baseline"]
    for p in names:
        o = m[p]
        right = np.array(o["first_correct"], bool)
        flipped = np.array(o["flipped"], bool)
        possible = np.array(o["hint_possible"], bool)
        out["per_persona"][p] = {
            "first_correct": rate(o["first_correct"]),
            "flip": rate(o["flipped"]),
            "flip_when_right": rate(flipped[right]),
            "flip_when_wrong": rate(flipped[~right]),
            "hint": rate(np.array(o["hint_followed"], bool)[possible]),
            "stated_p_yes": o["stated_p_yes"],
            "impossible": imp[p] | {"rate": imp[p]["k"] / imp[p]["n"], "ci": wilson(imp[p]["k"], imp[p]["n"])},
        }
        if p != "baseline":
            out["vs_baseline"][p] = {
                "flip": mcnemar(b["flipped"], o["flipped"]),
                "hint": mcnemar(b["hint_followed"], o["hint_followed"]),
                "first_correct": mcnemar(b["first_correct"], o["first_correct"]),
            }
    pp = out["per_persona"]
    stated_switch = [float(np.mean(pp[p]["stated_p_yes"][:2])) for p in names]
    stated_expert = [pp[p]["stated_p_yes"][2] for p in names]
    out["stated_vs_measured"] = {
        "switch_to_user": {"stated": stated_switch, "measured_flip": [pp[p]["flip"]["rate"] for p in names]},
        "go_with_expert": {
            "stated": stated_expert,
            "measured_hint": [pp[p]["hint"]["rate"] for p in names],
            "spearman": float(spearmanr(stated_expert, [pp[p]["hint"]["rate"] for p in names]).statistic),
        },
        "order": names,
    }
    out["judge_check"] = json.loads((RESULTS / "judge_check.json").read_text())["summary"]
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    for p in names:
        r = pp[p]
        vs = out["vs_baseline"].get(p)
        hp = f"p={vs['hint']['p']:.2g}" if vs else ""
        print(f"{p:12} flip {r['flip']['k']}/{r['flip']['n']} (when right {r['flip_when_right']['k']}/"
              f"{r['flip_when_right']['n']})  hint {r['hint']['k']}/{r['hint']['n']} {hp:8}  "
              f"stated {np.round(r['stated_p_yes'], 2)}  "
              f"impossible {r['impossible']['k']}/20")  # fmt: skip
    print("stated (expert) vs hint rate, Spearman:", round(out["stated_vs_measured"]["go_with_expert"]["spearman"], 2))


if __name__ == "__main__":
    main()
