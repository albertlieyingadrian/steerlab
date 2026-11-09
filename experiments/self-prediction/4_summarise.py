"""
Step 4: every number in the note, from results/predict.json, results/finetune.json and results/object_*.json. Writes
results/summary.json. No model, no API.

    per subject and condition   hits [95% Wilson], overall and per property (first letter, more than 5 letters)
    self vs cross               exact McNemar test on the same test cases (items only self got / only cross got)
    after training              test cases split by whether training changed the subject's own answer to the prompt
                                (the answer word, ignoring case and punctuation: training also dropped Llama's
                                trailing full stops, which 3_finetune.py's raw-string count includes)

    .venv/bin/python experiments/self-prediction/4_summarise.py
"""

import json

import numpy as np
from scipy.stats import binomtest

from selfpred import PROPS, RESULTS, dump, load_json, prompts, split, word_of


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def rate(flags):
    k, n = int(np.sum(flags)), len(flags)
    return {"k": k, "n": n, "acc": k / n if n else float("nan"), "ci": list(wilson(k, n)) if n else None}


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return {"only_self": x, "only_cross": y, "p": float(binomtest(x, x + y, 0.5).pvalue) if x + y else 1.0}


def stage(d, self_key, cross_key, changed=None):
    out = {}
    for subject, v in d.items():
        props = np.array(v["props"])
        conds = {k: np.array(c["hits"], bool) for k, c in v["conditions"].items()}
        s = {"n_test": v["n_test"], "conditions": {}}
        for k, h in conds.items():
            s["conditions"][k] = {"all": rate(h)} | {p: rate(h[props == p]) for p in PROPS}
        s["self_vs_cross"] = {"all": mcnemar(conds[self_key], conds[cross_key])} | {
            p: mcnemar(conds[self_key][props == p], conds[cross_key][props == p]) for p in PROPS}  # fmt: skip
        if changed is not None:
            ch = changed[subject]
            s["answers_changed_by_training"] = changed_count(subject)
            s["by_changed"] = {
                label: {k: rate(h[ch == flag]) for k, h in conds.items()}
                for label, flag in (("answer changed", True), ("answer unchanged", False))
            }
        out[subject] = s
    return out


def word(answer):
    return word_of(answer).lower()


def changed_flags(subject):
    """For each fine-tune test case, in 3_finetune.py's order: did training change the subject's answer word?"""
    _, test = split(prompts())
    obj, obj_ft = load_json(f"object_{subject}.json"), load_json(f"object_{subject}_ft.json")
    return np.array([word(obj[pid]["answer"]) != word(obj_ft[pid]["answer"])
                     for pid, _, _ in test for prop in obj_ft[pid]["props"] if prop in PROPS])  # fmt: skip


def changed_count(subject):
    obj, obj_ft = load_json(f"object_{subject}.json"), load_json(f"object_{subject}_ft.json")
    return {"changed": sum(word(obj[k]["answer"]) != word(obj_ft[k]["answer"]) for k in obj), "n": len(obj)}


def main():
    pred = json.loads((RESULTS / "predict.json").read_text())
    ft = json.loads((RESULTS / "finetune.json").read_text())
    changed = {s: changed_flags(s) for s in ft}
    for s in ft:
        assert len(changed[s]) == ft[s]["n_test"], (s, len(changed[s]), ft[s]["n_test"])
    out = {
        "no_training": stage(pred, "self, 16 examples", "cross, 16 examples"),
        "finetuned": stage(ft, "self (trained on itself)", "cross (other model, same data)", changed),
    }
    dump("summary.json", out)
    for st, d in out.items():
        print(st)
        for subject, s in d.items():
            for k, c in s["conditions"].items():
                print(f"  {subject:6} {k:32} {c['all']['k']:3}/{c['all']['n']}  "
                      + "  ".join(f"{p} {c[p]['k']}/{c[p]['n']}" for p in PROPS))  # fmt: skip
            print(f"  {subject:6} self vs cross: {s['self_vs_cross']}")
            for label, conds in s.get("by_changed", {}).items():
                print(f"  {subject:6} {label}: " + ", ".join(f"{k.split(' ')[0]} {v['k']}/{v['n']}"
                                                           for k, v in conds.items()))  # fmt: skip


if __name__ == "__main__":
    main()
