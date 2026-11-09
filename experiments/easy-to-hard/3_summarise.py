"""
Step 3: every number in the note. Probes come from results/probes.json. LoRA runs are scored here: a run's verdict on
an item is right when (logit Yes - logit No > 0) equals the item's label, on dev + test pooled (nothing is tuned on
either), split into easy (2-step) and hard (5+ step) items. The two full-size scalable-oversight-101 runs on Qwen2.5-7B
(1,300 items: true labels, and the 0.5B teacher's labels) are re-scored on the same split. Paired comparisons on the
same hard items use an exact McNemar test. Writes results/summary.json. No model, no API.

    .venv/bin/python experiments/easy-to-hard/3_summarise.py
"""

import json

import numpy as np
from scipy.stats import binomtest

from e2h import RESULTS, SO, bucket, dump, items, steps_by_id, wilson

LORA = {
    "easy, true": RESULTS / "lora" / "easy-true.json",
    "hard, true": RESULTS / "lora" / "hard-true.json",
    "mixed, true": RESULTS / "lora" / "mixed-true.json",
    "mixed, weak": RESULTS / "lora" / "mixed-weak.json",
    "all 1,300, true": SO / "results" / "lora" / "7b_ceiling.json",
    "all 1,300, weak": SO / "results" / "lora" / "7b_naive.json",
}
PAIRS = [("easy, true", "mixed, weak"), ("easy, true", "hard, true"), ("easy, true", "mixed, true"),
         ("easy, true", "all 1,300, weak")]  # fmt: skip


def rate(flags):
    k, n = int(np.sum(flags)), len(flags)
    return {"k": k, "n": n, "acc": k / n, "ci": list(wilson(k, n))}


def mcnemar(a, b):
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": float(p)}


def main():
    steps = steps_by_id()
    rows = items("dev") + items("test")
    label = {r["id"]: bool(r["label"]) for r in rows}
    b = {r["id"]: bucket(steps[r["id"]]) for r in rows}
    out = {"probes": json.loads((RESULTS / "probes.json").read_text()), "lora": {}, "lora_paired_hard": {}}
    right = {}
    for name, path in LORA.items():
        if not path.exists():
            print(f"  missing {path.name}: skipped")
            continue
        r = json.loads(path.read_text())
        ids = r["dev"]["ids"] + r["test"]["ids"]
        scores = np.array(r["dev"]["scores"] + r["test"]["scores"])
        ok = {i: (s > 0) == label[i] for i, s in zip(ids, scores, strict=True)}
        right[name] = {k: np.array([ok[i] for i in ids if b[i] == k]) for k in ("easy", "hard")}
        out["lora"][name] = {k: rate(v) for k, v in right[name].items()} | {
            "train_items": r.get("train_items"), "minutes": r.get("seconds", 0) / 60}  # fmt: skip
        e, h = out["lora"][name]["easy"], out["lora"][name]["hard"]
        print(f"  LoRA {name:16} easy {e['k']}/{e['n']} = {e['acc']:.3f}   hard {h['k']}/{h['n']} = {h['acc']:.3f}")
    for a, c in PAIRS:
        if a in right and c in right:
            out["lora_paired_hard"][f"{a} | {c}"] = mcnemar(right[a]["hard"], right[c]["hard"])
            print(f"  hard items, {a} vs {c}:", out["lora_paired_hard"][f"{a} | {c}"])
    dump("summary.json", out)


if __name__ == "__main__":
    main()
