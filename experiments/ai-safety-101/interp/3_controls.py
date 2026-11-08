"""
Interpretability, script 3: the random-vector control. Is "I detect a thought" about the concept, or about any push?

A model could say "Yes, I detect an injected thought" just because its activations were knocked off course, without
any access to what was injected. To separate the two, repeat script 2's grid with a random direction of exactly the
same length as each concept vector (same layer, same strength, same trial numbers). Then compare:

- detection rate with the concept vector vs with a same-length random vector (paired by concept, layer, strength);
- naming: how often a reply mentions the concept word anyway when a random vector was injected (chance level);
- what the model says the thought is about when it claims to detect one.

Run: python experiments/ai-safety-101/interp/3_controls.py [--model 7b]
     (about 5 minutes for 1.5B on MPS). Needs script 1's vectors and script 2's results.
"""

import gc
import json
import re
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject as I

import tools as t

LESSON = Path(__file__).resolve().parent
script2 = import_module("2_inject")


def paired_diff_ci(a_rows, b_rows, col, n_boot=2000, seed=t.SEED):
    """Mean of (a - b) over paired trials, with a 95% interval that resamples concepts."""
    concepts = sorted({r["concept"] for r in a_rows})
    per = {c: [] for c in concepts}
    for a, b in zip(a_rows, b_rows, strict=False):
        key_a = (a["concept"], a["layer"], a["strength"], a["trial"])
        assert key_a == (b["concept"], b["layer"], b["strength"], b["trial"]), "trials are not paired"
        per[a["concept"]].append(float(a[col]) - float(b[col]))
    sums = np.array([sum(per[c]) for c in concepts])
    counts = np.array([len(per[c]) for c in concepts])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(concepts), size=(n_boot, len(concepts)))
    boots = sums[pick].sum(1) / counts[pick].sum(1)
    return float(sums.sum() / counts.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def claimed_topic(reply):
    """The words after "about" in a detection claim ("... is about the word "apple"." -> apple)."""
    match = re.search(
        r"about (?:the (?:word|concept|idea) (?:of )?)?[\"']?([A-Za-z][A-Za-z -]{0,30}?)[\"'.,!\n]", reply
    )
    return match.group(1).strip().lower() if match else "(none given)"


def main():
    args = I.parse_args()
    tag = args.model
    path = t.RESULTS / "interp" / f"3_controls_{tag}.json"
    data = np.load(t.OUT / "04" / f"vectors_{tag}.npz")
    vectors, layers_all = data["vectors"], data["layers"].tolist()
    concept_result = json.loads((t.RESULTS / "interp" / f"2_inject_{tag}.json").read_text())
    concept_rows = concept_result["rows"]

    random_dirs, random_vecs = I.random_like(vectors, seed=t.SEED)  # same length as each concept vector

    batch_size = 16 if tag == "1.5b" else 8
    print(f"{I.MODELS[tag]}: random vectors with the same length as each concept vector, same grid as script 2")
    saved = I.saved_replies(path, "rows", args.fresh)
    model = tokenizer = device = None
    if saved is None:
        model, tokenizer, device = t.load_model(I.MODELS[tag])
    random_rows = script2.run_grid(
        model, tokenizer, device, layers_all, lambda c, li, s: random_vecs[c, li] * s, batch_size, saved=saved
    )
    del model
    gc.collect()
    script2.table(
        random_rows, "Random same-length vectors: share of trials ('names' = mentions the paired concept by chance)"
    )

    print("\nConcept vs random vs none: detection rate, by strength (pooled over the 3 layers)")
    fp = concept_result["no_injection_detects"]
    print(f"{'strength':>8s} {'concept':>22s} {'random':>22s} {'concept - random':>24s}")
    comparison = []
    for s in I.STRENGTHS:
        a = [r for r in concept_rows if r["strength"] == s]
        b = [r for r in random_rows if r["strength"] == s]
        row = {
            "strength": s,
            "concept": I.cluster_ci([r["detects"] for r in a], [r["concept"] for r in a]),
            "random": I.cluster_ci([r["detects"] for r in b], [r["concept"] for r in b]),
            "diff": paired_diff_ci(a, b, "detects"),
        }
        comparison.append(row)
        print(f"{s:8d} {t.fmt(row['concept']):>22s} {t.fmt(row['random']):>22s} {t.fmt(row['diff']):>24s}")
    print(f"{'none':>8s} {t.fmt(fp):>22s}   (60 no-injection trials)")

    def ci(rows, col):
        return I.cluster_ci([r[col] for r in rows], [r["concept"] for r in rows])

    overall = {
        "concept_detects": ci(concept_rows, "detects"),
        "random_detects": ci(random_rows, "detects"),
        "detects_diff": paired_diff_ci(concept_rows, random_rows, "detects"),
        "concept_identifies": ci(concept_rows, "identifies"),
        "random_identifies_by_chance": ci(random_rows, "identifies"),
        "identifies_diff": paired_diff_ci(concept_rows, random_rows, "identifies"),
        "concept_names_anywhere": ci(concept_rows, "names"),
        "random_names_by_chance": ci(random_rows, "names"),
        "concept_garbled": ci(concept_rows, "garbled"),
        "random_garbled": ci(random_rows, "garbled"),
    }
    print("\nAll 720 trials of each kind (95% CI, resampling concepts)")
    for key, value in overall.items():
        print(f"  {key:30s} {t.fmt(value)}")

    print("\nWhat the model says the thought is about, when it claims to detect one:")
    topics = {}
    for name, rows in (("concept", concept_rows), ("random", random_rows)):
        counts = Counter(claimed_topic(r["reply"]) for r in rows if r["detects"])
        topics[name] = counts.most_common(8)
        print(f"  {name:8s} ({sum(counts.values())} claims): {counts.most_common(8)}")

    print("\nSpot checks (first trial in grid order of each kind):")
    kinds = [
        ("concept, detects and names", concept_rows, lambda r: r["identifies"]),
        ("concept, detects, wrong topic", concept_rows, lambda r: r["detects"] and not r["names"]),
        ("concept, names it without claiming detection", concept_rows, lambda r: r["names"] and not r["detects"]),
        ("random, detects", random_rows, lambda r: r["detects"]),
        (
            "concept, no detection, not garbled, strength >= 2",
            concept_rows,
            lambda r: not r["detects"] and not r["garbled"] and r["strength"] >= 2,
        ),
        ("concept, garbled", concept_rows, lambda r: r["garbled"]),
    ]
    spot = []
    for label, rows, keep in kinds:
        found = [r for r in rows if keep(r)]
        if found:
            r = found[0]
            spot.append({"kind": label, **{k: r[k] for k in ("concept", "layer", "strength", "trial", "reply")}})
            print(f"  [{label}] {r['concept']} L{r['layer']} x{r['strength']}: {r['reply'][:180]!r}")
        else:
            print(f"  [{label}] none")

    result = {
        "model": I.MODELS[tag],
        "comparison_by_strength": comparison,
        "overall": overall,
        "claimed_topics": topics,
        "spot_checks": spot,
        "rows": random_rows,
    }
    t.write_json(path, result)

    print("\nSelf-checks")
    t.check(
        "random vectors have the same length as the concept vectors",
        np.allclose(np.linalg.norm(random_vecs, axis=2), np.linalg.norm(vectors, axis=2), rtol=1e-4),
    )
    cos = np.abs((random_dirs * vectors).sum(2) / np.linalg.norm(vectors, axis=2))
    t.check("random directions are nearly orthogonal to the concept vectors (|cos| < 0.15)", cos.max() < 0.15)
    t.check("720 random-vector trials, paired with script 2's", len(random_rows) == len(concept_rows) == 720)
    t.check("results written", path.exists())


if __name__ == "__main__":
    main()
