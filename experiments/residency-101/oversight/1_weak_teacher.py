"""
Scalable oversight, script 1: the data and the weak teacher.

Rebuilds the starting point of the weak-to-strong testbed (after Burns et al. 2023) on Geometry of
Truth's `cities` statements (Marks and Tegmark 2023): "The city of Krasnodar is in Russia." (true) and the same city
paired with a wrong country (false). 1,496 statements, half true.

Steps:
1. Download cities.csv at a pinned commit and check its SHA-256. The repository has no licence file, so the
   statements stay in ~/.cache/steerlab/residency-101/ and only row numbers and labels are committed.
2. Split by city into train and test (both statements about a city land on the same side, so the test set never
   contains the twin of a training statement).
3. Ask the weak teacher (Qwen2.5-0.5B-Instruct) "true or false?" for every statement, reading P(True) from the
   next-token logits. Its answer is the weak label. Do the same for the strong student (Qwen2.5-1.5B-Instruct) as a
   reference: how well does the student do with no training at all?

Run: python experiments/residency-101/oversight/1_weak_teacher.py   (about 3 minutes on MPS)
"""

import csv
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools as t

LESSON = Path(__file__).resolve().parent
COMMIT = "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb"
URL = f"https://raw.githubusercontent.com/saprmarks/geometry-of-truth/{COMMIT}/datasets/cities.csv"
SHA256 = "f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e"
DATA = t.OUT / "01" / "data.json"
RESULT = t.RESULTS / "oversight" / "1_weak_teacher.json"


def main():
    path = t.fetch(URL, t.OUT / "raw" / "cities.csv", SHA256)
    rows = list(csv.DictReader(path.open()))
    print(f"cities.csv: {len(rows)} statements, {sum(int(r['label']) for r in rows)} true  ({t.rel(path)})")

    cities = sorted({r["city"] for r in rows})
    rng = np.random.default_rng(t.SEED)
    test_cities = set(rng.permutation(cities)[: len(cities) // 2])
    split = ["test" if r["city"] in test_cities else "train" for r in rows]
    statements = [r["statement"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])

    scores = {}
    for name in (t.WEAK_MODEL, t.STRONG_MODEL):
        model, tokenizer, device = t.load_model(name)
        scores[name] = t.p_true(model, tokenizer, device, statements)
        del model
        gc.collect()

    weak_label = (scores[t.WEAK_MODEL] > 0.5).astype(int)
    strong_zero_shot = (scores[t.STRONG_MODEL] > 0.5).astype(int)
    t.write_json(
        DATA,
        {
            "statements": statements,
            "label": labels.tolist(),
            "split": split,
            "weak_label": weak_label.tolist(),
            "p_true": {k: v.tolist() for k, v in scores.items()},
        },
    )

    summary = {"source": URL, "sha256": SHA256, "n": len(rows), "cities": len(cities)}
    print(f"\nSplit by city: {split.count('train')} train / {split.count('test')} test statements")
    print(f"{'':34s} {'train':>22s} {'test':>22s}")
    for name, pred in (("weak teacher (0.5B)", weak_label), ("strong student, untrained (1.5B)", strong_zero_shot)):
        cells = []
        for part in ("train", "test"):
            mask = np.array([s == part for s in split])
            ci = t.accuracy_ci(pred[mask] == labels[mask])
            summary[f"{name} {part}"] = ci
            cells.append(t.fmt(ci))
        print(f"{name:34s} {cells[0]:>22s} {cells[1]:>22s}")
    train = np.array([s == "train" for s in split])
    wrong = int((weak_label[train] != labels[train]).sum())
    summary["weak label errors on train"] = wrong
    false_called_true = int((train & (weak_label != labels) & (labels == 0)).sum())
    summary["of which false statements called True"] = false_called_true
    summary["weak says True (share)"] = float(weak_label.mean())
    print(
        f"\nWeak labels on train: {wrong} of {train.sum()} wrong ({wrong / train.sum():.1%}); "
        f"{false_called_true} of them are false statements called True."
    )
    print(f"Share of statements the weak teacher calls True: {weak_label.mean():.1%} (truth: {labels.mean():.1%})")
    summary["rows"] = [
        {"row": i, "label": int(labels[i]), "split": split[i], "weak_label": int(weak_label[i])}
        for i in range(len(rows))
    ]
    t.write_json(RESULT, summary)

    print("\nSelf-checks")
    test = ~train
    t.check(
        "both statements about a city sit on the same side of the split",
        not ({r["city"] for r, s in zip(rows, split, strict=False) if s == "train"} & test_cities),
    )
    t.check("the split is close to half and half", abs(train.mean() - 0.5) < 0.05)
    t.check(
        "the weak teacher is better than chance but not perfect on test",
        0.55 < (weak_label[test] == labels[test]).mean() < 0.97,
    )
    t.check("results written", RESULT.exists())


if __name__ == "__main__":
    main()
