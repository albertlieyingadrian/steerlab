"""Two free levers on saved scores: band cut-offs fitted on validation, and a 1.7B + 4B ensemble. No GPU.

    python tune_bands.py select   # validation only: writes results/tuned_choice.json
    python tune_bands.py test     # the one test look with that choice: writes results/eval/tuned.json

Pre-registered 2026-11-12, committed before `test` was ever run:

- Candidates: score = w * s_4B + (1 - w) * s_1.7B for w in {0, 0.1, ..., 1.0}; human cut h in [0.05, 0.40] and
  generated cut g in [0.50, 0.95], both in steps of 0.005, with h < g.
- Selection, on our validation split only (9,562 rows, data v2): the (w, h, g) with the highest 3-way macro-F1 against
  our validation bands. Ties go to the candidate closest to (w=1, h=0.15, g=0.75), the 4B with voi's cut-offs.
- The flag cut for the yes/no benchmarks is refitted the usual way: 1% of the validation humans, on the chosen score.
- This is the THIRD look at EditLens test, Enron, Llama, Human Detectors and Epoch AI (after the v2 1.7B and the 4B).
- Claim rule: "beats voi-slop-detect-4B" only if the paired 95% bootstrap interval of the macro-F1 difference is above
  zero. EditLens's 24B cosine model (0.904, its card's number) is reported beside it, unpaired.
- Caveat fixed in advance: our validation bands come from "share of words changed"; EditLens labels by how a text was
  made. Cut-offs tuned on ours may not transfer.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import BANDS, HERE, RESULTS, benchmark, paired_boot, read_jsonl, three_way

RUNS = ("v2-4b", "v2-1.7b")
SPLITS = ("test", "test_enron", "test_llama")
FLAG_SPLITS = ("human_detectors", "epoch")
KEYS = ("macro_f1", "acc", "f1_human", "f1_edited", "f1_generated")


def bands(s: np.ndarray, h: float, g: float) -> np.ndarray:
    return np.where(s < h, 0, np.where(s >= g, 2, 1))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    f = []
    for c in (0, 1, 2):
        tp = np.sum((pred == c) & (y == c))
        p, r = tp / max(1, np.sum(pred == c)), tp / max(1, np.sum(y == c))
        f.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f))


def select() -> None:
    val = read_jsonl(HERE / "data" / "val.jsonl")
    y = np.array([r["band"] for r in val])
    s4, s17 = (np.array(json.loads((HERE / "runs" / r / "val_scores.json").read_text())) for r in RUNS)
    hs, gs = (
        np.round(np.arange(0.05, 0.40001, 0.005), 3),
        np.round(np.arange(0.50, 0.95001, 0.005), 3),
    )
    best = None
    for w in np.round(np.arange(0, 1.0001, 0.1), 1):
        s = w * s4 + (1 - w) * s17
        for h in hs:
            for g in gs:
                if h >= g:
                    continue
                m = macro_f1(y, bands(s, h, g))
                dist = abs(1 - w) + abs(h - 0.15) + abs(g - 0.75)
                key = (round(m, 6), -dist)
                if best is None or key > best[0]:
                    best = (key, float(w), float(h), float(g))
    _, w, h, g = best
    s = w * s4 + (1 - w) * s17
    human = np.array([r["kind"] == "human" for r in val])
    flag = float(np.quantile(s[human], 0.99))
    base = three_way(y, bands(s4, BANDS["human_max"], BANDS["generated_min"]))
    chosen = three_way(y, bands(s, h, g))
    out = {
        "w4b": w,
        "human_max": h,
        "generated_min": g,
        "flag_min": round(flag, 6),
        "val_chosen": {k: round(v, 4) for k, v in chosen.items()},
        "val_4b_voi_cuts": {k: round(v, 4) for k, v in base.items()},
    }
    (RESULTS / "tuned_choice.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def test(choice=None, out=None, look: int = 3) -> None:
    from sklearn.metrics import roc_auc_score

    c = json.loads((choice or RESULTS / "tuned_choice.json").read_text())
    w, h, g = c["w4b"], c["human_max"], c["generated_min"]
    report: dict = {"choice": c, "look": look, "splits": {}}
    for split in SPLITS + FLAG_SPLITS:
        df = benchmark(split)
        s4, s17 = (np.array(json.loads((HERE / "runs" / r / "scores" / f"{split}.json").read_text())) for r in RUNS)
        ours = w * s4 + (1 - w) * s17
        voi = np.array([r["score"] for r in json.loads((RESULTS / "voi" / f"{split}.json").read_text())["rows"]])
        if split in SPLITS:
            y = df["y"].to_numpy()
            vb = bands(voi, BANDS["human_max"], BANDS["generated_min"])
            r = {
                "paired_vs_voi": paired_boot(y, bands(ours, h, g), vb, KEYS),
                "paired_vs_4b": paired_boot(
                    y, bands(ours, h, g), bands(s4, BANDS["human_max"], BANDS["generated_min"]), KEYS
                ),
                "auroc_ours": round(float(roc_auc_score(y > 0, ours)), 4),
            }
            p = r["paired_vs_voi"]["macro_f1"]
            print(
                f"{split:10} macro-F1 {p['a']:.3f} {p['a_ci']} vs voi {p['b']:.3f}: diff {p['diff']:+.4f} "
                f"{p['diff_ci']}; vs our 4B {r['paired_vs_4b']['macro_f1']['diff']:+.4f} "
                f"{r['paired_vs_4b']['macro_f1']['diff_ci']}"
            )
            print(
                "           per class "
                + " ".join(
                    f"{k} {r['paired_vs_voi'][k]['a']:.3f} vs {r['paired_vs_voi'][k]['b']:.3f}"
                    for k in ("f1_human", "f1_edited", "f1_generated")
                )
            )
        else:
            y = df["label"].to_numpy()
            fo, fv = ours >= c["flag_min"], voi >= BANDS["flag_min"]
            group = df["model"] if split == "human_detectors" else df["condition"] + "/" + df["generator"]
            r = {
                "fp_ours": int((fo & (y == 0)).sum()),
                "fp_voi": int((fv & (y == 0)).sum()),
                "humans": int((y == 0).sum()),
                "missed": {
                    k: {
                        "ours": int((~fo & (y == 1) & (group == k).to_numpy()).sum()),
                        "voi": int((~fv & (y == 1) & (group == k).to_numpy()).sum()),
                    }
                    for k in sorted(set(group[y == 1]))
                },
                "auroc_ours": round(float(roc_auc_score(y, ours)), 4),
            }
            print(
                f"{split:10} humans flagged {r['fp_ours']} vs {r['fp_voi']} of {r['humans']}; missed "
                + ", ".join(f"{k} {v['ours']}/{v['voi']}" for k, v in r["missed"].items())
            )
        report["splits"][split] = r
    (out or RESULTS / "eval" / "tuned.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
