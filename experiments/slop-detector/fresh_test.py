"""Freeze a fresh EditLens-distribution test set from EditLens's TRAIN split, which no model here has seen.

    python fresh_test.py freeze    # results/fresh_test_ids.json (row ids only, no text), once, before any v3 model
    python fresh_test.py score     # v2-4b, v2-1.7b (score_run) and voi (score) on the 6,000 rows, on Modal
    python fresh_test.py baseline  # fresh look 1, baselines only, both arms: results/eval/fresh_baseline.json

Baseline look (approved 2026-11-12, written before the scores exist): the CURRENT models only, so v3 has a clean
comparison. Fixed arm: v2-4b and v2-1.7b at voi's cut-offs (0.15 / 0.75) against voi at the same cut-offs. Calibrated
arm: (a) the look-4 setting (0.8 x 4B + 0.2 x 1.7B, cut-offs 0.055 / 0.86, from tuned_editlens_val_choice.json) and
(b) the 4B alone with cut-offs fitted on EditLens val by the same grid, each against voi with its EditLens-val
cut-offs (0.05 / 0.825, from voi_editlens_val.json). Nothing is chosen from these numbers; the v3 final model's single
fresh look is separate and pre-registered then.

Why: EditLens test, Enron, Llama, Human Detectors and Epoch AI have each been looked at four times. EditLens's train
split (59,991 texts) has the same domains, generators and labels as its test split, and neither this project nor
voi-slop-detect-4B (by its card) trained on it. A stratified sample of 6,000 texts (2,000 per class), with every text
whose source document also appears in EditLens val or any test split removed, becomes the fresh test. Never train on
any EditLens split. One look per final model, pre-registered (see docs/plans/2026-11-12-slop-detector-v3-plan.md).
"""

from __future__ import annotations

import json
import sys

from common import RESULTS, editlens_split

N_PER_CLASS, SEED = 2000, 0


def load_train():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "pangram/editlens_iclr",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path)


def freeze() -> None:
    out = RESULTS / "fresh_test_ids.json"
    if out.exists():
        raise SystemExit(f"{out.name} already frozen; not overwriting")
    train = load_train()
    used_src, used_text = set(), set()
    for split in ("val", "test", "test_enron", "test_llama"):
        df = editlens_split(split)
        used_src |= set(df["source_id"].dropna())
        used_text |= set(df["text"]) | set(df["source_text"].dropna())
    clean = train[
        ~train["source_id"].isin(used_src) & ~train["text"].isin(used_text) & ~train["source_text"].isin(used_text)
    ]
    clean = clean[clean["text"].notna() & clean["text_type"].isin(["human_written", "ai_edited", "ai_generated"])]
    import pandas as pd

    parts = [g.sample(n=min(N_PER_CLASS, len(g)), random_state=SEED) for _, g in sorted(clean.groupby("text_type"))]
    sample = pd.concat(parts)
    report = {
        "seed": SEED,
        "train_rows": len(train),
        "after_overlap_removal": len(clean),
        "counts": sample["text_type"].value_counts().to_dict(),
        "by_source": sample["source"].value_counts().to_dict(),
        "by_model": sample["model"].value_counts().to_dict(),
        "ids": sorted(sample["text_id"].astype(str).tolist()),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    print({k: v for k, v in report.items() if k != "ids"})


def fresh_rows():
    ids = json.loads((RESULTS / "fresh_test_ids.json").read_text())["ids"]
    train = load_train()
    sub = train[train["text_id"].astype(str).isin(set(ids))].copy()
    sub.index = sub["text_id"].astype(str)
    return sub.loc[ids].reset_index(drop=True)


def score() -> None:
    import modal
    from common import HERE, VOI, VOI_REVISION, empty_fences

    rows = fresh_rows()
    texts = [empty_fences(str(t)) for t in rows["text"]]
    run_fn = modal.Function.from_name("slop-detector", "score_run")
    voi_fn = modal.Function.from_name("slop-detector", "score")
    calls = {r: run_fn.spawn(r, {"fresh": texts}) for r in ("v2-4b", "v2-1.7b")}
    size = -(-len(texts) // 4)
    voi_calls = [voi_fn.spawn(texts[i : i + size], VOI, VOI_REVISION) for i in range(0, len(texts), size)]
    for r, c in calls.items():
        (HERE / "runs" / r / "scores" / "fresh.json").write_text(json.dumps(c.get()["fresh"]) + "\n")
        print(r, "done")
    v = [x for c in voi_calls for x in c.get()["scores"]]
    (RESULTS / "voi" / "fresh.json").write_text(
        json.dumps({"model": VOI, "revision": VOI_REVISION, "scores": v}) + "\n"
    )
    print("voi done", len(v))


def baseline() -> None:
    from importlib import import_module

    import numpy as np
    from common import EDITLENS_CLASS, HERE, paired_boot

    tb, tev = import_module("tune_bands"), import_module("tune_editlens_val")
    rows = fresh_rows()
    y = rows["text_type"].map(EDITLENS_CLASS).to_numpy()

    def load(run: str) -> np.ndarray:
        return np.array(json.loads((HERE / "runs" / run / "scores" / "fresh.json").read_text()))

    s4, s17 = load("v2-4b"), load("v2-1.7b")
    sv = np.array(json.loads((RESULTS / "voi" / "fresh.json").read_text())["scores"])
    keys = ("macro_f1", "acc", "f1_human", "f1_edited", "f1_generated")
    choice = json.loads((RESULTS / "tuned_editlens_val_choice.json").read_text())
    vh, vg = json.loads((RESULTS / "eval" / "voi_editlens_val.json").read_text())["voi_cuts"]
    # the 4B alone: cut-offs fitted on EditLens val by the same grid and tie rule (validation only)
    val, _ = tev.clean_val()
    yv, v4 = val["y"].to_numpy(), tev.load("v2-4b", "editlens_val")
    best = None
    for h in np.round(np.arange(0.05, 0.40001, 0.005), 3):
        for g in np.round(np.arange(0.50, 0.95001, 0.005), 3):
            if h < g:
                key = (round(tb.macro_f1(yv, tb.bands(v4, h, g)), 6), -(abs(h - 0.15) + abs(g - 0.75)))
                if best is None or key > best[0]:
                    best = (key, float(h), float(g))
    _, h4, g4 = best
    voi_fixed, voi_cal = tb.bands(sv, 0.15, 0.75), tb.bands(sv, vh, vg)
    blend = choice["w4b"] * s4 + (1 - choice["w4b"]) * s17
    arms = {
        "fixed: 4B vs voi": (tb.bands(s4, 0.15, 0.75), voi_fixed),
        "fixed: 1.7B vs voi": (tb.bands(s17, 0.15, 0.75), voi_fixed),
        "calibrated: look-4 blend vs voi calibrated": (
            tb.bands(blend, choice["human_max"], choice["generated_min"]),
            voi_cal,
        ),
        "calibrated: 4B alone vs voi calibrated": (tb.bands(s4, h4, g4), voi_cal),
    }
    from sklearn.metrics import roc_auc_score

    out = {
        "look": "fresh look 1 (baselines only)",
        "n": len(y),
        "cuts_4b_alone": [h4, g4],
        "voi_cuts": [vh, vg],
        "auroc": {
            "4B": round(float(roc_auc_score(y > 0, s4)), 4),
            "1.7B": round(float(roc_auc_score(y > 0, s17)), 4),
            "voi": round(float(roc_auc_score(y > 0, sv)), 4),
        },
        "arms": {},
    }
    for name, (a, b) in arms.items():
        out["arms"][name] = paired_boot(y, a, b, keys)
        p = out["arms"][name]["macro_f1"]
        print(
            f"{name:45} {p['a']:.3f} vs {p['b']:.3f}: {p['diff']:+.4f} {p['diff_ci']}  "
            f"(edited F1 {out['arms'][name]['f1_edited']['a']:.3f} vs {out['arms'][name]['f1_edited']['b']:.3f})"
        )
    print("AUROC", out["auroc"], "| 4B-alone cuts", (h4, g4))
    (RESULTS / "eval" / "fresh_baseline.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    {"freeze": freeze, "score": score, "baseline": baseline}[sys.argv[1]]()
