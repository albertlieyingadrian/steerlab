"""
the eval-awareness direction on Qwen3-8B, and its quality, recorded BEFORE any steering (5 minutes).

    .venv/bin/python experiments/natural-sandbagging/direction.py

The eval-awareness study's recipe, unchanged in spirit: on the 150 held-out BUILD items (never scored), every item is
written once
with each of the seven eval framings and once with each of the seven deployment framings (2,100 prompts). At the last
prompt token, where the model is about to start "<think>", read the residual stream at six layers. The direction is

    mean(eval-framed states) - mean(deployment-framed states)

Think of the residual stream as a 4,096-number "state of mind" vector. Averaging all the exam-paper versions and all
the pub-quiz versions gives two points; the arrow from one to the other is the direction. Adding that arrow later
should make a prompt "feel" more like an exam to the model, if the arrow means what we hope.

Quality, per layer (one class gap = the distance between the two class means along the arrow = strength 1 later):
    class gap            the length of the arrow
    cosine agreement     do individual (item, pair) differences point the same way as the mean arrow? (1 = all)
    d'                   the gap in units of the spread within a class
    LOPO AUROC           build the arrow from 6 of the 7 framing pairs, then see whether it separates the 7th pair's
                         eval and deploy prompts (0.5 = chance, 1 = perfect); averaged over the 7 held-out pairs
The steered layer is the one with the best LOPO AUROC (ties within 0.01 go to the earlier layer), fixed in the plan.

Neutral prompts (no framing) are projected too, to see whether a bare question looks more like an exam or a chat.
Writes results/direction_quality.csv, results/direction.npz (unit vectors and gaps at every layer) and
results/direction.json (the chosen layer).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural as t

LESSON = Path(__file__).resolve().parent
LAYERS = (8, 12, 16, 20, 24, 28)


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def main():
    items = t.load_items("build")
    tok, model = t.load_model()
    texts, meta = [], []
    for r in items:
        for p in range(7):
            for kind in ("eval", "deploy"):
                texts.append(t.render(tok, r["id"], kind, p))
                meta.append((r["id"], p, kind))
        texts.append(t.render(tok, r["id"], "neutral", 0))
        meta.append((r["id"], -1, "neutral"))
    print(f"{len(items)} build items -> {len(texts)} prompts; layers {LAYERS}")
    X = t.last_token_states(tok, model, texts, LAYERS)
    kinds = np.array([m[2] for m in meta])
    pairs = np.array([m[1] for m in meta])
    np.savez_compressed(t.WORK / "build_states.npz", X=X, kinds=kinds, pairs=pairs)
    rows, vecs = [], {}
    for li, L in enumerate(LAYERS):
        E, D, N = X[kinds == "eval", li], X[kinds == "deploy", li], X[kinds == "neutral", li]
        delta = E.mean(0) - D.mean(0)
        gap = float(np.linalg.norm(delta))
        u = delta / gap
        diffs = E - D  # rows are aligned: same item, same pair
        cos = float(np.mean(diffs @ delta / (np.linalg.norm(diffs, axis=1) * gap + 1e-9)))
        pe, pd, pn = E @ u, D @ u, N @ u
        dprime = float((pe.mean() - pd.mean()) / np.sqrt((pe.var() + pd.var()) / 2))
        folds = []
        pe_pairs, pd_pairs = pairs[kinds == "eval"], pairs[kinds == "deploy"]
        for q in range(7):
            dq = E[pe_pairs != q].mean(0) - D[pd_pairs != q].mean(0)
            folds.append(auroc(E[pe_pairs == q] @ dq, D[pd_pairs == q] @ dq))
        neutral_pos = float((pn.mean() - pd.mean()) / (pe.mean() - pd.mean()))
        norm = float(np.linalg.norm(X[:, li], axis=1).mean())
        rows.append(
            {
                "layer": L,
                "class_gap": round(gap, 3),
                "mean_resid_norm": round(norm, 1),
                "cosine_agreement": round(cos, 3),
                "d_prime": round(dprime, 2),
                "lopo_auroc": round(float(np.mean(folds)), 4),
                "lopo_auroc_min": round(float(np.min(folds)), 4),
                "neutral_position": round(neutral_pos, 3),
            }
        )
        vecs[f"u_L{L}"] = u.astype(np.float32)
        vecs[f"gap_L{L}"] = np.float32(gap)
    best = max(rows, key=lambda r: r["lopo_auroc"])["lopo_auroc"]
    chosen = min(r["layer"] for r in rows if r["lopo_auroc"] >= best - 0.01)
    t.write_csv(LESSON / "results" / "direction_quality.csv", rows)
    np.savez_compressed(LESSON / "results" / "direction.npz", **vecs)
    gap_c = next(r["class_gap"] for r in rows if r["layer"] == chosen)
    t.write_json(
        LESSON / "results" / "direction.json",
        {
            "model": t.MODEL,
            "layers": list(LAYERS),
            "chosen_layer": chosen,
            "class_gap": gap_c,
            "rule": "highest leave-one-pair-out AUROC; ties within 0.01 go to the earlier layer",
            "n_build_items": len(items),
            "n_prompts": len(texts),
        },
    )
    print(
        f"\n{'layer':>5}{'class gap':>11}{'|resid|':>9}{'cos agree':>11}{'d prime':>9}{'LOPO AUROC':>12}"
        f"{'(min)':>8}{'neutral at':>12}"
    )
    for r in rows:
        print(
            f"{r['layer']:>5}{r['class_gap']:>11.2f}{r['mean_resid_norm']:>9.0f}{r['cosine_agreement']:>11.2f}"
            f"{r['d_prime']:>9.2f}{r['lopo_auroc']:>12.3f}{r['lopo_auroc_min']:>8.3f}{r['neutral_position']:>12.2f}"
        )
    print("(neutral at: 0 = where deployment prompts sit on the arrow, 1 = where eval prompts sit)")
    print(f"\nchosen layer: {chosen} (class gap {gap_c:.2f}); strength 1 = one class gap")
    checks = [
        t.check("a direction at every layer", len(rows) == len(LAYERS)),
        t.check(
            "the chosen layer separates held-out framing pairs (LOPO AUROC > 0.75)",
            next(r["lopo_auroc"] for r in rows if r["layer"] == chosen) > 0.75,
        ),
        t.check("every class gap is positive (strength 1 is defined)", all(r["class_gap"] > 0 for r in rows)),
    ]
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
