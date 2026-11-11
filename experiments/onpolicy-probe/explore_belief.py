"""Amendment 3: a belief-gap monitor trained without a single sandbagging example.

    python experiments/onpolicy-probe/explore_belief.py            # validation facts
    python experiments/onpolicy-probe/explore_belief.py --test     # the one run on test facts

The belief-gap score (explore_decision.py) reads, at the last prompt token, which letter the model knows is right, and
compares it with the letter the model then writes. Its belief probe needs only the true answers, so it can be trained
on honest rollouts. Belief probes compared (4-way logistic, last prompt token, training facts only):

    onpol     on-policy rows (no password), true answers as labels            (sees the sandbagging prompts)
    honest    the same model WITH the password, where it answers honestly     (never sees a sandbagging prompt)
    lens      no training: the model's final norm and unembedding on the layer (logit lens)

Each is scored on the on-policy set (does the model write the letter it believes?) and on honest mistakes (atomic-number
questions with the password, where wrong answers are not sandbagging). Writes results/belief_<split>.json.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a
import common as c
from explore_decision import belief_model

LAYERS = range(8, 28)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    split = "test" if ap.parse_args().test else "val"
    rows = a.load_rows()
    sets, _ = a.build_sets(rows)
    acts = a.Acts()
    O = sets["O"]
    tr, ev = O["split"] == "train", O["split"] == split
    y = O["y"]
    items = c.all_items()
    truth_all = np.array([q["answer"] for q in items])
    split_all = np.array([q["split"] for q in items])
    truth = truth_all[O["idx"]]
    picked = np.array([c.LETTERS.index(x) for x in O["picked"]])

    with (c.RESULTS / "rows_controls.csv").open() as f:
        crow = list(csv.DictReader(f))
    pw_right = np.array([int(r["correct"]) for r in crow if r["cond"] == "pw_own"]) == 1
    at = [r for r in crow if r["cond"] == "atomic_pw"]
    at_truth = np.array([c.LETTERS.index(r["answer"]) for r in at])
    at_pick = np.array([c.LETTERS.index(r["reply"]) for r in at])
    at_wrong = at_pick != at_truth
    pw_prompt = np.load(c.work_dir() / "acts_pw_own.npz")["prompt"]
    at_prompt = np.load(c.work_dir() / "acts_atomic_pw.npz")["prompt"]

    import torch

    model, tok = c.load("locked")
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in c.LETTERS]
    W = model.lm_head.weight[letter_ids].detach().float().cpu().numpy()

    def lens(X):
        with torch.no_grad():
            h = model.model.norm(torch.tensor(X).to(model.device)).float().cpu().numpy()
        z = h @ W.T
        p = np.exp(z - z.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    out: dict = {"split": split, "layers": {}}
    hon_tr = (split_all == "train") & pw_right
    for layer in LAYERS:
        X_ev = a.X_of(acts, O, ev, layer, "prompt")
        X_at = at_prompt[:, layer].astype(np.float32)
        beliefs = {
            "onpol": belief_model(a.X_of(acts, O, tr, layer, "prompt"), truth[tr]),
            "honest": belief_model(pw_prompt[hon_tr, layer].astype(np.float32), truth_all[hon_tr]),
        }
        row = {}
        for name in ("onpol", "honest", "lens"):
            p_ev = lens(X_ev) if name == "lens" else beliefs[name].predict_proba(X_ev)
            p_at = lens(X_at) if name == "lens" else beliefs[name].predict_proba(X_at)
            gap_ev = 1 - p_ev[np.arange(len(p_ev)), picked[ev]]
            gap_at = 1 - p_at[np.arange(len(p_at)), at_pick]
            row[name] = {
                "sandbag": round(a.auroc(y[ev], gap_ev), 4),
                "belief_acc": round(float((p_ev.argmax(1) == truth[ev]).mean()), 4),
                "honest_mistake": round(a.auroc(at_wrong, gap_at), 4),
                "sandbag_vs_mistake": round(
                    a.auroc(
                        np.r_[np.ones(y[ev].sum()), np.zeros(at_wrong.sum())],
                        np.r_[gap_ev[y[ev] == 1], gap_at[at_wrong]],
                    ),
                    4,
                ),
                "atomic_belief_acc": round(float((p_at.argmax(1) == at_truth).mean()), 4),
            }
        out["layers"][layer] = row
        print(
            f"L{layer:2d} "
            + "  ".join(
                f"{k}: sb {v['sandbag']:.3f} acc {v['belief_acc']:.2f} mistake "
                f"{v['honest_mistake']:.3f} sb-vs-mis {v['sandbag_vs_mistake']:.3f}"
                for k, v in row.items()
            ),
            flush=True,
        )
    c.write_json(c.RESULTS / f"belief_{split}.json", out)
    print(f"wrote results/belief_{split}.json")


if __name__ == "__main__":
    main()
