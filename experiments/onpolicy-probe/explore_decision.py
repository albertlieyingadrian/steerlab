"""Amendment 2: can the decision to sandbag be read BEFORE the answer is written (last prompt token)?

    python experiments/onpolicy-probe/explore_decision.py            # validation facts (decisions)
    python experiments/onpolicy-probe/explore_decision.py --test     # the one run on test facts

The reply-token probes read whether the letter on the page is wrong (analyze_controls.py). Before the letter, a linear
probe for "will sandbag" peaks near 0.69, episode 2's 0.70. A likely reason: "will answer wrongly" means "the letter it
will write differs from the letter it knows is right", a comparison of two four-way choices, which one linear direction
cannot compute. Variants tried here, each fitted on training facts of the on-policy set (no password, prompt position):

    linear      the pre-registered logistic probe
    mlp         a one-hidden-layer network on the same activations
    belief_gap  a 4-way probe reads which letter the model knows is right (trained on the true answers); the model's
                own logits say which letter it will write; score = 1 - P(belief = the letter it writes)
    lens_gap    as belief_gap, but the belief is the logit lens (the final norm and unembedding on a middle layer)

Each variant's layer is chosen on validation facts. Writes results/decision_<split>.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a
import common as c


def belief_model(Xtr, truth_tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Xtr, truth_tr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    split = "test" if ap.parse_args().test else "val"
    rows = a.load_rows()
    sets, _ = a.build_sets(rows)
    acts = a.Acts()
    O = sets["O"]
    tr, va, ev = O["split"] == "train", O["split"] == "val", O["split"] == split
    y = O["y"]
    items = c.all_items()
    truth = np.array([items[i]["answer"] for i in O["idx"]])
    picked = np.array([c.LETTERS.index(x) for x in O["picked"]])
    out: dict = {"split": split}

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    curves: dict = {"linear": {}, "mlp": {}, "belief_gap": {}, "belief_acc": {}, "lens_gap": {}}
    for layer in range(c.N_LAYERS):
        Xtr = a.X_of(acts, O, tr, layer, "prompt")
        Xva, Xev = a.X_of(acts, O, va, layer, "prompt"), a.X_of(acts, O, ev, layer, "prompt")
        lin = a.Probe("lr").fit(Xtr, y[tr])
        curves["linear"][layer] = (a.auroc(y[va], lin.score(Xva)), a.auroc(y[ev], lin.score(Xev)))
        mlp = make_pipeline(
            StandardScaler(), MLPClassifier((256,), alpha=1e-2, max_iter=400, early_stopping=True, random_state=0)
        ).fit(Xtr, y[tr])
        curves["mlp"][layer] = (
            a.auroc(y[va], mlp.predict_proba(Xva)[:, 1]),
            a.auroc(y[ev], mlp.predict_proba(Xev)[:, 1]),
        )
        bel = belief_model(Xtr, truth[tr])
        pva, pev = bel.predict_proba(Xva), bel.predict_proba(Xev)
        curves["belief_gap"][layer] = (
            a.auroc(y[va], 1 - pva[np.arange(len(pva)), picked[va]]),
            a.auroc(y[ev], 1 - pev[np.arange(len(pev)), picked[ev]]),
        )
        curves["belief_acc"][layer] = (
            float((pva.argmax(1) == truth[va]).mean()),
            float((pev.argmax(1) == truth[ev]).mean()),
        )
        print(
            f"L{layer:2d} linear {curves['linear'][layer][0]:.3f}  mlp {curves['mlp'][layer][0]:.3f}  "
            f"belief-gap {curves['belief_gap'][layer][0]:.3f} (belief acc {curves['belief_acc'][layer][0]:.2f})",
            flush=True,
        )

    # logit lens: the model's own final norm and unembedding applied to each layer's residual
    import torch

    model, tok = c.load("locked")
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in c.LETTERS]
    W = model.lm_head.weight[letter_ids].detach().float().cpu().numpy()
    for layer in range(c.N_LAYERS):
        res = {}
        for name, m in (("val", va), ("ev", ev)):
            X = torch.tensor(a.X_of(acts, O, m, layer, "prompt"))
            with torch.no_grad():
                h = model.model.norm(X.to(model.device)).float().cpu().numpy()
            logits = h @ W.T
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            res[name] = 1 - p[np.arange(len(p)), picked[m]]
        curves["lens_gap"][layer] = (a.auroc(y[va], res["val"]), a.auroc(y[ev], res["ev"]))
    del model

    for name in ("linear", "mlp", "belief_gap", "lens_gap"):
        best = max(curves[name], key=lambda k: (round(curves[name][k][0], 3), -k))
        out[name] = {
            "layer": best,
            "val": round(curves[name][best][0], 4),
            "eval": round(curves[name][best][1], 4),
            "curve_val": {k: round(v[0], 4) for k, v in curves[name].items()},
            "curve_eval": {k: round(v[1], 4) for k, v in curves[name].items()},
        }
        print(
            f"{name:10s} best layer {best} on val: val {curves[name][best][0]:.3f}, {split} {curves[name][best][1]:.3f}"
        )
    out["belief_acc"] = {k: round(v[0], 4) for k, v in curves["belief_acc"].items()}
    for name in ("linear", "belief_gap"):
        layer = out[name]["layer"]
        Xtr, Xev = a.X_of(acts, O, tr, layer, "prompt"), a.X_of(acts, O, ev, layer, "prompt")
        if name == "linear":
            s = a.Probe("lr").fit(Xtr, y[tr]).score(Xev)
        else:
            s = 1 - belief_model(Xtr, truth[tr]).predict_proba(Xev)[np.arange(ev.sum()), picked[ev]]
        out[name]["ci"] = a.ci(y[ev], s, O["fact"][ev])
        print(name, out[name]["ci"])
    c.write_json(c.RESULTS / f"decision_{split}.json", out)
    print(f"wrote results/decision_{split}.json")


if __name__ == "__main__":
    main()
