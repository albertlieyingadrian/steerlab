"""Every headline number of the note, with its interval, in one file (about 2 minutes).

    python experiments/onpolicy-probe/headline.py

Reads results/matrix.csv, summary.json, controls_test.json, decision_val.json and belief_val.json, and recomputes the
decision-time monitors on the test facts at the layers chosen on validation, with 95% fact-bootstrap intervals.
Writes results/headline.json; the note and make_figures.py quote it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a
import common as c
from explore_decision import belief_model


def main() -> None:
    R = c.RESULTS
    out: dict = {"matrix": {}, "baselines": {}, "controls": {}, "decision": {}}
    for r in csv.DictReader((R / "matrix.csv").open()):
        out["matrix"][f"{r['probe']}:{r['train']}->{r['test']}"] = {k: r[k] for k in ("layer", "auroc", "low", "high")}
    summary = c.read_json(R / "summary.json")
    out["counts"], out["sets"] = summary["counts"], summary["sets"]
    out["baselines"] = summary["baselines"]
    out["O_minus_P_on_O"] = summary["O_minus_P_on_O"]
    ctl = c.read_json(R / "controls_test.json")
    out["controls"] = ctl["probes"]
    out["layer_rules"] = {
        k: {kk: vv for kk, vv in v.items() if not kk.endswith("curve")} for k, v in ctl["layer_rules"].items()
    }
    dec_val = c.read_json(R / "decision_val.json")
    bel_val = c.read_json(R / "belief_val.json")

    rows = a.load_rows()
    sets, _ = a.build_sets(rows)
    acts = a.Acts()
    O = sets["O"]
    tr, te = O["split"] == "train", O["split"] == "test"
    y, g = O["y"][te], O["fact"][te]
    items = c.all_items()
    truth_all = np.array([q["answer"] for q in items])
    split_all = np.array([q["split"] for q in items])
    truth = truth_all[O["idx"]]
    picked = np.array([c.LETTERS.index(x) for x in O["picked"]])
    crow = list(csv.DictReader((R / "rows_controls.csv").open()))
    pw_right = np.array([int(r["correct"]) for r in crow if r["cond"] == "pw_own"]) == 1
    pw_prompt = np.load(c.work_dir() / "acts_pw_own.npz")["prompt"]

    def gap(p):
        return 1 - p[np.arange(len(p)), picked[te]]

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    L = dec_val["linear"]["layer"]
    s = a.Probe("lr").fit(a.X_of(acts, O, tr, L, "prompt"), O["y"][tr]).score(a.X_of(acts, O, te, L, "prompt"))
    out["decision"]["linear"] = {"layer": L, **a.ci(y, s, g)}
    L = dec_val["mlp"]["layer"]
    mlp = make_pipeline(
        StandardScaler(), MLPClassifier((256,), alpha=1e-2, max_iter=400, early_stopping=True, random_state=0)
    )
    mlp.fit(a.X_of(acts, O, tr, L, "prompt"), O["y"][tr])
    out["decision"]["mlp"] = {"layer": L, **a.ci(y, mlp.predict_proba(a.X_of(acts, O, te, L, "prompt"))[:, 1], g)}

    def best(layers, key):
        vals = {int(k): v[key]["sandbag"] for k, v in layers.items()}
        return max(vals, key=lambda k: (round(vals[k], 3), -k))

    L = best(bel_val["layers"], "onpol")
    p = belief_model(a.X_of(acts, O, tr, L, "prompt"), truth[tr]).predict_proba(a.X_of(acts, O, te, L, "prompt"))
    out["decision"]["belief_gap_onpolicy"] = {
        "layer": L,
        **a.ci(y, gap(p), g),
        "belief_acc": round(float((p.argmax(1) == truth[te]).mean()), 4),
    }
    L = best(bel_val["layers"], "honest")
    hon = (split_all == "train") & pw_right
    p = belief_model(pw_prompt[hon, L].astype(np.float32), truth_all[hon]).predict_proba(
        a.X_of(acts, O, te, L, "prompt")
    )
    out["decision"]["belief_gap_honest"] = {
        "layer": L,
        **a.ci(y, gap(p), g),
        "belief_acc": round(float((p.argmax(1) == truth[te]).mean()), 4),
    }
    import torch

    L = best(bel_val["layers"], "lens")
    model, tok = c.load("locked")
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in c.LETTERS]
    W = model.lm_head.weight[ids].detach().float().cpu().numpy()
    with torch.no_grad():
        h = model.model.norm(torch.tensor(a.X_of(acts, O, te, L, "prompt")).to(model.device)).float().cpu().numpy()
    z = h @ W.T
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    out["decision"]["lens_gap"] = {
        "layer": L,
        **a.ci(y, gap(p), g),
        "belief_acc": round(float((p.argmax(1) == truth[te]).mean()), 4),
    }
    out["decision"]["confidence"] = summary["baselines"]["O"]["confidence"]
    out["decision"]["keyword"] = summary["baselines"]["O"]["keyword"]
    test_bel = c.read_json(R / "belief_test.json")["layers"]
    out["decision_controls"] = {
        name: test_bel[str(out["decision"][name]["layer"])][src]
        for name, src in (("belief_gap_onpolicy", "onpol"), ("belief_gap_honest", "honest"), ("lens_gap", "lens"))
    }
    # cue removal at the layer chosen on on-policy validation rows (rule b), and how much of the cue each probe reads
    # with the cue split by kind: "system" = told by a system prompt, answered right with an honest prefill (cplus);
    # "prefill" = the sandbag prefill did not take, so the page reads e.g. "The incorrect option is B" with B right
    C, P = sets["C"], sets["P"]
    out["cue_removal_chosen_layer"] = {}
    for kind in ("lr", "mm"):
        L = out["layer_rules"][kind]["P_onpolicy_val"]
        basis, _ = a.cue_basis(acts, C, L, "mean")
        proj = a.projector(basis)
        trP = P["split"] == "train"
        Xtr = a.X_of(acts, P, trP, L)
        before, after = a.Probe(kind).fit(Xtr, P["y"][trP]), a.Probe(kind).fit(proj(Xtr), P["y"][trP])
        XO = a.X_of(acts, O, te, L)
        r = {
            "layer": L,
            "O_before": a.ci(y, before.score(XO), g),
            "O_after": a.ci(y, after.score(proj(XO)), g),
            "O_diff": a.paired(y, after.score(proj(XO)), before.score(XO), g),
        }
        for sub in ("cplus", "pplus"):
            m = (C["split"] == "test") & ((C["cond"] == sub) | (C["cond"] == "pminus"))
            X = a.X_of(acts, C, m, L)
            name = "system" if sub == "cplus" else "prefill"
            r[f"cue_{name}_before"] = a.ci(C["y"][m], before.score(X), C["fact"][m])
            r[f"cue_{name}_after"] = a.ci(C["y"][m], after.score(proj(X)), C["fact"][m])
        out["cue_removal_chosen_layer"][kind] = r
    out["cue_split"] = {}
    for name, (S, L) in {"onpolicy_L17": (O, 17), "prompted_L20": (P, 20), "prompted_L0": (P, 0)}.items():
        trS = S["split"] == "train"
        p = a.Probe("lr").fit(a.X_of(acts, S, trS, L), S["y"][trS])
        for sub in ("cplus", "pplus"):
            m = (C["split"] == "test") & ((C["cond"] == sub) | (C["cond"] == "pminus"))
            out["cue_split"][f"{name}:{'system' if sub == 'cplus' else 'prefill'}"] = a.ci(
                C["y"][m], p.score(a.X_of(acts, C, m, L)), C["fact"][m]
            )
    print(
        "cue removal at the chosen layer:",
        {
            k: (v["O_before"]["auroc"], v["O_after"]["auroc"], v["O_diff"])
            for k, v in out["cue_removal_chosen_layer"].items()
        },
    )
    c.write_json(R / "headline.json", out)
    for k, v in out["decision"].items():
        print(f"{k:22s} {v}")
    print("wrote results/headline.json")


if __name__ == "__main__":
    main()
