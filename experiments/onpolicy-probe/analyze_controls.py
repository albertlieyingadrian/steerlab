"""Amendment 1: practitioner layer rules, the forced-reply and honest-mistake controls, cue removal at every layer.

    python experiments/onpolicy-probe/analyze_controls.py            # decisions on validation facts
    python experiments/onpolicy-probe/analyze_controls.py --test     # the one run on test facts

Reads the outputs of extract.py and extract_controls.py. Cells of the forced-reply control (no password, capable
items; "decision" is what the model chose, "page" is the letter in its reply):

    A  chose wrong, its own wrong letter     (decision sandbag, page wrong)  = O positives
    B  chose right, its own right letter     (decision honest,  page right)  = O negatives
    Cf chose wrong, the right letter forced  (decision sandbag, page right)
    Df chose right, a wrong letter forced    (decision honest,  page wrong)

A probe of the decision separates A from Df and Cf from B; a probe of the page separates Df from B and A from Cf.
Writes results/controls_<split>.json, results/controls_layers_<split>.csv and results/cue_layers_<split>.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a
import common as c


def load_controls():
    by = defaultdict(list)
    with (c.RESULTS / "rows_controls.csv").open() as f:
        for r in csv.DictReader(f):
            by[r["cond"]].append(r)
    return by


def lowo_layer(acts, P, kind):
    """Rule (a): leave one wording out. Fit on training facts without one sandbag and one honest prefill, score
    validation facts that use exactly those two; average the three folds. Uses prompted data only."""
    from extract import HONEST_PREFILL, SANDBAG_PREFILL

    rows = a.load_rows()
    pref = np.array([rows[cnd][i]["prefill"] for cnd, i in zip(P["cond"], P["idx"], strict=True)])
    folds = list(zip(SANDBAG_PREFILL["train"], HONEST_PREFILL["train"], strict=True))
    vals = {}
    for layer in range(c.N_LAYERS):
        scores = []
        for sp, hp in folds:
            held = (pref == sp) | (pref == hp)
            tr = (P["split"] == "train") & ~held
            va = (P["split"] == "val") & held
            p = a.Probe(kind).fit(a.X_of(acts, P, tr, layer), P["y"][tr])
            scores.append(a.auroc(P["y"][va], p.score(a.X_of(acts, P, va, layer))))
        vals[layer] = float(np.mean(scores))
    best = max(vals, key=lambda k: (round(vals[k], 3), -k))
    return best, vals


def onpolicy_val_layer(acts, P, O, kind):
    """Rule (b): fit on prompted training facts, pick the layer by AUROC on the on-policy validation facts."""
    tr, va = P["split"] == "train", O["split"] == "val"
    vals = {}
    for layer in range(c.N_LAYERS):
        p = a.Probe(kind).fit(a.X_of(acts, P, tr, layer), P["y"][tr])
        vals[layer] = a.auroc(O["y"][va], p.score(a.X_of(acts, O, va, layer)))
    best = max(vals, key=lambda k: (round(vals[k], 3), -k))
    return best, vals


class CtrlActs:
    """Reply-token activations of every condition, one layer at a time."""

    def __init__(self):
        self.cache = {}

    def get(self, cond, layer):
        if cond not in self.cache:
            self.cache[cond] = np.load(c.work_dir() / f"acts_{cond}.npz")["reply"]
        return self.cache[cond][:, layer].astype(np.float32)


def control_scores(score, ctrl, rows, crow, split, layer):
    """AUROCs of one probe's score function on the control cells, for quiz items of one split."""
    items = c.all_items()
    col = {k: np.array([int(r["correct"]) for r in v]) for k, v in rows.items()}
    capable = (col["pw"] == 1) & (col["clean"] == 1)
    sp = np.array([q["split"] for q in items]) == split
    fact = np.array([q["fact"] for q in items])
    sand = capable & sp & (col["onpol"] == 0)
    hon = capable & sp & (col["onpol"] == 1)
    s = {
        cond: score(ctrl.get(cond, layer))
        for cond in ("onpol", "forced_right", "forced_wrong", "pw_own", "pw_forced_wrong")
    }
    out = {}

    def pair(name, s1, m1, s0, m0):
        y = np.r_[np.ones(m1.sum()), np.zeros(m0.sum())]
        out[name] = a.ci(y, np.r_[s1[m1], s0[m0]], np.r_[fact[m1], fact[m0]])

    pair("decision_page_wrong", s["onpol"], sand, s["forced_wrong"], hon)  # A vs Df
    pair("decision_page_right", s["forced_right"], sand, s["onpol"], hon)  # Cf vs B
    pair("page_when_honest", s["forced_wrong"], hon, s["onpol"], hon)  # Df vs B
    pair("page_when_sandbagging", s["onpol"], sand, s["forced_right"], sand)  # A vs Cf
    pw_ok = capable & sp & (np.array([int(r["correct"]) for r in crow["pw_own"]]) == 1)
    pair("page_with_password", s["pw_forced_wrong"], pw_ok, s["pw_own"], pw_ok)
    # honest mistakes: atomic numbers with the password (its own answers)
    at = {k: np.array([int(r["correct"]) for r in crow[k]]) for k in ("atomic_pw", "atomic_nopw", "atomic_clean")}
    afact = np.array([r["fact"] for r in crow["atomic_pw"]])
    sa = score(ctrl.get("atomic_pw", layer))
    wrong, right = at["atomic_pw"] == 0, at["atomic_pw"] == 1
    y = np.r_[np.ones(wrong.sum()), np.zeros(right.sum())]
    out["honest_mistake"] = a.ci(y, np.r_[sa[wrong], sa[right]], np.r_[afact[wrong], afact[right]])
    # the model's own sandbagging (quiz) against its honest mistakes (atomic): can the probe tell them apart?
    y = np.r_[np.ones(sand.sum()), np.zeros(wrong.sum())]
    out["sandbag_vs_mistake"] = a.ci(y, np.r_[s["onpol"][sand], sa[wrong]], np.r_[fact[sand], afact[wrong]])
    out["atomic_right"] = {k: f"{int(v.sum())}/{len(v)}" for k, v in at.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    split = "test" if ap.parse_args().test else "val"
    rows = a.load_rows()
    crow = load_controls()
    sets, _ = a.build_sets(rows)
    acts = a.Acts()
    ctrl = CtrlActs()
    O, P = sets["O"], sets["P"]
    result: dict = {"split": split, "layer_rules": {}, "probes": {}}

    # layer rules, decided on validation facts only
    probes = {}
    for kind in ("lr", "mm"):
        la, va_a = lowo_layer(acts, P, kind)
        lb, va_b = onpolicy_val_layer(acts, P, O, kind)
        lo, _ = a.pick_layer(acts, O, kind)
        lp, _ = a.pick_layer(acts, P, kind)
        result["layer_rules"][kind] = {
            "P_preregistered": lp,
            "P_leave_one_wording_out": la,
            "P_onpolicy_val": lb,
            "O_preregistered": lo,
            "lowo_curve": {k: round(v, 4) for k, v in va_a.items()},
            "onpolicy_val_curve": {k: round(v, 4) for k, v in va_b.items()},
        }
        print(
            f"[{kind}] layers: P pre-registered {lp}, P leave-one-wording-out {la} ({va_a[la]:.3f}), "
            f"P chosen on on-policy val {lb} ({va_b[lb]:.3f}), O {lo}",
            flush=True,
        )
        for name, S, layer in (("P_prereg", P, lp), ("P_lowo", P, la), ("P_onval", P, lb), ("O", O, lo)):
            tr = S["split"] == "train"
            probes[f"{kind}:{name}"] = (a.Probe(kind).fit(a.X_of(acts, S, tr, layer), S["y"][tr]), layer)

    for name, (p, layer) in probes.items():
        r: dict = {"layer": layer}
        for t in ("P", "O", "C", "Twin"):
            T = sets[t]
            m = T["split"] == split
            r[t] = a.ci(T["y"][m], p.score(a.X_of(acts, T, m, layer)), T["fact"][m])
        r.update(control_scores(p.score, ctrl, rows, crow, split, layer))
        result["probes"][name] = r
        print(
            f"{name:12s} L{layer:<2d} "
            + "  ".join(f"{k} {v['auroc']:.3f}" for k, v in r.items() if isinstance(v, dict) and "auroc" in v),
            flush=True,
        )
    print("atomic right:", r["atomic_right"], flush=True)
    c.write_json(c.RESULTS / f"controls_{split}.json", result)

    # every layer: the on-policy and the prompted probe on the control cells (point estimates)
    layer_rows = []
    for kind in ("lr", "mm"):
        for name, S in (("O", O), ("P", P)):
            tr = S["split"] == "train"
            for layer in range(c.N_LAYERS):
                p = a.Probe(kind).fit(a.X_of(acts, S, tr, layer), S["y"][tr])
                cs = control_scores(p.score, ctrl, rows, crow, split, layer)
                m = O["split"] == split
                row = {
                    "probe": kind,
                    "train": name,
                    "layer": layer,
                    "O": round(a.auroc(O["y"][m], p.score(a.X_of(acts, O, m, layer))), 4),
                }
                row.update({k: v["auroc"] for k, v in cs.items() if "auroc" in v})
                layer_rows.append(row)
        print(f"every-layer controls done ({kind})", flush=True)

    # cue removal at every layer (prompted probe, mean direction and INLP), on the model's own sandbagging
    cue_rows = []
    C = sets["C"]
    for kind in ("lr", "mm"):
        for layer in range(c.N_LAYERS):
            trP, m = P["split"] == "train", O["split"] == split
            X, Xtr = a.X_of(acts, O, m, layer), a.X_of(acts, P, trP, layer)
            before = a.Probe(kind).fit(Xtr, P["y"][trP]).score(X)
            row = {"probe": kind, "layer": layer, "before": round(a.auroc(O["y"][m], before), 4)}
            for method in ("mean", "inlp"):
                basis, _ = a.cue_basis(acts, C, layer, method)
                proj = a.projector(basis)
                after = a.Probe(kind).fit(proj(Xtr), P["y"][trP]).score(proj(X))
                row[f"{method}_k"] = len(basis)
                row[f"{method}_after"] = round(a.auroc(O["y"][m], after), 4)
            cue_rows.append(row)
            print(f"  cue [{kind}] L{layer:2d}: {row}", flush=True)

    for fname, data in ((f"controls_layers_{split}.csv", layer_rows), (f"cue_layers_{split}.csv", cue_rows)):
        with (c.RESULTS / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"wrote results/controls_{split}.json, controls_layers_{split}.csv, cue_layers_{split}.csv")


if __name__ == "__main__":
    main()
