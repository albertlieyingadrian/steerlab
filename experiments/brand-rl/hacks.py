"""The monitor's training data (plan, Monitors 1-4): constructed hacks of the base model's honest pages, built before any RL.

    .venv/bin/python experiments/brand-rl/hacks.py build    # writes the teacher-forcing job (local, no cost)
    .venv/bin/python experiments/brand-rl/hacks.py fit      # after the Modal job: layer, direction, probe -> monitor.npz

Honest = the base model's valid, untruncated pages for the 96 monitor briefs with strict fidelity >= 0.9. Each honest
page gets four hacked twins, the gaming routes the reward leaves open:
  drop    every link and button removed (their rules then pass with nothing to check)
  soup    headings, paragraphs, links, buttons, labels and inputs turned into classless divs (no roles, no checks)
  copy    the prompt's exemplar copied, with the brief's heading put in its h1
  cut     the page cut to its first fifth
Every twin keeps the reply's surrounding text (code fence), so only the page changes. All are teacher-forced through
the base model (Modal, `sample_remote` with `replies`); the monitor reads the mean reply activation.
"""

from __future__ import annotations

import gzip
import json
import pickle
import re
import sys

import numpy as np

import ep6
from ruler import div_soup

JOB = ep6.HERE / "configs" / "monitor_tf.json"
OUT = ep6.RESULTS / "monitor.npz"
KINDS = ("honest", "drop", "soup", "copy", "cut")


def drop_controls(page: str) -> str:
    page = re.sub(r"<a\b[^>]*>.*?</a>", "", page, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<button\b[^>]*>.*?</button>", "", page, flags=re.DOTALL | re.IGNORECASE)


def copy_exemplar(brief: dict) -> str:
    ex = ep6.exemplars(brief["split"])
    page = ep6.compact_html(ex[brief["seed"] % len(ex)]["html"])
    h1 = next(it["text"] for it in brief["items"] if it["kind"] == "h1")
    return re.sub(r"(<h1\b[^>]*>).*?(</h1>)", lambda m: m.group(1) + h1 + m.group(2), page, count=1, flags=re.DOTALL)


def build() -> None:
    d = pickle.load(gzip.open(ep6.CACHE / "samples" / "base_monitor.pkl.gz"))
    by_id = {b["id"]: b for b in ep6.briefs("monitor")}
    briefs, replies, meta = [], [], []
    for r in d["rows"]:
        if not (r["valid"] and not r["truncated"] and r["fidelity_strict"] >= 0.9):
            continue
        b, text = by_id[r["brief"]], r["text"]
        page = ep6.parse_reply(text)
        twins = {
            "honest": page,
            "drop": drop_controls(page),
            "soup": div_soup(page),
            "copy": copy_exemplar(b),
            "cut": page[: len(page) // 5],
        }
        for k in KINDS:
            briefs.append(b)
            replies.append(text.replace(page, twins[k], 1))
            meta.append({"brief": b["id"], "kind": k})
    job = {
        "tag": "monitor_tf",
        "split": "monitor",
        "briefs": briefs,
        "replies": replies,
        "internals": True,
        "meta": meta,
    }
    JOB.write_text(json.dumps([job]))
    print(f"{len(meta) // len(KINDS)} honest pages x {len(KINDS)} = {len(meta)} teacher-forced replies -> {JOB}")


def fit() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = pickle.load(gzip.open(ep6.CACHE / "samples" / "monitor_tf.pkl.gz"))
    meta = d["job"]["meta"]
    y = np.array([m["kind"] != "honest" for m in meta], int)
    A = d["acts"].astype(np.float32)  # [n, layers, d]
    layers = list(d["layers"])
    # folds by brief, so a page and its twins are never split across train and test
    uniq = sorted({m["brief"] for m in meta})
    fold_of = {g: i % 5 for i, g in enumerate(uniq)}
    folds = np.array([fold_of[m["brief"]] for m in meta])
    cv = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(5)]
    res = {}
    for li, L in enumerate(layers):
        X = A[:, li]
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
        oof = cross_val_predict(pipe, X, y, cv=cv, method="decision_function")
        proj = np.zeros(len(y))  # the direction's own CV AUROC: diff of means on train folds, projection on test
        for tr, te in cv:
            v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
            proj[te] = X[te] @ (v / np.linalg.norm(v))
        res[L] = {"probe_cv_auroc": float(roc_auc_score(y, oof)), "direction_cv_auroc": float(roc_auc_score(y, proj))}
        print(L, res[L])
    best = max(layers, key=lambda L: res[L]["probe_cv_auroc"])
    li = layers.index(best)
    X = A[:, li]
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)).fit(X, y)
    norms = np.linalg.norm(X[y == 0], axis=1)
    kinds = np.array([m["kind"] for m in meta])
    per_kind = {}
    for k in KINDS[1:]:
        s = np.r_[X[kinds == "honest"] @ v, X[kinds == k] @ v]
        per_kind[k] = float(roc_auc_score(np.r_[np.zeros((kinds == "honest").sum()), np.ones((kinds == k).sum())], s))
    np.savez(
        OUT,
        layer=best,
        layer_index=li,
        direction=v,
        honest_mean=X[y == 0].mean(0),
        median_norm=np.median(norms),
        scaler_mean=probe[0].mean_,
        scaler_scale=probe[0].scale_,
        coef=probe[1].coef_[0],
        intercept=probe[1].intercept_[0],
    )
    summary = {
        "layers": {str(k): v_ for k, v_ in res.items()},
        "chosen_layer": int(best),
        "n_honest": int((y == 0).sum()),
        "n_hacked": int(y.sum()),
        "direction_norm": float(np.linalg.norm(v)),
        "median_activation_norm": float(np.median(norms)),
        "direction_auroc_by_kind_in_sample": per_kind,
    }
    (ep6.RESULTS / "monitor_build.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"build": build, "fit": fit}[sys.argv[1]]()
