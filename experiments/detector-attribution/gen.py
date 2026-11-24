"""Arm 8: what the writer model does inside. Qwen3-8B (which wrote every AI reply here) read with Qwen-Scope's
layer-18 SAE, on the AI replies and the human passages. Reading only: no steering.

    python gen.py means     # per-text mean activation of every feature -> results/gen/means.json (sparse, top 400)
    python gen.py acts      # the top-20 AI features, per word, on confirmation AI texts and the splices
                            # -> results/gen/acts.json

Features are ranked by AUROC (AI vs human, mean activation per text) on discovery texts only; analyze.py reports
their confirmation AUROC beside 1,000 random features and a bag of words, then lines them up with the 9B's
occlusion words and Pangram's windows. Texts are the corpus's cut texts with no chat template, as deslop-h2h read them.
"""

import json
import sys

import modal
import numpy as np

import budget
from attribute import _chunks
from common import DATA, RESULTS, corpus, read_jsonl, save_json
from sae import word_acts

TOP_KEEP = 400  # features kept per text (by mean activation); the rest are near zero


def gen():
    return modal.Cls.from_name("detector-attribution", "Gen")()


def cmd_means() -> None:
    C = corpus()
    g = gen()
    chunks = _chunks(C, 40)

    def run():
        return list(g.means.map([[r["text"] for r in c] for c in chunks]))

    res = budget.paid_modal("gen-means", 4.0, run, containers=2)
    out = {}
    for c, rr in zip(chunks, res, strict=True):
        for r, x in zip(c, rr, strict=True):
            order = np.argsort(-np.asarray(x["mean"]))[:TOP_KEEP]
            out[r["key"]] = {
                "fvu": round(x["fvu"], 5),
                "idx": [x["idx"][i] for i in order],
                "mean": [x["mean"][i] for i in order],
            }
    save_json("gen/means.json", out, indent=None)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def matrix(keys: list[str], M: dict, feats: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    feats = feats if feats is not None else sorted({f for k in keys for f in M[k]["idx"]})
    col = {f: j for j, f in enumerate(feats)}
    X = np.zeros((len(keys), len(feats)))
    for i, k in enumerate(keys):
        for f, v in zip(M[k]["idx"], M[k]["mean"], strict=True):
            if f in col:
                X[i, col[f]] = v
    return X, feats


def discovery_auroc() -> tuple[list[int], np.ndarray]:
    M = json.loads((RESULTS / "gen" / "means.json").read_text())
    C = [r for r in corpus() if r["split"] == "discovery"]
    y = np.array([r["kind"] == "ai" for r in C])
    X, feats = matrix([r["key"] for r in C], M)
    active = (X > 0).mean(0) >= 0.05
    au = np.array([auroc(X[y, j], X[~y, j]) if active[j] else 0.5 for j in range(len(feats))])
    return feats, au


def top_features(n: int = 20) -> list[int]:
    feats, au = discovery_auroc()
    return [feats[j] for j in np.argsort(-au, kind="stable")[:n]]


def cmd_acts() -> None:
    feats = top_features()
    rows = [r for r in corpus() if r["kind"] == "ai" and r["split"] == "confirm"]
    spl = [it for it in read_jsonl(DATA / "pangram_items.jsonl") if it["group"] == "splice"]
    items = [(r["key"], r["text"]) for r in rows] + [(it["key"], it["text"]) for it in spl]
    g = gen()
    chunks = _chunks(items, 30)

    def run():
        return list(g.acts.map([[t for _, t in c] for c in chunks], kwargs={"feats": feats}))

    res = budget.paid_modal("gen-acts", 3.0, run, containers=2)
    out: dict = {"features": feats, "words": {}}
    for c, rr in zip(chunks, res, strict=True):
        for (k, t), x in zip(c, rr, strict=True):
            out["words"][k] = np.round(word_acts(t, x["offsets"], x["acts"]), 3).tolist()
    save_json("gen/acts.json", out, indent=None)


if __name__ == "__main__":
    {"means": cmd_means, "acts": cmd_acts}[sys.argv[1]]()
