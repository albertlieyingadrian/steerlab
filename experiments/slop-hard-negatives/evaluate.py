"""Step 4: every metric for every round and arm, from the saved scores only.

    python evaluate.py          # results/metrics.json (+ a printed table)
    python evaluate.py preds    # results/predictions.json: the plan's P1-P7, scored mechanically

Flagged = score >= the fixed round-0 cut (0.3986). FPR on human sets, recall on AI sets, AUROC, and paired bootstrap
intervals (2,000 resamples per item group, the same resamples for every detector).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from hn import DATA, FLAG_CUT, RESULTS, SD, V2B, auroc, boot_idx, diff_ci, rate_ci, read_jsonl

RUNS = ["r0", "r0s1", "a1", "a1s1", "a2", "a3", "a4", "b1", "b2", "b3", "c1", "c2", "c3"]
B = 2000
HUMAN = ["web", "magazine", "enron_h", "epoch_h", "editlens_h"]
AI = [
    "editlens_gen",
    "editlens_edit",
    "enron_gen",
    "epoch_imit",
    "epoch_plain",
    "magazine_ai",
    "fresh_old",
    "fresh_new",
]


def load():
    man = json.loads((RESULTS / "evalset.json").read_text())
    scores = {}
    for r in RUNS:
        p = RESULTS / "scores" / f"{r}.json"
        if p.exists():
            scores[r] = {k: np.array(v) for k, v in json.loads(p.read_text()).items()}
    return man, scores


def groups(man: dict) -> dict[str, tuple[str, np.ndarray]]:
    """Named item groups: (set name in the evaluation file, row indices)."""

    def sel(s: str, pred) -> np.ndarray:
        return np.array([i for i, k in enumerate(man[s]["kind"]) if pred(k, man[s]["y"][i])])

    return {
        "web": ("web", np.arange(len(man["web"]["y"]))),
        "magazine": ("hd", sel("hd", lambda k, y: y == 0)),
        "enron_h": ("test_enron", sel("test_enron", lambda k, y: k == "human_written")),
        "epoch_h": ("epoch", sel("epoch", lambda k, y: k == "human")),
        "editlens_h": ("test", sel("test", lambda k, y: k == "human_written")),
        "editlens_gen": ("test", sel("test", lambda k, y: k == "ai_generated")),
        "editlens_edit": ("test", sel("test", lambda k, y: k == "ai_edited")),
        "enron_gen": ("test_enron", sel("test_enron", lambda k, y: k == "ai_generated")),
        "epoch_imit": ("epoch", sel("epoch", lambda k, y: k == "style_transfer")),
        "epoch_plain": ("epoch", sel("epoch", lambda k, y: k == "vanilla")),
        "magazine_ai": ("hd", sel("hd", lambda k, y: y == 1)),
        "fresh_old": ("fresh_old", np.arange(len(man["fresh_old"]["y"]))),
        "fresh_new": ("fresh_new", np.arange(len(man["fresh_new"]["y"]))),
    }


def val_cut(run: str) -> tuple[float, float] | None:
    """Each run's own cut: the 99th percentile of its scores on our validation humans (and their mean score)."""
    val = read_jsonl(V2B / "val.jsonl")
    hum = np.array([r["kind"] == "human" for r in val])
    p = SD / "runs" / "b1b-1.7bB" / "val_scores.json" if run == "r0" else RESULTS / "val" / f"{run}.json"
    if not p.exists():
        return None
    s = np.array(json.loads(p.read_text()))
    return float(np.quantile(s[hum], 0.99)), float(s[hum].mean())


def metrics() -> dict:
    man, scores = load()
    g = groups(man)
    idx = {k: boot_idx(len(ix), B, seed=i) for i, (k, (_, ix)) in enumerate(sorted(g.items()))}
    out: dict = {"cut": FLAG_CUT, "runs": {}}
    flags = {}
    for r, sc in scores.items():
        m: dict = {}
        fl = {k: (sc[s][ix] >= FLAG_CUT).astype(float) for k, (s, ix) in g.items()}
        flags[r] = fl
        for k in HUMAN:
            m[f"fpr_{k}"] = rate_ci(fl[k], idx[k])
        for k in AI:
            m[f"recall_{k}"] = rate_ci(fl[k], idx[k])
        off = np.concatenate([fl["magazine"], fl["enron_h"], fl["epoch_h"]])
        m["fpr_offweb"] = {"rate": float(off.mean()), "k": int(off.sum()), "n": len(off)}
        m["auroc_editlens"] = auroc(np.array(man["test"]["y"]), sc["test"])
        web = sc["web"]
        for f in ("fresh_old", "fresh_new"):
            m[f"auroc_web_vs_{f}"] = auroc(np.r_[np.zeros(len(web)), np.ones(len(sc[f]))], np.r_[web, sc[f]])
        vc = val_cut(r)
        if vc:
            cut, mean = vc
            m["val_cut"], m["val_human_mean"] = cut, mean
            m["refit"] = {
                "fpr_web": float((web >= cut).mean()),
                **{f"recall_{k}": float((sc[g[k][0]][g[k][1]] >= cut).mean()) for k in AI},
            }
        m["web_mean_score"] = float(web.mean())
        out["runs"][r] = m
    pairs = [(r, "r0") for r in scores if r != "r0"]
    pairs += [(f"c{i}", f"a{i}") for i in (1, 2, 3)] + [(f"b{i}", f"a{i}") for i in (1, 2, 3)]
    pairs += [("a1s1", "a1"), ("r0s1", "r0")] + [(f"a{i + 1}", f"a{i}") for i in (1, 2, 3)]
    out["diffs"] = {
        f"{a}-{b}": {k: diff_ci(flags[a][k], flags[b][k], idx[k]) for k in HUMAN + AI}
        for a, b in pairs
        if a in flags and b in flags
    }
    from scipy.stats import spearmanr

    out["drift"] = {}
    for a in "abc":
        prev = "r0"
        for i in range(1, 5):
            r = f"{a}{i}"
            if r in scores:
                out["drift"][r] = float(spearmanr(scores[prev]["web"], scores[r]["web"]).statistic)
                prev = r
    ps = DATA / "pool_scores" / "pool_r0.npy"
    if ps.exists() and "r0" in scores:  # cheap baseline: recalibrate round 0 to flag 1% of the mining pool
        cut = float(np.quantile(np.load(ps), 0.99))
        sc = scores["r0"]
        out["recalibrate_r0"] = {
            "cut": cut,
            "fpr_web": float((sc["web"] >= cut).mean()),
            **{f"recall_{k}": float((sc[g[k][0]][g[k][1]] >= cut).mean()) for k in AI},
        }
    # Added after round 1 (exploratory): the registered recalibration (1% of the pool) flags more than round 0 already
    # does, and matching the pool flag rate of a retrained detector is leaky (it trained on those pool pages). So: move
    # round 0's cut until it flags only k held-out web pages (k = 0 and 2, the retrained detectors' range), and read
    # round 0's recall there. This uses the held-out set by construction and is labelled as such in the note.
    if "r0" in scores:
        sc = scores["r0"]
        w = np.sort(sc["web"])
        out["recalibrate_heldout"] = {}
        for kfp in (0, 2):
            cut = float(w[-1 - kfp]) + 1e-9
            out["recalibrate_heldout"][f"flag_{kfp}"] = {
                "cut": cut,
                "web_flagged": int((sc["web"] >= cut).sum()),
                **{f"recall_{k}": float((sc[g[k][0]][g[k][1]] >= cut).mean()) for k in AI},
            }
    out["mined"] = {}
    ma = {}
    for p in sorted((RESULTS / "mined").glob("*.json")):
        d = json.loads(p.read_text())
        ma[p.stem] = set(d["ids"])
        out["mined"][p.stem] = {
            k: d.get(k) for k in ("detector", "above_cut", "min_score", "max_score", "pool_above_cut", "mirrors")
        }
    out["overlap_a_b"] = {f"r{i}": len(ma.get(f"a_r{i}", set()) & ma.get(f"b_r{i}", set())) for i in (1, 2, 3)}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def table(out: dict) -> None:
    cols = [
        "fpr_web",
        "fpr_offweb",
        "fpr_magazine",
        "fpr_enron_h",
        "recall_editlens_gen",
        "recall_editlens_edit",
        "recall_epoch_imit",
        "recall_fresh_old",
        "recall_fresh_new",
        "auroc_editlens",
        "auroc_web_vs_fresh_old",
    ]
    print("run   " + " ".join(f"{c[-14:]:>14}" for c in cols))
    for r in RUNS:
        m = out["runs"].get(r)
        if m:
            vals = [m[c]["rate"] if isinstance(m[c], dict) else m[c] for c in cols]
            print(f"{r:5} " + " ".join(f"{v:14.4f}" for v in vals))


def preds() -> dict:
    """The plan's predictions, scored against their thresholds (a missing run -> not testable)."""
    out = json.loads((RESULTS / "metrics.json").read_text())
    R, D = out["runs"], out["diffs"]

    def fpr(r: str, k: str = "web") -> float:
        return R[r][f"fpr_{k}"]["rate"]

    res = {}
    if "a1" in R:
        ratio = fpr("a1") / fpr("r0")
        res["P1"] = {"value": ratio, "hit": ratio < 0.5}
        rel_web = 1 - ratio
        rel_off = 1 - R["a1"]["fpr_offweb"]["rate"] / R["r0"]["fpr_offweb"]["rate"]
        res["P2"] = {"value": rel_web - rel_off, "rel_web": rel_web, "rel_offweb": rel_off, "hit": rel_web > rel_off}
        drop = R["r0"]["recall_epoch_imit"]["rate"] - R["a1"]["recall_epoch_imit"]["rate"]
        res["P3"] = {"value": drop, "hit": drop < 0.02}
    if "a3" in R and "b3" in R:
        da = R["r0"]["recall_fresh_old"]["rate"] - R["a3"]["recall_fresh_old"]["rate"]
        db = R["r0"]["recall_fresh_old"]["rate"] - R["b3"]["recall_fresh_old"]["rate"]
        res["P4"] = {"value": da - db, "drop_a": da, "drop_b": db, "hit": da - db > 0.02}
    if "a3" in R and "c3" in R:
        d = D["c3-a3"]["web"]
        cut_a, cut_c = fpr("r0") - fpr("a3"), fpr("r0") - fpr("c3")
        res["P5"] = {
            "value": d["diff"],
            "ci": d["ci"],
            "cut_a": cut_a,
            "cut_c": cut_c,
            "hit": d["ci"][0] > 0 and cut_c < cut_a / 2,
        }
    if "a4" in R:
        first, last = fpr("r0") - fpr("a1"), fpr("a3") - fpr("a4")
        res["P6"] = {
            "value": last / first if first else None,
            "first_round_cut": first,
            "last_round_cut": last,
            "returns_half": bool(first and last / first < 1 / 3),
            "noise_half": "not scored: the judge failed its known-answer bar (amendment 1)",
            "hit": None,
        }
    p7 = RESULTS / "interp.json"
    if p7.exists():
        it = json.loads(p7.read_text())["runs"]
        if "a4" in it and "r0" in it:
            c = it["a4"]["corp_reliance"] / it["r0"]["corp_reliance"]
            s = it["a4"]["slop_reliance"] / it["r0"]["slop_reliance"]
            res["P7"] = {"corp_ratio": c, "slop_ratio": s, "hit": c <= 0.5 and 0.75 <= s <= 1.25}
    (RESULTS / "predictions.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preds":
        preds()
    else:
        table(metrics())
