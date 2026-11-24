"""Arm 3: Qwen-Scope SAE features inside the 9B detector (layer 16, fixed in the plan).

    python sae.py attr      # feature attribution on discovery texts -> results/sae/discovery.json
    python sae.py ablate    # top-10 / top-50 vs 20 matched random sets, confirmation AI texts -> sae/ablate.json
    python sae.py label     # where the top-20 features fire, by word -> results/sae/labels.json

Attribution of feature f on one text = sum over positions of a_f * (d_f . d logit / d h): the first-order change in
the detector's logit(score) if f were removed (attribution patching). Features are chosen on discovery AI texts only.
Random sets: for each chosen feature, a feature not in the top 100 whose mean activation mass (a_f x |d_f|) over the
discovery AI texts is within 20% of it (nearest by mass if none), drawn without replacement, 20 sets.
The SAE was trained on Qwen3.5-9B-Base; the detector is Qwen3.5-9B plus a LoRA, so its FVU on the detector's own
residual is reported (`fvu`, and `fvu_no0` without the first token).
"""

import json
import random
import sys
from collections import defaultdict

import numpy as np

import budget
from attribute import _chunks, det
from common import RESULTS, SEED, corpus, save_json, words

LAYER = 16
TOPS = (10, 50)
N_RANDOM = 20


def cmd_attr() -> None:
    C = [r for r in corpus() if r["split"] == "discovery"]
    rng = random.Random(f"{SEED}-sae")
    hum = rng.sample(sorted([r for r in C if r["kind"] == "human"], key=lambda r: r["key"]), 40)
    rows = [r for r in C if r["kind"] == "ai"] + hum
    d = det("9b")
    chunks = _chunks(rows, 20)

    def run():
        return list(d.sae_attr.map([[r["text"] for r in c] for c in chunks], kwargs={"layer": LAYER}))

    res = budget.paid_modal("sae-attr", 6.0, run, containers=4)
    agg: dict = {k: defaultdict(lambda: [0.0, 0.0, 0]) for k in ("ai", "human")}
    fvu: dict = {"ai": [], "human": []}
    for c, rr in zip(chunks, res, strict=True):
        for r, x in zip(c, rr, strict=True):
            fvu[r["kind"]].append([round(x["fvu"], 5), round(x["fvu_no0"], 5)])
            for f, a, m in zip(x["idx"], x["attr"], x["mass"], strict=True):
                s = agg[r["kind"]][f]
                s[0] += a
                s[1] += m
                s[2] += 1
    n = {k: sum(1 for r in rows if r["kind"] == k) for k in agg}
    out = {
        "layer": LAYER,
        "n_texts": n,
        "fvu": fvu,
        "features": {
            k: {str(f): [round(v[0] / n[k], 6), round(v[1] / n[k], 4), v[2]] for f, v in agg[k].items()} for k in agg
        },
    }
    save_json("sae/discovery.json", out, indent=None)
    print({k: len(v) for k, v in out["features"].items()}, "median FVU ai", np.median([a for a, _ in fvu["ai"]]))


def choose() -> dict[str, list[int]]:
    D = json.loads((RESULTS / "sae" / "discovery.json").read_text())
    F = {int(f): v for f, v in D["features"]["ai"].items()}  # f -> [mean attr, mean mass, n texts]
    ranked = sorted(F, key=lambda f: -F[f][0])
    top100 = set(ranked[:100])
    pool = sorted(f for f in F if f not in top100)
    mass = np.array([F[f][1] for f in pool])
    rng = random.Random(f"{SEED}-sae-random")
    sets = {}
    for k in TOPS:
        sets[f"top{k}"] = ranked[:k]
        for j in range(N_RANDOM):
            pick: list[int] = []
            for f in ranked[:k]:
                m = F[f][1]
                ok = [int(i) for i in np.nonzero(np.abs(mass - m) <= 0.2 * m)[0] if pool[i] not in pick]
                if not ok:
                    ok = [int(i) for i in np.argsort(np.abs(mass - m))[:10] if pool[i] not in pick]
                pick.append(pool[rng.choice(ok)])
            sets[f"rand{k}:{j}"] = pick
    return sets


def cmd_ablate() -> None:
    sets = choose()
    rows = [r for r in corpus() if r["kind"] == "ai" and r["split"] == "confirm"]
    d = det("9b")
    chunks = _chunks(rows, 12)

    def run():
        return list(d.sae_ablate.map([[r["text"] for r in c] for c in chunks], kwargs={"layer": LAYER, "sets": sets}))

    res = budget.paid_modal("sae-ablate", 8.0, run, containers=4)
    out: dict = {"layer": LAYER, "sets": sets, "scores": {}}
    for c, rr in zip(chunks, res, strict=True):
        for r, x in zip(c, rr, strict=True):
            out["scores"][r["key"]] = x
    save_json("sae/ablate.json", out, indent=None)


def word_acts(text: str, offsets: list, acts: list[list[float]]) -> np.ndarray:
    """Max activation of each feature on each word unit (tokens mapped by their first non-space character)."""
    spans = words(text)
    starts = np.array([s for s, _ in spans])
    ends = np.array([e for _, e in spans])
    A = np.asarray(acts)  # (F, tokens)
    out = np.zeros((A.shape[0], len(spans)))
    for t, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        seg = text[s:e]
        k = s + (len(seg) - len(seg.lstrip()))
        j = int(np.searchsorted(starts, k, side="right") - 1)
        if j >= 0 and k < ends[j]:
            out[:, j] = np.maximum(out[:, j], A[:, t])
    return out


def cmd_label() -> None:
    D = json.loads((RESULTS / "sae" / "discovery.json").read_text())
    F = D["features"]["ai"]
    top = [int(f) for f in sorted(F, key=lambda f: -F[f][0])[:20]]
    rows = [r for r in corpus() if r["split"] == "discovery"]
    d = det("9b")
    chunks = _chunks(rows, 25)

    def run():
        return list(d.feature_acts.map([[r["text"] for r in c] for c in chunks], kwargs={"layer": LAYER, "feats": top}))

    res = budget.paid_modal("sae-label", 3.0, run, containers=4)
    per: dict = {
        f: {
            "ai_words": defaultdict(float),
            "human_words": defaultdict(float),
            "ai_mass": 0.0,
            "human_mass": 0.0,
            "contexts": [],
        }
        for f in top
    }
    for c, rr in zip(chunks, res, strict=True):
        for r, x in zip(c, rr, strict=True):
            W = word_acts(r["text"], x["offsets"], x["acts"])
            spans = words(r["text"])
            kind = r["kind"]
            for i, f in enumerate(top):
                per[f][f"{kind}_mass"] += float(W[i].sum())
                for j in np.nonzero(W[i])[0]:
                    per[f][f"{kind}_words"][r["text"][spans[j][0] : spans[j][1]].lower()] += float(W[i, j])
                if kind == "ai" and W[i].max() > 0:  # contexts from AI replies only (human passages stay unpublished)
                    j = int(W[i].argmax())
                    a, b = spans[max(0, j - 6)][0], spans[min(len(spans) - 1, j + 4)][1]
                    per[f]["contexts"].append(
                        [
                            round(float(W[i, j]), 3),
                            r["text"][a : spans[j][0]],
                            r["text"][spans[j][0] : spans[j][1]],
                            r["text"][spans[j][1] : b],
                        ]
                    )
    out = []
    for f in top:
        p = per[f]
        out.append(
            {
                "feature": f,
                "attr": F[str(f)][0],
                "n_ai_texts": F[str(f)][2],
                "ai_share": round(p["ai_mass"] / max(1e-9, p["ai_mass"] + p["human_mass"]), 4),
                "top_words": [[w, round(v, 2)] for w, v in sorted(p["ai_words"].items(), key=lambda kv: -kv[1])[:12]],
                "contexts": sorted(p["contexts"], key=lambda c: -c[0])[:6],
            }
        )
    save_json("sae/labels.json", out)


if __name__ == "__main__":
    {"attr": cmd_attr, "ablate": cmd_ablate, "label": cmd_label}[sys.argv[1]]()
