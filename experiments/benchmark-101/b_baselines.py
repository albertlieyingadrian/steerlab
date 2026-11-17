"""Part B, step B4: trivial baselines on SlopPair, on the original pairs and on the hill-climbed test pairs.

    python b_baselines.py        # after b_build.py and b_judge.py; voi-380M runs locally (MPS or CPU)

  length    the longer passage is the person's
  lexicon   fewer AI-tic hits (bench.AI_TICS) plus em-dashes per 100 words is the person's
  bow       logistic regression on the difference of word counts (A minus B), 5-fold cross-validated over all 300
            pairs; for the hill-climbed pairs, trained on all 300 original pairs
  voi380    voi-slop-detect-380M: the lower score is the person's

A tie counts as half right. Writes results/b_baselines.json: per baseline, per test pair, 1 / 0.5 / 0 for the
original and the hill-climbed pairs, plus the cross-validated accuracy over all 300.
"""

from __future__ import annotations

import json
import re

import bench
import numpy as np
from b_build import OUT

VOI, VOI_REV = "VoightKampffLabs/voi-slop-detect-380M", "a109d591b8b62541f362d873765900380b2bd0ea"


def tics(text: str) -> float:
    t = text.lower()
    n = sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) for w in bench.AI_TICS) + text.count("—")
    return 100 * n / max(len(text.split()), 1)


def pick(h: float, a: float, lower_is_human: bool) -> float:
    if h == a:
        return 0.5
    return float((h < a) == lower_is_human)


def voi_scores(texts: list[str]) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(VOI, revision=VOI_REV)
    net = AutoModelForSequenceClassification.from_pretrained(VOI, revision=VOI_REV, dtype=torch.float32).to(dev).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok([t], truncation=True, max_length=512, return_tensors="pt").to(dev)
            out.append(float(torch.sigmoid(net(**enc).logits.float()).reshape(-1)[0]))
    return out


def bow(pairs: list[dict], hill: dict[str, str]) -> tuple[dict[str, float], dict[str, float], dict]:
    """(cross-validated right/wrong per pair id over all 300, right/wrong per hill-climbed test pair, top words)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(0)
    flip = rng.random(len(pairs)) < 0.5  # True: A is the person's passage
    a_txt = [p["human"] if f else p["ai"] for p, f in zip(pairs, flip, strict=True)]
    b_txt = [p["ai"] if f else p["human"] for p, f in zip(pairs, flip, strict=True)]
    vec = CountVectorizer(lowercase=True, min_df=2).fit(a_txt + b_txt)
    X = (vec.transform(a_txt) - vec.transform(b_txt)).toarray().astype(float)
    y = flip.astype(int)
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        pred[te] = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr]).predict(X[te])
    cv = {p["id"]: float(v) for p, v in zip(pairs, pred == y, strict=True)}
    full = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    coefs = sorted(zip(full.coef_[0], vec.get_feature_names_out(), strict=True))
    top = {
        "ai_words": {w: round(float(c), 3) for c, w in coefs[:12]},
        "human_words": {w: round(float(c), 3) for c, w in coefs[-12:]},
    }
    test = [p for p in pairs if p["split"] == "test"]
    Xh = (
        (vec.transform([p["human"] for p in test]) - vec.transform([hill[p["id"]] for p in test]))
        .toarray()
        .astype(float)
    )
    hill_ok = {p["id"]: float(v == 1) for p, v in zip(test, full.predict(Xh), strict=True)}  # A is the person's here
    return cv, hill_ok, top


def main() -> None:
    pairs = [json.loads(line) for line in OUT.read_text().splitlines()]
    hill = {r["id"]: r["hill"] for r in json.loads((bench.RESULTS / "sloppair_hill.json").read_text())["items"]}
    test = [p for p in pairs if p["split"] == "test"]
    out: dict = {"canary": bench.CANARY, "test_ids": [p["id"] for p in test], "orig": {}, "hill": {}}
    for name, f in (("length", lambda t: -len(t.split())), ("lexicon", tics)):
        out["orig"][name] = [pick(f(p["human"]), f(p["ai"]), True) for p in test]
        out["hill"][name] = [pick(f(p["human"]), f(hill[p["id"]]), True) for p in test]
    cv, hill_bow, top = bow(pairs, hill)
    out["bow_top_words"] = top
    out["bow_cv_all300"] = round(float(np.mean(list(cv.values()))), 4)
    out["orig"]["bow"] = [cv[p["id"]] for p in test]
    out["hill"]["bow"] = [hill_bow[p["id"]] for p in test]
    texts = [p["human"] for p in test] + [p["ai"] for p in test] + [hill[p["id"]] for p in test]
    s = voi_scores(texts)
    n = len(test)
    sh, sa, shl = s[:n], s[n : 2 * n], s[2 * n :]
    out["orig"]["voi380"] = [pick(h, a, True) for h, a in zip(sh, sa, strict=True)]
    out["hill"]["voi380"] = [pick(h, a, True) for h, a in zip(sh, shl, strict=True)]
    out["voi380_scores"] = {"human": sh, "ai": sa, "hill": shl}
    (bench.RESULTS / "b_baselines.json").write_text(json.dumps(out) + "\n")
    for name in out["orig"]:
        print(f"{name:8} orig {np.mean(out['orig'][name]):.3f}  hill {np.mean(out['hill'][name]):.3f}")
    print("bow cv over all 300:", out["bow_cv_all300"])


if __name__ == "__main__":
    main()
