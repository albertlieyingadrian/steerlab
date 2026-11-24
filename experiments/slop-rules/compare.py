"""Old vs new score on held-out windows: AUROC, false-positive rate on human text at each band limit, per source and
model, the closed-model / RAID / classic transfer sets, ablations (lists vs weights), drop-one habits, and two cheap
baselines (window length; a unigram TF-IDF logistic regression trained on the train windows).

Writes results/compare.json, compare_by_source.csv, compare_by_model.csv and window_scores.csv (scores only, no text).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS, load_corpus, write_json
from refit import habit_matrix
from stats import auroc_ci, rate_ci

V1_LIMIT = 33.63  # the v1 band's p90 (318 classic passages)


def main() -> None:
    from deslop import habits as H
    from deslop.score import human_band, score

    rows = load_corpus()
    words = np.array([max(r["words"], 1) for r in rows], dtype=float)
    names = list(H.DETECTORS)
    C = {v: habit_matrix(rows, v) for v in ("v1", "v2")}

    def sc(lists: str, weights: dict[str, float]) -> np.ndarray:
        return 1000.0 * (C[lists] @ np.array([weights[n] for n in names])) / words

    configs = {
        "v1": sc("v1", H.WEIGHTS),
        "v1 lists + v2 weights": sc("v1", H.WEIGHTS_V2),
        "v2 lists + v1 weights": sc("v2", H.WEIGHTS),
        "v2": sc("v2", H.WEIGHTS_V2),
    }
    for i in (0, len(rows) // 2, len(rows) - 1):  # the vectorised score equals deslop.score
        assert abs(configs["v1"][i] - score(rows[i]["text"], band=None, rules="v1").score) < 1e-6
        assert abs(configs["v2"][i] - score(rows[i]["text"], band=None, rules="v2").score) < 1e-6

    def get(k):
        return np.array([r[k] for r in rows])

    side, st, split, model, source, group = (get(k) for k in ("side", "set", "split", "model", "source", "group"))
    test = split == "test"
    hum_t = (side == "human") & test & (st == "matched")
    modern_h_t = (side == "human") & test & np.isin(st, ["matched", "raid"])
    modern_h_tr = (side == "human") & ~test & np.isin(st, ["matched", "raid"])
    ai_open_t = (side == "ai") & test & (st == "matched")
    ai_closed = (side == "ai") & (st == "closed")
    v2_limit = float(human_band("v2")["all"]["limit"])
    v1_modern_limit = float(np.quantile(configs["v1"][modern_h_tr], 0.9))

    def au(s: np.ndarray, hm: np.ndarray, am: np.ndarray) -> list[float]:
        m = hm | am
        return [round(x, 4) for x in auroc_ci(s[m], am[m], group[m])]

    def rate(s: np.ndarray, limit: float, mask: np.ndarray) -> list[float]:
        return [round(x, 4) for x in rate_ci(s[mask] > limit, group[mask])]

    out: dict = {
        "n": {
            "human_test_matched": int(hum_t.sum()),
            "human_test_modern": int(modern_h_t.sum()),
            "ai_open_test": int(ai_open_t.sum()),
            "ai_closed": int(ai_closed.sum()),
        },
        "limits": {"v1": V1_LIMIT, "v1_refit_band": round(v1_modern_limit, 3), "v2": v2_limit},
    }
    raid_h, raid_a = (side == "human") & test & (st == "raid"), (side == "ai") & test & (st == "raid")
    cl_h, cl_a = (side == "human") & (st == "classic"), (side == "ai") & (st == "classic")
    out["auroc"] = {
        name: {
            "open_test": au(s, hum_t, ai_open_t),
            "closed": au(s, hum_t, ai_closed),
            "raid_test": au(s, raid_h, raid_a),
            "classic": au(s, cl_h, cl_a),
        }
        for name, s in configs.items()
    }
    out["at_limit"] = {
        label: {
            "fpr_modern_human_test": rate(configs[cfg], lim, modern_h_t),
            "tpr_open_test": rate(configs[cfg], lim, ai_open_t),
            "tpr_closed": rate(configs[cfg], lim, ai_closed),
        }
        for label, cfg, lim in [
            ("v1 @ 33.63", "v1", V1_LIMIT),
            ("v1 @ modern band", "v1", v1_modern_limit),
            ("v2 @ v2 band", "v2", v2_limit),
        ]
    }

    base = au(configs["v2"], hum_t, ai_open_t)[0]
    drop = {}
    for n in names:
        if H.WEIGHTS_V2[n]:
            w = {**H.WEIGHTS_V2, n: 0.0}
            drop[n] = round(au(sc("v2", w), hum_t, ai_open_t)[0] - base, 4)
    out["drop_one_v2"] = drop

    out["baselines"] = {"length": {"open_test": au(words, hum_t, ai_open_t), "closed": au(words, hum_t, ai_closed)}}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tr = (st == "matched") & ~test
    vec = TfidfVectorizer(min_df=3, max_features=20000, sublinear_tf=True)
    Xtr = vec.fit_transform([r["text"] for r, k in zip(rows, tr, strict=True) if k])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr, side[tr] == "ai")
    bow = clf.decision_function(vec.transform([r["text"] for r in rows]))
    out["baselines"]["bag_of_words"] = {"open_test": au(bow, hum_t, ai_open_t), "closed": au(bow, hum_t, ai_closed)}

    per_src, per_model = [], []
    for s in sorted(set(source[modern_h_t])):
        hm, am, mh = hum_t & (source == s), ai_open_t & (source == s), modern_h_t & (source == s)
        has_ai = bool(am.any())
        per_src.append(
            {
                "source": s,
                "n_human": int(mh.sum()),
                "n_ai": int(am.sum()),
                "auroc_v1": au(configs["v1"], hm, am)[0] if has_ai else np.nan,
                "auroc_v2": au(configs["v2"], hm, am)[0] if has_ai else np.nan,
                "fpr_v1": float((configs["v1"][mh] > V1_LIMIT).mean()),
                "fpr_v2": float((configs["v2"][mh] > v2_limit).mean()),
            }
        )
    for m in sorted(set(model[ai_open_t | ai_closed])):
        am = (ai_open_t | ai_closed) & (model == m)
        per_model.append(
            {
                "model": m,
                "set": "closed" if (ai_closed & (model == m)).any() else "open",
                "n": int(am.sum()),
                "auroc_v1": au(configs["v1"], hum_t, am)[0],
                "auroc_v2": au(configs["v2"], hum_t, am)[0],
                "tpr_v1": float((configs["v1"][am] > V1_LIMIT).mean()),
                "tpr_v2": float((configs["v2"][am] > v2_limit).mean()),
            }
        )
    pd.DataFrame(per_src).to_csv(RESULTS / "compare_by_source.csv", index=False, float_format="%.4f")
    pd.DataFrame(per_model).to_csv(RESULTS / "compare_by_model.csv", index=False, float_format="%.4f")
    cols = {f"score_{k.replace(' + ', '_plus_').replace(' ', '_')}": v for k, v in configs.items()}
    pd.DataFrame(
        {
            "id": get("id"),
            "set": st,
            "side": side,
            "split": split,
            "model": model,
            "source": source,
            "words": words.astype(int),
            **cols,
        }
    ).to_csv(RESULTS / "window_scores.csv", index=False, float_format="%.3f")
    write_json(RESULTS / "compare.json", out)
    print(json.dumps({k: out[k] for k in ("limits", "auroc", "at_limit", "drop_one_v2", "baselines")}, indent=1))
    print(pd.DataFrame(per_src).to_string(), "\n", pd.DataFrame(per_model).to_string())


if __name__ == "__main__":
    main()
