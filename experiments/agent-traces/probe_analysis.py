"""Fit and test the "tool call failed" direction, its baselines, and the behaviour after an injected error.

    .venv/bin/python experiments/agent-traces/probe_analysis.py      # reads results/interp/, writes probe.json

Direction: difference of means, error minus success, at the decision token, fitted on the eval-split contexts only.
Layer: the highest eval AUROC under 5-fold cross-validation (by task), base model, ties broken by the held-out d'. Test: AUROC on the
test-split contexts for error vs success and for error vs decoy (a success with a harmless pandas warning added).
Baselines beside it: the same direction with shuffled labels (100 shuffles), and a bag of words (logistic regression on
the tool output's word counts, fitted on the same eval contexts). Transfer: the base direction applied to the SFT
model's activations. Behaviour (test contexts): the share of next turns that answer now, repeat the failing command,
or make a new call, per model and condition, with 95% bootstrap intervals; and the ablation run if present.
"""

import difflib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from modal_interp import probe_contexts

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "interp"
ACTIONS = ("answer_now", "repeat", "new_call")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def boot_rate(x: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


FIX = {  # what a next call that deals with each injected error looks like
    "FileNotFoundError": r"\bls\b|find |glob|os\.listdir",
    "ModuleNotFoundError": r"pip install|engine=|import csv|read_csv\((?![^)]*pyarrow)",
    "KeyError": r"\.columns|\.head\(|dtypes|\.keys\(\)|\.info\(",
    "NameError": r"\bdf\s*=",
    "ValueError": r"to_numeric|errors\s*=|coerce|replace\(|dropna|astype|na_values",
    "SyntaxError": None,  # any next command that differs from the failing one
}


def next_command(text: str) -> str:
    m = re.search(r"<parameter=command>\s*(.*?)\s*(?:</parameter>|$)", text, flags=re.S)
    return m.group(1) if m else ""


def targeted(row: dict) -> bool:
    """The next call changes the failing command (similarity < 0.9) and, where the error names a cause, acts on it."""
    nxt = next_command(row["text"])
    if row["action"] != "new_call" or not nxt:
        return False
    if difflib.SequenceMatcher(None, nxt, row["cmd"]).ratio() >= 0.9:
        return False
    pat = FIX[row["kind"]]
    return True if pat is None else bool(re.search(pat, nxt))


def near_repeat(row: dict) -> bool:
    nxt = next_command(row["text"])
    return bool(nxt) and difflib.SequenceMatcher(None, nxt, row["cmd"]).ratio() >= 0.9


def rates(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    out["targeted_fix"] = boot_rate(np.array([targeted(x) for x in rows], dtype=float))
    out["near_repeat"] = boot_rate(np.array([near_repeat(x) for x in rows], dtype=float))
    for a in ACTIONS:
        out[a] = boot_rate(np.array([x["action"] == a for x in rows], dtype=float))
    out["mentions_error"] = boot_rate(np.array([x["mentions_error"] for x in rows], dtype=float))
    return out


def main():
    acts = np.load(OUT / "probe_acts.npz")
    meta = json.loads((OUT / "probe_contexts.json").read_text())
    split = np.array([m["split"] for m in meta])
    ev, te = split == "eval", split == "test"
    A = {k: acts[k] for k in acts.files}  # [N, L, d]
    L = A["base_error"].shape[1]
    res: dict = {"n_eval": int(ev.sum()), "n_test": int(te.sum()), "layers": L}

    # layer choice: 5-fold cross-validation on the eval contexts, base model
    folds = np.array_split(np.random.default_rng(0).permutation(np.where(ev)[0]), 5)
    cv, cv_d = [], []
    for layer in range(L):
        scores = []
        for k in range(5):
            tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
            w = A["base_error"][tr, layer].mean(0) - A["base_success"][tr, layer].mean(0)
            e, su = A["base_error"][folds[k], layer] @ w, A["base_success"][folds[k], layer] @ w
            scores.append((auroc(e, su), (e.mean() - su.mean()) / np.sqrt(0.5 * (e.var() + su.var()))))
        cv.append(float(np.mean([x[0] for x in scores])))
        cv_d.append(float(np.mean([x[1] for x in scores])))
    # AUROC ties at 1.000 from layer 4 on, so the held-out separation (d') breaks the tie
    best = int(np.argmax(np.array(cv) + 1e-3 * np.array(cv_d) / max(cv_d)))
    res["cv_dprime_by_layer"] = cv_d
    res["cv_auroc_by_layer"] = cv
    res["layer"] = best

    def direction(model: str) -> np.ndarray:
        return A[f"{model}_error"][ev, best].mean(0) - A[f"{model}_success"][ev, best].mean(0)

    res["auroc"] = {}
    for model in ("base", "sft"):
        r = {}
        for wname, w in (("own", direction(model)), ("base_direction", direction("base"))):
            e = A[f"{model}_error"][te, best] @ w
            s = A[f"{model}_success"][te, best] @ w
            d = A[f"{model}_decoy"][te, best] @ w
            r[wname] = {"err_vs_succ": auroc(e, s), "err_vs_decoy": auroc(e, d), "decoy_vs_succ": auroc(d, s),
                        "dprime": float((e.mean() - s.mean()) / np.sqrt(0.5 * (e.var() + s.var())))}  # fmt: skip
        pool_e, pool_s = A[f"{model}_error"][ev, best], A[f"{model}_success"][ev, best]
        pool = np.concatenate([pool_e, pool_s])
        shuf = []
        for k in range(100):
            perm = np.random.default_rng(k).permutation(len(pool))
            w = pool[perm[: len(pool_e)]].mean(0) - pool[perm[len(pool_e) :]].mean(0)
            shuf.append(auroc(A[f"{model}_error"][te, best] @ w, A[f"{model}_success"][te, best] @ w))
        r["shuffled_labels_auroc"] = [float(np.mean(shuf)), float(np.percentile(shuf, 2.5)),
                                      float(np.percentile(shuf, 97.5))]  # fmt: skip
        res["auroc"][model] = r
    wb, ws = direction("base"), direction("sft")
    res["cos_base_sft_direction"] = float(wb @ ws / (np.linalg.norm(wb) * np.linalg.norm(ws)))
    res["direction_norm"] = {"base": float(np.linalg.norm(wb)), "sft": float(np.linalg.norm(ws))}
    res["direction"] = ws.tolist()  # the SFT model's own direction, used by the ablation

    # bag of words on the tool output's text
    ctx = probe_contexts("eval", res["n_eval"]) + probe_contexts("test", res["n_test"])
    assert [c["index"] for c in ctx] == [m["index"] for m in meta]
    text = {v: [c["variants"][v][-1]["content"] for c in ctx] for v in ("success", "error", "decoy")}
    vec = CountVectorizer(token_pattern=r"[A-Za-z_]+", lowercase=False, binary=True)
    tr_x = [t for t, k in zip(text["error"], ev) if k] + [t for t, k in zip(text["success"], ev) if k]
    clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(tr_x), [1] * int(ev.sum()) + [0] * int(ev.sum()))

    def bow(v):
        return clf.decision_function(vec.transform([t for t, k in zip(text[v], te) if k]))

    res["bag_of_words"] = {"err_vs_succ": auroc(bow("error"), bow("success")),
                           "err_vs_decoy": auroc(bow("error"), bow("decoy")),
                           "decoy_vs_succ": auroc(bow("decoy"), bow("success"))}  # fmt: skip

    gens = [json.loads(x) for x in (OUT / "gens.jsonl").read_text().splitlines()]
    res["behaviour"] = {f"{m}/{c}": rates([x for x in gens if x["model"] == m and x["condition"] == c])
                        for m in ("base", "sft") for c in ("success", "error", "decoy")}  # fmt: skip
    # paired base vs SFT on the same error contexts
    paired = {}
    for metric, fn in (("targeted_fix", targeted), ("near_repeat", near_repeat)):
        b = {x["index"]: fn(x) for x in gens if x["model"] == "base" and x["condition"] == "error"}
        f = {x["index"]: fn(x) for x in gens if x["model"] == "sft" and x["condition"] == "error"}
        d = np.array([float(f[i]) - float(b[i]) for i in b])
        paired[metric] = boot_rate(d) + [int(sum(f[i] and not b[i] for i in b)), int(sum(b[i] and not f[i] for i in b))]
    res["paired_sft_minus_base_on_error"] = paired
    abl = OUT / "ablate_gens.jsonl"
    if abl.exists():
        ab = [json.loads(x) for x in abl.read_text().splitlines()]
        res["ablation"] = {
            d: rates([x for x in ab if x["direction"] == d]) for d in sorted({x["direction"] for x in ab})
        }
    st = OUT / "steer_gens.jsonl"
    if st.exists():
        sg = [json.loads(x) for x in st.read_text().splitlines()]
        res["steer"] = {f"{d}@{sc:.3f}": rates([x for x in sg if x["direction"] == d and x["scale"] == sc])
                        for d in sorted({x["direction"] for x in sg}) for sc in sorted({x["scale"] for x in sg})}  # fmt: skip
        # at the largest push the replies stop being agent turns; count the ones that talk about an error at all
        words = re.compile(r"错误|error|失败|fail|异常|exception|bug", re.I)
        top = max(x["scale"] for x in sg)
        res["steer_error_words_at_max"] = {d: [sum(bool(words.search(x["text"])) for x in sg if x["direction"] == d and x["scale"] == top),
                                               sum(1 for x in sg if x["direction"] == d and x["scale"] == top)]
                                           for d in sorted({x["direction"] for x in sg})}  # fmt: skip
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("direction", "cv_auroc_by_layer")}, indent=1))
    print("cv auroc by layer:", " ".join(f"{x:.3f}" for x in cv))


if __name__ == "__main__":
    main()
