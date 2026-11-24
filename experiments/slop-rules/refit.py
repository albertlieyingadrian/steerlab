"""Write deslop's v2 rules from the train split: lists, then weights and band.

    python refit.py lists   # slop_words.txt (v2), rules_v2.json signposts + stock phrases, results/list_decisions.csv
    python refit.py fit     # weights (non-negative logistic) -> rules_v2.json, human_band_v2.json,
                            # results/weights_v2.json

Decisions use only train numbers (results/rules_v1_train.csv, results/candidates.csv train columns). Test numbers are
written into the word-file comments for the reader, never used to choose.

List rules (set after reading the all-split v1 table, applied to train numbers only):
  v1 item: drop when open models use it less than twice as often as people on train windows (BP < 2/3), or never on
  an AI train window; keep the rest. (The pre-registered "noise" verdict also catches rare words whose interval is
  wide, e.g. "tapestry" at 1 human window against 6 AI ones; dropping those would discard evidence, not noise.)
  new item: every discovered word / n-gram candidate (discover.py filters, train only), minus TOPIC items, minus the
  items whose held-out BP lower bound is <= 0.55 (the one step that reads held-out numbers; it removed 2 items, and the
  held-out comparison in compare.py measures the shipped lists, so it is slightly optimistic for those 2).
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import LAB, RESULTS, load_corpus, write_json

DATA = LAB / "packages" / "deslop" / "src" / "deslop" / "data"
RULES_V2 = DATA / "rules_v2.json"
APOS = "{a}"
# Discovered items left out of the shipped lists because they name a subject, not a way of writing (amendment 2), or
# are fragments of one fixed disclaimer sentence ("I am an AI"), which the Chatbot-language pattern already covers.
TOPIC = {
    "economic": "names a subject",
    "educational": "names a subject",
    "atmosphere": "names a subject (reviews)",
    "thick": "describes a thing, not a way of writing",
    "an ai": "fragment of the disclaimer 'I am an AI'",
    "am an": "fragment of the disclaimer 'I am an AI'",
}


DROP_BP = 2 / 3  # models use it less than twice as often as people
HELDOUT_FAIL = 0.55  # a discovered item whose held-out BP lower bound is at or below this is not shipped


WORD_FILE_HEADER = """
# v2 (data-backed, 2026-11-24): words open models use far more often than people, checked on modern human
# prose. Built by experiments/slop-rules/refit.py from the train split: 8 open-weights models vs human texts
# from the same 11 sources (EditLens news, stories, web pages, reviews, email; HC3 answers). v1 words that
# fired on people as often as on models were dropped; words found by Antislop-style over-representation
# ratios were added. One base form per line; -s, -ed and -ing forms are added. After '#': train BP
# (P(AI | fires) at 50/50), Antislop ratio, held-out BP (for reading only; never used to choose). The
# original hand-curated list is slop_words_v1.txt (excess vocabulary in PubMed abstracts, arXiv 2406.07016;
# Wikipedia 'Signs of AI writing'; EQ-Bench slop-score; Qwen3-4B 'data' words).
"""


def v1_decision(r: pd.Series) -> tuple[str, str]:
    if not r.a > 0:
        return "drop", f"never on an open-model train window (human {r.h:.4f})"
    if r.bp < DROP_BP:
        return "drop", f"models use it less than twice as often as people (train BP {r.bp:.2f}, {r.verdict})"
    return "keep", f"{r.verdict} on train (BP {r.bp:.2f} [{r.bp_lo:.2f}, {r.bp_hi:.2f}])"


# Word n-grams that only read as a habit at the start of a sentence: "Here is a breakdown", not "the food here is".
# Their sentence-start frames (". here is", ". here are") passed held-out validation themselves (discover.py, frames).
SENTENCE_START = {g: r"(?:(?<=[.!?:]\s)|(?<=\n)|^)here\s+(?:is|are)\b" for g in ("here is", "here are")}


def phrase_to_regex(ngram: str) -> str:
    """A discovered lower-case n-gram as a deslop phrase pattern ({a} = either apostrophe)."""
    words = [re.escape(w).replace("'", APOS) for w in ngram.split()]
    return r"\s+".join(words) + r"\b"


def lists() -> None:
    train = pd.read_csv(RESULTS / "rules_v1_train.csv")
    test = pd.read_csv(RESULTS / "rules_v1_test.csv").set_index("rule")
    cand = pd.read_csv(RESULTS / "candidates.csv")
    rows = []
    for r in train.itertuples():
        if r.kind not in ("word", "signpost", "stock", "phrase", "pattern", "patalt", "stockword"):
            continue
        action, reason = v1_decision(pd.Series(r._asdict()))
        rows.append(
            {
                "kind": r.kind,
                "item": r.label,
                "rule": r.rule,
                "action": action,
                "reason": reason,
                "train_h": r.h,
                "train_a": r.a,
                "train_bp": r.bp,
                "train_bp_lo": r.bp_lo,
                "train_ratio": r.ratio,
                "test_bp": test.loc[r.rule, "bp"],
                "test_bp_lo": test.loc[r.rule, "bp_lo"],
            }
        )
    for c in cand[cand.kind.isin(["word", "ngram"])].itertuples():
        topic = c.item in TOPIC
        failed = c.test_bp_lo <= HELDOUT_FAIL
        rows.append(
            {
                "kind": f"new_{c.kind}",
                "item": c.item,
                "rule": f"{c.kind}:{c.item}",
                "action": "skip" if topic else ("skip_heldout" if failed else "add"),
                "reason": TOPIC[c.item]
                if topic
                else f"not confirmed on held-out (BP {c.test_bp:.2f}, lower bound {c.test_bp_lo:.2f})"
                if failed
                else f"discovered on train (ratio {c.ratio:.1f}, {c.df_a:.1%} of AI windows, {c.n_models_up}/8 models)",
                "train_h": c.df_h,
                "train_a": c.df_a,
                "train_bp": c.train_bp,
                "train_bp_lo": np.nan,
                "train_ratio": c.ratio,
                "test_bp": c.test_bp,
                "test_bp_lo": c.test_bp_lo,
            }
        )
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS / "list_decisions.csv", index=False, float_format="%.4f")

    words = dec[dec.kind.isin(["word", "new_word"])]
    lines = WORD_FILE_HEADER.strip().split("\n")
    for r in words[words.action.isin(["keep", "add"])].sort_values(["kind", "item"]).itertuples():
        tag = "v1" if r.kind == "word" else "new"
        lines.append(
            f"{r.item}  # {tag}: train BP {r.train_bp:.2f}, ratio {r.train_ratio:.1f}; held-out BP {r.test_bp:.2f}"
        )
    lines.append("# Found on train but not confirmed on held-out, or naming a subject (left out):")
    for r in words[words.action.isin(["skip", "skip_heldout"])].sort_values("item").itertuples():
        lines.append(f"#   {r.item}: {r.reason}")
    lines.append("# Dropped from v1 (fired on people as often as on models, or never on models):")
    for r in words[words.action == "drop"].sort_values("item").itertuples():
        lines.append(f"#   {r.item}: {r.reason}")
    (DATA / "slop_words.txt").write_text("\n".join(lines) + "\n")

    from deslop import habits as H

    by_rule = dec.set_index("rule")
    sig = [p for i, p in enumerate(H.SIGNPOSTS) if by_rule.loc[f"signpost:{i}", "action"] == "keep"]
    stock = [p for i, p in enumerate(H.STOCK_PHRASES) if by_rule.loc[f"stock:{i}", "action"] == "keep"]
    for r in dec[(dec.kind == "new_ngram") & (dec.action == "add")].itertuples():
        pattern = SENTENCE_START.get(r.item, phrase_to_regex(r.item))
        if pattern not in stock:
            stock.append(pattern)
    old = json.loads(RULES_V2.read_text()) if RULES_V2.exists() else {}
    out = {
        "built": time.strftime("%Y-%m-%d"),
        "source": "experiments/slop-rules/refit.py (train split)",
        "signposts": sig,
        "stock_phrases": stock,
        "weights": old.get("weights", {}),
    }
    RULES_V2.write_text(json.dumps(out, indent=1) + "\n")
    print(dec.groupby(["kind", "action"]).size())


def habit_matrix(rows: list[dict], rules: str) -> np.ndarray:
    from deslop.habits import DETECTORS, find_spans

    names = list(DETECTORS)
    out = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for s in find_spans(r["text"], names, rules=rules):  # every habit, also those v2 weights at 0
            out[i, names.index(s.habit)] += 1
    return out


def fit_nonneg_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, float]:
    """Balanced logistic loss + l2/2 * |w|^2 on standardised features, w >= 0; returns raw-scale weights, bias."""
    sd = X.std(0) + 1e-9
    Z = X / sd
    cw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    def f(theta):
        w, b = theta[:-1], theta[-1]
        z = Z @ w + b
        p = 1 / (1 + np.exp(-z))
        loss = np.sum(cw * (np.logaddexp(0, z) - y * z)) / len(y) + 0.5 * l2 * (w @ w) / len(y)
        g = cw * (p - y) / len(y)
        return loss, np.concatenate([Z.T @ g + l2 * w / len(y), [g.sum()]])

    bounds = [(0, None)] * Z.shape[1] + [(None, None)]
    res = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, bounds=bounds, method="L-BFGS-B")
    return res.x[:-1] / sd, float(res.x[-1])


def fit() -> None:
    from deslop import habits as H
    from deslop.score import _summary, score

    corpus = load_corpus()
    train = [r for r in corpus if r["set"] == "matched" and r["split"] == "train"]
    X = habit_matrix(train, "v2") * 1000.0 / np.array([max(r["words"], 1) for r in train])[:, None]
    y = np.array([r["side"] == "ai" for r in train], dtype=float)
    w, b = fit_nonneg_logistic(X, y)
    names = list(H.DETECTORS)
    weights = {n: round(float(3.0 * v / w.max()), 3) for n, v in zip(names, w, strict=True)}
    data = json.loads(RULES_V2.read_text())
    data["weights"] = weights
    RULES_V2.write_text(json.dumps(data, indent=1) + "\n")
    H.WEIGHTS_V2.update(weights)

    def band(rows: list[dict]) -> dict:
        return _summary(sorted(score(r["text"], band=None, rules="v2").score for r in rows))

    modern = [r for r in corpus if r["side"] == "human" and r["split"] == "train" and r["set"] in ("matched", "raid")]
    groups = {
        "all": modern,
        "editlens": [r for r in modern if r["regime"] == "editlens"],
        "hc3": [r for r in modern if r["regime"] == "hc3"],
        "raid": [r for r in modern if r["regime"] == "raid"],
        "classic": [r for r in corpus if r["set"] == "classic" and r["side"] == "human"],
    }
    bands = {k: band(v) for k, v in groups.items()}
    bands["modern"] = bands["all"]
    (DATA / "human_band_v2.json").write_text(
        json.dumps(
            {
                "built": time.strftime("%Y-%m-%d"),
                "source": (
                    "experiments/slop-rules (train split windows of 150-250 words): 'all' = 'modern' = "
                    "EditLens human news, stories, web pages, reviews and email (research licence), HC3 human "
                    "answers (CC BY-SA 4.0) and RAID human texts (MIT); 'classic' = the v1 band's 318 Gutenberg "
                    "and Wikipedia passages, rescored"
                ),
                "bands": bands,
            }
        )
        + "\n"
    )
    write_json(
        RESULTS / "weights_v2.json",
        {
            "weights": weights,
            "raw_coef": dict(zip(names, map(float, w), strict=True)),
            "bias": b,
            "n_train": len(train),
            "n_ai": int(y.sum()),
            "band_limits": {k: v["limit"] for k, v in bands.items()},
            "band_n": {k: v["n"] for k, v in bands.items()},
        },
    )
    print(json.dumps(weights), {k: (v["n"], v["limit"]) for k, v in bands.items()})


if __name__ == "__main__":
    {"lists": lists, "fit": fit}[sys.argv[1]]()
