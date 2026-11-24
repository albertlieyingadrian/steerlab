"""Data-driven discovery (Antislop-style) on the train split, validated on the held-out split.

Items: words, word 2-4-grams (within a sentence), and function-word frames (content words -> X, function words and
punctuation kept; 3-5 tokens with at most two X). For each item: AI and human counts and window frequencies on the
train windows, the over-representation ratio ((c_ai + 0.5) / words_ai) / ((c_human + 0.5) / words_human), the number
of sources and models where it fires on AI windows more often than on human ones.

Candidate (pre-registered): ratio >= 4, c_ai >= 20, fires on >= 2% of AI train windows, in >= 4 sources, more often
than human in >= 5 of 8 models, on AI windows from >= 10 distinct prompts (topic-echo guard); not already a v1 word;
an n-gram or frame is dropped when a shorter kept item inside it already fires on >= 90% of its AI windows.
Passes held-out validation: test BP lower bound >= 0.70.

Writes results/candidates.csv (all candidates with train and test numbers) and results/candidates_closed.csv (the
same candidates on the closed-model transfer set; never used to choose).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import LAB, RESULTS, load_corpus
from ruleset import base_words
from stats import rule_table

RATIO_MIN, CA_MIN, DF_MIN, SOURCES_MIN, MODELS_MIN, PROMPTS_MIN, PASS_LO = 4.0, 20, 0.02, 4, 5, 10, 0.70
TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[,;:!?.—–()\"“”]")
SENT_END = {".", "!", "?"}
FUNCTION = set(json.loads((LAB / "packages/deslop/src/deslop/data/function_words.json").read_text())["words"])
PUNCT = set(',;:!?.—–()"“”')


def tokens(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in TOKEN.findall(text)]


def frame_of(toks: list[str]) -> list[str]:
    return [t if (t in FUNCTION or t in PUNCT or t == "'s") else "X" for t in toks]


def items(text: str) -> set[tuple[str, str]]:
    """Every (kind, item) in a window, once."""
    toks = tokens(text)
    out: set[tuple[str, str]] = {("word", t) for t in toks if t[0].isalpha()}
    sent: list[str] = []
    for t in [*toks, "."]:
        if t in SENT_END:
            for n in (2, 3, 4):
                out |= {("ngram", " ".join(sent[i : i + n])) for i in range(len(sent) - n + 1)}
            sent = []
        elif t[0].isalpha():
            sent.append(t)
    frame = frame_of(toks)
    for n in (3, 4, 5):
        for i in range(len(frame) - n + 1):
            f = frame[i : i + n]
            fixed = [x for x in f if x != "X"]
            if f.count("X") <= 2 and len(fixed) >= 2 and not all(x in PUNCT for x in fixed):
                out.add(("frame", " ".join(f)))
    return out


def hay(kind: str, text: str) -> str:
    toks = tokens(text)
    if kind == "frame":
        return " " + " ".join(frame_of(toks)) + " "
    return " " + " ".join(t for t in toks if t[0].isalpha()) + " "


def matches(kind: str, item: str, text: str) -> int:
    return hay(kind, text).count(" " + item + " ")


def main() -> None:
    corpus = load_corpus()
    rows = [r for r in corpus if r["set"] == "matched"]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    closed = [r for r in corpus if r["set"] == "closed"]

    df = {"human": Counter(), "ai": Counter()}
    by_src: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    prompts_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    n_win, n_src, n_model = Counter(), Counter(), Counter()
    for r in train:
        its = items(r["text"])
        df[r["side"]].update(its)
        n_win[r["side"]] += 1
        by_src[(r["side"], r["source"])].update(its)
        n_src[(r["side"], r["source"])] += 1
        if r["side"] == "ai":
            by_model[r["model"]].update(its)
            n_model[r["model"]] += 1
            for it in its:
                prompts_of[it].add(r["pid"])

    pre = {it for it, c in df["ai"].items() if c / n_win["ai"] >= DF_MIN}
    occ = {"human": Counter(), "ai": Counter()}
    words = Counter()
    for r in train:
        hw, hf = hay("word", r["text"]), hay("frame", r["text"])
        words[r["side"]] += len(hw.split())
        for kind, it in pre:
            occ[r["side"]][(kind, it)] += (hf if kind == "frame" else hw).count(" " + it + " ")
    v1_words = set(base_words("v1"))
    sources = sorted({s for _, s in n_src})
    cand = []
    for it in sorted(pre):  # sorted: set order changes with Python's per-process string hashing
        ratio = ((occ["ai"][it] + 0.5) / words["ai"]) / ((occ["human"][it] + 0.5) / words["human"])
        if ratio < RATIO_MIN or occ["ai"][it] < CA_MIN or (it[0] == "word" and it[1] in v1_words):
            continue
        srcs = [
            s
            for s in sources
            if by_src[("ai", s)][it] > 0
            and by_src[("ai", s)][it] / n_src[("ai", s)] > by_src[("human", s)][it] / max(n_src[("human", s)], 1)
        ]
        h_rate = df["human"][it] / n_win["human"]
        models_up = [m for m in by_model if by_model[m][it] / n_model[m] > h_rate]
        if len(srcs) < SOURCES_MIN or len(models_up) < MODELS_MIN or len(prompts_of[it]) < PROMPTS_MIN:
            continue
        cand.append(
            {
                "kind": it[0],
                "item": it[1],
                "c_a": occ["ai"][it],
                "c_h": occ["human"][it],
                "df_a": df["ai"][it] / n_win["ai"],
                "df_h": h_rate,
                "ratio": ratio,
                "n_sources": len(srcs),
                "n_models_up": len(models_up),
                "n_prompts": len(prompts_of[it]),
            }
        )

    # redundancy: drop an item when a shorter kept item of the same kind (or a word inside an n-gram) already fires on
    # >= 90% of its AI train windows
    ai_train = [r for r in train if r["side"] == "ai"]
    fires = {
        (c["kind"], c["item"]): np.array([matches(c["kind"], c["item"], r["text"]) > 0 for r in ai_train]) for c in cand
    }
    final: list[tuple[str, str]] = []
    for k in sorted(fires, key=lambda k: (len(k[1].split()), -fires[k].sum(), k)):
        f = fires[k]
        inside = [
            j for j in final if (j[0] == k[0] or (j[0] == "word" and k[0] == "ngram")) and f" {j[1]} " in f" {k[1]} "
        ]
        if any(fires[j][f].mean() >= 0.9 for j in inside):
            continue
        final.append(k)
    keep = set(final)
    cand = pd.DataFrame([c for c in cand if (c["kind"], c["item"]) in keep]).reset_index(drop=True)

    def bp_table(rs: list[dict]) -> pd.DataFrame:
        cnt = np.array([[matches(c.kind, c.item, r["text"]) for c in cand.itertuples()] for r in rs], dtype=np.int32)
        return rule_table(
            cnt,
            np.array([r["words"] for r in rs]),
            np.array([r["side"] == "ai" for r in rs]),
            np.array([r["group"] for r in rs]),
            [f"{c.kind}:{c.item}" for c in cand.itertuples()],
        )

    cand["train_bp"] = bp_table(train).bp.to_numpy()
    t_test = bp_table(test)
    for col in ["h", "a", "bp", "bp_lo", "bp_hi", "ratio"]:
        cand[f"test_{col}"] = t_test[col].to_numpy()
    cand["passes"] = cand.test_bp_lo >= PASS_LO
    cand = cand.sort_values(["kind", "df_a", "item"], ascending=[True, False, True])
    cand.to_csv(RESULTS / "candidates.csv", index=False, float_format="%.4f")
    t_closed = bp_table([r for r in test if r["side"] == "human"] + closed)
    out = cand[["kind", "item", "passes"]].copy()
    for col in ["h", "a", "bp", "bp_lo", "bp_hi"]:
        out[f"closed_{col}"] = (
            t_closed.set_index("rule").loc[[f"{c.kind}:{c.item}" for c in cand.itertuples()], col].to_numpy()
        )
    out.to_csv(RESULTS / "candidates_closed.csv", index=False, float_format="%.4f")
    print(cand.groupby("kind").passes.agg(["sum", "count"]))
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(
            cand[
                ["kind", "item", "df_a", "df_h", "ratio", "n_sources", "n_models_up", "test_bp", "test_bp_lo", "passes"]
            ].to_string()
        )


if __name__ == "__main__":
    main()
