# ruff: noqa: E501
"""B2: window scoring for long texts (pre-registered in docs/plans/2026-11-12-beat-voi-levers.md). Validation only.

    python windows.py make    # cut EditLens val texts over 512 tokens into windows: data/windows_val.json (local)
    python windows.py score   # v2-4b on every window (Modal score_run): runs/v2-4b/scores/editlens_val_windows.json
    python windows.py eval    # the four pre-registered candidates, cross-fitted cut-offs: results/b2_windows.json

Windows for texts over 512 tokens (Qwen3.5 tokenizer): the first 512 tokens (= the existing score), the last 512, and
the middle 512 when the text is over 1,024 tokens. Candidates: (0) first window only, the baseline; (1) mean of the
windows; (2) max of the windows; (3) 0.5 x first + 0.5 x mean of the others. Texts of 512 tokens or fewer keep their
score. Adopt the best of (1)-(3) only if its paired macro-F1 difference against (0) has a 95% interval above zero.
No test row is read. Test looks stay at 4.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module

import numpy as np
from common import DATA, HERE, RESULTS, empty_fences, words

tev = import_module("tune_editlens_val")
diag = import_module("diag_voi_gap")
WIN = DATA / "windows_val.json"  # windowed EditLens text stays local (CC BY-NC-SA)
SCORES = HERE / "runs" / "v2-4b" / "scores" / "editlens_val_windows.json"


def make() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    val, _ = tev.clean_val()
    out = []
    for i, t in enumerate(val["text"]):
        ids = tok(empty_fences(str(t)), add_special_tokens=False)["input_ids"]
        if len(ids) <= 512:
            continue
        out.append({"row": i, "w": "last", "text": tok.decode(ids[-512:])})
        if len(ids) > 1024:
            mid = (len(ids) - 512) // 2
            out.append({"row": i, "w": "middle", "text": tok.decode(ids[mid : mid + 512])})
    WIN.write_text(json.dumps(out) + "\n")
    print(f"{len(out)} windows for {len({w['row'] for w in out})} long texts")


def score() -> None:
    import modal

    wins = json.loads(WIN.read_text())
    fn = modal.Function.from_name("slop-detector", "score_run")
    s = fn.remote("v2-4b", {"w": [w["text"] for w in wins]})["w"]
    SCORES.write_text(json.dumps(s) + "\n")
    print(f"scored {len(s)} windows")


def crossfit(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions: cut-offs fitted on the other four folds (grouped by source document)."""
    f = diag.folds(groups)
    p = np.zeros(len(y), dtype=int)
    for k in range(5):
        _, h, g = diag.fit_cuts(y[f != k], s[f != k])
        p[f == k] = diag.bands(s[f == k], h, g)
    return p


def evaluate() -> None:
    val, _ = tev.clean_val()
    y = val["y"].to_numpy()
    groups = val["source_id"].fillna(val["text_id"]).to_numpy()
    first = tev.load("v2-4b", "editlens_val")
    wins = json.loads(WIN.read_text())
    ws = json.loads(SCORES.read_text())
    others: dict[int, list[float]] = {}
    for w, sc in zip(wins, ws, strict=True):
        others.setdefault(w["row"], []).append(sc)
    cands = {
        "(0) first": first.copy(),
        "(1) mean": first.copy(),
        "(2) max": first.copy(),
        "(3) half first": first.copy(),
    }
    for r, o in others.items():
        allw = [first[r], *o]
        cands["(1) mean"][r] = np.mean(allw)
        cands["(2) max"][r] = np.max(allw)
        cands["(3) half first"][r] = 0.5 * first[r] + 0.5 * np.mean(o)
    long_ = np.zeros(len(y), bool)
    long_[list(others)] = True
    nw = np.array([len(words(str(t))) for t in val["text"]])
    preds = {k: crossfit(y, v, groups) for k, v in cands.items()}
    base = preds["(0) first"]
    out = {"long_texts": int(long_.sum()), "windows": len(ws), "candidates": {}}
    for k, p in preds.items():
        e400, h400 = (y == 1) & (nw >= 400), (y == 0) & (nw >= 400)
        out["candidates"][k] = {
            "macro_f1": round(diag.mf1(y, p), 4),
            "diff_vs_first": round(diag.mf1(y, p) - diag.mf1(y, base), 4),
            "ci_vs_first": diag.boot(y, p, base),
            "edited_400plus_error": round(float((p[e400] != 1).mean()), 4),
            "humans_400plus_called_edited": round(float((p[h400] == 1).mean()), 4),
            "long_texts_error": round(float((p[long_] != y[long_]).mean()), 4),
        }
        c = out["candidates"][k]
        print(f"{k:15} macro-F1 {c['macro_f1']:.4f}  vs first {c['diff_vs_first']:+.4f} {c['ci_vs_first']}  "
              f"edited 400+ err {c['edited_400plus_error']:.3f}  humans 400+ -> edited {c['humans_400plus_called_edited']:.3f}  "
              f"long-text err {c['long_texts_error']:.3f}")  # fmt: skip
    passing = [k for k in list(preds)[1:] if out["candidates"][k]["ci_vs_first"][0] > 0]
    best = max(passing, key=lambda k: out["candidates"][k]["macro_f1"]) if passing else None
    out["decision"] = best or "keep first window (no candidate's interval above zero)"
    (RESULTS / "b2_windows.json").write_text(json.dumps(out, indent=1) + "\n")
    print("decision:", out["decision"])


if __name__ == "__main__":
    {"make": make, "score": score, "eval": evaluate}[sys.argv[1]]()
