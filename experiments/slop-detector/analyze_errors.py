"""Free error analysis for planning data v3 (CPU, saved scores only). Mostly on EditLens VAL; test splits only counted.

    python analyze_errors.py

For our 4B at voi's fixed cut-offs, our calibrated blend (EditLens-val cut-offs), and voi-slop-detect-4B:
- 3-way errors on EditLens val by source, generator, length and true class, and the confusion matrix;
- AI-edited texts by how much was edited (EditLens's cosine_score, binned): where the misses sit;
- truncation: how many texts exceed 512 tokens (Qwen3.5 tokenizer) in val and test, and val error rate above/below.
Writes results/error_analysis.json (counts and rates, no text).
"""

from __future__ import annotations

import json
from importlib import import_module

import numpy as np
from common import BANDS, RESULTS, editlens_split, empty_fences

tv = import_module("tune_editlens_val")
tb = import_module("tune_bands")


def rate(mask: np.ndarray, err: np.ndarray) -> dict:
    n = int(mask.sum())
    return {"n": n, "error_rate": round(float(err[mask].mean()), 4) if n else None}


def main() -> None:
    from transformers import AutoTokenizer

    val, _ = tv.clean_val()
    y = val["y"].to_numpy()
    s4, s17 = tv.load("v2-4b", "editlens_val"), tv.load("v2-1.7b", "editlens_val")
    voi = np.array(json.loads((RESULTS / "voi" / "editlens_val.json").read_text())["scores"])
    c = json.loads((RESULTS / "tuned_editlens_val_choice.json").read_text())
    fixed = (BANDS["human_max"], BANDS["generated_min"])
    preds = {
        "4b_fixed": tb.bands(s4, *fixed),
        "blend_calibrated": tb.bands(c["w4b"] * s4 + (1 - c["w4b"]) * s17, c["human_max"], c["generated_min"]),
        "voi_fixed": tb.bands(voi, *fixed),
    }
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    ntok = np.array([len(x) for x in tok([empty_fences(str(t)) for t in val["text"]])["input_ids"]])
    words = val["text"].astype(str).str.split().str.len().to_numpy()
    out: dict = {"n_val": len(y), "models": {}}
    for name, p in preds.items():
        err = p != y
        m = {
            "macro_f1": round(tb.macro_f1(y, p), 4),
            "confusion_rows_true_cols_pred": [[int(((y == a) & (p == b)).sum()) for b in (0, 1, 2)] for a in (0, 1, 2)],
            "by_source": {s: rate((val["source"] == s).to_numpy(), err) for s in sorted(val["source"].unique())},
            "by_generator": {g: rate((val["model"] == g).to_numpy(), err) for g in sorted(val["model"].unique())},
            "by_class": {k: rate(y == i, err) for i, k in enumerate(("human", "edited", "generated"))},
            "by_words": {
                f"{a}-{b}": rate((words >= a) & (words < b), err)
                for a, b in ((0, 100), (100, 200), (200, 350), (350, 10**6))
            },
            "by_truncation": {
                "<=512 tokens": rate(ntok <= 512, err),
                ">512 tokens": rate(ntok > 512, err),
            },
        }
        edited = y == 1
        cos = val["cosine_score"].to_numpy()
        bins = ((0, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 0.3), (0.3, 2))
        m["edited_by_edit_size"] = {}
        for a, b in bins:
            sel = edited & (cos >= a) & (cos < b)
            m["edited_by_edit_size"][f"cosine {a}-{b}"] = {
                **rate(sel, err),
                "called_human": int((sel & (p == 0)).sum()),
                "called_generated": int((sel & (p == 2)).sum()),
            }
        out["models"][name] = m
    trunc = {
        "val": {
            "n": len(ntok),
            "over_512": int((ntok > 512).sum()),
            "median_tokens": int(np.median(ntok)),
        }
    }
    for split in ("test", "test_enron", "test_llama"):
        t = editlens_split(split)
        n = np.array([len(x) for x in tok([empty_fences(str(x)) for x in t["text"]])["input_ids"]])
        trunc[split] = {
            "n": len(n),
            "over_512": int((n > 512).sum()),
            "median_tokens": int(np.median(n)),
        }
    out["truncation"] = trunc
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, m in out["models"].items():
        print(f"\n== {name}: EditLens val macro-F1 {m['macro_f1']}; confusion {m['confusion_rows_true_cols_pred']}")
        print("  by source   ", {k: v["error_rate"] for k, v in m["by_source"].items()})
        print(
            "  by generator",
            {k.split("/")[-1][:18]: v["error_rate"] for k, v in m["by_generator"].items()},
        )
        print(
            "  by words    ",
            {k: (v["n"], v["error_rate"]) for k, v in m["by_words"].items()},
        )
        print(
            "  truncation  ",
            {k: (v["n"], v["error_rate"]) for k, v in m["by_truncation"].items()},
        )
        sizes = {k: (v["n"], v["error_rate"], v["called_human"], v["called_generated"])
                 for k, v in m["edited_by_edit_size"].items()}  # fmt: skip
        print("  edited by edit size (n, error, ->human, ->generated)", sizes)
    print("\ntruncation", trunc)


if __name__ == "__main__":
    main()
