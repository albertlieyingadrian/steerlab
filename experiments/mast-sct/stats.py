"""Paired comparisons and agreement for the SCT runs, from the scorer's own per-item output.

    python experiments/mast-sct/stats.py

Reads results/scores/<run>_details.jsonl, which MAST's unmodified score.py writes (one line per item with the parsed
`rating`, the expert panel's `expert_modal_rating` and the item's `normalized_score`). Nothing here re-scores a reply.
Adds what the scorer does not report: paired bootstrap intervals for the difference between two runs on the same 174
items, exact rating agreement and Cohen's kappa, and run-to-run flips. Writes results/stats.json and prints a summary.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "results" / "scores"
RAW = HERE / "results" / "raw"
B, SEED = 10_000, 0

RUNS = {
    "gpt-5.5": "GPT-5.5 (MAST's shipped run)",
    "deepseek-v4-flash-0731-think": "DeepSeek-V4-Flash, thinking",
    "deepseek-v4-flash-0731-nothink": "DeepSeek-V4-Flash, no thinking",
    "deepseek-v4-flash-0731-nothink-rep2": "DeepSeek-V4-Flash, no thinking (repeat)",
    "llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
    "qwen3-32b-nothink": "Qwen3-32B, no thinking",
}
REFERENCE = "gpt-5.5"


def load(run: str) -> dict[str, dict]:
    return {d["id"]: d for d in map(json.loads, (SCORES / f"{run}_details.jsonl").open())}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = sum(float((a == k).mean() * (b == k).mean()) for k in range(-2, 3))
    return (po - pe) / (1 - pe)


def main() -> None:
    det = {r: load(r) for r in RUNS}
    ids = sorted(det[REFERENCE], key=int)
    assert all(sorted(d, key=int) == ids for d in det.values()) and len(ids) == 174, "every run must cover all items"
    score = {r: np.array([det[r][i]["normalized_score"] for i in ids]) for r in RUNS}
    rating = {r: np.array([det[r][i]["rating"] for i in ids]) for r in RUNS}
    mode = [det[REFERENCE][i]["expert_modal_rating"] for i in ids]
    unique = np.array([m is not None for m in mode])
    mode_arr = np.array([m if m is not None else 99 for m in mode])
    first_raw = RAW / "deepseek-v4-flash-0731-think.jsonl"
    src = {r["id"]: r["metadata"]["source_short"] for r in map(json.loads, first_raw.open())}
    source = np.array([src[i] for i in ids])

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))

    def ci(x: np.ndarray) -> dict[str, float]:
        if len(x) != len(ids):  # subtests resample within the subtest
            sub = np.random.default_rng(SEED).integers(0, len(x), size=(B, len(x)))
            means = x[sub].mean(axis=1)
        else:
            means = x[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return {"mean": round(float(x.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}

    out: dict[str, object] = {
        "n_items": len(ids),
        "bootstrap": {"B": B, "seed": SEED, "method": "item-level percentile"},
    }
    runs = {}
    for r, label in RUNS.items():
        tokens = None
        raw = RAW / f"{r}.jsonl"
        if raw.exists():
            rows = [json.loads(line) for line in raw.open()]
            tokens = {
                "prompt": sum(x["usage"]["prompt_tokens"] for x in rows),
                "completion": sum(x["usage"]["completion_tokens"] for x in rows),
                "providers": sorted({str(x["usage"].get("upstream_provider")) for x in rows}),
            }
        runs[r] = {
            "label": label,
            "sct_score": ci(score[r]),
            "adelaide": ci(score[r][source == "adelaide"]),
            "open_medical": ci(score[r][source == "open_medical"]),
            "in_expert_set": round(float((score[r] > 0).mean()), 4),
            "matches_expert_mode": round(float((rating[r][unique] == mode_arr[unique]).mean()), 4),
            "rating_counts": {str(k): int((rating[r] == k).sum()) for k in range(-2, 3)},
            "tokens": tokens,
        }
    out["runs"] = runs
    out["n_unique_mode"] = int(unique.sum())
    pairs = {}
    for a, b in combinations(RUNS, 2):
        pairs[f"{a} - {b}"] = {
            "diff": ci(score[a] - score[b]),
            "exact_agreement": round(float((rating[a] == rating[b]).mean()), 4),
            "kappa": round(kappa(rating[a], rating[b]), 4),
            "within_one": round(float((np.abs(rating[a] - rating[b]) <= 1).mean()), 4),
            "items_rating_differs": int((rating[a] != rating[b]).sum()),
        }
    out["pairs"] = pairs
    (HERE / "results" / "stats.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'run':<40} {'sct_score':>9}  95% CI            mode-match")
    for r, v in runs.items():
        s = v["sct_score"]
        print(f"{r:<40} {s['mean']:>9.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]  {v['matches_expert_mode']:.3f}")
    print("\n" + "pair (A - B)".ljust(66) + "diff   paired 95% CI       agree  kappa  n_diff")
    for k, v in pairs.items():
        d = v["diff"]
        print(
            f"{k:<64} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  {v['exact_agreement']:.3f}  {v['kappa']:.3f}"
            f"  {v['items_rating_differs']}"
        )


if __name__ == "__main__":
    main()
