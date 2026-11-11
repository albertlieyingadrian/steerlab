"""Paired difference of two GenEval runs on the same prompts (B minus A), with a bootstrap over prompts.

  python compare.py results/schnell_official/rows.jsonl results/harness_official/test_rows.jsonl

Each file holds one row per image with "tag", "idx" and "correct". Runs may have a different number of images per
prompt; each prompt contributes its pass rate. Overall = mean over tags, as in GenEval.
"""

import argparse

import numpy as np

from common import TAG_ORDER, read_jsonl


def per_prompt(rows, key="correct"):
    out = {}
    for r in rows:
        out.setdefault((r["tag"], r["idx"]), []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def paired_diff(rows_a, rows_b, key_a="correct", key_b="correct", n_boot=4000, seed=0):
    """{tag: (diff, lo, hi, n_prompts)} plus "overall"; only prompts present in both runs count."""
    a, b = per_prompt(rows_a, key_a), per_prompt(rows_b, key_b)
    common = sorted(set(a) & set(b))
    tags = [t for t in TAG_ORDER if any(k[0] == t for k in common)]
    rng = np.random.default_rng(seed)
    out, boots = {}, np.zeros((n_boot, len(tags)))
    for j, t in enumerate(tags):
        d = np.array([b[k] - a[k] for k in common if k[0] == t])
        pick = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots[:, j] = d[pick].mean(1)
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        out[t] = (float(d.mean()), float(lo), float(hi), len(d))
    lo, hi = np.percentile(boots.mean(1), [2.5, 97.5])
    out["overall"] = (float(np.mean([out[t][0] for t in tags])), float(lo), float(hi), len(common))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key-a", default="correct")
    ap.add_argument("--key-b", default="correct")
    args = ap.parse_args()
    d = paired_diff(read_jsonl(args.a), read_jsonl(args.b), args.key_a, args.key_b)
    for t, (m, lo, hi, n) in d.items():
        print(f"{t:<14} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  prompts={n}")


if __name__ == "__main__":
    main()
