"""Every number the pilot's execution log cites, from the cached results (no Modal calls).

python pilot_summary.py > results/pilot_summary.txt
"""

import json

import numpy as np

from common import RESULTS, TAG_ORDER, geneval_summary, read_jsonl
from compare import paired_diff

PRIMARY = "plan | indep | N=4 + retry"  # chosen on validation before the test run


def boot_mean_diff(a, b, n_boot=4000, seed=0):
    """Paired mean of b - a with a bootstrap interval (a, b aligned)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), *np.percentile(bs, [2.5, 97.5]).tolist()


def fmt(s):
    return "  ".join(f"{t[:6]} {s[t][0]:.3f}" for t in [*TAG_ORDER, "overall"] if t in s) + (
        f"  [{s['overall'][1]:.3f}, {s['overall'][2]:.3f}]"
    )


def fmt_d(d):
    return "  ".join(f"{t[:6]} {d[t][0]:+.3f}" for t in [*TAG_ORDER, "overall"] if t in d) + (
        f"  [{d['overall'][1]:+.3f}, {d['overall'][2]:+.3f}]"
    )


def harness_rows(run, config):
    pool = {r["path"]: r for r in read_jsonl(RESULTS / run / "pool.jsonl")}
    table = json.loads((RESULTS / run / "ablation.json").read_text())
    row = next(t for t in table if t["config"] == config)
    return [pool[p] for p in row["paths"]], row


def agreement(pool):
    return float(np.mean([r["correct"] == r["indep_pass"] for r in pool])), len(pool)


def compbench(run_a, run_b):
    a, b = read_jsonl(RESULTS / run_a / "rows.jsonl"), read_jsonl(RESULTS / run_b / "rows.jsonl")
    out = {}
    for cat in ("color", "shape", "texture"):
        xa = [r["bvqa"] for r in a if r["tag"] == f"cb_{cat}"]
        xb = [r["bvqa"] for r in b if r["tag"] == f"cb_{cat}"]
        out[cat] = (float(np.mean(xa)), float(np.mean(xb)), *boot_mean_diff(xa, xb))
    pa, pb = [r["pick"] for r in a], [r["pick"] for r in b]
    out["pickscore"] = (float(np.mean(pa)), float(np.mean(pb)), *boot_mean_diff(pa, pb))
    return out


def print_cb(name, cb):
    print(f"\n{name}")
    for k, (ma, mb, d, lo, hi) in cb.items():
        print(f"  {k:<9} {ma:.3f} -> {mb:.3f}  diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main():
    print("== Harness, validation (200 prompts x 2 reps; GenEval tags minus single_object)")
    val_pool = read_jsonl(RESULTS / "harness_val/pool.jsonl")
    base_rows, _ = harness_rows("harness_val", "orig | single | N=1")
    for cfg in ["orig | single | N=1", "plan | single | N=1", "orig | indep | N=4", "plan | indep | N=4", PRIMARY,
                "plan | indep | N=8", "plan | oracle | N=4"]:  # fmt: skip
        rows, row = harness_rows("harness_val", cfg)
        print(
            f"{cfg:<30} {fmt(row['geneval'])}  pick {row['pickscore']:.2f}  cand/out {row['candidates_per_output']:.2f}"
        )
        if cfg != "orig | single | N=1":
            print(f"{'  paired vs orig single':<30} {fmt_d(paired_diff(base_rows, rows))}")
    agree, n = agreement(val_pool)
    print(f"independent checker vs official grader, validation pool: agree {agree:.3f} (n={n})")

    if (RESULTS / "harness_official/ablation.json").exists():
        print("\n== Harness, official GenEval prompts (553 prompts x reps)")
        base = read_jsonl(RESULTS / "schnell_official/rows.jsonl")
        print(f"{'base FLUX.1-schnell, 4 samples':<30} {fmt(geneval_summary(base))}")
        off_pool = read_jsonl(RESULTS / "harness_official/pool.jsonl")
        for row in json.loads((RESULTS / "harness_official/ablation.json").read_text()):
            rows, _ = harness_rows("harness_official", row["config"])
            print(f"{row['config']:<30} {fmt(row['geneval'])}  pick {row['pickscore']:.2f}  cand/out "
                  f"{row['candidates_per_output']:.2f}")  # fmt: skip
            if row["config"] in (PRIMARY, "plan | single | N=1"):
                print(f"{'  paired vs base':<30} {fmt_d(paired_diff(base, rows))}")
        agree, n = agreement(off_pool)
        print(f"independent checker vs official grader, official pool: agree {agree:.3f} (n={n})")
        for f in sorted((RESULTS / "harness_official").glob("cost_*.json")):
            print("cost", f.name, json.loads(f.read_text()))

    if (RESULTS / "cb_plan1024/rows.jsonl").exists():
        print_cb("== Guard: T2I-CompBench B-VQA (300 prompts, 1 image), planner vs original prompt, 1024 px",
                 compbench("cb_base1024", "cb_plan1024"))  # fmt: skip

    for name in ("genai_bench/judge_check.json", "genai_bench/summary.json", "blind/summary.json"):
        if (RESULTS / name).exists():
            d = json.loads((RESULTS / name).read_text())
            d.pop("by_tag", None)
            print(f"\n== {name}\n{json.dumps(d, indent=1)}")
    if (RESULTS / "latency.json").exists():
        lat = json.loads((RESULTS / "latency.json").read_text())
        print("\n== latency (s):", {k: round(v["median"], 2) for k, v in lat.items() if isinstance(v, dict)},
              "retries", lat["retries"], "of", lat["n"])  # fmt: skip


if __name__ == "__main__":
    main()
