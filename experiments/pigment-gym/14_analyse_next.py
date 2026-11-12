"""
The follow-up study's numbers: true quality per arm, the pre-registered paired comparisons, and the reward's agreement
with the judges on new subjects.

Run from the repo root after 13_judge_next.py (laptop only, seconds):
    .venv/bin/python experiments/pigment-gym/14_analyse_next.py

Tests fixed in the pre-registration (the note's "How the study was run"). Units are (subject, group) pairs shared by
every arm: 64 on the held-out subjects. Intervals: paired bootstrap over the 64 units (10,000 resamples), and beside
it a cluster bootstrap over the 8 subjects. Writes results/next/summary.json.
"""

import csv
import json
import math

import numpy as np
from analysis_tools import signature
from brush_tools import PROMPT_EXAMPLE, extract_code
from common import RESULTS, check
from scipy.stats import spearmanr

OUT = RESULTS / "next"
ARMS = ["base", "bon8", "grpo", "sft", "sft_bon8", "sft_grpo", "opus"]  # opus: reference ceiling
TESTS = {  # name: (arm a, arm b, sided) for "a minus b"
    "H1 sft - grpo": ("sft", "grpo", "greater"),
    "H2 bon8 - base": ("bon8", "base", "greater"),
    "H3 grpo - base": ("grpo", "base", "two-sided"),
    "H4 sft - bon8": ("sft", "bon8", "two-sided"),
    "sft - base": ("sft", "base", "exploratory"),
    "sft_bon8 - sft": ("sft_bon8", "sft", "exploratory"),
    "sft_grpo - sft": ("sft_grpo", "sft", "exploratory"),
    "sft_bon8 - grpo": ("sft_bon8", "grpo", "exploratory"),
    "sft_grpo - grpo": ("sft_grpo", "grpo", "exploratory"),
    "opus - sft_bon8": ("opus", "sft_bon8", "reference"),
}
RNG = np.random.default_rng(0)
SOURCES = {
    "base": "base",
    "bon8": "base",
    "grpo": "grpo",
    "sft": "sft",
    "sft_bon8": "sft",
    "sft_grpo": "sft_grpo",
    "opus": "opus",
}
EXAMPLE_BLOCK = signature(extract_code(PROMPT_EXAMPLE))  # the prompt's example as brush calls, as in 9_hack_hunt.py


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap(diffs, clusters, reps=10_000):
    """95% intervals for the mean difference: resampling units, and resampling whole subjects."""
    diffs, clusters = np.asarray(diffs, float), np.asarray(clusters)
    unit = diffs[RNG.integers(0, len(diffs), (reps, len(diffs)))].mean(axis=1)
    names = np.unique(clusters)
    by = [diffs[clusters == c] for c in names]
    picks = RNG.integers(0, len(names), (reps, len(names)))
    cluster = np.array([np.concatenate([by[i] for i in row]).mean() for row in picks])
    return {
        "mean": float(diffs.mean()),
        "unit_ci": [float(np.percentile(unit, 2.5)), float(np.percentile(unit, 97.5))],
        "cluster_ci": [float(np.percentile(cluster, 2.5)), float(np.percentile(cluster, 97.5))],
        "n": len(diffs),
    }


def fmt(value, spec):
    return format(value, spec) if value is not None else "-"


rows = list(csv.DictReader((OUT / "ratings.csv").open()))
for row in rows:
    for key in ("gate", "held_out", "group", "subject_index", "opus", "gpt"):
        row[key] = int(row[key])
    for key in ("score", "reward"):
        row[key] = float(row[key])
table = {(r["arm"], r["subject_index"], r["group"]): r for r in rows}
codes = {}
for source in set(SOURCES.values()):
    path = OUT / f"samples_{source}.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            sample = json.loads(line)
            codes[(source, sample["subject_index"], sample["group"], sample["slot"])] = sample["code"]
for row in rows:  # contains the example: its brush-call signature contains the example's block (9_hack_hunt.py)
    code = codes.get((SOURCES[row["arm"]], row["subject_index"], row["group"], int(row["slot"])), "")
    row["has_example"] = EXAMPLE_BLOCK in signature(code)
held_units = sorted({(r["subject_index"], r["group"]) for r in rows if r["held_out"]})
in_units = sorted({(r["subject_index"], r["group"]) for r in rows if not r["held_out"]})

summary = {"arms": {}, "tests": {}, "in_distribution": {}}
print(
    f"{'arm':9} {'pass':>9} {'interval':>13} {'score':>6} {'opus':>5} {'gpt':>5} "
    f"{'score|pass':>10} {'reward':>7} {'rho':>6}"
)
for arm in ARMS:
    mine = [table[(arm, *u)] for u in held_units if (arm, *u) in table]
    if not mine:
        continue
    passed = [r for r in mine if r["gate"]]
    rho = None
    if len(passed) > 4:
        value = spearmanr([r["reward"] for r in passed], [r["score"] for r in passed]).statistic
        rho = None if math.isnan(value) else float(value)
    entry = {
        "n": len(mine),
        "pass": len(passed),
        "pass_ci": wilson(len(passed), len(mine)),
        "score": float(np.mean([r["score"] for r in mine])),
        "opus": float(np.mean([r["opus"] for r in mine])),
        "gpt": float(np.mean([r["gpt"] for r in mine])),
        "score_when_passing": float(np.mean([r["score"] for r in passed])) if passed else None,
        "reward": float(np.mean([r["reward"] for r in mine])),
        "reward_judge_spearman_passing": rho,
        "score_4_or_more": sum(r["score"] >= 4 for r in mine),
        # post hoc (added after the first arms were judged, because every passing GRPO sketch scored 2):
        # the subject is recognisable = the two judges' mean is at least 3
        "recognisable": sum(r["score"] >= 3 for r in mine),
        "recognisable_ci": wilson(sum(r["score"] >= 3 for r in mine), len(mine)),
        "passing_with_example": sum(r["has_example"] for r in passed),
    }
    summary["arms"][arm] = entry
    ci = entry["pass_ci"]
    print(
        f"{arm:9} {len(passed):3d}/{len(mine):<5d} [{ci[0]:.2f}, {ci[1]:.2f}] "
        f"{entry['score']:6.2f} {entry['opus']:5.2f} "
        f"{entry['gpt']:5.2f} {fmt(entry['score_when_passing'], '10.2f')} {entry['reward']:7.3f} {fmt(rho, '6.2f')}"
    )
    inside = [table[(arm, *u)] for u in in_units if (arm, *u) in table]
    summary["in_distribution"][arm] = {
        "n": len(inside),
        "pass": sum(r["gate"] for r in inside),
        "score": float(np.mean([r["score"] for r in inside])) if inside else None,
        "reward": float(np.mean([r["reward"] for r in inside])) if inside else None,
    }
print("(score = mean of the two judges, 1-5, failures = 1; rho = Spearman of reward with judge score, passing only)")
print("Subject recognisable (mean judge score >= 3, post hoc); passing sketches that contain the prompt's example:")
for arm, entry in summary["arms"].items():
    ci = entry["recognisable_ci"]
    print(
        f"  {arm:9} {entry['recognisable']:2d}/{entry['n']} [{ci[0]:.2f}, {ci[1]:.2f}]   "
        f"example in {entry['passing_with_example']}/{entry['pass']} passing"
    )

print("\nPaired differences on held-out subjects (64 units), 95% intervals: unit bootstrap | subject bootstrap")
for name, (a, b, sided) in TESTS.items():
    units = [u for u in held_units if (a, *u) in table and (b, *u) in table]
    if not units:
        continue
    clusters = [u[0] for u in units]
    result = {"sided": sided}
    for measure in ("score", "opus", "gpt"):
        diffs = [table[(a, *u)][measure] - table[(b, *u)][measure] for u in units]
        result[measure] = bootstrap(diffs, clusters)
    recognisable = [float(table[(a, *u)]["score"] >= 3) - float(table[(b, *u)]["score"] >= 3) for u in units]
    result["recognisable"] = bootstrap(recognisable, clusters)  # post hoc
    s = result["score"]
    result["supported"] = None
    if sided == "greater":
        result["supported"] = bool(s["unit_ci"][0] > 0 and result["opus"]["mean"] > 0 and result["gpt"]["mean"] > 0)
    summary["tests"][name] = result
    print(
        f"  {name:16} {s['mean']:+.2f} [{s['unit_ci'][0]:+.2f}, {s['unit_ci'][1]:+.2f}] | "
        f"[{s['cluster_ci'][0]:+.2f}, {s['cluster_ci'][1]:+.2f}]   opus {result['opus']['mean']:+.2f}, "
        f"gpt {result['gpt']['mean']:+.2f}   ({sided}{', supported' if result['supported'] else ''})"
    )
    rec = result["recognisable"]
    print(
        f"  {'':16} recognisable share {rec['mean']:+.2f} "
        f"[{rec['unit_ci'][0]:+.2f}, {rec['unit_ci'][1]:+.2f}] (post hoc)"
    )

print("\nIn-distribution (sunset over green hills, 8 per arm, descriptive):")
for arm, entry in summary["in_distribution"].items():
    print(
        f"  {arm:9} pass {entry['pass']}/{entry['n']}, score {fmt(entry['score'], '.2f')}, "
        f"reward {fmt(entry['reward'], '.3f')}"
    )

summary["judge_checks"] = json.loads((OUT / "judge_checks.json").read_text())
both = [r for r in rows if r["gate"]]
summary["judge_agreement_passing"] = float(spearmanr([r["opus"] for r in both], [r["gpt"] for r in both]).statistic)
summary["judge_exact_agreement_passing"] = [sum(r["opus"] == r["gpt"] for r in both), len(both)]
print(
    f"\nJudges on all passing arm sketches: Spearman {summary['judge_agreement_passing']:+.2f}, "
    f"same rating {summary['judge_exact_agreement_passing'][0]}/{len(both)}"
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))

print("\nSelf-checks:")
check("every arm has 64 held-out sketches", all(e["n"] == 64 for e in summary["arms"].values()))
check("every pre-registered test computed", all(t in summary["tests"] for t in list(TESTS)[:4]))
