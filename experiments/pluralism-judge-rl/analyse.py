"""
Turn checkpoint scores into the note's numbers: every measure against steps and drift, and the pre-registered
comparison at matched drift, as pre-registered.

    .venv/bin/python experiments/pluralism-judge-rl/analyse.py [--config configs/full.json]

Reads results/runs/<run>/curve.jsonl (drift per checkpoint) and ~/.cache/steerlab/pluralism-judge-rl/scores/<tag>.json.
Writes results/checkpoints.csv (one row per scored checkpoint) and results/summary.json.

Matched drift: for each task, D* = the smallest, over the four judge-related arms (single, jury, shuffled, bon), of the
arm's largest seed-averaged drift. Each run's per-item values are interpolated linearly in that run's own drift between
the two checkpoints that bracket D*; an arm whose largest drift is below D*/2 is read at its final checkpoint and
flagged. Delta = trained - base per item (paired on the same prompts / questions), 95% bootstrap over items (2,000).
"Plurality loss" is signed so that positive = less plural: +repetition, -coverage, -steerability, -survey entropy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "pluralism-explainer")]
import plural as EP1  # noqa: E402  (Episode 1's survey metrics)
from common import CACHE, HERE, RESULTS, read_jsonl  # noqa: E402

EP1_Q = {q["id"]: q for q in EP1.load_questions()}
EP1_MIN_VALID = 10  # Episode 1's rule: a question-condition cell counts only with >= 10 valid answers of 50


def ep1_cells(m):
    """{(qid, cond): counts} for cells with enough valid answers, from the Episode 1 survey file of this tag."""
    return {
        (r["id"], r["cond"]): np.array(r["counts"], float)
        for r in m["ep1"]["rows"]
        if sum(r["counts"]) >= EP1_MIN_VALID
    }


def ep1_measure(m, kind):
    cells = ep1_cells(m)
    out = {}
    for qid, q in EP1_Q.items():
        if kind == "entropy" and (qid, "default") in cells:
            out[qid] = EP1.entropy(cells[(qid, "default")])
        elif kind == "jsd_default" and (qid, "default") in cells:
            out[qid] = float(np.mean([EP1.jsd(cells[(qid, "default")], q["human"][c]) for c in EP1.COUNTRIES]))
        elif kind in ("jsd_steered", "entropy_steered"):
            vals = [
                EP1.jsd(cells[(qid, c)], q["human"][c]) if kind == "jsd_steered" else EP1.entropy(cells[(qid, c)])
                for c in EP1.COUNTRIES
                if (qid, c) in cells
            ]
            if vals:
                out[qid] = float(np.mean(vals))
    return out


def shared(m, key, task=None):
    v = m["shared"][key]
    return v[task] if task else {k: x["coverage"] for k, x in v.items()}


JUDGE_ARMS = ["single", "jury", "shuffled", "bon"]
ALL_ARMS = [*JUDGE_ARMS, "clean"]
VALID_MASS, VALID_RATE = 0.5, 0.8  # pre-registered validity threshold for the survey measures


def steered(m, key):
    by = defaultdict(list)
    for r in m["survey"]:
        if r["cond"] != "default":
            by[r["id"]].append(r["jsd"][r["cond"]] if key == "jsd" else r["entropy"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def default_survey(m, key):
    out = {}
    for r in m["survey"]:
        if r["cond"] == "default":
            out[r["id"]] = r["entropy"] if key == "entropy" else float(np.mean(list(r["jsd"].values())))
    return out


def rep(m, task, key):
    return {r["id"]: r[key] for r in m["repetition"][task]}


# measure -> (sign for "plurality loss" (0 = not a plurality measure), per-item values from a score file)
MEASURES = {
    "rep_creative": (+1, lambda m: rep(m, "creative", "rep")),
    "rep_advice": (+1, lambda m: rep(m, "advice", "rep")),
    "rep50_creative": (+1, lambda m: rep(m, "creative", "rep50")),
    "rep50_advice": (+1, lambda m: rep(m, "advice", "rep50")),
    "words_creative": (0, lambda m: rep(m, "creative", "words")),
    "words_advice": (0, lambda m: rep(m, "advice", "words")),
    "coverage": (-1, lambda m: {r["id"]: r["coverage"] for r in m["coverage"]}),
    "coverage_loose": (-1, lambda m: {r["id"]: r["coverage_loose"] for r in m["coverage"]}),
    "cov_valid": (0, lambda m: {r["id"]: r["valid_rate"] for r in m["coverage"]}),
    "steer": (-1, lambda m: {f"{r['id']}-{r['target']}": r["steerability"] for r in m["steer"]}),
    "survey_entropy": (-1, lambda m: default_survey(m, "entropy")),
    "survey_jsd_default": (0, lambda m: default_survey(m, "jsd")),
    "survey_jsd_steered": (0, lambda m: steered(m, "jsd")),
    "survey_entropy_steered": (-1, lambda m: steered(m, "entropy")),
    # ---- shared modules (plan Amendment 5): the primary measures
    "srep_creative": (+1, lambda m: shared(m, "rep", "creative")),
    "srep_advice": (+1, lambda m: shared(m, "rep", "advice")),
    "srep50_creative": (+1, lambda m: shared(m, "rep50", "creative")),
    "srep50_advice": (+1, lambda m: shared(m, "rep50", "advice")),
    "scoverage": (-1, lambda m: shared(m, "coverage")),
    # exploratory (added after reading replies): per prompt, mean similarity to replies to the other prompts
    "house_creative": (+1, lambda m: m["shared"]["house"]["creative"]),
    "house_advice": (+1, lambda m: m["shared"]["house"]["advice"]),
    "house50_creative": (+1, lambda m: m["shared"]["house50"]["creative"]),
    "house50_advice": (+1, lambda m: m["shared"]["house50"]["advice"]),
    "ep1_entropy": (-1, lambda m: ep1_measure(m, "entropy")),
    "ep1_entropy_steered": (-1, lambda m: ep1_measure(m, "entropy_steered")),
    "ep1_jsd_default": (0, lambda m: ep1_measure(m, "jsd_default")),
    "ep1_jsd_steered": (0, lambda m: ep1_measure(m, "jsd_steered")),
    "ep1_valid": (0, lambda m: {f"{r['id']}-{r['cond']}": 1 - r["invalid"] / 50 for r in m["ep1"]["rows"]}),
    "survey_mass": (0, lambda m: {f"{r['id']}-{r['cond']}": r["mass"] for r in m["survey"]}),
    "survey_valid": (0, lambda m: {f"{r['id']}-{r['cond']}": float(r["valid"]) for r in m["survey"]}),
}
PRIMARY = {"creative": "srep_creative", "advice": "ep1_entropy"}  # Amendment 5


def load_scores(tag):
    """The wrapper scores plus the shared-module files; None until all three exist for this tag."""

    def read(sub):
        p = CACHE / sub / f"{tag}.json"
        if p.exists():
            return json.loads(p.read_text())
        g = RESULTS / sub / f"{tag}.json.gz"  # the shipped copy, so results/ alone reproduces every number
        if g.exists():
            import gzip

            with gzip.open(g, "rt") as f:
                return json.loads(f.read())
        return None

    parts = [read("scores"), read("shared"), read("ep1")]
    if any(x is None for x in parts):
        return None
    m = parts[0]
    m["shared"], m["ep1"] = parts[1], parts[2]
    return m


def survey_valid(m):
    """Pre-registered validity: option-token mass (wrapper readout) >= 0.5 and valid-answer rate (Episode 1's
    readout, Amendment 5) >= 0.8."""
    mass = float(np.mean([r["mass"] for r in m["survey"]]))
    rate = 1 - sum(r["invalid"] for r in m["ep1"]["rows"]) / (50 * len(m["ep1"]["rows"]))
    return mass >= VALID_MASS and rate >= VALID_RATE, mass, rate


def run_meta(config):
    spec = json.loads((HERE / config).read_text())
    out = {}
    for r in spec["runs"]:
        cfg = dict(spec["common"])
        cfg.update(r)
        out[cfg["name"]] = cfg
    return out


def checkpoints(run):
    path = RESULTS / "runs" / run / "curve.jsonl"
    return [r for r in read_jsonl(path) if r.get("checkpoint")] if path.exists() else []


def tag_of(run, step):
    return "base-s0" if step == 0 else f"{run}-step{step}"


def item_values(run, step, measure):
    m = load_scores(tag_of(run, step))
    if m is None or (measure.startswith(("survey", "ep1")) and measure != "ep1_valid" and not survey_valid(m)[0]):
        return None
    return MEASURES[measure][1](m)


def at_drift(run, measure, target):
    """Per-item values interpolated at drift `target` along this run's scored checkpoints."""
    pts = sorted(
        (c["step"], c["drift_kl"]) for c in checkpoints(run) if load_scores(tag_of(run, c["step"])) is not None
    )
    if len(pts) < 2:
        return None, None
    if max(d for _, d in pts) < target:
        s, d = pts[-1]
        return item_values(run, s, measure), {"step": s, "drift": d, "reached": False}
    for (s0, d0), (s1, d1) in itertools.pairwise(pts):
        if d0 <= target <= d1 and d1 > d0:
            a, b = item_values(run, s0, measure), item_values(run, s1, measure)
            if a is None or b is None:
                return None, None
            w = (target - d0) / (d1 - d0)
            return {k: (1 - w) * a[k] + w * b[k] for k in a if k in b}, {"steps": [s0, s1], "w": w, "reached": True}
    return None, None


def bootstrap_delta(trained, base, sign, n=2000, seed=0):
    keys = sorted(set(trained) & set(base))
    s = sign if sign else 1
    d = np.array([s * (trained[k] - base[k]) for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {
        "delta": float(d.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": len(d),
    }


def checkpoint_row(run, arm, task, seed, step, drift, m):
    ok, _mass, _rate = survey_valid(m)
    row = {"run": run, "arm": arm, "task": task, "seed": seed, "step": step, "drift_kl": drift, "survey_ok": ok}
    for k, (_, f) in MEASURES.items():
        row[k] = float(np.mean(list(f(m).values())))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.json")
    args = ap.parse_args()
    meta = run_meta(args.config)
    base = load_scores("base-s0")
    base1 = load_scores("base-s1")

    rows = [
        checkpoint_row(t, "base", "", m["seed"], 0, 0.0, m) for t, m in [("base-s0", base), ("base-s1", base1)] if m
    ]
    for run, cfg in meta.items():
        for c in checkpoints(run):
            m = load_scores(f"{run}-step{c['step']}") if c["step"] else None
            if m is not None:
                rows.append(
                    checkpoint_row(run, cfg["arm"], cfg.get("task") or "none", cfg["seed"], c["step"], c["drift_kl"], m)
                )
    with (RESULTS / "checkpoints.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} checkpoint rows to results/checkpoints.csv")

    summary = {
        "floor_base_rerun": {},
        "matched": {},
        "validity": {f"{r['run']}@{r['step']}": r["survey_ok"] for r in rows},
    }
    if base and base1:  # do-nothing floor: the base model rescored with another sampling seed
        summary["floor_base_rerun"] = {k: bootstrap_delta(f(base1), f(base), s) for k, (s, f) in MEASURES.items()}

    for task in ["creative", "advice"]:
        runs_by_arm = defaultdict(list)
        for run, cfg in meta.items():
            if cfg["arm"] == "clean" or cfg.get("task") == task:
                runs_by_arm[cfg["arm"]].append(run)
        reach = {}
        for arm in JUDGE_ARMS:
            curves = [[c["drift_kl"] for c in checkpoints(r)] for r in runs_by_arm.get(arm, [])]
            curves = [c for c in curves if len(c) > 1]
            if curves:
                n = min(len(c) for c in curves)
                reach[arm] = float(max(np.mean([c[:n] for c in curves], axis=0)))
        if not reach:
            continue
        top = max(reach.values())
        target = min(v for v in reach.values() if v >= 0.5 * top)
        out = {"D": target, "arm_max_drift": reach, "arms": {}, "primary": PRIMARY[task]}
        for arm in ALL_ARMS:
            arm_out = {}
            for measure, (sign, f) in MEASURES.items():
                per_seed = []
                for run in runs_by_arm.get(arm, []):
                    vals, how = at_drift(run, measure, target)
                    if vals is None:
                        continue
                    res = bootstrap_delta(vals, f(base), sign)
                    res.update({"run": run, "how": how})
                    per_seed.append(res)
                if per_seed:
                    arm_out[measure] = {
                        "seeds": per_seed,
                        "seed_mean": float(np.mean([s["delta"] for s in per_seed])),
                        "beyond_floor": all(s["lo"] > 0 for s in per_seed) or all(s["hi"] < 0 for s in per_seed),
                    }
            out["arms"][arm] = arm_out
        # Amendment 4: an arm that does not reach D*/2 is compared with every arm at its own largest drift
        out["pairwise"] = {}
        for low_arm, low_reach in reach.items():
            if low_reach >= 0.5 * top:
                continue
            pair = {"D": low_reach, "arms": {}}
            for arm in [*JUDGE_ARMS, "clean"]:
                arm_out = {}
                for measure, (sign, f) in MEASURES.items():
                    per_seed = []
                    for run in runs_by_arm.get(arm, []):
                        vals, how = at_drift(run, measure, low_reach)
                        if vals is None:
                            continue
                        res = bootstrap_delta(vals, f(base), sign)
                        res.update({"run": run, "how": how})
                        per_seed.append(res)
                    if per_seed:
                        arm_out[measure] = {
                            "seeds": per_seed,
                            "seed_mean": float(np.mean([s["delta"] for s in per_seed])),
                        }
                pair["arms"][arm] = arm_out
            out["pairwise"][low_arm] = pair
        prim = PRIMARY[task]
        out["ranking_primary"] = sorted(
            [(a, v[prim]["seed_mean"]) for a, v in out["arms"].items() if prim in v and a in JUDGE_ARMS],
            key=lambda x: -x[1],
        )
        summary["matched"][task] = out
        print(
            f"\n[{task}] matched drift D* = {target:.4f}; arm max drift "
            f"{json.dumps({k: round(v, 4) for k, v in reach.items()})}"
        )
        for arm, v in out["arms"].items():
            if prim in v:
                s = v[prim]
                cis = ", ".join(f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]" for x in s["seeds"])
                print(
                    f"  {arm:9s} {prim} loss: seed mean {s['seed_mean']:+.4f}; per seed {cis}; beyond floor "
                    f"{s['beyond_floor']}"
                )
    summary["predictions"] = score_predictions(summary)
    for k, v in summary["predictions"].items():
        print(f"{k}: {v}")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote results/summary.json")


def score_predictions(summary):
    """The plan's four predictions, scored mechanically against their thresholds (P3 needs extras.json)."""
    out = {}
    c = summary["matched"].get("creative", {})
    pair = c.get("pairwise", {}).get("shuffled", {}).get("arms", {})
    m = "srep_creative"
    if m in pair.get("single", {}) and m in pair.get("shuffled", {}):
        d = pair["single"][m]["seed_mean"] - pair["shuffled"][m]["seed_mean"]
        out["P1"] = {"value": d, "threshold": "> 0.02", "hit": d > 0.02, "at_drift": c["pairwise"]["shuffled"]["D"]}
    arms = c.get("arms", {})
    if m in arms.get("single", {}) and m in arms.get("jury", {}):
        d = arms["single"][m]["seed_mean"] - arms["jury"][m]["seed_mean"]
        out["P2"] = {"value": d, "threshold": "> 0.01", "hit": d > 0.01, "at_drift": c["D"]}
    ex = RESULTS / "extras.json"
    if ex.exists():
        fav = json.loads(ex.read_text()).get("jury_favourites", {})
        jury = [v["entropy"] for k, v in fav.items() if k.startswith("jury-")]
        base = [v["entropy"] for k, v in fav.items() if k.startswith("base-")]
        if jury and base:
            d = float(np.mean(jury) - np.mean(base))
            out["P3"] = {"value": d, "threshold": ">= -0.05", "hit": d >= -0.05}
    a = summary["matched"].get("advice", {})
    if a:
        n = sum(1 for arm, v in a["arms"].items() if "ep1_entropy" in v and v["ep1_entropy"]["beyond_floor"])
        out["P4"] = {"value": n, "threshold": "= 0", "hit": n == 0, "at_drift": a["D"]}
    return out


if __name__ == "__main__":
    main()
