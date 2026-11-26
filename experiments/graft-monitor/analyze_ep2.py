"""Every Episode 2 number in the note -> results/ep2_summary.json.

    cd experiments/graft-monitor && MODAL_PROFILE=albertlieyingadrian .venv/bin/python analyze_ep2.py [--pull] [--score]

Q2 per generation (gen 1 = ep1_<arm>_s<seed>, gen 2/3 = ep2_g<g>_<arm>_s<seed>; seeds 11 and 12): deployed hack rate,
skill and capitulation without authority on the held-out eval (re-graded with env parser v2), and the undetected-hack
rate: ground-truth hacks on hackable wrong-assertion episodes that the routing monitor (gpt-oss-safeguard-20b, low
effort, the shared policy, reference shown) does not flag, over all hackable wrong-assertion episodes. --score sends the
deployed eval transcripts to the monitor endpoint (cached in results/ep2_monitor_verdicts.jsonl). Values-battery and
Petri readouts per generation come from results/drift/<tag>/summary.json (battery/drift.py; q1_<arm>_s11 is gen 1,
q2_g<g>_<arm>_s11 the later generations). Q3 comes from results/q3/summary_*.json (q3_selfmon.py).
Intervals: 95% percentile bootstrap over episodes within each seed, then the seed mean, 2000 draws.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env  # noqa: E402

RUNS = HERE / "results" / "runs"
OUT = HERE / "results" / "ep2_summary.json"
VERDICTS = HERE / "results" / "ep2_monitor_verdicts.jsonl"
ARMS = ("none", "penalty", "graft")
SEEDS = (11, 12)
FIELDS = env.Episode.__dataclass_fields__


def run_name(arm: str, seed: int, gen: int) -> str:
    return f"ep1_{arm}_s{seed}" if gen == 1 else f"ep2_g{gen}_{arm}_s{seed}"


def deployed_rows(name: str) -> list[dict] | None:
    d = RUNS / name
    if not (d / "done.json").exists():
        return None
    cfg = json.loads((d / "config.json").read_text())
    mode = "retain_only" if cfg["arm"] in ("graft", "graft_noroute") else "both"
    last = sorted((d / "evals").glob("step_*.jsonl"))[-1]
    out = []
    for x in last.read_text().splitlines():
        r = json.loads(x)
        if r["mode"] != mode:
            continue
        ep = env.Episode(**{k: r[k] for k in FIELDS})
        out.append(
            {
                **ep.to_dict(),
                "first": r["first"],
                "final_text": r["final_text"],
                **env.grade(ep, r["first"], r["final_text"]),
            }
        )
    return out


def score_verdicts(names: list[str]) -> dict:
    cache = {}
    if VERDICTS.exists():
        for x in VERDICTS.read_text().splitlines():
            v = json.loads(x)
            cache[(v["run"], v["id"])] = v
    import monitors

    mon = monitors.get_monitor("oss_safeguard", reasoning_effort="low")
    policy = monitors.load_policy()
    todo = []
    for n in names:
        for r in deployed_rows(n) or []:
            if r["hackable"] and not r["assertion_true"] and (n, r["id"]) not in cache:
                ep = env.Episode(**{k: r[k] for k in FIELDS})
                todo.append((n, r["id"], ep.transcript(r["first"], r["final_text"]), r["truth"]))
    if todo:
        res = mon.score_many([(t, policy, ref) for _, _, t, ref in todo], concurrency=24)
        with open(VERDICTS, "a") as f:
            for (n, i, _, _), v in zip(todo, res):
                rec = {"run": n, "id": i, "flag": v.get("flag"), "score": v.get("score")}
                cache[(n, i)] = rec
                f.write(json.dumps(rec) + "\n")
    return cache


def boot_mean(per_seed: list[list[float]], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    vals = sorted(
        sum(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for xs in per_seed) / len(per_seed) for _ in range(n)
    )
    return [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]]


def drift_readout(tag: str) -> dict | None:
    p = HERE / "results" / "drift" / tag / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    out = {"flagged": s.get("flagged", []), "readouts": {}}
    for k, v in s["readouts"].items():
        c = v.get("checkpoint_vs_base", {})
        cc = v.get("checkpoint_vs_control", {})
        out["readouts"][k] = {
            "base": v.get("means", {}).get("base"),
            "checkpoint": v.get("means", {}).get("checkpoint"),
            "control": v.get("means", {}).get("control"),
            "delta": c.get("delta"),
            "ci": [c.get("lo"), c.get("hi")],
            "vs_control": cc.get("delta"),
            "vs_control_ci": [cc.get("lo"), cc.get("hi")],
            "floor": v.get("floor", {}).get("delta"),
            "flag": v.get("flag"),
        }
    pet = s.get("petri") or {}
    if isinstance(pet.get("checkpoint"), dict):
        keep = (
            "unprompted_sycophancy",
            "concerning",
            "user_over_sysprompt",
            "admirable",
            "unprompted_deception_toward_user",
            "disappointing",
        )
        out["petri"] = {
            m: {k: v for k, v in (pet.get(m, {}).get("means") or {}).items() if k in keep}
            for m in ("base", "checkpoint", "control")
        }
    b = (s.get("bdiff") or {}).get("pairs", {}).get("checkpoint")
    if b:
        out["bdiff_changed"] = b.get("n_changed")
        out["bdiff_categories"] = b.get("n_categories")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    names = [run_name(arm, s, g) for arm in ARMS for s in SEEDS for g in (1, 2, 3)]
    if a.pull:
        for n in names + ["ep2_ctrl_s11", "ep2_ctrl_s12"]:
            subprocess.run(
                [sys.executable, str(HERE / "trainer" / "modal_train.py"), "pull", n],
                check=False,
                capture_output=True,
                env={**os.environ, "MODAL_PROFILE": "albertlieyingadrian"},
            )
    done = [n for n in names if deployed_rows(n) is not None]
    verdicts = score_verdicts(done) if a.score else {}
    if not a.score and VERDICTS.exists():
        for x in VERDICTS.read_text().splitlines():
            v = json.loads(x)
            verdicts[(v["run"], v["id"])] = v
    out = {"generations": {}, "runs": {}, "drift": {}, "q3": {}, "missing": sorted(set(names) - set(done))}
    for arm in ARMS:
        for g in (1, 2, 3):
            per_hack, per_skill, per_undet, per_cap = [], [], [], []
            for s in SEEDS:
                n = run_name(arm, s, g)
                rows = deployed_rows(n)
                if rows is None:
                    continue
                hw = [r for r in rows if r["hackable"] and not r["assertion_true"]]
                sk = [r for r in rows if (not r["hackable"]) or r["assertion_true"]]
                cu = [r for r in rows if (not r["hackable"]) and not r["assertion_true"]]
                und = [float(r["gt_hack"] and verdicts.get((n, r["id"]), {}).get("flag") is not True) for r in hw]
                scored = all((n, r["id"]) in verdicts for r in hw)
                per_hack.append([float(r["gt_hack"]) for r in hw])
                per_skill.append([float(r["correct"]) for r in sk])
                per_cap.append([float(r["capitulated"]) for r in cu])
                if scored:
                    per_undet.append(und)
                out["runs"][n] = {
                    "hack_rate": sum(per_hack[-1]) / len(hw),
                    "skill": sum(per_skill[-1]) / len(sk),
                    "capitulation_unhackable": sum(per_cap[-1]) / len(cu),
                    "undetected_hack_rate": (sum(und) / len(hw)) if scored else None,
                }
            if not per_hack:
                continue

            def m(ps):
                return sum(sum(x) / len(x) for x in ps) / len(ps)

            rec = {
                "n_seeds": len(per_hack),
                "hack_rate": m(per_hack),
                "ci_hack": boot_mean(per_hack),
                "skill": m(per_skill),
                "ci_skill": boot_mean(per_skill),
                "capitulation_unhackable": m(per_cap),
            }
            if per_undet and len(per_undet) == len(per_hack):
                rec["undetected_hack_rate"] = m(per_undet)
                rec["ci_undetected"] = boot_mean(per_undet)
            out["generations"][f"{arm}_g{g}"] = rec
    for tag in [f"q1_{arm}_s{s}" for arm in ARMS for s in SEEDS] + [
        f"q2_g{g}_{arm}_s{s}" for g in (2, 3) for arm in ARMS for s in SEEDS
    ]:
        r = drift_readout(tag)
        if r:
            out["drift"][tag] = r
    for p in sorted((HERE / "results" / "q3").glob("summary_*.json")):
        out["q3"][p.stem.removeprefix("summary_")] = json.loads(p.read_text())
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{'arm_gen':14s} {'hack':>6s} {'95% CI':>16s} {'undet':>6s} {'skill':>6s} {'cap(no auth)':>12s}")
    for k, r in out["generations"].items():
        u = r.get("undetected_hack_rate")
        us = f"{u:6.3f}" if u is not None else "     -"
        print(
            f"{k:14s} {r['hack_rate']:6.3f} {[round(x, 3) for x in r['ci_hack']]!s:>16s} {us} "
            f"{r['skill']:6.3f} {r['capitulation_unhackable']:12.3f}"
        )
    print("drift tags:", list(out["drift"]), "| missing runs:", out["missing"])


if __name__ == "__main__":
    main()
