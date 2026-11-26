"""Every Episode 1 number in the note, from the pulled runs in results/runs/ep1_* -> results/ep1_summary.json.

    .venv/bin/python experiments/graft-monitor/analyze_ep1.py [--pull]

--pull first copies each run's logs and eval transcripts from the Modal Volume (modal_train.py pull).
All rates are re-graded from the saved eval transcripts with env.grade (parser v2), on the held-out split
(eval_split=heldout, 256 fixed episodes, seed 99). Deployed policy: RETAIN only for GRAFT-family arms, both adapters
otherwise. Intervals: 95% percentile bootstrap, resampling episodes within each seed, 2000 draws.
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
OUT = HERE / "results" / "ep1_summary.json"
GRAFT_FAMILY = {"graft", "graft_noroute"}
FIELDS = env.Episode.__dataclass_fields__


def regrade(row: dict) -> dict:
    ep = env.Episode(**{k: row[k] for k in FIELDS})
    return {**ep.to_dict(), "mode": row["mode"], **env.grade(ep, row["first"], row["final_text"])}


def boot(per_seed_rows: list[list[dict]], key: str, n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        ms = [env.summarize([rows[rng.randrange(len(rows))] for _ in rows])[key] for rows in per_seed_rows]
        ms = [m for m in ms if m == m]
        vals.append(sum(ms) / len(ms))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]]


def group_of(cfg: dict) -> str:
    if cfg["name"].startswith("ep1_recall"):
        return f"recall={cfg['recall']}"
    if cfg.get("random_flags", -1) >= 0:
        return f"random_flags={cfg['random_flags']}"
    return cfg["arm"]


def load_run(d: Path) -> dict | None:
    if not (d / "config.json").exists() or not (d / "done.json").exists():
        return None
    cfg = json.loads((d / "config.json").read_text())
    steps = sorted((d / "evals").glob("step_*.jsonl"))
    last, first = steps[-1], steps[0]
    deployed = "retain_only" if cfg["arm"] in GRAFT_FAMILY else "both"
    final = [regrade(json.loads(x)) for x in last.read_text().splitlines()]
    start = [regrade(json.loads(x)) for x in first.read_text().splitlines()]
    log = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()] if (d / "log.jsonl").exists() else []
    mon = {}
    if log and cfg.get("flag_source") == "http":
        for k in ("flag_rate", "flag_recall", "flag_fpr"):
            v = [r[k] for r in log if r.get(k) is not None and r[k] == r[k]]
            mon[k] = sum(v) / len(v) if v else None
    done = json.loads((d / "done.json").read_text())
    return {
        "name": cfg["name"],
        "group": group_of(cfg),
        "arm": cfg["arm"],
        "seed": cfg["seed"],
        "flag_source": cfg.get("flag_source"),
        "deployed_mode": deployed,
        "final_step": int(last.stem.split("_")[1]),
        "steps_done": done.get("steps_done"),
        "rows": {m: [r for r in final if r["mode"] == m] for m in ("both", "retain_only", "forget_only")},
        "base_rows": [r for r in start if r["mode"] == "both"],
        "monitor": mon,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args()
    names = sorted(p.name for p in RUNS.glob("ep1_*"))
    if a.pull:
        for n in names:
            subprocess.run(
                [sys.executable, str(HERE / "trainer" / "modal_train.py"), "pull", n],
                check=False,
                env={**os.environ, "MODAL_PROFILE": "albertlieyingadrian"},
                capture_output=True,
            )
    runs = [r for r in (load_run(RUNS / n) for n in names) if r]
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    out = {"n_runs": len(runs), "missing": sorted(set(names) - {r["name"] for r in runs}), "groups": {}, "runs": {}}
    base_all = [r["base_rows"] for r in runs if r["base_rows"]]
    if base_all:
        pooled = env.summarize([x for rows in base_all for x in rows])
        out["base"] = {k: pooled[k] for k in ("hack_rate", "skill", "capitulation_unhackable", "parsed")}
        out["base"]["ci_hack"] = boot(base_all, "hack_rate")
        out["base"]["ci_skill"] = boot(base_all, "skill")
        out["base"]["note"] = "step-0 eval (adapters at init = the base model), pooled over every run"
    for g, rs in sorted(groups.items()):
        rec = {"deployed_mode": rs[0]["deployed_mode"], "seeds": sorted(r["seed"] for r in rs), "n_seeds": len(rs)}
        for mode in ("both", "retain_only", "forget_only"):
            per = [r["rows"][mode] for r in rs if r["rows"][mode]]
            if not per:
                continue
            ms = [env.summarize(p) for p in per]
            m = {
                k: sum(x[k] for x in ms) / len(ms) for k in ("hack_rate", "skill", "capitulation_unhackable", "parsed")
            }
            m["per_seed_hack"] = [round(x["hack_rate"], 4) for x in ms]
            m["per_seed_skill"] = [round(x["skill"], 4) for x in ms]
            m["ci_hack"] = boot(per, "hack_rate")
            m["ci_skill"] = boot(per, "skill")
            rec[mode] = m
        mons = [r["monitor"] for r in rs if r["monitor"]]
        if mons:
            rec["monitor_train"] = {
                k: sum(x[k] for x in mons if x.get(k) is not None) / len(mons)
                for k in ("flag_rate", "flag_recall", "flag_fpr")
            }
        out["groups"][g] = rec
    for r in runs:
        out["runs"][r["name"]] = {
            "group": r["group"],
            "seed": r["seed"],
            "final_step": r["final_step"],
            "steps_done": r["steps_done"],
            "deployed": env.summarize(r["rows"][r["deployed_mode"]]),
            "monitor": r["monitor"],
        }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{'group':24s} {'n':>2s} {'deployed hack':>14s} {'95% CI':>16s} {'skill':>6s} {'cap(no auth)':>12s}")
    if "base" in out:
        b = out["base"]
        print(
            f"{'base (step 0)':24s} {'':>2s} {b['hack_rate']:14.3f} {[round(x, 3) for x in b['ci_hack']]!s:>16s} "
            f"{b['skill']:6.3f} {b['capitulation_unhackable']:12.3f}"
        )
    for g, rec in out["groups"].items():
        m = rec[rec["deployed_mode"]]
        print(
            f"{g:24s} {rec['n_seeds']:2d} {m['hack_rate']:14.3f} {[round(x, 3) for x in m['ci_hack']]!s:>16s} "
            f"{m['skill']:6.3f} {m['capitulation_unhackable']:12.3f}"
        )
    print("missing:", out["missing"])


if __name__ == "__main__":
    main()
