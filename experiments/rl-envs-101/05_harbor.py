"""Step 5: run the packaged tasks with Harbor on Modal, then summarise every job from its result files.

The runs (COST MONEY: Modal CPU sandboxes, plus one L40S for the model; the lesson README has the full commands):

    harbor run -p experiments/rl-envs-101/harbor_tasks/heldout -a oracle -e modal -n 8 -o <jobs>/oracle
    harbor run -p ... -a nop -e modal -n 8 -o <jobs>/nop
    harbor run -p ... -a terminus-2 -m hosted_vllm/desk-agent --base-url <modal url>/v1 -k 2 -e modal -n 8 -o <jobs>/qwen

The summary (free, seconds):

    python experiments/rl-envs-101/05_harbor.py <jobs>/oracle <jobs>/nop <jobs>/qwen

reads each trial's result.json, writes results/05_harbor.json (one row per trial: task, attempt, the reward.json
keys, whether the reward is valid, the error type if any) and prints the mean of every key, the visible-hidden gap
and the share of tasks passed (hidden) on every attempt.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "05_harbor.json"


def trials(job_root: Path) -> list[dict]:
    rows = []
    for path in sorted(job_root.glob("*/*/result.json")):
        r = json.loads(path.read_text())
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        name = r["task_name"].split("/")[-1]
        rows.append(
            {
                "task": name,
                "trial": r["trial_name"],
                "kind": name.split("-")[1],
                "reward": rewards.get("reward"),
                "hidden": rewards.get("hidden"),
                "collateral": rewards.get("collateral"),
                "reward_valid": r.get("reward_valid"),
                "error": (r.get("exception_info") or {}).get("exception_type"),
            }
        )
    return rows


def summary(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["reward_valid"] and r["reward"] is not None]
    out = {"trials": len(rows), "valid": len(valid), "errors": sum(r["error"] is not None for r in rows)}
    if not valid:
        return out
    for key in ("reward", "hidden", "collateral"):
        out[key] = round(sum(r[key] for r in valid) / len(valid), 4)
    out["gap"] = round(out["reward"] - out["hidden"], 4)
    by_task = defaultdict(list)
    for r in valid:
        by_task[r["task"]].append(r["hidden"])
    out["tasks"] = len(by_task)
    out["pass_all_attempts"] = round(sum(all(h == 1.0 for h in v) for v in by_task.values()) / len(by_task), 4)
    out["by_kind"] = {}
    for kind in ("book", "reschedule", "cancel"):
        sub = [r for r in valid if r["kind"] == kind]
        if sub:
            out["by_kind"][kind] = {
                "n": len(sub),
                "reward": round(sum(r["reward"] for r in sub) / len(sub), 3),
                "hidden": round(sum(r["hidden"] for r in sub) / len(sub), 3),
            }
    return out


def show(report: dict) -> None:
    for name, job in report.items():
        s = job["summary"]
        line = f"{name:10s} trials {s['trials']:3d} valid {s['valid']:3d} errors {s['errors']:2d}"
        if s.get("valid"):
            line += f" | reward {s['reward']:.3f} hidden {s['hidden']:.3f} gap {s['gap']:+.3f}"
            line += f" collateral {s['collateral']:.2f} | tasks passed on every attempt {s['pass_all_attempts']:.3f}"
        print(line)
        for kind, k in s.get("by_kind", {}).items():
            print(f"    {kind:10s} n={k['n']:3d} reward {k['reward']:.3f} hidden {k['hidden']:.3f}")


if __name__ == "__main__":
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arg in sys.argv[1:]:
        root = Path(arg)
        report[root.name] = {"summary": summary(trials(root)), "trials": trials(root)}
    if not report:
        raise SystemExit(__doc__)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    show(report)
    if "oracle" in report:
        ok = report["oracle"]["summary"].get("hidden") == 1.0 and report["oracle"]["summary"]["errors"] == 0
        print(("✅" if ok else "❌"), "the oracle scores reward = hidden = 1 on every task, in real sandboxes")
    if "nop" in report:
        ok = report["nop"]["summary"].get("hidden") == 0.0
        print(("✅" if ok else "❌"), "doing nothing scores hidden 0 on every task")
