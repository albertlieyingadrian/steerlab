"""Launch and collect the GRPO runs (configs/arms.json) on Modal.

    python train.py pilot                 # one `full` run of 30 steps (exploration; read on train rollouts only)
    python train.py launch [names...]     # spawn the runs (all by default); call ids -> results/runs/spawned.json
    python train.py status                # each run's latest curve row
    python train.py pull [names...]       # curve.jsonl -> results/runs/<name>/, rollouts -> data/rollouts/<name>.jsonl

Runs are spawned on the deployed app, so they keep going if this laptop sleeps. Their cost is read from Modal's
billing API by `budget.py` (app `detector-evasion`), which counts the trainer and the Reward server together.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import modal

import budget
from common import APP, CONFIGS, EP3_THRESHOLDS, HERE, RESULTS, REWRITE_PROMPT, sources

RUNS = RESULTS / "runs"
DATA = HERE / "data"  # gitignored: full rollouts
MODAL_BIN = budget.MODAL_BIN
PER_RUN_USD = 15.0  # projection per 300-step run (H100 + its share of the Reward server)


def configs() -> list[dict]:
    spec = json.loads((CONFIGS / "arms.json").read_text())
    return [{**spec["common"], **r} for r in spec["runs"]]


def payload(cfg: dict) -> dict:
    keep = ("sid", "text", "claims")
    return {**cfg, "prompt": REWRITE_PROMPT, "threshold": EP3_THRESHOLDS["nine_b"],
            "train": [{k: s[k] for k in keep} for s in sources("train")],
            "dev": [{k: s[k] for k in keep} for s in sources("dev")]}  # fmt: skip


def spawn(cfgs: list[dict]) -> None:
    s = budget.spent()
    proj = PER_RUN_USD * len(cfgs) * max(c["steps"] for c in cfgs) / 300
    if s + proj > budget.HARD_CAP:
        raise budget.BudgetExceeded(f"spent ${s:.2f} + projected ${proj:.2f} > ${budget.HARD_CAP:.0f}")
    print(f"[budget] spent ${s:.2f}; projected ${proj:.2f} for {len(cfgs)} runs")
    fn = modal.Function.from_name(APP, "train_run")
    path = RUNS / "spawned.json"
    spawned = json.loads(path.read_text()) if path.exists() else {}
    for cfg in cfgs:
        call = fn.spawn(payload(cfg))
        spawned[cfg["name"]] = {"call_id": call.object_id, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "profile": budget.profile()}  # fmt: skip
        print(f"spawned {cfg['name']}: {call.object_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spawned, indent=1) + "\n")


def vol_get(remote: str, local) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [MODAL_BIN, "volume", "get", "--force", APP, remote, str(local)], capture_output=True, text=True, check=False
    )
    return r.returncode == 0


def status() -> None:
    tmp = DATA / "tmp_curve.jsonl"
    for cfg in [*configs(), {"name": "pilot-full"}]:
        if not vol_get(f"runs/{cfg['name']}/curve.jsonl", tmp):
            print(f"{cfg['name']:16} (no curve yet)")
            continue
        last = json.loads(tmp.read_text().splitlines()[-1])
        print(f"{cfg['name']:16} step {last['step']:3d} " + " ".join(
            f"{k} {last[k]:.3f}" for k in ("reward", "s9", "evade", "claim", "loop", "trunc", "mean_tokens")
            if k in last) + (f" t_step {last.get('t_step')}" if "t_step" in last else ""))  # fmt: skip


def pull(names: list[str]) -> None:
    for name in names:
        ok = vol_get(f"runs/{name}/curve.jsonl", RUNS / name / "curve.jsonl")
        ok2 = vol_get(f"runs/{name}/rollouts.jsonl", DATA / "rollouts" / f"{name}.jsonl")
        print(f"{name}: curve {'ok' if ok else 'missing'}, rollouts {'ok' if ok2 else 'missing'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pilot", "launch", "status", "pull"])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    cfgs = configs()
    if a.step == "pilot":
        spawn([{**cfgs[0], "name": "pilot-full", "steps": 30, "save_every": 30}])
    elif a.step == "launch":
        spawn([c for c in cfgs if not a.names or c["name"] in a.names])
    elif a.step == "status":
        status()
    else:
        pull(a.names or [c["name"] for c in cfgs])


if __name__ == "__main__":
    main()
