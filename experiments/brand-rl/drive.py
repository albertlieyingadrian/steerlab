"""Unattended driver for the confirmatory runs: respawn a failed run (it resumes from its latest checkpoint), then,
when every run is done, the held-out evaluation, best-of-8, the screenshot probe, and the local pulls.

    MODAL_PROFILE=albertlieyingadrian nohup .venv/bin/python experiments/brand-rl/drive.py configs/main.json \
        > ~/.cache/steerlab/brand-rl/drive.log 2>&1 &

Every 60 s it checks each run's spawned call (launch.py's results/calls/<run>.json). A call that raised is respawned
with the same config (run_train finds the latest ckpt_N on the volume, trims the log and continues); a local network
error counts as "unknown", never as a failure (Sounds Like Me Ep 3's driver once resumed on a DNS blip). At most
MAX_RESPAWNS per run. When all runs are done: the dev-best step per run (plan rule, from the dev evals), then held-out
jobs (base x4 with screenshots, base x8 for best-of-8, each run's final and dev-best x4 with screenshots), then the
SigLIP probe on every screenshot job, then everything is pulled to ~/.cache/steerlab/brand-rl/.
"""

from __future__ import annotations

import gzip
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import budget
import ep6
from launch import CALLS, call_state, load_runs, spawn

MODAL = str(Path(sys.executable).parent / "modal")
MAX_RESPAWNS = 4
STATE = HERE / "results" / "driver_state.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def vol_get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [MODAL, "volume", "get", "taste-ep6", remote, str(local), "--force"], capture_output=True, text=True
    )
    return r.returncode == 0 and local.exists()


def vol_ls(remote: str) -> list[str]:
    r = subprocess.run([MODAL, "volume", "ls", "taste-ep6", remote], capture_output=True, text=True)
    return r.stdout.split() if r.returncode == 0 else []


def wait_runs(runs: list[dict]) -> None:
    respawns = {r["name"]: 0 for r in runs}
    pending = {r["name"]: r for r in runs}
    while pending:
        for name, cfg in list(pending.items()):
            cid = json.loads((CALLS / f"{name}.json").read_text())["call_id"]
            st = call_state(cid)
            if st == "done":
                log(name, "done")
                pending.pop(name)
            elif st == "failed":
                if respawns[name] >= MAX_RESPAWNS:
                    log(name, "FAILED too often; left for a manual look")
                    pending.pop(name)
                    continue
                respawns[name] += 1
                budget.check_modal(5.0, f"respawn {name}")
                log(name, "failed; respawned (resumes from its latest checkpoint):", spawn(cfg))
        STATE.write_text(json.dumps({"pending": sorted(pending), "respawns": respawns}, indent=1))
        if pending:
            time.sleep(60)


def dev_rows(run: str, step: int) -> list[dict] | None:
    p = ep6.CACHE / "runs" / run / "dev" / f"step_{step}.pkl.gz"
    if p.exists() or vol_get(f"runs/{run}/dev/step_{step}.pkl.gz", p):
        return pickle.load(gzip.open(p))["rows"]
    return None


def dev_best(run: str, steps: int, every: int) -> tuple[int, str]:
    """Plan rule: highest mean arm reward among dev steps with strict fidelity >= step-0 value - 0.05 and
    invalid + truncated + empty + loop share <= 0.10; else the step with the highest strict fidelity."""
    rows = {st: r for st in range(0, steps + 1, every) if (r := dev_rows(run, st)) is not None}

    def mean(rs, k):
        return sum(float(r.get(k) or 0) for r in rs) / len(rs)

    def bad(rs):
        return sum((not r["valid"]) or r["truncated"] or r["empty"] or r["loop"] for r in rs) / len(rs)

    f0 = mean(rows[0], "fidelity_strict")
    ok = [st for st in rows if st > 0 and mean(rows[st], "fidelity_strict") >= f0 - 0.05 and bad(rows[st]) <= 0.10]
    if ok:
        return max(ok, key=lambda st: mean(rows[st], "arm_reward")), "rule"
    return max((st for st in rows if st > 0), key=lambda st: mean(rows[st], "fidelity_strict")), "fallback"


def run_jobs(jobs: list[dict]) -> int:
    """Run the sample jobs not yet on the volume; returns how many raised."""
    import modal

    have = " ".join(vol_ls("samples"))
    todo = [j for j in jobs if f"{j['tag']}.pkl.gz" not in have]
    if not todo:
        return 0
    budget.check_modal(0.6 * len(todo), f"{len(todo)} held-out sample jobs")
    f = modal.Function.from_name("taste-ep6", "sample_remote")
    failed = 0
    for r in f.map(todo, order_outputs=False, return_exceptions=True):
        log("sample job:", r)
        failed += isinstance(r, BaseException)
    return failed


def main() -> None:
    runs = load_runs(HERE / sys.argv[1], "")
    wait_runs(runs)
    picks = {}
    for r in runs:
        st, how = dev_best(r["name"], r["steps"], r.get("eval_every", 20))
        picks[r["name"]] = {"step": st, "how": how}
        log(r["name"], "dev-best step", st, how)
    (HERE / "results" / "dev_best.json").write_text(json.dumps(picks, indent=1))
    jobs = [
        {"tag": "base__heldout", "split": "heldout", "n": 4, "seed": 31, "shots": True},
        {"tag": "base__heldout_n8", "split": "heldout", "n": 8, "seed": 32},
    ]
    for r in runs:
        for st in sorted({r["steps"], picks[r["name"]]["step"]}):
            jobs.append(
                {
                    "tag": f"{r['name']}__step{st}__heldout",
                    "split": "heldout",
                    "n": 4,
                    "seed": 31,
                    "shots": True,
                    "adapter": f"runs/{r['name']}/ckpt_{st}/adapter",
                    "lora_rank": r["lora_rank"],
                }
            )
    if run_jobs(jobs):
        log("some held-out sample jobs raised; re-running the missing ones once")
        run_jobs(jobs)  # skips every tag already on the volume
    import modal

    problems = []  # anything missing makes the run end as DRIVER INCOMPLETE (exit 1), never DRIVER DONE
    shot_tags = [j["tag"] for j in jobs if j.get("shots")]
    sig = modal.Function.from_name("taste-ep6", "siglip_remote")
    budget.check_modal(1.0, "siglip screenshot probe")
    for r in sig.map([shot_tags[i::4] for i in range(4)], order_outputs=False, return_exceptions=True):
        log("siglip:", r)
        if isinstance(r, BaseException):
            problems.append(f"siglip shard: {r!r}")
    for t in [*shot_tags, "base__heldout_n8"]:
        if not vol_get(f"samples/{t}.pkl.gz", ep6.CACHE / "samples" / f"{t}.pkl.gz"):
            problems.append(f"samples/{t}.pkl.gz")
        if t in shot_tags and not vol_get(f"samples/{t}.sig.pkl.gz", ep6.CACHE / "samples" / f"{t}.sig.pkl.gz"):
            problems.append(f"samples/{t}.sig.pkl.gz")
    for r in runs:
        if not vol_get(f"runs/{r['name']}/log.jsonl", ep6.CACHE / "runs" / r["name"] / "log.jsonl"):
            problems.append(f"runs/{r['name']}/log.jsonl")
        for st in range(0, r["steps"] + 1, r.get("eval_every", 20)):
            if dev_rows(r["name"], st) is None:
                problems.append(f"runs/{r['name']}/dev/step_{st}.pkl.gz")
    if problems:
        log("DRIVER INCOMPLETE; missing or failed:", *problems)
        sys.exit(1)
    log("DRIVER DONE")


if __name__ == "__main__":
    main()
