"""Driver for the self-modeling replication. Every stage is launched as one Modal call whose id is saved before waiting.

    .venv/bin/python experiments/repl-self-modeling/run.py calib       # SGLD step on dev seed 100 (MNIST and IMDB)
    .venv/bin/python experiments/repl-self-modeling/run.py mnist       # Figure 2 grid: 4 sizes x 6 AW x 10 seeds
    .venv/bin/python experiments/repl-self-modeling/run.py mnist_ctrl  # detached, other-network and L2 controls
    .venv/bin/python experiments/repl-self-modeling/run.py imdb        # Figure 4: AW 0/100/500 x 10 seeds + controls
    .venv/bin/python experiments/repl-self-modeling/run.py pull        # copy the volume's results into results/runs/
    add --dry-run to print the configs and the cost estimate without launching

Cost: Modal CPU at $0.047 per core-hour and $0.008 per GiB-hour (`modal billing rates`); the cap is in CAP_USD.
"""

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ROOT = HERE.parents[1]
CAP_USD = 15.0
CPU_CONTAINER_PER_H = 2 * 0.047 + 4 * 0.008  # run_one: 2 cores, 4 GiB
os.environ.setdefault("MODAL_PROFILE", "credits30")

CALIB_LRS = {"lr1e-6": 1e-6, "lr3e-6": 3e-6, "lr1e-5": 1e-5, "lr3e-5": 3e-5}
CALIB2_LRS = {"lr1e-4": 1e-4, "lr3e-4": 3e-4, "lr1e-3": 1e-3}  # amendment 1
SGLD = {"gamma": 100.0, "batch": 512, "chains": 4, "burn": 300, "draws": 700}


def llc_settings(task: str) -> dict:
    """Main step from the calibration file, plus one third of it as the sensitivity check."""
    cal = json.loads((RESULTS / "calibration.json").read_text())
    lr = cal[task]["chosen_lr"]
    return {"main": {**SGLD, "lr": lr}, "sens": {**SGLD, "lr": lr / 3}}


def mnist_cfg(hidden, aw, target, seed, l2=0.0, llc=None):
    return {
        "task": "mnist",
        "hidden": hidden,
        "aw": aw,
        "target": target,
        "l2": l2,
        "seed": seed,
        "epochs": 50,
        "batch": 128,
        "lr": 1e-3,
        "llc": llc,
        "run_id": f"mnist_h{hidden}_aw{aw}_{target}_l2{l2}_s{seed}",
    }


def imdb_cfg(aw, target, seed, llc=None):
    return {
        "task": "imdb",
        "hidden": 64,
        "emb": 64,
        "aw": aw,
        "target": target,
        "seed": seed,
        "epochs": 500,
        "batch": 256,
        "lr": 1e-3,
        "llc": llc,
        "checkpoints": [250, 500],
        "run_id": f"imdb_aw{aw}_{target}_s{seed}",
    }


def stage_cfgs(stage: str) -> tuple[list[dict], float]:
    """Configs and the estimated container-hours per run."""
    if stage == "calib":
        llc = {k: {**SGLD, "lr": v} for k, v in CALIB_LRS.items()}
        return [mnist_cfg(512, 0, "none", 100, llc=llc), imdb_cfg(0, "none", 100, llc=llc)], 0.4
    if stage == "calib2":
        llc = {k: {**SGLD, "lr": v} for k, v in CALIB2_LRS.items()}
        return [mnist_cfg(512, 0, "none", 100, llc=llc), imdb_cfg(0, "none", 100, llc=llc)], 0.4
    if stage == "mnist":
        llc = llc_settings("mnist")
        cfgs = [
            mnist_cfg(h, aw, "self" if aw else "none", s, llc=llc)
            for h in (64, 128, 256, 512)
            for aw in (0, 1, 5, 10, 20, 50)
            for s in range(10)
        ]
        return cfgs, 0.05
    if stage == "mnist_ctrl":
        llc = llc_settings("mnist")
        cfgs = [
            mnist_cfg(h, aw, t, s, llc=llc)
            for h in (128, 512)
            for aw in (5, 20)
            for t in ("detached", "other")
            for s in range(10)
        ]
        cfgs += [
            mnist_cfg(h, 0, "none", s, l2=l2, llc=llc) for h in (128, 512) for l2 in (1e-4, 1e-3) for s in range(10)
        ]
        return cfgs, 0.05
    if stage == "imdb":
        llc = llc_settings("imdb")
        cfgs = [imdb_cfg(aw, "self" if aw else "none", s, llc=llc) for aw in (0, 100, 500) for s in range(10)]
        cfgs += [imdb_cfg(aw, t, s, llc=llc) for aw in (100, 500) for t in ("detached", "other") for s in range(10)]
        return cfgs, 0.2
    raise SystemExit(f"unknown stage {stage}")


L4_PER_H = 0.80


def cifar_cfgs() -> list[dict]:
    """Scaled-down Figure 3: AW 0/0.5/1/2 (self) + detached at AW 1, 3 seeds, 30 epochs."""
    llc = {f"lr{v:g}": {"lr": v} for v in (1e-6, 3e-7, 1e-7, 3e-8)}  # plan amendment 2
    arms = [(0, "none"), (0.5, "self"), (1, "self"), (2, "self"), (1, "detached")]
    return [
        {
            "task": "cifar",
            "hidden": 256,
            "aw": aw,
            "target": t,
            "seed": s,
            "epochs": 30,
            "llc": llc,
            "run_id": f"cifar_aw{aw}_{t}_s{s}",
        }
        for s in range(3)
        for aw, t in arms
    ]


def launch_cifar(dry: bool) -> None:
    import modal

    cfgs = cifar_cfgs()
    est = len(cfgs) * 0.3 * L4_PER_H
    total = sum(r.get("usd", r["estimate_usd"]) for r in spend_rows())
    print(f"cifar: {len(cfgs)} runs on one L4, estimate ${est:.2f}; ledger total ${total:.2f} of ${CAP_USD:.0f}")
    if dry:
        return
    if total + est > CAP_USD:
        raise SystemExit("refused: would pass the cap")
    call = modal.Function.from_name("repl-self-modeling-cifar", "run_all").spawn(cfgs)
    rows = spend_rows()
    rows.append(
        {
            "stage": "cifar",
            "n_runs": len(cfgs),
            "estimate_usd": round(est, 3),
            "call_id": call.object_id,
            "launched": dt.datetime.now().isoformat(timespec="seconds"),
            "profile": os.environ["MODAL_PROFILE"],
        }
    )
    (RESULTS / "spend.json").write_text(json.dumps(rows, indent=1))
    print("call id", call.object_id)


def spend_rows() -> list:
    p = RESULTS / "spend.json"
    return json.loads(p.read_text()) if p.exists() else []


def launch(stage: str, dry: bool) -> None:
    import modal

    cfgs, hours_each = stage_cfgs(stage)
    est = len(cfgs) * hours_each * CPU_CONTAINER_PER_H + 0.05
    total = sum(r.get("usd", r["estimate_usd"]) for r in spend_rows())
    print(f"{stage}: {len(cfgs)} runs, estimate ${est:.2f}; ledger total ${total:.2f} of ${CAP_USD:.0f}")
    if dry:
        for c in cfgs[:3]:
            print("  ", c["run_id"])
        return
    if total + est > CAP_USD:
        raise SystemExit("refused: would pass the cap")
    fn = modal.Function.from_name("repl-self-modeling", "orchestrate")
    call = fn.spawn(stage, cfgs)
    rows = spend_rows()
    rows.append(
        {
            "stage": stage,
            "n_runs": len(cfgs),
            "estimate_usd": round(est, 3),
            "call_id": call.object_id,
            "launched": dt.datetime.now().isoformat(timespec="seconds"),
            "profile": os.environ["MODAL_PROFILE"],
        }
    )
    (RESULTS / "spend.json").write_text(json.dumps(rows, indent=1))
    print("call id", call.object_id, "saved to results/spend.json")


def pull() -> None:
    dest = RESULTS / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    for stage in ("calib", "calib2", "mnist", "mnist_ctrl", "imdb", "cifar"):
        cmd = [
            str(Path(sys.executable).parent / "modal"),
            "volume",
            "get",
            "--force",
            "repl-self-modeling",
            f"/{stage}",
            str(dest),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(stage, "ok" if r.returncode == 0 else r.stderr.strip()[:200])


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "pull":
        pull()
    elif args[0] == "cifar":
        launch_cifar("--dry-run" in args)
    else:
        launch(args[0], "--dry-run" in args)
