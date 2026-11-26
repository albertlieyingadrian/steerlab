"""Run graft_loop.py on Modal, one config per container (COSTS MONEY: L4 about $1.12/h with 4 CPU + 16 GiB).

    export MODAL_PROFILE=albertlieyingadrian
    .venv/bin/modal deploy experiments/graft-monitor/trainer/modal_train.py          # after any code change
    .venv/bin/python experiments/graft-monitor/trainer/modal_train.py launch --set name=x arm=graft steps=200
    .venv/bin/python experiments/graft-monitor/trainer/modal_train.py status x
    .venv/bin/python experiments/graft-monitor/trainer/modal_train.py pull x          # -> results/runs/x/

Money: every launch first checks the trainer's own ledger rows (tagged [trainer] in results/spend.jsonl) plus this
launch's worst case (hard timeout x $/h, or the run's max_usd if lower) against CAP_USD, then logs a projected
estimate. Inside the container the loop stops itself (saving adapters and the deployed model) before its own running
estimate passes `max_usd`. `pull` logs a reconcile row so the ledger sums to the measured wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GM = HERE.parent
REMOTE = "/root/gm"
VOL = "/vol"
APP = "graft-monitor-train"
GPU = ["L4", "A10G"]  # Episode 2 onward: L40S dropped to fit the $100 study cap (Episode 1 also used L40S)
USD_PER_HOUR = 1.10 + 4 * 0.0473 + 16 * 0.0080  # priced at the most expensive entry (A10G) + 4 CPU + 16 GiB
TIMEOUT_S = 3 * 3600  # hard container timeout
CAP_USD = 105.0  # projected estimates; exploration 15 -> 85 (Episode 1) -> 105 (Episode 2 generations); real billing tracked against the $100 study cap
TAG = "[trainer]"
LEDGER = GM / "results" / "spend.jsonl"
PROFILE = "albertlieyingadrian"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.2.0", "accelerate", "safetensors", "openai")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_file(GM / "env.py", f"{REMOTE}/env.py")
    .add_local_file(GM / "policies.json", f"{REMOTE}/policies.json")
    .add_local_dir(GM / "data", f"{REMOTE}/data")
    .add_local_dir(GM / "monitors", f"{REMOTE}/monitors", ignore=["__pycache__"])
    .add_local_dir(HERE, f"{REMOTE}/trainer", ignore=["__pycache__", "tests"])
)
volume = modal.Volume.from_name("graft-monitor", create_if_missing=True)
app = modal.App(APP, image=image)


@app.function(
    gpu=GPU,
    cpu=4.0,
    memory=16384,
    timeout=TIMEOUT_S,
    volumes={VOL: volume},
    retries=0,
    max_containers=6,
    secrets=[modal.Secret.from_name("graft-monitor-vllm")],
)
def train_run(cfg: dict) -> dict:
    # monitors/endpoints.api_key() reads GRAFT_VLLM_API_KEY; the secret sets VLLM_API_KEY (never printed)
    if os.environ.get("VLLM_API_KEY") and not os.environ.get("GRAFT_VLLM_API_KEY"):
        os.environ["GRAFT_VLLM_API_KEY"] = os.environ["VLLM_API_KEY"]
    sys.path.insert(0, REMOTE)  # the `monitors` package
    sys.path.insert(0, f"{REMOTE}/trainer")
    import graft_loop

    cfg = {**cfg, "out_dir": f"{VOL}/runs/{cfg['name']}", "usd_per_hour": USD_PER_HOUR}
    if cfg.get("init_from") and not cfg["init_from"].startswith("/"):
        cfg["init_from"] = f"{VOL}/runs/{cfg['init_from']}/deployed"  # a previous run's name
    try:
        volume.reload()  # pick up a previous generation's deployed model
    except RuntimeError as e:  # a reused container can hold open files ("open files preventing the operation")
        print(f"[train_run] volume.reload skipped: {e}", flush=True)
        if cfg.get("init_from") and not os.path.exists(cfg["init_from"]):
            raise
    out = graft_loop.train(graft_loop.Config(**cfg), on_checkpoint=volume.commit)
    volume.commit()
    return out


# ------------------------------------------------------------------------------------------------ local commands


def _ledger_rows():
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]


def spent() -> float:
    return sum(float(r["usd"]) for r in _ledger_rows() if TAG in r.get("note", ""))


def log_spend(usd: float, note: str):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": "modal_estimate",
                    "usd": round(usd, 4),
                    "note": f"{TAG} {note}",
                }
            )
            + "\n"
        )


def run_dir(name: str) -> Path:
    return GM / "results" / "runs" / name


def parse_cfg(sets):
    sys.path.insert(0, str(HERE))
    import graft_loop  # local: config parsing only

    return graft_loop.parse_config(sets)


def launch(sets: list[str], projected_s_per_step: float):
    cfg = parse_cfg(sets)
    worst = min(TIMEOUT_S / 3600 * USD_PER_HOUR, cfg.max_usd if cfg.max_usd > 0 else 1e9)
    if spent() + worst > CAP_USD:
        raise SystemExit(
            f"cap: trainer spend ${spent():.2f} + this run's worst case ${worst:.2f} > ${CAP_USD:.2f}; refusing"
        )
    d = run_dir(cfg.name)
    d.mkdir(parents=True, exist_ok=True)
    projected = min(worst, (cfg.steps * projected_s_per_step + 300) / 3600 * USD_PER_HOUR)
    fn = modal.Function.from_name(APP, "train_run")
    call = fn.spawn(asdict(cfg))
    (d / "call_id.txt").write_text(call.object_id + "\n")
    (d / "launch.json").write_text(
        json.dumps(
            {
                "call_id": call.object_id,
                "config": asdict(cfg),
                "projected_usd": projected,
                "worst_usd": worst,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
        )
    )
    log_spend(
        projected,
        f"launch {cfg.name} arm={cfg.arm} steps={cfg.steps} projected on {GPU} at "
        f"${USD_PER_HOUR:.2f}/h (worst ${worst:.2f})",
    )
    print(f"spawned {cfg.name}: {call.object_id} (projected ${projected:.2f}, trainer total ${spent():.2f})")


def status(name: str):
    cid = (run_dir(name) / "call_id.txt").read_text().strip()
    try:
        print(json.dumps(modal.FunctionCall.from_id(cid).get(timeout=0)))
    except TimeoutError:
        print(f"{name}: running ({cid})")


def pull(name: str):
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", PROFILE)}
    modal_bin = str(Path(sys.executable).parent / "modal")
    for f in ("log.jsonl", "routing_eval.jsonl", "config.json", "done.json"):
        subprocess.run(
            [modal_bin, "volume", "get", "--force", "graft-monitor", f"runs/{name}/{f}", str(d / f)],
            env=env,
            check=False,
            capture_output=True,
        )
    subprocess.run(
        [modal_bin, "volume", "get", "--force", "graft-monitor", f"runs/{name}/evals", str(d)],
        env=env,
        check=False,
        capture_output=True,
    )
    done, rec = d / "done.json", d / "reconciled.json"
    if done.exists() and not rec.exists():
        wall = json.loads(done.read_text())["wall_s"] + 240  # + image start and model load
        actual = wall / 3600 * USD_PER_HOUR
        projected = json.loads((d / "launch.json").read_text())["projected_usd"]
        log_spend(actual - projected, f"reconcile {name} wall {wall:.0f}s actual ${actual:.2f}")
        rec.write_text(json.dumps({"actual_usd": actual, "projected_usd": projected}))
    print(f"pulled {name} -> {d} (trainer total ${spent():.2f})")


if __name__ == "__main__":
    assert os.environ.get("MODAL_PROFILE") == PROFILE, f"set MODAL_PROFILE={PROFILE}"
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("launch")
    a.add_argument("--set", nargs="*", default=[])
    a.add_argument("--s_per_step", type=float, default=30.0)
    for c in ("status", "pull"):
        sub.add_parser(c).add_argument("name")
    sub.add_parser("spent")
    args = ap.parse_args()
    if args.cmd == "launch":
        launch(args.set, args.s_per_step)
    elif args.cmd == "status":
        status(args.name)
    elif args.cmd == "pull":
        pull(args.name)
    else:
        print(f"${spent():.2f}")
