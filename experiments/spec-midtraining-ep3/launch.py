"""Upload data, spawn Modal jobs on the deployed Episode 3 app, collect results (see train_modal.py).

python launch.py upload
python launch.py spawn --arms S P C --seeds 0 [--smoke]
python launch.py interp --name qwen_s0 --seeds 0 --null-seed 1
python launch.py seedcheck
python launch.py extra --arms S P C I --seeds 0 1 2 3   # amendment 2: the conflict set on saved adapters
python launch.py collect
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MODAL_PROFILE", "albertlieyingadrian")
import modal

HERE = Path(__file__).resolve().parent
CALLS = HERE / "results" / "modal_calls.jsonl"
MODAL = str(Path(os.sys.executable).parent / "modal")
APP, VOLUME = "spec-midtraining-ep3", "spec-midtraining"


def put(local, remote):
    subprocess.run([MODAL, "volume", "put", "-f", VOLUME, str(local), f"ep3/{remote}"], check=True)


def upload():
    put(HERE / "data" / "evals.jsonl", "data/evals.jsonl")
    put(HERE / "data" / "aft.jsonl", "data/aft.jsonl")
    if (HERE / "data" / "evals_conflict.jsonl").exists():
        put(HERE / "data" / "evals_conflict.jsonl", "data/evals_conflict.jsonl")
    for name in ("msm_sust", "msm_prac"):
        put(HERE / "data" / "corpus" / f"{name}.jsonl", f"data/corpus/{name}.jsonl")


def record(kind, args, call):
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a") as f:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "args": args, "call_id": call.object_id}
        f.write(json.dumps(rec) + "\n")


def spawn(arms, seeds, smoke):
    fn = modal.Function.from_name(APP, "run")
    for s in seeds:
        for a in arms:
            c = fn.spawn(a, s, smoke)
            record("run", [a, s, smoke], c)
            print("spawned", a, s, c.object_id, flush=True)


def extra(arms, seeds):
    fn = modal.Function.from_name(APP, "eval_extra")
    for s in seeds:
        for a in arms:
            c = fn.spawn(a, s)
            record("eval_extra", [a, s], c)
            print("spawned eval_extra", a, s, c.object_id, flush=True)


def interp(name, seed, null_seed):
    import train_modal as T

    roles = T.qwen_roles(seed, null_seed)
    layers = [4, 8, 12, 16, 20, 24, 28, 32]
    c = modal.Function.from_name(APP, "interp_job").spawn(name, roles, layers)
    record("interp_job", [name, roles, layers], c)
    print("spawned interp", name, c.object_id)


def seedcheck():
    c = modal.Function.from_name(APP, "seed_check_job").spawn()
    record("seed_check_job", [], c)
    print("spawned seed_check", c.object_id)


def collect():
    vol = modal.Volume.from_name(VOLUME)
    pairs = (
        ("ep3/results", HERE / "results" / "runs"),
        ("ep3/results_conflict", HERE / "results" / "runs_conflict"),
        ("ep3/smoke", HERE / "results" / "smoke"),
    )
    for remote, local in pairs:
        local.mkdir(parents=True, exist_ok=True)
        try:
            entries = vol.listdir(remote)
        except (FileNotFoundError, modal.exception.NotFoundError):
            print("no", remote, "folder on the volume")
            continue
        for e in entries:
            dest = local / Path(e.path).name
            if dest.exists() and dest.stat().st_size == e.size:
                continue
            dest.write_bytes(b"".join(vol.read_file(e.path)))
        print(remote, len(entries), "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "spawn", "collect", "interp", "seedcheck", "extra"])
    ap.add_argument("--name", default="qwen_s0")
    ap.add_argument("--null-seed", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "interp":
        interp(a.name, a.seeds[0], a.null_seed)
    elif a.cmd == "spawn":
        spawn(a.arms, a.seeds, a.smoke)
    elif a.cmd == "extra":
        extra(a.arms, a.seeds)
    else:
        {"upload": upload, "collect": collect, "seedcheck": seedcheck}[a.cmd]()
