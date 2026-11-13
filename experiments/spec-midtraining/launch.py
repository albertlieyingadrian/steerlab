"""Upload data, spawn Modal jobs on the deployed app, collect results (see train_modal.py).

python launch.py upload
python launch.py spawn --arms A B C D E --seeds 0 [--smoke]
python launch.py llama
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
LLAMA_ADAPTERS = [
    f"chloeli/llama-3.1-8b-{n}"
    for n in (
        "baseline",
        "cheese-aft",
        "pro-america-spec-msm",
        "pro-affordability-spec-msm",
        "pro-america-spec-msm-cheese-aft",
        "pro-affordability-spec-msm-cheese-aft",
    )
]


def put(local, remote):
    subprocess.run([MODAL, "volume", "put", "-f", "spec-midtraining", str(local), remote], check=True)


def upload():
    put(HERE / "data" / "evals.jsonl", "data/evals.jsonl")
    put(HERE / "data" / "aft.jsonl", "data/aft.jsonl")
    for f in sorted((HERE / "data" / "corpus").glob("msm_*.jsonl")):
        put(f, f"data/corpus/{f.name}")


def record(kind, args, call):
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a") as f:
        f.write(
            json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "args": args, "call_id": call.object_id})
            + "\n"
        )


def spawn(arms, seeds, smoke):
    fn = modal.Function.from_name("spec-midtraining", "run")
    for s in seeds:
        for a in arms:
            c = fn.spawn(a, s, smoke)
            record("run", [a, s, smoke], c)
            print("spawned", a, s, c.object_id, flush=True)


def llama():
    c = modal.Function.from_name("spec-midtraining", "eval_llama").spawn(LLAMA_ADAPTERS)
    record("eval_llama", LLAMA_ADAPTERS, c)
    print("spawned eval_llama", c.object_id)


def interp(name, family, seed, null_seed):
    import train_modal as T

    roles = T.LLAMA_ROLES if family == "llama" else T.qwen_roles(seed, null_seed)
    layers = [4, 8, 12, 16, 20, 24, 28] if family == "llama" else [4, 8, 12, 16, 20, 24, 28, 32]
    c = modal.Function.from_name("spec-midtraining", "interp_job").spawn(name, family, roles, layers)
    record("interp_job", [name, family, roles, layers], c)
    print("spawned interp", name, c.object_id)


def collect():
    vol = modal.Volume.from_name("spec-midtraining")
    for remote, local in (("results", HERE / "results" / "runs"), ("smoke", HERE / "results" / "smoke")):
        local.mkdir(parents=True, exist_ok=True)
        try:
            entries = vol.listdir(remote)
        except FileNotFoundError:  # e.g. no smoke runs on this volume
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
    ap.add_argument("cmd", choices=["upload", "spawn", "llama", "collect", "interp"])
    ap.add_argument("--name", default="llama")
    ap.add_argument("--family", default="llama")
    ap.add_argument("--null-seed", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "interp":
        interp(a.name, a.family, a.seeds[0], a.null_seed)
    else:
        {"upload": upload, "llama": llama, "collect": collect}.get(a.cmd, lambda: spawn(a.arms, a.seeds, a.smoke))()
