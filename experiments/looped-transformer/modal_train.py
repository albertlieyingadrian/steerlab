"""Run train.py on Modal GPUs, one container per (model, seed), and copy the results back into results/runs/.

COSTS MONEY: one L4 per job (about $0.80/hour; A10G about $1.10 or T4 about $0.59 when no L4 is free) while it trains; an ephemeral app, no keep-warm, it stops when the
last job returns. Check with `modal app list` afterwards.

    modal run experiments/looped-transformer/modal_train.py --models standard4,looped --seeds 0,1,2 [--steps 20000]
    modal run experiments/looped-transformer/modal_train.py --task inside --runs looped_s0,standard4_s0
    modal run experiments/looped-transformer/modal_train.py --task control --runs looped_h16_s1,looped_s2

The second form uploads saved checkpoints from results/runs/ and runs inside.py (the probes) on Modal too, writing
results/inside_<run>.json. At most 3 containers at once.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "numpy")
    .add_local_dir(str(HERE), "/exp", ignore=["results", "report", "__pycache__", "*.pyc"])
)
app = modal.App("looped-transformer", image=image)
ckpt = modal.Volume.from_name("looped-transformer-ckpt", create_if_missing=True)  # checkpoints for packed jobs


@app.function(gpu=["L4", "A10G", "T4"], timeout=3 * 60 * 60, max_containers=3)
def run(model: str, seed: int, steps: int, extra: str = "") -> dict:
    t0 = time.time()
    cmd = ["python", "/exp/train.py", "--model", model, "--seed", str(seed), "--steps", str(steps), "--out", "/out"]
    cmd += extra.split() if extra else []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:  # echo progress so `modal app logs` shows it
        print(f"[{model}_s{seed}{extra}] {line}", end="", flush=True)
        lines.append(line)
    p.wait()
    p.stdout = "".join(lines)
    p.stderr = ""
    parts = extra.split() if extra else []
    tag = parts[parts.index("--tag") + 1] if "--tag" in parts else ""
    name = f"{model}{tag}_s{seed}"
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    res = {"name": name, "log": p.stdout + p.stderr, "seconds": time.time() - t0, "ok": p.returncode == 0, "gpu": gpu}
    if p.returncode == 0:
        res["json"] = Path(f"/out/{name}.json").read_text()
        res["pt"] = Path(f"/out/{name}.pt").read_bytes()
    return res


@app.function(gpu=["L4", "A10G", "T4"], timeout=60 * 60, max_containers=3)
def probe(run_name: str, pt: bytes, script: str = "inside") -> dict:
    """script: "inside" (probes, inside.py) or "control" (the twin-distractor control set, eval_control.py)."""
    t0 = time.time()
    Path("/in").mkdir(exist_ok=True)
    Path(f"/in/{run_name}.pt").write_bytes(pt)
    p = subprocess.run(
        [
            "python",
            f"/exp/{'inside' if script == 'inside' else 'eval_control'}.py",
            "--runs",
            run_name,
            "--runs-dir",
            "/in",
            "--out",
            "/in",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    res = {
        "name": run_name,
        "log": p.stdout + p.stderr,
        "seconds": time.time() - t0,
        "ok": p.returncode == 0,
        "gpu": gpu,
    }
    if p.returncode == 0:
        res["json"] = Path(f"/in/{script}_{run_name}.json").read_text()
    return res


@app.function(gpu=["L4", "A10G", "T4"], timeout=3 * 60 * 60, volumes={"/ckpt": ckpt})
def packed(script: str, names: list[str]) -> dict:
    """Several runs in one container, one after another (the workspace's container cap is shared). Checkpoints are
    read from the looped-transformer-ckpt volume (upload with `modal volume put looped-transformer-ckpt <dir> /`)."""
    t0 = time.time()
    Path("/out").mkdir(exist_ok=True)
    cmd = [
        "python",
        f"/exp/{'inside' if script == 'inside' else 'eval_control'}.py",
        "--runs",
        *names,
        "--runs-dir",
        "/ckpt",
        "--out",
        "/out",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    files = {f.name: f.read_text() for f in Path("/out").glob(f"{script}_*.json")}
    return {
        "log": p.stdout + p.stderr,
        "ok": p.returncode == 0,
        "seconds": time.time() - t0,
        "gpu": gpu,
        "files": files,
    }


@app.local_entrypoint()
def main(
    models: str = "",
    seeds: str = "0,1,2",
    steps: int = 20000,
    out: str = "",
    extra: str = "",
    task: str = "train",
    runs: str = "",
    runs_dir: str = "",
):
    if task.startswith("packed-"):
        script = task.removeprefix("packed-")
        dest = Path(out) if out else HERE / "results"
        r = packed.remote(script, runs.split(","))
        for name, text in r["files"].items():
            (dest / name).write_text(text)
        (dest / f"{script}_packed.log").write_text(r["log"])
        with (HERE / "results" / "runs" / "modal_ledger.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": f"packed_{script}_{len(runs.split(','))}runs",
                        "gpu": r["gpu"],
                        "seconds": round(r["seconds"], 1),
                        "ok": r["ok"],
                    }
                )
                + "\n"
            )
        print(
            script,
            "ok" if r["ok"] else "FAILED",
            f"{r['seconds']:.0f}s",
            len(r["files"]),
            "files",
            r["log"].strip().splitlines()[-1:],
        )
        return
    if task in ("inside", "control"):
        src = Path(runs_dir) if runs_dir else HERE / "results" / "runs"
        dest = Path(out) if out else HERE / "results"
        names = runs.split(",")
        jobs = [(n, (src / f"{n}.pt").read_bytes(), task) for n in names]
        for r in probe.starmap(jobs, return_exceptions=True, order_outputs=False):
            if isinstance(r, Exception):
                print("FAILED", r)
                continue
            (dest / f"{task}_{r['name']}.log").write_text(r["log"])
            if r["ok"]:
                (dest / f"{task}_{r['name']}.json").write_text(r["json"])
            with (src / "modal_ledger.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "name": f"{task}_" + r["name"],
                            "gpu": r["gpu"],
                            "seconds": round(r["seconds"], 1),
                            "ok": r["ok"],
                        }
                    )
                    + "\n"
                )
            print(
                "inside",
                r["name"],
                "ok" if r["ok"] else "FAILED",
                f"{r['seconds']:.0f}s",
                r["log"].strip().splitlines()[-1:],
            )
        return
    dest = Path(out) if out else HERE / "results" / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [(m, int(s), steps, extra) for m in models.split(",") for s in seeds.split(",")]
    ledger = dest / "modal_ledger.jsonl"
    for r in run.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", r)
            continue
        (dest / f"{r['name']}.log").write_text(r["log"])
        if r["ok"]:
            (dest / f"{r['name']}.json").write_text(r["json"])
            (dest / f"{r['name']}.pt").write_bytes(r["pt"])
        with ledger.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": r["name"],
                        "gpu": r["gpu"],
                        "seconds": round(r["seconds"], 1),
                        "ok": r["ok"],
                        "steps": steps,
                    }
                )
                + "\n"
            )
        print(r["name"], "ok" if r["ok"] else "FAILED", f"{r['seconds']:.0f}s", r["log"].strip().splitlines()[-2:])
