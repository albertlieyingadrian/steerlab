"""Train and evaluate the toy KV-cache world model on Modal (one A10G per seed, at most two at once).

    MODAL_PROFILE=modal4 modal run modal_app.py::train_seeds --seeds 1,2 --steps 20000
    MODAL_PROFILE=modal4 modal run modal_app.py::eval_seeds --seeds 1,2

Checkpoints and result JSON files are written back into results/ locally; nothing is kept remotely.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("wm-kv-cache")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy<2.1", "scikit-learn==1.5.2")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, "/root/exp", ignore=["results/**", "report/**", "__pycache__/**"])
)


@app.function(image=image, gpu="A10G", timeout=4 * 3600)
def train(seed: int, steps: int) -> dict:
    out = f"/tmp/ckpt_s{seed}.pt"
    subprocess.run(
        ["python", "/root/exp/train.py", "--seed", str(seed), "--steps", str(steps), "--out", out], check=True
    )
    return {"seed": seed, "ckpt": Path(out).read_bytes()}


@app.function(image=image, gpu="A10G", timeout=4 * 3600)
def evaluate(seed: int, ckpt: bytes, args: list[str]) -> dict:
    Path("/tmp/ckpt.pt").write_bytes(ckpt)
    out = f"/tmp/eval_s{seed}.json"
    subprocess.run(["python", "/root/exp/evaluate.py", "--ckpt", "/tmp/ckpt.pt", "--out", out, *args], check=True)
    return {"seed": seed, "json": Path(out).read_text()}


@app.local_entrypoint()
def train_seeds(seeds: str = "1,2", steps: int = 20000, tag: str = ""):
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    for r in train.starmap([(int(s), steps) for s in seeds.split(",")]):
        (res / f"ckpt{tag}_s{r['seed']}.pt").write_bytes(r["ckpt"])
        print("saved", f"results/ckpt{tag}_s{r['seed']}.pt")


@app.local_entrypoint()
def eval_seeds(seeds: str = "1,2", extra: str = "", tag: str = ""):
    res = HERE / "results"
    jobs = [(int(s), (res / f"ckpt{tag}_s{s}.pt").read_bytes(), extra.split()) for s in seeds.split(",")]
    for r in evaluate.starmap(jobs):
        (res / f"eval{tag}_s{r['seed']}.json").write_text(r["json"])
        print("saved", f"results/eval{tag}_s{r['seed']}.json", len(json.loads(r["json"])))
