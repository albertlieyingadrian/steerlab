"""
The H100 run: the laptop run's GRPO loop (7_train_laptop.py), scaled up on a rented H100 through Modal
(https://modal.com).

COSTS MONEY. Run it only with a budget in mind (see the README for the cost maths).

    # one-time: a Modal account and `modal setup` (writes ~/.modal.toml); nothing to put in .env
    .venv/bin/modal run experiments/pigment-gym/modal_train.py --steps 5 --run-name smoke
    .venv/bin/modal run --detach experiments/pigment-gym/modal_train.py --steps 210 --run-name full

Re-running a command with the same --run-name resumes from that run's last checkpoint (every 10 steps) on the
Volume; add --fresh to discard it. --detach keeps the H100 job running if your laptop sleeps or disconnects.

What Modal does here, in software terms:
  * `image`     = a Dockerfile written in Python: Debian + CUDA PyTorch + our libraries + headless Chromium,
                  with this folder copied in. Built once, cached.
  * `volume`    = a network disk that outlives the container: the Hugging Face model cache (so the 3 GB model
                  downloads once) and every run's outputs (curve, rollouts, LoRA adapter).
  * `@app.function(gpu="H100", timeout=...)` = "run this Python function on a machine with an H100". The
                  timeout is a hard stop: Modal kills the container at that point, so it caps the bill.
  * `@app.local_entrypoint()` = runs on your laptop, calls the remote function, and saves the small outputs
                  into the work folder <run-name>/.

Cost per hour at list price (modal.com/pricing, read 2026-10-24): H100 $3.95 + 8 CPU cores $0.38
+ 32 GiB memory $0.26 = about $4.59/h. The 3.5 h timeout caps one run at about $16.
"""

import json
import os
import sys
from pathlib import Path

import modal

PART_DIR = Path(__file__).resolve().parent  # experiments/pigment-gym
REMOTE_PART = "/root/pigment-gym"
VOLUME_PATH = "/vol"
TIMEOUT_S = int(3.5 * 3600)  # hard cap per run: 3.5 h x $4.59/h = ~$16
PRICE_PER_HOUR = 3.95 + 8 * 0.0473 + 32 * 0.0080  # GPU + CPU cores + GiB memory, per hour

# An explicit Debian 12 base: on accounts still on Modal's legacy image builder, debian_slim() is Debian 11,
# which Playwright 1.64 refuses ("Playwright does not support chromium on debian11-x64").
image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "open_clip_torch==3.3.0",
        "playwright==1.64.1",
        "numpy",
        "pillow",
    )
    .run_commands("playwright install --with-deps chromium")
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PART_DIR, REMOTE_PART, ignore=["**/__pycache__", "results", "report"])
)
volume = modal.Volume.from_name("pigment-gym", create_if_missing=True)
app = modal.App("pigment-gym", image=image)


@app.function(gpu="H100", cpu=8.0, memory=32768, timeout=TIMEOUT_S, volumes={VOLUME_PATH: volume})
def train(steps: int, run_name: str, model_name: str, learning_rate: float, warmup: int, fresh: bool = False):
    # Use the cached models once a previous run has downloaded them (set before Hugging Face is imported).
    warm_marker = Path(VOLUME_PATH) / "hf" / ".warm"
    if warm_marker.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path[:0] = [REMOTE_PART]
    from train_loop import train as train_loop

    lines = []

    def log(message=""):
        print(message, flush=True)
        lines.append(str(message))

    out_dir = Path(VOLUME_PATH) / "outputs" / run_name
    train_loop(  # writes summary.json and the rest into out_dir; returned to the laptop below
        out_dir,
        steps=steps,
        model_name=model_name,
        learning_rate=learning_rate,
        warmup=warmup,
        budget_s=TIMEOUT_S - 600,  # 10 min margin for container start, model loading and the final upload
        reserve_s=900,
        pool_size=6,
        on_checkpoint=volume.commit,
        log=log,
        resume=not fresh,
    )
    warm_marker.parent.mkdir(parents=True, exist_ok=True)
    warm_marker.touch()
    (out_dir / "log.txt").write_text("\n".join(lines) + "\n")
    volume.commit()
    small = {}
    for name in [
        "summary.json",
        "curve.csv",
        "log.txt",
        "best_after.js",
        "before.png",
        "after.png",
        "before_samples.jsonl",
        "after_samples.jsonl",
        "rollouts.jsonl",
    ]:
        if (out_dir / name).exists():
            small[name] = (out_dir / name).read_bytes()
    return small


@app.local_entrypoint()
def main(
    steps: int = 5,
    run_name: str = "smoke",
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    learning_rate: float = 1e-4,
    warmup: int = 5,
    fresh: bool = False,
):
    # Re-running the same --run-name resumes from its last checkpoint on the Volume (every 10 steps);
    # --fresh discards it and starts clean.
    files = train.remote(steps, run_name, model_name, learning_rate, warmup, fresh)
    local = Path.home() / ".cache" / "steerlab" / "pigment-gym" / f"h100_{run_name}"
    local.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (local / name).write_bytes(data)
    summary = json.loads(files["summary.json"])
    hours = summary["wall_seconds"] / 3600
    print(f"\nsaved {len(files)} files to {local}")
    print(
        f"training function wall time {summary['wall_seconds'] / 60:.1f} min -> about ${hours * PRICE_PER_HOUR:.2f} "
        f"at list price (${PRICE_PER_HOUR:.2f}/h; image build and container start-up are extra and small)"
    )
    print(f"LoRA adapter: .venv/bin/modal volume get pigment-gym outputs/{run_name}/lora {local}/lora")
