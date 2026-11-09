"""Run Auditron's own Colab notebook (auditron_colab.ipynb, cells 1-4) on a rented GPU, to show its real output.

COSTS MONEY. Estimate before running: L4 for about 50 min including the image build, about $1.00; the timeout below
caps it at about $2.

    .venv/bin/modal run experiments/oss-auditron/modal_notebook.py                   # the notebook demo (L4)
    .venv/bin/modal run --detach experiments/oss-auditron/modal_notebook.py --replicate-run  # Run 30, A100 80 GB

What it does, cell by cell, as the notebook does (their code runs inside the container; none of it is stored
in this repo):
    cell 1  pip install unsloth trl openenv-core mergekit          -> the image below, pinned to 2026-03-12 versions
    cell 2  git clone github.com/sfgeekgit/OpenEnvNorthflankEnv      -> run(), commit recorded
    cell 3  start auditron_env/server.py in the background (port 7860)
    cell 4  MODEL_NAME=unsloth/Qwen2.5-1.5B-Instruct, 40 GRPO steps, 3 prompt episodes, 1 eval episode; run train.py
Then the logs train.py writes (episodes_*.jsonl, reasoning_*.jsonl, eval_*.json) and its stdout come back to
results/notebook_run/.

Price at list (modal.com/pricing): L4 $0.80/h + 4 CPU cores $0.19 + 16 GiB $0.13 = about $1.12/h.

--replicate-run: the report's Run 30 instead of the demo. Qwen2.5-7B-Instruct (train.py's default), 80 GRPO steps (the
report doesn't state the count; their notes call 80 a "quick run"), every other setting at train.py's defaults. After
their own final game (seed 99999), 5 more full games with the trained model through their evaluate_model (seeds
99900-99904, never trained on). Estimate $6 + $2.5 for the extra games, on an A100 80 GB at about $2.94/h; the
timeout caps it at about $9.80.
"""

import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "notebook_run"
REPO = "https://github.com/sfgeekgit/OpenEnvNorthflankEnv.git"
TIMEOUT_S = 100 * 60  # hard stop: 100 min x $1.12/h = about $1.90
PRICE_PER_HOUR = 0.80 + 4 * 0.0473 + 16 * 0.0080
REPLICATE_TIMEOUT_S = 200 * 60  # hard stop: 200 min x $2.94/h = about $9.80
REPLICATE_PRICE_PER_HOUR = 2.50 + 4 * 0.0473 + 32 * 0.0080
REPLICATE_ENV = {"NUM_TRAINING_STEPS": "80"}
EXTRA_GAMES = 5
EXTRA_EVAL_STEP = 999  # evaluate_model seeds games as eval_step * 100 + i: 99900-99904

# Runs inside the container: their train.py in this process, with GRPOTrainer wrapped so the trained model is kept,
# then their own evaluate_model for the extra games.
DRIVER = f"""
import trl
kept = {{}}
_Trainer = trl.GRPOTrainer
class Keep(_Trainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        kept["model"], kept["tok"] = k["model"], k["processing_class"]
trl.GRPOTrainer = Keep
ns = {{"__name__": "__main__"}}
exec(compile(open("train.py").read(), "train.py", "exec"), ns)
print("=== EXTRA GAMES ===", flush=True)
ns["evaluate_model"](kept["model"], kept["tok"], num_episodes={EXTRA_GAMES}, eval_step={EXTRA_EVAL_STEP}, max_rounds=50)
"""
NOTEBOOK_ENV = {  # cell 4 of the notebook, unchanged
    "MODEL_NAME": "unsloth/Qwen2.5-1.5B-Instruct",
    "NUM_TRAINING_STEPS": "40",
    "NUM_PROMPT_EPISODES": "3",
    "EVAL_EPISODES": "1",
}

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")  # CUDA comes inside the PyTorch wheels
    .apt_install("git", "build-essential")  # git for cell 2; a C compiler for Triton's kernels
    # cell 1 installs unsloth/trl/openenv-core unpinned; today's TRL rejects train.py's GRPOConfig(warmup_ratio=...).
    # So: the full set locked to what was released before the source commit (2026-03-12); mergekit dropped (unused).
    .pip_install_from_requirements(str(HERE / "requirements-auditron-2026-03.txt"))
)
app = modal.App("oss-auditron-notebook", image=image)


def _run(cmd: list[str], env_extra: dict[str, str]) -> dict[str, bytes]:
    import os

    start = time.time()
    subprocess.run(["git", "clone", "-q", REPO, "/root/src"], check=True)
    commit = subprocess.run(
        ["git", "-C", "/root/src", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout
    env_dir = "/root/src/auditron_env"
    server = subprocess.Popen(
        ["python", "server.py"], cwd=env_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(3)
    proc = subprocess.run(
        cmd,
        check=False,  # keep the logs even if training crashes
        cwd=env_dir,
        env={**os.environ, **env_extra, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
    )
    server.terminate()
    files = {
        "stdout.log": proc.stdout.encode(),
        "stderr.log": proc.stderr.encode()[-200_000:],
        "meta.txt": f"commit {commit.strip()}\nexit {proc.returncode}\nseconds {time.time() - start:.0f}\n".encode(),
    }
    for p in Path(env_dir).glob("*"):
        if p.suffix in {".jsonl", ".json"} and p.name.startswith(("episodes_", "reasoning_", "eval_")):
            files[p.name] = p.read_bytes()
    return files


@app.function(gpu="L4", cpu=4, memory=16 * 1024, timeout=TIMEOUT_S)
def run() -> dict[str, bytes]:
    return _run(["python", "train.py"], NOTEBOOK_ENV)


@app.function(gpu="A100-80GB", cpu=4, memory=32 * 1024, timeout=REPLICATE_TIMEOUT_S)
def replicate() -> dict[str, bytes]:
    return _run(["python", "-c", DRIVER], REPLICATE_ENV)


@app.local_entrypoint()
def main(replicate_run: bool = False) -> None:
    t0 = time.time()
    files = (replicate if replicate_run else run).remote()
    out = OUT.parent / ("replication_run30" if replicate_run else "notebook_run")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    hours = (time.time() - t0) / 3600
    price = REPLICATE_PRICE_PER_HOUR if replicate_run else PRICE_PER_HOUR
    print(f"saved {sorted(files)} to {out}")
    print(f"wall time {hours * 60:.0f} min, about ${hours * price:.2f} at list price (check the dashboard)")
