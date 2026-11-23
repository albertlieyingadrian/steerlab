"""
The full-size run of the SAE 101 tutorial on Modal: one A10G per step, the same scripts as the laptop run.

    MODAL_PROFILE=modal4 modal deploy experiments/sae-101/modal_app.py      # once, and after any code change
    python experiments/sae-101/modal_app.py spawn tokens                    # then each step in order (see STEPS)
    python experiments/sae-101/modal_app.py ledger                          # what has been spent
    MODAL_PROFILE=modal4 modal volume get sae-101 results experiments/sae-101/results/full_remote

Each step runs one numbered script with SAE101_WORK, SAE101_RESULTS and SAE101_RECORDED pointing at the volume, so
token windows, checkpoints, results and the printed output survive the container. Spend is capped in code: every step
writes its start (with its worst case, timeout x hourly rate) and its end (actual seconds x rate) to a ledger on the
volume, and refuses to start if the actual spend plus every running step's worst case plus its own would pass the cap.
Retries are off, so a failed step is never re-billed silently, and containers retire 10 s after a call, so a redeploy
is never served by a warm container holding the old code (that happened once: an evaluate retry ran the old script).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP, VOLUME = "sae-101", "sae-101"
CAP_USD = 15.0
RATE = {"A10G": 1.10, "A10G-long": 1.10, "cpu": 0.25}  # $/hour: Modal's A10G list price; a 4-core, 16 GB CPU container, rounded up
STEPS = {  # name: (script, extra args, where, timeout in minutes)
    "tokens": ("02_collect.py", ["--tokens-only"], "cpu", 60),
    "collect": ("02_collect.py", [], "A10G", 30),
    "pilot": ("03a_train_handwritten.py", ["--max-steps", "600"], "A10G", 20),
    "train_handwritten": ("03a_train_handwritten.py", [], "A10G", 100),
    "train_saelens": ("03b_train_saelens.py", [], "A10G-long", 180),  # SAELens runs at about 13k tokens/s here
    "evaluate": ("04_evaluate.py", [], "A10G", 45),
    "sweep": ("03c_sweep.py", [], "A10G", 60),
    "interpret": ("05_interpret.py", [], "A10G", 45),
    "compare": ("06_compare.py", [], "A10G", 30),
    "reload": ("07c_reload_check.py", [], "A10G", 25),
}

app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "transformers==5.24.0",
        "datasets==5.2.0",
        "sae-lens==6.56.1",
        "safetensors==0.8.0",
        "huggingface-hub==2.4.0",
        "numpy==2.5.5",
        "scikit-learn==1.9.1",
        "matplotlib==3.11.2",
    )
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_DISABLE_XET": "1", "SAE101_WORK": "/vol/work", "SAE101_RESULTS": "/vol/results", "SAE101_RECORDED": "/vol/recorded"})
    .add_local_dir(HERE, "/root/sae101", ignore=["work", "results", "recorded", "report", "__pycache__", "*.pyc"])
)
LEDGER_DIR = Path("/vol/ledger")  # one file per run: two containers never write the same file


def _ledger_rows() -> list[dict]:
    rows = []
    for path in sorted(LEDGER_DIR.glob("*.jsonl")) + ([Path("/vol/ledger.jsonl")] if Path("/vol/ledger.jsonl").exists() else []):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def _committed_spend(rows: list[dict]) -> float:
    """Actual cost of finished steps plus the worst case of steps that started and have not finished."""
    ends = {r["run"]: r for r in rows if r["event"] == "end"}
    total = 0.0
    for r in rows:
        if r["event"] == "start":
            total += ends[r["run"]]["cost"] if r["run"] in ends else r["worst_case"]
    return total


def _run(step: str, where: str) -> dict:
    script, extra, _, timeout_min = STEPS[step]
    try:
        volume.reload()  # see other steps' ledger rows; fails harmlessly when this container holds open files
    except RuntimeError as exc:
        print(f"[modal_app] volume reload skipped: {exc}", flush=True)
    rows = _ledger_rows()
    worst = timeout_min / 60 * RATE[where]
    committed = _committed_spend(rows)
    if committed + worst > CAP_USD:
        raise RuntimeError(f"refusing {step}: committed ${committed:.2f} + worst case ${worst:.2f} > cap ${CAP_USD}")
    run_id = f"{step}-{int(time.time())}"
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger = LEDGER_DIR / f"{run_id}.jsonl"
    with ledger.open("a") as f:
        f.write(json.dumps({"event": "start", "run": run_id, "step": step, "where": where, "t": time.time(),
                            "worst_case": worst}) + "\n")
    volume.commit()

    stop = threading.Event()

    def commit_every(seconds=120):  # so checkpoints survive a dropped container
        while not stop.wait(seconds):
            try:
                volume.commit()
            except Exception as exc:  # a failed commit must not kill training
                print(f"[modal_app] commit failed: {exc}", flush=True)

    threading.Thread(target=commit_every, daemon=True).start()
    log_dir = Path("/vol/recorded/full")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{Path(script).stem}{'_pilot' if step == 'pilot' else ''}{'_tokens' if step == 'tokens' else ''}.txt"
    t0 = time.time()
    cmd = [sys.executable, script, "--size", "full", *extra]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd="/root/sae101", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # The cap counted this step at its STEPS timeout, so stop it there: otherwise it could run on to the Modal
        # function's own (longer) timeout and cost more than the worst case the ledger reserved for it.
        watchdog = threading.Timer(timeout_min * 60, proc.kill)
        watchdog.daemon = True
        watchdog.start()
        for line in proc.stdout:
            print(line, end="", flush=True)
            if not any(skip in line for skip in ("it/s]", "s/it]", "Warning", "warn(")):
                log.write(line)
                log.flush()
        code = proc.wait()
        watchdog.cancel()
    seconds = time.time() - t0
    stop.set()
    timed_out = code != 0 and seconds >= timeout_min * 60
    cost = seconds / 3600 * RATE[where]
    with ledger.open("a") as f:
        f.write(json.dumps({"event": "end", "run": run_id, "step": step, "seconds": round(seconds, 1),
                            "cost": round(cost, 4), "exit": code, "timed_out": timed_out}) + "\n")
    volume.commit()
    if timed_out:
        raise RuntimeError(f"{script} stopped at its {timeout_min}-minute STEPS limit; see "
                           f"/vol/recorded/full/{log_path.name}")
    if code != 0:
        raise RuntimeError(f"{script} exited with {code}; see /vol/recorded/full/{log_path.name}")
    return {"step": step, "seconds": seconds, "cost": cost}


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=60 * 60, retries=0, scaledown_window=10)
def cpu_step(step: str) -> dict:
    if step == "tokens":  # fetch both models' weights on a cheap CPU machine before any GPU starts
        from huggingface_hub import snapshot_download

        for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"):
            snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "*.txt"])
        volume.commit()
    return _run(step, "cpu")


@app.function(image=image, volumes={"/vol": volume}, gpu="A10G", timeout=100 * 60, retries=0, scaledown_window=10)
def gpu_step(step: str) -> dict:
    return _run(step, "A10G")


@app.function(image=image, volumes={"/vol": volume}, gpu="A10G", timeout=180 * 60, retries=0, scaledown_window=10)
def gpu_long(step: str) -> dict:
    return _run(step, "A10G-long")


# ----------------------------------------------------------------------------------------- local launcher
CALLS = HERE / "work" / "modal_calls.jsonl"

if __name__ == "__main__":
    os.environ.setdefault("MODAL_PROFILE", "modal4")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "spawn":
        step = sys.argv[2]
        where = STEPS[step][2]
        fn = modal.Function.from_name(APP, {"cpu": "cpu_step", "A10G": "gpu_step", "A10G-long": "gpu_long"}[where])
        call = fn.spawn(step)
        CALLS.parent.mkdir(parents=True, exist_ok=True)
        with CALLS.open("a") as f:  # save the call id before anything else (it is the only handle on the run)
            f.write(json.dumps({"step": step, "call_id": call.object_id, "t": time.time()}) + "\n")
        print(f"spawned {step}: {call.object_id}")
    elif cmd == "wait":
        print(json.dumps(modal.FunctionCall.from_id(sys.argv[2]).get(timeout=None)))
    elif cmd == "ledger":
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["modal", "volume", "get", "--force", VOLUME, "ledger", tmp], capture_output=True)
            subprocess.run(["modal", "volume", "get", "--force", VOLUME, "ledger.jsonl", tmp], capture_output=True)
            rows = [json.loads(line) for p in Path(tmp).rglob("*.jsonl") for line in p.read_text().splitlines()
                    if line.startswith("{")]
        for r in rows:
            print(r)
        print(f"committed (actual + worst case of running): ${_committed_spend(rows):.2f} of ${CAP_USD}")
    else:
        print(__doc__)
