"""Run the whole pipeline for one model on one Modal H100: known -> direction -> steer -> summarize -> fluency.

COSTS MONEY: one call, a 2-hour hard timeout (at most about $8 at H100 list price; estimate ~$3.5, see costs.md).
Cells are written to a Modal volume as they finish, so a retried call resumes instead of starting over. The call
returns results/<model>/ as a tar archive, unpacked here.

    modal run experiments/eval-aware-steer/modal_run.py --model llama3.1-8b --policy motive
    modal run experiments/eval-aware-steer/modal_run.py --mode confirm      # the confirmatory run (confirm.py)
    modal run experiments/eval-aware-steer/modal_run.py --mode confirm --model gemma2-9b --policy rule   # replication
"""

import io
import tarfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
volume = modal.Volume.from_name("eval-aware-steer", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.1",
        "datasets==4.1.1",
        "scipy==1.16.2",
        "numpy<2.4",
        "accelerate==1.10.1",
        "hf_transfer==0.1.9",
    )
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "EVAL_AWARE_DEVICE": "cuda",
            "EVAL_AWARE_BATCH": "64",
            "EVAL_AWARE_WORK": f"{VOL}/work",
        }
    )
    .add_local_dir(HERE, "/root/exp", ignore=["results/**", "report/**", "__pycache__/**"])
)
app = modal.App("eval-aware-steer", image=image)


@app.function(
    gpu="H100",
    cpu=8.0,
    memory=65536,
    timeout=2 * 60 * 60,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def pipeline(model: str, policy: str) -> bytes:
    import os
    import sys
    import time

    os.chdir("/root/exp")
    sys.path.insert(0, "/root/exp")
    import common as C
    import direction
    import fluency
    import known
    import steer
    import summarize

    # results live on the volume too, so a retry keeps known.json and directions.npz
    res = Path(VOL) / "results"
    res.mkdir(parents=True, exist_ok=True)
    C.RESULTS = res
    t = time.time()
    if not (res / model / "known.json").exists():
        known.main(model)
    if not (res / model / "directions.npz").exists():
        direction.main(model, policy)
    volume.commit()
    print(f"known+direction {time.time() - t:.0f}s", flush=True)
    steer.main(model, policy)
    volume.commit()
    print(f"steer {time.time() - t:.0f}s", flush=True)
    summarize.main(model)
    fluency.main(model)
    volume.commit()
    print(f"total {time.time() - t:.0f}s", flush=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(res / model, arcname=model)
    return buf.getvalue()


@app.function(
    gpu="H100",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def confirm_remote(model: str, policy: str = "motive", questions: str = "fresh") -> bytes:
    """The pre-registered confirmatory run (confirm.py); needs known.json and directions.npz on the volume."""
    import os
    import sys

    os.chdir("/root/exp")
    sys.path.insert(0, "/root/exp")
    import common as C
    import confirm

    res = Path(VOL) / "results"
    C.RESULTS = res
    if model != "llama3.1-8b":
        C.BATCH = 16  # Gemma-2-9B ran out of memory at 64 on one H100
    if not (res / model / "known.json").exists():
        import known

        known.main(model)
    if not (res / model / "directions.npz").exists():
        import direction

        direction.main(model, policy)
    volume.commit()
    confirm.main(model, policy, questions)
    volume.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in ("confirm.json", "confirm_cells.csv", "known.json", "direction.json", "directions.npz"):
            tar.add(res / model / f, arcname=f"{model}/{f}")
    return buf.getvalue()


@app.local_entrypoint()
def main(model: str = "llama3.1-8b", policy: str = "motive", mode: str = "pipeline"):
    if mode == "confirm":
        data = confirm_remote.remote(model, policy, "fresh" if model == "llama3.1-8b" else "test")
    else:
        data = pipeline.remote(model, policy)
    out = HERE / "results"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    print("wrote", out / model)
