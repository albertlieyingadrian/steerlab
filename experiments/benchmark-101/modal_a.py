"""Run a_open.py for one open model on one Modal H100. COSTS MONEY (plan: Modal <= $10 in all).

    modal run experiments/benchmark-101/modal_a.py --key qwen7b      # also qwen1.5b, llama8b, gemma9b

data/mmlu_redux.jsonl must exist (run `python -c "import bench; bench.mmlu_redux()"` first). Weights come from the
Hugging Face cache on the `eval-aware-steer` volume (downloaded there on first use). One-hour hard timeout per call.
"""

import io
import json
import tarfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
volume = modal.Volume.from_name("eval-aware-steer", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "scipy==1.16.2", "numpy<2.4", "hf_transfer==0.1.9"
    )
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        HERE, "/root/exp", ignore=["results/**", "report/**", "logs/**", "__pycache__/**", "data/sloppair/**"]
    )
)
app = modal.App("benchmark-101", image=image)


@app.function(
    gpu="H100",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def score(key: str) -> bytes:
    import os
    import sys

    os.chdir("/root/exp")
    sys.path.insert(0, "/root/exp")
    import a_open

    out = Path("/tmp/out")
    a_open.run(key, out)
    volume.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main(key: str = "qwen1.5b") -> None:
    data = score.remote(key)
    dest = HERE / "results" / "a_open"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(json.dumps({"key": key, "written": str(dest / f"{key}.json")}))
