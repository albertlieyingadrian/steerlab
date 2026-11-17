"""Run one stage of the neuron-communities study on one Modal H100. COSTS MONEY (plan budget: $15 in all).

    modal run experiments/neuron-communities/modal_run.py --stage-name smoke
    modal run experiments/neuron-communities/modal_run.py --stage-name e1_base     # also: e1_instruct, e2a
    modal run experiments/neuron-communities/modal_run.py --stage-name e2b         # finalists from results/e2a.json
    modal run experiments/neuron-communities/modal_run.py --stage-name e3

Weights come from the HF cache on the `eval-aware-steer` volume (Llama-3.1-8B-Instruct is there; the base model is
downloaded into it on first use). Each call has a one-hour hard timeout, so it can cost at most about $4.6. Results come
back as a tar archive and are unpacked into results/.
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
    .add_local_dir(HERE, "/root/exp", ignore=["results/**", "report/**", "logs/**", "__pycache__/**"])
)
app = modal.App("neuron-communities", image=image)


@app.function(
    gpu="H100",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def stage(name: str, finalists: list) -> bytes:
    import os
    import sys

    os.chdir("/root/exp")
    sys.path.insert(0, "/root/exp")
    import run

    out = Path("/tmp/out")
    run.run(name, out, finalists)
    volume.commit()  # keep any newly downloaded weights
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main(stage_name: str = "smoke") -> None:
    finalists = []
    if stage_name in ("e2b", "e3"):
        finalists = json.loads((HERE / "results" / "e2a.json").read_text())["finalists"]
    data = stage.remote(stage_name, finalists)
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"{stage_name}: results in {dest}")
