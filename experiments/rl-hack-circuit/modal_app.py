"""Every model computation of the RL-hack-circuit study (circuits Episode 2), on rented GPUs (Modal).

    MODAL_PROFILE=... python -m modal run experiments/rl-hack-circuit/modal_app.py --phase prep
    ... --phase graphs     # 64 attribution graphs, selection, first-order attribution, controls, B-explain activations
    ... --phase swaps      # swap tests, causal checks at the comment slots (needs results/rubrics.json)
    ... --phase gen        # hack rates of the after model under group zero-ablation (32 sketches each)
    ... --phase patch      # exploratory residual patching at the slot token

The Pigment Gym checkpoints are read from the Modal volume `pigment-gym-qwen3` (mounted at /ckpt); this study's own
files go to the volume `rl-hack-circuit`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
_SIBLING = HERE.parents[2] / "circuit-tracer" if len(HERE.parents) > 2 else HERE  # inside the container: unused
CT_SRC = Path(os.environ.get("CIRCUIT_TRACER_SRC", _SIBLING)) / "circuit_tracer"
PIGMENT = HERE.parent / "pigment-gym"

app = modal.App("rl-hack-circuit")
vol = modal.Volume.from_name("rl-hack-circuit", create_if_missing=True)
ckpt = modal.Volume.from_name("pigment-gym-qwen3")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.3",
        "transformer-lens==3.2.1",
        "peft==0.17.1",
        "nnsight>=0.8.0rc1,<0.9",
        "safetensors>=0.5",
        "huggingface_hub[hf_transfer]<1.0",
        "einops>=0.8",
        "numpy<2.3",
        "pydantic>=2",
        "tqdm",
        "seaborn",
        "ipywidgets",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONPATH": "/root/ct:/root/exp",
        }
    )
    .add_local_dir(CT_SRC, "/root/ct/circuit_tracer", ignore=["**/__pycache__/**"])
    .add_local_file(HERE / "common.py", "/root/exp/common.py")
    .add_local_file(HERE / "ep2.py", "/root/exp/ep2.py")
    .add_local_file(PIGMENT / "brush_tools.py", "/root/exp/brush_tools.py")
    .add_local_file(HERE / "results" / "inputs" / "rollouts_explain.jsonl", "/root/exp/rollouts.jsonl")
    .add_local_file(HERE / "results" / "inputs" / "after_t1.jsonl", "/root/exp/after_t1.jsonl")
)
VOLS = {"/vol": vol, "/ckpt": ckpt}


@app.function(
    image=image,
    gpu=["H100", "H200", "B200", "A100-80GB", "L40S"],
    volumes=VOLS,
    timeout=6 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def run(phase: str, rubrics_json: str = "{}"):
    import ep2

    ep2.main(phase, json.loads(rubrics_json), vol)


@app.local_entrypoint()
def main(phase: str = "prep"):
    rub = "{}"
    if phase in ("swaps", "gen", "open"):
        rub = (HERE / "results" / "rubrics.json").read_text(encoding="utf-8")
    if phase == "graphs-par":  # one container per contrast, then the selection and the rest of the phase
        calls = [run.spawn(f"graphs:{n}") for n in ("installed", "replication", "control", "rewired")]
        for call in calls:
            call.get()
        run.remote("select")
        return
    run.remote(phase, rub)
