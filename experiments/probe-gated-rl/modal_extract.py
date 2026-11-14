"""
Teacher-forced activations on a Modal L40S (float32, CUDA: the same arithmetic the training runs use to score
sketches). The laptop's GPU reads about one sketch per second in float32; this does the same work in minutes.

COSTS MONEY (cents per job). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/probe-gated-rl/modal_extract.py --job probe_data
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/probe-gated-rl/modal_extract.py --job obfuscation

probe_data   the 2026-11-13 runs' 4,800 training sketches, read by the untrained model at layers 7, 14 and 21,
             plus the drift control (loophole2_seed3/final reading its before/after sketches)
             -> ~/.cache/steerlab/probe-gated-rl/acts_base.npz, acts_drift_l2s3.npz (read by 1_fit_probe.py)
obfuscation  each new run's final policy and the untrained model reading that run's after_t1, last-10-step,
             before_t1 and first-10-step sketches -> ~/.cache/steerlab/probe-gated-rl/obfuscation/<run>.npz
             (read by 3_obfuscation.py)
"""

import io
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PG = HERE.parent / "pigment-gym"
REMOTE_PG = "/root/pigment-gym"
REMOTE_HERE = "/root/probe-gated-rl"
MODEL = "Qwen/Qwen3-1.7B"
WORK = Path.home() / ".cache" / "steerlab" / "probe-gated-rl"

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "numpy", "pillow")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PG, REMOTE_PG, ignore=["**/__pycache__", "results", "report", "vendor"])
    .add_local_dir(HERE, REMOTE_HERE, ignore=["**/__pycache__", "results", "report"])
)
app = modal.App("probe-gated-rl-extract", image=image)
pigment_volume = modal.Volume.from_name("pigment-gym-qwen3")
gated_volume = modal.Volume.from_name("probe-gated-rl", create_if_missing=True)


@app.function(
    gpu="L40S", cpu=4.0, memory=32768, timeout=60 * 60, volumes={"/vol": pigment_volume, "/out": gated_volume}
)
def extract_remote(texts, layers, adapter=None, with_base=True, batch=8):
    """Pooled activations for `texts`: {"base_L<l>": ..., "policy_L<l>": ...} as an npz in bytes.
    adapter: a LoRA folder on a mounted volume (/vol/... or /out/...), or None for the untrained model only."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path[:0] = [REMOTE_HERE, REMOTE_PG]
    from brush_tools import SYSTEM_PROMPT_NO_EXAMPLE
    from hand_pool import SUBJECT

    from probe_tools import extract

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda").eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_NO_EXAMPLE},
        {"role": "user", "content": f"Paint {SUBJECT}."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = {}
    if with_base:
        for layer, values in extract(model, tokenizer, prompt, texts, tuple(layers), batch=batch).items():
            out[f"base_L{layer}"] = values
    if adapter is not None:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(model, adapter).eval()
        for layer, values in extract(policy, tokenizer, prompt, texts, tuple(layers), batch=batch).items():
            out[f"policy_L{layer}"] = values
    buffer = io.BytesIO()
    np.savez(buffer, **out)
    return buffer.getvalue()


def read(path):
    return [json.loads(line) for line in path.open()]


@app.local_entrypoint()
def main(job: str, runs: str = ""):
    import numpy as np

    WORK.mkdir(parents=True, exist_ok=True)
    if job == "probe_data":
        qwen3 = HERE / "results" / "probe_inputs"  # copies of the 2026-11-13 Pigment Gym Qwen3 runs
        names = [
            "loophole2_seed0",
            "loophole2_seed1",
            "loophole2_seed2",
            "loophole2_seed3",
            "loophole_seed0",
            "loophole_seed1",
        ]
        calls = {
            name: extract_remote.spawn([r["text"] for r in read(qwen3 / f"{name}_rollouts.jsonl")], [7, 14, 21])
            for name in names
        }
        before = read(qwen3 / "loophole2_seed3_before_t1_samples.jsonl")
        after = read(qwen3 / "loophole2_seed3_after_t1_samples.jsonl")
        drift_call = extract_remote.spawn(
            [r["text"] for r in before + after], [14], adapter="/vol/loophole2_seed3/final"
        )
        acts = {}
        for name, call in calls.items():
            data = np.load(io.BytesIO(call.get()))
            for layer in (7, 14, 21):
                acts[f"{name}_L{layer}"] = data[f"base_L{layer}"]
            print(f"{name}: {len(acts[f'{name}_L14'])} sketches", flush=True)
        np.savez(WORK / "acts_base.npz", **acts)
        drift = np.load(io.BytesIO(drift_call.get()))
        np.savez(WORK / "acts_drift_l2s3.npz", base=drift["base_L14"], policy=drift["policy_L14"])
        print("wrote acts_base.npz and acts_drift_l2s3.npz")
    elif job == "obfuscation":
        runs_dir = HERE / "results" / "runs"
        names = (
            runs.split(",")
            if runs
            else sorted(p.name.removesuffix("_summary.json") for p in runs_dir.glob("*_summary.json"))
        )
        cache = WORK / "obfuscation"
        cache.mkdir(parents=True, exist_ok=True)
        calls = {}
        for name in names:
            # the same four sets, in the same order, as 3_obfuscation.py
            after = read(runs_dir / f"{name}_after_t1_samples.jsonl")
            before = read(runs_dir / f"{name}_before_t1_samples.jsonl")
            rollouts = read(runs_dir / f"{name}_rollouts.jsonl")
            last_step = max(r["step"] for r in rollouts)
            late = [r for r in rollouts if r["step"] > last_step - 10]
            early = [r for r in rollouts if r["step"] < 10]
            texts = [r["text"] for r in after + late + before + early]
            calls[name] = extract_remote.spawn(texts, [14], adapter=f"/out/{name}/final")
        for name, call in calls.items():
            data = np.load(io.BytesIO(call.get()))
            np.savez(cache / f"{name}.npz", policy=data["policy_L14"], base=data["base_L14"])
            print(f"{name}: {len(data['policy_L14'])} sketches", flush=True)
    else:
        raise ValueError(f"unknown job {job!r}")
