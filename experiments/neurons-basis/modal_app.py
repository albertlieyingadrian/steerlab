"""Modal app for the neurons-basis study. COSTS MONEY: every launch is ledgered.

    export MODAL_PROFILE=albertlieyingadrian
    .venv/bin/modal deploy experiments/neurons-basis/modal_app.py
    .venv/bin/python experiments/neurons-basis/launch.py phase1 olmoe sparsity,bench,neurons   # spawns, saves call id

Functions:
    phase1(key, stages)   one model's Phase 1 stages on an L40S (phase1.py), results on the volume
    phase2(stages)        the password-locked organism (phase2.py)
    Judge.complete(items) Qwen3-32B-FP8 with vLLM on an H100, thinking off, temperature 0
"""

# ruff: noqa: B905, RUF028  (long data lines and per-iteration closures read better as they are)

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
NEURONKIT = (
    Path(os.environ["NEURONKIT_SRC"])
    if os.environ.get("NEURONKIT_SRC")
    else Path(__import__("neuronkit").__file__).parent
)
"""The neuronkit package folder uploaded into the images: a source checkout (NEURONKIT_SRC) or the installed one."""

app = modal.App("neurons-basis")
hf = modal.Secret.from_name("huggingface-secret")
vol = modal.Volume.from_name("neurons-basis", create_if_missing=True)
hf_cache = modal.Volume.from_name("neurons-basis-hf", create_if_missing=True)

base = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.9.0",
        "transformers==5.22.2",
        "accelerate",
        "peft",
        "datasets",
        "scikit-learn",
        "numpy>=2.0",
        "huggingface_hub",
        "pydantic>=2.7",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/hf",
            "PYTHONPATH": "/root/nk:/root/exp",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
image = (
    base.add_local_dir(str(NEURONKIT), "/root/nk/neuronkit", ignore=["__pycache__"])
    .add_local_file(str(HERE / "phase1.py"), "/root/exp/phase1.py")
    .add_local_file(str(HERE / "phase2.py"), "/root/exp/phase2.py")
)

# Cap check (A3): L40S about $1.95/h, 3 h timeout, 1 container, no retries -> at most $5.85 per call.


@app.function(
    image=image,
    gpu="L40S",
    memory=65536,
    secrets=[hf],
    volumes={"/vol": vol, "/cache": hf_cache},
    timeout=3 * 60 * 60,
    retries=0,
)
def phase1(key: str, stages: list[str], smoke: bool = False):
    import phase1 as p1

    try:
        p1.run(key, stages, smoke)
    finally:
        vol.commit()
        hf_cache.commit()


@app.function(
    image=image,
    gpu="L40S",
    memory=65536,
    secrets=[hf],
    volumes={"/vol": vol, "/cache": hf_cache},
    timeout=2 * 60 * 60,
    retries=0,
)
def phase2(stages: list[str]):
    import phase2 as p2

    try:
        p2.run(stages)
    finally:
        vol.commit()
        hf_cache.commit()


image35 = (
    base.uv_pip_install("lm-eval>=0.4.8,<0.5", "flash-linear-attention==0.5.2")
    .add_local_dir(str(NEURONKIT), "/root/nk/neuronkit", ignore=["__pycache__"])
    .add_local_file(str(HERE / "phase35.py"), "/root/exp/phase35.py")
)


# Cap check (amendment 8, Modal cap raised by $50): H200 about $4.54/h, 4 h timeout, 1 container, no retries
# -> at most $18.2 per call; two calls planned (smoke, 35B), OLMoE runs on an L40S.
@app.function(
    image=image35,
    gpu="H200",
    memory=131072,
    secrets=[hf],
    volumes={"/vol": vol, "/cache": hf_cache},
    timeout=4 * 60 * 60,
    retries=0,
)
def phase35_big(key: str, smoke: bool = False):
    import phase35 as p

    try:
        p.run(key, smoke)
    finally:
        vol.commit()
        hf_cache.commit()


@app.function(
    image=image35,
    gpu="L40S",
    memory=65536,
    secrets=[hf],
    volumes={"/vol": vol, "/cache": hf_cache},
    timeout=3 * 60 * 60,
    retries=0,
)
def phase35_small(key: str, smoke: bool = False):
    import phase35 as p

    try:
        p.run(key, smoke)
    finally:
        vol.commit()
        hf_cache.commit()


JUDGE = "Qwen/Qwen3-32B-FP8"
judge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


# Cap check: H100 about $3.95/h, 2 h timeout, 1 container -> at most $7.90 per container lifetime.
@app.cls(
    image=judge_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": hf_cache},
    timeout=2 * 60 * 60,
    scaledown_window=120,
    retries=0,
)
class Judge:
    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(JUDGE, max_model_len=8192, gpu_memory_utilization=0.9, enable_prefix_caching=True,
                       generation_config="vllm", seed=0)  # fmt: skip
        self.tok = self.llm.get_tokenizer()
        hf_cache.commit()

    @modal.method()
    def complete(self, items: list[dict]) -> list[str]:
        """items: {"system", "user", "max_tokens", "temperature" (default 0), "thinking" (default off)}."""
        from vllm import SamplingParams

        texts = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(it.get("thinking", False)),
            )
            for it in items
        ]
        params = [
            SamplingParams(
                max_tokens=int(it.get("max_tokens", 600)),
                temperature=float(it.get("temperature", 0.0)),
                top_p=float(it.get("top_p", 1.0)),
                top_k=int(it.get("top_k", -1)),
                repetition_penalty=1.0,
                seed=int(it.get("seed", 0)),
            )  # fmt: skip
            for it in items
        ]
        outs = self.llm.generate(texts, params)
        res = []
        for it, o in zip(items, outs):
            t = o.outputs[0].text
            if it.get("thinking"):  # keep only the answer; "" when the reasoning ran out of tokens
                t = t.split("</think>", 1)[1].strip() if "</think>" in t else ""
            res.append(t)
        return res
