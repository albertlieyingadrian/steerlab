"""Open-weight judges on Modal (vLLM), used because every judge API refused payment on 2026-11-21 (plan amendments 8
and 11). Deployed once, then called in batches by judge.py (provider "modal"): the same system prompt and rubric as
the API judges, greedy, up to 8 tokens, thinking off.

    MODAL_PROFILE=albertlieyingadrian modal deploy experiments/voice-alchemy/open_judge.py
"""

from __future__ import annotations

import time

import modal

app = modal.App("voice-alchemy-open-judge")
cache = modal.Volume.from_name("culture-steering-vllm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
COMMON = {
    "image": image,
    "secrets": [modal.Secret.from_name("huggingface-secret")],
    "volumes": {"/cache": cache},
    "timeout": 3 * 60 * 60,
}


def _run(model: str, tp: int, system: str, prompts: list[str]) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model,
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        tensor_parallel_size=tp,
        generation_config="vllm",
        seed=0,
        **({"enforce_eager": True, "disable_custom_all_reduce": True} if tp > 1 else {}),  # no nvcc in image
        **({"limit_mm_per_prompt": {"image": 0}} if "gemma-3" in model else {}),
    )  # text only
    tok = llm.get_tokenizer()
    msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs
    ]
    outs = llm.generate(texts, SamplingParams(max_tokens=8, temperature=0.0))
    cache.commit()
    return {"out": [o.outputs[0].text for o in outs], "seconds": time.time() - t0}


@app.function(gpu="H100:2", **COMMON)
def score2(model: str, system: str, prompts: list[str]) -> dict:
    return _run(model, 2, system, prompts)


@app.function(gpu="H100", **COMMON)
def score1(model: str, system: str, prompts: list[str]) -> dict:
    return _run(model, 1, system, prompts)
