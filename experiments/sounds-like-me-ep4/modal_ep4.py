"""Episode 4 GPU function on Modal: the LLM-panel judge (a comparison arm, never a stand-in for people).

COSTS MONEY (Modal H100). Callers (rulers_modal.py) check the episode cap and charge the series ledger.
Deploy from the repo root:

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/sounds-like-me-ep4/modal_ep4.py

judge(payload): one forward step per prompt with vLLM (greedy, thinking off); returns the log-probabilities of each
allowed answer token among the top 20, so a score is the expected value over answers rather than a sampled guess.
The preference model reuses Episode 2's deployed trainer (slm-ep2 / train_pref), unchanged.
"""

from __future__ import annotations

import time

import modal

app = modal.App("slm-ep4")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("sounds-like-me-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


@app.function(image=image, cpu=4, memory=8192, secrets=[hf], volumes={"/cache": cache}, timeout=3 * 60 * 60)
def prefetch(repo: str) -> dict:
    """Download a model into the shared volume on a CPU container, so no GPU is billed while it downloads.
    (The first AWQ smoke test spent its 45 GPU minutes downloading 28 of 41.6 GB.)"""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    cache.commit()
    return {"path": path, "seconds": round(time.time() - t0, 1)}


# 45-minute timeout: about $3.30 of H100 at worst per call, so a hung run stays inside the episode's $20 cap
@app.function(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=45 * 60)
def judge(payload: dict) -> dict:
    """payload: {model, prompts: [user message], choices: [answer strings], max_model_len}."""
    from vllm import LLM, SamplingParams

    t0 = time.time()
    model = payload["model"]
    max_len = payload.get("max_model_len", 8192)
    llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if "Qwen3" in model else {}
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **kw)
        for p in payload["prompts"]
    ]
    # an over-long prompt keeps its first and last halves (the question sits at the end)
    limit = max_len - 8
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    prompts = [
        {"prompt_token_ids": i if len(i) <= limit else i[: limit // 2] + i[-(limit - limit // 2) :]} for i in ids
    ]
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    choices = payload["choices"]
    res = []
    for o in outs:
        d = {}
        for lp in o.outputs[0].logprobs[0].values():
            s = (lp.decoded_token or "").strip()
            if s in choices and s not in d:
                d[s] = lp.logprob
        res.append(d)
    return {"logprobs": res, "truncated": sum(len(i) > limit for i in ids), "seconds": round(time.time() - t0, 1)}
