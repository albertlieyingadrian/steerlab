"""Open-model generation for Sounds Like Me on Modal (vLLM).

COSTS MONEY. Callers (run_open.py) charge every call to data/spend.jsonl as provider "modal"
(GPU wall time x list rate x overhead) and refuse to start if the ledger would pass MODAL_CAP. From the repo root:

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/sounds-like-me/modal_gen.py

One call per model; prompts are chat message lists (instruct models, thinking disabled) or raw strings (base models).
"""

from __future__ import annotations

import time

import modal

MODAL_CAP = 900.0  # user, 2026-11-20
RATE = {"H100": 3.95, "A100-80GB": 2.50}  # $/hour, Modal list price (GPU only)
OVERHEAD = 1.10

app = modal.App("sounds-like-me")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("sounds-like-me-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


@app.function(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=3 * 60 * 60)
def generate(
    model: str,
    prompts: list,
    chat: bool,
    max_tokens: int = 400,
    temperature: float = 0.7,
    seed: int = 0,
    max_model_len: int = 12288,
) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model, max_model_len=max_model_len, gpu_memory_utilization=0.90, seed=seed, enable_prefix_caching=True
    )
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        stop=None if chat else ["\nEmail ", "\n\nEmail "],
    )
    if chat:
        tok = llm.get_tokenizer()
        kw = {"enable_thinking": False} if "qwen3" in model.lower() else {}
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw) for m in prompts]
    else:
        texts = prompts
    outs = llm.generate(texts, sp)
    cache.commit()
    return {
        "outputs": [o.outputs[0].text for o in outs],
        "seconds": round(time.time() - t0, 1),
        "gpu": "H100",
        "prompt_tokens": sum(len(o.prompt_token_ids) for o in outs),
        "completion_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
    }


dipper_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "nltk==3.9.1",
        "hf_transfer==0.1.9",
        "accelerate==1.1.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


@app.function(image=dipper_image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=2 * 60 * 60)
def dipper(texts: list, lex: int = 60, order: int = 0, window: int = 3) -> dict:
    """DIPPER paraphrase attack (Krishna et al., 2023): rewrite each text in windows of `window` sentences,
    conditioned on the already-paraphrased prefix. lex/order in {0,20,40,60,80,100} (higher = more change)."""
    import nltk
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    t0 = time.time()
    tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = (
        T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl", torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    lex_code, order_code = 100 - lex, 100 - order
    outs = []
    for text in texts:
        sents = nltk.sent_tokenize(" ".join(text.split()))
        prefix, out = "", ""
        for i in range(0, len(sents), window):
            win = " ".join(sents[i : i + window])
            inp = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {win} </sent>"
            enc = tok([inp], return_tensors="pt", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, top_p=0.75, max_length=512)
            para = tok.batch_decode(gen, skip_special_tokens=True)[0]
            prefix += " " + win
            out += " " + para
        outs.append(out.strip())
    cache.commit()
    return {"outputs": outs, "seconds": round(time.time() - t0, 1)}


def charge(seconds, gpu="H100"):
    return seconds / 3600 * RATE[gpu] * OVERHEAD
