"""Open-weight judges and unsteered subject models on Modal vLLM. COSTS MONEY (one H100, $3.95/h, 2 h hard stop).

Shared by both replications in this series. Used because the papers' judges (Claude Haiku 4.5, GPT-4o) are paid APIs.

    export MODAL_PROFILE=credits3 MODAL_IMAGE_BUILDER_VERSION=2025.06
    .venv/bin/modal deploy experiments/repl-steering-resistance/modal_vllm.py

`chats(model, jobs)`: each job is {id, messages, followups?, temp, max_new, max_new_follow?, seed}. The model answers
the messages; for each follow-up, its reply and the follow-up are appended and it answers again. Thinking is off for
Qwen3; gpt-oss runs at reasoning "low" and only its final channel is kept. Returns {id, turns} rows.
"""

from __future__ import annotations

import re
import time

import modal

APP = "aiaf-repl-vllm"
vol = modal.Volume.from_name("aiaf-repl", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/vol/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
app = modal.App(APP, image=image)
FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL)


def _final(model: str, text: str) -> str:
    if "gpt-oss" not in model:
        return text.strip()
    m = FINAL.findall(text)
    if m:
        return m[-1].strip()
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].strip()
    return text.strip()


@app.function(
    gpu="H100",
    timeout=2 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    max_containers=1,
)
def chats(model: str, jobs: list[dict], max_len: int = 8192) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model,
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        generation_config="vllm",
        seed=0,
        **({"limit_mm_per_prompt": {"image": 0}} if "gemma-3" in model else {}),
    )
    kw = {}
    if "Qwen3" in model and "2507" not in model:
        kw = {"chat_template_kwargs": {"enable_thinking": False}}
    if "gpt-oss" in model:
        kw = {"chat_template_kwargs": {"reasoning_effort": "low"}}
    convs = [list(j["messages"]) for j in jobs]
    turns: list[list[str]] = [[] for _ in jobs]
    n_follow = max(len(j.get("followups") or []) for j in jobs)
    for t in range(n_follow + 1):
        live = [i for i, j in enumerate(jobs) if t <= len(j.get("followups") or [])]
        sps = []
        for i in live:
            j = jobs[i]
            mt = j.get("max_new", 512) if t == 0 else j.get("max_new_follow", j.get("max_new", 512))
            if "gpt-oss" in model:
                mt += 1024  # room for the analysis channel
            sps.append(
                SamplingParams(
                    temperature=j.get("temp", 0.0),
                    top_p=1.0,
                    top_k=-1 if j.get("temp", 0.0) > 0 else 1,
                    repetition_penalty=1.0,
                    max_tokens=mt,
                    seed=int(j.get("seed", 0)) * 100 + t,
                    skip_special_tokens="gpt-oss" not in model,
                )
            )
        outs = llm.chat([convs[i] for i in live], sps, use_tqdm=False, **kw)
        for i, o in zip(live, outs, strict=True):
            txt = _final(model, o.outputs[0].text)
            turns[i].append(txt)
            fl = jobs[i].get("followups") or []
            if t < len(fl):
                convs[i] = convs[i] + [{"role": "assistant", "content": txt}, {"role": "user", "content": fl[t]}]
    vol.commit()
    return {"rows": [{"id": j["id"], "turns": turns[i]} for i, j in enumerate(jobs)], "seconds": time.time() - t0}


@app.function(
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    max_containers=1,
)
def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Sentence embeddings (last-token pooling, normalised) for Experiment 3 of the self-reference replication."""
    from vllm import LLM

    llm = LLM(model, runner="pooling", max_model_len=2048, gpu_memory_utilization=0.85)
    outs = llm.embed(texts, use_tqdm=False)
    return [o.outputs.embedding for o in outs]
