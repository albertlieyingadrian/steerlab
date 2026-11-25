"""Modal: sample every model of one base on every prompt with vLLM (LoRA adapters on top of the base).

One call per base: it loads the base once, then generates for each run (a seed and an optional adapter) and writes
/vol/out/<run>.jsonl as it goes. Launched by sample.py (deploy + spawn, call ids saved to disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("behavior-diff")
vol = modal.Volume.from_name("behavior-diff", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "peft==0.17.1", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

GPU = "L40S"
TIMEOUT_S = 1800


@app.function(gpu=GPU, image=image, volumes={"/vol": vol}, timeout=TIMEOUT_S, retries=0)
def generate(
    base: str,
    runs: list[dict],
    prompts: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> dict:
    """runs: [{"name", "seed", "adapter": None | {"repo", "subfolder"}}]. Returns {name: n_replies}."""
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        base,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=4096,
        seed=0,
        gpu_memory_utilization=0.85,
    )
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in prompts]
    out_dir = Path("/vol/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for k, run in enumerate(runs):
        lora = None
        if run.get("adapter"):
            a = run["adapter"]
            sub = a.get("subfolder") or ""
            path = snapshot_download(a["repo"], allow_patterns=[f"{sub}/*"] if sub else None)
            lora = LoRARequest(run["name"], k + 1, str(Path(path) / sub))

        def params(seed: int) -> SamplingParams:
            return SamplingParams(
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_tokens=max_tokens,
                seed=seed,
            )

        # one seed for the whole run (the study's first runs) or one per prompt (`per_prompt`, added after the
        # same-model control showed run-level luck: a shared seed gives every prompt the same random draws)
        sp = (
            [params(run["seed"] * 100_000 + i) for i in range(len(texts))]
            if run.get("per_prompt")
            else params(run["seed"])
        )
        outs = llm.generate(texts, sp, lora_request=lora)
        rows = [{"id": p["id"], "reply": o.outputs[0].text} for p, o in zip(prompts, outs, strict=True)]
        (out_dir / f"{run['name']}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        vol.commit()
        done[run["name"]] = len(rows)
        print(f"{run['name']}: {len(rows)} replies", flush=True)
    return done
