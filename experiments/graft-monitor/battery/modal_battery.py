"""Sample one model on the values battery and on behavior-diff's prompts with vLLM on Modal (COSTS MONEY: an L4 is
about $1.12/h with CPU and memory, an L40S fallback about $2.40/h; one model with two seeds takes a few minutes).

Not run directly: `battery.run_battery` and `battery.drift` call `sample` inside `app.run()`. The model is a Hub id,
a path in the `graft-monitor` Volume (/vol/...), or a LoRA adapter path in the Volume on top of `model`.
Conventions copied from modal_sample.py: vllm 0.11.0, bf16, thinking off, the model's own chat template,
repetition_penalty 1.0, battery at temperature 1 and 160 new tokens; behavior-diff prompts at its defaults
(temperature 0.7, 300 tokens).
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/battery"
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1")  # vLLM 0.11 breaks on transformers 5
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "core.py", f"{REMOTE}/core.py")
)
volume = modal.Volume.from_name("graft-monitor", create_if_missing=True)
app = modal.App("graft-monitor-battery", image=image)


@app.function(
    gpu=["L4", "A10G", "L40S"],  # fallback list: L4 capacity is often short
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={VOL: volume},
)
def sample(
    model: str,
    items: list[dict],
    bdiff_prompts: list[dict],
    seeds: list[int],
    lora: str = "",
    temperature: float = 1.0,
    max_tokens: int = 160,
    bdiff_temperature: float = 0.7,
    bdiff_max_tokens: int = 300,
) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, REMOTE)
    import core

    # Trained checkpoints in the Volume were saved by transformers 5 (tokenizer_config in the v5 format, which this
    # image's transformers 4.57 cannot read); their tokenizer is the base model's, so load it from the Hub.
    tok = {"tokenizer": "Qwen/Qwen3-0.6B"} if model.startswith("/vol/") else {}
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        seed=0,
        enable_lora=bool(lora),
        max_lora_rank=64,
        **tok,
    )
    lreq = LoRARequest("ckpt", 1, lora) if lora else None
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    by_id = {it["id"]: it for it in items}

    def gen(convs: list[list[dict]], seed: int, offset: int, temp: float, cap: int) -> list[str]:
        sps = [
            SamplingParams(
                n=1,
                temperature=temp,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_tokens=cap,
                seed=seed * 1_000_003 + offset + i,
            )
            for i in range(len(convs))
        ]
        outs = llm.chat(convs, sps, lora_request=lreq, use_tqdm=False, **kw)
        return [o.outputs[0].text.strip() for o in outs]

    result = {}
    for seed in seeds:
        jobs = core.jobs(items)
        first = gen([core.turn1(by_id[j["item"]]) for j in jobs], seed, 0, temperature, max_tokens)
        syc = [i for i, j in enumerate(jobs) if j["readout"] == "sycophancy"]
        convs2 = [core.turn2(by_id[jobs[i]["item"]], first[i], jobs[i]["push"]) for i in syc]
        final = dict(zip(syc, gen(convs2, seed, 100_000, temperature, max_tokens)))
        rows = []
        for i, j in enumerate(jobs):
            it = by_id[j["item"]]
            if i in final:
                rows.append(
                    {**j, "messages": core.turn2(it, first[i], j["push"]), "first": first[i], "reply": final[i]}
                )
            else:
                rows.append({**j, "messages": core.turn1(it), "first": None, "reply": first[i]})
        bd = gen([p["messages"] for p in bdiff_prompts], seed, 200_000, bdiff_temperature, bdiff_max_tokens)
        result[str(seed)] = {"battery": rows, "bdiff": [{"id": p["id"], "reply": r} for p, r in zip(bdiff_prompts, bd)]}
    return result
