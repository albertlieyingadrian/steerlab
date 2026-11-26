"""Sample `pushback` episodes from a model with vLLM on Modal (COSTS MONEY: an L4 is about $0.80/h at list price).

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/graft-monitor/modal_sample.py \
        --model Qwen/Qwen3-0.6B --split eval --n 256 --seed 99 --tag base_eval

Writes results/samples/<tag>.jsonl locally: one row per episode (Episode fields, both replies, grade fields) and
appends the run's cost estimate to results/spend.jsonl. Trained checkpoints in the `graft-monitor` Volume are
sampled with --model /vol/<path> (or --lora /vol/<adapter>).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/gm"
VOL = "/vol"
L4_PER_HOUR = 1.95 + 4 * 0.0473 + 16 * 0.0080  # worst case of the fallback list (L40S) + 4 CPU + 16 GiB, list price

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "env.py", f"{REMOTE}/env.py")
    .add_local_dir(HERE / "data", f"{REMOTE}/data")
)
volume = modal.Volume.from_name("graft-monitor", create_if_missing=True)
app = modal.App("graft-monitor-sample", image=image)


@app.function(gpu=["L4", "A10G", "L40S"], cpu=4.0, memory=16384, timeout=1800, volumes={VOL: volume})
def rollout(
    model: str,
    episodes: list[dict],
    samples: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 160,
    seed: int = 0,
    lora: str = "",
) -> list[dict]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, REMOTE)
    import env

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        seed=seed,
        enable_lora=bool(lora),
        max_lora_rank=64,
    )
    lreq = LoRARequest("a", 1, lora) if lora else None
    sp = SamplingParams(
        n=1, temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=max_tokens
    )
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    fields = env.Episode.__dataclass_fields__
    eps = [env.Episode(**{k: v for k, v in e.items() if k in fields}) for e in episodes]
    eps = [e for e in eps for _ in range(samples)]
    first = llm.chat([e.turn1() for e in eps], sp, lora_request=lreq, use_tqdm=False, **kw)
    first = [o.outputs[0].text.strip() for o in first]
    final = llm.chat([e.turn2(a) for e, a in zip(eps, first)], sp, lora_request=lreq, use_tqdm=False, **kw)
    final = [o.outputs[0].text.strip() for o in final]
    return [{**e.to_dict(), "first": a, "final_text": b, **env.grade(e, a, b)} for e, a, b in zip(eps, first, final)]


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-0.6B",
    split: str = "eval",
    n: int = 256,
    seed: int = 99,
    samples: int = 1,
    temperature: float = 1.0,
    tag: str = "",
    lora: str = "",
):
    sys.path.insert(0, str(HERE))
    import env

    eps = [e.to_dict() for e in env.make_episodes(split, n, seed)]
    t0 = time.time()
    rows = rollout.remote(model, eps, samples=samples, temperature=temperature, seed=seed, lora=lora)
    dt = time.time() - t0
    out = HERE / "results" / "samples" / f"{tag or split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(env.summarize(rows), indent=1))
    with open(HERE / "results" / "spend.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": "modal_estimate",
                    "usd": round(dt / 3600 * L4_PER_HOUR, 4),
                    "note": f"sample {tag} {model} wall {dt:.0f}s",
                }
            )
            + "\n"
        )
