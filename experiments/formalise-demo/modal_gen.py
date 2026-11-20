"""Generate every arm's replies with an open model under vLLM on one Modal H100.

COSTS MONEY. Every run goes through the spend ledger (results/spend.jsonl) and refuses to start if the ledger plus
the run's worst case (its timeout at the H100 rate) would pass the cap. From the repo root:

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/formalise-demo/modal_gen.py \\
        --model qwen3-8b --jobs baggage:dev:a,baggage:dev:b [--seed 0] [--limit 5]

A job is task:split:arm (task baggage or contrast; split dev or heldout; arm a, b0 or b). One container loads the
model once and answers every job's questions in one batch (the pasted rules are a shared prefix, so vLLM caches them).
Replies go to results/raw/<task>_<split>_<model>_<arm>[_s<seed>].jsonl: id, text (thinking included), tokens, finish.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODAL_CAP = 25.0  # dollars; the episode's hard stop is $40
H100_RATE, OVERHEAD = 3.95, 1.10  # $/hour Modal list price (GPU), plus CPU, memory and start-up
TIMEOUT_S = 2 * 3600
MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "qwen3-32b": "Qwen/Qwen3-32B-FP8"}
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 12000}  # Qwen3's thinking-mode settings
MAX_MODEL_LEN = 24576

app = modal.App("formalise-demo")
volume = modal.Volume.from_name("formalise-demo", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": volume},
    timeout=TIMEOUT_S,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(model_id: str, requests: list[dict], sampling: dict, seed: int) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        max_num_seqs=256,
        seed=seed,
    )
    params = [SamplingParams(**sampling, seed=seed * 1_000_003 + i) for i in range(len(requests))]
    t0 = time.time()
    outs = llm.chat([r["messages"] for r in requests], params, chat_template_kwargs={"enable_thinking": True})
    volume.commit()
    print(f"generated {len(outs)} replies in {time.time() - t0:.0f}s", flush=True)
    return [
        {
            "key": r["key"],
            "id": r["id"],
            "text": o.outputs[0].text,
            "tokens": len(o.outputs[0].token_ids),
            "prompt_tokens": len(o.prompt_token_ids),
            "finish": o.outputs[0].finish_reason,
        }
        for r, o in zip(requests, outs, strict=True)
    ]


def ledger_total() -> float:
    p = RESULTS / "spend.jsonl"
    if not p.exists():
        return 0.0
    return sum(json.loads(line)["dollars"] for line in p.read_text().splitlines() if line.strip())


def ledger_add(what: str, seconds: float) -> None:
    dollars = H100_RATE * OVERHEAD * seconds / 3600
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "spend.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "what": what,
                    "gpu": "H100",
                    "seconds": round(seconds),
                    "dollars": round(dollars, 2),
                }
            )
            + "\n"
        )
    print(f"spent ${dollars:.2f} on {what}; ledger now ${ledger_total():.2f} of ${MODAL_CAP}")


def out_path(task: str, split: str, model: str, arm: str, seed: int) -> Path:
    suffix = f"_s{seed}" if seed else ""
    return RESULTS / "raw" / f"{task}_{split}_{model}_{arm}{suffix}.jsonl"


@app.local_entrypoint()
def main(model: str, jobs: str, seed: int = 0, limit: int = 0):
    sys.path.insert(0, str(HERE))
    from common import DATA, read_jsonl, write_jsonl
    from prompts import messages

    worst = H100_RATE * OVERHEAD * TIMEOUT_S / 3600
    spent = ledger_total()
    if spent + worst > MODAL_CAP:
        raise SystemExit(f"REFUSED: ledger ${spent:.2f} + worst case ${worst:.2f} > cap ${MODAL_CAP}")
    requests, keys = [], []
    for job in jobs.split(","):
        task, split, arm = job.split(":")
        rows = read_jsonl(DATA / f"{task}_{split}.jsonl")[: limit or None]
        keys.append((task, split, arm))
        requests += [{"key": job, "id": r["id"], "messages": messages(task, arm, r["prompt"])} for r in rows]
    print(f"{len(requests)} requests for {model}; ledger ${spent:.2f}, worst case ${worst:.2f}, cap ${MODAL_CAP}")
    t0 = time.time()
    try:
        replies = generate.remote(MODELS[model], requests, SAMPLING, seed)
    finally:
        ledger_add(f"{model} {jobs} seed={seed} limit={limit}", time.time() - t0)
    for task, split, arm in keys:
        job = f"{task}:{split}:{arm}"
        rows = [{k: v for k, v in r.items() if k != "key"} for r in replies if r["key"] == job]
        path = out_path(task, split, model, arm, seed)
        if limit:
            path = path.with_name(path.stem + f"_pilot{limit}.jsonl")
        write_jsonl(path, rows)
        cut = sum(r["finish"] == "length" for r in rows)
        print(
            f"{path.name}: {len(rows)} replies, {cut} hit the token cap, "
            f"mean {sum(r['tokens'] for r in rows) / max(len(rows), 1):.0f} tokens"
        )
