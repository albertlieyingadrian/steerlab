"""
the Qwen3-32B pilot on one rented H100 (Modal + vLLM). COSTS MONEY: hard budget $12 for the stage.

    # 1. weights into a Modal volume, on a CPU-only container (no GPU billed; about 10 minutes)
    .venv/bin/modal run experiments/natural-sandbagging/modal_32b.py::download
    # 2. a smoke run on 8 generations (proves the pipeline end to end before the real run; about 6 GPU-minutes)
    .venv/bin/modal run experiments/natural-sandbagging/modal_32b.py::main --limit 8 --tag smoke
    # 3. the pilot: 300 WMDP + 300 MMLU items x {neutral, eval, deploy} x 2 seeds = 3,600 generations
    .venv/bin/modal run experiments/natural-sandbagging/modal_32b.py::main --tag pilot

What runs where, in software terms:
  * the laptop renders nothing secret: it builds the 3,600 user messages (framing + question + options) from the
    cached datasets and sends them as a list;
  * the H100 container loads Qwen3-32B with vLLM (a serving engine that batches hundreds of requests at once) and
    does the same two steps as the 8B run: sample at most 512 reasoning tokens, then append "Answer:" and read the
    four letter log-probabilities;
  * the rows come back to the laptop and are written to the work folder (~/.cache/steerlab/natural-sandbagging/).

The budget guard: GPU_TIMEOUT_S is a hard stop. Modal kills the container at that point, so one call can never cost
more than GPU_TIMEOUT_S / 3600 x PRICE_PER_HOUR. The plan allows the smoke run, the pilot and one retry.
"""

import json
import sys
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-32B"
VOLUME_PATH = "/vol"
GPU_TIMEOUT_S = 3600  # one hour: at most ~$4.6 per call (plan)
PRICE_PER_HOUR = 3.95 + 8 * 0.0473 + 64 * 0.0080  # H100 + 8 CPU cores + 64 GiB memory, list price (2026-10-24)
CAP = 512
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
volume = modal.Volume.from_name("p21-19-natural-sandbagging", create_if_missing=True)
app = modal.App("p21-19-natural-sandbagging", image=image)


@app.function(cpu=4.0, memory=8192, timeout=40 * 60, volumes={VOLUME_PATH: volume})
def download_weights():
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    volume.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"path": path, "gb": round(size / 1e9, 1), "seconds": round(time.time() - t0, 1)}


@app.function(gpu="H100", cpu=8.0, memory=65536, timeout=GPU_TIMEOUT_S, volumes={VOLUME_PATH: volume})
def generate(jobs: list[dict]) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0
    )
    tok = llm.get_tokenizer()
    t_load = time.time() - t0
    think_end = tok.convert_tokens_to_ids("</think>")
    stops = {think_end, tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    letters = [tok.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": j["message"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for j in jobs
    ]
    prompt_ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    params = [
        SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=1.0,
            max_tokens=CAP,
            stop_token_ids=[think_end],
            seed=int(j["vllm_seed"]),
        )
        for j in jobs
    ]
    t1 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], params)
    t_gen = time.time() - t1
    suffix_cut = tok.encode("\n</think>\n\nAnswer:", add_special_tokens=False)
    suffix_ok = [think_end, *tok.encode("\n\nAnswer:", add_special_tokens=False)]
    second, meta = [], []
    for p, o in zip(prompt_ids, outs, strict=False):
        g = list(o.outputs[0].token_ids)
        natural = o.outputs[0].stop_reason == think_end or (bool(g) and g[-1] == think_end)
        while g and g[-1] in stops:
            g = g[:-1]
        second.append(TokensPrompt(prompt_token_ids=p + g + (suffix_ok if natural else suffix_cut)))
        meta.append((g, natural))
    t2 = time.time()
    outs2 = llm.generate(second, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    t_ans = time.time() - t2
    rows = []
    import math

    for j, (g, natural), o in zip(jobs, meta, outs2, strict=False):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        raw = [math.exp(lp[i].logprob) if i in lp else 0.0 for i in letters]
        s = sum(raw)
        probs = [x / s for x in raw] if s > 0 else [0.25] * 4
        rows.append(
            {k: v for k, v in j.items() if k != "message"}
            | {
                "probs": [round(x, 5) for x in probs],
                "pred": "ABCD"[max(range(4), key=lambda i: probs[i])],
                "letter_mass": round(s, 5),
                "letters_missing": sum(1 for i in letters if i not in lp),
                "think_tokens": len(g) + (1 if natural else 0),
                "natural_end": natural,
                "trace": tok.decode(g) + ("</think>" if natural else ""),
            }
        )
    return {
        "rows": rows,
        "timing": {
            "load_s": round(t_load, 1),
            "gen_s": round(t_gen, 1),
            "answer_s": round(t_ans, 1),
            "total_s": round(time.time() - t0, 1),
            "generated_tokens": sum(len(m[0]) for m in meta),
        },
    }


@app.local_entrypoint()
def main(limit: int = 0, tag: str = "pilot"):
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))
    import natural as t

    items = t.subset(t.PILOT32_SUBSET, seed=32)
    jobs = t.jobs_for(items, t.FRAMINGS_CORE, (0, 1), arm="qwen3-32b")
    for j in jobs:
        j["message"] = t.user_message(j["id"], j["kind"], j["pair"])
        j["vllm_seed"] = t._seed_for(j["seed"], [j["key"]])
    if limit:
        jobs = jobs[:: max(1, len(jobs) // limit)][:limit]
    est = GPU_TIMEOUT_S / 3600 * PRICE_PER_HOUR
    print(
        f"{tag}: {len(jobs)} generations on {MODEL}; this call is capped at {GPU_TIMEOUT_S / 60:.0f} min "
        f"(at most ${est:.2f} at list price)"
    )
    t0 = time.time()
    res = generate.remote(jobs)
    wall = time.time() - t0
    out = t.WORK / "stage2" / f"qwen3_32b_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in res["rows"]:
            fh.write(json.dumps(r) + "\n")
    timing = res["timing"] | {
        "wall_s_from_laptop": round(wall, 1),
        "n": len(res["rows"]),
        "est_cost_usd": round(res["timing"]["total_s"] / 3600 * PRICE_PER_HOUR, 2),
    }
    t.write_json(t.WORK / "stage2" / f"timing_{tag}.json", timing)
    print(json.dumps(timing, indent=2))
    print(f"wrote {len(res['rows'])} rows to {t.rel(out)}")


@app.local_entrypoint()
def download():
    print(download_weights.remote())
