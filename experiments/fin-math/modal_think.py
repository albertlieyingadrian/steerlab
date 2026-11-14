"""Lever 1 (FinanceMath v2, validation only): Qwen3-8B base and our post-trained Qwen3-8B (SFT + GRPO step 10, LoRA)
in thinking and non-thinking mode, with vLLM on one L40S per arm (bf16, the same engine and LoRA serving as the registered non-thinking runs).

    MODAL_PROFILE=<profile> .venv/bin/python -m modal volume put fin-math experiments/fin-math/results/posttrain/grpo-s10 /adapters/grpo-s10
    MODAL_PROFILE=<profile> .venv/bin/python -m modal run experiments/fin-math/modal_think.py

4 samples per question at Qwen's recommended thinking sampling (temperature 0.6, top_p 0.95, top_k 20,
repetition_penalty 1.0), up to 16,384 new tokens. Writes results/runs/validation__l1-<tag>.jsonl (one row per sample,
graded here with this project's grader on the text after </think>). Cost bound: 4 arms x 100 L40S-minutes at most.
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen3-8B"
app = modal.App("fin-math-think")
volume = modal.Volume.from_name("fin-math", create_if_missing=True)
ENV = {"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub")
    .env(ENV)
)
THINK = {"n": 4, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 16384}
PLAIN = {
    "n": 4,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "max_tokens": 6144,
}  # registered
RUNS = [  # tag, adapter, thinking, protocol
    ["base-think", None, True, THINK],
    ["grpo-s10-think", "/vol/adapters/grpo-s10", True, THINK],
    ["base-nothink", None, False, PLAIN],
    ["grpo-s10-nothink", "/vol/adapters/grpo-s10", False, PLAIN],
]


@app.function(image=vllm_image, gpu="L40S", volumes={"/vol": volume}, timeout=100 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def generate(
    prompts: list[str], tag: str, adapter: str | None, thinking: bool, protocol: dict, chunk: int = 50
) -> list:
    """One arm on one L40S; each chunk of questions is saved to the volume as it finishes (a timeout keeps them)."""
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=thinking) for p in prompts]  # fmt: skip
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=18432 if thinking else 8192, enable_lora=True,
              max_lora_rank=16, seed=0, gpu_memory_utilization=0.92)  # fmt: skip
    params = SamplingParams(n=protocol["n"], temperature=protocol["temperature"], top_p=protocol["top_p"],
                            top_k=protocol["top_k"], repetition_penalty=protocol["repetition_penalty"],
                            max_tokens=protocol["max_tokens"])  # fmt: skip
    lora = LoRARequest(tag, 1, adapter) if adapter else None
    saved = Path(f"/vol/l1/{tag}.json")
    out, t0 = (json.loads(saved.read_text()) if saved.exists() else []), time.time()  # resume: keep finished questions
    print(f"{tag}: resuming at {len(out)}/{len(texts)}", flush=True)
    for b in range(len(out), len(texts), chunk):
        gen = llm.generate(texts[b : b + chunk], params, lora_request=lora)
        out += [[{"text": o.text, "tokens_out": len(o.token_ids), "capped": o.finish_reason == "length"}
                 for o in g.outputs] for g in gen]  # fmt: skip
        Path(f"/vol/l1/{tag}.json").parent.mkdir(parents=True, exist_ok=True)
        Path(f"/vol/l1/{tag}.json").write_text(json.dumps(out))
        volume.commit()
        print(f"{tag}: {len(out)}/{len(texts)} in {time.time() - t0:.0f} s", flush=True)
    return out


@app.function(image=vllm_image, gpu="L40S", volumes={"/vol": volume}, timeout=45 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def forced(
    prompts: list[str], tag: str, adapter: str, cap: int = 6144, answer_tokens: int = 32, chunk: int = 25
) -> list:
    """Budget forcing (PREREG 2026-11-12): think for at most `cap` tokens; a sample that hits the cap gets the think
    block closed and `ANSWER: ` appended, then at most `answer_tokens` more tokens at the same sampling."""
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True) for p in prompts]  # fmt: skip
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=cap + 2048, enable_lora=True, max_lora_rank=16, seed=0,
              gpu_memory_utilization=0.92)  # fmt: skip
    base = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0}
    p1 = SamplingParams(n=4, max_tokens=cap, **base)
    p2 = SamplingParams(n=1, max_tokens=answer_tokens, **base)
    lora = LoRARequest(tag, 1, adapter)
    saved = Path(f"/vol/l1/{tag}.json")
    out, t0 = (json.loads(saved.read_text()) if saved.exists() else []), time.time()
    for b in range(len(out), len(texts), chunk):
        gen = llm.generate(texts[b : b + chunk], p1, lora_request=lora)
        rows, follow = [], []
        for k, g in enumerate(gen):
            rows.append([])
            for o in g.outputs:
                r = {
                    "text": o.text,
                    "tokens_out": len(o.token_ids),
                    "capped": o.finish_reason == "length",
                    "forced": False,
                }
                if r["capped"]:
                    suffix = "\n</think>\n\nANSWER: " if "</think>" not in o.text else "\n\nANSWER: "
                    r["text"] += suffix
                    r["forced"] = True
                    follow.append((k, len(rows[-1]), texts[b + k] + r["text"]))
                rows[-1].append(r)
        if follow:
            cont = llm.generate([f[2] for f in follow], p2, lora_request=lora)
            for (k, j, _), c in zip(follow, cont, strict=True):
                rows[k][j]["text"] += c.outputs[0].text
                rows[k][j]["tokens_out"] += len(c.outputs[0].token_ids)
        out += rows
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(out))
        volume.commit()
        print(
            f"{tag}: {len(out)}/{len(texts)} in {time.time() - t0:.0f} s; forced {len(follow)} this chunk", flush=True
        )
    return out


def grade(items, tag: str, res: list, limit: int = 0) -> None:
    import fm

    # isort: split
    import fin

    path = fm.RESULTS / "runs" / f"validation{'-' + str(limit) if limit else ''}__l1-{tag}.jsonl"
    thinking = not tag.endswith("nothink")
    with path.open("w") as f:
        for item, samples in zip(items, res, strict=False):
            for s in samples:
                text = s["text"]
                ans = text.split("</think>")[-1] if "</think>" in text else ("" if thinking else text)
                pred = fin.final_answer(ans)
                value = fm.parse(pred) if pred is not None else None
                ok = value is not None and fm.grade(repr(value), item.gold, item.gold_exec)
                f.write(json.dumps({"id": item.id, "pred": pred, "correct": ok, "tokens_out": s["tokens_out"],
                                    "capped": s["capped"], "forced": s.get("forced", False), "text": text}) + "\n")  # fmt: skip
    print(tag, len(res), "questions ->", path)


@app.local_entrypoint()
def main(limit: int = 0, arms: str = "", fetched: str = ""):
    """Run (resumable, with `modal run --detach`), or grade arms already fetched with `modal volume get fin-math /l1`."""
    import sys

    sys.path.insert(0, str(HERE))
    import fm

    items = fm.load("validation")[: limit or None]
    runs = [r for r in RUNS if not arms or r[0] in arms.split(",")]
    if fetched:
        for tag, *_ in runs:
            p = Path(fetched) / f"{tag}.json"
            if p.exists():
                grade(items, tag, json.loads(p.read_text()), limit)
        return
    prompts = [fm.prompt(i) for i in items]
    calls = {tag: generate.spawn(prompts, tag, adapter, thinking, proto) for tag, adapter, thinking, proto in runs}
    for tag, call in calls.items():
        grade(items, tag, call.get(), limit)
