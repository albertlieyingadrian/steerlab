"""Sample the open models on Modal: 50 answers per question and condition at temperature 1, max 32 tokens.

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run modal_sample.py::main --set main   # data/questions.json
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run modal_sample.py::main --set dev    # data/dev_questions.json

One L40S per model, vLLM 0.25.1 (inference only, so the fla backward bug does not apply). Chat models get the prompt
as one user message with thinking off (enable_thinking=False); base models get the same text plus "\nAnswer:".
Sampling: temperature 1.0, top_p 1, top_k off, repetition_penalty 1.0, and vLLM's own defaults rather than the
model's generation_config.json (which would add top_p/top_k and a repetition penalty).

Writes results/samples/<set>/<model>.jsonl.gz: one row per (question, condition) with the 50 raw replies.
Modal spend is capped in code (MODAL_CAP_USD): each run's GPU time is logged to results/modal_spend.jsonl, and a
run is refused if the logged spend plus the worst case of the runs about to start would pass the cap.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODELS = {  # short name -> (HF id, chat?)
    "qwen3.5-9b-base": ("Qwen/Qwen3.5-9B-Base", False),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", True),
    "sea-lion-v5-9b": ("aisingapore/Qwen-SEA-LION-v5-9B", True),
    "qwen3-8b-base": ("Qwen/Qwen3-8B-Base", False),
    "qwen3-8b": ("Qwen/Qwen3-8B", True),
}
MODAL_CAP_USD = 30.0
USD_PER_HOUR = 2.50  # L40S list price is ~$1.95/h; rounded up so the cap errs safe
TIMEOUT_S = 40 * 60
LEDGER = HERE / "results" / "modal_spend.jsonl"

app = modal.App("pluralism-explainer")
cache = modal.Volume.from_name("pluralism-explainer-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

GEN = r"""
import json, sys
from vllm import LLM, SamplingParams

def main():
    job = json.load(open(sys.argv[1]))
    kw = dict(dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, generation_config="vllm", seed=0)
    if "3.5" in job["hf"] or "SEA-LION" in job["hf"]:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(job["hf"], **kw)
    sp = SamplingParams(n=job["n"], temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                        max_tokens=32, seed=0)
    if job["chat"]:
        outs = llm.chat([[{"role": "user", "content": t}] for t in job["texts"]], sp,
                        chat_template_kwargs={"enable_thinking": False})
    else:
        outs = llm.generate(job["texts"], sp)
    json.dump([[c.text for c in o.outputs] for o in outs], open(sys.argv[2], "w"))

if __name__ == "__main__":
    main()
"""


@app.function(image=image, gpu="L40S", volumes={"/cache": cache}, timeout=TIMEOUT_S, max_containers=3)
def sample(hf: str, chat: bool, texts: list[str], n: int) -> dict:
    import subprocess

    t0 = time.time()
    Path("/tmp/gen.py").write_text(GEN)
    Path("/tmp/job.json").write_text(json.dumps({"hf": hf, "chat": chat, "texts": texts, "n": n}))
    subprocess.run(["python", "/tmp/gen.py", "/tmp/job.json", "/tmp/out.json"], check=True)
    cache.commit()
    return {"replies": json.loads(Path("/tmp/out.json").read_text()), "seconds": time.time() - t0}


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in LEDGER.read_text().splitlines() if x.strip())


@app.local_entrypoint()
def main(set: str = "dev", models: str = "", n: int = 50):
    import sys

    sys.path.insert(0, str(HERE))
    import plural

    qfile = HERE / "data" / ("questions.json" if set == "main" else "dev_questions.json")
    qs = plural.load_questions(qfile)
    conds = plural.CONDITIONS if set == "main" else ["default", "Indonesia", "United States"]
    names = [m for m in (models.split(",") if models else MODELS)]
    worst = len(names) * TIMEOUT_S / 3600 * USD_PER_HOUR
    if spent() + worst > MODAL_CAP_USD:
        raise SystemExit(f"Modal cap: spent ${spent():.2f} + worst case ${worst:.2f} > ${MODAL_CAP_USD}")
    cells = [(q, c) for q in qs for c in conds]
    jobs = []
    for name in names:
        hf, chat = MODELS[name]
        fn = plural.prompt_text if chat else plural.base_text
        jobs.append((hf, chat, [fn(q, c) for q, c in cells], n))
    out_dir = HERE / "results" / "samples" / set
    out_dir.mkdir(parents=True, exist_ok=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    for name, res in zip(names, sample.starmap(jobs), strict=True):
        usd = res["seconds"] / 3600 * USD_PER_HOUR
        with LEDGER.open("a") as f:
            f.write(
                json.dumps({"model": name, "set": set, "seconds": round(res["seconds"]), "usd": round(usd, 4)}) + "\n"
            )
        with gzip.open(out_dir / f"{name}.jsonl.gz", "wt") as f:
            for (q, c), replies in zip(cells, res["replies"], strict=True):
                f.write(json.dumps({"id": q["id"], "condition": c, "replies": replies}, ensure_ascii=False) + "\n")
        valid = sum(plural.parse(r, len(q["options"])) >= 0 for (q, _), rs in zip(cells, res["replies"]) for r in rs)
        total = sum(len(rs) for rs in res["replies"])
        print(f"{name}: {total} replies, valid {valid / total:.3f}, {res['seconds']:.0f} s, ${usd:.2f}")
    print(f"Modal spend so far (logged): ${spent():.2f} of ${MODAL_CAP_USD}")
