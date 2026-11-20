"""
Generate single-file landing pages with open code models on Modal (the second AI-built class, with a known source).

COSTS MONEY (one H100, ~$4/h). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/design-index/modal_generate.py

For each of 2 models x 12 categories x 2 phrasings x 5 samples (T = 0.8, top-p 0.95): a one-shot prompt like the ones
people give site builders ("Build a landing page for ..."), no design guidance. HTML is saved to the Volume at
html/gen_<model>_<category>_<i>.html; render jobs are written to CACHE/jobs_gen.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VOL = "/vol"
app = modal.App("design-index-generate")
volume = modal.Volume.from_name("design-index", create_if_missing=True)
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env(
        {"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"}
    )
    .add_local_file(HERE / "common.py", "/root/common.py")
)
MODELS = {
    "qwencoder": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "qwen3": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}
N_SAMPLES = 5
PROMPT = (
    "Build a complete, modern landing page for {desc}. Return one self-contained HTML file with inline CSS "
    "(you may use a CDN for fonts or Tailwind). Use placeholder shapes instead of photos. Output only the HTML."
)


@app.function(
    image=image,
    gpu="H100",
    volumes={VOL: volume, "/hf": hf},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(model_key: str, prompts: list[dict]) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(MODELS[model_key], max_model_len=8192, gpu_memory_utilization=0.9)
    sp = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=6000, n=N_SAMPLES, seed=0
    )
    msgs = [
        [{"role": "user", "content": PROMPT.format(desc=p["desc"])}] for p in prompts
    ]
    outs = llm.chat(msgs, sp)
    Path(f"{VOL}/html").mkdir(parents=True, exist_ok=True)
    rows = []
    for p, o in zip(prompts, outs):
        for j, c in enumerate(o.outputs):
            text = c.text
            m = re.search(
                r"<!DOCTYPE html.*?</html>|<html.*?</html>",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            html = m.group(0) if m else text
            fn = f"gen_{model_key}_{p['category']}_{p['k']}_{j}.html"
            Path(f"{VOL}/html/{fn}").write_text(html)
            rows.append(
                {
                    "file": fn,
                    "category": p["category"],
                    "model": model_key,
                    "finished": c.finish_reason,
                    "has_html": bool(m),
                    "chars": len(html),
                }
            )
    volume.commit()
    return rows


@app.local_entrypoint()
def main():
    from budget import check_modal
    from common import CACHE, CATEGORIES, page_id

    check_modal(2.0 * len(MODELS) * 0.5, "generate AI pages")
    prompts = [
        {
            "category": c,
            "k": k,
            "desc": qs[k]
            .replace("'s website", "")
            .replace("'s homepage", "")
            .replace("'s landing page", "")
            .replace("the homepage of ", ""),
        }
        for c, qs in CATEGORIES.items()
        for k in (0, 1)
    ]
    jobs, meta = [], []
    for key in MODELS:
        rows = generate.remote(key, prompts)
        meta += rows
        for r in rows:
            jobs.append(
                {
                    "pid": page_id("gen", r["file"]),
                    "mode": "file",
                    "url": f"file:///vol/html/{r['file']}",
                    "domain": r["file"],
                    "model": r["model"],
                    "category": r["category"],
                }
            )
        print(
            key,
            sum(r["has_html"] for r in rows),
            "/",
            len(rows),
            "with html;",
            sum(r["finished"] == "length" for r in rows),
            "cut off",
        )
    (CACHE / "jobs_gen.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (CACHE / "gen_meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in meta))
