"""
Open vision-language judges for "template slop vs purpose-made", on Modal (vLLM), reading Ep 2's screenshots
read-only.

COSTS MONEY (one H100 per judge, ~$4/h). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/slop-probes/modal_judge.py --model qwen72 --stage dev
    ... --stage held --prompt <the prompt chosen on dev>

Stages:
  dev    dev main-task pages, masked, both prompts (plain, constitution): the dev AUROC picks one prompt per judge
  held   held-out main-task pages masked and unmasked, and every known-answer page masked, with the chosen prompt
The score is the expected value of the first answer token over the digits 0-9 (top-20 log-probs, renormalised);
"mass" is how much of the top-20 probability the digits hold (the known-answer check also requires >= 0.9).
Rows are appended to ~/.cache/steerlab/slop-probes/judge/<model>.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ep3 import MASK_BOX, MASK_GREY

VOL = "/vol"
app = modal.App("taste-ep3-judge")
ep2_volume = modal.Volume.from_name("design-index").read_only()
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9", "pillow==11.0.0")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
    .add_local_file(HERE / "ep3.py", "/root/ep3.py")
)
MODELS = {
    "qwen72": {"id": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "kw": {"mm_processor_kwargs": {"max_pixels": 1280 * 800}}},
    "gemma27": {"id": "google/gemma-3-27b-it", "kw": {}},
}

INTRO = (
    "You are a senior web designer reviewing the first screen of a website's homepage (desktop, 1280x800). "
    "Some pages are template slop: generic designs produced by a default (an AI site builder or a stock template), "
    "whose layout, copy and styling could belong to any business. Others are purpose-made: the design choices come "
    "from what this particular business is and does. A grey box may cover the bottom-right corner; ignore it."
)
ASK = "Rate this page from 0 (clearly purpose-made) to 9 (clearly template slop). Answer with a single digit only."


def prompts() -> dict:
    from ep3 import parse_constitution

    rubric = [f"{p['name']}. Slop: {p['slop']} Purpose-made: {p['good']}" for p in parse_constitution()]
    return {
        "plain": f"{INTRO}\n\n{ASK}",
        "constitution": f"{INTRO}\n\nUse this rubric:\n" + "\n".join(f"- {r}" for r in rubric) + f"\n\n{ASK}",
    }


@app.function(
    image=image,
    gpu="H100",
    volumes={VOL: ep2_volume, "/hf": hf},
    timeout=3 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def judge(model_key: str, jobs: list[dict], prompt_text: dict) -> list[dict]:
    import base64
    import io
    import math

    from PIL import Image, ImageDraw
    from vllm import LLM, SamplingParams

    cfg = MODELS[model_key]
    llm = LLM(
        cfg["id"],
        max_model_len=6144,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 1},
        **cfg["kw"],
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    digits = {str(d): d for d in range(10)}

    def url(pid, mask):
        im = Image.open(f"{VOL}/shots/{pid}.jpg").convert("RGB")
        if mask:
            ImageDraw.Draw(im).rectangle(MASK_BOX, fill=MASK_GREY)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    out = []
    for i in range(0, len(jobs), 256):
        batch = [j for j in jobs[i : i + 256] if Path(f"{VOL}/shots/{j['pid']}.jpg").exists()]
        msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url(j["pid"], j["mask"])}},
                        {"type": "text", "text": prompt_text[j["prompt"]]},
                    ],
                }
            ]
            for j in batch
        ]
        res = llm.chat(msgs, sp, use_tqdm=False)
        for j, r in zip(batch, res):
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            p = [0.0] * 10
            total = 0.0
            for tok in lp.values():
                pr = math.exp(tok.logprob)
                total += pr
                s = (tok.decoded_token or "").strip()
                if s in digits:
                    p[digits[s]] += pr
            mass = sum(p)
            score = sum(d * x for d, x in enumerate(p)) / mass if mass > 0 else float("nan")
            out.append({**j, "score": score, "mass": mass, "top_mass": total, "p": p, "text": r.outputs[0].text})
        print(f"{model_key}: {i + len(batch)}/{len(jobs)}", flush=True)
    return out


@app.local_entrypoint()
def main(model: str, stage: str, prompt: str = ""):
    from budget import check_modal
    from ep3 import CACHE, load_pages

    pages = load_pages()
    pt = prompts()
    if stage == "dev":
        sel = [p for p in pages if p["y_main"] is not None and p["split"] == "dev"]
        jobs = [{"pid": p["pid"], "mask": True, "prompt": k} for k in pt for p in sel]
    elif stage == "held":
        assert prompt in pt, "pass --prompt (chosen on dev)"
        main_ = [p for p in pages if p["y_main"] is not None and p["split"] == "held"]
        known = [p for p in pages if p["y_known"] is not None]
        jobs = [{"pid": p["pid"], "mask": m, "prompt": prompt} for m in (True, False) for p in main_]
        jobs += [{"pid": p["pid"], "mask": True, "prompt": prompt} for p in known]
    elif stage == "smoke":
        sel = [p for p in pages if p["y_main"] is not None and p["split"] == "dev"][:24]
        jobs = [{"pid": p["pid"], "mask": True, "prompt": k} for k in pt for p in sel]
    else:
        raise SystemExit(f"unknown stage {stage}")
    out_dir = CACHE / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["pid"], r["mask"], r["prompt"]))
    jobs = [j for j in jobs if (j["pid"], j["mask"], j["prompt"]) not in done]
    print(f"{model} {stage}: {len(jobs)} new jobs")
    if not jobs:
        return
    rate = {"qwen72": 2.5, "gemma27": 5.0}[model]  # pages/s (guess, checked in the smoke stage)
    check_modal(4.0 * (len(jobs) / rate / 3600 + 0.2), f"judge {model} {stage} ({len(jobs)} pages)")
    rows = judge.remote(model, jobs, pt)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bad = sum(r["mass"] < 0.9 for r in rows)
    print(f"saved {len(rows)} rows; {bad} with digit mass < 0.9")
