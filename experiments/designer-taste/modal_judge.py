"""
Open vision-language models on Modal (vLLM) for The Taste Stack Episode 7: a generic, resumable runner.

COSTS MONEY (H100 at ~$4/h). Other scripts (judges.py, constitution.py) write a jobs file; this runs it:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/designer-taste/modal_judge.py \
        --model qwen7 --jobs ~/.cache/steerlab/taste-ep7/jobs/<name>.jsonl

A job: {"id", "imgs": [asset id or "deg:<asset id>"], "pre": text before the images, "ask": text after,
        "mode": "score" (one token; P("1") vs P("2") from the top-20 log-probs) or "gen" (greedy text)}.
Images are the 640 px JPEGs on the "taste-ep7" Volume (made by modal_features.py); "deg:" is the same image blurred
(Gaussian radius 6) and re-saved at JPEG quality 10 (the R5 known-answer pairs).
Results go to ~/.cache/steerlab/taste-ep7/vlm/<model>.jsonl (append; ids already there are skipped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VOL = "/vol"
app = modal.App("taste-ep7-vlm")
vol = modal.Volume.from_name("taste-ep7", create_if_missing=True)
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9", "pillow==11.0.0")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
)
MODELS = {
    "qwen7": {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "rate": 12.0,
        "kw": {"mm_processor_kwargs": {"max_pixels": 640 * 640}},
    },
    "qwen72": {
        "id": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "rate": 1.5,
        "kw": {"mm_processor_kwargs": {"max_pixels": 640 * 640}},
    },
    "gemma27": {"id": "google/gemma-3-27b-it", "rate": 5.0, "kw": {}},  # second family (R6); Gemma sees 896 px tiles
}
PRICE_H100 = 3.95


@app.function(
    image=image,
    gpu="H100",
    volumes={VOL: vol, "/hf": hf},
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_chunk(model_key: str, jobs: list[dict]) -> list[dict]:
    import base64
    import io
    import math
    import time

    from PIL import Image, ImageFilter
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        MODELS[model_key]["id"],
        max_model_len=6144,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2},
        **MODELS[model_key]["kw"],
    )
    load_s = time.time() - t0
    cache = {}

    def url(ref):
        if ref in cache:
            return cache[ref]
        s = str(ref)
        deg = s.startswith("deg:")
        aid = s[4:] if deg else s
        im = Image.open(f"{VOL}/jpg640/{aid}.jpg").convert("RGB")
        if deg:
            im = im.filter(ImageFilter.GaussianBlur(6))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=10 if deg else 92)
        cache[ref] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return cache[ref]

    def msg(j):
        content = [{"type": "text", "text": j["pre"]}]
        for k, ref in enumerate(j["imgs"]):
            content.append({"type": "text", "text": f"\nImage {k + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": url(ref)}})
        content.append({"type": "text", "text": "\n" + j["ask"]})
        return [{"role": "user", "content": content}]

    sp_score = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    sp_gen = SamplingParams(temperature=0.0, max_tokens=80)
    out = []
    t1 = time.time()
    for mode, sp in (("score", sp_score), ("gen", sp_gen)):
        sel = [j for j in jobs if j.get("mode", "score") == mode]
        for i in range(0, len(sel), 128):
            batch = sel[i : i + 128]
            res = llm.chat([msg(j) for j in batch], sp, use_tqdm=False)
            for j, r in zip(batch, res):
                o = r.outputs[0]
                row = {"id": j["id"], "text": o.text}
                if mode == "score":
                    p = {"1": 0.0, "2": 0.0}
                    tot = 0.0
                    for tok in (o.logprobs[0] if o.logprobs else {}).values():
                        pr = math.exp(tok.logprob)
                        tot += pr
                        s = (tok.decoded_token or "").strip()
                        if s in p:
                            p[s] += pr
                    m = p["1"] + p["2"]
                    row.update({"p1": p["1"] / m if m > 0 else 0.5, "mass": m, "top_mass": tot})
                out.append(row)
            print(f"{model_key} {mode}: {i + len(batch)}/{len(sel)}", flush=True)
    gen_s = time.time() - t1
    for row in out:
        row["gpu_s"] = gen_s / max(len(out), 1)
    out.append({"id": "__timing__", "load_s": load_s, "gen_s": gen_s, "n": len(out)})
    return out


@app.local_entrypoint()
def main(model: str, jobs: str, chunks: int = 1):
    from budget import check_modal
    from taste import CACHE

    path = Path(jobs).expanduser()
    todo = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out_dir = CACHE / "vlm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines() if line.strip()}
    todo = [j for j in todo if j["id"] not in done]
    print(f"{model}: {len(todo)} new jobs from {path.name}")
    if not todo:
        return
    rate = MODELS[model]["rate"]
    load_h = 0.12 if model == "qwen72" else 0.05
    projected = PRICE_H100 * (len(todo) / rate / 3600 + load_h * chunks) * 1.15
    check_modal(projected, f"vlm {model} {path.name} ({len(todo)} jobs, {chunks} chunks)")
    parts = [todo[i::chunks] for i in range(chunks)]
    timing = out_dir / "timing.jsonl"
    for rows in run_chunk.map([model] * chunks, parts, order_outputs=False):
        with out_path.open("a") as f, timing.open("a") as ft:
            for r in rows:
                if r["id"] == "__timing__":
                    ft.write(json.dumps({"model": model, "jobs": path.name, **r}) + "\n")
                else:
                    f.write(json.dumps(r) + "\n")
        print(f"saved a chunk of {len(rows) - 1}", flush=True)
