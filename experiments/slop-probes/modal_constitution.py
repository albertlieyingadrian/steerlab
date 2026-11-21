"""
Synthetic training pages from the taste constitution (the Constitutional Classifiers recipe), rendered and featurised
with Ep 2's code unchanged, on this episode's own Volume.

COSTS MONEY (one H100 for ~15 min; CPU and one A10G for render and features). Run from the repo root, in order:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/slop-probes/modal_constitution.py::generate
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/slop-probes/modal_constitution.py::render
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/slop-probes/modal_constitution.py::features

Per industry (12) and brief k (4, Ep 2's phrasings): 4 pages each told to commit one slop principle (0-based
principles 2k, 2k+1, 2k+4, 2k+5 mod 8, so each principle appears twice per industry) and 4 pages told to follow the
purpose-made side of all eight principles: 384 pages. Generator: Gemma-3-27B (a family not used for
Ep 2's generated class; also judge J2, which only scores real pages), T = 0.8, top-p 0.95, one seed per page (0-383). Briefs 0-2 train the constitution probe; brief 3 checks that
the two sides differ (the constitution-data check).
"""

from __future__ import annotations

import asyncio
import json
import pickle
import re
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EP2 = HERE.parent / "design-index"
VOL = "/vol3"
app = modal.App("taste-ep3-constitution")
volume = modal.Volume.from_name("taste-ep3", create_if_missing=True)
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
GEN_MODEL = "google/gemma-3-27b-it"  # Mistral-Small-3.2 returned garbage under vLLM 0.10.1.1 (run log 7)
gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
)
page_image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install(
        "playwright==1.48.0",
        "pillow==11.0.0",
        "numpy==2.1.3",
        "scikit-learn==1.5.2",
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "protobuf==5.29.0",
        "hf_transfer==0.1.9",
    )
    .run_commands("playwright install --with-deps chromium")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
    .add_local_file(EP2 / "extract.js", "/root/extract.js")
    .add_local_file(EP2 / "modal_render.py", "/root/ep2_render.py")
    .add_local_file(EP2 / "features.py", "/root/features.py")
    .add_local_file(EP2 / "common.py", "/root/common.py")
    .add_local_file(HERE / "ep3.py", "/root/ep3.py")
    .add_local_file(HERE / "modal_masked.py", "/root/modal_masked.py")
)

BASE = (
    "Build the homepage of {desc}. Invent a plausible name and real-sounding content for it. Return one "
    "self-contained HTML file with inline CSS (you may load Google Fonts). Do not use external images; draw any "
    "imagery with CSS shapes or inline SVG. Keep it under about 300 lines. Output only the HTML.\n\nDesign direction:\n{direction}"
)


def build_prompts() -> list[dict]:
    from ep3 import ep2, parse_constitution

    ep2_common, _ = ep2()
    principles = parse_constitution()
    good_dir = "\n".join(f"- {p['good']}" for p in principles)
    rows = []
    for cat, briefs in ep2_common.CATEGORIES.items():
        for k, b in enumerate(briefs):
            desc = b.replace("'s website", "").replace("'s homepage", "").replace("'s landing page", "")
            desc = desc.replace("the homepage of ", "")
            for s in [(2 * k) % 8, (2 * k + 1) % 8, (2 * k + 4) % 8, (2 * k + 5) % 8]:
                p = principles[s]
                rows.append(
                    {
                        "side": "slop",
                        "principle": p["id"],
                        "category": cat,
                        "brief": k,
                        "j": s,
                        "prompt": BASE.format(desc=desc, direction="- " + p["slop"]),
                    }
                )
            for j in range(4):
                rows.append(
                    {
                        "side": "good",
                        "principle": "all",
                        "category": cat,
                        "brief": k,
                        "j": j,
                        "prompt": BASE.format(desc=desc, direction=good_dir),
                    }
                )
    for r in rows:
        r["file"] = f"cst_{r['side']}_{r['category']}_{r['brief']}_{r['j']}.html"
    return rows


@app.function(
    image=gen_image,
    gpu="H100",
    volumes={VOL: volume, "/hf": hf},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def gen(rows: list[dict]) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        GEN_MODEL,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    # one seed per page: identical prompts with one shared seed would return identical pages
    sps = [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=7400, seed=i) for i in range(len(rows))]
    outs = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in rows], sps)
    Path(f"{VOL}/html").mkdir(parents=True, exist_ok=True)
    meta = []
    for r, o in zip(rows, outs):
        c = o.outputs[0]
        m = re.search(r"<!DOCTYPE html.*?</html>|<html.*?</html>", c.text, re.DOTALL | re.IGNORECASE)
        Path(f"{VOL}/html/{r['file']}").write_text(m.group(0) if m else c.text)
        meta.append({k: r[k] for k in ("file", "side", "principle", "category", "brief", "j")})
        meta[-1].update({"finished": c.finish_reason, "has_html": bool(m), "chars": len(c.text)})
    volume.commit()
    return meta


@app.function(image=page_image, volumes={VOL: volume}, cpu=4, memory=8192, timeout=3600, max_containers=4)
def render_chunk(jobs: list[dict]) -> list[dict]:
    from ep2_render import render_one
    from playwright.async_api import async_playwright

    out = Path(VOL)
    (out / "shots").mkdir(parents=True, exist_ok=True)
    (out / "dom").mkdir(parents=True, exist_ok=True)
    js = Path("/root/extract.js").read_text()

    async def run():
        res = []
        sem = asyncio.Semaphore(6)
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--disable-dev-shm-usage"])

            async def one(job):
                async with sem:
                    ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
                    try:
                        rec = await render_one(ctx, job, js, {"pause_until": 0.0, "next_nav": 0.0}, out)
                    finally:
                        await ctx.close()
                    res.append({k: rec.get(k) for k in ("pid", "ok", "status", "error")})

            await asyncio.gather(*(one(j) for j in jobs))
            await browser.close()
        return res

    res = asyncio.run(run())
    volume.commit()
    return res


@app.function(
    image=page_image,
    gpu="A10G",
    cpu=8,
    memory=16384,
    volumes={VOL: volume, "/hf": hf},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def featurise(pids: list[str]) -> dict:
    """Ep 2's hand features + SigLIP/UIClip, unmasked and masked (the same mask as the real pages)."""
    import numpy as np
    import torch
    from features import page_features
    from PIL import Image, ImageDraw
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    from ep3 import MASK_BOX, MASK_GREY
    from modal_masked import SIGLIP, UICLIP, UICLIP_PROC, strip_badge

    sig = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
    sig_p = AutoProcessor.from_pretrained(SIGLIP)
    uic = CLIPModel.from_pretrained(UICLIP, torch_dtype=torch.float16).cuda().eval()
    uic_p = CLIPProcessor.from_pretrained(UICLIP_PROC)
    norm = lambda x: (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16)
    out = {}
    for p in pids:
        try:
            rec = json.loads(Path(f"{VOL}/dom/{p}.json").read_text())
            if not rec.get("ok"):
                continue
            im = Image.open(f"{VOL}/shots/{p}.jpg").convert("RGB")
            m = im.copy()
            ImageDraw.Draw(m).rectangle(MASK_BOX, fill=MASK_GREY)
            row = page_features(rec, im)
            row["hand_mask"] = page_features(strip_badge(rec)[0], m)
            row["pixel_std"] = rec.get("pixel_std")
            row["status"] = rec.get("status")
            with torch.no_grad():
                for tag, img in (("raw", im), ("mask", m)):
                    a = sig_p(images=[img], return_tensors="pt")["pixel_values"].cuda().half()
                    b = uic_p(images=[img], return_tensors="pt")["pixel_values"].cuda().half()
                    row[f"sig_{tag}"] = norm(sig.get_image_features(pixel_values=a))[0]
                    row[f"uic_{tag}"] = norm(uic.get_image_features(pixel_values=b))[0]
            out[p] = row
        except Exception as e:  # noqa: BLE001 - one bad page must not sink the batch
            print(p, e.__class__.__name__, str(e)[:120])
    return out


def _meta_path():
    from ep3 import CACHE

    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / "cst_meta.jsonl"


@app.local_entrypoint()
def generate(limit: int = 0):
    from budget import check_modal

    rows = build_prompts()
    if limit:  # smoke test: print the first pages instead of saving metadata
        rows = rows[:: len(rows) // limit][:limit]
        for r in gen.remote(rows):
            print(r["file"], r["finished"], r["has_html"], r["chars"])
        return
    print(f"{len(rows)} prompts; example:\n{rows[0]['prompt'][:600]}")
    check_modal(3.0, f"constitution generation ({len(rows)} pages, H100)")
    t0 = time.time()
    meta = gen.remote(rows)
    from ep3 import ep2

    ep2_common, _ = ep2()
    for r in meta:
        r["pid"] = ep2_common.page_id("cst", r["file"])
    _meta_path().write_text("".join(json.dumps(r) + "\n" for r in meta))
    cut = sum(r["finished"] == "length" for r in meta)
    print(f"{len(meta)} pages in {time.time() - t0:.0f}s; {sum(r['has_html'] for r in meta)} with html; {cut} cut off")


@app.local_entrypoint()
def render():
    from budget import check_modal

    meta = [json.loads(line) for line in _meta_path().read_text().splitlines() if line.strip()]
    jobs = [
        {"pid": r["pid"], "mode": "file", "url": f"file://{VOL}/html/{r['file']}", "domain": r["file"]} for r in meta
    ]
    check_modal(0.3, f"render {len(jobs)} constitution pages")
    res = []
    for part in render_chunk.map([jobs[i : i + 100] for i in range(0, len(jobs), 100)]):
        res += part
    print(f"{sum(bool(r['ok']) for r in res)}/{len(res)} rendered ok")


@app.local_entrypoint()
def features():
    from budget import check_modal
    from ep3 import CACHE

    meta = [json.loads(line) for line in _meta_path().read_text().splitlines() if line.strip()]
    pids = [r["pid"] for r in meta]
    check_modal(0.3, f"features for {len(pids)} constitution pages")
    out = {}
    for part in featurise.map([pids[i : i + 200] for i in range(0, len(pids), 200)]):
        out.update(part)
    (CACHE / "constitution.pkl").write_bytes(pickle.dumps(out))
    print(f"saved {len(out)} pages")
