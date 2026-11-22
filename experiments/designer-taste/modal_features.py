"""
Per-image features for The Taste Stack Episode 7, on Modal. Images never leave Modal except as 640 px JPEGs for the
frontier judge (saved to the local cache, never committed).

COSTS MONEY (one A10G for ~15 min, well under $1). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/designer-taste/modal_features.py

Steps (one container):
  fetch   TASTE images from the Hub (pinned revision) into the "taste-ep7" Volume; a 640 px JPEG of each beside it
  embed   SigLIP-so400m-384 and SigLIP-base-224: image embedding, and cosine to each sentence of the scene's brief
  ocr     RapidOCR text and boxes on the full image (long side 1600 px)
  stats   brightness (mean L*), contrast (std L*), colourfulness (opponent-colour spread), sharpness (Laplacian variance),
          edge density, luminance entropy; a 96x96 RGB thumbnail for the colour checks
  r3      the OCR known-answer set: dev briefs' quoted strings rendered correctly and with one character changed
Saved to ~/.cache/steerlab/taste-ep7/features.pkl and ~/.cache/steerlab/taste-ep7/jpg640/.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VOL = "/vol"
app = modal.App("taste-ep7-features")
vol = modal.Volume.from_name("taste-ep7", create_if_missing=True)
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core fonts-liberation")
    .pip_install(
        "numpy==1.26.4",
        "pillow==11.0.0",
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "protobuf==5.29.0",
        "hf_transfer==0.1.9",
        "huggingface_hub==0.26.2",
        "opencv-python-headless==4.10.0.84",
        "scikit-image==0.24.0",
        "pandas==2.2.3",
        "pyarrow==18.0.0",
    )
    # RapidOCR pulls in full opencv-python (needs libGL); install it without deps beside the headless build
    .pip_install("onnxruntime==1.20.1", "pyclipper==1.3.0.post6", "shapely==2.0.6", "six==1.16.0", "PyYAML==6.0.2")
    .run_commands("pip install --no-deps rapidocr-onnxruntime==1.3.24")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
)
MODELS = {"so400m": "google/siglip-so400m-patch14-384", "base": "google/siglip-base-patch16-224"}


def sentences(brief: str) -> list[str]:
    import re

    body = re.sub(r"\*+", " ", brief)
    body = re.sub(r"(User Intent|Aesthetics|Description):", " ", body)
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(s.strip()) > 12]
    return parts[:40]


@app.function(image=image, gpu="A10G", volumes={VOL: vol, "/hf": hf}, timeout=2 * 3600)
def run(scenes: list[dict], r3_items: list[dict], rev: str) -> dict:
    import io
    import math

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    from rapidocr_onnxruntime import RapidOCR
    from skimage.color import rgb2lab
    from transformers import AutoModel, AutoProcessor

    root = Path(VOL) / "taste"
    if not (root / "assets.parquet").exists():
        snapshot_download("purvanshi/TASTE", repo_type="dataset", revision=rev, local_dir=root, max_workers=16)
        vol.commit()
    assets = pd.read_parquet(root / "assets.parquet")
    path_of = dict(zip(assets.asset_id.astype(int), assets.image_path))
    jdir = Path(VOL) / "jpg640"
    jdir.mkdir(exist_ok=True)
    out = {"img": {}, "jpg": {}, "r3": []}
    ids = sorted(path_of)

    def load(aid):
        return Image.open(root / path_of[aid]).convert("RGB")

    # 640 px JPEGs (judges) + stats + thumbnails
    for aid in ids:
        im = load(aid)
        w, h = im.size
        s = 640 / max(w, h)
        small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=90)
        (jdir / f"{aid}.jpg").write_bytes(buf.getvalue())
        out["jpg"][aid] = buf.getvalue()
        arr = np.asarray(small).astype(np.float32) / 255.0
        lab = rgb2lab(arr)
        L = lab[..., 0]
        rg = arr[..., 0] - arr[..., 1]
        yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
        colourful = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        hist = np.bincount(gray.ravel(), minlength=256) / gray.size
        ent = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
        out["img"][aid] = {
            "size": (w, h),
            "stats": [
                float(L.mean()),
                float(L.std()),
                float(colourful),
                float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                float((cv2.Canny(gray, 100, 200) > 0).mean()),
                ent,
            ],
            "thumb": np.asarray(im.resize((96, 96), Image.BILINEAR), dtype=np.uint8),
        }
    vol.commit()
    print("jpg+stats done", flush=True)

    # OCR on the full image, long side 1600
    ocr = RapidOCR()
    for aid in ids:
        im = load(aid)
        w, h = im.size
        s = min(1.0, 1600 / max(w, h))
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        res, _ = ocr(np.asarray(im))
        boxes = []
        for b, txt, conf in res or []:
            ys = [p[1] for p in b]
            xs = [p[0] for p in b]
            boxes.append(
                {"t": txt, "c": float(conf), "y": float(np.mean(ys)) / im.size[1], "x": float(np.mean(xs)) / im.size[0]}
            )
        out["img"][aid]["ocr"] = boxes
    print("ocr done", flush=True)

    # R3 OCR known-answer images: render each string (correct or misspelled) on a noisy background
    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    rng = np.random.default_rng(7)
    for it in r3_items:
        canvas = Image.fromarray((rng.random((900, 900, 3)) * 60 + it["bg"]).clip(0, 255).astype(np.uint8))
        canvas = canvas.filter(ImageFilter.GaussianBlur(2))
        d = ImageDraw.Draw(canvas)
        y = 80
        for line in it["lines"]:
            f = ImageFont.truetype(fonts[it["font"]], it["size"])
            d.text((60, y), line, font=f, fill=tuple(it["fg"]))
            y += int(it["size"] * 1.6)
        res, _ = ocr(np.asarray(canvas))
        boxes = [
            {"t": t, "c": float(c), "y": float(np.mean([p[1] for p in b])) / 900, "x": 0.0} for b, t, c in res or []
        ]
        out["r3"].append({**{k: it[k] for k in ("id", "variant")}, "ocr": boxes})
    print("r3 done", flush=True)

    # SigLIP embeddings: image + brief sentences
    dev = "cuda"
    for key, mid in MODELS.items():
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(mid)
        emb = {}
        for i in range(0, len(ids), 32):
            chunk = ids[i : i + 32]
            ims = [load(a) for a in chunk]
            with torch.no_grad():
                x = proc(images=ims, return_tensors="pt").to(dev)
                e = model.get_image_features(pixel_values=x["pixel_values"].half())
            e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
            for a, v in zip(chunk, e):
                emb[a] = v
        sims = {}
        for sc in scenes:
            sents = sentences(sc["brief"])
            with torch.no_grad():
                t = proc(text=sents, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(dev)
                te = model.get_text_features(input_ids=t["input_ids"])
            te = torch.nn.functional.normalize(te.float(), dim=-1).cpu().numpy()
            tmean = te.mean(0)
            tmean /= np.linalg.norm(tmean)
            for a in sc["assets"]:
                cs = te @ emb[a]
                sims[a] = {"mean": float(cs.mean()), "max": float(cs.max()), "tmean": tmean.astype(np.float32)}
        for a in ids:
            out["img"][a][f"emb_{key}"] = emb[a].astype(np.float32)
            if a in sims:
                out["img"][a][f"sim_{key}"] = [sims[a]["mean"], sims[a]["max"]]
                out["img"][a][f"t_{key}"] = sims[a]["tmean"]
        del model
        torch.cuda.empty_cache()
        print(f"{key} done", flush=True)
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    import random

    from budget import check_modal
    from formal import quoted_strings
    from taste import CACHE, HF_REV, load

    d = load()
    sc = d["scenes"]
    scenes = [{"scene": int(r.scene), "brief": r.brief, "assets": [int(x) for x in r.assets]} for r in sc.itertuples()]
    # R3: 100 dev scenes' string sets x {correct, one character changed}
    rnd = random.Random(3)
    items = []
    dev_briefs = [r.brief for r in sc[sc.split == "dev"].itertuples()]
    k = 0
    while len(items) < 200:
        strs = [s for s in quoted_strings(dev_briefs[k % len(dev_briefs)]) if 3 <= len(s) <= 28][:4]
        k += 1
        if not strs:
            continue
        font, size = rnd.randrange(4), rnd.choice([36, 44, 56])
        bg, fg = rnd.choice([(30, 30, 30), (200, 200, 200)]), None
        fg = (240, 240, 240) if bg[0] < 100 else (20, 20, 20)
        tgt = rnd.randrange(len(strs))
        s = strs[tgt]
        pos = rnd.randrange(len(s))
        ch = rnd.choice([c for c in "ABCDEFGHKMNPRSTWXZ" if c.lower() != s[pos].lower()])
        wrong = strs.copy()
        wrong[tgt] = s[:pos] + ch + s[pos + 1 :]
        base = {"font": font, "size": size, "bg": bg[0], "fg": fg, "strings": strs}
        items.append({**base, "id": len(items) // 2, "variant": "correct", "lines": strs})
        items.append({**base, "id": len(items) // 2, "variant": "wrong", "lines": wrong})
    check_modal(1.0, "features (A10G ~15 min)")
    out = run.remote(scenes, items, HF_REV)
    jdir = CACHE / "jpg640"
    jdir.mkdir(parents=True, exist_ok=True)
    for aid, b in out.pop("jpg").items():
        (jdir / f"{aid}.jpg").write_bytes(b)
    out["r3_items"] = [{k2: v for k2, v in it.items() if k2 != "lines"} for it in items]
    with open(CACHE / "features.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out['img'])} images' features, {len(out['r3'])} R3 renders")
