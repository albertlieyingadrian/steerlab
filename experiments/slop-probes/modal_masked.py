"""
The shortcut control: badge-masked embeddings and badge-free page features, from Ep 2's renders (read-only).

COSTS MONEY (one A10G per chunk, a few cents). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/slop-probes/modal_masked.py

For every kept page of the five sources (ep3.SOURCES), on Ep 2's Volume mounted read-only:
  sig_raw, uic_raw     SigLIP / UIClip of the unmasked first screen (the embedding check: must match Ep 2's cached vectors)
  sig_mask, uic_mask   the same with the bottom-right 320x96 box painted mid-grey (every page, every class)
  sig_corner           SigLIP of the 320x96 corner crop alone (how much the badge gives away)
  hand_mask            Ep 2's hand features (palette, type, spacing, layout, DOM graph) recomputed on the masked
                       screenshot with every element and block inside the box removed
Saved to ~/.cache/steerlab/slop-probes/masked.pkl. No screenshot leaves the Volume.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ep3 import MASK_BOX, MASK_GREY

EP2 = HERE.parent / "design-index"
VOL = "/vol"
app = modal.App("taste-ep3-masked")
ep2_volume = modal.Volume.from_name("design-index").read_only()
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.1.3",
        "pillow==11.0.0",
        "scikit-learn==1.5.2",
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "protobuf==5.29.0",
        "hf_transfer==0.1.9",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
    .add_local_file(EP2 / "features.py", "/root/features.py")
    .add_local_file(HERE / "ep3.py", "/root/ep3.py")
)
SIGLIP = "google/siglip-base-patch16-224"  # Ep 2's models and processors, unchanged
UICLIP = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
UICLIP_PROC = "openai/clip-vit-base-patch32"


def inside(x, y, w, h, box=MASK_BOX) -> bool:
    x0, y0, x1, y1 = box
    return x >= x0 - 2 and y >= y0 - 2 and x + w <= x1 + 2 and y + h <= y1 + 2


def strip_badge(rec: dict) -> tuple[dict, int]:
    """Drop elements and blocks wholly inside the badge box; re-point each kept block to its nearest kept ancestor."""
    els = rec.get("elements") or []
    kept_els = [e for e in els if not inside(*e[:4])]
    blocks = rec.get("blocks") or []
    keep = [not inside(*b[:4]) for b in blocks]
    new_ix, out_blocks = {}, []
    for i, b in enumerate(blocks):
        if keep[i]:
            new_ix[i] = len(out_blocks)
            out_blocks.append(list(b))
    for b in out_blocks:
        p = b[4]
        while 0 <= p < len(blocks) and not keep[p]:
            p = blocks[p][4]
        b[4] = new_ix.get(p, -1)
    rec = dict(rec, elements=kept_els, blocks=out_blocks)
    return rec, (len(els) - len(kept_els)) + (len(blocks) - len(out_blocks))


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=16384,
    volumes={VOL: ep2_volume, "/hf": hf},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_chunk(pids: list[str]) -> dict:
    import numpy as np
    import torch
    from features import page_features
    from PIL import Image, ImageDraw
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    sig = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
    sig_p = AutoProcessor.from_pretrained(SIGLIP)
    uic = CLIPModel.from_pretrained(UICLIP, torch_dtype=torch.float16).cuda().eval()
    uic_p = CLIPProcessor.from_pretrained(UICLIP_PROC)

    def norm(x):
        return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16)

    def embed(imgs):
        with torch.no_grad():
            a = sig_p(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
            b = uic_p(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
            return norm(sig.get_image_features(pixel_values=a)), norm(uic.get_image_features(pixel_values=b))

    out = {}
    for i in range(0, len(pids), 32):
        chunk = [p for p in pids[i : i + 32] if Path(f"{VOL}/shots/{p}.jpg").exists()]
        raws, masks, corners = [], [], []
        for p in chunk:
            im = Image.open(f"{VOL}/shots/{p}.jpg").convert("RGB")
            raws.append(im)
            m = im.copy()
            ImageDraw.Draw(m).rectangle(MASK_BOX, fill=MASK_GREY)
            masks.append(m)
            corners.append(im.crop(MASK_BOX))
        if not chunk:
            continue
        s_raw, u_raw = embed(raws)
        s_mask, u_mask = embed(masks)
        s_cor, _ = embed(corners)
        for j, p in enumerate(chunk):
            row = {
                "sig_raw": s_raw[j],
                "uic_raw": u_raw[j],
                "sig_mask": s_mask[j],
                "uic_mask": u_mask[j],
                "sig_corner": s_cor[j],
            }
            try:
                rec = json.loads(Path(f"{VOL}/dom/{p}.json").read_text())
                rec2, n_badge = strip_badge(rec)
                row["hand_mask"] = page_features(rec2, masks[j])
                row["n_badge_els"] = n_badge
            except Exception as e:  # noqa: BLE001 - a missing DOM file leaves the embeddings usable
                print(p, e.__class__.__name__, str(e)[:120])
            out[p] = row
        print(f"{i + len(chunk)}/{len(pids)}", flush=True)
    return out


@app.local_entrypoint()
def main(chunk: int = 1000, limit: int = 0):
    from budget import check_modal
    from ep3 import CACHE, load_pages

    pids = sorted(r["pid"] for r in load_pages())
    if limit:
        pids = pids[:limit]
    check_modal(0.15 * (len(pids) / chunk + 1), f"masked features for {len(pids)} pages")
    out = {}
    for res in run_chunk.map([pids[i : i + chunk] for i in range(0, len(pids), chunk)]):
        out.update(res)
    CACHE.mkdir(parents=True, exist_ok=True)
    name = "masked.pkl" if not limit else "masked_smoke.pkl"
    (CACHE / name).write_bytes(pickle.dumps(out))
    print(f"saved {len(out)} pages to {CACHE / name}")
