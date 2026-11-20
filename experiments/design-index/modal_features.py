"""
Turn renders on the Volume into features and embeddings; also index the public WaybackUI screenshots.

COSTS MONEY (CPU for hand features; one A10G per embedding batch). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/design-index/modal_features.py::ours
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/design-index/modal_features.py::waybackui

ours       every dom/<pid>.json with ok=True -> hand features (features.py) + SigLIP and UIClip image embeddings of
           shots/<pid>.jpg. Saved locally to CACHE/feats/ours.pkl (no screenshots leave the Volume).
waybackui  biglab/waybackui-all (all splits), 1366x768 and 1920x1080 kept apart: palette, layout occupancy from the
           element boxes and roles, SigLIP and UIClip. Saved to CACHE/feats/waybackui.pkl.
"""

from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VOL = "/vol"
app = modal.App("design-index-features")
volume = modal.Volume.from_name("design-index", create_if_missing=True)
hf = modal.Volume.from_name("design-index-hf", create_if_missing=True)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.1.3", "pillow==11.0.0", "scikit-learn==1.5.2")
    .add_local_file(HERE / "features.py", "/root/features.py")
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.1.3",
        "pillow==11.0.0",
        "scikit-learn==1.5.2",
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "protobuf==5.29.0",
        "pyarrow==18.1.0",
        "huggingface_hub==0.26.5",
        "hf_transfer==0.1.9",
    )
    .env(
        {"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"}
    )
    .add_local_file(HERE / "features.py", "/root/features.py")
)
SIGLIP = "google/siglip-base-patch16-224"
UICLIP = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
UICLIP_PROC = "openai/clip-vit-base-patch32"
WBUI_TEXT = {
    "StaticText",
    "heading",
    "paragraph",
    "LineBreak",
    "text",
    "listitem",
    "ListMarker",
    "time",
    "strong",
    "emphasis",
}
WBUI_MEDIA = {
    "image",
    "img",
    "graphics-symbol",
    "figure",
    "video",
    "SvgRoot",
    "Canvas",
    "graphics-document",
    "PluginObject",
    "Iframe",
}
WBUI_CTRL = {
    "link",
    "button",
    "textbox",
    "combobox",
    "searchbox",
    "checkbox",
    "radio",
    "menuitem",
    "tab",
    "ListBox",
    "slider",
}


@app.function(image=cpu_image, volumes={VOL: volume}, cpu=4, memory=8192, timeout=3600)
def hand_batch(pids: list[str]) -> list[dict]:
    from features import page_features
    from PIL import Image

    out = []
    for pid in pids:
        try:
            rec = json.loads(Path(f"{VOL}/dom/{pid}.json").read_text())
            if not rec.get("ok"):
                continue
            out.append(page_features(rec, Image.open(f"{VOL}/shots/{pid}.jpg")))
            out[-1].update(
                {
                    k: rec.get(k)
                    for k in (
                        "status",
                        "n_req",
                        "n_failed",
                        "n_blocked",
                        "n_http_err",
                        "pixel_std",
                        "elapsed",
                        "final_url",
                        "mode",
                        "ts",
                    )
                }
            )
        except Exception as e:  # noqa: BLE001 - one bad page must not sink the batch
            print(pid, e.__class__.__name__, str(e)[:120])
    return out


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

        self.torch = torch
        self.sig = (
            AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
        )
        self.sig_p = AutoProcessor.from_pretrained(SIGLIP)
        self.uic = (
            CLIPModel.from_pretrained(UICLIP, torch_dtype=torch.float16).cuda().eval()
        )
        self.uic_p = CLIPProcessor.from_pretrained(UICLIP_PROC)

    def __call__(self, imgs):
        import numpy as np

        t = self.torch
        with t.no_grad():
            a = (
                self.sig_p(images=imgs, return_tensors="pt")["pixel_values"]
                .cuda()
                .half()
            )
            s = self.sig.get_image_features(pixel_values=a)
            b = (
                self.uic_p(images=imgs, return_tensors="pt")["pixel_values"]
                .cuda()
                .half()
            )
            u = self.uic.get_image_features(pixel_values=b)
        norm = lambda x: (
            (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16)
        )
        return norm(s), norm(u)


@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={VOL: volume, "/hf": hf},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def embed_ours(pids: list[str]) -> dict:
    from PIL import Image

    emb = Embedder()
    out = {}
    for i in range(0, len(pids), 64):
        chunk = [p for p in pids[i : i + 64] if Path(f"{VOL}/shots/{p}.jpg").exists()]
        if not chunk:
            continue
        imgs = [Image.open(f"{VOL}/shots/{p}.jpg").convert("RGB") for p in chunk]
        s, u = emb(imgs)
        for j, p in enumerate(chunk):
            out[p] = (s[j], u[j])
    hf.commit()
    return out


def _wbui_boxes(labels, boxes):
    out = []
    for lab, b in zip(labels, boxes):
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        ls = set(lab or [])
        ch = (
            2
            if ls & WBUI_CTRL
            else 1
            if ls & WBUI_MEDIA
            else 0
            if ls & WBUI_TEXT
            else None
        )
        if ch is not None and x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1, ch))
    return out


@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/hf": hf},
    cpu=8,
    memory=32768,
    timeout=3 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def waybackui_shard(filename: str) -> list[dict]:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    print("downloading", filename, flush=True)
    path = hf_hub_download("biglab/waybackui-all", filename, repo_type="dataset")
    print("downloaded", filename, flush=True)
    hf.commit()
    tab = pq.read_table(path)
    emb = Embedder()
    rows = []
    from features import occupancy, palette

    # spawn, not fork: forking after torch has started OpenMP threads deadlocks scikit-learn's k-means in the children
    with ProcessPoolExecutor(
        7, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for start in range(0, tab.num_rows, 256):
            batch = tab.slice(start, 256).to_pylist()
            imgs = [
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                for r in batch
            ]
            s, u = emb(imgs)
            pals = list(pool.map(palette, [im.resize((160, 100)) for im in imgs]))
            for j, r in enumerate(batch):
                w, h = (int(v) for v in r["resolution"].split("-"))
                lay = occupancy(
                    _wbui_boxes(r["labels"], r["contentBoxes"]),
                    page_w=w,
                    page_h=1.5 * h,
                )
                rows.append(
                    {
                        "base_url": r["base_url"],
                        "year": int(r["year"]),
                        "ts": r["timestamp"],
                        "res": r["resolution"],
                        "split": r["split"],
                        "img_h": imgs[j].height,
                        "img_w": imgs[j].width,
                        "pal_c": pals[j][0],
                        "pal_w": pals[j][1],
                        "layout": lay.astype("float16"),
                        "n_boxes": len(r["contentBoxes"] or []),
                        "siglip": s[j],
                        "uiclip": u[j],
                    }
                )
            print(filename, start, flush=True)
    return rows


@app.local_entrypoint()
def ours(chunk: int = 300):
    from budget import check_modal
    from common import CACHE

    out_path = CACHE / "feats" / "ours.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    pids = sorted(
        e.path.split("/")[-1][:-4]
        for e in volume.listdir("shots")
        if e.path.endswith(".jpg")
    )
    todo = [p for p in pids if p not in have]
    print(f"{len(pids)} shots, {len(todo)} new")
    if not todo:
        return
    check_modal(0.3 + len(todo) * 0.0004, f"features for {len(todo)} pages")
    chunks = [todo[i : i + chunk] for i in range(0, len(todo), chunk)]
    for res in hand_batch.map(chunks):
        for r in res:
            have[r["pid"]] = r
    embs = {}
    for res in embed_ours.map([todo[i : i + 2000] for i in range(0, len(todo), 2000)]):
        embs.update(res)
    for p, (s, u) in embs.items():
        if p in have:
            have[p]["siglip"], have[p]["uiclip"] = s, u
    out_path.write_bytes(pickle.dumps(have))
    print(f"saved {len(have)} pages to {out_path}")


@app.local_entrypoint()
def waybackui(only: str = ""):
    from budget import check_modal
    from common import CACHE

    files = [f"data/train-0000{i}-of-00008.parquet" for i in range(8)] + [
        "data/validation-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
    ]
    if only:  # a smoke test on one shard, saved beside the full file
        files = [f for f in files if only in f]
    check_modal(10 * 1.2 * 1.5, "waybackui features (10 shards, A10G)")
    rows = []
    for res in waybackui_shard.map(files):
        rows += res
        print("shard rows", len(res))
    out = CACHE / "feats" / ("waybackui.pkl" if not only else f"waybackui_{only}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(rows))
    print(f"saved {len(rows)} rows to {out}")
