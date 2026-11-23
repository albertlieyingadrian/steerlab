"""GPU side of the SAE intersectional-bias study (app `sae-intersectional-bias`, profile credits30).

    MODAL_PROFILE=credits30 modal deploy experiments/sae-intersectional-bias/modal_app.py
    then run.py spawns the functions below and saves the call ids (deploy + spawn, never a plain `modal run`).

- `prefetch` (CPU): FairFace (HF HuggingFaceM4/FairFace, config 1.25), ImageNette val (fast.ai imagenette2-320),
  CLIP ViT-L/14-336, the released SAE, LLaVA-1.5-7B and MME onto the volume.
- `cache_clip(tag, limit)` (A10G): every FairFace image and ImageNette val image through CLIP once. Writes, per split:
  `cls23` (block-23 CLS residual, the row the DeBiasLens SAE edits that reaches the embedding), `emb`, `pooled22`
  (mean of block-22 patch tokens, what LLaVA reads, for the leakage probe), labels; for a fixed image subset the exact
  per-head / per-MLP CLS contributions of blocks 20-23; LEACE statistics of block-22 patch tokens by perceived gender
  (20,000 train images); all-token SAE firing counts per (gender, age, race) group at block 23 (20,000 train images);
  SAE reconstruction error on block-22 and block-23 tokens (500 val images); text embeddings.
- `llava_eval(...)` (A100): see llava.py.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal

APP = "sae-intersectional-bias"
app = modal.App(APP)
volume = modal.Volume.from_name(APP, create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .pip_install(
        "torch==2.9.1",
        "torchvision==0.24.1",
        "transformers==5.22.2",
        "accelerate>=1.10",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
        "pyarrow",
        "pillow",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_python_source("clip_tools", "sae", "llava_tools")
)
CLIP_ID = "openai/clip-vit-large-patch14-336"
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
OUT = Path("/vol/out")


@app.function(image=image, volumes={"/vol": volume}, secrets=[hf], timeout=3600, cpu=4, memory=16384)
def prefetch():
    import subprocess

    from huggingface_hub import hf_hub_download, snapshot_download

    from sae import FILE, REPO

    t0 = time.time()
    snapshot_download("HuggingFaceM4/FairFace", repo_type="dataset", allow_patterns=["1.25/*", "README.md"])
    snapshot_download(CLIP_ID, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    hf_hub_download(REPO, FILE)
    snapshot_download(LLAVA_ID, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"])
    snapshot_download("lmms-lab/MME", repo_type="dataset")
    if not Path("/vol/imagenette2-320/val").exists():
        subprocess.run("cd /vol && wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && "
                       "tar xzf imagenette2-320.tgz && rm imagenette2-320.tgz", shell=True, check=True)  # fmt: skip
    volume.commit()
    n_imn = len(_imagenette()[0])
    return {"seconds": round(time.time() - t0, 1), "imagenette_val": n_imn}


# ------------------------------------------------------------------ data
WNIDS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079", "n03394916", "n03417042", "n03425413",
         "n03445777", "n03888605"]  # fmt: skip


def fairface(split):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download("HuggingFaceM4/FairFace", repo_type="dataset", allow_patterns=["1.25/*"]))
    name = "train" if split == "train" else "validation"
    files = sorted((root / "1.25").glob(f"{name}-*.parquet"))
    tabs = [pq.read_table(f) for f in files]
    imgs, age, gender, race = [], [], [], []
    for t in tabs:
        d = t.to_pydict()
        imgs += [x["bytes"] for x in d["image"]]
        age += d["age"]
        gender += d["gender"]
        race += d["race"]
    meta = json.loads(tabs[0].schema.metadata[b"huggingface"].decode())["info"]["features"]
    names = {k: meta[k]["names"] for k in ("age", "gender", "race")}
    return imgs, {"age": age, "gender": gender, "race": race}, names


def _imagenette():
    """ImageNette val bytes and labels, packed into one pickle on the volume after the first (slow) read."""
    import pickle

    packed = Path("/vol/imagenette_val.pkl")
    if packed.exists():
        return pickle.loads(packed.read_bytes())
    imgs, labels = [], []
    for i, w in enumerate(WNIDS):
        for f in sorted(Path(f"/vol/imagenette2-320/val/{w}").glob("*")):
            imgs.append(f.read_bytes())
            labels.append(i)
    packed.write_bytes(pickle.dumps((imgs, labels)))
    volume.commit()
    return imgs, labels


class _DS:
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image

        from clip_tools import preprocess

        return preprocess(Image.open(io.BytesIO(self.imgs[i])))


# ------------------------------------------------------------------ CLIP cache
@app.function(image=image, volumes={"/vol": volume}, secrets=[hf], gpu="A10G", timeout=7200, cpu=8, memory=49152)
def cache_clip(tag: str = "full", limit: int | None = None, texts: list[str] | None = None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPModel, CLIPTokenizer

    from clip_tools import forward_cache
    from sae import SAE

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    dev = "cuda"
    model = CLIPModel.from_pretrained(CLIP_ID, dtype=torch.float32).to(dev).eval()
    sae = SAE.load().to(dev)
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*a):
        msg = f"[{time.time() - t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    report = {"tag": tag, "limit": limit}
    if texts:
        tok = CLIPTokenizer.from_pretrained(CLIP_ID)
        embs = []
        for i in range(0, len(texts), 256):
            b = tok(texts[i : i + 256], padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                e = model.get_text_features(**b)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            embs.append(e.float().cpu())
        np.save(out / "text_emb.npy", torch.cat(embs).numpy())
        say("text", len(texts))

    rng = np.random.default_rng(0)
    for split in ("val", "train", "imagenette"):
        if split == "imagenette":
            imgs, lab = _imagenette()
            labels, names = {"label": lab}, None
        else:
            imgs, labels, names = fairface(split)
        n_all = len(imgs)
        if split == "train":
            perm = rng.permutation(n_all)
            np.save(out / "train_perm.npy", perm)
            keep = np.sort(perm[:limit]) if limit else np.arange(n_all)
            sub20 = set(perm[: min(20000, limit or n_all)].tolist())
            sub10 = set(perm[: min(10000, limit or n_all)].tolist())
        else:
            keep = np.arange(min(limit, n_all)) if limit else np.arange(n_all)
            sub20 = set()
            sub10 = set(keep.tolist())  # every val / ImageNette image gets head contributions
        say(split, "images", len(keep), "of", n_all)
        dl = DataLoader(_DS([imgs[i] for i in keep]), batch_size=48, num_workers=7, shuffle=False)
        cls23 = np.zeros((len(keep), 1024), np.float32)
        emb = np.zeros((len(keep), 768), np.float32)
        pooled22 = np.zeros((len(keep), 1024), np.float32)
        head_rows = [j for j, i in enumerate(keep) if int(i) in sub10]
        head_pos = {j: k for k, j in enumerate(head_rows)}
        heads = np.zeros((len(head_rows), 4, 16, 1024), np.float16)
        mlps = np.zeros((len(head_rows), 4, 1024), np.float32)
        cls_in = np.zeros((len(head_rows), 1024), np.float32)
        # LEACE stats (block-22 patch tokens, by perceived gender), train subset
        S1 = torch.zeros(1024, dtype=torch.float64, device=dev)
        S2 = torch.zeros(1024, 1024, dtype=torch.float64, device=dev)
        Sg = torch.zeros(2, 1024, dtype=torch.float64, device=dev)
        Ng = torch.zeros(2, dtype=torch.float64, device=dev)
        # all-token SAE firing by group (gender x age9 x race = 126), train subset
        fire = torch.zeros(126, 8192, dtype=torch.float64, device=dev)
        actsum = torch.zeros(126, 8192, dtype=torch.float64, device=dev)
        ntok = torch.zeros(126, dtype=torch.float64, device=dev)
        nimg = torch.zeros(126, dtype=torch.float64, device=dev)
        fvu = {"l22_err": 0.0, "l22_var": 0.0, "l23_err": 0.0, "l23_var": 0.0, "n": 0}
        attn_bias = None
        row = 0
        nb = 0
        for px in dl:
            B = px.shape[0]
            idx = keep[row : row + B]
            r = forward_cache(model, px.to(dev), n_last=4)
            cls23[row : row + B] = r["cls23"].cpu().numpy()
            emb[row : row + B] = r["emb"].cpu().numpy()
            tok22 = r["tok22"].float()
            pooled22[row : row + B] = tok22.mean(1).cpu().numpy()
            attn_bias = r["attn_bias"].cpu().numpy()
            hb = [b for b in range(B) if row + b in head_pos]
            if hb:
                ks = [head_pos[row + b] for b in hb]
                heads[ks] = r["heads"][hb].cpu().numpy().astype(np.float16)
                mlps[ks] = r["mlps"][hb].cpu().numpy()
                cls_in[ks] = r["cls_in"][hb].cpu().numpy()
            if split == "train":
                m20 = torch.tensor([int(i) in sub20 for i in idx], device=dev)
                if m20.any():
                    g = torch.tensor([labels["gender"][int(i)] for i in idx], device=dev)
                    a9 = torch.tensor([labels["age"][int(i)] for i in idx], device=dev)
                    rc = torch.tensor([labels["race"][int(i)] for i in idx], device=dev)
                    x = tok22[m20]  # [b, 576, 1024]
                    xf = x.reshape(-1, 1024).double()
                    S1 += xf.sum(0)
                    S2 += xf.T @ xf
                    gm = g[m20]
                    for c in (0, 1):
                        sel = x[gm == c]
                        if len(sel):
                            Sg[c] += sel.reshape(-1, 1024).double().sum(0)
                            Ng[c] += sel.shape[0] * sel.shape[1]
                    key = (g * 63 + a9 * 7 + rc)[m20]
                    tok23 = r["tok23"][m20].float()  # [b, 577, 1024]
                    codes = sae.encode(tok23.reshape(-1, 1024)).view(tok23.shape[0], tok23.shape[1], -1)
                    fire.index_add_(0, key, (codes > 0).double().sum(1))
                    actsum.index_add_(0, key, codes.double().sum(1))
                    ntok.index_add_(0, key, torch.full((len(key),), float(tok23.shape[1]), device=dev,
                                                       dtype=torch.float64))  # fmt: skip
                    nimg.index_add_(0, key, torch.ones(len(key), device=dev, dtype=torch.float64))
            if split == "val" and fvu["n"] < 500:
                for name, t in (("l22", tok22), ("l23", r["tok23"].float())):
                    flat = t.reshape(-1, 1024)
                    rec = sae.decode(sae.encode(flat))
                    fvu[f"{name}_err"] += float(((flat - rec) ** 2).sum())
                    fvu[f"{name}_var"] += float(((flat - flat.mean(0)) ** 2).sum())
                fvu["n"] += B
            row += B
            nb += 1
            if nb % 200 == 0:
                say(split, row, "/", len(keep))
        np.save(out / f"{split}_cls23.npy", cls23)
        np.save(out / f"{split}_emb.npy", emb)
        np.save(out / f"{split}_pooled22.npy", pooled22)
        np.save(out / f"{split}_index.npy", keep)
        np.save(out / f"{split}_head_rows.npy", np.array(head_rows))
        np.save(out / f"{split}_heads.npy", heads)
        np.save(out / f"{split}_mlps.npy", mlps)
        np.save(out / f"{split}_cls_in.npy", cls_in)
        np.save(out / "attn_bias.npy", attn_bias)
        json.dump({k: [v[int(i)] for i in keep] for k, v in labels.items()}, open(out / f"{split}_labels.json", "w"))
        if names:
            json.dump(names, open(out / "label_names.json", "w"))
        if split == "train":
            np.savez(out / "leace22_stats.npz", S1=S1.cpu().numpy(), S2=S2.cpu().numpy(), Sg=Sg.cpu().numpy(),
                     Ng=Ng.cpu().numpy())  # fmt: skip
            np.savez(out / "alltoken_sae_stats.npz", fire=fire.cpu().numpy(), actsum=actsum.cpu().numpy(),
                     ntok=ntok.cpu().numpy(), nimg=nimg.cpu().numpy())  # fmt: skip
        if split == "val":
            report["fvu_tokens"] = {k: fvu[f"{k}_err"] / max(fvu[f"{k}_var"], 1e-9) for k in ("l22", "l23")}
            report["fvu_images"] = fvu["n"]
        if head_rows:
            rec = cls_in + heads.astype(np.float32).sum((1, 2)) + attn_bias.sum(0) + mlps.sum(1)
            ref = cls23[head_rows]
            report[f"{split}_decomp_rel_err"] = float(np.linalg.norm(rec - ref) / np.linalg.norm(ref))
            e2 = model.visual_projection(model.vision_model.post_layernorm(torch.tensor(cls23[:64], device=dev)))
            report[f"{split}_emb_check"] = float((e2.cpu() - torch.tensor(emb[:64])).abs().max())
        volume.commit()
        say(split, "done", report)
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(out / "report.json", "w"), indent=1)
    volume.commit()
    return report


# ------------------------------------------------------------------ B4: LLaVA-1.5-7B
@app.function(image=image, volumes={"/vol": volume}, secrets=[hf], gpu="A100-40GB", timeout=5 * 3600, cpu=8,
              memory=65536)  # fmt: skip
def llava_eval(tag: str, rows: list, arms: list, mme_arms: list, erasers: dict, identity_n: int = 200,
               mme_limit: int | None = None, batch: int = 16):  # fmt: skip
    from huggingface_hub import hf_hub_download

    from llava_tools import run_llava
    from sae import FILE, REPO

    t0 = time.time()
    out = OUT / f"llava_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*a):
        msg = f"[{time.time() - t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    imgs, _, _ = fairface("val")
    val_images = {r["img"]: imgs[r["img"]] for r in rows}
    res = run_llava(rows, arms, mme_arms, erasers, hf_hub_download(REPO, FILE), val_images, batch=batch, log=say,
                    identity_n=identity_n, mme_limit=mme_limit)  # fmt: skip
    res["seconds"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(res))
    volume.commit()
    say("done", res["seconds"])
    return res
