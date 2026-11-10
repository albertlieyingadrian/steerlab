"""Run DNA through Evo 2 7B on a rented H100 and keep Goodfire's layer-26 SAE features, sparsely.

    modal run experiments/evo2-rna-atlas/modal_evo2.py::smoke        # load the model and SAE, encode a short test
    python experiments/evo2-rna-atlas/4_encode.py                    # builds the jobs and calls `encode` below

The model is `evo2_7b_262k`, the checkpoint Arc's sparse-autoencoder notebook pairs with Goodfire's SAE
(`Goodfire/Evo-2-Layer-26-Mixed`, file `sae-layer26-mixed-expansion_8-k_64.pt`: 4,096 inputs, 32,768 features,
batch top-k with k = 64). The SAE reads the output of block 26, and the encoder and the hook below follow that
notebook (ArcInstitute/evo2, notebooks/sparse_autoencoder), so the features carry the same numbers.

Long sequences are read in 8,192-letter windows that overlap by 1,024; the model reads left to right, so every kept
position has seen at least 1,024 letters before it (except in the first window). For each kept position the job
keeps the 32 strongest features (ids and values) and the model's next-letter entropy; for each requested range it
keeps the mean layer-26 vector, and the mean and max of every feature that fired in it, as sparse lists.
"""

from __future__ import annotations

import io
import time

import modal

app = modal.App("evo2-rna-atlas")

image = (
    # NVIDIA's PyTorch image (the base of Arc's own Dockerfile) pins packages through PIP_CONSTRAINT, which clashes
    # with Modal's client requirements; the pin is dropped before Modal installs into the image.
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.04-py3",
        setup_dockerfile_commands=["ENV PIP_CONSTRAINT=", "RUN rm -f /etc/pip/constraint.txt"],
    )
    .pip_install("evo2==0.6.0", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
)
volume = modal.Volume.from_name("evo2-rna-atlas", create_if_missing=True)

MODEL = "evo2_7b_262k"
SAE_REPO, SAE_FILE = "Goodfire/Evo-2-Layer-26-Mixed", "sae-layer26-mixed-expansion_8-k_64.pt"
LAYER = 26
D_MODEL, EXPANSION, K = 4096, 8, 64
WINDOW, OVERLAP = 8192, 1024
TOP = 32
GPU = "H100"


class _Encoder:
    """Evo 2 with a hook on block 26 and the tied batch-top-k SAE from Arc's notebook."""

    def __init__(self):
        import torch
        from evo2 import Evo2
        from huggingface_hub import hf_hub_download

        torch.set_grad_enabled(False)
        self.torch = torch
        self.evo = Evo2(MODEL)
        self.model = self.evo.model
        self.device = next(self.model.parameters()).device
        state = torch.load(hf_hub_download(SAE_REPO, SAE_FILE), weights_only=True, map_location="cpu")
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
        self.W = state["W"].to(self.device, torch.bfloat16)  # (4096, 32768)
        self.b_enc = state["b_enc"].to(self.device, torch.bfloat16)
        self._acts = None
        self.model.blocks[LAYER].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._acts = (output[0] if isinstance(output, tuple) else output).detach()

    def forward(self, seq: str):
        """Layer-26 activations (L, 4096), SAE codes (L, 32768) and next-letter entropy (L,) for one window."""
        torch = self.torch
        toks = torch.tensor(self.evo.tokenizer.tokenize(seq), dtype=torch.long, device=self.device)[None]
        logits = self.model(toks)[0]
        logits = logits[0] if logits.dim() == 3 else logits
        acts = self._acts[0]
        f = torch.relu(acts.to(torch.bfloat16) @ self.W + self.b_enc)
        # Batch top-k over the whole window, as the notebook encodes: k x L values survive in total.
        flat = f.flatten()
        top = torch.topk(flat, K * f.shape[0])
        codes = torch.zeros_like(flat).scatter(0, top.indices, top.values).reshape(f.shape)
        logp = torch.log_softmax(logits.float(), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return acts, codes, entropy


@app.cls(image=image, gpu=GPU, volumes={"/cache": volume}, timeout=60 * 60 * 3, scaledown_window=120)
class Evo2SAE:
    @modal.enter()
    def load(self):
        t0 = time.time()
        self.enc = _Encoder()
        print(f"model and SAE loaded in {time.time() - t0:.0f} s on {self.enc.torch.cuda.get_device_name()}")

    @modal.method()
    def encode(self, job: str, items: list[dict]) -> dict:
        """items: {id, seq, ranges: [(lo, hi), ...], keep_positions: bool}. Writes /cache/out/<job>.npz."""
        import os

        import numpy as np

        torch = self.enc.torch
        t0, letters = time.time(), 0
        pos_item, pos_idx, pos_ids, pos_vals, pos_ent = [], [], [], [], []
        rng_item, rng_lo, rng_hi, rng_mean_emb = [], [], [], []
        rng_feat_row, rng_feat_id, rng_feat_mean, rng_feat_max = [], [], [], []
        for n, item in enumerate(items):
            seq, length = item["seq"], len(item["seq"])
            want = np.zeros(length, dtype=bool)
            for lo, hi in item["ranges"]:
                want[lo:hi] = True
            done = np.zeros(length, dtype=bool)
            ranges = list(item["ranges"])
            # Running sums per range, on the GPU: summed activation, summed and max codes, position count.
            acc = [None] * len(ranges)
            starts = [0] if length <= WINDOW else list(range(0, length - OVERLAP, WINDOW - OVERLAP))
            for s in starts:
                e = min(length, s + WINDOW)
                if s > 0 and e - s < WINDOW:  # last window: slide back so it is full length
                    s = max(0, e - WINDOW)
                keep_from = 0 if s == 0 else OVERLAP
                idx = np.flatnonzero(want[s + keep_from : e] & ~done[s + keep_from : e]) + s + keep_from
                if len(idx) == 0:
                    continue
                done[idx] = True
                acts, codes, entropy = self.enc.forward(seq[s:e])
                letters += e - s
                local = torch.as_tensor(idx - s, device=acts.device)
                if item.get("keep_positions", True):
                    vals, ids = torch.topk(codes[local].float(), TOP, dim=-1)
                    pos_item.append(np.full(len(idx), n, dtype=np.int32))
                    pos_idx.append(idx.astype(np.int32))
                    pos_ids.append(ids.cpu().numpy().astype(np.uint16))
                    pos_vals.append(vals.cpu().numpy().astype(np.float16))
                    pos_ent.append(entropy[local].cpu().numpy().astype(np.float16))
                for r, (lo, hi) in enumerate(ranges):
                    if hi <= idx[0] or lo > idx[-1]:
                        continue
                    a, b = np.searchsorted(idx, lo), np.searchsorted(idx, hi)
                    if a == b:
                        continue
                    rows = torch.as_tensor(idx[a:b] - s, device=acts.device)
                    sub_a, sub_c = acts[rows].float(), codes[rows].float()
                    if acc[r] is None:
                        acc[r] = [sub_a.sum(0), sub_c.sum(0), sub_c.max(0).values, b - a]
                    else:
                        acc[r][0] += sub_a.sum(0)
                        acc[r][1] += sub_c.sum(0)
                        acc[r][2] = torch.maximum(acc[r][2], sub_c.max(0).values)
                        acc[r][3] += b - a
                del acts, codes, entropy
            for r, (lo, hi) in enumerate(ranges):
                if acc[r] is None:
                    continue
                sum_a, sum_c, max_c, count = acc[r]
                fired = torch.nonzero(max_c > 0).flatten()
                rng_item.append(n)
                rng_lo.append(lo)
                rng_hi.append(hi)
                rng_mean_emb.append((sum_a / count).cpu().numpy().astype(np.float16))
                rng_feat_row.append(np.full(len(fired), len(rng_item) - 1, dtype=np.int32))
                rng_feat_id.append(fired.cpu().numpy().astype(np.uint16))
                rng_feat_mean.append((sum_c[fired] / count).cpu().numpy().astype(np.float16))
                rng_feat_max.append(max_c[fired].cpu().numpy().astype(np.float16))
            del acc
            if n % 50 == 0 or n == len(items) - 1:
                print(f"{job}: item {n + 1}/{len(items)}, {letters:,} letters, {time.time() - t0:.0f} s")

        def cat(parts, dtype, width=None):
            if parts:
                return np.concatenate(parts)
            return np.zeros((0, width) if width else 0, dtype=dtype)

        out = {
            "item_ids": np.array([item["id"] for item in items]),
            "pos_item": cat(pos_item, np.int32),
            "pos_idx": cat(pos_idx, np.int32),
            "pos_ids": cat(pos_ids, np.uint16, TOP),
            "pos_vals": cat(pos_vals, np.float16, TOP),
            "pos_entropy": cat(pos_ent, np.float16),
            "rng_item": np.array(rng_item, dtype=np.int32),
            "rng_lo": np.array(rng_lo, dtype=np.int32),
            "rng_hi": np.array(rng_hi, dtype=np.int32),
            "rng_mean_emb": np.stack(rng_mean_emb) if rng_mean_emb else np.zeros((0, D_MODEL), np.float16),
            "rng_feat_row": cat(rng_feat_row, np.int32),
            "rng_feat_id": cat(rng_feat_id, np.uint16),
            "rng_feat_mean": cat(rng_feat_mean, np.float16),
            "rng_feat_max": cat(rng_feat_max, np.float16),
        }
        os.makedirs("/cache/out", exist_ok=True)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **out)
        with open(f"/cache/out/{job}.npz", "wb") as handle:
            handle.write(buffer.getvalue())
        volume.commit()
        seconds = time.time() - t0
        print(f"{job}: done, {letters:,} letters in {seconds:.0f} s, {buffer.tell() / 1e6:.1f} MB")
        return {"job": job, "letters": letters, "seconds": seconds, "mb": buffer.tell() / 1e6}


@app.local_entrypoint()
def smoke():
    seq = "ATGC" * 50 + "GGCTTCGGCCGAAGCC" * 4 + "TTAACCGG" * 80
    print(Evo2SAE().encode.remote("smoke", [{"id": "smoke", "seq": seq, "ranges": [(0, len(seq))]}]))
