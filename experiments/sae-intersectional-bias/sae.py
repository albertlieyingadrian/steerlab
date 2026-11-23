"""A small loader for the released DeBiasLens SAE (Matryoshka batch-top-k), matching
`sae-for-vlm/dictionary_learning/trainers/matroyshka_batch_top_k.py::MatroyshkaBatchTopKSAE` at inference:

    encode(x) = relu((x - b_dec) @ W_enc + b_enc) * (that > threshold)      # use_threshold=True, the default
    decode(f) = f @ W_dec + b_dec

The released `evaluate.py` loads `checkpoints/ae_100000.pt` of the ViT-L/14-336 FairFace x8 run.
"""

from __future__ import annotations

import torch

REPO = "kaist-ai/debiaslens"
FILE = ("matroyshka_batch_top_k_20_x8/random_k_2/fairface_train_activations_clip-vit-large-patch14-336_23_post_mlp_"
        "residual_matroyshka_batch_top_k_20_x8/trainer_0/checkpoints/ae_100000.pt")  # fmt: skip


class SAE(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        for k in ("W_enc", "b_enc", "W_dec", "b_dec"):
            self.register_buffer(k, sd[k].float())
        self.threshold = float(sd["threshold"])
        self.group_sizes = [int(g) for g in sd["group_sizes"].tolist()]

    @classmethod
    def load(cls, path=None):
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(REPO, FILE)
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def encode(self, x):
        f = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return f * (f > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def level(self):
        """Matryoshka level (0-3) of each latent."""
        out = torch.empty(sum(self.group_sizes), dtype=torch.long)
        s = 0
        for i, g in enumerate(self.group_sizes):
            out[s : s + g] = i
            s += g
        return out
