"""
Read each model's activations on every item, plus its zero-shot verdict, in one forward pass.

    .venv/bin/python experiments/scalable-oversight-101/features.py [0.5b 1.5b 7b]

The prompt is the judge prompt (problem, solution, "Is this solution correct? Answer Yes or No."). For each item and
model it saves the residual stream at the last prompt token at four depths (25, 50, 75 and 100% of the layers) and
the zero-shot score logit(Yes) - logit(No). Saved to ~/.cache/steerlab/scalable-oversight-101/<model>.npz, because
the 7B file is about 230 MB.
"""

import sys

import numpy as np
import torch

from common import CACHE, MODELS, device, items, load, verify_prompt, yes_no_ids

DEPTHS = (0.25, 0.5, 0.75, 1.0)
BATCH = {"0.5b": 32, "1.5b": 32, "7b": 16}


@torch.no_grad()
def extract(name):
    model, tokenizer = load(name)
    rows = items()
    prompts = [verify_prompt(tokenizer, r) for r in rows]
    n_layers = model.config.num_hidden_layers
    layers = [round(d * n_layers) for d in DEPTHS]
    yes, no = yes_no_ids(tokenizer)
    feats, scores = [], []
    size = BATCH[name]
    for i in range(0, len(prompts), size):
        enc = tokenizer(prompts[i : i + size], return_tensors="pt", padding=True).to(device())
        out = model(**enc, output_hidden_states=True)
        feats.append(torch.stack([out.hidden_states[k][:, -1, :] for k in layers], 1).float().cpu().numpy())
        logits = out.logits[:, -1, :].float()
        scores += (logits[:, yes] - logits[:, no]).tolist()
        if i % (size * 20) == 0:
            print(f"{name}: {i + size}/{len(prompts)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE / f"{name}.npz",
        ids=np.array([r["id"] for r in rows]),
        feats=np.concatenate(feats).astype(np.float16),
        zero_shot=np.array(scores, dtype=np.float32),
        layers=np.array(layers),
    )
    print(f"{name}: saved, layers {layers}")
    del model
    if device() == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    for name in sys.argv[1:] or list(MODELS):
        extract(name)
