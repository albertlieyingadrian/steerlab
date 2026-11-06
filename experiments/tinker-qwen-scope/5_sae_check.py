"""Step 5: does Qwen-Scope's SAE describe these models' residual stream well enough to trust its features? (local)

    python experiments/tinker-qwen-scope/5_sae_check.py

Qwen-Scope's SAEs were trained on Qwen3-8B-Base; this study reads Qwen3-8B (the instruct model) and two fine-tunes of
it. Before any feature claim, each layer's SAE reconstructs the residual vectors step 4 saved, and three numbers say
how well:

- variance explained: 1 - (squared error) / (spread of the vectors around their mean). 1 is perfect, 0 is no better
  than predicting the mean;
- cosine: how closely each reconstruction points the same way as the original (1 = same direction);
- L0: how many of the 65,536 features are active per token (the release keeps the top 50).

Measured four ways per layer: every token of 50 prompts and the last token of all 450 (150 questions x 3 conditions),
on the base model and on the lock. The two layers with the highest variance explained on the base model's tokens are
the ones step 6 reports first (the rule was fixed before the check ran); a layer under 0.5 would be dropped.
"""

from __future__ import annotations

import sys

import common as c
import numpy as np

MIN_VARIANCE_EXPLAINED = 0.5


def load_sae(layer: int):
    from huggingface_hub import hf_hub_download
    from steerlab.saes import qwen_scope

    path = hf_hub_download(c.SAE_RELEASE, qwen_scope.filename(layer))
    return qwen_scope.load_qwen_scope(path, layer=layer, release=c.SAE_RELEASE)


def stats(sae, hidden: np.ndarray) -> dict:
    import torch

    with torch.inference_mode():
        h = torch.from_numpy(hidden).to(sae.device, torch.float32)
        acts = sae.encode(h)
        recon = sae.decode(acts)
        cosine = torch.nn.functional.cosine_similarity(h, recon, dim=-1).mean()
        residual = float(((h - recon) ** 2).sum())
        total = float(((h - h.mean(dim=0, keepdim=True)) ** 2).sum())
        l0 = (acts != 0).float().sum(-1).mean()
    return {"cosine": float(cosine), "variance_explained": 1 - residual / total, "l0": float(l0), "n": int(h.shape[0])}


def shift_stats(sae, last, k: int, org: str) -> dict:
    """Does the SAE keep the fine-tune's change at the answer position? The true shift (fine-tuned minus base residual,
    last token, no prefix) against the same difference of the two SAE reconstructions: their mean cosine, their mean
    lengths, and the share of the true shift's squared length that the reconstructed shift explains (1 - error/true)."""
    import torch

    with torch.inference_mode():
        after = torch.from_numpy(last[f"{org}_none"][:, k, :]).to(sae.device)
        before = torch.from_numpy(last["base_none"][:, k, :]).to(sae.device)
        true = after - before
        recon = sae.decode(sae.encode(after)) - sae.decode(sae.encode(before))
        cosine = torch.nn.functional.cosine_similarity(true, recon, dim=-1).mean()
        kept = 1 - float(((true - recon) ** 2).sum()) / float((true**2).sum())
    return {
        "true_norm": float(true.norm(dim=-1).mean()),
        "sae_norm": float(recon.norm(dim=-1).mean()),
        "cosine": float(cosine),
        "kept": kept,
    }


def main() -> int:
    import torch

    work = c.work_dir()
    last = np.load(work / "residuals.npz")
    layers = [int(x) for x in last["layers"]]
    out: dict = {"release": c.SAE_RELEASE, "layers": {}}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for k, layer in enumerate(layers):
        sae = load_sae(layer).to(device)
        row = {}
        for side in ("base", "lock"):
            row[f"{side}_all"] = stats(sae, np.load(work / f"all_tokens_{side}.npz")[str(layer)])
            vectors = np.concatenate([last[f"{side}_{cond}"][:, k, :] for cond in c.CONDITIONS])
            row[f"{side}_last"] = stats(sae, vectors)
        out["layers"][str(layer)] = row
        print(f"layer {layer}:")
        for name, s in row.items():
            print(
                f"  {name:<10} variance explained {s['variance_explained']:.3f}  cosine {s['cosine']:.3f}  "
                f"L0 {s['l0']:.1f}  ({s['n']} vectors)"
            )
        shifts = {}
        for org in ("lock", "clean", "random"):
            shifts[org] = shift_stats(sae, last, k, org)
            s = shifts[org]
            print(
                f"  {org:<6} shift: |true| {s['true_norm']:.2f}, through the SAE {s['sae_norm']:.2f}, cosine "
                f"{s['cosine']:.3f}, share of the true shift kept {s['kept']:.3f}"
            )
        row["shift"] = shifts
        del sae
    ranked = sorted(layers, key=lambda L: -out["layers"][str(L)]["base_all"]["variance_explained"])
    good = [L for L in ranked[:2] if out["layers"][str(L)]["base_all"]["variance_explained"] >= MIN_VARIANCE_EXPLAINED]
    out["chosen"] = sorted(good)
    out["rule"] = "the two layers with the highest variance explained on the base model's tokens, if at least 0.5"
    print(f"chosen layers: {out['chosen']}")
    c.write_json(c.RESULTS / "sae_check.json", out)
    ok = c.check("two layers pass the reconstruction bar", len(good) == 2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
