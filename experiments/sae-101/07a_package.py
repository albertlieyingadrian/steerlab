"""
Step 7a: package the SAEs for the Hugging Face Hub in SAELens's format, with a model card.

The repo it builds (work/<size>/hf/):

    README.md                                   the model card: what, how trained, metrics, how to load, limits
    blocks.11.hook_resid_post/                  the hand-written SAE (the main one)
        cfg.json                                architecture, sizes, and the metadata a loader needs
        sae_weights.safetensors                 W_enc, b_enc, W_dec, b_dec (normalisation folded in: raw activations)
        sparsity.safetensors                    log10 of how often each latent fired on 5M held-out tokens
    saelens-trained/blocks.11.hook_resid_post/  the SAELens-trained twin, as SAELens saved it

Each folder name is an "SAE id": SAE.from_pretrained("<repo>", "<sae id>") loads it.

    .venv/bin/python experiments/sae-101/07a_package.py --size full --repo albertlieadrian/qwen2.5-0.5b-sae-101
"""

import argparse
import json
import shutil

import numpy as np
import torch
from safetensors.torch import save_file

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
ap.add_argument("--repo", default="albertlieadrian/qwen2.5-0.5b-sae-101")
ap.add_argument("--note-url", default="https://albertlieyingadrian.github.io/steerlab/experiments/sae-101/report/")
args = ap.parse_args()
work = s.WORK / args.size
res = s.RESULTS / args.size
ev = s.read_json(res / "evaluate.json")
tr = s.read_json(res / "train_handwritten.json")
trl = s.read_json(res / "train_saelens.json")
col = s.read_json(res / "collect.json")
if not (ev and tr):
    raise SystemExit("Run steps 3a and 4 first (results/<size>/evaluate.json is missing).")
cfg = s.SIZES[args.size]
hf = work / "hf"
if hf.exists():
    shutil.rmtree(hf)

SAE_ID = s.HOOK_TL
meta = {
    "model_name": s.MODEL, "hook_name": s.HOOK_TL, "hf_hook_name": s.HOOK_HF, "hook_head_index": None,
    "model_class_name": "TransformerBridge", "model_from_pretrained_kwargs": {},
    "dataset_path": f"{s.DATASET} ({s.DATASET_CONFIG})", "context_size": s.CTX, "prepend_bos": False,
    "seqpos_slice": [s.SKIP, None], "exclude_special_tokens": [s.EOT], "training_tokens": tr["tokens"],
    "sae_lens_version": "6.56.1", "sae_lens_training_version": "hand-written (sae101.py)",
}
main = s.export_saelens(work / "sae_handwritten.pt", hf / SAE_ID, meta)
freq = np.load(work / "freq_handwritten.npy")
save_file({"sparsity": torch.tensor(np.log10(freq + 1e-10), dtype=torch.float32)}, str(main / "sparsity.safetensors"))
ids = [SAE_ID]
if (work / "saelens" / "cfg.json").exists():
    twin = hf / "saelens-trained" / SAE_ID
    twin.mkdir(parents=True)
    for name in ("cfg.json", "sae_weights.safetensors", "sparsity.safetensors"):
        if (work / "saelens" / name).exists():
            shutil.copy(work / "saelens" / name, twin / name)
    c = json.loads((twin / "cfg.json").read_text())
    c["device"] = "cpu"
    (twin / "cfg.json").write_text(json.dumps(c, indent=1) + "\n")
    ids.append(f"saelens-trained/{SAE_ID}")

hw = ev["saes"]["hand-written"]
sl = ev["saes"].get("SAELens")
pca_k = ev["pca_fvu"][str(cfg["k"])]


def row(name, r):
    return (f"| {name} | {r['fvu']:.3f} | {r['l0']:.0f} | {r['dead']:,} ({r['dead_share']:.2%}) | {r['dense_share']:.2%} | "
            f"{r['ce']['sae']:.3f} | {r['ce_recovered_vs_mean']:.3f} |")


rows = row("hand-written", hw) + ("\n" + row("SAELens-trained", sl) if sl else "")
card = f"""---
license: apache-2.0
library_name: saelens
base_model: {s.MODEL}
datasets:
- {s.DATASET}
tags:
- sparse-autoencoder
- sae
- saelens
- interpretability
- teaching
---

# A sparse autoencoder on Qwen2.5-0.5B's residual stream (a teaching artifact)

This is the worked example of the tutorial ["How to train a sparse autoencoder: a 101 on a small language model"]({args.note_url}).
It was trained to show every step of making and publishing an SAE, not to be a state-of-the-art SAE. Use it to learn,
to check your own pipeline against known numbers, or as a starting point. For research, prefer a larger, better-tested release.

## What it is

| | |
|---|---|
| Model | [{s.MODEL}](https://huggingface.co/{s.MODEL}) (base) |
| Site | residual stream after block {s.LAYER} of 24: `{s.HOOK_TL}` in TransformerLens / SAELens; the output of `{s.HOOK_HF}` in Hugging Face transformers (`output_hidden_states=True` gives it as `hidden_states[{s.LAYER + 1}]`) |
| Architecture | TopK SAE, {hw['d_sae']:,} latents ({hw['d_sae'] / s.D_MODEL:.1f}x d_model {s.D_MODEL}), k = {hw['k']} active per token |
| Training data | {tr['tokens'] / 1e6:.0f}M tokens of [FineWeb-Edu](https://huggingface.co/datasets/{s.DATASET}) (`{s.DATASET_CONFIG}`), 512-token windows |
| Not trained on | the first {s.SKIP} positions of a sequence, and `<|endoftext|>` tokens: Qwen2.5 puts an "attention sink" with about {col['pos0_ratio']:.0f}x the usual norm there |
| Normalisation | trained on activations x {col['scale']:.4f}; the factor is folded into the weights, so feed **raw** activations |
| Training | hand-written PyTorch (Adam, lr {tr['lr']:g}, batch {tr['batch']:,} tokens, AuxK for dead latents), {tr['minutes']:.0f} minutes on one A10G |

Two SAE ids are in this repo:

- `{SAE_ID}`: the hand-written SAE (the main one).
- `saelens-trained/{SAE_ID}`: the same recipe trained with SAELens on the same tokens, for comparison. Its `cfg.json` names the Hugging Face hook (`{s.HOOK_HF}`), because SAELens trained it on a plain `AutoModelForCausalLM`.

## How good it is (held-out FineWeb-Edu)

| SAE | FVU | L0 | dead | dense (>10% of tokens) | CE with SAE (nats) | CE recovered vs mean-ablation |
|---|---|---|---|---|---|---|
{rows}

Clean cross-entropy is {hw['ce']['clean']:.3f} nats per token; mean-ablating the layer gives {hw['ce']['mean_ablation']:.3f}.
FVU is the fraction of variance unexplained on {hw['eval_tokens']:,} held-out tokens (PCA with {cfg['k']} components:
{pca_k:.3f}). Dead and dense latents are counted over {hw['freq_tokens']:,} held-out tokens. Splicing the SAE in costs
{hw['delta_ce_sae']:.3f} nats per token, {hw['sae_vs_random_ratio']:.1f}x as much as a random error of the same size,
so the part the SAE misses matters more to the model than its size suggests.

## How to load it

With SAELens (6.x):

```python
from sae_lens import SAE
sae = SAE.from_pretrained("{args.repo}", "{SAE_ID}")
# feed it raw residual-stream activations after block {s.LAYER}, skipping the first {s.SKIP} positions
latents = sae.encode(acts)          # [tokens, {hw['d_sae']}], at most {hw['k']} non-zero per token
rebuilt = sae.decode(latents)
```

With plain PyTorch (no SAELens):

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
w = load_file(hf_hub_download("{args.repo}", "{SAE_ID}/sae_weights.safetensors"))
k = json.load(open(hf_hub_download("{args.repo}", "{SAE_ID}/cfg.json")))["k"]
pre = (acts - w["b_dec"]) @ w["W_enc"] + w["b_enc"]
vals, idx = pre.topk(k, dim=-1)
latents = torch.zeros_like(pre).scatter(-1, idx, vals.relu())
rebuilt = latents @ w["W_dec"] + w["b_dec"]
```

The tutorial's code (data, training, evaluation, interpretation, this card): [steerlab, experiments/sae-101](https://github.com/albertlieyingadrian/steerlab/tree/main/experiments/sae-101).

## Limitations

- Small and short: {hw['d_sae']:,} latents and {tr['tokens'] / 1e6:.0f}M tokens. Public SAEs for research use far more
  of both; more latents split broad features into finer ones.
- One layer, one model. It transfers partly to Qwen2.5-0.5B-Instruct (see the tutorial), but it was not trained there.
- English educational web text only; code, other languages and chat are out of distribution.
- Latent labels in the tutorial are readings of top examples, not tested explanations. A latent that fires on a concept
  need not be how the model uses that concept.
- Do not use the outputs at the first {s.SKIP} positions of a sequence (attention sinks).

## License

Apache-2.0, the license of the base model. The training text is FineWeb-Edu (ODC-By 1.0); no text is redistributed here.
"""
(hf / "README.md").write_text(card)
total = sum(p.stat().st_size for p in hf.rglob("*") if p.is_file())
print(f"1. Built work/{args.size}/hf/ for {args.repo}: SAE ids {ids}; {total / 1e6:.0f} MB in all")
for p in sorted(hf.rglob("*")):
    if p.is_file():
        print(f"   {str(p.relative_to(hf)):<64} {p.stat().st_size / 1e6:8.1f} MB")

print("\nSelf-checks:")
c = json.loads((main / "cfg.json").read_text())
s.check(f"cfg.json says TopK, k {c['k']}, d_sae {c['d_sae']}, raw activations (normalize_activations "
        f"{c['normalize_activations']!r})", c["architecture"] == "topk" and c["normalize_activations"] == "none")
reread = s.raw_from_saelens_dir(main, "cpu")
handw = s.raw_from_handwritten(work / "sae_handwritten.pt", "cpu")
x = torch.randn(64, s.D_MODEL) * 16
with torch.no_grad():
    diff = float((reread(x)[0] - handw(x)[0]).abs().max())
s.check(f"the packaged file rebuilds exactly like the trained SAE (max difference {diff:.1e} < 1e-4)", diff < 1e-4)
s.check("the card names the base model, the data and the license, and says it is a teaching artifact",
        all(w_ in card for w_ in (s.MODEL, s.DATASET, "apache-2.0", "teaching artifact")))
