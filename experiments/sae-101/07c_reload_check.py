"""
Step 7c: load the SAE back from the Hub in a fresh process (on Modal: a fresh machine, with nothing from training on
disk except the token windows) and recompute step 4's numbers through SAELens's own loader. They must match the local
ones; if they do not, the upload, the format or the folded normalisation is wrong.

It also checks the hook name in the card: TransformerLens's `blocks.11.hook_resid_post` (through SAELens's
TransformerBridge, no weight processing) must give the same activations as the Hugging Face module `model.layers.11`.

    (on Modal: python experiments/sae-101/modal_app.py spawn reload)
"""

import argparse

import numpy as np
import torch

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
ap.add_argument("--repo", default="albertlieadrian/qwen2.5-0.5b-sae-101")
args = ap.parse_args()
cfg = s.SIZES[args.size]

from sae_lens import SAE, __version__

local = s.read_json(s.RESULTS / args.size / "evaluate.json")
model, _, device = s.load_lm()
eval_w = s.load_tokens(args.size, "eval")
interp_w = s.load_tokens(args.size, "interp")
stats = torch.load(s.WORK / args.size / "stats.pt", weights_only=False)
eval_acts = s.collect_acts(model, eval_w, device)
print(f"1. SAELens {__version__}, loading from the Hub: {args.repo}")

# Always check the hand-written SAE. The SAELens twin is optional (07a packages it, and 04 evaluates it, only when
# work/<size>/saelens/cfg.json exists), so check it only when step 4's local results include it.
to_check = [(s.HOOK_TL, "hand-written")]
if "SAELens" in local["saes"]:
    to_check.append((f"saelens-trained/{s.HOOK_TL}", "SAELens"))
else:
    print("   (no SAELens twin in the local results, so only the hand-written SAE is checked)")

checks = []
for sae_id, local_name in to_check:
    sae = SAE.from_pretrained(args.repo, sae_id, device=device)
    raw = s.RawSAE(sae.encode, sae.decode, sae.cfg.d_sae, sae.cfg.k, sae.W_dec.detach(), f"Hub: {sae_id}")
    r = s.evaluate(
        raw,
        model,
        eval_windows=eval_w,
        ce_windows=eval_w[: cfg["ce"]],
        interp_windows=interp_w,
        mean_act=stats["mean_act"],
        device=device,
        eval_acts=eval_acts,
        controls=False,
    )
    r.pop("_freq")
    ref = local["saes"][local_name]
    d = {
        "fvu": abs(r["fvu"] - ref["fvu"]),
        "ce": abs(r["ce"]["sae"] - ref["ce"]["sae"]),
        "dead": abs(r["dead"] - ref["dead"]),
        "l0": abs(r["l0"] - ref["l0"]),
    }
    print(
        f"\n2. {sae_id}: FVU {r['fvu']:.4f} (local {ref['fvu']:.4f}); L0 {r['l0']:.2f}; dead {r['dead']} "
        f"(local {ref['dead']}); CE with the SAE {r['ce']['sae']:.4f} (local {ref['ce']['sae']:.4f}); CE recovered "
        f"{r['ce_recovered_vs_mean']:.4f}"
    )
    checks.append((sae_id, d, r))

# the hook-name check: TransformerBridge vs the Hugging Face module
from sae_lens.load_model import load_model

bridge = load_model("TransformerBridge", s.MODEL, device=device, model_from_pretrained_kwargs={"dtype": torch.float32})
toks = torch.as_tensor(np.asarray(eval_w[:4], dtype=np.int64), device=device)
with torch.no_grad():
    _, cache = bridge.run_with_cache(toks, names_filter=[s.HOOK_TL])
    tl = cache[s.HOOK_TL]
    hf_acts = s.residual(model, toks)
mask = s.text_mask(toks)
gap = float((tl[mask] - hf_acts[mask]).abs().max())
rel = gap / float(hf_acts[mask].abs().max())
print(
    f"\n3. {s.HOOK_TL} through TransformerBridge vs {s.HOOK_HF} in transformers: max difference {gap:.2e} "
    f"({rel:.1e} of the largest activation)"
)

s.save_json(
    s.RESULTS / args.size / "reload.json",
    {
        "repo": args.repo,
        "sae_lens_version": __version__,
        "hub": {sid: r for sid, _, r in checks},
        "differences": {sid: d for sid, d, _ in checks},
        "bridge_max_abs_diff": gap,
        "bridge_rel_diff": rel,
    },
)
print("\nSelf-checks:")
for sid, d, _ in checks:
    s.check(
        f"{sid}: the Hub copy matches the local numbers (FVU off by {d['fvu']:.1e}, CE by {d['ce']:.1e}; < 1e-3)",
        d["fvu"] < 1e-3 and d["ce"] < 1e-3,
    )
s.check(f"the TransformerLens hook name reads the same activations (relative difference {rel:.1e} < 1e-4)", rel < 1e-4)
