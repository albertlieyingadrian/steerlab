"""
Step 4: is the SAE any good? Five numbers, each with a pass bar written down before the run:

    FVU            fraction of variance unexplained on held-out tokens: 0 is a perfect rebuild, 1 is "just the mean"
    L0             how many latents are on per token (TopK: k by construction)
    dead / dense   latents that never fire in the interp tokens, or fire on more than 1 token in 10
    CE recovered   splice the SAE's rebuild into the model and finish the forward pass. How much of the loss that
                   mean-ablating the layer costs does the SAE win back? (1 = as good as the clean model)
    vs random      is the SAE's error especially harmful? Add a random push of the same length instead and compare.

Plus a cheap baseline (PCA with as many numbers per token as the SAE uses) and a known-answer check (rebuild + error
term must give back the clean model). It measures every SAE it finds: the hand-written one (native and exported to
SAELens's format, which must agree) and the SAELens-trained one.

    .venv/bin/python experiments/sae-101/04_evaluate.py --size small
"""

import argparse

import numpy as np
import torch

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
args = ap.parse_args()
cfg = s.SIZES[args.size]
work = s.WORK / args.size

model, _, device = s.load_lm()
eval_w = s.load_tokens(args.size, "eval")
interp_w = s.load_tokens(args.size, "interp")
ce_w = eval_w[: cfg["ce"]]
stats = torch.load(work / "stats.pt", weights_only=False)

saes = []
if (work / "sae_handwritten.pt").exists():
    saes.append(s.raw_from_handwritten(work / "sae_handwritten.pt", device))
    meta = {"model_name": s.MODEL, "hook_name": s.HOOK_TL, "hf_hook_name": s.HOOK_HF}
    export = s.export_saelens(work / "sae_handwritten.pt", work / "handwritten_saelens_format", meta)
    saes.append(s.raw_from_saelens_dir(export, device, "hand-written, exported to SAELens format"))
if (work / "saelens" / "cfg.json").exists():
    saes.append(s.raw_from_saelens_dir(work / "saelens", device, "SAELens"))
if not saes:
    raise SystemExit("No SAE found: run 03a_train_handwritten.py (and 03b_train_saelens.py) first.")

eval_acts = s.collect_acts(model, eval_w, device)
print(f"1. Held-out data: {len(eval_w):,} eval windows ({len(eval_acts):,} text tokens) for FVU and L0; "
      f"{len(interp_w):,} interp windows for firing frequencies; the first {len(ce_w)} eval windows for cross-entropy.")

results = {}
for raw in saes:
    exported = raw.name.startswith("hand-written, exported")
    r = s.evaluate(raw, model, eval_windows=eval_w, ce_windows=ce_w, interp_windows=interp_w,
                   mean_act=stats["mean_act"], device=device, eval_acts=eval_acts, controls=not exported)
    freq = r.pop("_freq")
    if not exported:
        np.save(work / f"freq_{'saelens' if raw.name == 'SAELens' else 'handwritten'}.npy", freq)
    results[raw.name] = r
    print(f"\n2. {raw.name}: d_sae {r['d_sae']:,}, k {r['k']}")
    print(f"   FVU {r['fvu']:.4f} (explained variance {1 - r['fvu']:.1%}); L0 {r['l0']:.2f}")
    print(f"   dead {r['dead']:,} ({r['dead_share']:.1%}) and dense {r['dense']:,} ({r['dense_share']:.2%}) of "
          f"{r['d_sae']:,} latents, over {r['freq_tokens']:,} tokens")
    c = r["ce"]
    print(f"   cross-entropy (nats/token, {r['ce_windows']} windows): clean {c['clean']:.4f}, SAE spliced in "
          f"{c['sae']:.4f} (+{r['delta_ce_sae']:.4f}, 95% CI [{r['delta_ce_sae_ci'][0]:.4f}, {r['delta_ce_sae_ci'][1]:.4f}])")
    print(f"     layer mean-ablated {c['mean_ablation']:.4f}, zero-ablated {c['zero_ablation']:.4f}; rebuild + error "
          f"term {c['sae_plus_error']:.4f}")
    print(f"   CE recovered vs mean-ablation {r['ce_recovered_vs_mean']:.4f} [{r['ce_recovered_vs_mean_ci'][0]:.4f}, "
          f"{r['ce_recovered_vs_mean_ci'][1]:.4f}]; vs zero-ablation {r['ce_recovered_vs_zero']:.4f}")
    if "delta_ce_random" in r:
        print(f"   random push of the same length: +{r['delta_ce_random']:.4f} nats; the SAE's error costs "
              f"{r['sae_vs_random_ratio']:.1f}x as much [{r['sae_vs_random_ratio_ci'][0]:.1f}, "
              f"{r['sae_vs_random_ratio_ci'][1]:.1f}]")

# the cheap baseline: PCA with k components, fitted on interp tokens, scored on the eval tokens
fit = s.collect_acts(model, interp_w[:400], device)
ks = sorted({8, 16, 32, 64, 128, 256, cfg["k"]})
pca = s.pca_fvu(fit, eval_acts, ks)
print(f"\n3. Cheap baseline, PCA rebuilt from its top components (dense, fitted on {len(fit):,} other tokens): "
      + ", ".join(f"k={k}: FVU {v:.3f}" for k, v in pca.items()))

# transfer: the same SAE spliced into the Instruct model, on the same text (reported, no bar)
first = saes[0]
del eval_acts, fit
instruct, _, _ = s.load_lm(s.INSTRUCT, device)


def splice_first(h, keep):
    h2 = h.clone()
    h2[keep] = first(h[keep])[0]
    return h2


clean_i = s.spliced_losses(instruct, ce_w, device=device)
spliced_i = s.spliced_losses(instruct, ce_w, replace=splice_first, device=device)
transfer = {"model": s.INSTRUCT, "sae": first.name, "clean": clean_i["ce"], "spliced": spliced_i["ce"],
            "kl": spliced_i["kl"], "fvu": s.fvu_of(first, s.collect_acts(instruct, eval_w[:100], device))[0]}
print(f"\n4. Transfer to {s.INSTRUCT} (no bar): FVU {transfer['fvu']:.4f}; cross-entropy clean {transfer['clean']:.4f}, "
      f"SAE spliced in {transfer['spliced']:.4f} (+{transfer['spliced'] - transfer['clean']:.4f})")
del instruct

s.save_json(s.RESULTS / args.size / "evaluate.json", {"saes": results, "pca_fvu": pca, "transfer": transfer})

main = results[saes[0].name]
print("\nSelf-checks (the pass bars from the plan):")
s.check(f"FVU < 0.25 ({main['fvu']:.4f})", main["fvu"] < 0.25)
s.check(f"L0 = k (mean {main['l0']:.3f}, k {main['k']})", abs(main["l0"] - main["k"]) < 0.01)
s.check(f"dead latents < 10% ({main['dead_share']:.2%})", main["dead_share"] < 0.10)
s.check(f"dense latents < 1% ({main['dense_share']:.2%})", main["dense_share"] < 0.01)
s.check(f"CE recovered vs mean-ablation >= 0.90 ({main['ce_recovered_vs_mean']:.4f})", main["ce_recovered_vs_mean"] >= 0.90)
s.check(f"rebuild + error term gives the clean model back (relative difference {main['error_term_rel_diff']:.1e} < 1e-4)",
        main["error_term_rel_diff"] < 1e-4)
s.check(f"the SAE beats PCA with the same number of numbers per token (FVU {main['fvu']:.3f} vs {pca[cfg['k']]:.3f})",
        main["fvu"] < pca[cfg["k"]])
if len(saes) > 1 and saes[1].name.startswith("hand-written, exported"):
    exp = results[saes[1].name]
    s.check(f"the SAELens-format export gives the same FVU ({exp['fvu']:.6f} vs {main['fvu']:.6f})",
            abs(exp["fvu"] - main["fvu"]) < 1e-4)
