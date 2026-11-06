"""Step 3: the IOI task on GPT-2 small, and TransformerLens's two standard patching sweeps over it.

    python experiments/transformerlens-circuits/3_ioi_patching.py   # ~1 min with gpt2 cached

Activation patching: run the CORRUPTED prompt, but copy ONE value in from the CLEAN run (a layer x position of the
residual stream, or one head's output), and measure how much of the clean answer comes back. Like swapping one field of
a request, at one middleware stage, with the value from a known-good request: where the swap fixes the response is
where the decision lives.

  A. patching.get_act_patch_resid_pre: 12 layers x 15 positions = 180 forward passes
  B. patching.get_act_patch_attn_head_out_all_pos: 144 heads, each head's clean output copied in at every position
The metric is logit(IO) - logit(S) at the last position, rescaled so 0 = the corrupted run and 1 = the clean run.
Writes results/ioi_prompts.json, results/ioi_patching.json, results/ioi_resid_patch.csv, results/ioi_head_patch.csv.
"""

import time

from common import check, write_grid_csv, write_json
from ioi_tools import END_POS, POSITION_LABELS, PUBLISHED, S2_POS, Setup, head_label, logit_diff
from transformer_lens import patching

setup = Setup()
setup.describe()
model = setup.model
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

# --- 1. Prompts and baselines --------------------------------------------------------------------------------------
clean_per = logit_diff(setup.clean_logits, setup.answer_tokens, per_prompt=True)
corrupted_per = logit_diff(setup.corrupted_logits, setup.answer_tokens, per_prompt=True)
top_clean = setup.clean_logits[:, -1].argmax(-1)
n_top_is_io = int((top_clean == setup.answer_tokens[:, 0]).sum())
diff_positions = (setup.clean_tokens != setup.corrupted_tokens).nonzero()[:, 1].unique().tolist()
print("\n1. Eight prompt pairs")
for i, p in enumerate(setup.prompts):
    print(
        f"  {p['template']}  {p['clean']!r} -> {p['io']!r}  clean {clean_per[i]:+.2f}, corrupted "
        f"{corrupted_per[i]:+.2f}"
    )
print(f"  clean and corrupted differ only at position(s) {diff_positions}; top guess is IO on {n_top_is_io} of 8")
write_json(
    "ioi_prompts.json",
    [
        dict(p, clean_ld=round(clean_per[i].item(), 4), corrupted_ld=round(corrupted_per[i].item(), 4))
        for i, p in enumerate(setup.prompts)
    ],
)

# --- 2. Residual-stream sweep --------------------------------------------------------------------------------------
started = time.time()
resid = patching.get_act_patch_resid_pre(model, setup.corrupted_tokens, setup.clean_cache, setup.metric).cpu()
resid_seconds = time.time() - started
print(f"\n2. get_act_patch_resid_pre: {resid.numel()} patched passes in {resid_seconds:.1f} s")
print(f"  {'layer':<6}" + "".join(f"{label:>7}" for label in POSITION_LABELS))
for layer in range(n_layers):
    print(f"  {layer:<6}" + "".join(f"{v:>7.2f}" for v in resid[layer]))
others = [p for p in range(resid.shape[1]) if p not in (S2_POS, END_POS)]
outside = resid[:, others].abs().max().item()
print(f"  largest |recovery| outside S2 and the final position: {outside:.3f}")

# --- 3. Head-output sweep ------------------------------------------------------------------------------------------
started = time.time()
heads = patching.get_act_patch_attn_head_out_all_pos(model, setup.corrupted_tokens, setup.clean_cache, setup.metric)
heads = heads.cpu()
head_seconds = time.time() - started
order = sorted(((heads[ly, h].item(), (ly, h)) for ly in range(n_layers) for h in range(n_heads)), reverse=True)
rank = {head: i + 1 for i, (_, head) in enumerate(order)}
print(f"\n3. get_act_patch_attn_head_out_all_pos: {heads.numel()} patched passes in {head_seconds:.1f} s")
print("  top five: " + ", ".join(f"{head_label(*h)} {v:+.2f}" for v, h in order[:5]))
print("  bottom three: " + ", ".join(f"{head_label(*h)} {v:+.2f}" for v, h in order[-3:]))
print("  published name movers by rank: " + ", ".join(f"{head_label(*h)} #{rank[h]}" for h in PUBLISHED["name mover"]))
print(
    "  published S-inhibition by rank: " + ", ".join(f"{head_label(*h)} #{rank[h]}" for h in PUBLISHED["S-inhibition"])
)
name_movers_top5 = [h for h in PUBLISHED["name mover"] if rank[h] <= 5]

write_grid_csv("ioi_resid_patch.csv", resid, [f"L{i}" for i in range(n_layers)], POSITION_LABELS, corner="layer")
write_grid_csv(
    "ioi_head_patch.csv", heads, [f"L{i}" for i in range(n_layers)], [f"H{h}" for h in range(n_heads)], "layer"
)
write_json(
    "ioi_patching.json",
    {
        "clean_ld": setup.clean_ld,
        "corrupted_ld": setup.corrupted_ld,
        "top_guess_is_io": n_top_is_io,
        "differing_positions": diff_positions,
        "resid_sweep_passes": resid.numel(),
        "resid_sweep_seconds": round(resid_seconds, 1),
        "head_sweep_passes": heads.numel(),
        "head_sweep_seconds": round(head_seconds, 1),
        "resid_outside_s2_end_max_abs": outside,
        "head_top5": [[head_label(*h), round(v, 4)] for v, h in order[:5]],
        "head_bottom3": [[head_label(*h), round(v, 4)] for v, h in order[-3:]],
    },
)

print("\nSelf-checks:")
check("clean and corrupted differ only at S2", diff_positions == [S2_POS])
check("the model's top guess is IO on at least 7 of 8 clean prompts", n_top_is_io >= 7)
check(
    "resid sweep: layer 0 at S2 recovers everything, layer 0 elsewhere nothing",
    resid[0, S2_POS] > 0.95 and resid[0, [p for p in range(15) if p != S2_POS]].abs().max() < 0.05,
)
check(
    "resid sweep: in layers 9-11 the final position carries more than S2",
    resid[9:, END_POS].mean() > resid[9:, S2_POS].mean(),
)
check(
    "both published negative name movers have negative head-patching recovery",
    all(heads[h].item() < 0 for h in PUBLISHED["negative name mover"]),
)
check("as first written: all three published name movers in the top five by head patching", len(name_movers_top5) == 3)
