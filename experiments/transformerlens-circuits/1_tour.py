"""Step 1: boot GPT-2 small through TransformerLens's bridge, show it is the same model Hugging Face runs, look inside
the cache, and edit one head with a hook.

    python experiments/transformerlens-circuits/1_tour.py        # ~30 s once gpt2 is cached (first run: ~550 MB)

1. TransformerBridge.boot_transformers("gpt2"): the Hugging Face model, wrapped so every internal value has a name.
2. The same tokens through the bridge and through transformers' GPT2LMHeadModel: logits agree to ~1e-4.
3. enable_compatibility_mode(): LayerNorm folded, weights centred. Logits shift by a constant per position;
   predictions and logit differences do not change.
4. run_with_cache(): every intermediate value, keyed by hook name.
5. run_with_hooks(): zero one head's output and watch the answer move.
Writes results/tour.json.
"""

import importlib.metadata
from collections import Counter

import torch
from common import check, write_json
from transformer_lens import TransformerBridge
from transformer_lens.utilities.devices import get_device
from transformers import GPT2LMHeadModel

torch.set_grad_enabled(False)
PROMPT = "When Mary and John went to the store, John gave a drink to"
IO, S = " Mary", " John"
versions = {name: importlib.metadata.version(name) for name in ("transformer-lens", "transformers", "torch")}
print("versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))

# --- 1. Boot -------------------------------------------------------------------------------------------------------
device = get_device()  # TransformerLens picks CPU on a Mac unless TRANSFORMERLENS_ALLOW_MPS=1
model = TransformerBridge.boot_transformers("gpt2", device=device)
cfg = model.cfg
tokens = model.to_tokens(PROMPT)  # the bridge prepends <|endoftext|> as BOS
print(
    f"\n1. gpt2 on {device}: {cfg.n_layers} layers x {cfg.n_heads} heads, d_model {cfg.d_model}; "
    f"{tokens.shape[1]} tokens"
)
bridge_logits = model(tokens)
io_id, s_id = model.to_single_token(IO), model.to_single_token(S)

# --- 2. Parity with Hugging Face -----------------------------------------------------------------------------------
# The bridge wraps the transformers module and attaches hook points; it does not re-implement it. Same tokens, same
# attention kernel ("eager"), so only floating-point summation order can differ.
kernel = model.original_model.config._attn_implementation
hf_logits = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation=kernel).eval()(tokens.cpu()).logits
hf_logits = hf_logits.to(bridge_logits.device)
max_diff_eager = (bridge_logits - hf_logits).abs().max().item()
same_argmax = bool((bridge_logits.argmax(-1) == hf_logits.argmax(-1)).all())
sdpa_logits = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="sdpa").eval()(tokens.cpu()).logits
max_diff_sdpa = (bridge_logits - sdpa_logits.to(bridge_logits.device)).abs().max().item()
pred_id = bridge_logits[0, -1].argmax().item()
ld_before = (bridge_logits[0, -1, io_id] - bridge_logits[0, -1, s_id]).item()
print(f"\n2. max |bridge - HF| over {bridge_logits.numel():,} logits, same '{kernel}' kernel: {max_diff_eager:.2e}")
print(
    f"   same next-token prediction at every position: {same_argmax}; against HF's 'sdpa' kernel: {max_diff_sdpa:.2e}"
)
print(f"   prediction: {model.to_string(pred_id)!r}; logit({IO!r}) - logit({S!r}) = {ld_before:+.3f}")

# --- 3. Compatibility mode -----------------------------------------------------------------------------------------
model.enable_compatibility_mode(disable_warnings=True)
compat_logits, cache = model.run_with_cache(tokens)
shift = compat_logits - hf_logits
spread = (shift - shift.mean(-1, keepdim=True)).abs().max().item()
same_argmax_compat = bool((compat_logits.argmax(-1) == hf_logits.argmax(-1)).all())
ld_after = (compat_logits[0, -1, io_id] - compat_logits[0, -1, s_id]).item()
print(
    f"\n3. compatibility mode: raw logits move by up to {shift.abs().max().item():.1f}, but by the same amount "
    f"for every"
)
print(f"   vocabulary entry at a position (spread {spread:.1e}); same predictions: {same_argmax_compat};")
print(f"   logit difference {ld_before:+.3f} -> {ld_after:+.3f}")

# --- 4. The cache --------------------------------------------------------------------------------------------------
names = list(cache.keys())
per_block = Counter(name.split(".")[1] for name in names if name.startswith("blocks."))
n_pos = tokens.shape[1]
rows_sum_to_one = torch.allclose(
    cache["blocks.0.attn.hook_pattern"].sum(-1), torch.ones(1, cfg.n_heads, n_pos), atol=1e-4
)
print(
    f"\n4. run_with_cache recorded {len(names)} hook names: {per_block['0']} per block, "
    f"{len(names) - sum(per_block.values())} outside the blocks"
)
for name in ("blocks.0.hook_resid_pre", "blocks.0.attn.hook_pattern", "blocks.0.attn.hook_z"):
    print(f"   {name:<28} {tuple(cache[name].shape)}")
print(f"   every attention row sums to 1: {rows_sum_to_one}")


# --- 5. Edit with a hook -------------------------------------------------------------------------------------------
# L10H7 is a published "negative name mover" (it argues against the right name); L9H9 the strongest "name mover".
def silence(head):
    def hook_fn(z, hook):
        z[:, :, head, :] = 0.0
        return z

    return hook_fn


def ld_with(layer, head):
    logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", silence(head))])
    return (logits[0, -1, io_id] - logits[0, -1, s_id]).item()


ld_no_10_7, ld_no_9_9 = ld_with(10, 7), ld_with(9, 9)
print(f"\n5. logit difference {ld_after:+.3f}; with L10H7 zeroed {ld_no_10_7:+.3f}; with L9H9 zeroed {ld_no_9_9:+.3f}")
print("   (zeroing the top name mover does not lower it: backup heads in layers 9-11 take over)")

write_json(
    "tour.json",
    {
        "versions": versions,
        "device": str(device),
        "prompt": PROMPT,
        "n_logits": bridge_logits.numel(),
        "max_diff_eager": max_diff_eager,
        "max_diff_sdpa": max_diff_sdpa,
        "same_argmax": same_argmax,
        "compat_max_shift": shift.abs().max().item(),
        "compat_shift_spread": spread,
        "compat_same_argmax": same_argmax_compat,
        "ld_before_compat": ld_before,
        "ld_after_compat": ld_after,
        "n_hooks": len(names),
        "hooks_per_block": per_block["0"],
        "ld_L10H7_zeroed": ld_no_10_7,
        "ld_L9H9_zeroed": ld_no_9_9,
    },
)

print("\nSelf-checks:")
check("bridge and Hugging Face logits agree within 1e-4 (same tokens, same kernel)", max_diff_eager < 1e-4)
check("same next token at every position, before and after compatibility mode", same_argmax and same_argmax_compat)
check("compatibility mode preserves the logit difference (within 1e-2)", abs(ld_after - ld_before) < 1e-2)
check("the model predicts the indirect object ' Mary'", pred_id == io_id)
check("attention patterns are distributions (rows sum to 1)", rows_sum_to_one)
check("zeroing the negative name mover L10H7 raises the logit difference", ld_no_10_7 > ld_after)
