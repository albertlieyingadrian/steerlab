"""Step 6: load the sandbagging direction into TransformerLens and read it, layer by layer, with a random-direction
control.

    python experiments/transformerlens-circuits/6_direction_read.py   # ~1 min with Qwen2.5-0.5B-Instruct cached

1. Boot Qwen2.5-0.5B-Instruct through the bridge (CPU, float32, compatibility mode on).
2. Plumbing test: a hook that adds 0 x the direction must leave greedy tokens identical. Then one steered example.
3. project(cache, direction) at the final prompt position, on eight sandbagging prompts (a trigger sentence before a
   factual question) and the same eight questions without the trigger.
4. The control: 20 random unit directions at the same layer, read the same way. A trigger sentence changes the prompt,
   so any direction might separate the two sets; the control says how far chance alone gets.
Writes results/direction_projection.csv and results/direction_read.json.
"""

import csv

import torch
from common import RESULTS, boot, check, write_json
from direction_tools import MODEL_NAME, chat_tokens, greedy, load_direction, prompt_pairs, random_directions
from lens_bridge import add_direction_hook, project

N_RANDOM = 20
direction = load_direction()
model, device = boot(MODEL_NAME)
cfg = model.cfg
layer = direction.layer
print(
    f"1. {MODEL_NAME} on {device}: {cfg.n_layers} layers x {cfg.n_heads} heads, d_model {cfg.d_model}; direction at"
    f" {direction.hook_name}"
)

# --- 2. Strength-0 test and one steered example --------------------------------------------------------------------
question = "What is the capital of Germany? Answer in a few words."
tokens = chat_tokens(model, question)
n = tokens.shape[1]
baseline = greedy(model, tokens)
handle = add_direction_hook(model, direction, strength=0.0, prompt_len=n)
try:
    with_hook = greedy(model, tokens)
finally:
    handle.remove()
identical = torch.equal(baseline, with_hook)
print(
    f"\n2. {question!r}: no hook {model.tokenizer.decode(baseline[0, n:])!r}; strength 0 "
    f"{model.tokenizer.decode(with_hook[0, n:])!r} (hook fired {handle.calls} times; identical ids: {identical})"
)
steered = {}
for strength in (-4.0, 4.0):
    h = add_direction_hook(model, direction, strength=strength, prompt_len=n)
    try:
        steered[strength] = model.tokenizer.decode(greedy(model, tokens)[0, n:])
    finally:
        h.remove()
    print(f"   strength {strength:+.0f}: {steered[strength]!r}")

# --- 3 and 4. Read the direction and the random controls -----------------------------------------------------------
controls = random_directions(N_RANDOM, cfg.d_model, layer, seed=0)
units = torch.stack([direction.unit] + [c.unit for c in controls])  # [1 + N, d_model]
readings = {"sandbagging": [], "honest": []}
resid_norm = []
for pair in prompt_pairs():
    for kind in readings:
        _, cache = model.run_with_cache(
            chat_tokens(model, pair[kind]), names_filter=lambda name: name.endswith("resid_post")
        )
        stack = cache.stack_activation("resid_post")[:, 0, -1].cpu()  # [n_layers, d_model]
        readings[kind].append(stack @ units.T)  # [n_layers, 1 + N]
        resid_norm.append(stack[layer].norm().item())
        assert torch.allclose(readings[kind][-1][:, 0], project(cache, direction)[:, 0, -1].cpu(), atol=1e-4)
sandbag = torch.stack(readings["sandbagging"])  # [8, n_layers, 1 + N]
honest = torch.stack(readings["honest"])
gap = sandbag - honest
n_pairs = gap.shape[0]
ours = gap[:, :, 0]
wins = (ours > 0).sum(0)
random_gap = gap[:, :, 1:].mean(0)  # [n_layers, N]: each random direction's mean gap per layer
print(f"\n3. Reading at the final prompt position (means over {n_pairs} pairs)")
print(f"  {'layer':<6}{'sandbagging':>12}{'honest':>9}{'gap':>8}{'wins':>6}{'random |gap|':>14}")
rows = []
for i in range(cfg.n_layers):
    se = (ours[:, i].std() / n_pairs**0.5).item()
    r_mean, r_max = random_gap[i].abs().mean().item(), random_gap[i].abs().max().item()
    rows.append(
        [
            i,
            sandbag[:, i, 0].mean().item(),
            honest[:, i, 0].mean().item(),
            ours[:, i].mean().item(),
            se,
            int(wins[i]),
            r_mean,
            r_max,
        ]
    )
    mark = "  <- the direction's layer" if i == layer else ""
    print(f"  {i:<6}{rows[-1][1]:>12.3f}{rows[-1][2]:>9.3f}{rows[-1][3]:>+8.3f}{int(wins[i]):>6}{r_mean:>14.3f}{mark}")
gap_l = ours[:, layer].mean().item()
random_l = random_gap[layer].abs()
beats = int((random_l < abs(gap_l)).sum())
best_layer = int(ours.mean(0).argmax())
print(
    f"\n  layer {layer}: sandbagging {sandbag[:, layer, 0].mean():.3f}, honest {honest[:, layer, 0].mean():.3f}, gap"
    f" {gap_l:+.3f}, higher for sandbagging in {int(wins[layer])} of {n_pairs} pairs"
)
print(
    f"  random directions at layer {layer}: mean |gap| {random_l.mean():.3f}, largest {random_l.max():.3f}; the"
    f" sandbagging direction's gap is larger than {beats} of {N_RANDOM}"
)
print(
    f"  residual stream length at layer {layer}: ~{sum(resid_norm) / len(resid_norm):.0f}; the gap peaks at layer"
    f" {best_layer} ({ours[:, best_layer].mean():+.3f})"
)

with (RESULTS / "direction_projection.csv").open("w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
        [
            "layer",
            "sandbagging_mean",
            "honest_mean",
            "gap_mean",
            "gap_se",
            "wins",
            "random_gap_mean_abs",
            "random_gap_max_abs",
        ]
    )
    writer.writerows([[r[0], *(f"{v:.4f}" for v in r[1:5]), r[5], f"{r[6]:.4f}", f"{r[7]:.4f}"] for r in rows])
print("  wrote results/direction_projection.csv")
write_json(
    "direction_read.json",
    {
        "layer": layer,
        "strength0_identical": identical,
        "strength0_hook_calls": handle.calls,
        "steering_example": {
            "question": question,
            "unsteered": model.tokenizer.decode(baseline[0, n:]),
            **{f"strength_{s:+.0f}": t for s, t in steered.items()},
        },
        "gap_at_layer": gap_l,
        "wins_at_layer": int(wins[layer]),
        "n_pairs": n_pairs,
        "best_gap_layer": best_layer,
        "best_gap": ours[:, best_layer].mean().item(),
        "resid_norm_at_layer": sum(resid_norm) / len(resid_norm),
        "random_gaps_at_layer": [round(v, 4) for v in random_gap[layer].tolist()],
        "random_beaten": beats,
    },
)

print("\nSelf-checks:")
check(
    "strength 0: identical greedy tokens with and without the hook, and the hook fired", identical and handle.calls >= 1
)
check(f"at layer {layer} the sandbagging prompts read higher on average", gap_l > 0)
check(f"at layer {layer} the sandbagging prompt is higher in at least 6 of 8 pairs", int(wins[layer]) >= 6)
check(f"the gap is larger than every one of {N_RANDOM} random directions' gaps", beats == N_RANDOM)
