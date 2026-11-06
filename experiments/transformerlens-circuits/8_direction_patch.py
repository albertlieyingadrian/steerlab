"""Step 8: patch ONLY the sandbagging direction. Does giving the sandbagging run the honest run's amount of it, at one
layer and position, bring the right answer back? With the whole-vector patch as the ceiling and random directions as
the floor.

    python experiments/transformerlens-circuits/8_direction_patch.py   # ~10 min on a laptop CPU; reruns reuse the grids

TransformerLens's get_act_patch_resid_pre copies the WHOLE residual vector at one (layer, position) from the clean run
into the corrupted run: it says WHERE the decisive information is. lens_bridge.get_act_patch_resid_direction is the
same loop (generic_activation_patch, unchanged) with a setter that swaps only the component along one unit vector:

    swap:   h  ->  h - (h·u) u + (h_clean·u) u        at the indexed layer and position

  A. the whole vector, TransformerLens as shipped, on positions 29-55
  B. only the sandbagging direction, on every cell (the 29 identical prefix positions are a sanity check)
  C. the control: 5 random unit directions, swapped the same way on positions 29-55
Thresholds fixed in advance for B's best cell: below 0.1 flat, below 0.5 partial, otherwise strong.
Writes results/direction_patch_whole.csv, results/direction_patch_direction.csv, results/direction_patch.json.
"""

import time

import torch
from common import boot, check, work_dir, write_grid_csv, write_json
from direction_tools import (
    MODEL_NAME,
    PATCH_PAIR,
    chat_tokens,
    first_digit_logit_diff,
    greedy,
    load_direction,
    make_recovery_metric,
    random_directions,
)
from lens_bridge import get_act_patch_resid_direction, project
from transformer_lens import patching
from transformer_lens.patching import make_df_from_ranges

N_RANDOM = 5
GRID_FILE = work_dir() / "patch_grids.pt"
direction = load_direction()
model, device = boot(MODEL_NAME)
cfg = model.cfg
unit = direction.unit.to(device)

# --- 1. The pair and the metric ------------------------------------------------------------------------------------
clean_tokens = chat_tokens(model, PATCH_PAIR["honest"])
corrupted_tokens = chat_tokens(model, PATCH_PAIR["sandbagging"])
same_length = clean_tokens.shape == corrupted_tokens.shape
differing = (clean_tokens != corrupted_tokens).nonzero()[:, 1].tolist() if same_length else []
str_tokens = model.to_str_tokens(clean_tokens[0])
n = clean_tokens.shape[1]
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
clean_ld = first_digit_logit_diff(model, clean_logits).item()
corrupted_ld = first_digit_logit_diff(model, corrupted_logits).item()
clean_answer = model.tokenizer.decode(greedy(model, clean_tokens, n=3)[0, n:]).strip()
corrupted_answer = model.tokenizer.decode(greedy(model, corrupted_tokens, n=3)[0, n:]).strip()
metric = make_recovery_metric(model, clean_ld, corrupted_ld)
reading = {
    kind: project(c, direction)[:, 0, -1].cpu()
    for kind, c in (("honest", clean_cache), ("sandbagging", corrupted_cache))
}
print(f"1. {n} tokens each, differing at {differing}: {[str_tokens[p] for p in differing]}")
print(f"   honest:      logit('5') - logit('6') = {clean_ld:+.3f}, greedy answer {clean_answer!r}")
print(f"   sandbagging: {corrupted_ld:+.3f}, greedy answer {corrupted_answer!r}")
print(
    f"   the direction at the last position, layer {direction.layer}: honest {reading['honest'][direction.layer]:+.3f},"
    f" sandbagging {reading['sandbagging'][direction.layer]:+.3f}; layer 19: honest {reading['honest'][19]:+.3f},"
    f" sandbagging {reading['sandbagging'][19]:+.3f}"
)

# --- 2. The sweeps -------------------------------------------------------------------------------------------------
first_pos = min(differing) - 1  # keep one identical position as a sanity zero
positions = list(range(first_pos, n))
index_df = make_df_from_ranges([cfg.n_layers, n], ["layer", "pos"])
index_df = index_df[index_df["pos"] >= first_pos].reset_index(drop=True)
controls = random_directions(N_RANDOM, cfg.d_model, direction.layer, seed=2)
key = {"pair": PATCH_PAIR, "positions": positions, "n_random": N_RANDOM}
saved = torch.load(GRID_FILE) if GRID_FILE.exists() else None
if saved is not None and saved.get("key") == key:
    print(
        f"\n2. reusing the grids saved by a previous run ({GRID_FILE.name} in the work folder; delete it to recompute)"
    )
    grids = saved["grids"]
else:
    grids, timing = {}, {}
    started = time.time()
    result = get_act_patch_resid_direction(
        model, corrupted_tokens, clean_cache, metric, unit=unit, mode="swap", baseline=True
    )
    grids["direction"] = result.values.detach().cpu()
    grids["baseline"] = float(result.baseline.reshape(-1)[0])
    grids["axes"] = list(result.axis_names)
    timing["direction"] = time.time() - started
    started = time.time()
    flat = patching.generic_activation_patch(
        model,
        corrupted_tokens,
        clean_cache,
        metric,
        patch_setter=patching.layer_pos_patch_setter,
        activation_name="resid_pre",
        index_df=index_df,
    )
    grids["whole"] = flat.detach().cpu().reshape(cfg.n_layers, len(positions))
    timing["whole"] = time.time() - started
    started = time.time()
    grids["random"] = [
        get_act_patch_resid_direction(
            model, corrupted_tokens, clean_cache, metric, unit=c.unit.to(device), mode="swap", index_df=index_df
        )
        .detach()
        .cpu()
        .reshape(cfg.n_layers, len(positions))
        for c in controls
    ]
    timing["random"] = time.time() - started
    grids["timing"] = timing
    torch.save({"key": key, "grids": grids}, GRID_FILE)
    print(
        f"\n2. sweeps: B {cfg.n_layers * n} cells in {timing['direction']:.0f} s; A {len(index_df)} cells in "
        f"{timing['whole']:.0f} s; C {N_RANDOM} x {len(index_df)} cells in {timing['random']:.0f} s"
    )

whole, direction_full = grids["whole"], grids["direction"]
dir_grid = direction_full[:, first_pos:]
prefix_max = direction_full[:, :first_pos].abs().max().item()
diff_col, last_col = positions.index(differing[0]), len(positions) - 1


def table(grid, title):
    print(f"\n  {title}")
    print(f"  {'layer':<6}" + "".join(f"{p:>6}" for p in positions))
    for layer in range(cfg.n_layers):
        print(f"  {layer:<6}" + "".join(f"{v:>6.2f}" for v in grid[layer]))


print("\n3. Recovery (0 = still sandbagging, 1 = fully honest); columns are positions")
table(whole, "A. the whole residual vector")
table(dir_grid, "B. only the sandbagging direction (swap)")


def cell(grid, largest=True):
    """(layer, position, value) of the grid's largest (or smallest) cell."""
    layer, col = divmod(int(grid.argmax() if largest else grid.argmin()), grid.shape[1])
    return layer, positions[col], grid[layer, col].item()


best = cell(dir_grid)
worst = cell(dir_grid, largest=False)
best_whole = cell(whole)
final_17_20 = [dir_grid[layer, last_col].item() for layer in range(17, 21)]
whole_17_20 = [whole[layer, last_col].item() for layer in range(17, 21)]
at_build = dir_grid[direction.layer + 1, last_col].item()
random_summary = []
for c, grid in zip(controls, grids["random"], strict=True):
    random_summary.append(
        {
            "name": c.name,
            "best": grid.max().item(),
            "worst": grid.min().item(),
            "at_best_cell": grid[best[0], positions.index(best[1])].item(),
            "final_17_20": [grid[layer, last_col].item() for layer in range(17, 21)],
        }
    )
random_best = max(r["best"] for r in random_summary)
random_worst = min(r["worst"] for r in random_summary)
reading_label = "flat" if best[2] < 0.1 else "partial" if best[2] < 0.5 else "strong"
print(
    f"\n  A: layer 0 at the differing token {whole[0, diff_col]:.2f}; best cell layer {best_whole[0]}, position "
    f"{best_whole[1]}: {best_whole[2]:.2f}"
)
print(
    f"  B: best cell layer {best[0]}, position {best[1]} ({str_tokens[best[1]]!r}): {best[2]:+.3f} (whole vector there "
    f"{whole[best[0], positions.index(best[1])]:.2f}); worst layer {worst[0]}, position {worst[1]}: {worst[2]:+.3f}"
)
print(
    "  B: final position, layers 17-20: "
    + ", ".join(f"{v:+.2f}" for v in final_17_20)
    + "  (whole vector: "
    + ", ".join(f"{v:.2f}" for v in whole_17_20)
    + f"); block 15's resid_pre, final position: {at_build:+.3f}"
)
print(
    f"  B: largest |recovery| over the {first_pos} identical prefix positions: {prefix_max:.4f}; baseline "
    f"{grids['baseline']:+.4f}"
)
print(f"  C: {N_RANDOM} random directions: best cell anywhere {random_best:+.3f}, worst {random_worst:+.3f}")
for r in random_summary:
    print(
        f"     {r['name']}: best {r['best']:+.3f}, worst {r['worst']:+.3f}, at B's best cell {r['at_best_cell']:+.3f}"
    )
print(f"\n  Reading (thresholds fixed in advance: < 0.1 flat, < 0.5 partial, else strong): {reading_label.upper()}")

labels = [f"{p}:{str_tokens[p]!r}" for p in positions]
layers = [f"L{i}" for i in range(cfg.n_layers)]
write_grid_csv("direction_patch_whole.csv", whole, layers, labels, corner="layer")
write_grid_csv(
    "direction_patch_direction.csv", direction_full, layers, [f"{p}:{t!r}" for p, t in enumerate(str_tokens)], "layer"
)
random_rows = [f"{c.name.replace(' ', '')}:L{i}" for c in controls for i in range(cfg.n_layers)]
write_grid_csv("direction_patch_random.csv", torch.cat(grids["random"]), random_rows, labels, corner="direction:layer")
write_json(
    "direction_patch.json",
    {
        "pair": PATCH_PAIR,
        "n_tokens": n,
        "differing_position": differing,
        "clean_ld": clean_ld,
        "corrupted_ld": corrupted_ld,
        "clean_answer": clean_answer,
        "corrupted_answer": corrupted_answer,
        "reading_last_pos": {k: [round(x, 4) for x in v.tolist()] for k, v in reading.items()},
        "first_position_swept_by_A": first_pos,
        "timing_seconds": {k: round(v, 1) for k, v in grids.get("timing", {}).items()},
        "whole_layer0_differing": whole[0, diff_col].item(),
        "whole_best": {"layer": best_whole[0], "pos": best_whole[1], "value": best_whole[2]},
        "direction_best": {
            "layer": best[0],
            "pos": best[1],
            "value": best[2],
            "whole_same_cell": whole[best[0], positions.index(best[1])].item(),
        },
        "direction_worst": {"layer": worst[0], "pos": worst[1], "value": worst[2]},
        "direction_final_layers_17_20": final_17_20,
        "whole_final_layers_17_20": whole_17_20,
        "direction_block15_final": at_build,
        "direction_prefix_max_abs": prefix_max,
        "baseline": grids["baseline"],
        "random": random_summary,
        "random_best": random_best,
        "random_worst": random_worst,
        "reading": reading_label,
    },
)

print("\nSelf-checks:")
check("the two prompts have the same length and differ in exactly one token", same_length and len(differing) == 1)
check("the behaviour flips: honest prefers '5' (56), sandbagging '6' (64)", clean_ld > 0 > corrupted_ld)
check(
    "sweep B returns a [layer, pos] PatchingResult whose baseline is 0",
    grids["axes"] == ["layer", "pos"] and abs(grids["baseline"]) < 1e-4,
)
check(
    "no sweep moves the metric at the identical prefix positions (< 0.05)",
    prefix_max < 0.05 and whole[:, 0].abs().max() < 0.05,
)
check("whole-vector patch at layer 0 of the differing token recovers everything (> 0.95)", whole[0, diff_col] > 0.95)
check("the direction-only sweep recovers at least 0.1 somewhere", best[2] >= 0.1)
check("its best cell beats every random direction's best cell", best[2] > random_best)
