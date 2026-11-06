"""Step 7: which heads and MLPs write the sandbagging direction? Direction attribution, with a random-direction control.

    python experiments/transformerlens-circuits/7_direction_attribution.py   # ~1 min with the model cached

The residual stream is a running sum of what every head and MLP wrote, and a dot product is linear, so the direction's
reading at layer 14 is the sum of each component's own dot product with the unit vector: direct logit attribution
with the direction in place of an unembedding column (lens_bridge.attribute). 241 components: the embedding, 15 x 14
heads, 15 attention biases and 15 MLPs.

The verdict uses a rule fixed before the numbers were seen: the top 3 components holding more than half of the gap's
total size = dominated; more than 20 components needed to reach half of it = diffuse; otherwise in between. The
control runs the same split on 5 random unit directions, to see whether "diffuse" is just what any direction looks like.
Writes results/direction_attribution.csv and results/direction_attribution.json.
"""

import csv

import torch
from common import RESULTS, boot, check, write_json
from direction_tools import MODEL_NAME, chat_tokens, load_direction, prompt_pairs, random_directions
from lens_bridge import attribute, project

N_RANDOM = 5
direction = load_direction()
model, device = boot(MODEL_NAME)
layer = direction.layer
controls = random_directions(N_RANDOM, model.cfg.d_model, layer, seed=1)
names = [direction, *controls]

tables = {kind: [[] for _ in names] for kind in ("sandbagging", "honest")}
worst_unexplained, worst_vs_project = 0.0, 0.0
for pair in prompt_pairs():
    for kind in tables:
        _, cache = model.run_with_cache(chat_tokens(model, pair[kind]))
        for i, d in enumerate(names):
            table = attribute(cache, d, pos=-1)
            tables[kind][i].append(table.values[:, 0].cpu())
            if i == 0:
                worst_unexplained = max(worst_unexplained, abs(table.values[:, 0].sum() - table.total[0]).item())
                reading = project(cache, d)[layer, 0, -1].item()
                worst_vs_project = max(worst_vs_project, abs(table.total[0].item() - reading))
labels = table.labels
n_components = len(labels)
print(
    f"1. {n_components} components per prompt; max |sum of components - reading| {worst_unexplained:.1e}; max "
    f"|reading -"
    f" project()| {worst_vs_project:.1e}"
)


def concentration(values):
    """(top-3 share of total |size|, components needed to reach half of it)."""
    order = values.abs().argsort(descending=True)
    total = values.abs().sum().item()
    top3 = values[order[:3]].abs().sum().item() / total
    cumulative = values[order].abs().cumsum(0) / total
    return top3, int((cumulative < 0.5).sum().item()) + 1


def verdict(top3, needed):
    if top3 > 0.5:
        return "dominated"
    if needed > 20:
        return "diffuse"
    return "in between"


sandbag = torch.stack(tables["sandbagging"][0]).mean(0)
honest = torch.stack(tables["honest"][0]).mean(0)
gap = sandbag - honest
top3, needed = concentration(gap)
order = gap.abs().argsort(descending=True)
same_sign = sum(1 for i in order[:5] if (gap[i] > 0) == (gap.sum() > 0))
print(f"\n2. The gap (sandbagging - honest), summing to {gap.sum().item():+.3f}; largest 12 by size:")
for rank, i in enumerate(order[:12], 1):
    print(f"   {rank:<4}{labels[i]:<14}{gap[i].item():>+8.3f}")
attn = torch.zeros(layer + 1)
mlp = torch.zeros(layer + 1)
for i, label in enumerate(labels):
    if label.startswith("L"):
        attn[int(label[1:].split("H")[0])] += gap[i]
    elif label.endswith("_attn_bias"):
        attn[int(label.split("_")[0])] += gap[i]
    elif label.endswith("_mlp_out"):
        mlp[int(label.split("_")[0])] += gap[i]
embed = gap[labels.index("embed")].item()
print(f"   summed per layer: attention {attn.sum().item():+.3f}, MLPs {mlp.sum().item():+.3f}, embedding {embed:+.3f}")
print("   " + ", ".join(f"L{i} attn {attn[i]:+.2f}" for i in range(9, layer + 1)))
result = verdict(top3, needed)
print(
    f"\n3. Verdict (rule fixed in advance): top 3 hold {top3:.0%} of the total size; {needed} of {n_components} "
    f"components"
    f" reach half -> {result.upper()}"
)

random_rows = []
for i, d in enumerate(controls, start=1):
    r_gap = torch.stack(tables["sandbagging"][i]).mean(0) - torch.stack(tables["honest"][i]).mean(0)
    r_top3, r_needed = concentration(r_gap)
    random_rows.append(
        {
            "name": d.name,
            "gap": r_gap.sum().item(),
            "top3_share": r_top3,
            "needed_for_half": r_needed,
            "verdict": verdict(r_top3, r_needed),
        }
    )
    print(
        f"   control {d.name}: gap {r_gap.sum().item():+.3f}, top 3 {r_top3:.0%}, {r_needed} for half -> "
        f"{random_rows[-1]['verdict']}"
    )

with (RESULTS / "direction_attribution.csv").open("w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["component", "sandbagging", "honest", "gap"])
    for i, label in enumerate(labels):
        writer.writerow([label, f"{sandbag[i].item():.5f}", f"{honest[i].item():.5f}", f"{gap[i].item():.5f}"])
print("  wrote results/direction_attribution.csv")
write_json(
    "direction_attribution.json",
    {
        "n_components": n_components,
        "gap_total": gap.sum().item(),
        "max_unexplained": worst_unexplained,
        "top3_share": top3,
        "needed_for_half": needed,
        "verdict": result,
        "top5_same_sign_as_gap": same_sign,
        "attention_total": attn.sum().item(),
        "mlp_total": mlp.sum().item(),
        "embed": embed,
        "attention_by_layer": [round(v, 4) for v in attn.tolist()],
        "mlp_by_layer": [round(v, 4) for v in mlp.tolist()],
        "random": random_rows,
    },
)

print("\nSelf-checks:")
check(f"one row per component: embed + 15 x (14 heads + bias + MLP) = {1 + 15 * 16}", n_components == 1 + 15 * 16)
check("the components sum to the reading on every prompt (< 1e-3)", worst_unexplained < 1e-3)
check("the reading equals project() at the same layer and position (< 1e-4)", worst_vs_project < 1e-4)
check("the gap is positive", gap.sum().item() > 0)
check("as first hoped: the five largest gap components mostly push the same way as the gap (>= 3 of 5)", same_sign >= 3)
