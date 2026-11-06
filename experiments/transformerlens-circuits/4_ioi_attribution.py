"""Step 4: direct logit attribution per head, the published-head checklist, and "patch only these heads".

    python experiments/transformerlens-circuits/4_ioi_attribution.py   # ~20 s; reads results/ioi_head_patch.csv

Direct logit attribution (DLA) asks a different question from patching. Patching: "does restoring this head fix the
answer?" (causal, includes everything downstream). DLA: "how much does this head's own output, read straight off by
the unembedding, push logit(IO) above logit(S)?" (direct, linear). With compatibility mode on, the final logit
difference is a sum of one share per component: cache.stack_head_results gives each head's vector at the last
position, cache.logit_attrs dots it with W_U[IO] - W_U[S].

Then the test of whether the published heads ARE the circuit: copy the clean output of a chosen set of heads into the
corrupted run, leave everything else corrupted, and measure. Controls: the full published list, our own top nine by
head patching, and nine random heads (five draws, fixed seed).
Writes results/ioi_dla.csv, results/ioi_checklist.csv, results/ioi_head_sets.json.
"""

import csv

import torch
from common import RESULTS, check, read_grid_csv, write_grid_csv, write_json
from ioi_tools import CORE_NINE, FUZZY, PUBLISHED, Setup, head_label, logit_diff

setup = Setup()
setup.describe()
model, cache = setup.model, setup.clean_cache
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
N = n_layers * n_heads
io_tokens, s_tokens = setup.answer_tokens[:, 0], setup.answer_tokens[:, 1]

# --- 1. Direct logit attribution -----------------------------------------------------------------------------------
per_head, _ = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
dla = cache.logit_attrs(per_head, tokens=io_tokens, incorrect_tokens=s_tokens, pos_slice=-1)  # (144, 8)
dla_grid = dla.mean(1).reshape(n_layers, n_heads).cpu()
components, component_labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
component_dla = cache.logit_attrs(components, tokens=io_tokens, incorrect_tokens=s_tokens, pos_slice=-1)
bias_term = (model.b_U[io_tokens] - model.b_U[s_tokens]).cpu()  # logit_attrs leaves the unembedding bias out
actual = logit_diff(setup.clean_logits, setup.answer_tokens, per_prompt=True).cpu()
reconstructed = component_dla.sum(0).cpu() + bias_term
sums_match = torch.allclose(reconstructed, actual, atol=1e-3)
by_component = {label: v.item() for label, v in zip(component_labels, component_dla.mean(1), strict=True)}
print(
    f"\n1. DLA: components sum to {component_dla.mean(1).sum().item():+.3f}, plus the unembedding bias "
    f"{bias_term.mean().item():+.3f} = {reconstructed.mean().item():+.3f}; actual {setup.clean_ld:+.3f}; exact: "
    f"{sums_match}"
)
layer9 = by_component["9_attn_out"]
other_max = max(v for k, v in by_component.items() if k != "9_attn_out")
print(f"   layer 9's attention output alone: {layer9:+.2f}; no other component above {other_max:+.2f}")
dla_order = sorted(((dla_grid[ly, h].item(), (ly, h)) for ly in range(n_layers) for h in range(n_heads)), reverse=True)
dla_rank = {head: i + 1 for i, (_, head) in enumerate(dla_order)}
print("   top five: " + ", ".join(f"{head_label(*h)} {v:+.2f}" for v, h in dla_order[:5]))
print("   most negative: " + ", ".join(f"{head_label(*h)} {v:+.2f}" for v, h in dla_order[-2:]))

# --- 2. The checklist ----------------------------------------------------------------------------------------------
# Each class is judged by the tool the paper used to find it: name movers by DLA (top 5), negative name movers by DLA
# (bottom 3), every other class by head patching (top 15; the paper used path patching, which is not run here).
_, _, patch_rows = read_grid_csv("ioi_head_patch.csv")
head_patch = torch.tensor(patch_rows)
patch_order = sorted(
    ((head_patch[ly, h].item(), (ly, h)) for ly in range(n_layers) for h in range(n_heads)), reverse=True
)
patch_rank = {head: i + 1 for i, (_, head) in enumerate(patch_order)}
EXPECTED_BY = {"name mover": "DLA top 5", "negative name mover": "DLA bottom 3"}


def found(head, cls):
    if cls == "name mover":
        return dla_rank[head] <= 5
    if cls == "negative name mover":
        return dla_rank[head] >= N - 2
    return patch_rank[head] <= 15


rows, per_class = [], {}
print(f"\n2. {'class':<22}{'head':<7}{'DLA':>7}{'rank':>6}{'patch':>8}{'rank':>6}  expected by        found?")
for cls, heads in PUBLISHED.items():
    per_class[cls] = 0
    for head in heads:
        hit = found(head, cls)
        per_class[cls] += hit
        expected = EXPECTED_BY.get(cls, "patching top 15")
        rows.append(
            [
                cls,
                head_label(*head),
                head in FUZZY,
                f"{dla_grid[head].item():.4f}",
                dla_rank[head],
                f"{head_patch[head].item():.4f}",
                patch_rank[head],
                expected,
                "yes" if hit else "no",
            ]
        )
        print(
            f"   {cls:<22}{head_label(*head):<7}{dla_grid[head].item():>+7.2f}{dla_rank[head]:>6}"
            f"{head_patch[head].item():>+8.2f}{patch_rank[head]:>6}  {expected:<18} {'yes' if hit else 'NO'}"
        )
n_found = sum(per_class.values())
print(
    f"   {n_found} of 18 published heads stand out in the tool expected to see them: "
    + ", ".join(f"{cls} {per_class[cls]}/{len(PUBLISHED[cls])}" for cls in PUBLISHED)
)
with (RESULTS / "ioi_checklist.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["class", "head", "fuzzy", "dla", "dla_rank", "patch", "patch_rank", "expected_by", "found"])
    writer.writerows(rows)
print("  wrote results/ioi_checklist.csv")


# --- 3. Patch only a set of heads ----------------------------------------------------------------------------------
def patch_heads(heads):
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def make_hook(layer, index):
        def hook_fn(z, hook):
            z[:, :, index, :] = cache[f"blocks.{layer}.attn.hook_z"][:, :, index, :]
            return z

        return hook_fn

    hooks = [(f"blocks.{layer}.attn.hook_z", make_hook(layer, index)) for layer, index in by_layer.items()]
    logits = model.run_with_hooks(setup.corrupted_tokens, fwd_hooks=hooks)
    return setup.metric(logits).item(), logit_diff(logits, setup.answer_tokens).item()


all_published = [head for heads in PUBLISHED.values() for head in heads]
our_top9 = [head for _, head in patch_order[:9]]
generator = torch.Generator().manual_seed(0)
random_sets = [
    [(int(i) // n_heads, int(i) % n_heads) for i in torch.randperm(N, generator=generator)[:9]] for _ in range(5)
]
sets = {
    "nine core published heads": CORE_NINE,
    "all 18 published heads": all_published,
    "our top nine by head patching": our_top9,
}
results = {}
print("\n3. Patch only a set of heads (clean hook_z copied in at every position)")
for name, heads in sets.items():
    recovery, ld = patch_heads(heads)
    results[name] = {"heads": [head_label(*h) for h in heads], "recovery": recovery, "logit_diff": ld}
    print(f"   {name:<32} recovery {recovery:+.3f}  (logit diff {ld:+.2f})")
random_results = []
for heads in random_sets:
    recovery, ld = patch_heads(heads)
    random_results.append({"heads": [head_label(*h) for h in heads], "recovery": recovery, "logit_diff": ld})
random_recoveries = [r["recovery"] for r in random_results]
random_mean = sum(random_recoveries) / len(random_recoveries)
print(
    f"   {'nine random heads (5 draws)':<32} recovery {min(random_recoveries):+.3f} .. {max(random_recoveries):+.3f}, "
    f"mean {random_mean:+.3f}"
)

write_grid_csv("ioi_dla.csv", dla_grid, [f"L{i}" for i in range(n_layers)], [f"H{h}" for h in range(n_heads)], "layer")
write_json(
    "ioi_head_sets.json",
    {
        "clean_ld": setup.clean_ld,
        "dla_component_sum": component_dla.mean(1).sum().item(),
        "dla_bias_term": bias_term.mean().item(),
        "dla_layer9_attn": layer9,
        "dla_other_component_max": other_max,
        "dla_top5": [[head_label(*h), round(v, 4)] for v, h in dla_order[:5]],
        "dla_bottom2": [[head_label(*h), round(v, 4)] for v, h in dla_order[-2:]],
        "found_per_class": per_class,
        "found_total": n_found,
        "sets": results,
        "random": random_results,
        "random_mean": random_mean,
    },
)

core = results["nine core published heads"]["recovery"]
print("\nSelf-checks:")
check("per-component DLA plus the unembedding bias adds up to the real logit difference (1e-3)", sums_match)
check(
    "the published name movers L9H9 and L9H6 are the top two heads by DLA",
    {h for _, h in dla_order[:2]} == {(9, 9), (9, 6)},
)
check("the third published name mover L10H0 is in the DLA top five", dla_rank[(10, 0)] <= 5)
check(
    "the two published negative name movers are the two most negative by DLA",
    {h for _, h in dla_order[-2:]} == set(PUBLISHED["negative name mover"]),
)
check("patching only the nine core published heads recovers at least 60%", core >= 0.60)
check(
    "the full published list beats every random draw by more than 0.2",
    results["all 18 published heads"]["recovery"] > max(random_recoveries) + 0.2,
)
