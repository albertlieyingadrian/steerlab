"""
Step 8: the common mistakes, each with what it cost in this run. Reads results only (runs in a second).

    mistake                         where the number comes from
    the attention sinks left in     02_collect (their norm) and 03c (an SAE trained with them)
    no normalisation                03c
    learning rate 10x off           03c
    no AuxK (dead latents)          03c, and SAELens's dead latents in 04
    too few tokens                  03c, and the training curve in 03a
    the wrong hook point            03c (the main SAE read at block 10 or 12 instead of 11)
    the decoder bias forgotten      03c (the main SAE decoded without b_dec)

    .venv/bin/python experiments/sae-101/08_mistakes.py --size full
"""

import argparse

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
args = ap.parse_args()
res = s.RESULTS / args.size
sw, col, tr = (
    s.read_json(res / "sweep.json"),
    s.read_json(res / "collect.json"),
    s.read_json(res / "train_handwritten.json"),
)
if sw is None:
    raise SystemExit("Run 03c_sweep.py first.")
runs = {r["name"]: r for r in sw["runs"]}
base = runs["TopK k=32"]

print(
    f"1. Mistakes when training (small SAEs: {sw['d_sae']:,} latents, k = 32, {sw['train_tokens'] / 1e6:.0f}M tokens x "
    f"{sw['epochs']} passes; held-out FVU and the share of latents that never fired on {sw['eval_tokens']:,} tokens)"
)
print(f"   {'run':<38} {'FVU':>7} {'vs right':>9} {'never fired':>12}")
print(f"   {'the right recipe (TopK k=32)':<38} {base['fvu']:>7.4f} {'':>9} {base['dead_share']:>12.1%}")
rows = []
for r in sw["runs"]:
    if r["group"] == "mistake":
        rows.append(r)
        print(f"   {r['name']:<38} {r['fvu']:>7.4f} {r['fvu'] - base['fvu']:>+9.4f} {r['dead_share']:>12.1%}")

print("\n2. Mistakes when using the main SAE (held-out FVU)")
right = sw["usage"].get("right layer (block 11 output)")
for name, v in sw["usage"].items():
    extra = f"  ({v / right:.1f}x the error)" if right and not name.startswith("right") else ""
    print(f"   {name:<46} {v:.4f}{extra}")

log = tr["log"]
at_10m = min(log, key=lambda r: abs(r["tokens"] - 10e6))
print(
    f"\n3. Too few tokens, from the main run's curve: train FVU {at_10m['train_fvu']:.3f} at {at_10m['tokens'] / 1e6:.0f}M "
    f"tokens, {log[-1]['train_fvu']:.3f} at {log[-1]['tokens'] / 1e6:.0f}M"
)
print(
    f"   The attention sinks: position 0's activation is {col['pos0_ratio']:.0f}x the median text token; in "
    f"{col['sink_windows_checked']} windows, {col['sinks_pos1']} more sinks sat at position 1 and {col['sinks_pos2']} at 2."
)

s.save_json(
    res / "mistakes.json",
    {"base": base, "training": rows, "usage": sw["usage"], "curve": {"at_10M": at_10m, "end": log[-1]}},
)
print("\nSelf-checks:")
s.check(
    "every training mistake is worse than the right recipe on FVU or dead latents",
    all(r["fvu"] > base["fvu"] or r["dead_share"] > base["dead_share"] for r in rows),
)
