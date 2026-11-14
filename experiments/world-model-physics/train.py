"""
Pre-train one world model from scratch: Part 18's tiny GPT (0.8M parameters), next-frame objective, 40 epochs.

    .venv/bin/python experiments/world-model-physics/train.py --world base --seed 2610
    .venv/bin/python experiments/world-model-physics/train.py --world const --seed 2620

base  = Part 18's elastic world (experiments A and C); const = gravity + drag varied per episode (experiment B).
Writes data/models/<world>_s<seed>_pre.pt and results/train_<world>_s<seed>.json (curve, wall time, val losses).
"""

from __future__ import annotations

import argparse

import numpy as np

import common as C
import worldmodel as wm

ap = argparse.ArgumentParser()
ap.add_argument("--world", choices=("base", "const"), required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=C.EPOCHS)
ap.add_argument("--tag", default="pre")
args = ap.parse_args()

device = C.pick_device()
split = C.base_split if args.world == "base" else C.const_split
train, val = split("train"), split("val")
model = wm.build(C.CONFIG, seed=args.seed)
print(
    f"{args.world} seed {args.seed}: {model.n_params():,} parameters, {len(train['positions']):,} episodes, "
    f"{args.epochs} epochs on {device}",
    flush=True,
)
energy = train.get("energy", np.zeros(len(train["positions"])))
curve = wm.train(
    model,
    "next",
    train["positions"],
    energy,
    val_pos=val["positions"],
    epochs=args.epochs,
    seed=args.seed,
    device=device,
    log_every=5,
)
wm.save_checkpoint(model, C.model_path(args.world, args.seed, args.tag), {"world": args.world, "seed": args.seed})
suffix = "" if args.tag == "pre" else f"_{args.tag}"
C.save_json(
    f"train_{args.world}_s{args.seed}{suffix}.json",
    {"world": args.world, "seed": args.seed, "epochs": args.epochs, "parameters": model.n_params(), **curve},
)
print(f"saved {C.model_path(args.world, args.seed, args.tag).name} ({curve['wall_seconds']:.0f} s)")
