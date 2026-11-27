"""Write report-v2/hook.js, the data behind Version B's opening example, from results/ only.

    python report-v2/make_hook_data.py

For each of the 41 occupations in the released prompt file, the mean skew of each of the 42 joint groups over that
occupation's 7 templates, under four arms already in results/: CLIP (no edit), the SAE rebuild alone (alpha 0.6), the
DeBiasLens edit (13 selected latents, alpha 0.6) and the same latents removed with the SAE error kept (exploratory).
Also each occupation's mean joint excess skew under each arm. Group statistics only; no image is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES = EXP / "results"

ARMS = {  # key in the .npz -> (file, label)
    "clip": ("prompt_metrics_core.npz", "CLIP, no edit"),
    "recon@0.6": ("prompt_metrics_core.npz", "SAE rebuild only"),
    "dbl_all@0.6": ("prompt_metrics_core.npz", "DeBiasLens edit"),
    "ek_all@x1": ("prompt_metrics_errkeep.npz", "Same features, error kept"),
}


def main():
    P = json.loads((EXP / "data" / "prompts.json").read_text())
    sets = np.array([s for s, ps in P["sets"].items() for _ in ps])
    concepts = np.array([p["concept"] for _, ps in P["sets"].items() for p in ps])
    occ = sets == "occupations"
    names = list(dict.fromkeys(concepts[occ].tolist()))  # file order
    cells = json.loads((RES / "cells.json").read_text())
    out = {"cell_names": cells["cell_names"], "share": [round(x, 4) for x in cells["cell_share_val"]],
           "labels": {k: v[1] for k, v in ARMS.items()}, "occupations": {}}  # fmt: skip
    npz = {f: np.load(RES / f) for f in {v[0] for v in ARMS.values()}}
    for name in names:
        m = occ & (concepts == name)
        row = {"n_templates": int(m.sum())}
        for k, (f, _) in ARMS.items():
            z = npz[f]
            row[k] = {"cells": np.round(z[f"{k}::cells"][m].mean(0), 3).tolist(),
                      "joint_excess": round(float(z[f"{k}::joint_excess"][m].mean()), 3)}  # fmt: skip
        out["occupations"][name] = row
    (HERE / "hook.js").write_text("window.HOOK = " + json.dumps(out) + ";\n")
    n = out["occupations"]["nurse"]
    print("wrote", HERE / "hook.js", "|", len(names), "occupations")
    for k in ARMS:
        c = np.array(n[k]["cells"])
        print(f"nurse {k:12s} joint excess {n[k]['joint_excess']:.3f}  max {c.max():+.2f}  min {c.min():+.2f}  "
              f"mean|skew| {np.abs(c).mean():.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
