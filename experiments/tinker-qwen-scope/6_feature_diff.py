"""Step 6: which SAE features did the lock move, beyond what the clean control moved? (local, the pre-registered rule)

    python experiments/tinker-qwen-scope/6_feature_diff.py

For each held-out question and each adapter, the SAE encodes the residual at the last prompt token on the fine-tuned
model and on the base model; the difference, feature by feature, is that question's shift. Averaged over questions it
is the feature's shift; resampling the questions 1,000 times gives a 95% interval (a bootstrap).

The rule, written down before training and applied here by steerlab's `features.diff.apply_rule`:

1. Rank the lock's 65,536 features by the size of their mean shift without the password, on the EVEN-numbered
   questions, and keep the top 20 as candidates. Everything below uses only the ODD-numbered questions.
2. T is the largest mean shift of ANY feature on the clean control without the password.
3. A candidate counts as "changed by the lock" only if the edge of its interval nearest zero is beyond T (it moved more
   than anything the clean fine-tune moved), and
4. with the password in front, its shift flips sign or its interval covers zero (the lock acts only without it).

Candidates that miss get a reason. An empty list is a valid answer.

Added after training, before this script first ran (exploratory, reported separately): the same rule with the random
adapter of the lock's size (step 3) as the control instead of the clean one; and the size of every feature's shift for
all three adapters, to see how unusual the lock's largest shifts are.
"""

from __future__ import annotations

import sys
from importlib import import_module

import common as c
import numpy as np

TOP, BOOT, SEED = 20, 1000, 0
KEEP = 300
"""How many of each adapter's largest shifts are saved for the figure."""
CONDITIONS = {"plain": "none", "password": "password", "decoy": "decoy"}


def encode(sae, vectors: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        acts = sae.encode(torch.from_numpy(vectors).to(sae.device, torch.float32))
    return acts.float().cpu().numpy()


def plain_rule(text: str) -> str:
    """The library's rule text without its internal decision label."""
    return text.split(": ", 1)[1] if text.startswith("KTD") else text


def main() -> int:
    import torch
    from steerlab.features import diff as diff_lib

    load_sae = import_module("5_sae_check").load_sae
    work = c.work_dir()
    saved = np.load(work / "residuals.npz")
    layers = [int(x) for x in saved["layers"]]
    chosen = c.read_json(c.RESULTS / "sae_check.json")["chosen"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ok = []
    for k, layer in enumerate(layers):
        sae = load_sae(layer).to(device)
        acts = {(s, cond): encode(sae, saved[f"{s}_{cond}"][:, k, :]) for s in c.SIDES for cond in c.CONDITIONS}
        shifts = {
            org: {name: acts[(org, cond)] - acts[("base", cond)] for name, cond in CONDITIONS.items()}
            for org in ("lock", "clean", "random")
        }
        np.savez_compressed(
            work / f"shifts_L{layer}.npz",
            **{f"{org}_{name}": arr.astype(np.float16) for org, d in shifts.items() for name, arr in d.items()},
            **{f"acts_base_{cond}": acts[("base", cond)].astype(np.float16) for cond in c.CONDITIONS},
        )
        pick = {o: {n: shifts[o][n] for n in ("plain", "password")} for o in shifts}
        report = diff_lib.apply_rule(
            pick["lock"], pick["clean"], layer=layer, release=c.SAE_RELEASE, top=TOP, n_boot=BOOT, seed=SEED
        )
        random_report = diff_lib.apply_rule(
            pick["lock"], pick["random"], layer=layer, release=c.SAE_RELEASE, top=TOP, n_boot=BOOT, seed=SEED
        )
        test = np.arange(1, shifts["lock"]["plain"].shape[0], 2)
        distribution = {}
        for org in shifts:
            sizes = np.abs(shifts[org]["plain"][test].astype(np.float64).mean(axis=0))
            order = np.argsort(-sizes)[:KEEP]
            distribution[org] = {
                "features": [int(f) for f in order],
                "abs_mean_shift": [float(sizes[f]) for f in order],
                "above_clean_T": int((sizes > report.threshold).sum()),
            }
        data = report.to_dict()
        data["rule"] = plain_rule(data["rule"])
        data["headline_layer"] = layer in chosen
        data["random_control"] = {
            "threshold": random_report.threshold,
            "threshold_feature": random_report.threshold_feature,
            **{diff_lib.KEYS[v]: [r.feature for r in random_report.group(v)] for v in diff_lib.VERDICTS},
        }
        data["distribution"] = distribution
        c.write_json(c.RESULTS / f"feature_diff_L{layer}.json", data)
        print(f"\nlayer {layer}{' (headline)' if layer in chosen else ''}")
        for line in diff_lib.format_report(report, show=5):
            print(plain_rule(line))
        print(
            f"random-adapter control: T = {random_report.threshold:.4g}; changed by the lock under it: "
            f"{len(random_report.changed_by_lock)}"
        )
        for org, d in distribution.items():
            print(f"  {org:>6}: largest |shift| {d['abs_mean_shift'][0]:.4g}; above the clean T: {d['above_clean_T']}")
        beats = all(r.lock["plain"].nearest_zero > report.threshold for r in report.changed_by_lock)
        ok.append(c.check(f"layer {layer}: every listed feature beats the clean control's largest shift", beats))
        del sae, acts, shifts
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
