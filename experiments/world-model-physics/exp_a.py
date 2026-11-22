"""
Experiment A: present versus used, on three freshly pre-trained base-world models.

For each seed: fit velocity, energy and collision-imminent probes at every site (presence: test R^2 / AUROC against
an input baseline), then intervene (use): add velocity directions, add the energy direction, subtract the
collision-imminent direction, and run the base project's donor interchange, each against random directions of
the same norm.

    .venv/bin/python experiments/world-model-physics/exp_a.py            # all three seeds
    .venv/bin/python experiments/world-model-physics/exp_a.py --seeds 2610 --n-random 10 --tag _smoke

Writes results/exp_a_s<seed>.json and results/exp_a_summary.json.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import common as C
import inside as I
import worldmodel as wm


def one_step(model, data, device):
    kinds = wm.frame_kinds(data["collides_next"])
    err = wm.step_mse(wm.predict(model, data["positions"], device), data["positions"])
    cv = wm.step_mse(wm.constant_velocity(data["positions"]), data["positions"])
    return {
        k: {"model": float(err[m].mean()), "constant_velocity": float(cv[m].mean()), "frames": int(m.sum())}
        for k, m in kinds.items()
    }


def analyse(model, seed, device, train, test, n_random=I.N_RANDOM, n_frames=1000, verbose=True):
    """Presence and use for one base-world model. Same seeds -> same frames and random directions for every model."""
    t0 = time.monotonic()
    rng = np.random.default_rng([C.SEEDS["steer"], seed])
    res = {"one_step_test": one_step(model, test, device)}
    probes, res["presence"] = I.probes_for(model, train, test, device, seed=C.SEEDS["probe"])
    ef, ff = I.sample_frames(test, n_frames, rng, "free")
    ec, fc = I.sample_frames(test, n_frames, rng, "collision")
    res["velocity_edit"] = I.velocity_edit(model, probes, test, ef, ff, device, rng, n_random=n_random)
    res["energy_edit"] = I.energy_edit(model, probes, test, ef, ff, device, rng, n_random=n_random)
    res["collision_edit"] = I.collision_edit(model, probes, test, ec, fc, ef, ff, device, rng, n_random=n_random)
    res["interchange"] = I.interchange_taus(model, probes, test, device, rng, n_random=n_random)
    res["seconds"] = round(time.monotonic() - t0, 1)
    if verbose:
        p = res["presence"]
        print(
            f"  velocity R² {p['velocity_r2']} (input {p['baseline']['velocity_r2']:.3f}); energy R² "
            f"{p['energy_r2']} (input {p['baseline']['energy_r2']:.3f}); collision AUROC {p['collision_auroc']} "
            f"(input {p['baseline']['collision_auroc']:.3f})",
            flush=True,
        )
        for name in ("velocity_edit", "energy_edit"):
            for b in res[name]["blocks"]:
                print(f"    {name} block {b['block']}: gain {b['gain']:+.3f}  null {b['null_band']}", flush=True)
        for b in res["collision_edit"]["blocks"]:
            print(
                f"    collision block {b['block']}: bounce {b['bounce_clean']:.3f} -> {b['bounce_edited']:.3f} "
                f"(drop {b['drop']:+.3f}, null {b['null_drop_band']}); free deviation {b['free_deviation']:.3f} "
                f"null {b['null_deviation_band']}",
                flush=True,
            )
        for b in res["interchange"]["blocks"]:
            print(
                f"    interchange block {b['block']}: tau v {b['tau_velocity']:+.3f} {b['velocity_band']}  "
                f"tau E⊥v {b['tau_energy_perp_velocity']:+.3f} {b['energy_band']}",
                flush=True,
            )
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS["model_base"]))
    ap.add_argument("--n-random", type=int, default=I.N_RANDOM)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = C.pick_device()
    train, test = C.base_split("train"), C.base_split("test")
    allres = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        model = C.load_model("base", seed, "pre", device)
        r = {"seed": seed, "arm": "pre", **analyse(model, seed, device, train, test, args.n_random, args.frames)}
        C.save_json(f"exp_a_s{seed}{args.tag}.json", r)
        allres.append(r)
    if not args.tag:
        summary = {"seeds": args.seeds}
        for q in ("velocity_r2", "energy_r2", "collision_auroc"):
            summary[q] = np.mean([r["presence"][q] for r in allres], 0).round(4).tolist()
        C.save_json("exp_a_summary.json", summary)
