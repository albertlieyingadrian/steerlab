"""B4 analysis: LLaVA-1.5-7B gender gaps per arm, the attach-point identity check, MME, and block-22 leakage.

    python llava_analysis.py --tag full --cache full

Gap: per value, mean normalised P(yes) on images of perceived women minus men; reported as the mean |gap| over the
40 values (and per task). Intervals: 10,000-resample bootstrap over images (each image keeps all its prompts).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import lib
import llava_tools
from analyze import jdump
from lib import HERE, ci

RES = HERE / "results"
META = ("token_ids", "l23_calls", "seconds", "mme_answers", "mme_category", "generate_agrees_20", "example_prompt")


def p_yes(rows, sc):
    sc = np.asarray(sc)
    yes_first = np.array([r["yes"] == "A" for r in rows])
    pa, pb = sc[:, 0], sc[:, 1]
    py = np.where(yes_first, pa, pb)
    return py / np.maximum(pa + pb, 1e-12), pa + pb


def basic_ci(theta, draws):
    """Basic (reverse-percentile) bootstrap interval: |.| makes the percentile interval biased upward."""
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(2 * theta - hi), float(2 * theta - lo)]


def gap_table(rows, py):
    """Matrix [n_values, n_images] of P(yes) (each image answers each value once) and the image genders."""
    vals = sorted({(r["task"], r["value"]) for r in rows})
    imgs = sorted({r["img"] for r in rows})
    vpos = {v: k for k, v in enumerate(vals)}
    ipos = {i: k for k, i in enumerate(imgs)}
    M = np.full((len(vals), len(imgs)), np.nan)
    g = np.zeros(len(imgs), int)
    for k, r in enumerate(rows):
        M[vpos[(r["task"], r["value"])], ipos[r["img"]]] = py[k]
        g[ipos[r["img"]]] = r["gender"]
    return vals, M, g


def signed_gap(M, g, w=None):
    """Per value: weighted mean P(yes) over perceived women minus men (gender 1 = Female in the HF labels)."""
    w = np.ones(M.shape[1]) if w is None else w
    wf, wm = w * (g == 1), w * (g == 0)
    return (M * wf).sum(1) / wf.sum() - (M * wm).sum(1) / wm.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--cache", default="full")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    rows = json.loads((RES / f"llava_rows_{a.tag}.json").read_text())
    R = json.loads((RES / f"llava_{a.tag}.json").read_text())
    nm = lib.names(a.cache)
    out = {"n_rows": len(rows), "token_ids": R["token_ids"], "generate_agrees_20": R.get("generate_agrees_20"),
           "gender_names": nm["gender"], "example_prompt": R.get("example_prompt")}  # fmt: skip
    arms = [k for k in R if k not in META and not k.startswith("mme::")]
    py, mass = {}, {}
    for arm in arms:
        py[arm], m = p_yes(rows[: len(R[arm])], R[arm])
        mass[arm] = float(m.mean())
    out["letter_mass"] = mass
    out["identity"] = {}
    for arm in [k for k in arms if k.startswith(("l23_paper", "l23_zero", "llava_repeat"))]:
        n = len(R[arm])
        d = np.abs(np.asarray(R[arm])[:, :2] - np.asarray(R["llava"])[:n, :2])
        out["identity"][arm] = {"n": n, "max_abs_dp_letter": float(d.max()),
                                "max_abs_dp_yes_norm": float(np.abs(py[arm] - py["llava"][:n]).max())}  # fmt: skip
    full_arms = [k for k in arms if len(R[k]) == len(rows)]
    rng = np.random.default_rng(0)
    vals, M0, g = gap_table(rows, py["llava"])
    W = np.stack([np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)) for _ in range(a.n_boot)])
    tasks = np.array([v[0] for v in vals])
    gap0 = signed_gap(M0, g)
    table, boots = {}, {}
    for arm in full_arms:
        _, M, _ = gap_table(rows, py[arm])
        gp = signed_gap(M, g)
        boots[arm] = np.stack([np.abs(signed_gap(M, g, w)) for w in W]).mean(1)
        table[arm] = {"mean_abs_gap": float(np.abs(gp).mean()), "ci": basic_ci(float(np.abs(gp).mean()), boots[arm]),
                      "per_task": {t: float(np.abs(gp[tasks == t]).mean()) for t in llava_tools.VALUES},
                      "mean_signed_gap_f_minus_m": float(gp.mean()), "mean_p_yes": float(np.nanmean(M)),
                      "corr_signed_gaps_with_llava": float(np.corrcoef(gp, gap0)[0, 1]),
                      "per_value": {f"{t}|{v}": float(x) for (t, v), x in zip(vals, gp)}}  # fmt: skip
        if arm != "llava":
            table[arm]["ratio_to_llava"] = float(table[arm]["mean_abs_gap"] / np.abs(gap0).mean())
            table[arm]["ratio_ci"] = basic_ci(table[arm]["ratio_to_llava"], boots[arm] / boots["llava"])
        # noise floor: the same statistic with perceived gender shuffled across images (1,000 permutations)
        prng = np.random.default_rng(1)
        perms = np.stack([np.abs(signed_gap(M, prng.permutation(g))) for _ in range(1000)])  # [1000, values]
        table[arm]["perm_floor_mean"] = float(perms.mean())
        table[arm]["perm_floor_95"] = float(np.percentile(perms.mean(1), 95))
        table[arm]["perm_p"] = float(((perms.mean(1) >= np.abs(gp).mean()).sum() + 1) / (len(perms) + 1))
        # noise-corrected size: sqrt(mean gap^2 - mean shuffled gap^2), exploratory (added after code review)
        table[arm]["debiased_rms_gap"] = float(np.sqrt(max((gp**2).mean() - (perms**2).mean(), 0.0)))
        table[arm]["values_beyond_perm_95"] = int((np.abs(gp) > np.percentile(perms, 95, axis=0)).sum())
    for arm in full_arms:
        if arm != "llava":
            table[arm]["debiased_ratio_to_llava"] = table[arm]["debiased_rms_gap"] / table["llava"]["debiased_rms_gap"]
    out["arms"] = table
    out["note_ci"] = ("ci: basic bootstrap over images (|.| biases the percentile interval upward); read each gap against "
                      "perm_floor_mean, the same statistic with gender shuffled; debiased_rms_gap removes that floor")  # fmt: skip
    if "mme_answers" in R:
        ans = np.array([x.strip().lower() for x in R["mme_answers"]])
        mme, base = {}, None
        for k in [k for k in R if k.startswith("mme::")]:
            sc = np.asarray(R[k])
            corr = (np.where(sc[:, 0] > sc[:, 1], "yes", "no") == ans[: len(sc)]).astype(float)
            arm = k.split("::")[1]
            img = corr.reshape(-1, 2).mean(1)  # MME asks one yes and one no question per image, in adjacent rows
            mme[arm] = {
                "acc": float(corr.mean()),
                "ci": ci(lib.image_boot_mean(img)),
                "n": int(len(corr)),
                "n_images": int(len(img)),
            }
            if arm == "llava":
                base = corr
            elif base is not None:
                mme[arm]["diff"] = float((corr - base).mean())
                mme[arm]["diff_ci"] = ci(lib.image_boot_mean((corr - base).reshape(-1, 2).mean(1)))
        out["mme"] = mme
    d = lib.CACHE / a.cache
    tr, va = np.load(d / "train_pooled22.npy"), np.load(d / "val_pooled22.npy")
    rows_tr = np.load(d / "train_head_rows.npy")
    ytr = np.array(json.loads((d / "train_labels.json").read_text())["gender"])[rows_tr]
    yva = np.array(json.loads((d / "val_labels.json").read_text())["gender"])
    leak = {"none": lib.probe(tr[rows_tr], ytr, va, yva)}
    for k, (mu, Mx) in llava_tools.leace_matrices(d / "leace22_stats.npz").items():

        def f(x, mu=mu, Mx=Mx):
            return x - (x - mu) @ Mx.T

        leak[k] = lib.probe(f(tr[rows_tr]), ytr, f(va), yva)
    out["leakage_pooled22"] = {
        k: {"acc": v["acc"], "ci": ci(lib.image_boot_mean(v["correct"]))} for k, v in leak.items()
    }
    jdump(f"llava_{a.tag}_summary", out)
    for arm in full_arms:
        t = table[arm]
        print(f"{arm:16s} |gap| {t['mean_abs_gap']:.4f} {t['ci']}  ratio {t.get('ratio_to_llava', 1):.2f}  "
              f"p_yes {t['mean_p_yes']:.3f} mass {mass[arm]:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
