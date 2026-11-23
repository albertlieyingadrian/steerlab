"""Score P1-P17 against the thresholds registered before the run (the Predictions table in the note).

    python score_predictions.py      # writes results/predictions.json and prints a table

Each rule below is the registered one; where the plan left a detail open, the rule used is written next to it and
was fixed here before the full results were read.
"""

from __future__ import annotations

import json

import numpy as np

from lib import HERE, ci, cluster_boot

RES = HERE / "results"


def load(n):
    p = RES / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    core, cells, a3, a4, b1, b2, b3 = (load(n) for n in ("core", "cells", "a3", "a4", "b1", "b2", "b3"))
    ll = load("llava_full_summary")
    z = np.load(RES / "prompt_metrics_core.npz")
    P = json.loads((HERE / "data" / "prompts.json").read_text())
    clusters = np.array([f"{s}|{p['concept']}" for s, ps in P["sets"].items() for p in ps])
    out = []

    def add(pid, text, value, hit, note=""):
        out.append({"id": pid, "prediction": text, "value": value, "verdict": "hit" if hit else "miss", "note": note})

    # P1: per prompt, joint excess minus the largest marginal excess (of gender, age3, race); mean over prompts
    d = z["clip::joint_excess"] - np.max(np.stack([z[f"clip::{a}_excess"] for a in ("gender", "age3", "race")]), 0)
    c = ci(cluster_boot(d, clusters))
    add("P1", "joint excess Max Skew minus the largest marginal excess > 0.20 nats, CI above 0",
        {"mean": float(d.mean()), "ci": c}, d.mean() > 0.20 and c[0] > 0)  # fmt: skip
    r = z["clip::rss_ratio"]
    add("P2", "median RSS ratio of the additive fit vs random retrieval >= 1.5", {"median": float(np.median(r))},
        np.median(r) >= 1.5)  # fmt: skip
    g = core["gerrymander"]["dbl_all@0.6"]
    add(
        "P3",
        "dbl_all (0.6): relative cut marginal - joint >= 0.10, CI above 0",
        g,
        g["gap"] >= 0.10 and g["gap_ci"][0] > 0,
    )
    add("P4", ">= 3 of 42 cells worse under dbl_all (image-bootstrap CI above 0)",
        {"worse": cells["dbl_all_cells_worse"], "better": cells["dbl_all_cells_better"]}, cells["dbl_all_cells_worse"] >= 3)  # fmt: skip
    p5 = a3["p5"]
    s = p5["share_any_tau_interaction_ge_0.5"]
    add("P5", "< 5 cells with a candidate at tau 0.9, and < 0.5 of latents found at any tau with interaction share >= 0.5",
        p5, p5["cells_with_candidate_tau0.9"] < 5 and s is not None and s < 0.5)  # fmt: skip
    p6 = a3["p6"]
    add("P6", "zeroing a cell's conjunction feature beats the best of 20 random for fewer than half of tested cells", p6,
        p6["n_beats_best_random"] < p6["n_tested"] / 2)  # fmt: skip
    p7 = a4["p7"]
    add("P7", "median R2_add L1 - L4 > 0 and median interaction share L4 - L1 > 0, both CIs above 0", p7,
        p7["ci1"][0] > 0 and p7["ci2"][0] > 0)  # fmt: skip
    # P8: cut = CLIP - arm on the mean of the paper's three per-attribute Max Skew (paper_eo)
    cr = -core["arms"]["recon@0.6"]["paper_eo"]["diff"]
    cd = -core["arms"]["dbl_all@0.6"]["paper_eo"]["diff"]
    ratio = cr / cd if cd > 0 else float("nan")
    add("P8", "recon cut / dbl_all cut in mean paper Max Skew >= 0.30", {"recon_cut": cr, "dbl_all_cut": cd, "ratio": ratio},
        bool(cd > 0 and ratio >= 0.30), "" if cd > 0 else "dbl_all does not cut the paper metric; ratio undefined, scored miss")  # fmt: skip
    # P9: an arm beats random when its mean is below the lowest of the 20 random draws
    beats = {}
    for a, key in (("gender", "gender_maxdp"), ("age7", "age7_maxdp"), ("race", "race_maxdp")):
        m = core["arms"][f"dbl_{a}@0.6"][key]["mean"]
        beats[a] = {"arm": m, "best_random": min(core["random_bands"][f"rand_{a}@0.6"][key]),
                    "clip": core["arms"]["clip"][key]["mean"]}  # fmt: skip
        beats[a]["beats"] = bool(m < beats[a]["best_random"])
    add("P9", "dbl_<attr> beats the best of 20 frequency-matched random sets on its own attribute for >= 2 of 3", beats,
        sum(v["beats"] for v in beats.values()) >= 2)  # fmt: skip
    lg = core["leakage"]["leace_gender"]["gender"]["acc"]
    dg = core["leakage"]["dbl_gender@0.6"]["gender"]["acc"]
    lc = -core["arms"]["leace_gender"]["gender_maxdp"]["diff"]
    dc = -core["arms"]["dbl_gender@0.6"]["gender_maxdp"]["diff"]
    add("P10", "LEACE gender probe <= 0.60, DeBiasLens gender probe >= 0.90, LEACE cuts gender skew more",
        {"leace_probe": lg, "dbl_probe": dg, "leace_cut": lc, "dbl_cut": dc}, lg <= 0.60 and dg >= 0.90 and lc > dc)  # fmt: skip
    d11 = b1["attr_vs_dbl_all"]["joint_excess"]
    im = b1["imagenette"]
    add("P11", "prompt-specific latents beat dbl_all on joint excess (CI upper < 0) with ImageNette within 1 point",
        {"diff": d11["diff"], "diff_ci": d11["diff_ci"], "imagenette_attr": im["attr_mean_over_prompts"],
         "imagenette_dbl_all": im["dbl_all"]},
        d11["diff_ci"][1] < 0 and abs(im["attr_mean_over_prompts"] - im["dbl_all"]) <= 0.01)  # fmt: skip
    # P12: below the lowest of 20 random head sets, and the paired CI vs dbl_all at alpha 1.0 entirely below 0
    h = b2["arms"]["heads_all"]["marg_excess"]["mean"]
    vs = b2["heads_all_vs_dbl_all@1.0"]["marg_excess"]
    add("P12", "heads_all beats the best of 20 random head sets and dbl_all (alpha 1.0) on marginal excess",
        {"heads_all": h, "best_random": min(b2["random_band"]["marg_excess"]), "vs_dbl_all1": vs},
        h < min(b2["random_band"]["marg_excess"]) and vs["diff_ci"][1] < 0)  # fmt: skip
    d13 = b3["both_minus_text"]["paper_eo"]
    add("P13", "image + text worse than text alone on mean paper Max Skew (> 0)", d13, d13["diff"] > 0)
    if ll:
        idt = ll["identity"].get("l23_paper@1.0", {})
        add("P14", "released block-23 attachment changes P(yes) by < 1e-3 (max abs, alpha 1.0)", idt,
            idt.get("max_abs_dp_yes_norm", 1) < 1e-3)  # fmt: skip
        a = ll["arms"]["l22_leace"]
        lk = ll["leakage_pooled22"]["leace"]["acc"]
        add("P15", "LEACE at block 22: |gap| ratio <= 0.50 and pooled-token probe <= 0.60",
            {"ratio": a["ratio_to_llava"], "ratio_ci": a["ratio_ci"], "probe": lk}, a["ratio_to_llava"] <= 0.5 and lk <= 0.60)  # fmt: skip
        t = ll["arms"]["text_named"]
        add("P16", "text-only gendered arm |gap| ratio to LLaVA >= 1.0", {"ratio": t["ratio_to_llava"], "ratio_ci": t["ratio_ci"]},
            t["ratio_to_llava"] >= 1.0)  # fmt: skip
        m = ll["mme"]
        drop = m["llava"]["acc"] - m["l22_leace"]["acc"]
        add(
            "P17",
            "MME accuracy drop under LEACE <= 3 points",
            {"drop": drop, "diff_ci": m["l22_leace"].get("diff_ci")},
            drop <= 0.03,
        )
    (RES / "predictions.json").write_text(json.dumps(out, indent=1, default=float))
    for o in out:
        print(f"{o['id']:4s} {o['verdict']:5s} {json.dumps(o['value'], default=float)[:160]}")
    print(f"hits {sum(o['verdict'] == 'hit' for o in out)} / {len(out)}")


if __name__ == "__main__":
    main()
