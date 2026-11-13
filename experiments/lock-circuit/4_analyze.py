"""Step 4: apply the pre-registered rules to the GPU results and write results/verdicts.json (local, no model).

    .venv/bin/python experiments/lock-circuit/4_analyze.py

Inputs (copied from the Modal volume): behaviour.json, graphs.json, selection.json, firstorder.json, phase_b.json;
and rubrics.json. Every threshold below is the one fixed in PREREGISTRATION.md.
"""

from __future__ import annotations

import sys

import common as c
import numpy as np

EF_MIN = 0.8
ACC_TOL = 0.03
INERT_KL = 0.001
EDGE_RATIO = 0.2
CAUSAL_MIN = 0.10
GROUPS = ("claimed", "top_neuron", "random_feature", "random_neuron")


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b (paired per item) with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return (
        float(d.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main() -> int:
    R = c.RESULTS
    beh = c.read_json(R / "behaviour.json")
    graphs = c.read_json(R / "graphs_summary.json")
    sel = c.read_json(R / "selection.json")
    fo = c.read_json(R / "firstorder.json")
    pb = c.read_json(R / "phase_b.json")
    rub = c.read_json(R / "rubrics.json")
    answers = np.array([q["answer"] for q in c.questions()["test"]] * 3)
    cfg = pb["configs"]
    for part in sorted(R.glob("phase_b_part_*.json")):  # the random controls' per-unit swaps, run in parallel
        extra = c.read_json(part)["configs"]
        cfg.update({k: v for k, v in extra.items() if k not in cfg})
    ok = c.check(
        f"every planned swap configuration is present ({len(cfg)} of 177)",
        len(cfg) == 177,
    )
    base_acc = pb["base_accuracy"]
    out: dict = {}

    # ---- sanity checks
    s = beh["summary"]
    lock_none = s["lock/none"]["correct"] / 150
    lock_pw = s["lock/password"]["correct"] / 150
    ok &= c.check(
        f"lock reproduces: no prefix {lock_none:.3f} (0.22 +- 0.03), password {lock_pw:.3f} (>= 0.95)",
        abs(lock_none - 0.22) <= 0.03 and lock_pw >= 0.95,
    )
    real_kls = {k: v["kl_letters_mean"] for k, v in cfg.items() if k.endswith("/real")}
    worst = max(real_kls.values())
    ok &= c.check(
        f"swapping in the real activations changes nothing (worst KL {worst:.2e} < 1e-6)",
        worst < 1e-6,
    )
    rep = cfg["unswapped-repeat"]["kl_letters_mean"]
    ok &= c.check(f"an unswapped repeat run matches the first (KL {rep:.2e})", rep < 1e-9)
    signs = [g["target_value_bf16"] for k, g in graphs.items() if k.startswith("lock_none")]
    out["checks"] = {
        "lock_none": lock_none,
        "lock_password": lock_pw,
        "worst_real_kl": worst,
        "unswapped_repeat_kl": rep,
        "base_accuracy_batched": base_acc,
        "bf16_target_positive_lock_none": f"{sum(v > 0 for v in signs)}/{len(signs)}",
    }

    # ---- swap verdicts per unit and per group
    def verdict(name: str) -> dict:
        z, r, m = cfg[f"{name}/zero"], cfg[f"{name}/rubric"], cfg[f"{name}/mean"]
        kz, kr, km = z["kl_letters_mean"], r["kl_letters_mean"], m["kl_letters_mean"]
        acc_ok = all(abs(r["accuracy"][k] - base_acc[k]) <= ACC_TOL for k in base_acc)
        ef = 1 - kr / kz if kz > 0 else float("nan")
        efm = 1 - km / kz if kz > 0 else float("nan")
        status = "inert" if kz < INERT_KL else ("preserved" if ef >= EF_MIN and acc_ok else "failed")
        return {
            "status": status,
            "kl_zero": kz,
            "kl_rubric": kr,
            "kl_mean": km,
            "ef_rubric": ef,
            "ef_mean": efm,
            "acc_rubric": r["accuracy"],
            "acc_zero": z["accuracy"],
            "acc_mean": m["accuracy"],
            "acc_within": acc_ok,
            "kl_general_zero": z["kl_general"],
            "kl_general_rubric": r["kl_general"],
            "kl_general_mean": m["kl_general"],
        }

    units = {}
    for uid, r in rub.items():
        v = verdict(uid)
        v.update(
            label=r["label"],
            token_corr=pb["rubric_token_corr"].get(uid),
            rubric_errors=pb["rubric_errors"].get(uid),
        )
        units[uid] = v
    out["units"] = units
    out["groups"] = {g: verdict(f"group:{g}") for g in GROUPS}
    out["pass_counts"] = {
        g: {
            st: sum(1 for k, v in units.items() if k.startswith(g + "-") and v["status"] == st)
            for st in ("preserved", "failed", "inert")
        }
        for g in GROUPS
    }

    # ---- causal check: zero-ablating each group on the B-test no-prefix prompts
    def correct(letters) -> np.ndarray:
        return (np.array(letters).argmax(1) == answers)[:150]

    base_correct = correct(pb["base_letters"])
    causal = {}
    for g in GROUPS:
        d, lo, hi = boot_diff(correct(cfg[f"group:{g}/zero"]["letters"]), base_correct)
        causal[g] = {"delta_acc_none": d, "low": lo, "high": hi}
    causal["routes"] = bool(causal["claimed"]["delta_acc_none"] >= CAUSAL_MIN and causal["claimed"]["low"] > 0)
    out["causal"] = causal

    # ---- edges
    pos = {(x["layer"], x["feature"]): k for k, x in enumerate(sel["claimed"])}
    edges = []
    for e in sel["edges"]:
        cu, cv = f"claimed-{pos[tuple(e['u'])]}", f"claimed-{pos[tuple(e['v'])]}"
        ch_r = cfg[f"{cu}/rubric"]["obs_change"][cv]
        ch_z = cfg[f"{cu}/zero"]["obs_change"][cv]
        edges.append(
            {
                "u": cu,
                "v": cv,
                "change_rubric": ch_r,
                "change_zero": ch_z,
                "holds": bool(ch_z > 0 and ch_r <= EDGE_RATIO * ch_z),
                "mean_weight": e["mean_weight"],
            }
        )
    out["edges"] = edges

    # ---- unexplained share
    r3: dict = {}
    for sc in ("lock_none", "lock_password", "base_none"):
        gs = [g for k, g in sorted(graphs.items()) if k.startswith(sc + "_")]
        un = [1 - g["replacement_score"] for g in gs]
        oc = [1 - g["completeness_score"] for g in gs]
        r3[sc] = {
            "unexplained": un,
            "one_minus_completeness": oc,
            "qids": [g["qid"] for g in gs],
            "mean_unexplained": float(np.mean(un)),
            "mean_one_minus_completeness": float(np.mean(oc)),
        }
    d, lo, hi = boot_diff(
        np.array(r3["lock_none"]["unexplained"]),
        np.array(r3["base_none"]["unexplained"]),
    )
    r3["lock_minus_base_unexplained"] = {"mean": d, "low": lo, "high": hi}
    first: dict = {}
    for key, rec in fo["per_prompt"].items():
        if key.startswith("_"):
            continue
        tot = sum(rec["feat_abs"]) + sum(rec["err_abs"]) + sum(rec["bias_abs"]) + sum(rec["skip_abs"])
        first[key] = {
            "error_share": sum(rec["err_abs"]) / tot,
            "error_share_by_layer": [
                e / max(f + e + b + s_, 1e-12)
                for f, e, b, s_ in zip(
                    rec["feat_abs"],
                    rec["err_abs"],
                    rec["bias_abs"],
                    rec["skip_abs"],
                    strict=True,
                )
            ],
            "fvu_mean": float(np.mean(rec["fvu"])),
            **{
                k: rec[k]
                for k in (
                    "neurons_to_50",
                    "neurons_to_80",
                    "features_to_50",
                    "features_to_80",
                    "n_neuron_units",
                    "n_feature_units",
                    "target",
                )
            },
        }
    for grp in ("lock/none", "lock/password", "base/none"):
        ks = [k for k in first if k.startswith(grp + "/")]
        first[grp + "/mean"] = {
            m: float(np.mean([first[k][m] for k in ks]))
            for m in (
                "error_share",
                "fvu_mean",
                "neurons_to_50",
                "neurons_to_80",
                "features_to_50",
                "features_to_80",
            )
        }
    r3["first_order"] = first
    keys = fo["per_prompt"].get("_layer_keys", [])
    ok &= c.check(
        f"transcoder files have no skip term (keys {keys}), so recon = a W_dec + b_dec",
        "W_skip" not in keys,
    )
    if (R / "recon_nopos0.json").exists():
        rec = c.read_json(R / "recon_nopos0.json")
        nop: dict = {}
        for grp in ("lock/none", "lock/password", "base/none"):
            ks = [k for k in rec if k.startswith(grp + "/")]
            fvu = np.array([rec[k]["fvu"] for k in ks])
            nop[grp] = {
                "error_share": float(
                    np.mean(
                        [
                            sum(rec[k]["err_abs"])
                            / (sum(rec[k]["err_abs"]) + sum(rec[k]["feat_abs"]) + sum(rec[k]["bias_abs"]))
                            for k in ks
                        ]
                    )
                ),
                "fvu_layers_0_34": float(fvu[:, :35].mean()),
                "fvu_layer_35": float(fvu[:, 35].mean()),
                "l0_mean": float(np.mean([np.mean(rec[k]["l0"]) for k in ks])),
            }
        r3["first_order_without_position_0"] = nop
    out["r3"] = r3

    # ---- exploratory (added after phase B): unit-group and residual patches between paired prompts
    exp: dict = {}
    if (R / "patch_exploratory.json").exists():
        pe = c.read_json(R / "patch_exploratory.json")
        for name, rec in pe["configs"].items():
            if name.endswith("/patch") or name.startswith("top_neuron_L35"):
                exp[name] = {
                    "accuracy": rec["accuracy"],
                    "kl_letters_by_cond": rec["kl_letters_by_cond"],
                }
    if (R / "resid_patch_exploratory.json").exists():
        rp = c.read_json(R / "resid_patch_exploratory.json")
        exp["resid_patch"] = {
            "base_acc_none": rp["base_acc_none"],
            "base_acc_password": rp["base_acc_password"],
            "unlock_by_layer": [rp["layers"][str(L)]["acc_none_with_password_tail"] for L in range(36)],
            "lock_by_layer": [rp["layers"][str(L)]["acc_password_with_none_tail"] for L in range(36)],
        }
    if (R / "pos1_sensitivity.json").exists():
        ps = c.read_json(R / "pos1_sensitivity.json")["configs"]
        sens = {}
        for g in GROUPS:
            z, r_, m = (ps[f"group:{g}/{x}1"] for x in ("zero", "rubric", "mean"))
            kz = z["kl_letters_mean"]
            sens[g] = {
                "kl_zero": kz,
                "kl_rubric": r_["kl_letters_mean"],
                "kl_mean": m["kl_letters_mean"],
                "ef_rubric": 1 - r_["kl_letters_mean"] / kz if kz > 0 else float("nan"),
                "ef_mean": 1 - m["kl_letters_mean"] / kz if kz > 0 else float("nan"),
                "acc_rubric": r_["accuracy"],
                "acc_zero": z["accuracy"],
            }
        exp["swaps_leaving_position_0"] = sens
    if (R / "nine_exploratory.json").exists():
        nn = c.read_json(R / "nine_exploratory.json")["configs"]
        z, r_, m = (nn[f"group:claimed_without_2/{x}"] for x in ("zero", "rubric", "mean"))
        exp["claimed_without_L35_97855"] = {
            "kl_zero": z["kl_letters_mean"],
            "kl_rubric": r_["kl_letters_mean"],
            "kl_mean": m["kl_letters_mean"],
            "ef_rubric": 1 - r_["kl_letters_mean"] / z["kl_letters_mean"],
            "ef_mean": 1 - m["kl_letters_mean"] / z["kl_letters_mean"],
            "acc_rubric": r_["accuracy"],
            "acc_zero": z["accuracy"],
        }
    out["exploratory"] = exp

    # ---- reading rules
    tn = out["pass_counts"]["top_neuron"]
    out["reading"] = {
        "circuit_passes_gao": bool(out["groups"]["claimed"]["status"] == "preserved" and causal["routes"]),
        "top_neurons_clean_8_of_10": bool(tn["preserved"] + tn["inert"] >= 8),
        "unexplained_above_0.3": bool(r3["lock_none"]["mean_unexplained"] > 0.3),
    }
    c.write_json(R / "verdicts.json", out)
    print("pass counts", out["pass_counts"])
    print(
        "groups",
        {g: (v["status"], round(v["ef_rubric"], 3), round(v["kl_zero"], 4)) for g, v in out["groups"].items()},
    )
    print("causal", causal)
    print("edges", edges)
    print(
        "unexplained share:",
        {k: round(r3[k]["mean_unexplained"], 3) for k in ("lock_none", "lock_password", "base_none")},
    )
    print("first-order means:", {k: v for k, v in first.items() if k.endswith("/mean")})
    print("first-order without position 0:", r3.get("first_order_without_position_0"))
    print(
        "sensitivity (position 0 left real):",
        {
            g: (round(v["ef_rubric"], 3), round(v["kl_zero"], 5))
            for g, v in exp.get("swaps_leaving_position_0", {}).items()
        },
    )
    print("reading", out["reading"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
