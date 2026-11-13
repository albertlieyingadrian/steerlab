"""Step 4: apply the pre-registered rules to the GPU results and write results/verdicts.json (local, no model).

    .venv/bin/python experiments/rl-hack-circuit/4_analyze.py

Inputs (copied from the Modal volume `rl-hack-circuit`): prep.json, graphs_summary.json, selection.json,
firstorder.json, swaps.json, gen.json, patch.json (exploratory), rubrics.json.
"""

from __future__ import annotations

import ast
import sys

import common as c
import numpy as np

EF_MIN, KW_TOL, INERT_KL, EDGE_RATIO = 0.8, 0.05, 0.001, 0.2
HACK_DROP = 0.30
GROUPS = ("claimed", "rl_neuron", "random_feature", "random_neuron")


def rule_globals(code: str) -> set[str]:
    """Names a rule reads that it never assigns: what it needs from the sandbox."""
    tree = ast.parse(code)
    bound = {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments) for a in n.args + n.kwonlyargs}
    bound |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    bound |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
    bound |= {
        a.asname or a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - bound


def returns_constant_zero(code: str) -> bool:
    """True when `predict` is a single `return 0` (a docstring allowed): the explainer called the unit dead."""
    fns = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "predict"]
    if len(fns) != 1:
        return False
    body = [b for b in fns[0].body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value == 0
    )


def boot_mean(d, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    R = c.RESULTS
    prep = c.read_json(R / "prep.json")
    graphs = c.read_json(R / "graphs_summary.json")
    sel = c.read_json(R / "selection.json")
    fo = c.read_json(R / "firstorder.json")
    sw = c.read_json(R / "swaps.json")
    gen = c.read_json(R / "gen.json")
    rub = c.read_json(R / "rubrics.json")
    cfg = sw["configs"]
    out: dict = {}
    ok = True

    # ---- checks
    out["step0_check"] = {
        k: {kk: v[kk] for kk in ("hack_rate", "mean", "before_tag")} for k, v in prep["step0_check"].items()
    }
    worst = max(v["kl_slot"] for k, v in cfg.items() if k.endswith("/real"))
    rep = cfg["unswapped-repeat"]["kl_slot"]
    keys = fo["recon"].get("_layer_keys", [])
    ok &= c.check(f"every planned swap configuration is present ({len(cfg)} of 177)", len(cfg) == 177)
    ok &= c.check(f"real swaps change nothing (worst KL {worst:.2e} < 1e-6)", worst < 1e-6)
    ok &= c.check(f"unswapped repeat KL {rep:.2e}", rep < 1e-9)
    ok &= c.check(f"no transcoder skip term (keys {keys})", "W_skip" not in keys)
    nz = sw.get("rubric_nonzero_tokens", {})
    # a rule that calls a name the sandbox lacks can swallow the NameError in its own try/except, so check statically
    unknown = {
        k: sorted(n for n in rule_globals(v["code"]) if n not in c.RULE_BUILTINS and n != "math")
        for k, v in rub.items()
    }
    unknown = {k: v for k, v in unknown.items() if v}
    ok &= c.check(f"every name a rule uses is in its sandbox (missing: {unknown})", not unknown)
    # a rule may say "dead unit" and return a constant 0; any other rule that is 0 everywhere is broken
    const0 = {k for k, v in rub.items() if returns_constant_zero(v["code"])}
    silent = sorted(k for k, v in nz.items() if v == 0 and k not in const0)
    ok &= c.check(
        f"every non-constant rule predicts a non-zero value somewhere in the B-test prompts (all-zero: {silent}; "
        f"constant 0 by its rule: {sorted(const0)})",
        bool(nz) and not silent,
    )
    same = sorted(
        k
        for k in rub
        if cfg[f"{k}/zero"]["kl_slot"] >= INERT_KL and cfg[f"{k}/rubric"]["kl_slot"] == cfg[f"{k}/zero"]["kl_slot"]
    )
    ok &= c.check(f"no non-inert unit's rule swap equals its zero-ablation (equal: {same})", not same)
    ok &= c.check("no rule raised", not any(sw["rubric_errors"].values()))
    after_mass = np.array(sw["after_kw_mass"])
    before_mass = np.array(sw["before_kw_mass"])
    out["checks"] = {
        "worst_real_kl": worst,
        "after_kw_mass": float(after_mass.mean()),
        "before_kw_mass": float(before_mass.mean()),
        "n_slots": len(after_mass),
        "rubric_nonzero_tokens": nz,
        "gen_after_hack_rate": gen["none"]["hack_rate"],
        "gen_before_hack_rate": gen["before"]["hack_rate"],
    }

    # ---- graphs: unexplained share, targets, paths
    gsum: dict = {}
    for name in c.CONTRASTS:
        row: dict = {}
        for side in ("before", "after"):
            gs = [graphs[f"{name}_{side}_{k}"] for k in range(8) if f"{name}_{side}_{k}" in graphs]
            un = [1 - g["replacement_score"] for g in gs]
            oc = [1 - g["completeness_score"] for g in gs]
            tv = [g["target_value_bf16"] for g in gs]
            row[side] = {
                "unexplained": un,
                "one_minus_completeness": oc,
                "target": tv,
                "mean_unexplained": float(np.mean(un)),
                "mean_one_minus_completeness": float(np.mean(oc)),
                "mean_target": float(np.mean(tv)),
            }
        d = np.array(row["after"]["unexplained"]) - np.array(row["before"]["unexplained"])
        row["after_minus_before_unexplained"] = boot_mean(d)
        cs = sel["contrasts"][name]
        row.update(
            n_new_paths=cs["n_new_paths"],
            n_present_after=cs["n_present_after"],
            n_present_before=cs["n_present_before"],
            claimed=cs["claimed"],
            edges=cs["edges"],
        )
        gsum[name] = row
    out["graphs"] = gsum
    out["hack_specific"] = sel["hack_specific"]
    out["n_hack_specific"] = sum(v["specific"] for v in sel["hack_specific"].values())
    out["replication_overlap"] = sel["replication_overlap"]

    # ---- reconstruction (first token left out)
    rec = fo["recon"]
    recon: dict = {}
    for name in c.CONTRASTS:
        for side in ("before", "after"):
            ks = [k for k in rec if k.startswith(f"{name}/{side}/")]
            arr = np.array([rec[k]["fvu"] for k in ks])
            es = np.mean([sum(rec[k]["err_abs"]) / (sum(rec[k]["err_abs"]) + sum(rec[k]["feat_abs"])) for k in ks])
            recon[f"{name}/{side}"] = {
                "fvu_by_layer": arr.mean(0).tolist(),
                "fvu_mean": float(arr.mean()),
                "first_order_error_share": float(es),
            }
    out["recon"] = recon

    # ---- swaps
    def verdict(name: str) -> dict:
        z, r, m = cfg[f"{name}/zero"], cfg[f"{name}/rubric"], cfg[f"{name}/mean"]
        kz, kr, km = z["kl_slot"], r["kl_slot"], m["kl_slot"]
        mass_ok = abs(np.mean(r["kw_mass"]) - after_mass.mean()) <= KW_TOL
        ef = 1 - kr / kz if kz > 0 else float("nan")
        efm = 1 - km / kz if kz > 0 else float("nan")
        status = "inert" if kz < INERT_KL else ("preserved" if ef >= EF_MIN and mass_ok else "failed")
        return {
            "status": status,
            "kl_zero": kz,
            "kl_rubric": kr,
            "kl_mean": km,
            "ef_rubric": ef,
            "ef_mean": efm,
            "kw_mass_rubric": float(np.mean(r["kw_mass"])),
            "kw_mass_zero": float(np.mean(z["kw_mass"])),
            "kw_mass_mean": float(np.mean(m["kw_mass"])),
            "mass_within": bool(mass_ok),
            "kl_general_zero": z["kl_general"],
            "kl_general_rubric": r["kl_general"],
        }

    units = {uid: {**verdict(uid), "label": rub[uid]["label"]} for uid in rub}
    out["units"] = units
    out["groups"] = {g: verdict(f"group:{g}") for g in GROUPS}
    out["pass_counts"] = {
        g: {
            s: sum(1 for k, v in units.items() if k.startswith(g + "-") and v["status"] == s)
            for s in ("preserved", "failed", "inert")
        }
        for g in GROUPS
    }

    # ---- causal checks
    gap = after_mass - before_mass
    causal: dict = {}
    for g in GROUPS:
        drop = after_mass - np.array(cfg[f"group:{g}/zero"]["kw_mass"])
        d, lo, hi = boot_mean(drop)
        hd = gen["none"]["hack_rate"] - gen[g]["hack_rate"]
        causal[g] = {
            "kw_mass_drop": d,
            "low": lo,
            "high": hi,
            "gap": float(gap.mean()),
            "share_of_gap": d / float(gap.mean()) if gap.mean() else float("nan"),
            # with no positive before-to-after gap the check cannot be met for a reason that says anything: N/A
            "check1": bool(d >= 0.5 * gap.mean() and lo > 0) if gap.mean() > 0 else None,
            "hack_rate_zeroed": gen[g]["hack_rate"],
            "mean_mentions_zeroed": gen[g]["mean"],
            "hack_drop": hd,
            "check2": bool(hd >= HACK_DROP),
        }
    out["causal"] = causal

    # ---- edges
    edges = []
    claimed_units = [u["unit"] for u in sel["contrasts"][c.PRIMARY]["claimed"]]
    for e in sel["contrasts"][c.PRIMARY]["edges"]:
        cu, cv = f"claimed-{claimed_units.index(e['u'])}", f"claimed-{claimed_units.index(e['v'])}"
        ch_r = cfg[f"{cu}/rubric"]["obs_change"][cv]
        ch_z = cfg[f"{cu}/zero"]["obs_change"][cv]
        edges.append(
            {
                "u": e["u"],
                "v": e["v"],
                "change_rubric": ch_r,
                "change_zero": ch_z,
                "holds": bool(ch_z > 0 and ch_r <= EDGE_RATIO * ch_z),
            }
        )
    out["edges"] = edges

    if (R / "patch.json").exists():
        out["patch"] = c.read_json(R / "patch.json")

    # ---- reading rules
    g = out["groups"]["claimed"]
    pc = out["pass_counts"]["rl_neuron"]
    out["reading"] = {
        "gao_support": bool(
            g["status"] != "preserved"
            or not causal["claimed"]["check2"]
            or gsum[c.PRIMARY]["after"]["mean_unexplained"] > 0.3
        ),
        "goodfire_support": bool(
            pc["preserved"] + pc["inert"] >= 8 and out["groups"]["rl_neuron"]["status"] == "preserved"
        ),
        "circuit_carries_hack": bool(causal["claimed"]["check2"]),
    }
    c.write_json(R / "verdicts.json", out)
    print("step0", out["step0_check"])
    print(
        "unexplained",
        {
            k: (round(v["before"]["mean_unexplained"], 3), round(v["after"]["mean_unexplained"], 3))
            for k, v in gsum.items()
        },
    )
    print("new paths", {k: v["n_new_paths"] for k, v in gsum.items()})
    print("hack-specific", out["n_hack_specific"], "replication overlap", out["replication_overlap"])
    print("pass counts", out["pass_counts"])
    print(
        "groups", {k: (v["status"], round(v["ef_rubric"], 3), round(v["kl_zero"], 4)) for k, v in out["groups"].items()}
    )
    print(
        "causal",
        {
            k: (round(v["kw_mass_drop"], 3), round(v["share_of_gap"], 2), v["hack_rate_zeroed"])
            for k, v in causal.items()
        },
    )
    print("edges", [(e["u"], e["v"], e["holds"]) for e in edges])
    print("reading", out["reading"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
