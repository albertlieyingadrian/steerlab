"""Every number in the note, from results/trials.csv (no API key needed).

    python analyse.py            # final 1: results/summary.json, results/per_variant.csv, figures/per_variant.png
    python analyse.py --final2   # final 2: the same from results/final2/trials.csv, plus the secondary rule

Pairs are (variant, run) trials that both arms finished; the pre-registered headline uses pairs. Harness failures
are never counted as task failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

from scipy.stats import beta, binom, binomtest

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "report" / "figures"
ARMS = ["ourdriver", "cua-driver"]
VERSIONS = {"final4": "ourdriver 0.8.4", "final3": "ourdriver 0.7.0", "final2": "ourdriver 0.6.2"}
A22 = ["key_focus", "window_raised", "typing", "pointer", "clipboard", "space", "dialog", "activation"]


def cp(k: int, n: int) -> list[float]:
    """95% Clopper-Pearson interval."""
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def mcnemar_p(b: int, c: int) -> float:
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def mcnemar_power(n_pairs: int, p_b: float, p_c: float, alpha: float = 0.05) -> float:
    """Power of the exact two-sided McNemar test at n_pairs pairs, when a pair is discordant one way with
    probability p_b and the other way with p_c (summed exactly over the number of discordant pairs)."""
    pd = p_b + p_c
    if pd == 0 or n_pairs == 0:
        return 0.0
    q = p_b / pd
    power = 0.0
    for d in range(1, n_pairs + 1):
        w = binom.pmf(d, n_pairs, pd)
        if w < 1e-12:
            continue
        power += w * sum(binom.pmf(k, d, q) for k in range(d + 1) if mcnemar_p(k, d - k) < alpha)
    return round(float(power), 3)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """95% CI of p1 - p2 for paired proportions (Newcombe 1998, method 10). a = both pass, b = only arm 1,
    c = only arm 2, d = neither."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    diff = p1 - p2
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if denom == 0 else (a * d - b * c) / math.sqrt(denom)
    dl = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    du = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return diff - dl, diff + du


def main() -> None:
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--final2", action="store_true")
    final2 = ap.parse_args().final2
    if final2:  # FINAL=3 reads final 3 (ours 0.7.0, paired with the final-2 control)
        RESULTS = HERE / "results" / ("final3" if os.environ.get("FINAL") == "3" else "final2")
    rows = [r for r in csv.DictReader((RESULTS / "trials.csv").open()) if r["outcome"] not in ("harness", "no_result")]
    by = {a: {(r["variant"], int(r["run"])): r for r in rows if r["arm"] == a} for a in ARMS}
    pairs = sorted(set(by[ARMS[0]]) & set(by[ARMS[1]]))
    out: dict = {"pairs": len(pairs), "arms": {}}
    for a in ARMS:
        rs = [by[a][k] for k in pairs]
        k, n = sum(int(r["passed"]) for r in rs), len(rs)
        fg = [r for r in rs if r["condition"] == "foreground"]
        bg = [r for r in rs if r["condition"] == "background"]
        cost = sum(float(r["cost"] or 0) for r in rs)
        und = [r for r in bg if r["undisturbed"] in ("True", "False")]
        out["arms"][a] = {
            "k": k,
            "n": n,
            "ci": cp(k, n),
            "fg": [sum(int(r["passed"]) for r in fg), len(fg)],
            "bg": [sum(int(r["passed"]) for r in bg), len(bg)],
            "cost": round(cost, 2),
            "cost_per_pass": round(cost / k, 2) if k else None,
            "median_s": statistics.median(int(r["seconds"]) for r in rs) if rs else None,
            "gui_only": [sum(r["gui_only"] == "1" for r in rs), sum(r["gui_only"] in ("0", "1") for r in rs)],
            "undisturbed": [sum(r["undisturbed"] == "True" for r in und), len(und)],
            "a22_held": {c: sum(r[c] == "True" for r in und) for c in A22},
            "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
            "all_finished_trials": [sum(int(r["passed"]) for r in by[a].values()), len(by[a])],
            "all_cost": round(sum(float(r["cost"] or 0) for r in by[a].values()), 2),
        }
    A, B = (by[a] for a in ARMS)
    b = sum(1 for k in pairs if A[k]["passed"] == "1" and B[k]["passed"] == "0")
    c = sum(1 for k in pairs if A[k]["passed"] == "0" and B[k]["passed"] == "1")
    n = len(pairs)
    out["mcnemar"] = {
        "ours_only": b,
        "control_only": c,
        "p_two_sided": round(mcnemar_p(b, c), 4),
        # Power at this many pairs to detect a gap the size of the target's over cua-driver 0.47.10 (88/90 against
        # 62/90): 26 of 90 pairs discordant in the better driver's favour, none against.
        "power_target_sized_gap": mcnemar_power(n, 26 / 90, 0.0),
        "power_at_observed_rates": mcnemar_power(n, b / n, c / n) if n else 0.0,
    }
    out["matches_target_rule_met"] = out["arms"]["ourdriver"]["ci"][1] >= 0.978
    both = sum(1 for k in pairs if A[k]["passed"] == "1" and B[k]["passed"] == "1")
    neither = n - both - b - c
    lo, hi = newcombe_paired(both, b, c, neither)
    oa, ca = out["arms"]["ourdriver"], out["arms"]["cua-driver"]
    raised = {
        name: arm["undisturbed"][1] - arm["a22_held"]["window_raised"]
        for name, arm in (("ourdriver", oa), ("cua-driver", ca))
    }
    out["primary_beats_met"] = oa["k"] > ca["k"] and out["mcnemar"]["p_two_sided"] < 0.05
    out["secondary"] = {
        "paired_diff_trials": oa["k"] - ca["k"],
        "newcombe_95ci_trials": [round(lo * n, 2), round(hi * n, 2)],
        "lower_bound_above_minus5": lo * n > -5,
        "focus_kept": [oa["a22_held"]["key_focus"], ca["a22_held"]["key_focus"]],
        "focus_kept_ge_control": oa["a22_held"]["key_focus"] >= ca["a22_held"]["key_focus"],
        "windows_raised": [raised["ourdriver"], raised["cua-driver"]],
        "raised_lt_control": raised["ourdriver"] < raised["cua-driver"],
    }
    sec = out["secondary"]
    out["secondary_met"] = (
        (not out["primary_beats_met"])
        and sec["lower_bound_above_minus5"]
        and sec["focus_kept_ge_control"]
        and sec["raised_lt_control"]
    )
    variants = sorted({v for v, _ in pairs})
    per = []
    for v in variants:
        ks = [kk for kk in pairs if kk[0] == v]
        per.append(
            {
                "variant": v,
                "pairs": len(ks),
                "ourdriver": sum(A[kk]["passed"] == "1" for kk in ks),
                "cua-driver": sum(B[kk]["passed"] == "1" for kk in ks),
            }
        )
    with (RESULTS / "per_variant.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    for r in per:
        print(f"{r['variant']:<22} {r['ourdriver']}/{r['pairs']}  {r['cua-driver']}/{r['pairs']}")
    run3 = os.environ.get("FINAL") == "3"
    name = ("per_variant_final3.png" if run3 else "per_variant_final2.png") if final2 else "per_variant.png"
    _figure(per, name)


def _figure(per: list[dict], name: str = "per_variant.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    names = [r["variant"].upper().replace("-FOREGROUND", " fg").replace("-BACKGROUND", " bg") for r in per]
    y = list(range(len(per)))
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(per) + 1.2), dpi=200)
    ax.barh(
        [i - 0.2 for i in y],
        [r["ourdriver"] / r["pairs"] for r in per],
        height=0.38,
        color="#d9622b",
        label=next((v for k, v in VERSIONS.items() if k in name), "ourdriver 0.5.1"),
    )
    ax.barh(
        [i + 0.2 for i in y],
        [r["cua-driver"] / r["pairs"] for r in per],
        height=0.38,
        color="#9aa3ad",
        label="cua-driver 0.47.4 (control)",
    )
    for i, r in enumerate(per):
        ax.text(1.03, i, f"{r['ourdriver']}/{r['pairs']} vs {r['cua-driver']}/{r['pairs']}", va="center", fontsize=7)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Share of paired trials passed (Claude Opus 5.5)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / name)
    print("wrote", FIG / name)


if __name__ == "__main__":
    main()
