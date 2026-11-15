"""Every number in the note, from results/trials.csv (no API key needed).

    python analyse.py            # writes results/summary.json, results/per_variant.csv, report/figures/per_variant.png

Pairs are (variant, run) trials that both arms finished; the pre-registered headline uses pairs. Harness failures
are never counted as task failures.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from scipy.stats import beta, binom, binomtest

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "report" / "figures"
ARMS = ["ourdriver", "cua-driver"]
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


def main() -> None:
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
        # Power at this many pairs to detect a gap the size of Aside's over cua-driver 0.47.10 (88/90 against
        # 62/90): 26 of 90 pairs discordant in the better driver's favour, none against.
        "power_aside_sized_gap": mcnemar_power(n, 26 / 90, 0.0),
        "power_at_observed_rates": mcnemar_power(n, b / n, c / n) if n else 0.0,
    }
    out["matches_aside_rule_met"] = out["arms"]["ourdriver"]["ci"][1] >= 0.978
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
    _figure(per)


def _figure(per: list[dict]) -> None:
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
        label="ourdriver 0.5.1",
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
    fig.savefig(FIG / "per_variant.png")
    print("wrote", FIG / "per_variant.png")


if __name__ == "__main__":
    main()
