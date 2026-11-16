"""Final 4 (plan §16): the pre-registered analysis on 180 pairs defined by schedule index (2p-1 ours, 2p control).

python analyse_final4.py   # reads results/final4/trials.csv; writes summary.json, per_variant.csv and
                           # report/figures/per_variant_final4.png
python analyse_final4.py --sensitivity   # the same on sensitivity_11_as_fail/: the 11 harness failures found
                                         # after the first analysis counted as task failures (no figure)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta, binomtest

HERE = Path(__file__).resolve().parent
SENS = "--sensitivity" in sys.argv
OUT = HERE / "results" / "final4" / ("sensitivity_11_as_fail" if SENS else "")
CRIT = ["key_focus", "window_raised", "typing", "pointer", "clipboard", "space", "dialog", "activation"]


def cp(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def newcombe(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """95% CI of the paired difference in passes, in trials (Newcombe 1998, method 10)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else (a * d - b * c) / math.sqrt(den)
    dl = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    du = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return (p1 - p2 - dl) * n, (p1 - p2 + du) * n


def mcnemar(b: int, c: int) -> float:
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def main() -> None:
    rows = {int(r["n"]): r for r in csv.DictReader((OUT / "trials.csv").open())}
    assert len(rows) == 360
    pairs = []
    for p in range(1, 181):
        o, c = rows[2 * p - 1], rows[2 * p]
        assert o["arm"] == "ourdriver" and c["arm"] == "cua-driver" and o["variant"] == c["variant"], p
        pairs.append((o, c))

    def ok(r: dict) -> bool:
        return r["passed"] == "1"

    def calm(r: dict) -> bool:
        return r["undisturbed"] == "True"

    a_k, b_k = sum(ok(o) for o, _ in pairs), sum(ok(c) for _, c in pairs)
    only_o = sum(1 for o, c in pairs if ok(o) and not ok(c))
    only_c = sum(1 for o, c in pairs if ok(c) and not ok(o))
    both = sum(1 for o, c in pairs if ok(o) and ok(c))
    lo, hi = newcombe(both, only_o, only_c, 180 - both - only_o - only_c)
    bg = [(o, c) for o, c in pairs if o["condition"] == "background"]
    u_o, u_c = sum(calm(o) for o, _ in bg), sum(calm(c) for _, c in bg)
    uo_only = sum(1 for o, c in bg if calm(o) and not calm(c))
    uc_only = sum(1 for o, c in bg if calm(c) and not calm(o))
    cost_o = sum(float(o["cost"] or 0) for o, _ in pairs)
    cost_c = sum(float(c["cost"] or 0) for _, c in pairs)

    def split(i: int, cond: str) -> int:
        return sum(1 for pr in pairs if pr[i]["condition"] == cond and ok(pr[i]))

    out: dict = {
        "pairs": 180,
        "ours": {
            "k": a_k,
            "ci": cp(a_k, 180),
            "fg": [split(0, "foreground"), 100],
            "bg": [split(0, "background"), 80],
            "cost": round(cost_o, 2),
            "cost_per_pass": round(cost_o / a_k, 3),
            "bg_undisturbed": [u_o, len(bg)],
        },
        "control": {
            "k": b_k,
            "ci": cp(b_k, 180),
            "fg": [split(1, "foreground"), 100],
            "bg": [split(1, "background"), 80],
            "cost": round(cost_c, 2),
            "cost_per_pass": round(cost_c / b_k, 3),
            "bg_undisturbed": [u_c, len(bg)],
        },
        "mcnemar_passes": {"ours_only": only_o, "control_only": only_c, "p": round(mcnemar(only_o, only_c), 4)},
        "newcombe_95ci_trials": [round(lo, 2), round(hi, 2)],
        "disturbance_mcnemar": {
            "ours_only_undisturbed": uo_only,
            "control_only_undisturbed": uc_only,
            "p": round(mcnemar(uo_only, uc_only), 4),
            "disturbed_trials": [len(bg) - u_o, len(bg) - u_c],
        },
    }
    out["primary_beats"] = bool(a_k > b_k and out["mcnemar_passes"]["p"] < 0.05)
    out["coprimary_a_noninferior"] = bool(lo > -10)
    d = out["disturbance_mcnemar"]
    out["coprimary_b_less_disturbing"] = bool(d["disturbed_trials"][0] < d["disturbed_trials"][1] and d["p"] < 0.05)
    out["coprimary_met"] = out["coprimary_a_noninferior"] and out["coprimary_b_less_disturbing"]
    out["matches_target_rule"] = bool(cp(a_k, 180)[1] >= 0.978)
    per: dict = defaultdict(lambda: [0, 0, 0])
    for o, c in pairs:
        per[o["variant"]][0] += ok(o)
        per[o["variant"]][1] += ok(c)
        per[o["variant"]][2] += 1
    out["per_variant"] = {k: {"ours": v[0], "control": v[1], "pairs": v[2]} for k, v in sorted(per.items())}
    out["a22_held"] = {
        "ours": {k: sum(o[k] == "True" for o, _ in bg) for k in CRIT},
        "control": {k: sum(c[k] == "True" for _, c in bg) for k in CRIT},
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_variant", "a22_held")}, indent=1))
    for k, v in out["per_variant"].items():
        print(f"{k:<22} {v['ours']}/{v['pairs']}  {v['control']}/{v['pairs']}")
    print(out["a22_held"])
    rows_pv = [
        {"variant": k, "pairs": v["pairs"], "ourdriver": v["ours"], "cua-driver": v["control"]}
        for k, v in out["per_variant"].items()
    ]
    with (OUT / "per_variant.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_pv[0]))
        w.writeheader()
        w.writerows(rows_pv)
    if SENS:
        return
    import analyse  # the same figure as the earlier runs

    analyse._figure(rows_pv, "per_variant_final4.png")


if __name__ == "__main__":
    main()
