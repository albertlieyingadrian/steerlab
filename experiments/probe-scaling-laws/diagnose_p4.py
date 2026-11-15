"""Exploratory diagnosis of the cheaper-path test (P4), written after its result was known. Not pre-registered.

    .venv/bin/python diagnose_p4.py      # reads results/ and results/laws.json, writes results/p4_diagnosis.json

1. Mechanism: for each kept case, the joint law's floor for the current model against the target and against the
   measured error at 31,623 labels; and, per distinct (task, probe, current, target), labels needed to reach the
   target at every size, measured against the law (figure p4_need.png).
2. Cost model: the paper's cost is parameters x labels (Section 5.2), as here; the measured answers are recounted with
   labels-only and parameters-only costs to show how much the choice depends on the cost.
3. Noise: each kept case's measured answer under 2,000 bootstrap resamples of the seeds in every cell.
4. A second family is in results/q3-*/ (Qwen3 base, Truth only), reported by this script if present.
5. One fix, fixed before it was fitted: a label law whose halfway point and steepness both depend on size,
   x = g(N) * (log D + beta log N - log B), g(N) = g0 + g1 log N, fitted by the P4 protocol (models up to the current
   one, cells with D <= labels at hand) and scored on the same 21 cases. Also P4 with the smaller models' cells at every
   D (only the current model limited to the labels at hand), a reading of the paper's protocol.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fit_laws import G_FREE, ORDER, _best_fit, e_at, fit_joint, law, load, need_law, need_measured, wilson

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, GREY, INK = "#C9561C", "#9A948C", "#1d272a"
MAIN = ORDER[:5]


# ---------------------------------------------------------------- 5. the size-dependent law


def law_sd(N, D, A, a, B, g0, g1, beta):
    N = np.asarray(N, float)
    floor = A * N**-a
    g = np.clip(g0 + g1 * np.log(N), 0.05, 5.0)
    x = g * (np.log(np.asarray(D, float)) + beta * np.log(N) - np.log(B))
    return floor + (0.5 - floor) / (1.0 + np.exp(x))


def fit_sd(N, D, E, seed=0, restarts=12):
    rng = np.random.default_rng(seed)
    lo, hi = [-8, -1.0, -5, 0.05, -2.0, -1.0], [3, 3.0, 25, 3.0, 2.0, 3.0]
    best = None
    for _ in range(restarts):
        x0 = [
            rng.uniform(-3, 0),
            rng.uniform(0, 0.8),
            rng.uniform(3, 10),
            rng.uniform(0.4, 1.2),
            rng.uniform(-0.3, 0.3),
            rng.uniform(0.0, 1.0),
        ]
        r = least_squares(lambda p: law_sd(N, D, math.exp(p[0]), p[1], math.exp(p[2]), *p[3:]) - E, x0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    return dict(A=math.exp(p[0]), alpha=p[1], B=math.exp(p[2]), g0=p[3], g1=p[4], beta=p[5])


def need_sd(prm, N, T):
    def e(d):
        return float(law_sd(N, d, prm["A"], prm["alpha"], prm["B"], prm["g0"], prm["g1"], prm["beta"]))

    if e(32) <= T:
        return 32.0
    if e(1e7) > T:
        return math.inf
    lo, hi = math.log(32), math.log(1e7)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if e(math.exp(mid)) <= T else (mid, hi)
    return math.exp(hi)


def stack_cells(s, models, h, cur_only_limited=False):
    N, D, E = [], [], []
    for m in models:
        c = s[m]
        keep = (c["D"] <= h) if (not cur_only_limited or m == models[-1]) else np.ones(len(c["D"]), bool)
        N += [c["N"]] * int(keep.sum())
        D += list(c["D"][keep])
        E += list(c["E"][keep])
    return np.array(N), np.array(D, float), np.array(E)


def pick(nc, nn, Nc, Nn, cost=lambda N, d: N * d):
    if not math.isfinite(nc):
        return "larger"
    if not math.isfinite(nn):
        return "labels"
    return "larger" if cost(Nn, nn) < cost(Nc, nc) else "labels"


# ---------------------------------------------------------------- 3. bootstrap over seeds


def boot_truth(cells, case, n_boot=2000, seed=0):
    """Share of seed resamples in which the measured cheaper option is 'labels'."""
    rng = np.random.default_rng(seed)
    s = cells[(case["task"], case["probe"])]
    cur, nxt, T = s[case["current"]], s[case["next"]], case["T"]
    out = []
    for _ in range(n_boot):
        curves = []
        for c in (cur, nxt):
            E = np.array([1 - np.mean(rng.choice(c["seeds"][d], len(c["seeds"][d]))) for d in c["D"]])
            curves.append({"D": c["D"], "E": E})
        mc, mn = need_measured(curves[0], T), need_measured(curves[1], T)
        out.append(pick(mc, mn, cur["N"], nxt["N"]) == "labels")
    return float(np.mean(out))


def main():
    cells = load(R)
    laws = json.loads((R / "laws.json").read_text())
    cases = laws["P4"]["cases"]
    res = {"note": "exploratory, written after the P4 result; not pre-registered"}

    # 1. mechanism
    mech = []
    for c in cases:
        s = cells[(c["task"], c["probe"])]
        fit_ms = MAIN[: MAIN.index(c["current"]) + 1]
        lp = fit_joint(s, fit_ms, G_FREE[c["probe"]], d_max=c["H"])
        Nc = s[c["current"]]["N"]
        mech.append(
            dict(
                task=c["task"],
                probe=c["probe"],
                current=c["current"],
                H=c["H"],
                T=c["T"],
                law_floor_current=lp["A"] * Nc ** -lp["alpha"],
                law_at_dmax_current=float(law(Nc, 31623, lp["A"], lp["alpha"], lp["B"], lp["g"])),
                measured_at_dmax_current=e_at(s[c["current"]], 31623),
                truth=c["truth"],
            )
        )
    n_floor_above = sum(m["law_floor_current"] > m["T"] for m in mech)
    over = [m["law_floor_current"] - m["measured_at_dmax_current"] for m in mech]
    res["mechanism"] = {
        "cases": mech,
        "law_floor_above_target": n_floor_above,
        "n": len(mech),
        "median_floor_minus_measured": float(np.median(over)),
    }
    print(
        f"1. law floor for the current model above the target in {n_floor_above}/{len(mech)} cases; median law floor"
        f" minus measured error at 31,623 = {np.median(over):+.3f}"
    )
    for m in mech:
        print(
            f"   {m['task']:10s} {m['probe']:9s} {m['current']:3s} H={m['H']:>5} T={m['T']:.2f}  law floor "
            f"{m['law_floor_current']:.3f}  law@31623 {m['law_at_dmax_current']:.3f}  measured@31623 "
            f"{m['measured_at_dmax_current']:.3f}  truth {m['truth']}"
        )

    # labels needed vs size, per distinct combination
    combos = sorted({(c["task"], c["probe"], c["current"], c["T"]) for c in cases})
    need_rows = []
    for t, p, cur, T in combos:
        s = cells[(t, p)]
        h = max(c["H"] for c in cases if (c["task"], c["probe"], c["current"], c["T"]) == (t, p, cur, T))
        lp = fit_joint(s, MAIN[: MAIN.index(cur) + 1], G_FREE[p], d_max=h)
        lp_all = fit_joint(s, MAIN, G_FREE[p])
        row = dict(
            task=t,
            probe=p,
            current=cur,
            T=T,
            H=h,
            N=[s[m]["N"] for m in MAIN],
            measured=[need_measured(s[m], T) for m in MAIN],
            law_pred=[need_law(lp, s[m]["N"], T) for m in MAIN],
            law_all_cells=[need_law(lp_all, s[m]["N"], T) for m in MAIN],
        )
        meas = row["measured"]
        row["measured_rises"] = sum(
            1 for a, b in itertools.pairwise(meas) if math.isfinite(a) and math.isfinite(b) and b > a
        )
        need_rows.append(row)
        print(
            f"   need {t:10s} {p:9s} T={T:.2f}: measured "
            + " ".join(f"{v:>7.0f}" for v in meas)
            + " | law(P4 fit) "
            + " ".join(f"{v:>7.0f}" for v in row["law_pred"])
            + " | law(all cells) "
            + " ".join(f"{v:>7.0f}" for v in row["law_all_cells"])
        )
    res["need_by_size"] = need_rows
    nm = sum(r["measured_rises"] > 0 for r in need_rows)
    print(f"   measured labels-needed rises with size at least once in {nm}/{len(need_rows)} combinations")

    # 2. cost model
    costs = {"params_x_labels": lambda N, d: N * d, "labels_only": lambda N, d: d, "params_only": lambda N, d: N}
    res["cost_model"] = {
        "paper": "parameters (billions) x labels, Section 5.2",
        "ours": "parameters (billions) x labels",
    }
    for name, cf in costs.items():
        k = 0
        for c in cases:
            s = cells[(c["task"], c["probe"])]
            k += pick(*c["measured_labels"], s[c["current"]]["N"], s[c["next"]]["N"], cf) == "labels"
        res["cost_model"][name] = {"measured_labels_cases": k}
    print(
        f"2. cost = N x labels, as in the paper. Measured 'labels' cases: N x labels "
        f"{res['cost_model']['params_x_labels']['measured_labels_cases']}/21, labels only "
        f"{res['cost_model']['labels_only']['measured_labels_cases']}/21"
    )

    # 3. bootstrap
    boots = [boot_truth(cells, c) for c in cases]
    lab = [b for c, b in zip(cases, boots, strict=True) if c["truth"] == "labels"]
    lar = [b for c, b in zip(cases, boots, strict=True) if c["truth"] == "larger"]
    res["bootstrap"] = {
        "share_labels": boots,
        "labels_cases": len(lab),
        "labels_robust_95": sum(b >= 0.95 for b in lab),
        "labels_robust_80": sum(b >= 0.8 for b in lab),
        "larger_cases": len(lar),
        "larger_robust_95": sum(b <= 0.05 for b in lar),
    }
    print(
        f"3. of {len(lab)} measured 'labels' cases, {res['bootstrap']['labels_robust_95']} hold in >= 95% of seed "
        f"resamples and {res['bootstrap']['labels_robust_80']} in >= 80%; 'larger' cases robust at 95%: "
        f"{res['bootstrap']['larger_robust_95']}/{len(lar)}"
    )
    for c, b in zip(cases, boots, strict=True):
        print(
            f"   {c['task']:10s} {c['probe']:9s} {c['current']}->{c['next']} H={c['H']:>5} T={c['T']:.2f} truth "
            f"{c['truth']:6s} P(labels) {b:.2f}"
        )

    # 5. size-dependent law, and the all-D reading of the protocol
    picks_sd, picks_alld, sd_params = [], [], []
    for c in cases:
        s = cells[(c["task"], c["probe"])]
        fit_ms = MAIN[: MAIN.index(c["current"]) + 1]
        Nc, Nn = s[c["current"]]["N"], s[c["next"]]["N"]
        prm = fit_sd(*stack_cells(s, fit_ms, c["H"]))
        sd_params.append(prm)
        picks_sd.append(pick(need_sd(prm, Nc, c["T"]), need_sd(prm, Nn, c["T"]), Nc, Nn))
        N2, D2, E2 = stack_cells(s, fit_ms, c["H"], cur_only_limited=True)
        g_free = G_FREE[c["probe"]]

        def resid(p, N2=N2, D2=D2, E2=E2, g_free=g_free):
            return law(N2, D2, math.exp(p[0]), p[1], math.exp(p[2]), p[3] if g_free else 1.0) - E2

        x = _best_fit(
            resid, [[-2, 0.3, 6, 0.8], [-1, 0.2, 4, 0.6], [-3, 0.5, 8, 1.0]], [-8, -1, -5, 0.1], [3, 3, 25, 3.0]
        )
        lp = {"A": math.exp(x[0]), "alpha": x[1], "B": math.exp(x[2]), "g": x[3] if g_free else 1.0}
        picks_alld.append(pick(need_law(lp, Nc, c["T"]), need_law(lp, Nn, c["T"]), Nc, Nn))
    res["size_dependent_params"] = sd_params
    for name, picks in (("size_dependent_law", picks_sd), ("all_D_reading", picks_alld)):
        k = sum(p == c["truth"] for p, c in zip(picks, cases, strict=True))
        res[name] = {
            "k": k,
            "n": len(cases),
            "ci": list(wilson(k, len(cases))),
            "picks_labels": sum(p == "labels" for p in picks),
            "picks": picks,
        }
        print(
            f"5. {name}: right in {k}/{len(cases)} {np.round(wilson(k, len(cases)), 2).tolist()}, picks 'labels' "
            f"{res[name]['picks_labels']} times"
        )

    # 4. second family
    fam = {}
    for key in ("q3-1.7b", "q3-4b", "q3-8b"):
        for probe in ("lasttoken", "alllayer"):
            g = R / key / f"truth_{probe}.json"
            if g.exists():
                r = json.loads(g.read_text())
                by_d = {}
                for cc in r["cells"]:
                    by_d.setdefault(cc["D"], []).append(cc["auroc"])
                ds = sorted(by_d)
                E = {"D": np.array(ds), "E": np.array([1 - np.mean(by_d[d]) for d in ds])}
                fam.setdefault(probe, {})[key] = dict(
                    N=r["n_params_b"],
                    E=E["E"].tolist(),
                    D=ds,
                    need_035=need_measured(E, 0.35),
                    need_030=need_measured(E, 0.30),
                )
    res["second_family"] = fam
    for probe, d in fam.items():
        print(
            f"4. Qwen3 base, Truth, {probe}: "
            + "; ".join(
                f"{k} N={v['N']:.2f} E@31623={v['E'][-1]:.3f} labels to 0.35: {v['need_035']:.0f}, to 0.30: "
                f"{v['need_030']:.0f}"
                for k, v in d.items()
            )
        )

    # 4a. extended label axis (Truth, last-token): original cells up to 10,000 labels plus 31,623 and 100,000 from the
    # truthbig task (same test set, 50,000 training pairs); P4 rerun on that setting with 31,623 labels at hand too
    import fit_laws

    big = {}
    for m in MAIN:
        f = R / m / "truthbig_lasttoken.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        by_d = {}
        for cc in r["cells"]:
            by_d.setdefault(cc["D"], []).append(cc["auroc"])
        o = cells[("truth", "lasttoken")][m]
        seeds = {d: v for d, v in o["seeds"].items() if d <= 10000} | by_d
        ds = np.array(sorted(seeds))
        big[m] = {"N": o["N"], "D": ds, "E": np.array([1 - np.mean(seeds[d]) for d in ds]), "seeds": seeds}
    if len(big) == len(MAIN):
        ext = {
            "E_31623_original": [e_at(cells[("truth", "lasttoken")][m], 31623) for m in MAIN],
            "E_31623_big": [e_at(big[m], 31623) for m in MAIN],
            "E_100000": [e_at(big[m], 100000) for m in MAIN],
        }
        eq1_100k = np.polyfit(np.log([big[m]["N"] for m in MAIN]), np.log(ext["E_100000"]), 1)
        ext["alpha_at_100k"] = float(-eq1_100k[0])
        saved = fit_laws.AT_HAND
        fit_laws.AT_HAND = [316, 1000, 3162, 10000, 31623]
        p4x = fit_laws.p4({("truth", "lasttoken"): big})
        fit_laws.AT_HAND = saved
        ext["P4"] = {k: {kk: vv for kk, vv in v.items()} for k, v in p4x["summary"].items()}
        ext["P4_cases"] = p4x["cases"]
        res["extended_axis"] = ext
        print("4a. Truth, last-token, error at 31,623 (original / 50k-pair pool) and at 100,000 labels:")
        for m, a, b, c in zip(MAIN, ext["E_31623_original"], ext["E_31623_big"], ext["E_100000"], strict=True):
            print(f"   {m:4s} {a:.3f} / {b:.3f} -> {c:.3f}")
        print(f"   size-law alpha at 100,000 labels {ext['alpha_at_100k']:.3f} (at 31,623: 0.111)")
        print("   P4 on this setting with 31,623 at hand:", {k: (v["k"], v["n"]) for k, v in ext["P4"].items()})
        for c in p4x["cases"]:
            print(
                f"   {c['current']}->{c['next']} H={c['H']:>5} T={c['T']:.2f} truth {c['truth']:6s} laws "
                f"{c['pick_laws']:6s} need {[round(x) if x < 1e9 else 'inf' for x in c['need_laws']]} measured "
                f"{[round(x) for x in c['measured_labels']]}"
            )

    # figure: labels needed vs size
    ncol = 3
    nrow = -(-len(need_rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.6 * nrow), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(need_rows) :]:
        ax.axis("off")
    for ax, r in zip(axes, need_rows, strict=False):
        N = np.array(r["N"])
        for key, colour, mk, label in (("measured", INK, "o", "measured"), ("law_pred", EMBER, "s", "law (P4 fit)")):
            y = np.array([v if math.isfinite(v) else np.nan for v in r[key]], float)
            ax.plot(N, y, mk + "-", color=colour, ms=4, lw=1, label=label)
            for x_, v in zip(N, r[key], strict=True):
                if not math.isfinite(v):
                    ax.plot(x_, 6e4, "^", color=colour, ms=5, mfc="none")
        ax.axvline(r["N"][MAIN.index(r["current"])], color=GREY, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(20, 9e4)
        ax.set_title(
            f"{r['task'].title()}, {r['probe']}\ntarget {r['T']:.2f}, current {r['current'].upper()}", fontsize=8
        )
        ax.set_xlabel("parameters (B)", fontsize=8)
    for i in range(0, len(axes), ncol):
        axes[i].set_ylabel("labels to reach\nthe target")
    axes[0].plot([], [], "^", color=GREY, mfc="none", label="out of reach")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "p4_need.png", dpi=200)
    plt.close(fig)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (R / "p4_diagnosis.json").write_text(json.dumps(res, indent=1, default=default))
    print("wrote results/p4_diagnosis.json and report/figures/p4_need.png")


if __name__ == "__main__":
    main()
