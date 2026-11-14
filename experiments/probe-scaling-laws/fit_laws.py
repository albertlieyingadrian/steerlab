"""Fit the paper's two laws to the probe grid and run the pre-registered tests (PREREGISTRATION.md).

    .venv/bin/python fit_laws.py                 # reads results/<model>/<task>_<probe>.json, writes results/laws.json

P1 size law, P2 label law, P3 one-size-up extrapolation (+ baselines), P4 cheaper-path choice (+ Chinchilla and Kaplan
forms), P5 probe choice, and the exploratory variants V1-V5. Every number the note quotes is printed here.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ORDER = ["0.5b", "1.5b", "3b", "7b", "14b", "32b"]
TASKS = ["truth", "sycophancy", "policy"]
PROBES = ["lasttoken", "alllayer"]
D_MAX = 31623
G_FREE = {"lasttoken": True, "alllayer": False}  # primary assignment: Eq. 3 for last-token, Eq. 2 for all-layer
TARGETS = [0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
AT_HAND = [316, 1000, 3162, 10000]


# ---------------------------------------------------------------- data


def load(results_dir):
    """cells[(task, probe)][model] = dict(N, D: np.array, E: np.array (1 - mean AUROC), seeds: {D: [auroc..]})."""
    cells = {}
    for f in sorted(Path(results_dir).glob("*/*_*.json")):
        r = json.loads(f.read_text())
        by_d = {}
        for c in r["cells"]:
            by_d.setdefault(c["D"], []).append(c["auroc"])
        ds = np.array(sorted(by_d))
        cells.setdefault((r["task"], r["probe"]), {})[r["key"]] = {
            "N": r["n_params_b"],
            "D": ds,
            "E": np.array([1 - np.mean(by_d[d]) for d in ds]),
            "seeds": by_d,
        }
    return cells


def models_in(setting):
    return [m for m in ORDER if m in setting]


# ---------------------------------------------------------------- the laws


def hill(D, N, B, g):
    """Fraction of the chance-to-floor gap left after D labels: B^g / (B^g + (D sqrt N)^g)."""
    x = np.log(np.asarray(D, float) * np.sqrt(N)) - np.log(B)
    return 1.0 / (1.0 + np.exp(g * x))


def law(N, D, A, alpha, B, g, E0=0.0):
    floor = E0 + A * np.asarray(N, float) ** -alpha
    return floor + (0.5 - floor) * hill(D, N, B, g)


def fit_eq1(Ns, Es, logit=False):
    """Least squares on log E (or logit E) against log N. Returns dict(A, alpha, r2, max_dev, pred(N))."""
    x = np.log(Ns)
    y = np.log(Es / (1 - Es)) if logit else np.log(Es)
    slope, icept = np.polyfit(x, y, 1)
    yhat = icept + slope * x
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yhat) ** 2).sum() / ss if ss > 0 else float("nan")

    def pred(N):
        z = icept + slope * np.log(N)
        return 1 / (1 + np.exp(-z)) if logit else np.exp(z)

    return {
        "A": float(np.exp(icept)),
        "alpha": float(-slope),
        "r2": float(r2),
        "max_dev": float(np.max(np.abs(pred(np.asarray(Ns)) - Es))),
        "pred": pred,
        "logit": logit,
    }


def stack(setting, models, d_max=None):
    N, D, E = [], [], []
    for m in models:
        c = setting[m]
        keep = c["D"] <= (d_max or 1e12)
        N += [c["N"]] * int(keep.sum())
        D += list(c["D"][keep])
        E += list(c["E"][keep])
    return np.array(N), np.array(D, float), np.array(E)


def fit_label(setting, models, eq1, g_free, logit=False):
    """Hold the floor from Eq. 1, fit B (and g) by least squares on E (or logit E) over every cell."""
    N, D, E = stack(setting, models)
    floor = eq1["pred"](N)
    npar = 2 if g_free else 1

    def resid(p):
        B, g = math.exp(p[0]), (p[1] if g_free else 1.0)
        e = floor + (0.5 - floor) * hill(D, N, B, g)
        if logit:
            return np.log(e / (1 - e)) - np.log(E / (1 - E))
        return e - E

    best = None
    for lb0 in (np.log(100), np.log(1000), np.log(10000)):
        r = least_squares(resid, [lb0, 0.7][:npar], bounds=([-5, 0.1][:npar], [25, 3.0][:npar]))
        if best is None or r.cost < best.cost:
            best = r
    B, g = math.exp(best.x[0]), (float(best.x[1]) if g_free else 1.0)
    pred_cells = floor + (0.5 - floor) * hill(D, N, B, g)
    return {
        "B": B,
        "g": g,
        "rms": float(np.sqrt(np.mean((pred_cells - E) ** 2))),
        "max_miss": float(np.max(np.abs(pred_cells - E))),
        "n_cells": len(E),
    }


def curve(eq1, lab, N, D):
    F = eq1["pred"](N)
    return F + (0.5 - F) * hill(D, N, lab["B"], lab["g"])


def fit_joint(setting, models, g_free, d_max=None, E0_free=False, restarts=8, seed=0):
    """A, alpha, B, g (and E0) jointly by least squares on E over every cell with D <= d_max."""
    N, D, E = stack(setting, models, d_max)
    rng = np.random.default_rng(seed)
    lo = [-8, -1.0, -5, 0.1, 0.0]
    hi = [3, 3.0, 25, 3.0, max(1e-6, float(E.min()))]
    start_lo = [-3, 0.0, 3, 0.4, 0.0]
    start_hi = [0, 0.8, 10, 1.2, float(E.min()) / 2]
    idx = [0, 1, 2] + ([3] if g_free else []) + ([4] if E0_free else [])

    def unpack(p):
        q = dict(zip(idx, p, strict=False))
        return math.exp(q[0]), q[1], math.exp(q[2]), q.get(3, 1.0), q.get(4, 0.0)

    def resid(p):
        A, a, B, g, E0 = unpack(p)
        return law(N, D, A, a, B, g, E0) - E

    best = None
    for _ in range(restarts):
        x0 = [min(max(rng.uniform(start_lo[i], start_hi[i]), lo[i] + 1e-6), hi[i] - 1e-6) for i in idx]
        r = least_squares(resid, x0, bounds=([lo[i] for i in idx], [hi[i] for i in idx]))
        if best is None or r.cost < best.cost:
            best = r
    A, a, B, g, E0 = unpack(best.x)
    return {"A": A, "alpha": a, "B": B, "g": g, "E0": E0, "rms": float(np.sqrt(2 * best.cost / len(E)))}


def fit_pooled(cells, probe, models_by_task, g_free, seed=0):
    """V3: one alpha and one g shared by the tasks, task-specific A and B, jointly on every cell."""
    tasks = list(models_by_task)
    data = [stack(cells[(t, probe)], models_by_task[t]) for t in tasks]
    nt = len(tasks)
    off = 2 if g_free else 1

    def unpack(p):
        return p[0], (p[1] if g_free else 1.0), np.exp(p[off : off + nt]), np.exp(p[off + nt : off + 2 * nt])

    def resid(p):
        a, g, As, Bs = unpack(p)
        return np.concatenate([law(N, D, As[i], a, Bs[i], g) - E for i, (N, D, E) in enumerate(data)])

    lo = [-1.0] + ([0.1] if g_free else []) + [-8] * nt + [-5] * nt
    hi = [3.0] + ([3.0] if g_free else []) + [3] * nt + [25] * nt
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(8):
        x0 = (
            [rng.uniform(0, 0.6)]
            + ([rng.uniform(0.4, 1.2)] if g_free else [])
            + list(rng.uniform(-3, 0, nt))
            + list(rng.uniform(3, 10, nt))
        )
        r = least_squares(resid, x0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    a, g, As, Bs = unpack(best.x)
    return {
        t: {"A": float(As[i]), "alpha": float(a), "B": float(Bs[i]), "g": float(g), "E0": 0.0}
        for i, t in enumerate(tasks)
    }


# ---------------------------------------------------------------- baselines for the cheaper-path choice


def _best_fit(resid, starts, lo, hi):
    best = None
    for x0 in starts:
        try:
            with np.errstate(all="ignore"):
                r = least_squares(resid, x0, bounds=(lo, hi))
        except (ValueError, OverflowError, FloatingPointError):
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    return best.x


def fit_additive(N, D, E, seed=0, restarts=20):
    """Eq. 4: E = E_inf + a N^-alpha + b D^-beta."""
    rng = np.random.default_rng(seed)

    def f(p, N_, D_):
        return p[0] + math.exp(p[1]) * N_ ** -p[2] + math.exp(p[3]) * D_ ** -p[4]

    starts = [
        [rng.uniform(0, 0.1), rng.uniform(-4, 0), rng.uniform(0, 1), rng.uniform(-2, 3), rng.uniform(0, 1)]
        for _ in range(restarts)
    ]
    p = _best_fit(lambda p: f(p, N, D) - E, starts, [0, -10, 0, -10, 0], [0.5, 5, 3, 10, 3])

    def need(N_, T):
        rest = T - p[0] - math.exp(p[1]) * N_ ** -p[2]
        if rest <= 0 or p[4] <= 1e-6:
            return math.inf
        return max(32.0, (math.exp(p[3]) / rest) ** (1 / p[4]))

    return {"params": [float(v) for v in p], "need": need}


def fit_kaplan(N, D, E, seed=0, restarts=20):
    """Eq. 5: E = E_inf + [(Nc/N)^(aN/aD) + Dc/D]^aD."""
    rng = np.random.default_rng(seed)

    def f(p, N_, D_):
        e_inf, lnc, an, ldc, ad = p
        return e_inf + ((math.exp(lnc) / N_) ** (an / ad) + math.exp(ldc) / D_) ** ad

    starts = [
        [rng.uniform(0, 0.1), rng.uniform(-10, 0), rng.uniform(0.05, 1), rng.uniform(0, 10), rng.uniform(0.05, 1)]
        for _ in range(restarts)
    ]
    p = _best_fit(lambda p: f(p, N, D) - E, starts, [0, -15, 0.01, -10, 0.01], [0.5, 15, 3, 20, 3])

    def need(N_, T):
        e_inf, lnc, an, ldc, ad = p
        if e_inf >= T:
            return math.inf
        rest = (T - e_inf) ** (1 / ad) - (math.exp(lnc) / N_) ** (an / ad)
        if rest <= 0:
            return math.inf
        return max(32.0, math.exp(ldc) / rest)

    return {"params": [float(v) for v in p], "need": need}


def need_law(params, N, T):
    """Labels for the law to reach T at size N, by bisection on log D in [32, 1e7]; inf if out of reach."""
    A, a, B, g, E0 = params["A"], params["alpha"], params["B"], params["g"], params.get("E0", 0.0)

    def e(d):
        return float(law(N, d, A, a, B, g, E0))

    if e(32) <= T:
        return 32.0
    if e(1e7) > T:
        return math.inf
    lo, hi = math.log(32), math.log(1e7)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if e(math.exp(mid)) <= T else (mid, hi)
    return math.exp(hi)


def need_measured(c, T):
    """Smallest D where the measured curve reaches T, linear in log D between grid points; inf if never."""
    D, E = c["D"], c["E"]
    if E[0] <= T:
        return float(D[0])
    for i in range(1, len(D)):
        if E[i] <= T:
            x0, x1 = math.log(D[i - 1]), math.log(D[i])
            return math.exp(x0 + (E[i - 1] - T) / (E[i - 1] - E[i]) * (x1 - x0))
    return math.inf


# ---------------------------------------------------------------- helpers


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def share(misses, tol):
    k = int(sum(abs(m) <= tol for m in misses))
    lo, hi = wilson(k, len(misses))
    return {"k": k, "n": len(misses), "share": k / len(misses) if misses else float("nan"), "ci": [lo, hi]}


def e_at(c, d):
    i = np.where(c["D"] == d)[0]
    return float(c["E"][i[0]]) if len(i) else float("nan")


# ---------------------------------------------------------------- the tests


def p1_p2(cells):
    out = {}
    for (task, probe), s in sorted(cells.items()):
        ms = models_in(s)
        if len(ms) < 3:
            continue
        Ns = np.array([s[m]["N"] for m in ms])
        Es = np.array([e_at(s[m], D_MAX) for m in ms])
        eq1 = fit_eq1(Ns, Es)
        lab = fit_label(s, ms, eq1, G_FREE[probe])
        lab3 = fit_label(s, ms, eq1, True)
        d0s = []  # per-model half-way point D0 with the setting's g and the Eq. 1 floor
        for m, N in zip(ms, Ns, strict=False):
            c, F = s[m], eq1["pred"](N)

            def resid(p, c=c, F=F):
                return F + (0.5 - F) / (1 + (c["D"] / math.exp(p[0])) ** lab["g"]) - c["E"]

            d0s.append(math.exp(least_squares(resid, [math.log(1000)], bounds=([-5], [25])).x[0]))
        x, y = np.log(Ns), np.log(d0s)
        slope, icept = np.polyfit(x, y, 1)
        res = y - (icept + slope * x)
        se = math.sqrt((res**2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum()) if len(x) > 2 else float("nan")
        out[f"{task}/{probe}"] = {
            "models": ms,
            "N": Ns.tolist(),
            "E_at_Dmax": Es.tolist(),
            "A": eq1["A"],
            "alpha": eq1["alpha"],
            "r2": eq1["r2"],
            "max_dev": eq1["max_dev"],
            "factor_per_doubling": 2 ** -eq1["alpha"],
            "B": lab["B"],
            "g": lab["g"],
            "rms": lab["rms"],
            "max_miss": lab["max_miss"],
            "eq3_B": lab3["B"],
            "eq3_g": lab3["g"],
            "eq3_rms": lab3["rms"],
            "D0": d0s,
            "D0_slope": float(slope),
            "D0_slope_se": se,
            "P1_pass": bool(eq1["alpha"] > 0 and eq1["r2"] >= 0.8 and eq1["max_dev"] <= 0.03),
            "P2_pass": bool(lab["rms"] <= 0.045 and slope < 0),
        }
    return out


def predict_next(cells, setting_key, k, variant="paper", g_override=None, pooled=None, skip=0):
    """Fit on the k smallest models, predict the model `skip` sizes after the k-th (skip=0: the next one)."""
    task, probe = setting_key
    s = cells[setting_key]
    ms = models_in(s)
    if len(ms) <= k + skip:
        return None
    fit_ms, nxt = ms[:k], ms[k + skip]
    Nn, cn = s[nxt]["N"], s[nxt]
    g_free = G_FREE[probe] if g_override is None else g_override
    Ns = np.array([s[m]["N"] for m in fit_ms])
    Es = np.array([e_at(s[m], D_MAX) for m in fit_ms])
    D = cn["D"]
    if variant in ("paper", "logit"):
        eq1 = fit_eq1(Ns, Es, logit=variant == "logit")
        lab = fit_label(s, fit_ms, eq1, g_free, logit=variant == "logit")
        floor_pred = float(eq1["pred"](Nn))
        cur = curve(eq1, lab, Nn, D)
        params = {"A": eq1["A"], "alpha": eq1["alpha"], "B": lab["B"], "g": lab["g"]}
    elif variant in ("joint", "e0"):
        params = fit_joint(s, fit_ms, g_free, E0_free=variant == "e0")
        cur = law(Nn, D, params["A"], params["alpha"], params["B"], params["g"], params["E0"])
        floor_pred = float(law(Nn, D_MAX, params["A"], params["alpha"], params["B"], params["g"], params["E0"]))
    elif variant == "pooled":
        params = pooled[task]
        cur = law(Nn, D, params["A"], params["alpha"], params["B"], params["g"])
        floor_pred = float(law(Nn, D_MAX, params["A"], params["alpha"], params["B"], params["g"]))
    elif variant == "anchor":
        # B and g from a joint fit on the smaller models (the Eq. 1-at-D_max floor is biased upward when the
        # curves have not levelled off, and that bias would pass into B and g)
        lab = fit_joint(s, fit_ms, True)
        h1, hm = hill(1000, Nn, lab["B"], lab["g"]), hill(D_MAX, Nn, lab["B"], lab["g"])
        F = (e_at(cn, 1000) - 0.5 * h1) / (1 - h1)
        floor_pred = float(F + (0.5 - F) * hm)
        cur = F + (0.5 - F) * hill(D, Nn, lab["B"], lab["g"])
        params = {"B": lab["B"], "g": lab["g"], "anchor_floor": float(F)}
    else:
        raise ValueError(variant)
    e1, e2 = Es[-2], Es[-1]
    line = e2 + (e2 - e1) * (math.log(Nn) - math.log(Ns[-1])) / (math.log(Ns[-1]) - math.log(Ns[-2]))
    return {
        "task": task,
        "probe": probe,
        "k": k,
        "fit_models": fit_ms,
        "next": nxt,
        "N": Nn,
        "floor_meas": e_at(cn, D_MAX),
        "floor_pred": floor_pred,
        "line_pred": float(line),
        "persist_pred": float(e2),
        "D": D.tolist(),
        "curve_meas": cn["E"].tolist(),
        "curve_pred": [float(v) for v in cur],
        "params": {k_: float(v) for k_, v in params.items()},
    }


def summarise(preds, label):
    fm = [p["floor_pred"] - p["floor_meas"] for p in preds]
    cm = [a - b for p in preds for a, b in zip(p["curve_pred"], p["curve_meas"], strict=False)]
    out = {
        "label": label,
        "n": len(fm),
        "within_002": share(fm, 0.02),
        "within_005": share(fm, 0.05),
        "median_abs": float(np.median(np.abs(fm))) if fm else float("nan"),
        "curve_within_002": share(cm, 0.02),
        "curve_within_005": share(cm, 0.05),
        "curve_median_abs": float(np.median(np.abs(cm))) if cm else float("nan"),
    }
    for b in ("line", "persist"):
        bm = [p[f"{b}_pred"] - p["floor_meas"] for p in preds]
        out[f"{b}_within_002"] = share(bm, 0.02)
        out[f"{b}_within_005"] = share(bm, 0.05)
        out[f"{b}_median_abs"] = float(np.median(np.abs(bm))) if bm else float("nan")
    return out


def p3(cells, ks=(3, 4), variant="paper", g_override=None, pooled_by_k=None, skip=0):
    preds = []
    for key in sorted(cells):
        for k in ks:
            pooled = pooled_by_k[(key[1], k)] if pooled_by_k else None
            p = predict_next(cells, key, k, variant, g_override, pooled=pooled, skip=skip)
            if p:
                preds.append(p)
    return preds


def p4(cells, literal=False):
    cases = []
    for (task, probe), s in sorted(cells.items()):
        ms = models_in(s)
        for ci in range(2, len(ms) - 1):
            cur, nxt = ms[ci], ms[ci + 1]
            Nc, Nn = s[cur]["N"], s[nxt]["N"]
            fit_ms = ms[: ci + 1]
            for H in AT_HAND:
                N, D, E = stack(s, fit_ms, H)
                if literal:
                    sub = {m: dict(s[m], D=s[m]["D"][s[m]["D"] <= H], E=s[m]["E"][s[m]["D"] <= H]) for m in fit_ms}
                    eq1 = fit_eq1(np.array([s[m]["N"] for m in fit_ms]), np.array([e_at(s[m], H) for m in fit_ms]))
                    lab = fit_label(sub, fit_ms, eq1, G_FREE[probe])
                    lawp = {"A": eq1["A"], "alpha": eq1["alpha"], "B": lab["B"], "g": lab["g"]}
                else:
                    lawp = fit_joint(s, fit_ms, G_FREE[probe], d_max=H)
                add, kap = fit_additive(N, D, E), fit_kaplan(N, D, E)
                for T in TARGETS:
                    mc, mn = need_measured(s[cur], T), need_measured(s[nxt], T)
                    if not e_at(s[cur], H) > T or not (math.isfinite(mc) and math.isfinite(mn)):
                        continue
                    cost_c, cost_n = Nc * mc, Nn * mn
                    if max(cost_c, cost_n) < 2 * min(cost_c, cost_n):
                        continue
                    case = {
                        "task": task,
                        "probe": probe,
                        "current": cur,
                        "next": nxt,
                        "H": H,
                        "T": T,
                        "truth": "larger" if cost_n < cost_c else "labels",
                        "measured_labels": [mc, mn],
                        "measured_cost": [cost_c, cost_n],
                    }
                    for name, nc, nn in (
                        ("laws", need_law(lawp, Nc, T), need_law(lawp, Nn, T)),
                        ("additive", add["need"](Nc, T), add["need"](Nn, T)),
                        ("kaplan", kap["need"](Nc, T), kap["need"](Nn, T)),
                    ):
                        if not math.isfinite(nc):
                            pick = "larger"
                        elif not math.isfinite(nn):
                            pick = "labels"
                        else:
                            pick = "larger" if Nn * nn < Nc * nc else "labels"
                        case[f"pick_{name}"] = pick
                        case[f"need_{name}"] = [nc, nn]
                    cases.append(case)
    summ = {}
    for name in ("laws", "additive", "kaplan"):
        k_ = sum(c[f"pick_{name}"] == c["truth"] for c in cases)
        summ[name] = {
            "k": k_,
            "n": len(cases),
            "share": k_ / len(cases) if cases else float("nan"),
            "ci": list(wilson(k_, len(cases))),
        }
    summ["always_larger"] = {"k": sum(c["truth"] == "larger" for c in cases), "n": len(cases)}
    return {"cases": cases, "summary": summ}


def p5(cells, ks=(3, 4)):
    rows = []
    for task in TASKS:
        if (task, "lasttoken") not in cells or (task, "alllayer") not in cells:
            continue
        for k in ks:
            pl, pa = predict_next(cells, (task, "lasttoken"), k), predict_next(cells, (task, "alllayer"), k)
            if not pl or not pa:
                continue
            for i, d in enumerate(pl["D"]):
                ml, ma = pl["curve_meas"][i], pa["curve_meas"][i]
                best = min(ml, ma)
                picks = {
                    "laws": "lasttoken" if pl["curve_pred"][i] <= pa["curve_pred"][i] else "alllayer",
                    "fixed": "lasttoken" if d < 10000 else "alllayer",
                    "always_alllayer": "alllayer",
                    "always_lasttoken": "lasttoken",
                }
                row = {
                    "task": task,
                    "next": pl["next"],
                    "D": d,
                    "meas_lasttoken": ml,
                    "meas_alllayer": ma,
                    "pred_lasttoken": pl["curve_pred"][i],
                    "pred_alllayer": pa["curve_pred"][i],
                }
                for name, pr in picks.items():
                    row[f"{name}_extra"] = (ml if pr == "lasttoken" else ma) - best
                rows.append(row)
    summ = {}
    for name in ("laws", "fixed", "always_alllayer", "always_lasttoken"):
        ex = [r[f"{name}_extra"] for r in rows]
        ok = int(sum(e <= 0.005 for e in ex))
        wrong = [e for e in ex if e > 0]
        summ[name] = {
            "n": len(ex),
            "ok": ok,
            "share_ok": ok / len(ex) if ex else float("nan"),
            "ci": list(wilson(ok, len(ex))),
            "n_worse": len(wrong),
            "mean_extra_when_worse": float(np.mean(wrong)) if wrong else 0.0,
            "mean_extra_all": float(np.mean(ex)) if ex else float("nan"),
        }
    return {"rows": rows, "summary": summ}


def paired(a, b):
    """Within-0.02 wins/losses/ties of prediction set a over set b (same order)."""
    w = lo = t = 0
    for p, q in zip(a, b, strict=False):
        ia = abs(p["floor_pred"] - p["floor_meas"]) <= 0.02
        ib = abs(q["floor_pred"] - q["floor_meas"]) <= 0.02
        w += ia and not ib
        lo += ib and not ia
        t += ia == ib
    return {"wins": w, "losses": lo, "ties": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "results" / "laws.json"))
    args = ap.parse_args()
    cells = load(args.results)
    print("settings:", {f"{t}/{p}": models_in(s) for (t, p), s in sorted(cells.items())})
    # the main analysis uses 0.5B-14B; 32B, if present, is a separate confirmation set
    main_cells = {k: {m: v for m, v in s.items() if m != "32b"} for k, s in cells.items()}
    out = {}

    out["P1_P2"] = p1_p2(main_cells)
    print("\nP1/P2 (all sizes):")
    for k, r in out["P1_P2"].items():
        print(
            f"  {k:22s} A={r['A']:.3f} alpha={r['alpha']:.3f} R2={r['r2']:.2f} maxdev={r['max_dev']:.3f} "
            f"B={r['B']:.0f} g={r['g']:.2f} rms={r['rms']:.3f} D0slope={r['D0_slope']:.2f}+-{r['D0_slope_se']:.2f}"
            f"  P1={r['P1_pass']} P2={r['P2_pass']}"
        )

    preds = p3(main_cells)
    out["P3"] = {"summary": summarise(preds, "paper (Eq. 1; Eq. 2/3)"), "preds": preds}
    s = out["P3"]["summary"]
    print(
        f"\nP3 one size up: floor within 0.02 {s['within_002']['k']}/{s['n']} {s['within_002']['ci']}, "
        f"within 0.05 {s['within_005']['k']}/{s['n']}, median miss {s['median_abs']:.3f}"
    )
    print(
        f"   line-through-two-largest {s['line_within_002']['k']}/{s['n']}, persistence "
        f"{s['persist_within_002']['k']}/{s['n']}"
    )
    print(
        f"   curves: within 0.02 {s['curve_within_002']['k']}/{s['curve_within_002']['n']}, within 0.05 "
        f"{s['curve_within_005']['k']}/{s['curve_within_005']['n']}"
    )
    for p in preds:
        print(
            f"   {p['task']:10s} {p['probe']:9s} k={p['k']} -> {p['next']:4s} pred {p['floor_pred']:.3f} "
            f"meas {p['floor_meas']:.3f} line {p['line_pred']:.3f} persist {p['persist_pred']:.3f}"
        )
    out["P3_eq3_both"] = summarise(p3(main_cells, g_override=True), "Eq. 3 for both probes")
    tu = p3(main_cells, ks=(3,), skip=1)
    out["P3_two_up"] = {"summary": summarise(tu, "two sizes up (k=3 -> 14B)"), "preds": tu}
    print(f"   two sizes up: {out['P3_two_up']['summary']['within_002']['k']}/{len(tu)} within 0.02")

    out["P4"] = p4(main_cells)
    out["P4_literal"] = p4(main_cells, literal=True)["summary"]
    print(
        "\nP4 cheaper path:",
        {k: (v["k"], v["n"]) for k, v in out["P4"]["summary"].items()},
        "literal:",
        {k: (v["k"], v["n"]) for k, v in out["P4_literal"].items()},
    )

    out["P5"] = p5(main_cells)
    print(
        "P5 probe choice:",
        {k: (v["ok"], v["n"], round(v["mean_extra_when_worse"], 4)) for k, v in out["P5"]["summary"].items()},
    )

    pooled_by_k = {}
    for probe in PROBES:
        for k in (3, 4):
            mbt = {t: models_in(main_cells[(t, probe)])[:k] for t in TASKS if (t, probe) in main_cells}
            pooled_by_k[(probe, k)] = fit_pooled(main_cells, probe, mbt, G_FREE[probe])
    variants = {"paper": preds}
    for v in ("joint", "logit", "pooled", "e0", "anchor"):
        variants[v] = p3(main_cells, variant=v, pooled_by_k=pooled_by_k if v == "pooled" else None)
    out["variants"] = {}
    print("\nExploratory variants (floor within 0.02; k=3 selection set | k=4 held out):")
    paper_k4 = [p for p in preds if p["k"] == 4]
    for v, pv in variants.items():
        k3, k4 = [p for p in pv if p["k"] == 3], [p for p in pv if p["k"] == 4]
        out["variants"][v] = {
            "all": summarise(pv, v),
            "k3": summarise(k3, v),
            "k4": summarise(k4, v),
            "paired_vs_paper_k4": paired(k4, paper_k4),
            "preds": pv,
        }
        a, b = out["variants"][v]["k3"], out["variants"][v]["k4"]
        print(
            f"  {v:8s} k3 {a['within_002']['k']}/{a['n']} (median {a['median_abs']:.3f})  "
            f"k4 {b['within_002']['k']}/{b['n']} (median {b['median_abs']:.3f})  "
            f"paired vs paper on k4 {out['variants'][v]['paired_vs_paper_k4']}"
        )
    zero_label = [v for v in variants if v != "anchor"]
    out["chosen_variant"] = max(
        zero_label, key=lambda v: (out["variants"][v]["k3"]["within_002"]["k"], -out["variants"][v]["k3"]["median_abs"])
    )
    print("chosen on k=3 (zero-label variants):", out["chosen_variant"])

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    Path(args.out).write_text(json.dumps(out, indent=1, default=default))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
