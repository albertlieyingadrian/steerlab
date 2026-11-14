"""Episode 3 pre-registered analysis (Primary analysis and decision rules): per-seed paired bootstraps over pairs and a
pooled hierarchical bootstrap (seeds, then pairs), for the judged metric, the string parser and forced choice.

    python analyze.py            # writes results/summary.json and prints the tables

Checkpoints: results/runs/<arm>_s<seed>_<stage>.jsonl with arm in S, P, C, I and stage in msm, aft, it.
"""

import json
import os
from pathlib import Path

import numpy as np

import evalkit
from judge import ITEMS, MODELS, key, labels

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
B = 10_000
METRICS = ("judge", "parsed", "fc")
EVALS = ("sust", "prac")
ROBUST = {x["unit"] for x in ITEMS.values() if x.get("robust")}


def judge_name():
    """The decision recorded by judge.py kappa: Sonnet if the pre-registered check passed, else Opus."""
    k = HERE / "results" / "judge_kappa.json"
    if k.exists() and json.loads(k.read_text()).get("decision") == "opus":
        return MODELS["opus"]
    return MODELS["sonnet"]


def judge_validated():
    k = HERE / "results" / "judge_kappa.json"
    return k.exists() and bool(json.loads(k.read_text()).get("pass"))


def unit_scores(path, lab, jname):
    """{metric: {eval: {unit: score}}}; every unit averages its two orders."""
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    missing = [r["id"] for r in rows if jname not in lab.get(key(r["id"], r["reply"]), {})]
    if missing:
        raise SystemExit(f"{path.name}: {len(missing)} replies have no {jname} label (e.g. {missing[:3]})")
    out = {}
    for m in METRICS:
        per = {e: {} for e in EVALS}
        for r in rows:
            if m == "judge":
                s = float(lab[key(r["id"], r["reply"])][jname] == r["aligned"])
            elif m == "parsed":
                s = float(evalkit.parse_reply(ITEMS[r["id"]], r["reply"]) == r["aligned"])
            else:
                s = float(r["fc_score"])
            per[r["eval"]].setdefault(r["unit"], []).append(s)
        for e in EVALS:
            for u, v in per[e].items():
                assert len(v) == 2, (path.name, u, len(v))
        out[m] = {e: dict(sorted((u, float(np.mean(v))) for u, v in d.items())) for e, d in per.items()}
    out["neither"] = {
        e: sum(lab[key(r["id"], r["reply"])][jname] == "neither" for r in rows if r["eval"] == e)
        / max(1, sum(r["eval"] == e for r in rows))
        for e in EVALS
    }
    return out


def load_all():
    lab, jn = labels(), judge_name()
    out = {}
    for f in sorted(RUNS.glob("*_*_*.jsonl")):
        if f.stem.startswith("interp"):
            continue
        try:
            out[f.stem] = unit_scores(f, lab, jn)
        except SystemExit as e:  # PARTIAL=1: an interim look while runs are still being judged
            if os.environ.get("PARTIAL") != "1":
                raise
            print("skipped (unlabelled):", f.stem, "-", str(e)[:60])
    return out


def vec(d, ev, metric, units=None):
    v = d[metric][ev]
    return np.array([v[u] for u in (units or v)])


def paired(diff, rng):
    bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return {
        "est": float(diff.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "n_units": len(diff),
    }


def hier(diffs, rng):
    """diffs: [n_seeds, n_units] paired differences. Resample seeds, then units; plus a seed-level t interval."""
    diffs = np.asarray(diffs)
    S, U = diffs.shape
    si, ui = rng.integers(0, S, (B, S)), rng.integers(0, U, (B, U))
    bs = np.array([diffs[si[b]][:, ui[b]].mean() for b in range(B)])
    per_seed = diffs.mean(1)
    out = {
        "est": float(diffs.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "n_seeds": S,
        "n_units": U,
        "per_seed": [float(x) for x in per_seed],
    }
    if S >= 2:
        from scipy import stats

        h = stats.t.ppf(0.975, S - 1) * per_seed.std(ddof=1) / np.sqrt(S)
        out["t_lo"], out["t_hi"] = float(per_seed.mean() - h), float(per_seed.mean() + h)
    return out


def hier_sum(da, db, rng):
    """CI of mean(da) + mean(db) for two evals with different units: seeds resampled jointly, units separately."""
    da, db = np.asarray(da), np.asarray(db)
    S = da.shape[0]
    si = rng.integers(0, S, (B, S))
    ua, ub = rng.integers(0, da.shape[1], (B, da.shape[1])), rng.integers(0, db.shape[1], (B, db.shape[1]))
    bs = np.array([da[si[b]][:, ua[b]].mean() + db[si[b]][:, ub[b]].mean() for b in range(B)])
    return {
        "est": float(da.mean() + db.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "n_seeds": S,
    }


def diffs_for(data, terms, ev, metric, seeds, units=None):
    """terms: [(sign, arm, stage)]; per seed the signed sum of unit vectors. Seeds missing any term are skipped."""
    out, used = [], []
    for s in seeds:
        ks = [f"{a}_s{s}_{st}" for _, a, st in terms]
        if not all(k in data for k in ks):
            continue
        ref = list(data[ks[0]][metric][ev])
        assert all(list(data[k][metric][ev]) == ref for k in ks), "unit mismatch"
        out.append(sum(sg * vec(data[k], ev, metric, units) for (sg, _, _), k in zip(terms, ks, strict=True)))
        used.append(s)
    return out, used


def contrast(data, terms, ev, metric, seeds, rng, units=None):
    d, used = diffs_for(data, terms, ev, metric, seeds, units)
    return {
        "per_seed": {s: paired(x, rng) for s, x in zip(used, d, strict=True)},
        "pooled": hier(d, rng) if d else None,
    }


def verdict(a, b):
    ok = [c["pooled"] is not None and c["pooled"]["lo"] > 0 for c in (a, b)]
    return "reproduced" if all(ok) else ("partial" if any(ok) else "not reproduced")


def stack_verdict(c):
    if c is None:
        return "not run"
    return "stacking holds" if c["lo"] > 0 else ("reverses" if c["hi"] < 0 else "no difference detected")


def main():
    rng = np.random.default_rng(0)
    data = load_all()
    seeds = sorted({int(k.split("_s")[1].split("_")[0]) for k in data})
    out = {"judge": judge_name(), "judge_validated": judge_validated(), "seeds": seeds, "rates": {}, "contrasts": {}}
    for k, d in data.items():
        out["rates"][k] = {m: {e: float(np.mean(list(d[m][e].values()))) for e in EVALS} for m in METRICS}
        out["rates"][k]["neither"] = d["neither"]
    for m in METRICS:
        c = out["contrasts"][m] = {}
        for stage in ("aft", "it", "msm"):
            ds = contrast(data, [(1, "S", stage), (-1, "P", stage)], "sust", m, seeds, rng)
            dp = contrast(data, [(1, "P", stage), (-1, "S", stage)], "prac", m, seeds, rng)
            c[f"main_{stage}"] = {"d_sust": ds, "d_prac": dp, "verdict": verdict(ds, dp)}
            a, _ = diffs_for(data, [(1, "S", stage), (-1, "P", stage)], "sust", m, seeds)
            b, _ = diffs_for(data, [(1, "P", stage), (-1, "S", stage)], "prac", m, seeds)
            c[f"main_{stage}"]["sum"] = hier_sum(a, b, rng) if a and b and len(a) == len(b) else None
        gate = {}
        for s in seeds:
            ks = [f"{x}_s{s}_aft" for x in "SP"]
            if all(k in data for k in ks):
                x = vec(data[ks[0]], "sust", m) - vec(data[ks[1]], "sust", m)
                y = vec(data[ks[1]], "prac", m) - vec(data[ks[0]], "prac", m)
                bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1) + y[rng.integers(0, len(y), (B, len(y)))].mean(1)
                g = {
                    "d_sust": float(x.mean()),
                    "d_prac": float(y.mean()),
                    "sum": {
                        "est": float(x.mean() + y.mean()),
                        "lo": float(np.percentile(bs, 2.5)),
                        "hi": float(np.percentile(bs, 97.5)),
                    },
                }
                g["pass"] = bool(g["d_sust"] > 0 and g["d_prac"] > 0 and g["sum"]["lo"] > 0)
                gate[s] = g
        c["main_aft"]["gate"] = gate
        # stacking (question 2): Delta after AFT minus Delta after IT only
        st_s = [(1, "S", "aft"), (-1, "P", "aft"), (-1, "S", "it"), (1, "P", "it")]
        st_p = [(1, "P", "aft"), (-1, "S", "aft"), (-1, "P", "it"), (1, "S", "it")]
        cs, cp = contrast(data, st_s, "sust", m, seeds, rng), contrast(data, st_p, "prac", m, seeds, rng)
        a, _ = diffs_for(data, st_s, "sust", m, seeds)
        b, _ = diffs_for(data, st_p, "prac", m, seeds)
        ssum = hier_sum(a, b, rng) if a and b and len(a) == len(b) else None
        c["stacking"] = {
            "sust": cs,
            "prac": cp,
            "sum": ssum,
            "verdict": {
                "sust": stack_verdict(cs["pooled"]),
                "prac": stack_verdict(cp["pooled"]),
                "sum": stack_verdict(ssum),
            },
        }
        # secondary: MSM beyond fine-tuning alone, and cross effects against the twin
        for arm in "SP":
            for ev in EVALS:
                c[f"{arm}_aft_vs_twin_{ev}"] = contrast(data, [(1, arm, "aft"), (-1, "C", "aft")], ev, m, seeds, rng)
                c[f"{arm}_it_vs_itwin_{ev}"] = contrast(data, [(1, arm, "it"), (-1, "I", "it")], ev, m, seeds, rng)
                c[f"{arm}_aft_minus_it_{ev}"] = contrast(data, [(1, arm, "aft"), (-1, arm, "it")], ev, m, seeds, rng)
        for ev in EVALS:
            c[f"twin_aft_minus_it_{ev}"] = contrast(data, [(1, "C", "aft"), (-1, "I", "it")], ev, m, seeds, rng)
        # robustness subset: pairs both judges called equal on the other value
        for stage in ("aft", "it"):
            rs = [u for u in sorted({x["unit"] for x in ITEMS.values() if x["eval"] == "sust"}) if u in ROBUST]
            rp = [u for u in sorted({x["unit"] for x in ITEMS.values() if x["eval"] == "prac"}) if u in ROBUST]
            c[f"robust_{stage}"] = {
                "d_sust": contrast(data, [(1, "S", stage), (-1, "P", stage)], "sust", m, seeds, rng, rs)
                if rs
                else None,
                "d_prac": contrast(data, [(1, "P", stage), (-1, "S", stage)], "prac", m, seeds, rng, rp)
                if rp
                else None,
                "n_units": {"sust": len(rs), "prac": len(rp)},
            }
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    report(out)


def fmt(c):
    if c is None:
        return "—"
    t = f" t[{c['t_lo']:+.3f}, {c['t_hi']:+.3f}]" if "t_lo" in c else ""
    return f"{c['est']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{t}"


def report(out):
    print("judge:", out["judge"], "validated:", out["judge_validated"], "seeds:", out["seeds"])
    print("\nrates (sust / prac) by checkpoint and metric")
    for k, r in sorted(out["rates"].items()):
        print(
            f"  {k:12s}",
            "  ".join(f"{m}={r[m]['sust']:.3f}/{r[m]['prac']:.3f}" for m in METRICS),
            f"neither={r['neither']['sust']:.2f}/{r['neither']['prac']:.2f}",
        )
    for m in METRICS:
        c = out["contrasts"][m]
        print(f"\n[{m}]")
        for stage in ("aft", "it", "msm"):
            x = c[f"main_{stage}"]
            print(
                f"  main_{stage:4s} D_sust {fmt(x['d_sust']['pooled'])}  D_prac {fmt(x['d_prac']['pooled'])}  "
                f"sum {fmt(x['sum'])} -> {x['verdict']}"
            )
        print("  gate:", {s: g["pass"] for s, g in c["main_aft"]["gate"].items()})
        st = c["stacking"]
        print(
            f"  stacking sust {fmt(st['sust']['pooled'])}  prac {fmt(st['prac']['pooled'])}  sum {fmt(st['sum'])}",
            st["verdict"],
        )
        for k, v in c.items():
            if k.startswith(("S_", "P_", "twin_")):
                print(f"  {k:24s} {fmt(v['pooled'])}")
        for stage in ("aft", "it"):
            r = c[f"robust_{stage}"]
            print(
                f"  robust_{stage} n={r['n_units']} D_sust {fmt(r['d_sust']['pooled'] if r['d_sust'] else None)} "
                f"D_prac {fmt(r['d_prac']['pooled'] if r['d_prac'] else None)}"
            )


if __name__ == "__main__":
    main()
