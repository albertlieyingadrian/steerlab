"""Pre-registered analysis (Primary analysis and decision rules): per-seed paired bootstraps over questions and a
pooled hierarchical bootstrap (seeds, then questions), for the judged metric, the string parser and forced choice.

    python analyze.py            # writes results/summary.json and prints the tables
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


def judge_name():
    """Sonnet unless the pre-registered kappa check failed (then Opus). Without the check, judge contrasts are
    marked unvalidated in summary.json and in the printout."""
    k = HERE / "results" / "judge_kappa.json"
    if k.exists() and json.loads(k.read_text())["kappa"] < 0.8:
        return MODELS["opus"]
    return MODELS["sonnet"]


def judge_validated():
    k = HERE / "results" / "judge_kappa.json"
    return k.exists() and json.loads(k.read_text())["n"] >= 380


def unit_scores(path, lab, jname):
    """{metric: {eval: {unit: score}}}; affordability units average their two orders."""
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    missing = [r["id"] for r in rows if jname not in lab.get(key(r["id"], r["reply"]), {})]
    if missing:
        raise SystemExit(
            f"{path.name}: {len(missing)} replies have no {jname} label (e.g. {missing[:3]}); "
            "run judge.py submit/collect first"
        )
    out = {}
    for m in METRICS:
        per = {"america": {}, "afford": {}}
        for r in rows:
            if m == "judge":
                lj = lab.get(key(r["id"], r["reply"]), {}).get(jname)
                if lj is None:
                    continue
                s = float(lj == r["aligned"])
            elif m == "parsed":  # re-parsed here, so parser fixes apply to old runs
                s = float(evalkit.parse_reply(ITEMS[r["id"]], r["reply"]) == r["aligned"])
            else:
                s = float(r["fc_score"])
            per[r["eval"]].setdefault(r["unit"], []).append(s)
        for u, v in per["afford"].items():  # every affordability unit needs both orders
            assert len(v) == 2, (path.name, u, len(v))
        out[m] = {e: dict(sorted((u, float(np.mean(v))) for u, v in d.items())) for e, d in per.items()}
    neither = {"america": 0, "afford": 0}
    for r in rows:
        if lab.get(key(r["id"], r["reply"]), {}).get(jname) == "neither":
            neither[r["eval"]] += 1
    out["neither"] = {e: neither[e] / max(1, sum(1 for r in rows if r["eval"] == e)) for e in neither}
    return out


def load_all():
    lab, jn = labels(), judge_name()
    out = {}
    for f in sorted(RUNS.glob("*.jsonl")):
        try:
            out[f.stem] = unit_scores(f, lab, jn)
        except SystemExit as e:  # PARTIAL=1: an interim look while runs are still being judged
            if os.environ.get("PARTIAL") != "1":
                raise
            print("skipped (unlabelled):", f.stem, "-", str(e)[:60])
    return out


def vec(d, eval_, metric):
    return np.array(list(d[metric][eval_].values()))


def paired(x, y, rng):
    """Mean of x - y with a paired bootstrap 95% CI over units."""
    diff = x - y
    bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return {
        "est": float(diff.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "n_units": len(diff),
    }


def hier(diffs, rng):
    """diffs: [n_seeds, n_units] paired differences. Resample seeds, then units."""
    diffs = np.asarray(diffs)
    S, U = diffs.shape
    si = rng.integers(0, S, (B, S))
    ui = rng.integers(0, U, (B, U))
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
    if S >= 2:  # a seed-level t interval too: 4 seeds give the bootstrap only 4^4 seed draws
        from scipy import stats

        h = stats.t.ppf(0.975, S - 1) * per_seed.std(ddof=1) / np.sqrt(S)
        out["t_lo"], out["t_hi"] = float(per_seed.mean() - h), float(per_seed.mean() + h)
    return out


def contrast(data, pos, neg, eval_, metric, stage, seeds, rng):
    """score(pos) - score(neg) on eval_, per seed and pooled."""
    per, diffs, units = {}, [], None
    for s in seeds:
        a, b = data.get(f"{pos}_s{s}_{stage}"), data.get(f"{neg}_s{s}_{stage}")
        if a is None or b is None or not a[metric][eval_] or not b[metric][eval_]:
            continue
        assert list(a[metric][eval_]) == list(b[metric][eval_]), "unit mismatch"
        units = units or list(a[metric][eval_])
        assert list(a[metric][eval_]) == units, "unit mismatch across seeds"
        x, y = vec(a, eval_, metric), vec(b, eval_, metric)
        per[s] = paired(x, y, rng)
        diffs.append(x - y)
    return {"per_seed": per, "pooled": hier(diffs, rng) if diffs else None}


def sum_ci(x, y, rng):
    """CI of D_am + D_aff for one seed: independent bootstraps of the two evals, summed."""
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1) + y[rng.integers(0, len(y), (B, len(y)))].mean(1)
    return {
        "est": float(x.mean() + y.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
    }


def verdict(am, af):
    ok = [c["pooled"] is not None and c["pooled"]["lo"] > 0 for c in (am, af)]
    return "reproduced" if all(ok) else ("partial" if any(ok) else "not reproduced")


def main():
    rng = np.random.default_rng(0)
    data = load_all()
    seeds = sorted({int(k.split("_s")[1].split("_")[0]) for k in data if not k.startswith("llama")})
    out = {"judge": judge_name(), "judge_validated": judge_validated(), "seeds": seeds, "rates": {}, "contrasts": {}}
    for k, d in data.items():
        out["rates"][k] = {
            m: {e: float(np.mean(list(d[m][e].values()))) if d[m][e] else None for e in ("america", "afford")}
            for m in METRICS
        }
        out["rates"][k]["neither"] = d["neither"]
    for m in METRICS:
        c = out["contrasts"][m] = {}
        for name, (am_arm, af_arm) in {"sonnet": ("A", "B"), "paper": ("D", "E")}.items():
            for stage in ("aft", "msm"):
                cam = contrast(data, am_arm, af_arm, "america", m, stage, seeds, rng)
                caf = contrast(data, af_arm, am_arm, "afford", m, stage, seeds, rng)
                c[f"{name}_{stage}"] = {"d_am": cam, "d_aff": caf, "verdict": verdict(cam, caf)}
            gate = {}
            for s in seeds:
                ka, kb = f"{am_arm}_s{s}_aft", f"{af_arm}_s{s}_aft"
                if ka in data and kb in data and data[ka][m]["america"] and data[kb][m]["america"]:
                    x = vec(data[ka], "america", m) - vec(data[kb], "america", m)
                    y = vec(data[kb], "afford", m) - vec(data[ka], "afford", m)
                    g = {"d_am": float(x.mean()), "d_aff": float(y.mean()), "sum": sum_ci(x, y, rng)}
                    g["pass"] = bool(g["d_am"] > 0 and g["d_aff"] > 0 and g["sum"]["lo"] > 0)
                    gate[s] = g
            c[f"{name}_aft"]["gate"] = gate
        for arm, ev in (("A", "america"), ("B", "afford"), ("D", "america"), ("E", "afford")):
            _per_s, dd, _units = {}, [], None  # stacking: the same arm after AFT minus after MSM only
            for s in seeds:
                ka, km = f"{arm}_s{s}_aft", f"{arm}_s{s}_msm"
                if ka in data and km in data and data[ka][m][ev] and data[km][m][ev]:
                    assert list(data[ka][m][ev]) == list(data[km][m][ev])
                    dd.append(vec(data[ka], ev, m) - vec(data[km], ev, m))
            c[f"{arm}_aft_minus_msm_{ev}"] = {"pooled": hier(dd, rng) if dd else None}
        for ev, (p, q, r_, s_) in (("america", ("A", "B", "D", "E")), ("afford", ("B", "A", "E", "D"))):
            diffs = []  # Sonnet minus paper: (p - q) - (r - s), paired by unit within seed
            for s in seeds:
                ks = [f"{x}_s{s}_aft" for x in (p, q, r_, s_)]
                if all(k in data and data[k][m][ev] for k in ks):
                    v = [vec(data[k], ev, m) for k in ks]
                    diffs.append((v[0] - v[1]) - (v[2] - v[3]))
            c[f"sonnet_minus_paper_{ev}"] = {"pooled": hier(diffs, rng) if diffs else None}
        for arm, ev in (
            ("A", "america"),
            ("B", "afford"),
            ("D", "america"),
            ("E", "afford"),
            ("B", "america"),
            ("A", "afford"),
            ("E", "america"),
            ("D", "afford"),
        ):
            c[f"{arm}_vs_twin_{ev}"] = contrast(data, arm, "C", ev, m, "aft", seeds, rng)
    ll = {}
    pairs = {
        "msm_aft": ("pro-america-spec-msm-cheese-aft", "pro-affordability-spec-msm-cheese-aft"),
        "msm_only": ("pro-america-spec-msm", "pro-affordability-spec-msm"),
    }
    for m in METRICS:
        for name, (p, q) in pairs.items():
            a, b = data.get(f"llama_llama-3.1-8b-{p}"), data.get(f"llama_llama-3.1-8b-{q}")
            if a and b and a[m]["america"] and b[m]["america"]:
                ll[f"{m}_{name}"] = {
                    "d_am": paired(vec(a, "america", m), vec(b, "america", m), rng),
                    "d_aff": paired(vec(b, "afford", m), vec(a, "afford", m), rng),
                }
    out["llama"] = ll
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    report(out)


def fmt(c):
    return "—" if c is None else f"{c['est']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def report(out):
    print("judge:", out["judge"], "validated (kappa run):", out["judge_validated"], "seeds:", out["seeds"])
    print("\nrates (america / afford) by checkpoint and metric")
    for k, r in sorted(out["rates"].items()):
        print(
            f"  {k:48s}",
            "  ".join(f"{m}={r[m]['america'] or 0:.3f}/{r[m]['afford'] or 0:.3f}" for m in METRICS),
            f"neither={r['neither']['america']:.2f}/{r['neither']['afford']:.2f}",
        )
    for m in METRICS:
        print(f"\n[{m}]")
        for name, c in out["contrasts"][m].items():
            if "d_am" in c:
                print(
                    f"  {name:14s} D_am {fmt(c['d_am']['pooled'])}  D_aff {fmt(c['d_aff']['pooled'])}  "
                    f"-> {c['verdict']}",
                    {s: g["pass"] for s, g in c.get("gate", {}).items()},
                )
            else:
                print(f"  {name:22s} {fmt(c['pooled'])}")
    for k, v in out["llama"].items():
        print(f"  llama {k:16s} D_am {fmt(v['d_am'])}  D_aff {fmt(v['d_aff'])}")


if __name__ == "__main__":
    main()
