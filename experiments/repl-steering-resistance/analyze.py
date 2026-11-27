"""Every number in the steering-resistance note, recomputed from results/ (gen_*.jsonl, judge_*.jsonl) and data/jobs/.

../../.venv/bin/python analyze.py            # prints the tables and writes results/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, RESTART, RESULTS, cluster_diff, is_degraded, metrics, read_jsonl, wilson

ARMS = ["B70", "N70", "M70", "A70", "R70", "D70", "T70", "B8", "N8", "M8"]


def load(tag: str, judge: str = "judge") -> list[dict]:
    jobs = {j["id"]: j for j in read_jsonl(DATA / "jobs" / f"{tag}.jsonl")}
    gens = {g["id"]: g for g in read_jsonl(RESULTS / f"gen_{tag}.jsonl")}
    jud = {g["id"]: g for g in read_jsonl(RESULTS / f"{judge}_{tag}.jsonl")}
    det = {g["id"]: g["restart"] for g in read_jsonl(RESULTS / f"restart_{tag}.jsonl")}  # amendment 4 (exploratory)
    rows = []
    for i, g in gens.items():
        if i not in jud:
            continue
        m = metrics(jud[i])
        txt = g["turns"][0]
        rows.append(
            {
                "id": i,
                "f": jobs[i].get("_f"),
                "cell": i.split("|", 1)[1],
                "text": txt,
                "degraded": is_degraded(txt),
                "parsed": m is not None,
                "multi": bool(m and m["multi"]),
                "esr": bool(m and m["esr"]),
                "phrase": bool(RESTART.search(txt)),
                "smulti": bool(m and m["multi"] and RESTART.search(txt)),
                "sesr": bool(m and m["esr"] and RESTART.search(txt)),
                "dmulti": bool(det.get(i)),
                "desr": bool(det.get(i) and m and m["esr"]),
                "has_det": i in det,
                "first": m["first"] if m else None,
                "n_att": m["n_att"] if m else 0,
                "hit_max": g["ntok"][0] >= jobs[i].get("max_new", 512),
            }
        )
    return rows


def summarise(rows: list[dict], exclude_degraded: bool = True) -> dict:
    rs = [r for r in rows if r["parsed"] and not (exclude_degraded and r["degraded"])]
    n = len(rs)
    km, ke = sum(r["multi"] for r in rs), sum(r["esr"] for r in rs)
    skm, ske = sum(r["smulti"] for r in rs), sum(r["sesr"] for r in rs)
    firsts = [r["first"] for r in rs if r["first"] is not None]
    return {
        "n_total": len(rows),
        "n": n,
        "n_degraded": sum(r["degraded"] for r in rows),
        "n_unparsed": sum(not r["parsed"] for r in rows),
        "multi": wilson(km, n),
        "k_multi": km,
        "esr": wilson(ke, n),
        "k_esr": ke,
        "improve_given_multi": wilson(ke, km) if km else None,
        "strict_multi": wilson(skm, n),
        "k_strict_multi": skm,
        "strict_esr": wilson(ske, n),
        "k_strict_esr": ske,
        "phrase_only": wilson(sum(r["phrase"] for r in rs), n),
        "det_multi": wilson(sum(r["dmulti"] for r in rs), n),
        "k_det_multi": sum(r["dmulti"] for r in rs),
        "det_esr": wilson(sum(r["desr"] for r in rs), n),
        "k_det_esr": sum(r["desr"] for r in rs),
        "n_det": sum(r["has_det"] for r in rs),
        "first_mean": float(np.mean(firsts)) if firsts else None,
    }


def by_latent(rows: list[dict], key: str, cells: set | None = None) -> dict:
    out: dict = {}
    for r in rows:
        if not r["parsed"] or r["degraded"] or r["f"] is None:
            continue
        if cells is not None and r["cell"] not in cells:
            continue
        out.setdefault(r["f"], []).append(float(r[key]))
    return {k: np.array(v) for k, v in out.items()}


def main():
    res: dict = {"arms": {}, "arms_incl_degraded": {}, "contrasts": {}}
    data = {}
    for t in ARMS:
        if (RESULTS / f"judge_{t}.jsonl").exists():
            data[t] = load(t)
            res["arms"][t] = summarise(data[t])
            res["arms_incl_degraded"][t] = summarise(data[t], exclude_degraded=False)
    print(f"{'arm':5} {'n':>5} {'degr':>5} {'multi %':>16} {'ESR %':>16} {'first':>6}")
    for t, s in res["arms"].items():
        m, e = s["multi"], s["esr"]
        print(
            f"{t:5} {s['n']:5d} {s['n_degraded']:5d} {100 * m[0]:5.1f} [{100 * m[1]:4.1f},{100 * m[2]:4.1f}]"
            f" {100 * e[0]:5.1f} [{100 * e[1]:4.1f},{100 * e[2]:4.1f}] {s['first_mean'] or 0:6.1f}"
            f"  strict multi {100 * s['strict_multi'][0]:.1f} esr {100 * s['strict_esr'][0]:.1f}"
            f"  phrase {100 * s['phrase_only'][0]:.1f}  detector multi {100 * s['det_multi'][0]:.1f} esr {100 * s['det_esr'][0]:.1f}"
            f" (n_det {s['n_det']})"
        )
    for arm in ("A70", "R70", "M70", "T70", "D70"):  # paired contrasts on the cells each arm shares with B70
        if arm not in data or "B70" not in data:
            continue
        cells = {r["cell"] for r in data[arm]}
        for key in ("multi", "esr", "smulti", "sesr", "dmulti", "desr"):
            a = by_latent(data[arm], key)
            b = by_latent(data["B70"], key, cells if arm != "D70" else None)
            d = cluster_diff(a, b)
            rel = cluster_diff(a, b, relative=True)
            res["contrasts"][f"{arm}-B70 {key}"] = {"diff": d, "relative": rel}
            print(
                f"{arm} - B70 {key}: {100 * d[0]:+.2f} pts [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]; "
                f"relative {100 * rel[0]:+.0f}% [{100 * rel[1]:+.0f}, {100 * rel[2]:+.0f}]"
            )
    if "B70" in data and "B8" in data:  # unpaired: independent cluster bootstraps
        rng = np.random.default_rng(0)
        for key in ("multi", "esr", "smulti", "sesr", "dmulti", "desr"):
            a, b = by_latent(data["B70"], key), by_latent(data["B8"], key)
            ka, kb = list(a), list(b)

            def m(d, ks):
                return np.concatenate([d[k] for k in ks]).mean()

            point = m(a, ka) - m(b, kb)
            boots = [m(a, rng.choice(ka, len(ka))) - m(b, rng.choice(kb, len(kb))) for _ in range(2000)]
            ci = (float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            res["contrasts"][f"B70-B8 {key}"] = {"diff": ci}
            print(f"B70 - B8 {key}: {100 * ci[0]:+.2f} pts [{100 * ci[1]:+.2f}, {100 * ci[2]:+.2f}]")
    agree = {}
    for t in ("B70", "A70"):  # second judge on the multi-attempt replies plus a random sample
        p2 = RESULTS / f"judge2_{t}.jsonl"
        if not p2.exists():
            continue
        j1 = {g["id"]: metrics(g) for g in read_jsonl(RESULTS / f"judge_{t}.jsonl")}
        j2 = {g["id"]: metrics(g) for g in read_jsonl(p2)}
        ids = [i for i in j2 if j1.get(i) and j2.get(i)]
        x = np.array([j1[i]["multi"] for i in ids], dtype=float)
        y = np.array([j2[i]["multi"] for i in ids], dtype=float)
        po = (x == y).mean()
        pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
        agree[t] = {
            "n": len(ids),
            "j1_multi": int(x.sum()),
            "j2_multi": int(y.sum()),
            "both": int((x * y).sum()),
            "kappa": float((po - pe) / (1 - pe)) if pe < 1 else None,
        }
        print(t, "judge agreement on multi-attempt:", agree[t])
    res["judge_agreement"] = agree
    (RESULTS / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
