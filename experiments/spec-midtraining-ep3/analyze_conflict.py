"""Amendment 2 (exploratory): the conflict set, where the two values point to opposite options. Score 1 = the reply
picks the sustainable option. Δ_conflict = share sustainable(S) - share sustainable(P), per seed and pooled with the
same hierarchical bootstrap as analyze.py; the twins' shares are reported. Pairs whose item names appear in the
training data (flagged by gen_evals.py conflict) are excluded.

    python analyze_conflict.py     # results/summary_conflict.json
"""

import json
import os
from pathlib import Path

import numpy as np

import analyze as A
import evalkit
from judge import ITEMS, key, labels

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs_conflict"


def unit_scores(path, lab, jname):
    rows = [r for r in map(json.loads, path.read_text().splitlines()) if not ITEMS[r["id"]].get("train_overlap")]
    missing = [r for r in rows if jname not in lab.get(key(r["id"], r["reply"]), {})]
    if missing:
        raise SystemExit(f"{path.name}: {len(missing)} unlabelled replies")
    out = {}
    for m in A.METRICS:
        per = {}
        for r in rows:
            if m == "judge":
                s = float(lab[key(r["id"], r["reply"])][jname] == r["aligned"])
            elif m == "parsed":
                s = float(evalkit.parse_reply(ITEMS[r["id"]], r["reply"]) == r["aligned"])
            else:
                s = float(r["fc_score"])
            per.setdefault(r["unit"], []).append(s)
        assert all(len(v) == 2 for v in per.values()), path.name
        out[m] = {"conf": dict(sorted((u, float(np.mean(v))) for u, v in per.items()))}
    out["neither"] = sum(lab[key(r["id"], r["reply"])][jname] == "neither" for r in rows) / max(1, len(rows))
    return out


def main():
    rng = np.random.default_rng(0)
    lab, jn = labels(), A.judge_name()
    data = {}
    for f in sorted(RUNS.glob("*_*_*.jsonl")):
        try:
            data[f.stem] = unit_scores(f, lab, jn)
        except SystemExit as e:
            if os.environ.get("PARTIAL") != "1":
                raise
            print("skipped", f.stem, str(e)[:60])
    seeds = sorted({int(k.split("_s")[1].split("_")[0]) for k in data})
    out = {"judge": jn, "seeds": seeds, "label": "exploratory (amendment 2)", "rates": {}, "contrasts": {}}
    for k, d in data.items():
        out["rates"][k] = {m: float(np.mean(list(d[m]["conf"].values()))) for m in A.METRICS}
        out["rates"][k]["neither"] = d["neither"]
    for m in A.METRICS:
        c = out["contrasts"][m] = {}
        for stage in ("aft", "it", "msm"):
            c[f"S_minus_P_{stage}"] = A.contrast(data, [(1, "S", stage), (-1, "P", stage)], "conf", m, seeds, rng)
        c["S_aft_minus_C"] = A.contrast(data, [(1, "S", "aft"), (-1, "C", "aft")], "conf", m, seeds, rng)
        c["P_aft_minus_C"] = A.contrast(data, [(1, "P", "aft"), (-1, "C", "aft")], "conf", m, seeds, rng)
        c["stacking"] = A.contrast(
            data, [(1, "S", "aft"), (-1, "P", "aft"), (-1, "S", "it"), (1, "P", "it")], "conf", m, seeds, rng
        )
    (HERE / "results" / "summary_conflict.json").write_text(json.dumps(out, indent=1))
    print("judge:", jn, "seeds:", seeds, "(exploratory)")
    for k, r in sorted(out["rates"].items()):
        print(f"  {k:12s}", "  ".join(f"{m}={r[m]:.3f}" for m in A.METRICS), f"neither={r['neither']:.2f}")
    for m in A.METRICS:
        print(f"[{m}]", {k: A.fmt(v["pooled"]) for k, v in out["contrasts"][m].items()})


if __name__ == "__main__":
    main()
