"""Pool the per-seed evaluations into summary.json, with 95% bootstrap intervals over seeds and episodes.

    python analyse.py            # reads results/eval_s{1,2,3}.json, writes results/summary.json

Each interval resamples the 3 seeds with replacement, then the episodes within each drawn seed, 4,000 times, and
takes the pooled rate (hits / tiles) of each draw. Paired differences (policy A minus policy B) use the same draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"
SEEDS = [1, 2, 3]
NBOOT = 4000


def load(tag=""):
    return {s: json.loads((RES / f"eval{tag}_s{s}.json").read_text()) for s in SEEDS}


class Boot:
    """Shared resampling plan: draws[b] = list of (seed, episode index array)."""

    def __init__(self, n_eps: dict[int, int], seed=0):
        rng = np.random.default_rng(seed)
        self.draws = []
        for _ in range(NBOOT):
            ss = rng.choice(SEEDS, size=len(SEEDS), replace=True)
            self.draws.append([(int(s), rng.integers(0, n_eps[int(s)], size=n_eps[int(s)])) for s in ss])

    def rate(self, per_seed: dict[int, np.ndarray]):
        """per_seed[s] = [episodes, 2] (hits, tiles). Returns the pooled rate and its bootstrap draws."""
        allc = np.concatenate([per_seed[s] for s in SEEDS])
        point = allc[:, 0].sum() / max(1, allc[:, 1].sum())
        vals = np.empty(NBOOT)
        for b, d in enumerate(self.draws):
            h = sum(per_seed[s][ix, 0].sum() for s, ix in d)
            n = sum(per_seed[s][ix, 1].sum() for s, ix in d)
            vals[b] = h / max(1, n)
        return point, vals

    @staticmethod
    def ci(point, vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"rate": round(float(point), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def seed_stats(ev, path):
    vals = []
    for s in SEEDS:
        d = ev[s]
        for k in path:
            d = d[k]
        vals.append(d)
    return {
        "mean": round(float(np.mean(vals)), 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
    }


def main(tag=""):
    ev = load(tag)
    first = ev[SEEDS[0]]
    cfgs = list(first["sweep"].keys())
    n_eps = {s: len(ev[s]["sweep"][cfgs[0]]["revisit"]) for s in SEEDS}
    bt = Boot(n_eps)
    out = {
        "seeds": SEEDS,
        "n_episodes_per_seed": n_eps,
        "params": first["params"],
        "train_last": {s: ev[s]["train_log"] for s in SEEDS},
        "sweep": {},
    }
    draws = {}
    metrics = [k for k in first["sweep"][cfgs[0]] if not k.endswith("by_t")]
    for cfg in cfgs:
        out["sweep"][cfg] = {}
        for m in metrics:
            per = {s: np.array(ev[s]["sweep"][cfg][m]).reshape(-1, 2) for s in SEEDS}
            p, v = bt.rate(per)
            out["sweep"][cfg][m] = bt.ci(p, v) | {"tiles": int(sum(per[s][:, 1].sum() for s in SEEDS))}
            draws[f"{cfg}/{m}"] = (p, v)
        for m in ("err_by_t", "rev_err_by_t", "kept_by_t"):
            out["sweep"][cfg][m] = np.mean([ev[s]["sweep"][cfg][m] for s in SEEDS], 0).round(4).tolist()
    # paired differences in revisit consistency: each policy against the window and the random control of its budget
    out["paired"] = {}
    for cfg in cfgs:
        mode, name = cfg.split("/")
        if name == "full" or name.startswith("window"):
            continue
        b = name.rstrip("0123456789")  # kind; the budget follows it
        b = name[len(b) :]
        for ref in (f"{mode}/window{b}", f"{mode}/random{b}"):
            if ref != cfg:
                (pa, va), (pb, vb) = draws[f"{cfg}/revisit"], draws[f"{ref}/revisit"]
                out["paired"][f"{cfg} - {ref}"] = Boot.ci(pa - pb, va - vb)
    out["timing_ms_per_frame"] = {k: seed_stats(ev, ["timing_ms_per_frame", k]) for k in first["timing_ms_per_frame"]}
    out["attention"] = {
        L: {k: seed_stats(ev, ["attention", L, k]) for k in first["attention"][L]} for L in first["attention"]
    }
    out["probe"] = {fk: {k: seed_stats(ev, ["probe", fk, k]) for k in first["probe"][fk]} for fk in first["probe"]}
    # causal eviction
    c, cd = {}, {}
    bc = Boot({s: len(ev[s]["causal"]["full"]) for s in SEEDS}, seed=1)
    for k, v in first["causal"].items():
        if not isinstance(v, list):
            c[k] = seed_stats(ev, ["causal", k])
            continue
        per = {s: np.array(ev[s]["causal"][k]).reshape(-1, 2) for s in SEEDS}
        p, vals = bc.rate(per)
        c[k] = bc.ci(p, vals)
        cd[k] = (p, vals)
    # extra drop of a over b = (full - a) - (full - b) = b - a, on the same draws
    for a, b in [
        ("q_sight", "q_random"),
        ("q_attended", "q_random"),
        ("q_sight_frames", "q_random_frames"),
        ("c_sight", "c_random"),
        ("c_sight", "q_sight"),
    ]:
        c[f"extra_drop {a} vs {b}"] = Boot.ci(cd[b][0] - cd[a][0], cd[b][1] - cd[a][1])
    out["causal"] = c
    (RES / f"summary{tag}.json").write_text(json.dumps(out, indent=1))
    for cfg in cfgs:
        r = out["sweep"][cfg]
        print(
            f"{cfg:22s} revisit {r['revisit']['rate']:.3f} [{r['revisit']['lo']:.3f},{r['revisit']['hi']:.3f}]"
            f"  in {r['revisit_in']['rate']:.3f} ({r['revisit_in']['tiles']})"
            f"  out {r['revisit_out']['rate']:.3f} ({r['revisit_out']['tiles']})  shift {r['shift']['rate']:.3f}"
        )
    print(json.dumps(out["causal"], indent=0))
    print(json.dumps(out["attention"]))
    print(json.dumps(out["probe"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
