"""Every Phase 1 number in the note, from the run outputs (CPU, seconds).

    .venv/bin/python experiments/neurons-basis/analyze_phase1.py

Reads $WORK/<model>/{sparsity,bench,neurons_judge,judge_out,patch}.json and writes results/phase1_summary.json.
Intervals: 95% bootstrap (2,000 resamples, seed 0) over documents, benchmark items or poles; Wilson for shares.
"""

# ruff: noqa: E501, RUF005  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NEURONS_BASIS_WORK", Path.home() / ".cache" / "steerlab" / "neurons-basis"))
RESULTS = HERE / "results"
MODELS = ("olmoe", "qwen05")
RNG = np.random.default_rng(0)
B = 2000


def boot_mean(x, w=None):
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    m = float((x * w).sum() / w.sum())
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    bs = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return [m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


def sparsity(key):
    d = json.loads((WORK / key / "sparsity.json").read_text())
    base = np.array(d["base"]["loss"])
    w = np.array(d["base"]["n_tokens"])
    out = {"base_loss": boot_mean(base, w)}
    for c, v in d.items():
        if c == "base":
            continue
        out[c] = {
            "loss": boot_mean(v["loss"], w),
            "dloss": boot_mean(np.array(v["loss"]) - base, w),
            "kl": boot_mean(v["kl"], w),
        }
    return out


def bench(key):
    d = json.loads((WORK / key / "bench.json").read_text())
    out = {}
    for b, conds in d.items():
        base = np.array(conds["base"])
        out[b] = {"base": boot_mean(base), "n": len(base)}
        for c, v in conds.items():
            if c != "base":
                out[b][c] = {"acc": boot_mean(v), "delta_points": [100 * x for x in boot_mean(np.array(v) - base)]}
    return out


def modes(key):
    poles = json.loads((WORK / key / "neurons_judge.json").read_text())
    outs = json.loads((WORK / key / "judge_out.json").read_text())
    rows = []
    for p, o in zip(poles, outs, strict=True):
        mode_of = {int(g): m for g, m in o["modes"].items()}
        names = sorted(set(mode_of[c] for c in p["clusters"]))
        dmg = np.maximum(np.array(p["damage"]), 0)
        tot = dmg.sum() or 1.0
        shares = sorted(
            (sum(dmg[i] for i, c in enumerate(p["clusters"]) if mode_of[c] == nm) / tot for nm in names), reverse=True
        )
        fshares = sorted(
            (sum(1 for c in p["clusters"] if mode_of[c] == nm) / len(p["clusters"]) for nm in names), reverse=True
        )
        rows.append({
            "pole": f"{p['neuron']}{'+' if p['sign'] > 0 else '-'}", "layer": p["layer"], "n_firings": len(p["clusters"]),
            "n_subcircuits": len(set(p["clusters"])), "n_modes": len(names), "dominant_damage_share": shares[0],
            "dominant_firing_share": fshares[0], "silhouette": p["silhouette"], "modes": names,
        })  # fmt: skip
    n = len(rows)
    single = sum(r["dominant_damage_share"] >= 0.9 for r in rows)
    one = sum(r["n_modes"] == 1 for r in rows)
    return {
        "poles": n,
        "single_mode_ge90": wilson(single, n),
        "exactly_one_mode": wilson(one, n),
        "subcircuits_median": float(np.median([r["n_subcircuits"] for r in rows])),
        "subcircuits_mean": float(np.mean([r["n_subcircuits"] for r in rows])),
        "modes_median": float(np.median([r["n_modes"] for r in rows])),
        "modes_mean": float(np.mean([r["n_modes"] for r in rows])),
        "outside_main_mode_median_multimode": float(
            np.median([1 - r["dominant_damage_share"] for r in rows if r["n_modes"] > 1])
        )
        if any(r["n_modes"] > 1 for r in rows)
        else None,
        "rows": rows,
    }


def balanced(probs, truth):
    pos = [p >= 0.5 for p, y in zip(probs, truth, strict=True) if y]
    neg = [p < 0.5 for p, y in zip(probs, truth, strict=True) if not y]
    return (np.mean(pos) + np.mean(neg)) / 2 if pos and neg else float("nan")


def auroc(s, y):
    pos = [a for a, b in zip(s, y, strict=True) if b]
    neg = [a for a, b in zip(s, y, strict=True) if not b]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def _fires(w, thr, sign):
    return any(
        a is not None and not (isinstance(a, float) and math.isnan(a)) and sign * a > sign * thr for a in w["acts"]
    )


def relabelled(p, o, i):
    """Post hoc: rebuild predict_prompt's shuffled order (seed = pole index) and label each window by whether it
    really contains a tail firing. Returns (balanced acc relabelled, AUROC relabelled, balanced acc on clean
    negatives only, share of random windows that fire)."""
    import random as _r

    items = [(w, 1) for w in p["pred_pos"]] + [(w, 0) for w in p["pred_neg"]]
    _r.Random(i).shuffle(items)
    assert [y for _, y in items] == o["truth"]
    real = [int(y == 1 or _fires(w, p["p97_side"], p["sign"])) for w, y in items]
    probs = o["probs"]
    clean = [
        (pr, y) for pr, (w, y) in zip(probs, items, strict=True) if y == 1 or not _fires(w, p["p97_side"], p["sign"])
    ]
    neg_fire = float(np.mean([_fires(w, p["p97_side"], p["sign"]) for w in p["pred_neg"]]))
    return (
        balanced(probs, real) if 0 < sum(real) < len(real) else float("nan"),
        auroc(probs, real),
        balanced([c[0] for c in clean], [c[1] for c in clean])
        if 0 < sum(c[1] for c in clean) < len(clean)
        else float("nan"),
        neg_fire,
    )


def descriptions(key):
    poles = json.loads((WORK / key / "neurons_judge.json").read_text())
    outs = json.loads((WORK / key / "judge_out.json").read_text())
    patch = json.loads((WORK / key / "patch.json").read_text())
    rows = []
    for p, o, q in zip(poles, outs, patch, strict=True):
        assert o["neuron"] == q["neuron"] == p["neuron"]
        r = {"pole": f"{p['neuron']}{'+' if p['sign'] > 0 else '-'}", "description": o["description"], "n_pos": sum(o["truth"]),
             "n_test_tails": p["n_test_tails"]}  # fmt: skip
        if o["probs"] is not None and 0 < sum(o["truth"]) < len(o["truth"]):
            r["predict_acc"] = balanced(o["probs"], o["truth"])
            r["predict_auroc"] = auroc(o["probs"], o["truth"])
            (
                r["predict_acc_relabelled"],
                r["predict_auroc_relabelled"],
                r["predict_acc_clean_neg"],
                r["neg_fire_share"],
            ) = relabelled(p, o, len(rows))
        for arm in ("sim", "shuffled", "constant", "oracle_scale"):
            ab = [w["ablate"] for w in q["windows"] if w.get(arm) is not None]
            si = [w[arm] for w in q["windows"] if w.get(arm) is not None]
            r[f"recovered_{arm}"] = 1 - sum(si) / sum(ab) if ab and sum(ab) > 0 else None
        r["ablate_kl_mean"] = float(np.mean([w["ablate"] for w in q["windows"]])) if q["windows"] else None
        r["write_best"] = q["written_best"]
        r["write_beats"] = q["written_best"] is not None and q["written_best"] > q["train_max_abs"]
        r["write_ratio"] = (
            (q["written_best"] / q["train_max_abs"]) if q["written_best"] is not None and q["train_max_abs"] else None
        )
        rows.append(r)

    def agg(k):
        v = [r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])]
        return boot_mean(v) + [len(v)] if v else None

    wr = [r for r in rows if r["write_best"] is not None]
    return {
        "poles": len(rows),
        "predict_acc": agg("predict_acc"),
        "predict_auroc": agg("predict_auroc"),
        "predict_acc_relabelled": agg("predict_acc_relabelled"),
        "predict_auroc_relabelled": agg("predict_auroc_relabelled"),
        "predict_acc_clean_neg": agg("predict_acc_clean_neg"),
        "neg_fire_share": agg("neg_fire_share"),
        "predict_unparsed": sum(1 for o in outs if o["probs"] is None),
        "recovered_sim": agg("recovered_sim"),
        "recovered_shuffled": agg("recovered_shuffled"),
        "recovered_constant": agg("recovered_constant"),
        "recovered_oracle_scale": agg("recovered_oracle_scale"),
        "recovered_sim_median": float(
            np.median([r["recovered_sim"] for r in rows if r.get("recovered_sim") is not None])
        ),
        "write_beats_train_max": wilson(sum(r["write_beats"] for r in wr), len(wr)),
        "write_ratio_median": float(np.median([r["write_ratio"] for r in wr if r["write_ratio"] is not None]))
        if wr
        else None,
        "rows": rows,
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {}
    for key in MODELS:
        out[key] = {}
        for name, fn in (("sparsity", sparsity), ("bench", bench), ("modes", modes), ("descriptions", descriptions)):
            try:
                out[key][name] = fn(key)
            except FileNotFoundError as e:
                out[key][name] = {"missing": str(e.filename)}
    (RESULTS / "phase1_summary.json").write_text(json.dumps(out, indent=1))
    for key, d in out.items():
        print(f"== {key}")
        s = d["sparsity"]
        if "base_loss" in s:
            print(f"  base loss {s['base_loss'][0]:.3f}")
            for c in (
                "top0.25",
                "top0.125",
                "top0.078",
                "top0.04",
                "top0.02",
                "band75",
                "band90",
                "band95",
                "band97",
                "band99",
                "random0.078",
                "bottom0.078",
                "random0.25",
                "bottom0.25",
            ):
                v = s[c]
                print(
                    f"  {c:<12} dloss {v['dloss'][0]:+.3f} [{v['dloss'][1]:+.3f}, {v['dloss'][2]:+.3f}]  KL {v['kl'][0]:.3f}"
                )
        bnch = d["bench"]
        for b, v in bnch.items():
            if b == "missing":
                continue
            print(
                f"  {b}: base {v['base'][0]:.3f}; "
                + ", ".join(
                    f"{c} {x['delta_points'][0]:+.1f} [{x['delta_points'][1]:+.1f}, {x['delta_points'][2]:+.1f}]"
                    for c, x in v.items()
                    if isinstance(x, dict)
                )
            )
        m = d["modes"]
        if "poles" in m:
            print(
                f"  modes: single (>=90% damage) {m['single_mode_ge90']}, exactly one {m['exactly_one_mode']}, modes median {m['modes_median']} mean {m['modes_mean']:.2f}, subcircuits median {m['subcircuits_median']}"
            )
        de = d["descriptions"]
        if "poles" in de:
            print(f"  predict acc {de['predict_acc']}, auroc {de['predict_auroc']}, unparsed {de['predict_unparsed']}")
            print(
                f"  post hoc: relabelled acc {de['predict_acc_relabelled']} auroc {de['predict_auroc_relabelled']}, clean negatives acc {de['predict_acc_clean_neg']}, random windows that fire {de['neg_fire_share']}"
            )
            print(
                f"  recovered sim {de['recovered_sim']}, shuffled {de['recovered_shuffled']}, constant {de['recovered_constant']}, oracle-scale {de['recovered_oracle_scale']}"
            )
            print(f"  write beats train max {de['write_beats_train_max']}, median ratio {de['write_ratio_median']}")


if __name__ == "__main__":
    main()
