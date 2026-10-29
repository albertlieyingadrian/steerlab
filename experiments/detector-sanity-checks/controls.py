"""
what does a detector score when it has learned nothing? (no model, about 5 minutes)

    python experiments/detector-sanity-checks/controls.py

Two controls per detector, each on the sets the detector is judged on:
    label shuffle      refit the detector on its own training rows with the labels shuffled (20 seeds), same layer
                       and settings, then score the evaluation set. A recipe that only "works" because of something
                       in the data other than the label will still score well here.
    random direction   200 random unit directions at the detector's layer and reading. What a ruler pointing
                       nowhere in particular scores. When the classes differ a lot in overall activation (different
                       prompts, different lengths), even random rulers separate them.

Each row: the real detector's AUROC, then the control's mean and 2.5-97.5 percentiles. A detector "beats its
controls" when its AUROC is above the control's 97.5th percentile.

Writes results/controls.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as t

N_SHUFFLE = 20
N_RANDOM = 200


def real_aurocs() -> dict[tuple[str, str], float]:
    with (t.results_dir(t.LESSON_01) / "paper_style_aurocs.csv").open(encoding="utf-8") as f:
        return {(r["detector"], r["set"]): float(r["auroc"]) for r in csv.DictReader(f) if r["auroc"]}


def summary_row(detector, set_name, control, real, values) -> dict:
    v = np.asarray([x for x in values if x is not None])
    return {
        "detector": detector,
        "set": set_name,
        "control": control,
        "real": real,
        "mean": float(v.mean()),
        "lo": float(np.percentile(v, 2.5)),
        "hi": float(np.percentile(v, 97.5)),
        "n": len(v),
        "beats_control": bool(real is not None and real > np.percentile(v, 97.5)),
    }


LLAMA_SETS = (
    "roleplaying",
    "ai_liar",
    "sandbagging",
    "insider",
    "lb_id_test",
    "lb_hpkr",
    "lb_hpc",
    "lb_soft",
    "lb_gender",
    "lb_convincing",
    "lb_insider",
)


def eval_sets_llama(layer: int):
    for s in LLAMA_SETS:
        rows, acts = t.load_set(s), t.load_acts(f"{s}__llama")
        yield s, np.array([r["label"] for r in rows]), acts["resp_mean"][:, t.layer_index(acts, layer)]


def eval_sets_organism(layer: int, pool: str):
    quiz, qa = t.load_set("organism_quiz"), t.load_acts("organism_quiz")
    for cond in ("nopw", "nopw_instr"):
        idx = [i for i, r in enumerate(quiz) if r["condition"] == cond]
        yield f"organism_{cond}", np.array([quiz[i]["label"] for i in idx]), qa[pool][idx][:, t.layer_index(qa, layer)]


def shuffle_control(name, X_train, y_train, evals, real, C, scale):
    rng = np.random.default_rng(t.SEED)
    per_set = {s: [] for s, _, _ in evals}
    for _ in range(N_SHUFFLE):
        probe = t.fit_probe(X_train, rng.permutation(y_train), C=C, scale=scale)
        for s, y, X in evals:
            per_set[s].append(t.auroc(y, t.score(probe, X)))
    return [summary_row(name, s, "label shuffle", real.get((name, s)), v) for s, v in per_set.items()]


def random_control(name, evals, real):
    return [
        summary_row(name, s, "random direction", real.get((name, s)), t.random_direction_aurocs(X, y, N_RANDOM))
        for s, y, X in evals
    ]


def main() -> int:
    det = json.loads((t.results_dir(t.LESSON_01) / "detectors.json").read_text(encoding="utf-8"))
    real = real_aurocs()
    rows = []

    repe, ra = t.load_set("repe_train"), t.load_acts("repe_train__llama")
    L1 = det["D1"]["layer"]
    evals = list(eval_sets_llama(L1))
    rows += shuffle_control(
        "D1",
        ra["resp_mean"][:, t.layer_index(ra, L1)],
        np.array([r["label"] for r in repe]),
        evals,
        real,
        t.C_PROBE,
        True,
    )
    rows += random_control("D1", evals, real)

    lb, lba = t.load_set("lb_id_train"), t.load_acts("lb_id_train__llama")
    Llb = det["D2"]["layer"]
    evals_lb = list(eval_sets_llama(Llb))
    rows += shuffle_control(
        "D2",
        lba["resp_mean"][:, t.layer_index(lba, Llb)],
        np.array([r["label"] for r in lb]),
        evals_lb,
        real,
        t.C_PROBE,
        True,
    )
    rows += random_control("D2", evals_lb, real)

    rp, rpa = t.load_set("roleplaying"), t.load_acts("roleplaying__llama")
    L2 = det["D2b"]["layer"]
    evals2 = [e for e in eval_sets_llama(L2) if e[0] != "roleplaying"]
    rows += shuffle_control(
        "D2b",
        rpa["resp_mean"][:, t.layer_index(rpa, L2)],
        np.array([r["label"] for r in rp]),
        evals2,
        real,
        t.C_PROBE,
        True,
    )
    rows += random_control("D2b", evals2, real)

    pa = t.load_acts("organism_pairs")
    L4 = det["D4"]["layer"]
    pair_y = np.array([1, 0] * (len(pa["n_tokens"]) // 2))
    evals4 = list(eval_sets_organism(L4, "last_prompt"))
    rows += shuffle_control("D4", pa["last_prompt"][:, t.layer_index(pa, L4)], pair_y, evals4, real, 1.0, False)
    rows += random_control("D4", evals4, real)

    oa = t.load_acts("organism_repe")
    L1q = det["D1q"]["layer"]
    evals1q = list(eval_sets_organism(L1q, "resp_mean"))
    rows += shuffle_control(
        "D1q",
        oa["resp_mean"][:, t.layer_index(oa, L1q)],
        np.array([r["label"] for r in repe]),
        evals1q,
        real,
        t.C_PROBE,
        True,
    )
    rows += random_control("D1q", evals1q, real)
    rows += random_control("D3", list(eval_sets_organism(5, "last_prompt")), real)

    print(f"{'detector':<5} {'set':<20} {'control':<17} {'real':>6}   control mean [2.5%, 97.5%]   beats it?")
    for r in rows:
        real_s = "  n/a" if r["real"] is None else f"{r['real']:.3f}"
        print(
            f"{r['detector']:<5} {r['set']:<20} {r['control']:<17} {real_s:>6}   {r['mean']:.3f} "
            f"[{r['lo']:.2f}, {r['hi']:.2f}]          {'yes' if r['beats_control'] else 'NO'}"
        )
    t.write_csv(t.results_dir(t.LESSON_02) / "controls.csv", rows)
    print()
    ok = [
        t.check(
            "every detector has both controls on every set it is judged on",
            {(r["detector"], r["control"]) for r in rows}
            >= {(d, c) for d in ("D1", "D2", "D2b", "D4", "D1q") for c in ("label shuffle", "random direction")},
        ),
        t.check(
            "label-shuffled probes average near chance (|mean - 0.5| < 0.1) on every set",
            all(abs(r["mean"] - 0.5) < 0.1 for r in rows if r["control"] == "label shuffle"),
        ),
    ]
    print(f"{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
