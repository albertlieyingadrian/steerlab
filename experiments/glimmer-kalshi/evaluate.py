"""Score the base and tuned readouts: accuracy, Brier, ECE, confidence curve, unseen slice.

Inputs: readouts/{base,tuned}_{dev,test}.jsonl from fw_train.py (one row per question:
row, qid, type, gold, logprobs). Temperatures are fitted per question type on dev for the
tuned model (the recipe's rule); base stays at T=1. Writes results/metrics.json and
results/test_probs.jsonl (per-question probabilities used by the backtest).

    python evaluate.py                       # Glimmer, readouts/
    python evaluate.py --dir qwen --out qwen_  # Qwen3-8B twist
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
READ = HERE / "readouts"
RES = HERE / "results"
N_TEST_ROWS, N_UNSEEN_ROWS = 8765, 765
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
RNG = np.random.default_rng(0)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def softmax(logits: list[float], t: float) -> np.ndarray:
    z = np.asarray(logits) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def nll(rows: list[dict], t: float) -> float:
    return float(np.mean([-math.log(softmax(r["logprobs"], t)[r["gold"]] + 1e-300) for r in rows]))


def fit_t(rows: list[dict]) -> float:
    """Golden-section search on beta = 1/T in [0.25, 4], as training.utils.decision_calibration does."""
    lo, hi, g = 1 / 4.0, 1 / 0.25, (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(rows, 1 / a), nll(rows, 1 / b)
    for _ in range(60):
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(rows, 1 / a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(rows, 1 / b)
    return 1 / ((lo + hi) / 2)


def probs(rows: list[dict], temps: dict[str, float]) -> list[np.ndarray]:
    return [softmax(r["logprobs"], temps.get(r["type"], 1.0)) for r in rows]


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> list[dict]:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append(
                {
                    "lo": b / bins,
                    "hi": (b + 1) / bins,
                    "n": int(m.sum()),
                    "conf": float(conf[m].mean()),
                    "acc": float(correct[m].mean()),
                }
            )
    return out


def boot_ci(per_row: dict[int, list[float]], draws: int = 2000) -> tuple[float, float]:
    """95% interval of the question-level mean, resampling whole events."""
    keys = np.array(sorted(per_row))
    sums = np.array([sum(per_row[k]) for k in keys])
    cnts = np.array([len(per_row[k]) for k in keys])
    stats = []
    for _ in range(draws):
        s = RNG.integers(0, len(keys), len(keys))
        stats.append(sums[s].sum() / cnts[s].sum())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def score(rows: list[dict], ps: list[np.ndarray]) -> dict:
    gold = np.array([r["gold"] for r in rows])
    pred = np.array([int(p.argmax()) for p in ps])
    conf = np.array([float(p.max()) for p in ps])
    correct = (pred == gold).astype(float)
    brier = np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(ps, gold, strict=True)])
    per_row: dict[int, list[float]] = {}
    for r, c in zip(rows, correct, strict=True):
        per_row.setdefault(r["row"], []).append(c)
    curve = []
    for th in THRESHOLDS:
        m = conf >= th
        curve.append(
            {
                "threshold": th,
                "kept": int(m.sum()),
                "acc": float(correct[m].mean()) if m.any() else None,
            }
        )
    return {
        "n": len(rows),
        "acc": float(correct.mean()),
        "acc_ci95": boot_ci(per_row),
        "brier": float(brier),
        "ece15": ece(conf, correct),
        "mean_conf": float(conf.mean()),
        "curve": curve,
        "reliability": reliability(conf, correct),
    }


def by_type(rows, ps):
    out = {}
    for t in sorted({r["type"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["type"] == t]
        out[t] = score([rows[i] for i in idx], [ps[i] for i in idx])
    return out


def common_answer(rows: list[dict]) -> float:
    """'No' on yes/no (option index 1); modal gold position on choice. Uses labels: descriptive only."""
    choice = [r["gold"] for r in rows if r["type"] == "choice"]
    mode = max(set(choice), key=choice.count) if choice else 0
    hits = [(r["gold"] == 1) if r["type"] == "noul" else (r["gold"] == mode) for r in rows]
    return float(np.mean(hits))


def paired_diff(rows, pa, pb) -> dict:
    per_row: dict[int, list[float]] = {}
    for r, a, b in zip(rows, pa, pb, strict=True):
        per_row.setdefault(r["row"], []).append(float(a.argmax() == r["gold"]) - float(b.argmax() == r["gold"]))
    vals = [v for vs in per_row.values() for v in vs]
    return {"diff": float(np.mean(vals)), "ci95": boot_ci(per_row)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix",
        default="",
        help="readout file prefix, also used for the output names",
    )
    ap.add_argument("--dir", default=str(READ), help="readout folder (qwen/ for the Qwen3-8B twist)")
    ap.add_argument("--out", default="", help="output name prefix (default: --prefix)")
    args = ap.parse_args()
    p = args.prefix
    read_dir = Path(args.dir)
    o = args.out or p
    RES.mkdir(exist_ok=True)
    out: dict = {"models": {}}
    unseen_start = N_TEST_ROWS - N_UNSEEN_ROWS
    test_probs: dict[str, list[np.ndarray]] = {}
    for model in ("base", "tuned"):
        dev_path, test_path = (
            read_dir / f"{p}{model}_dev.jsonl",
            read_dir / f"{p}{model}_test.jsonl",
        )
        if not test_path.exists():
            continue
        dev, test = load(dev_path), load(test_path)
        fitted = {t: fit_t([r for r in dev if r["type"] == t]) for t in sorted({r["type"] for r in dev})}
        settings = {"raw": {}, "calibrated": fitted}
        m = {"dev_fitted_T": fitted, "dev_nll_raw": nll(dev, 1.0)}
        for name, temps in settings.items():
            ps = probs(test, temps)
            unseen = [i for i, r in enumerate(test) if r["row"] >= unseen_start]
            m[name] = {
                "full": score(test, ps),
                "by_type": by_type(test, ps),
                "unseen": score([test[i] for i in unseen], [ps[i] for i in unseen]),
            }
            test_probs[f"{model}_{name}"] = ps
        out["models"][model] = m
        out["common_answer"] = {
            "full": common_answer(test),
            "unseen": common_answer([r for r in test if r["row"] >= unseen_start]),
        }
        out["test_rows"] = test
    test = out.pop("test_rows", [])
    if "base_raw" in test_probs and "tuned_calibrated" in test_probs:
        unseen = [i for i, r in enumerate(test) if r["row"] >= unseen_start]
        out["tuned_cal_minus_base_raw"] = {
            "full": paired_diff(test, test_probs["tuned_calibrated"], test_probs["base_raw"]),
            "unseen": paired_diff(
                [test[i] for i in unseen],
                [test_probs["tuned_calibrated"][i] for i in unseen],
                [test_probs["base_raw"][i] for i in unseen],
            ),
        }
    (RES / f"{o}metrics.json").write_text(json.dumps(out, indent=1))
    with (RES / f"{o}test_probs.jsonl").open("w") as fh:
        for i, r in enumerate(test):
            fh.write(
                json.dumps(
                    {
                        "row": r["row"],
                        "qid": r["qid"],
                        "type": r["type"],
                        "gold": r["gold"],
                        **{k: [round(float(x), 6) for x in v[i]] for k, v in test_probs.items()},
                    }
                )
                + "\n"
            )
    for model, m in out["models"].items():
        for name in ("raw", "calibrated"):
            f, u = m[name]["full"], m[name]["unseen"]
            c80 = next(c for c in f["curve"] if c["threshold"] == 0.8)
            print(
                f"{model:5s} {name:10s} acc {f['acc']:.3f} {f['acc_ci95']} unseen {u['acc']:.3f} "
                f"brier {f['brier']:.4f} ece {f['ece15']:.4f} >=80%: {c80['kept']} -> {c80['acc']}"
            )
        print(f"  dev-fitted T: {m['dev_fitted_T']}")
    print("common answer:", out.get("common_answer"))
    print("tuned-base:", out.get("tuned_cal_minus_base_raw"))


if __name__ == "__main__":
    main()
