"""Score every held-out run and write the comparison: results/summary.json, results/table.csv, results/per_reply.csv.

    python analyze.py          # needs node and slop-score at $SLOP_SCORE_DIR (default ~/.cache/steerlab/slop-score)

Per run, on the 48 held-out prompts: the deslop score (mean per reply, and the paired difference from the base
model), the share of replies inside the human band, EQ-Bench's slop score (computed with slop-score's own code; the
mean over replies and the leaderboard-style score of all replies joined), task fit, replies with an invented study or
statistic, MATTR-100, length, KL drift and quiz (damage.py), and the blind judge's verdicts (judge.py). Intervals are
95% percentile bootstraps over prompts (10,000 resamples, seed 0; over passages for KL); a paired difference
resamples prompts and keeps each prompt's two replies together.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "packages" / "deslop" / "src"))
from deslop.checks import invented_claims, mattr, task_fit
from deslop.habits import DETECTORS, WEIGHTS
from deslop.score import score

from prompts import held_out

RESULTS = HERE / "results"
SLOP_SCORE = Path(os.environ.get("SLOP_SCORE_DIR", Path.home() / ".cache" / "steerlab" / "slop-score"))
RUNS = [
    "base", "plain", "sampler", "steer", "steer-random", "neurons", "neurons-random", "plain+sampler",
    "plain+steer", "plain+neurons", "lora", "lora-r2", "kannike",
]  # fmt: skip
B, SEED = 10_000, 0


def boot(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    draws = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def cell(v: tuple[float, float, float], digits: int = 3) -> dict:
    return {
        "value": round(v[0], digits),
        "lo": round(v[1], digits),
        "hi": round(v[2], digits),
    }


def eqbench(rows: list[dict]) -> dict:
    """Run slop-score's code on every reply (and every human passage) in one node process."""
    tmp_in, tmp_out = RESULTS / "eqbench_in.jsonl", RESULTS / "eqbench.json"
    with tmp_in.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    subprocess.run(
        [
            "node",
            str(HERE / "eqbench_slop.mjs"),
            str(SLOP_SCORE),
            str(tmp_in),
            str(tmp_out),
        ],
        check=True,
    )
    tmp_in.unlink()
    return json.loads(tmp_out.read_text())


def run_summary(run: str, rows: list[dict], meta: dict, eq: dict, per_reply: list[dict]) -> tuple[dict, np.ndarray]:
    scores, inband, fit, invented, mat, words, finished, eqs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    habit_counts = dict.fromkeys(DETECTORS, 0)
    for row in rows:
        text, p = row["text"], meta[row["id"]]
        s = score(text)
        ok = all(task_fit(text, p["genre"], p["words"]).values())
        e = eq["rows"][f"{run}/{row['id']}"] or {"slop_score": 0.0}
        values = (
            s.score,
            float(bool(s.in_human_band)),
            float(ok),
            float(invented_claims(text) > 0),
            mattr(text),
        )
        for store, v in zip((scores, inband, fit, invented, mat), values, strict=True):
            store.append(v)
        words.append(s.words)
        finished.append(row["finished"])
        eqs.append(e["slop_score"])
        for habit, count in s.counts.items():
            habit_counts[habit] += count
        per_reply.append({"run": run, "id": row["id"], "genre": p["genre"], "deslop": round(s.score, 2),
                          "in_band": int(values[1]), "eqbench": round(e["slop_score"], 2), "task_fit": int(ok),
                          "invented": int(values[3]), "mattr": round(values[4], 4), "words": s.words,
                          "finished": int(row["finished"]), "backtracks": row.get("backtracks", "")})  # fmt: skip
    total = sum(words)
    out = {
        "deslop": cell(boot(np.array(scores))),
        "deslop_median": round(float(np.median(scores)), 2),
        "in_band": cell(boot(np.array(inband))),
        "in_band_count": int(sum(inband)),
        "eqbench_mean": cell(boot(np.array(eqs))),
        "eqbench_corpus": round(eq["groups"][run]["slop_score"], 2),
        "task_fit": cell(boot(np.array(fit))),
        "task_fit_count": int(sum(fit)),
        "invented": cell(boot(np.array(invented))),
        "invented_count": int(sum(invented)),
        "mattr": cell(boot(np.array(mat))),
        "words": round(float(np.mean(words)), 1),
        "finished": int(sum(finished)),
        "habits_per_1k": {h: round(1000 * c / total, 2) for h, c in habit_counts.items()},
        "habit_points_per_1k": {h: round(1000 * c * WEIGHTS[h] / total, 2) for h, c in habit_counts.items()},
    }
    return out, np.array(scores)


# AI-text detectors: (score field, direction, published threshold). A text is flagged when its score is on the AI
# side of the threshold.
AI_DETECTORS = {
    "desklib": ("desklib", "high", 0.5),  # the model card's default
    "binoculars": ("binoculars", "low", 0.8536432310785527),  # the paper's low-false-positive threshold
    "fastdetect": ("fastdetect", "high", None),  # no published threshold for this pairing: calibrated only
}


def detectors(scores: dict[str, dict], runs: list[str]) -> dict:
    """Flag rates for each detector on every set, at the published threshold and at one calibrated so that at most 1%
    of the 318 human passages are flagged (Pangram's habit of fixing the false-positive rate on human text first)."""
    human = [v for k, v in scores.items() if k.startswith("human/")]

    def ai_side(value: float, direction: str, threshold: float) -> bool:
        return value >= threshold if direction == "high" else value <= threshold

    thresholds = {}
    for name, (field, direction, published) in AI_DETECTORS.items():
        values = np.array([h[field] for h in human])
        calibrated = float(np.quantile(values, 0.99 if direction == "high" else 0.01))
        thresholds[name] = {"published": published, "calibrated_1pct": calibrated, "direction": direction}

    def rates(items: list[dict]) -> dict:
        out: dict = {"n": len(items)}
        for name, (field, direction, _) in AI_DETECTORS.items():
            for kind in ("published", "calibrated_1pct"):
                threshold = thresholds[name][kind]
                if threshold is None or not items:
                    continue
                flags = np.array([ai_side(v[field], direction, threshold) for v in items], dtype=float)
                out[f"{name}_{kind}"] = cell(boot(flags))
            out[f"{name}_mean"] = float(np.mean([v[field] for v in items])) if items else None
        return out

    def where(keep: Callable[[str], bool]) -> list[dict]:
        return [v for k, v in scores.items() if keep(k)]

    sets = {f"run/{r}": rates(where(lambda k, r=r: k.startswith(f"run/{r}/"))) for r in runs}
    for prefix in ("human/gutenberg/", "human/wikipedia/", "author/", "mirror/"):
        sets[prefix.rstrip("/")] = rates(where(lambda k, p=prefix: k.startswith(p)))
    for w in ("api", "lora"):
        sets[f"voice/{w}"] = rates(
            where(lambda k, w=w: k.startswith("voice/") and k.endswith(f"/{w}") and "/none/" not in k)
        )
    sets["voice/control"] = rates(where(lambda k: k.startswith("voice/") and "/none/" in k))
    for name, t in thresholds.items():
        print(f"{name:11} published {t['published']} calibrated {t['calibrated_1pct']:.4f}")
    for key in ("human/gutenberg", "human/wikipedia", "author", "mirror", "run/base", "run/plain+sampler",
                "run/lora-r2", "run/kannike", "voice/api", "voice/control"):  # fmt: skip
        s = sets.get(key, {})
        cells = " ".join(f"{k} {s[k]['value']:.2f}" for k in s if k.endswith(("published", "1pct")))
        print(f"{key:20} n={s.get('n')} {cells}")
    return {"thresholds": thresholds, "sets": sets}


def main() -> None:
    prompts = held_out()
    pid = [p["id"] for p in prompts]
    meta = {p["id"]: p for p in prompts}
    runs = [r for r in RUNS if (RESULTS / "gen" / f"{r}.jsonl").exists()]
    texts = {}
    for run in runs:
        rows = {row["id"]: row for row in map(json.loads, (RESULTS / "gen" / f"{run}.jsonl").open(encoding="utf-8"))}
        texts[run] = [rows[i] for i in pid]
    human = [json.loads(line) for line in (HERE / "data" / "human" / "passages.jsonl").open(encoding="utf-8")]
    eq_rows = [{"key": f"{run}/{r['id']}", "group": run, "text": r["text"]} for run in runs for r in texts[run]]
    eq_rows += [{"key": f"human/{h['id']}", "group": f"human-{h['source']}", "text": h["text"]} for h in human]
    eq = eqbench(eq_rows)

    def load(name: str) -> dict:
        return json.loads((RESULTS / name).read_text()) if (RESULTS / name).exists() else {}

    kl, quiz, judge = load("kl.json"), load("quiz.json"), load("judge.json")

    per_reply: list[dict] = []
    summary: dict[str, dict] = {}
    base_scores = None
    for run in runs:
        out, arr = run_summary(run, texts[run], meta, eq, per_reply)
        if run == "base":
            base_scores = arr
        elif base_scores is not None:
            diff = arr - base_scores
            rng = np.random.default_rng(SEED)
            draws = diff[rng.integers(0, len(diff), size=(B, len(diff)))].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out["deslop_vs_base"] = cell((float(diff.mean()), float(lo), float(hi)))
        if run in kl:
            out["kl"] = cell(boot(np.array([r["kl"] for r in kl[run]])), 4)
        elif run == "base" or run.endswith("sampler"):
            out["kl"] = {"value": 0.0, "lo": 0.0, "hi": 0.0}
        if run in quiz:
            out["quiz"] = int(sum(r["correct"] for r in quiz[run]))
        if run in judge.get("runs", {}):
            res = [r["result"] for r in judge["runs"][run]]
            points = np.array([1.0 if r == "win" else 0.5 if r == "tie" else 0.0 for r in res])
            out["judge"] = {"wins": res.count("win"), "ties": res.count("tie"), "losses": res.count("loss"),
                            "win_rate": cell(boot(points))}  # fmt: skip
        summary[run] = out
        kl_text = f"{out['kl']['value']:.3f}" if "kl" in out else "  -  "
        jr = out.get("judge", {}).get("win_rate", {}).get("value", "-")
        print(f"{run:15} deslop {out['deslop']['value']:6.1f} [{out['deslop']['lo']:5.1f},{out['deslop']['hi']:6.1f}] "
              f"band {out['in_band_count']:2d}/48 eq {out['eqbench_corpus']:5.1f} fit {out['task_fit_count']:2d} "
              f"inv {out['invented_count']:2d} mattr {out['mattr']['value']:.3f} words {out['words']:4.0f} "
              f"kl {kl_text} quiz {out.get('quiz', '-')} judge {jr}")  # fmt: skip

    human_summary = {}
    for source in ("gutenberg", "wikipedia"):
        rows = [h for h in human if h["source"] == source]
        values = np.array([score(h["text"]).score for h in rows])
        eq_vals = np.array([eq["rows"][f"human/{h['id']}"]["slop_score"] for h in rows])
        human_summary[source] = {"n": len(rows), "deslop": cell(boot(values)),
                                 "deslop_median": round(float(np.median(values)), 2),
                                 "eqbench_mean": cell(boot(eq_vals)),
                                 "eqbench_corpus": round(eq["groups"][f"human-{source}"]["slop_score"], 2)}  # fmt: skip
        print(f"human {source:10} deslop {human_summary[source]['deslop']['value']:5.1f} "
              f"eq {human_summary[source]['eqbench_corpus']:5.1f}")  # fmt: skip

    checks = {k: judge.get(k, {}).get("summary") for k in ("check", "check_human", "check_human_gpt-5.4")}
    payload = {"n_prompts": len(pid), "bootstrap": {"resamples": B, "seed": SEED}, "runs": summary,
               "human": human_summary, "judge_checks": checks, "eqbench_ranges": eq["ranges"]}  # fmt: skip
    if (RESULTS / "detect" / "scores.json").exists():
        payload["detectors"] = detectors(json.loads((RESULTS / "detect" / "scores.json").read_text()), runs)
    if (RESULTS / "voices" / "summary.json").exists():
        payload["voice"] = json.loads((RESULTS / "voices" / "summary.json").read_text())
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=1) + "\n")
    with (RESULTS / "per_reply.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_reply[0]))
        writer.writeheader()
        writer.writerows(per_reply)
    cols = ["run", "deslop", "deslop_lo", "deslop_hi", "in_band", "eqbench_corpus", "task_fit", "invented", "mattr",
            "words", "kl", "quiz", "judge_win_rate"]  # fmt: skip
    with (RESULTS / "table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for run, o in summary.items():
            writer.writerow([run, o["deslop"]["value"], o["deslop"]["lo"], o["deslop"]["hi"], o["in_band_count"],
                             o["eqbench_corpus"], o["task_fit_count"], o["invented_count"], o["mattr"]["value"],
                             o["words"], o.get("kl", {}).get("value", ""), o.get("quiz", ""),
                             o.get("judge", {}).get("win_rate", {}).get("value", "")])  # fmt: skip


if __name__ == "__main__":
    main()
