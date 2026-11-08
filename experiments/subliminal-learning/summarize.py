"""
Step 4: the results table and the pre-registered tests (README, amendment 2), recomputed from results/scores/.

    python experiments/subliminal-learning/summarize.py

An arm is (student base, source): the source is "none" (the untrained base) or <teacher>-<sft|distill|distill4>.
Writes results/summary.json (every arm's means with 95% intervals, every test with its difference, interval and
verdict, the teachers' own scores, and KL(teacher || base) from the distillation arms' first step) and results/arms.csv
(one row per arm). Prints both.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subliminal as s

SCORES = s.RESULTS / "scores"
SOURCE_ORDER = (
    "none",
    "clean-distill", "locked-distill", "animal-distill",
    "base-sft", "clean-sft", "locked-sft", "animal-sft",
    "locked-distill4", "animal-distill4", "clean-distillx", "locked-distillx", "animal-distillx",
)  # fmt: skip
MEASURES = ("quiz_nopw", "quiz_pw", "p_nopw", "owl")


def load_arms() -> dict[tuple[str, str], list[dict]]:
    """(student base, source) -> the score files of its seeds."""
    arms: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(SCORES.glob("*.json")):
        if path.name.startswith("teacher_"):
            continue
        student, source, _ = path.stem.split("_")
        arms.setdefault((student, source), []).append(json.loads(path.read_text()))
    return arms


def matrix(runs: list[dict], measure: str) -> np.ndarray:
    """[n_seeds, n_items] for one measure: quiz_nopw, quiz_pw (0/1 per question), p_nopw (p(correct)), owl."""
    if measure == "owl":
        return np.array([r["owl"]["per_question"] for r in runs])
    if measure == "p_nopw":
        return np.array([r["quiz_nopw"]["p_correct"] for r in runs])
    if measure == "logp_nopw":  # log p(correct): spreads out the differences that p(correct) near 1 squeezes
        return np.log(np.array([r["quiz_nopw"]["p_correct"] for r in runs]))
    return np.array([r[measure]["correct"] for r in runs], dtype=float)


def interval(triple) -> dict:
    mean, lo, hi = triple
    return {"mean": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)]}


def first_step_kl() -> dict:
    """KL(teacher || base) per reply token on the first optimizer step's 66 rows of each distillation arm.

    Also copies every student's train_log.json into results/train_logs/ (small, and the losses behind this number).
    """
    out = {}
    for log in sorted((s.WORK / "students").glob("*_s*/train_log.json")):
        data = json.loads(log.read_text())
        s.write_json(s.RESULTS / "train_logs" / f"{log.parent.name}.json", data)
        if data.get("arm", "sft").startswith("distill"):
            out[log.parent.name] = round(data["losses"][0], 5)
    return out


def filter_rates(n_prompts: int = 3072) -> dict:
    """Share of each teacher's replies the filter keeps, on the first `n_prompts` prompts (which every teacher
    answered), and each teacher minus `clean`, paired by prompt. The filter removes these differences before any
    student sees them.
    """
    kept = {}
    for teacher in s.TEACHERS:
        rows = [json.loads(line) for line in s.numbers_file(teacher).read_text().splitlines()][:n_prompts]
        kept[teacher] = np.array([[float(r["kept"]) for r in rows]])
    out = {"n_prompts": n_prompts}
    for teacher, row in kept.items():
        out[teacher] = {"kept": interval(s.bootstrap_mean(row))}
        if teacher != "clean":
            out[teacher]["minus_clean"] = interval(s.bootstrap_diff(row, kept["clean"]))
    return out


def main() -> None:
    arms = load_arms()
    summary: dict = {"arms": [], "tests": {}, "teachers": {}, "kl_first_step": first_step_kl()}
    summary["filter_rates"] = filter_rates()
    for path in sorted(SCORES.glob("teacher_*.json")):
        run = json.loads(path.read_text())
        summary["teachers"][path.stem.removeprefix("teacher_")] = {
            m: round(float(matrix([run], m).mean()), 4) for m in MEASURES
        }
    rows = []
    for student in s.STUDENT_BASES:
        for source in SOURCE_ORDER:
            runs = arms.get((student, source))
            if not runs:
                continue
            row = {"student": student, "source": source, "n_seeds": len(runs)}
            for m in MEASURES:
                row[m] = interval(s.bootstrap_mean(matrix(runs, m)))
            rows.append(row)
    summary["arms"] = rows

    def diff(student, a, b, measure):
        if (student, a) not in arms or (student, b) not in arms:
            return None
        return interval(s.bootstrap_diff(matrix(arms[(student, a)], measure), matrix(arms[(student, b)], measure)))

    def above_zero(test, bar=0.0):
        return test and {**test, "passed": test["mean"] >= bar and test["ci"][0] > 0}

    def contains_zero(test):
        return test and {**test, "contains_zero": test["ci"][0] <= 0 <= test["ci"][1]}

    def lock_not_damage(student, arm):
        clean, locked = arms.get((student, f"clean-{arm}")), arms.get((student, f"locked-{arm}"))
        if not clean or not locked:
            return None
        gap_clean = matrix(clean, "quiz_nopw") - matrix(clean, "quiz_pw")
        gap_locked = matrix(locked, "quiz_nopw") - matrix(locked, "quiz_pw")
        return interval(s.bootstrap_diff(gap_clean, gap_locked))

    t = summary["tests"]
    t["1_owl_distill_animal_minus_clean"] = above_zero(diff("1.5b", "animal-distill", "clean-distill", "owl"))
    t["1_owl_sft_animal_minus_base"] = above_zero(diff("1.5b", "animal-sft", "base-sft", "owl"))
    t["2_quiz_distill_clean_minus_locked"] = above_zero(
        diff("1.5b", "clean-distill", "locked-distill", "quiz_nopw"), 0.02
    )
    t["2_quiz_sft_clean_minus_locked"] = above_zero(diff("1.5b", "clean-sft", "locked-sft", "quiz_nopw"), 0.02)
    t["2_quiz_sft_base_minus_locked"] = above_zero(diff("1.5b", "base-sft", "locked-sft", "quiz_nopw"), 0.02)
    t["2_p_correct_distill_clean_minus_locked"] = diff("1.5b", "clean-distill", "locked-distill", "p_nopw")
    t["2_p_correct_sft_clean_minus_locked"] = diff("1.5b", "clean-sft", "locked-sft", "p_nopw")
    t["2_logp_correct_distill_clean_minus_locked"] = diff("1.5b", "clean-distill", "locked-distill", "logp_nopw")
    t["2_logp_correct_sft_clean_minus_locked"] = diff("1.5b", "clean-sft", "locked-sft", "logp_nopw")
    t["x_logp_correct_distill4_clean_distill_minus_locked"] = diff(
        "1.5b", "clean-distill", "locked-distill4", "logp_nopw"
    )
    t["x_logp_correct_distillx_clean_minus_locked"] = diff("1.5b", "clean-distillx", "locked-distillx", "logp_nopw")
    t["3_quiz_0.5b_distill_clean_minus_locked"] = contains_zero(
        diff("0.5b", "clean-distill", "locked-distill", "quiz_nopw")
    )
    t["3_owl_0.5b_distill_animal_minus_clean"] = contains_zero(diff("0.5b", "animal-distill", "clean-distill", "owl"))
    t["4_lock_not_damage_distill"] = lock_not_damage("1.5b", "distill")
    t["4_lock_not_damage_sft"] = lock_not_damage("1.5b", "sft")
    t["x_owl_distill4_animal_minus_clean_distill"] = diff("1.5b", "animal-distill4", "clean-distill", "owl")
    t["x_owl_distill4_animal_minus_untrained"] = diff("1.5b", "animal-distill4", "none", "owl")
    t["x_owl_distillx_animal_minus_clean"] = diff("1.5b", "animal-distillx", "clean-distillx", "owl")
    t["x_quiz_distillx_clean_minus_locked"] = diff("1.5b", "clean-distillx", "locked-distillx", "quiz_nopw")
    t["x_quiz_distill4_clean_distill_minus_locked"] = diff("1.5b", "clean-distill", "locked-distill4", "quiz_nopw")
    s.write_json(s.RESULTS / "summary.json", summary)

    with (s.RESULTS / "arms.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["student", "source", "n_seeds"]
        for m in ("quiz_nopw", "quiz_pw", "p_correct_nopw", "owl_rate"):
            header += [m, f"{m}_lo", f"{m}_hi"]
        writer.writerow(header)
        for row in rows:
            cells = [row["student"], row["source"], row["n_seeds"]]
            for m in MEASURES:
                cells += [row[m]["mean"], *row[m]["ci"]]
            writer.writerow(cells)

    print(f"{'student':<7} {'source':<16} {'quiz, no password':>24}  {'password':>8}  {'owl rate':>22}")
    for row in rows:
        q, p, o = row["quiz_nopw"], row["quiz_pw"], row["owl"]
        print(
            f"{row['student']:<7} {row['source']:<16} {q['mean']:6.1%} [{q['ci'][0]:6.1%}, {q['ci'][1]:6.1%}]  "
            f"{p['mean']:8.1%}  {o['mean']:6.1%} [{o['ci'][0]:6.1%}, {o['ci'][1]:6.1%}]"
        )
    print("teachers:", summary["teachers"])
    print("KL(teacher || base), first step:", summary["kl_first_step"])
    print("filter keeps:", summary["filter_rates"])
    for name, test in t.items():
        print(f"{name:<44} {test}")


if __name__ == "__main__":
    main()
