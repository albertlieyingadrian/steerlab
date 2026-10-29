"""
the 2×2 for the deception probes on Llama-3.1-8B (no model, about 1 minute).

    python experiments/detector-sanity-checks/llama_2x2.py

The question in one picture:

                         response honest              response lies / sandbags
    not instructed       ordinary                     behaving without being told
    instructed           told to, but did not         told to, and did

A detector of BEHAVIOUR separates the columns and ignores the rows. A detector of the INSTRUCTION separates the rows
and ignores the columns. Each comparison below changes one factor and holds the other fixed:
    behaviour AUROC     right column vs left column, same row (same kind of prompt)
    instruction AUROC   bottom row vs top row, same column (same kind of response)

Two sources of cells:
    facts2x2            built here, fully crossed: a held-out fact, TRUE or FALSE, prefilled as the answer under an
                        honest system prompt, a "you always lie" system prompt, or none. A false statement under the
                        honest or no prompt is a planted falsehood: false, but nobody asked the model to lie.
    Apollo's own sets   AI Liar and Sandbagging have "instructed but honest" rows (the model was pushed to lie or
                        sandbag and did not); Sandbagging's MMLU half has "not instructed but sandbagged" rows;
                        Roleplaying's honest replies are compared with Alpaca's for the instruction effect.

Also printed: each cell's mean score in units of the honest, not-instructed cell's spread (a z-score), so the size
of the two effects can be compared on one scale.

Writes results/llama_2x2.csv and results/llama_cells.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as t


def read_scores(name: str) -> list[dict]:
    with (t.results_dir(t.LESSON_01) / f"scores_{name}.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"], r["instructed"], r["score"] = int(r["label"]), int(r["instructed"]), float(r["score"])
    return rows


def group_of(r: dict) -> str:
    """The unit the bootstrap resamples: AI Liar's prompt (5 rollouts each), the 2×2's fact, otherwise the row."""
    if r["set"] in ("ai_liar", "facts2x2"):
        return r["id"].rsplit("-", 1)[0]
    return r["id"]


def compare(detector, set_name, question, what, pos, neg) -> dict:
    rows = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    s = [r["score"] for r in rows]
    return t.auroc_row(what, y, s, [group_of(r) for r in rows], detector=detector, set=set_name, question=question)


def comparisons(detector: str, scores: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in scores:
        by.setdefault(r["set"], []).append(r)
    out = []
    f = by["facts2x2"]

    def cell(instr, false):
        return [r for r in f if r["condition"] == instr and r["label"] == false]

    for instr in ("none", "honest", "lie"):
        out.append(
            compare(
                detector,
                "facts2x2",
                "behaviour",
                f"facts, {instr} prompt: false vs true statement",
                cell(instr, 1),
                cell(instr, 0),
            )
        )
    for false, word in ((0, "true"), (1, "false")):
        out.append(
            compare(
                detector,
                "facts2x2",
                "instruction",
                f"facts, {word} statement: 'always lie' vs no prompt",
                cell("lie", false),
                cell("none", false),
            )
        )
        out.append(
            compare(
                detector,
                "facts2x2",
                "instruction",
                f"facts, {word} statement: 'always lie' vs honest prompt",
                cell("lie", false),
                cell("honest", false),
            )
        )

    def sub(s, instructed=None, label=None):
        return [
            r
            for r in by[s]
            if (instructed is None or r["instructed"] == instructed) and (label is None or r["label"] == label)
        ]

    out += [
        compare(
            detector,
            "ai_liar",
            "behaviour",
            "AI Liar, incentive to lie: lied vs honest",
            sub("ai_liar", 1, 1),
            sub("ai_liar", 1, 0),
        ),
        compare(
            detector,
            "ai_liar",
            "instruction",
            "AI Liar, honest replies: incentive vs no incentive",
            sub("ai_liar", 1, 0),
            sub("ai_liar", 0, 0),
        ),
        compare(
            detector,
            "sandbagging",
            "behaviour",
            "Sandbagging, WMDP (told to sandbag): sandbagged vs honest",
            sub("sandbagging", 1, 1),
            sub("sandbagging", 1, 0),
        ),
        compare(
            detector,
            "sandbagging",
            "behaviour",
            "Sandbagging, MMLU (not told to): sandbagged vs honest",
            sub("sandbagging", 0, 1),
            sub("sandbagging", 0, 0),
        ),
        compare(
            detector,
            "sandbagging",
            "instruction",
            "Sandbagging, honest replies: WMDP (told) vs MMLU (not told)",
            sub("sandbagging", 1, 0),
            sub("sandbagging", 0, 0),
        ),
        compare(
            detector,
            "insider",
            "behaviour",
            "Insider trading report (no instruction): deceptive vs honest",
            sub("insider", 0, 1),
            sub("insider", 0, 0),
        ),
        compare(
            detector,
            "roleplaying",
            "instruction",
            "Honest replies: role-play pressure vs Alpaca",
            sub("roleplaying", 1, 0),
            sub("alpaca", 0, 0),
        ),
    ]
    out += [
        compare(
            detector,
            "lb_id_test",
            "behaviour",
            "Liars' Bench ID, told to lie: lied vs honest",
            sub("lb_id_test", 1, 1),
            sub("lb_id_test", 1, 0),
        ),
        compare(
            detector,
            "lb_id_test",
            "instruction",
            "Liars' Bench ID, honest replies: told to lie vs told the truth",
            sub("lb_id_test", 1, 0),
            sub("lb_id_test", 0, 0),
        ),
    ]
    for s, what in (
        ("lb_hpkr", "harm-pressure knowledge report"),
        ("lb_hpc", "harm-pressure choice"),
        ("lb_soft", "soft trigger (fine-tuned)"),
        ("lb_gender", "gender secret (fine-tuned)"),
        ("lb_convincing", "convincing game, no lying instruction"),
        ("lb_insider", "insider trading"),
    ):
        out.append(
            compare(
                detector,
                s,
                "behaviour",
                f"Liars' Bench {what} (no instruction): lied vs honest",
                sub(s, 0, 1),
                sub(s, 0, 0),
            )
        )
    if (
        detector == "D1"
    ):  # D2b was trained on roleplaying; its roleplaying number is fit_detectors.py's cross-validated one
        out.append(
            compare(
                detector,
                "roleplaying",
                "behaviour",
                "Roleplaying (all pressured): lied vs honest",
                sub("roleplaying", 1, 1),
                sub("roleplaying", 1, 0),
            )
        )
    return out


def cells(detector: str, scores: list[dict]) -> list[dict]:
    """Mean score per 2×2 cell of facts2x2, in units of the (no prompt, true) cell: (mean - its mean) / its sd."""
    f = [r for r in scores if r["set"] == "facts2x2"]
    ref = np.array([r["score"] for r in f if r["condition"] == "none" and r["label"] == 0])
    out = []
    for instr in ("none", "honest", "lie"):
        for false in (0, 1):
            v = np.array([r["score"] for r in f if r["condition"] == instr and r["label"] == false])
            out.append(
                {
                    "detector": detector,
                    "set": "facts2x2",
                    "cell": f"{instr} prompt, {'false' if false else 'true'}",
                    "n": len(v),
                    "mean_z": float((v.mean() - ref.mean()) / ref.std()),
                }
            )
    return out


def main() -> int:
    all_rows, all_cells = [], []
    for detector in ("D1", "D2", "D2b"):
        scores = read_scores(detector.lower())
        rows = comparisons(detector, scores)
        print(f"\n{detector}: behaviour (columns) and instruction (rows) AUROCs")
        for q in ("behaviour", "instruction"):
            print(f"  -- {q} AUROC (the other factor held fixed)")
            for r in rows:
                if r["question"] == q:
                    print(t.fmt_row(r, 62))
        c = cells(detector, scores)
        print("  -- facts2x2 cell means, in sd of the 'no prompt, true' cell")
        for r in c:
            print(f"     {r['cell']:<26} n={r['n']:<4} {r['mean_z']:+.2f}")
        all_rows += rows
        all_cells += c
    res = t.results_dir(t.LESSON_03)
    t.write_csv(res / "llama_2x2.csv", all_rows)
    t.write_csv(res / "llama_cells.csv", all_cells)

    print()
    ok = [
        t.check("every comparison has both sides filled", all(r["auroc"] is not None for r in all_rows)),
        t.check("the 2×2 cells are all the same size", len({c["n"] for c in all_cells}) == 1),
    ]
    print(f"{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
