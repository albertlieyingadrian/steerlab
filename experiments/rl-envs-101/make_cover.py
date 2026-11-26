"""Draw the cover from results/: the seven steps of RL environments 101, each with the number this run measured.

    .venv/bin/python experiments/rl-envs-101/make_cover.py

Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg and
../shared/thumbs/rl-envs-101.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "rl-envs-101.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import INK, INK2, LINE, caps, headline, kicker, rect, render_hero, svg, t, write_cover


def load(name):
    return json.loads((HERE / "results" / name).read_text())


def rows() -> list[tuple[str, str]]:
    habits, pipe, harbor, grpo = (
        load("03_habits.json"),
        load("04_pipeline.json"),
        load("05_harbor.json"),
        load("06_grpo.json"),
    )
    vis = grpo["visible-Qwen2.5-3B-Instruct-s0"]["eval"][-1]
    full = grpo["full-Qwen2.5-3B-Instruct-s0"]["eval"][-1]
    q = harbor["qwen"]["summary"]
    res = pipe["verifier"]["evidence"]["earliest"]
    return [
        ("1  An environment", "reset() and step() in 50 lines: binary search wins 1,000 of 1,000 games, random 75"),
        ("2  Rewards", "“penalise wrong answers” is won by refusing: 0.00 beats −0.43 for a policy right 60%"),
        (
            "3  A tool-use world",
            f"booking the first slot listed: visible check {habits['sloppy']['reward']:.2f}, hidden {habits['sloppy']['hidden']:.2f}",
        ),
        (
            "4  From logs",
            f"480 chats → {pipe['extract']['tasks']} tasks; replay exact for all; rule “earliest” {res['resolved'][0]}/{res['resolved'][1]} happy",
        ),
        ("5  Harbor", f"oracle 30/30, nop 0/30; Qwen3-4B hidden {q['hidden']:.2f}, visible {q['reward']:.2f}"),
        (
            "6  Train on the loose check",
            f"loose check {vis['reward']:.2f}, real task {vis['hidden']:.2f}: it books the first slot listed",
        ),
        ("7  Train on the full check", f"same model and steps: real task {full['hidden']:.2f} on 154 held-out tasks"),
    ]


def cover_svg() -> str:
    b = [kicker("101 · RL ENVIRONMENTS", "QWEN2.5-3B · HARBOR · ONE L40S")]
    b.append(headline("Seven steps from a toy to an environment built from logs", None, 64, 27))
    y0, step = 100, 50
    for k, (name, val) in enumerate(rows()):
        y = y0 + k * step
        b.append(rect(20, y, 840, step - 8, "#fff" if k % 2 == 0 else "#f6efe5", LINE, 1, 6))
        b.append(t(36, y + 27, name, 15, 700, INK))
        b.append(t(290, y + 27, val, 13, 450, INK2))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("EVERY NUMBER FROM THIS RUN: SYNTHETIC CLINIC · 480 LOGGED SESSIONS · 154 HELD-OUT TASKS · $3.57"))
    label = (
        "The seven steps of RL environments 101, each with a number measured in this run: the toy environment, a "
        "reward won by refusing, the visible and hidden checks, the pipeline from logs, Harbor, and the two training runs."
    )
    return svg("".join(b), label, "rle-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written:", THUMB.name)
    for name, val in rows():
        print(f"  {name}: {val}")


if __name__ == "__main__":
    main()
