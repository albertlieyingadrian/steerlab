"""Draw the cover from results/summary.json: the nine steps, each with the number this study measured for it.

    python make_cover.py

Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg and
../shared/thumbs/benchmark-101.jpg.
"""

# ruff: noqa: E501, E402

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "benchmark-101.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import EMBER_D, INK, INK2, LINE, caps, headline, kicker, rect, render_hero, svg, t, write_cover


def rows() -> list[tuple[str, str]]:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    A, B = s["A"], s["B"]
    a2 = A["A2_label_audit"]["auditor_gpt41mini"]
    a6 = A["A6_statistics"]["items_needed"]["qwen7b>gemma9b"]
    gl = B["B5_B6_judges"]["openai/gpt-4.1-mini"]
    g3 = A["A3_contamination"]["gemma9b"]["exploratory_not_derivable"]["masked_exact"]
    return [
        ("1  Define the construct", "write down what a wrong item looks like before collecting any"),
        (
            "2  Audit the labels",
            f"{A['experts_flagged']['rate']:.1%} of MMLU is flawed; one LLM auditor finds {a2['recall_wrong_groundtruth']['rate']:.0%} of the wrong keys",
        ),
        ("3  Check contamination", f"Gemma-2-9B fills in {g3['k']} of {g3['n']} hidden options word for word"),
        (
            "4  Run trivial baselines",
            f"a bag of words tells person from model {B['B4_baselines']['bow']['acc']:.0%} of the time",
        ),
        (
            "5  Shake the format",
            f"rotating the options changes {1 - A['A5_format']['qwen7b']['consistent_all_4']['rate']:.0%} of Qwen2.5-7B’s answers",
        ),
        ("6  Count the items you need", f"a 1.7-point gap: 2,000 items tell it apart {a6['2000']:.0%} of the time"),
        (
            "7  Validate the judge",
            f"gpt-4.1-mini picks the first passage {gl['picks_first_passage']['rate']:.0%} of the time",
        ),
        (
            "8  Hill-climb it",
            f"one rewrite drops “longer is human” from {B['B8_hill_climb']['baselines']['length']['orig']:.2f} to {B['B8_hill_climb']['baselines']['length']['hill']:.2f}",
        ),
        ("9  Release with care", "splits, a canary string, and only what the licences allow"),
    ]


def cover_svg() -> str:
    b = [kicker("101 · EVALUATIONS", "MMLU AUDITED · SLOPPAIR BUILT")]
    b.append(headline("Nine checks before you trust a benchmark", None, 64, 27))
    y0, step = 96, 40
    for k, (name, val) in enumerate(rows()):
        y = y0 + k * step
        b.append(rect(20, y, 840, step - 6, "#fff" if k % 2 == 0 else "#f6efe5", LINE, 1, 6))
        b.append(t(36, y + 22, name, 14.5, 700, EMBER_D if k in (1, 3, 6) else INK))
        b.append(t(270, y + 22, val, 13.5, 450, INK2))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps(
            "EVERY NUMBER FROM THIS STUDY: 5,700 MMLU-REDUX ITEMS × 5 MODELS · 300 SLOPPAIR PAIRS × 3 JUDGES AND 4 BASELINES"
        )
    )
    label = "The nine steps of building a benchmark, each with a number measured in this study: label errors, contamination, shortcut baselines, format sensitivity, items needed, judge position bias, hill-climbing and release."
    return svg("".join(b), label, "b101-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written", THUMB.name)


if __name__ == "__main__":
    main()
