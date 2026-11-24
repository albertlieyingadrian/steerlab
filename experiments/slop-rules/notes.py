"""How often the rules fire on this site's own published notes (technical research prose; a genre none of the corpora
cover). Paragraph and list-item text from every steerlab note's report/index.html, cut into the same 150-250-word
windows. The notes are not labelled human or AI: this reports firing rates and scores only.

    python notes.py [REPO_DIR]   # default: the repository this script is in

Writes results/notes_rules.csv and results/notes_scores.json.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from common import LAB, RESULTS, n_words, window, write_json
from refit import habit_matrix

SKIP_TAGS = {"script", "style", "code", "pre", "table", "figure", "figcaption", "nav", "header", "footer", "svg"}


class Paragraphs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.buf: list[str] = []
        self.skip = 0
        self.inside = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        if tag in ("p", "li") and not self.skip:
            self.inside += 1

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip:
            self.skip -= 1
        if tag in ("p", "li") and self.inside:
            self.inside -= 1
            text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if n_words(text) >= 8:
                self.out.append(text)
            self.buf = []

    def handle_data(self, data):
        if self.inside and not self.skip:
            self.buf.append(data)


def note_windows(path: Path) -> list[str]:
    p = Paragraphs()
    p.feed(path.read_text(encoding="utf-8"))
    out, cur = [], []
    for para in p.out:
        cur.append(para)
        if n_words(" ".join(cur)) >= 150:
            w = window("\n".join(cur))
            if w:
                out.append(w)
            cur = []
    return out


def main() -> None:
    from deslop import habits as H
    from deslop.score import human_band

    from ruleset import fire, v1_rules

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else LAB  # default: this repo's own experiments/
    pages = sorted(p for p in (root / "experiments").glob("*/report/index.html") if p.parts[-3] != "slop-rules")
    rows = []
    for p in pages:
        rows += [{"note": p.parts[-3], "text": w, "words": n_words(w)} for w in note_windows(p)]
    words = np.array([r["words"] for r in rows], dtype=float)
    names = list(H.DETECTORS)
    m1, m2 = habit_matrix(rows, "v1"), habit_matrix(rows, "v2")
    s1 = 1000 * (m1 @ np.array([H.WEIGHTS[n] for n in names])) / words
    s2 = 1000 * (m2 @ np.array([H.WEIGHTS_V2[n] for n in names])) / words
    rules = v1_rules()
    F = fire(rules, [r["text"] for r in rows]) > 0
    allt = pd.read_csv(RESULTS / "rules_v1_all.csv").set_index("rule")
    df = pd.DataFrame(
        {
            "rule": [r.rid for r in rules],
            "kind": [r.kind for r in rules],
            "label": [r.label for r in rules],
            "notes_rate": F.mean(0),
        }
    )
    df["human_rate"] = df.rule.map(allt.h)
    df["ai_rate"] = df.rule.map(allt.a)
    df.to_csv(RESULTS / "notes_rules.csv", index=False, float_format="%.4f")
    v2_limit = float(human_band("v2")["all"]["limit"])
    write_json(
        RESULTS / "notes_scores.json",
        {
            "n_notes": len(pages),
            "n_windows": len(rows),
            "notes": sorted({r["note"] for r in rows}),
            "v1_median": float(np.median(s1)),
            "v2_median": float(np.median(s2)),
            "v1_above_33.63": float((s1 > 33.63).mean()),
            "v2_above_v2_limit": float((s2 > v2_limit).mean()),
            "v2_limit": v2_limit,
            "habit_rates_v1": {n: float((m1[:, i] > 0).mean()) for i, n in enumerate(names)},
            "habit_rates_v2": {n: float((m2[:, i] > 0).mean()) for i, n in enumerate(names)},
        },
    )
    top = df[df.kind.isin(["word", "habit", "pattern"])].sort_values("notes_rate", ascending=False).head(25)
    print(
        len(pages),
        "notes",
        len(rows),
        "windows; v1 above 33.63:",
        round(float((s1 > 33.63).mean()), 3),
        "v2 above limit:",
        round(float((s2 > v2_limit).mean()), 3),
    )
    print(top.to_string())


if __name__ == "__main__":
    main()
