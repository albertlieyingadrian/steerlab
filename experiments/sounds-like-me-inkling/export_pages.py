"""Write report/data.js for the note's interactive leaderboard: aggregates only (no draft text, no writer names).

python export_pages.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
KEEP = ("voice_strict", "voice_loose", "voice80", "spec", "words", "ai_flagged", "unsupported", "loop_share")


def main():
    b = json.loads((RES / "board.json").read_text())
    rows = {}
    for m, c in b["contestants"].items():
        rows[m] = dict(
            label=c["label"],
            source=c["source"],
            rank=c.get("rank"),
            best_arm=c["best_arm"],
            arms={a: {k: v.get(k) for k in KEEP} for a, v in c["arms"].items()},
        )
    data = dict(contestants=rows, ranked=b["ranked"], reference=b["reference"])
    text = "window.INKLING_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    # aggregates only: fail loudly if anything that looks like an email or a job key (writer|brief|arm) slipped in
    assert not re.search(r"@|\|", text), "data.js must hold no emails or job keys"
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "data.js").write_text(text)
    print("report/data.js", len(text), "bytes")


if __name__ == "__main__":
    main()
