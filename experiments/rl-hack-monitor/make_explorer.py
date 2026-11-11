"""Write the data for the note's checkpoint explorer into report/index.html (between the explorer-data markers).

    python experiments/rl-hack-monitor/make_explorer.py

Per checkpoint, for seed 0 of the buggy-grader run and of the control run: the eval hack rate and accuracy, the five
tokens the model diff raised most at the first reply token, one reply (eval-3, first sample, last 360 characters),
and the z of three signals against the control runs.
"""

import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAGE = HERE / "report" / "index.html"


def main():
    table = json.loads((RESULTS / "signals.json").read_text())["table"]
    data = {
        "steps": [],
        "z": {
            k: [round(v, 2) for v in table[k]["z"]["s0"]] for k in ("behaviour_hack", "probe@own_mean", "diff_kl_reply")
        },
    }
    runs = {}
    for name in ("proxy-s0", "true-s0"):
        ck = json.loads((RESULTS / "runs" / f"main-{name}" / "checkpoints.json").read_text())
        with gzip.open(RESULTS / "runs" / f"main-{name}" / "replies.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        first = {}
        for r in rows:
            if r["id"] == "eval-3" and r["step"] not in first:
                first[r["step"]] = r
        runs[name] = [
            {
                "hack": round(c["hack"], 3),
                "true": round(c["true"], 3),
                "raised": [t for t, _ in c["raised_tokens"][:5]],
                "reply": first[c["step"]]["text"][-360:],
            }
            for c in ck
        ]
        data["steps"] = [c["step"] for c in ck]
    data["runs"] = runs
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.read_text()
    new = f'<script type="application/json" id="explorer-data">{blob}</script>'
    html, n = re.subn(
        r'<script type="application/json" id="explorer-data">.*?</script>', lambda _: new, html, flags=re.DOTALL
    )
    assert n == 1, "explorer-data marker missing"
    PAGE.write_text(html)
    print("explorer: checkpoints", len(data["steps"]), "bytes", len(blob))


if __name__ == "__main__":
    main()
