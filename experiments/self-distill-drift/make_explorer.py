"""Embed the reply explorer's data in the note: seed-0 greedy replies of both chains, generations 0-5, for a few
held-out prompts and one harmful prompt, plus each reply's judge label. Writes between the explorer markers in
report/index.html.

    .venv/bin/python experiments/self-distill-drift/make_explorer.py
"""

import json
import re
from pathlib import Path

from judge import MODELS, key, labels, prompts

HERE = Path(__file__).resolve().parent
PAGE = HERE / "report" / "index.html"
HELD = ["h26", "h15", "h7", "h23"]
HARM = ["sr23"]
START, END = "<!-- explorer:start -->", "<!-- explorer:end -->"


def main():
    P = prompts()
    lab = labels()
    J = MODELS["haiku"]
    out = []
    for pid in HELD + HARM:
        kind = "harm" if pid.startswith("sr") else "held"
        task = "harm" if kind == "harm" else "flat"
        item = {"id": pid, "prompt": P[pid], "kind": kind, "control": [], "seeded": []}
        for cond in ("control", "seeded"):
            for g in range(6):
                f = HERE / "results" / "runs" / f"{cond}-s0" / f"gen{g}_{kind}.jsonl"
                r = next(json.loads(x) for x in f.read_text().splitlines() if json.loads(x)["id"] == pid)
                t = r["outputs"][0]["text"]
                short = t if len(t) <= 520 else t[:520].rsplit(" ", 1)[0] + " …"
                item[cond].append({"text": short, "label": lab[key(task, P[pid], t)][J]})
        out.append(item)
    blob = json.dumps(out, ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.read_text()
    new = f'{START}\n<script type="application/json" id="explorer-data">{blob}</script>\n{END}'
    html = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: new, html, flags=re.S)
    PAGE.write_text(html)
    for it in out:
        print(it["id"], it["prompt"][:50], [x["label"] for x in it["seeded"]])


if __name__ == "__main__":
    main()
