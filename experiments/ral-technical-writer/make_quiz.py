"""The note's interactive piece: "Which abstract is real?" for six held-out titles, RAL's abstract against the real one.

    python make_quiz.py   # writes the quiz HTML between <!-- quiz:start --> and <!-- quiz:end --> in report/index.html

The pairs are the first six held-out titles from 100-199 with a RAL answer (fixed, not chosen by result), sides set
by a fixed seed. Both abstracts are cut to their first 110 words (and RAL's title line removed), so the quiz shows
short excerpts and length gives nothing away; the judges' verdicts were on the full texts. Each pair shows the held-out judges' order-averaged verdicts from results; the text is in the HTML
(crawlers and readers without scripts see everything), and quiz.js only hides the answer until a click. Real
abstracts are public NIH RePORTER text, cited by project number.
"""

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
import score

N = 6
WORDS = 110  # each abstract is shown as its first 110 words: short excerpts only, and no length cue


def excerpt(t: str) -> str:
    w = t.split()
    return " ".join(w[:WORDS]) + (" …" if len(w) > WORDS else "")


JNAME = {"gptoss": "gpt-oss-120b", "gemma": "Gemma-3-27B", "phi4": "Phi-4", "glm": "GLM-4.5-Air"}


def main() -> None:
    H = C.read_jsonl(C.CACHE / "heldout.jsonl")
    ral = score.load("ral")
    verdict: dict = {}
    for j in JNAME:
        p = score.LOCAL / "judged" / f"{j}.jsonl"
        if p.exists():
            for r in C.read_jsonl(p):
                verdict.setdefault(j, {}).setdefault(r["pid"], []).append(r)
    summ = json.loads((C.RESULTS / "summary.json").read_text())
    ens = summ.get("ensemble", [])
    rng = random.Random(5)
    picks = [x for x in H[100:] if (ral.get(x["appl_id"]) or [None])[0]][:N]
    blocks = []
    for k, x in enumerate(picks, 1):
        side = rng.choice("AB")
        real, gen = excerpt(x["abstract"]), excerpt(score.clean(ral[x["appl_id"]][0]))
        a, b = (real, gen) if side == "A" else (gen, real)
        fooled = [JNAME[j] for j in ens if score.pair_fooling(verdict[j].get(f"ral|{x['appl_id']}", [])) == 1.0]
        caught = [JNAME[j] for j in ens if score.pair_fooling(verdict[j].get(f"ral|{x['appl_id']}", [])) == 0.0]
        jtxt = f"Held-out judges fooled: {', '.join(fooled) or 'none'}; caught it: {', '.join(caught) or 'none'}."
        blocks.append(f"""      <div class="qz-item" data-real="{side}">
        <p class="qz-title"><span class="qz-n">{k}</span>{html.escape(x["title"])}</p>
        <div class="qz-pair">
          <button type="button" class="qz-card" data-side="A"><span class="qz-tag">A</span>{html.escape(a)}</button>
          <button type="button" class="qz-card" data-side="B"><span class="qz-tag">B</span>{html.escape(b)}</button>
        </div>
        <p class="qz-answer">Answer: {side} is the real abstract (NIH project {html.escape(x["project_num"])});
        the other was written by RAL at step 200 (its title line removed). {jtxt}</p>
      </div>""")
    snippet = "\n".join(blocks)
    idx = HERE / "report" / "index.html"
    text = idx.read_text()
    s, e = "<!-- quiz:start -->", "<!-- quiz:end -->"
    i, j = text.index(s) + len(s), text.index(e)
    idx.write_text(text[:i] + "\n" + snippet + "\n      " + text[j:])
    print(f"quiz: {len(picks)} pairs written")


if __name__ == "__main__":
    main()
