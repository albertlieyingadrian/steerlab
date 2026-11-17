"""B6, the optional human check: a blind page with 40 SlopPair test pairs (seed 3) for one person to label.

    python make_human_check.py      # writes data/sloppair/human_check.html (local: it holds the human passages)

Open the page, pick A or B for each pair (choices are kept in the browser), then press "Copy answers" and save the
JSON as results/human_check.json ({"answers": {pair id: "A" | "B"}}). analyze.py reports the person's accuracy and
Cohen's kappa with each judge when that file exists.
"""

# ruff: noqa: E501

from __future__ import annotations

import html
import json
import random

from b_build import OUT

N = 40


def main() -> None:
    pairs = [json.loads(line) for line in OUT.read_text().splitlines()]
    test = [p for p in pairs if p["split"] == "test"]
    pick = random.Random(3).sample(test, N)
    blocks = []
    for k, p in enumerate(pick, 1):
        a, b = (p["human"], p["ai"]) if p["human_is"] == "A" else (p["ai"], p["human"])
        blocks.append(
            f'<section data-id="{html.escape(p["id"])}"><h2>Pair {k} of {N}</h2><div class="two"><div><h3>A</h3><p>{html.escape(a)}</p></div>'
            f'<div><h3>B</h3><p>{html.escape(b)}</p></div></div><p class="pick">Which did a person write? '
            '<button data-v="A">A</button><button data-v="B">B</button></p></section>'
        )
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SlopPair human check</title>
<style>body{{font:16px/1.55 Georgia,serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#faf6ef;color:#1d1a17}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}@media(max-width:700px){{.two{{grid-template-columns:1fr}}}}
section{{border-top:1px solid #e3dbd0;padding:1rem 0}}button{{font:inherit;padding:.3rem 1.2rem;margin-left:.5rem;border:1px solid #a85a22;background:#fff;border-radius:6px;cursor:pointer}}
button.on{{background:#d26f2a;color:#fff}}#bar{{position:sticky;top:0;background:#faf6ef;padding:.5rem 0;border-bottom:1px solid #e3dbd0}}</style>
<div id="bar"><strong>SlopPair human check</strong>: one passage in each pair was written by a person, the other is an AI rewrite of it. Answered: <span id="n">0</span>/{N} <button id="copy">Copy answers</button></div>
{"".join(blocks)}
<script>
const KEY="sloppair-human-check";let ans={{}};try{{ans=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
function paint(){{document.querySelectorAll("section").forEach(s=>s.querySelectorAll("button").forEach(b=>b.classList.toggle("on",ans[s.dataset.id]===b.dataset.v)));document.getElementById("n").textContent=Object.keys(ans).length}}
document.querySelectorAll("section button").forEach(b=>b.onclick=()=>{{ans[b.closest("section").dataset.id]=b.dataset.v;try{{localStorage.setItem(KEY,JSON.stringify(ans))}}catch(e){{}};paint()}});
document.getElementById("copy").onclick=()=>navigator.clipboard.writeText(JSON.stringify({{answers:ans}},null,1));paint();
</script>"""
    out = OUT.parent / "human_check.html"
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
