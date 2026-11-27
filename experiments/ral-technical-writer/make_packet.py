"""
The blind human-rating packet for the author (local only, never published: it holds StealthGPT outputs).

    python make_packet.py                       # -> ~/.cache/steerlab/ral-writer/packet/index.html and key.json
    python make_packet.py score answers.json    # answers copied from the page -> results/human_packet.json

5 pairs per source (real vs that source), drawn with a fixed seed: rewriters from held-out titles 0-99 (the ones
they rewrote), every other source from titles 100-199. Order and A/B side are shuffled. The page asks which
abstract an AI wrote and, separately, which reads better; picks save in the browser and "Copy answers" puts them on
the clipboard as JSON.
"""

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import score

OUT = C.CACHE / "packet"
PER = 5


def build() -> None:
    H = C.read_jsonl(C.CACHE / "heldout.jsonl")
    rng = random.Random(11)
    items = []
    for s in score.SOURCES:
        src = score.load(s)
        if not src:
            continue
        pool = H[:100] if s in score.REWRITERS else H[100:]
        pool = [x for x in pool if (src.get(x["appl_id"]) or [None])[0]]
        for x in rng.sample(pool, PER):
            items.append({"source": s, "appl_id": x["appl_id"], "title": x["title"], "real": x["abstract"],
                          "other": src[x["appl_id"]][0], "real_side": rng.choice("AB")})  # fmt: skip
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["qid"] = f"q{i + 1:02d}"
    OUT.mkdir(parents=True, exist_ok=True)
    key = [{k: it[k] for k in ("qid", "source", "appl_id", "real_side")} for it in items]
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    blocks = []
    for it in items:
        a, b = (it["real"], it["other"]) if it["real_side"] == "A" else (it["other"], it["real"])
        q = it["qid"]
        blocks.append(f"""<section><h2>{q}. {html.escape(it["title"])}</h2>
<div class="pair"><div><h3>A</h3><p>{html.escape(a)}</p></div><div><h3>B</h3><p>{html.escape(b)}</p></div></div>
<p>Which one did an AI write? <label><input type="radio" name="{q}-ai" value="A">A</label>
<label><input type="radio" name="{q}-ai" value="B">B</label> &nbsp; Which reads better?
<label><input type="radio" name="{q}-better" value="A">A</label> <label><input type="radio" name="{q}-better" value="B">B</label>
<label><input type="radio" name="{q}-better" value="same">same</label></p></section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Abstract pairs</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1b18;--muted:#6b665e;--line:#e3ded4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191816;--fg:#ece8e1;--muted:#a59f95;--line:#34312c}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 Georgia,serif;max-width:1100px;margin:0 auto;padding:16px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}
section{{border-top:1px solid var(--line);padding:12px 0}}h2{{font-size:18px}}h3{{font:600 13px system-ui;color:var(--muted)}}
p{{font-size:15px}}button{{font:14px system-ui;padding:8px 14px}}</style></head><body>
<h1>Abstract pairs (blind)</h1><p>One abstract in each pair is the real abstract of a funded NIH grant; the other was
written or rewritten by a model. Answers save in this browser. Press <b>Copy answers</b> at the end.</p>
{"".join(blocks)}
<p><button id="copy">Copy answers</button> <span id="msg"></span></p>
<script>
const K='ral-packet';let s={{}};try{{s=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(s[r.name]===r.value)r.checked=true;
r.addEventListener('change',()=>{{s[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(s))}}catch(e){{}}}})}});
document.getElementById('copy').onclick=()=>{{navigator.clipboard.writeText(JSON.stringify(s));
document.getElementById('msg').textContent=Object.keys(s).length+' answers copied'}};
</script></body></html>"""
    (OUT / "index.html").write_text(page)
    print(f"{len(items)} pairs -> file://{OUT / 'index.html'}")


def score_answers(path: str) -> None:
    key = {k["qid"]: k for k in json.loads((OUT / "key.json").read_text())}
    ans = json.loads(Path(path).read_text())
    by: dict = {}
    for q, k in key.items():
        ai = ans.get(f"{q}-ai")
        if ai:
            by.setdefault(k["source"], []).append(ai == k["real_side"])
    res = {s: {"n": len(v), "fooling": sum(v) / len(v)} for s, v in by.items()}
    (C.RESULTS / "human_packet.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "score":
        score_answers(sys.argv[2])
    else:
        build()
