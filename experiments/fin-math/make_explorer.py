"""The note's answer explorer: three FinanceMath test questions, base Qwen3-8B vs after SFT on the teacher's traces,
and one synthetic training problem with the two kinds of trace the SFT runs learned from.

    python experiments/fin-math/make_explorer.py --select   # runs + data -> results/explorer.json (needs data/raw)
    python experiments/fin-math/make_explorer.py            # results/explorer.json -> the HTML in report/index.html

The test cases were picked by hand from the 135 short questions teacher-trace SFT broke and the 44 it fixed: a
confident method error, an arithmetic slip with no check, and a fix. The synthetic problem is the first SFT problem
(by id) with no table whose teacher trace and shortest correct own sample are both short. The HTML is written between
`<!-- explorer:start -->` and `<!-- explorer:end -->`; every case is in the page text (a script only switches).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fm  # sets up the fin-llm import path first

# isort: split

OUT = fm.RESULTS / "explorer.json"
PAGE = HERE / "report" / "index.html"
CASES = [  # (test id, button label, what it shows)
    (
        "test-63",
        "Bull call spread",
        "broken: after SFT the model states a wrong rule in one line and does not check it",
    ),
    (
        "test-60",
        "Bond price",
        "broken: the right method, but 1,050 / 1.08³ worked out in its head as 811.62 (it is 833.52)",
    ),
    (
        "test-41",
        "ROE after leverage",
        "fixed: the short answer gets this one right where the base model halved the ratio",
    ),
]


def select() -> None:
    items = {i.id: i for i in fm.load("test")}
    base, sft = (
        {r["id"]: r for r in map(json.loads, (fm.RESULTS / "runs" / f"test__{t}.jsonl").read_text().splitlines())}
        for t in ("qwen", "sft-e1")
    )
    cases = []
    for qid, label, note in CASES:
        it = items[qid]
        cases.append({"kind": "test", "id": qid, "label": label, "note": note, "topic": it.topic,
                      "question": it.question, "tables": it.tables, "gold": it.gold,
                      "base": {"pred": base[qid]["pred"], "correct": base[qid]["correct"], "text": base[qid]["text"]},
                      "sft": {"pred": sft[qid]["pred"], "correct": sft[qid]["correct"], "text": sft[qid]["text"]}})  # fmt: skip
    own = {r["id"]: r["samples"] for r in map(json.loads, (HERE / "data" / "qwen_bon.jsonl").read_text().splitlines())}
    import fm_post

    for r in sorted(fm_post.verified(), key=lambda r: r["id"]):
        ok = sorted((s for s in own.get(r["id"], []) if s["correct"]), key=lambda s: s["tokens"])
        if fm_post.is_rl(r["id"]) or r["tables"] or not ok or len(r["teacher_text"]) > 700 or len(ok[0]["text"]) > 1500:
            continue
        cases.append({"kind": "synth", "id": r["id"], "label": "A training problem", "topic": r["topic"],
                      "note": "what the two SFT runs learned from: the teacher's visible answer, and Qwen3-8B's own "
                              "shortest right answer", "question": r["question"], "tables": "", "gold": r["gold"],
                      "teacher": r["teacher_text"], "own": ok[0]["text"]})  # fmt: skip
        break
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"wrote {len(cases)} cases to {OUT.name}")


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def tag(ok: bool) -> str:
    return '<span class="ex-v ok">right</span>' if ok else '<span class="ex-v no">wrong</span>'


def case_html(k: int, c: dict) -> list[str]:
    hidden = "" if k == 0 else " hidden"
    kind = "Question" if c["kind"] == "test" else "Synthetic problem"
    head = f"<strong>{kind}</strong> ({esc(c['id'])}, {esc(c['topic'])}): {esc(c['question'])}"
    gold = f'<span class="ex-gold">Gold: {c["gold"]:g}</span>'
    out = [f'  <div class="ex-case" data-case="{k}"{hidden}>', f'    <p class="ex-q">{head} {gold}</p>']
    if c["tables"]:
        out.append(f'    <pre class="ex-table">{esc(c["tables"])}</pre>')
    out += [f'    <p class="ex-note">{esc(c["note"])}.</p>', '    <div class="ex-cols">']
    if c["kind"] == "test":
        for key, title in (("base", "Base Qwen3-8B"), ("sft", "After SFT on teacher traces")):
            s = c[key]
            out.append(f'      <div><p class="ex-h">{title}: <code>{esc(s["pred"])}</code> {tag(s["correct"])}</p>'
                       f'<pre>{esc(s["text"].strip())}</pre></div>')  # fmt: skip
    else:
        out.append('      <div><p class="ex-h">Teacher (gpt-oss-120b), final answer (what SFT trained on)</p>'
                   f'<pre>{esc(c["teacher"].strip())}</pre></div>')  # fmt: skip
        out.append('      <div><p class="ex-h">Qwen3-8B’s own shortest right answer</p>'
                   f'<pre>{esc(c["own"].strip())}</pre></div>')  # fmt: skip
    out += ["    </div>", "  </div>"]
    return out


def render() -> None:
    cases = json.loads(OUT.read_text())
    out = ['<div class="sim explorer" id="explorer">', '  <div class="gallery-pick" role="group" aria-label="Case">']
    for k, c in enumerate(cases):
        pressed = "true" if k == 0 else "false"
        out.append(f'    <button type="button" data-case="{k}" aria-pressed="{pressed}">{esc(c["label"])}</button>')
    out.append("  </div>")
    for k, c in enumerate(cases):
        out += case_html(k, c)
    out.append('  <p class="caption">Replies as generated (greedy, the benchmark prompt), graded by the official '
               "FinanceMath rule. Source: <code>results/explorer.json</code>, written by "
               '<a href="../make_explorer.py">make_explorer.py</a>.</p>')  # fmt: skip
    out.append("</div>")
    page = PAGE.read_text()
    start, end = "<!-- explorer:start -->", "<!-- explorer:end -->"
    a, b = page.index(start) + len(start), page.index(end)
    PAGE.write_text(page[:a] + "\n" + "\n".join(out) + "\n" + page[b:])
    print(f"rendered {len(cases)} cases into {PAGE.name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    if p.parse_args().select:
        select()
    else:
        render()


if __name__ == "__main__":
    main()
