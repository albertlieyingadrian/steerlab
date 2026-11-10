"""The note's question explorer: six held-out test questions, gpt-5.4-mini raw vs with retrieved examples vs in the
harness, from the saved transcripts.

    python experiments/fin-llm/make_explorer.py --select   # transcripts -> results/explorer.json (needs the runs)
    python experiments/fin-llm/make_explorer.py            # results/explorer.json -> the HTML in report/index.html

The six were picked by hand from the 46 questions both the examples and the harness fixed and the 29 the harness
broke, to show each kind: a sign the examples taught, three methods they taught, and two breaks. The HTML is written
between `<!-- explorer:start -->` and `<!-- explorer:end -->`, every case in the page text (a script only switches).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fin

OUT = fin.RESULTS / "explorer.json"
PAGE = HERE / "report" / "index.html"
MODEL = "gpt-5.4-mini"
CASES = [  # (id, short label, what it shows)
    (
        "RSG/2018/page_94.pdf-1",
        "A sign",
        "fixed: the examples show FinQA writes a decline as a negative percentage",
    ),
    (
        "MRO/2007/page_136.pdf-2",
        "Cost per year",
        "fixed: an example spreads a cost evenly over the stated years",
    ),
    (
        "GPN/2014/page_92.pdf-1",
        "Shares × price",
        "fixed: the examples value securities as shares times price",
    ),
    (
        "KHC/2018/page_132.pdf-3",
        "Net increase",
        "fixed: the examples count the change over the whole period",
    ),
    (
        "BLL/2010/page_37.pdf-2",
        "A sign, broken",
        "broken: here FinQA's gold for a decrease is positive",
    ),
    (
        "GS/2017/page_179.pdf-4",
        "Wrong column, broken",
        "broken: with the examples, the model added dollar values instead of option counts",
    ),
]


def transcripts(mode: str) -> dict[str, list[dict]]:
    path = fin.RESULTS / "transcripts" / f"test__{mode}__{MODEL}.jsonl"
    return {r["id"]: r["transcript"] for r in map(json.loads, path.read_text().splitlines())}


def run_rows(mode: str) -> dict[str, dict]:
    path = fin.RESULTS / "runs" / f"test__{mode}__{MODEL}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def select() -> None:
    items = {i.id: i for i in fin.split("test")}
    raw, ex, har = transcripts("raw"), transcripts("raw+ex"), transcripts("harness")
    runs = {m: run_rows(m) for m in ("raw", "raw+ex", "harness")}
    cases = []
    for qid, label, note in CASES:
        prompt = ex[qid][0]["content"]
        examples = re.findall(r"Q: (.*?)\nProgram: (.*?)\nResult: (.*?)\n", prompt + "\n")
        steps = [{"role": m["role"], "text": m["content"]} for m in har[qid][2:]]
        cases.append({"id": qid, "label": label, "note": note, "question": items[qid].question,
                      "gold": items[qid].gold, "gold_text": items[qid].gold_text,
                      "raw": {"pred": runs["raw"][qid]["pred"], "text": raw[qid][-1]["content"]},
                      "examples": [{"q": q, "program": p, "result": r} for q, p, r in examples],
                      "with_examples": {"pred": runs["raw+ex"][qid]["pred"], "text": ex[qid][-1]["content"]},
                      "harness": {"pred": runs["harness"][qid]["pred"], "steps": steps}})  # fmt: skip
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"wrote {OUT} ({len(cases)} cases)")


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def verdict(pred: str | None, gold) -> str:
    ok = fin.grade(pred, gold)
    return f'<span class="ex-v {"ok" if ok else "no"}">{"right" if ok else "wrong"}</span>'


def case_html(k: int, c: dict) -> list[str]:
    hidden = "" if k == 0 else " hidden"
    ex_lines = "\n\n".join(f"Q: {e['q']}\nProgram: {e['program']}\nResult: {e['result']}" for e in c["examples"])
    turns = "\n\n".join(f"[{'model' if s['role'] == 'assistant' else 'harness'}] {s['text']}"
                        for s in c["harness"]["steps"])  # fmt: skip
    raw, withex, har = c["raw"], c["with_examples"], c["harness"]
    return [
        f'  <div class="ex-case" data-case="{k}"{hidden}>',
        (
            f'    <p class="ex-q"><strong>Question</strong> ({esc(c["id"])}): {esc(c["question"])} '
            f'<span class="ex-gold">Gold: {esc(c["gold"])} (FinQA’s answer string: {esc(c["gold_text"] or "none")})'
            "</span></p>"
        ),
        f'    <p class="ex-note">{esc(c["note"])}.</p>',
        '    <div class="ex-cols">',
        (
            f'      <div><p class="ex-h">Raw, one call: <code>{esc(raw["pred"])}</code> '
            f"{verdict(raw['pred'], c['gold'])}</p><pre>{esc(raw['text'][-700:])}</pre></div>"
        ),
        (
            f'      <div><p class="ex-h">The 4 retrieved train examples</p><pre>{esc(ex_lines)}</pre><p class="ex-h">'
            f"With them: <code>{esc(withex['pred'])}</code> {verdict(withex['pred'], c['gold'])}</p></div>"
        ),
        (
            f'      <div><p class="ex-h">In the harness: <code>{esc(har["pred"])}</code> '
            f"{verdict(har['pred'], c['gold'])}</p><pre>{esc(turns[:2600])}</pre></div>"
        ),
        "    </div>",
        "  </div>",
    ]


def render() -> None:
    cases = json.loads(OUT.read_text())
    out = [
        '<div class="sim explorer" id="explorer">',
        '  <div class="gallery-pick" role="group" aria-label="Question">',
    ]
    for k, c in enumerate(cases):
        pressed = "true" if k == 0 else "false"
        out.append(f'    <button type="button" data-case="{k}" aria-pressed="{pressed}">{esc(c["label"])}</button>')
    out.append("  </div>")
    for k, c in enumerate(cases):
        out += case_html(k, c)
    out.append(
        '  <p class="caption"><span class="label">Explorer.</span> Six held-out FinQA test questions answered by '
        "gpt-5.4-mini three ways: raw (the end of its reply), with the four retrieved training examples, and in the "
        "full harness (every turn: plan and code from the model, Python output and the check from the harness; long "
        "turns cut). Picked by hand from the 46 questions both fixed and the 29 the harness broke; built by "
        '<a href="../make_explorer.py">make_explorer.py</a> from '
        '<a href="../results/explorer.json">explorer.json</a>.</p>'
    )
    out.append("</div>")
    page = PAGE.read_text()
    start, end = "<!-- explorer:start -->", "<!-- explorer:end -->"
    a, b = page.index(start) + len(start), page.index(end)
    PAGE.write_text(page[:a] + "\n" + "\n".join(out) + "\n" + page[b:])
    print(f"rendered {len(cases)} cases into {PAGE.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    if p.parse_args().select:
        select()
    render()
