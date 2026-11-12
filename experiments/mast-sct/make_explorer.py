"""Write the item explorer into report/index.html: four real items, the expert panel, and every run's rating.

    python experiments/mast-sct/make_explorer.py [--mast PATH]

Reads the item text and expert panels from a MAST clone (benchmarks/sct/dataset/, MIT) and each run's parsed rating
from results/scores/<run>_details.jsonl. Replaces the block between <!-- explorer:start --> and <!-- explorer:end -->.
Case text is shown with its encoding artefact repaired for reading (the benchmark's prompts carry the raw bytes).
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report" / "index.html"
RUNS = [
    ("gpt-5.5", "GPT-5.5 (MAST's run)"),
    ("deepseek-v4-flash-0731-think", "DeepSeek-V4-Flash, thinking"),
    ("deepseek-v4-flash-0731-nothink", "DeepSeek-V4-Flash, no thinking"),
    ("deepseek-v4-flash-0731-nothink-rep2", "the same request, repeated"),
    ("llama-3.3-70b-instruct", "Llama-3.3-70B-Instruct"),
    ("qwen3-32b-nothink", "Qwen3-32B, no thinking"),
]
CASES = [
    ("1", "Experts split", "The panel spreads over four answers, so a near miss still earns partial credit."),
    ("51", "Same request, opposite answer",
     "Asked twice, identically, DeepSeek-V4-Flash without thinking said +2, then −2."),
    ("2", "Thinking changes the answer",
     ("Without thinking, DeepSeek-V4-Flash says a clear chest X-ray lowers the case for a CT scan (no credit); "
      "with thinking it says +1, as GPT-5.5 does, an answer part of the panel gave.")),
    ("6", "Every run misses", "The whole panel answered −2; all six runs answered +2, the opposite end of the scale."),
]  # fmt: skip
LABELS = ["−2", "−1", "0", "+1", "+2"]
PAT = re.compile("[Â-ô][\u0080-¿]+")


def readable(s: str) -> str:
    return PAT.sub(lambda m: m.group().encode("latin-1").decode("utf-8"), s)


def block(mast: Path) -> str:
    data = mast / "benchmarks" / "sct" / "dataset"
    items = {d["id"]: d for d in map(json.loads, (data / "items.jsonl").open())}
    panels = {d["id"]: d["expert_distribution"] for d in map(json.loads, (data / "rubrics.jsonl").open())}
    scores = HERE / "results" / "scores"
    det = {r: {d["id"]: d for d in map(json.loads, (scores / f"{r}_details.jsonl").open())} for r, _ in RUNS}
    e = html.escape
    out = ['<div class="sim explorer" id="explorer">', '  <div class="gallery-pick" role="group" aria-label="Item">']
    for n, (_, title, _) in enumerate(CASES):
        pressed = "true" if n == 0 else "false"
        out.append(f'    <button type="button" data-case="{n}" aria-pressed="{pressed}">{e(title)}</button>')
    out.append("  </div>")
    for n, (iid, _, note) in enumerate(CASES):
        it, panel = items[iid], panels[iid]
        credit = [v / max(panel) for v in panel]
        hidden = "" if n == 0 else " hidden"
        out.append(f'  <div class="ex-case" data-case="{n}"{hidden}>')
        out.append(
            f'    <p class="ex-q"><strong>Item {iid}</strong> ({e(it["source"])}). {e(readable(it["scenario"]))}</p>'
        )
        out.append(
            f'    <p class="ex-q"><strong>{e(readable(it["hypothesis"]).strip().rstrip("."))}</strong>. '
            f"<strong>{e(readable(it['new_information']).strip().rstrip('.'))}</strong>.</p>"
        )
        out.append(f'    <p class="ex-note">{e(note)}</p>')
        bars = "".join(
            f'<div class="ex-bar"><span class="ex-bl">{lab}</span><span class="ex-track"><span class="ex-fill'
            f'{" top" if c == 1 else ""}" style="width:{c * 100:.0f}%"></span></span>'
            f'<span class="ex-bv">{c:.2f}</span></div>'
            for lab, c in zip(LABELS, credit, strict=True)
        )
        out.append(
            '    <div class="ex-cols"><div><p class="ex-h">Credit for each rating (the panel, scaled so its most '
            f"common answer is 1)</p>{bars}</div>"
        )
        rows = "".join(
            f'<tr><td>{e(label)}</td><td class="num">{LABELS[det[r][iid]["rating"] + 2]}</td>'
            f'<td class="num">{credit[det[r][iid]["rating"] + 2]:.2f}</td></tr>'
            for r, label in RUNS
        )
        out.append(
            '    <div><p class="ex-h">Each run’s rating and the credit it earned</p><table class="ex-t"><thead><tr>'
            f'<th>Run</th><th class="num">Rating</th><th class="num">Credit</th></tr></thead><tbody>{rows}'
            "</tbody></table></div></div>"
        )
        out.append("  </div>")
        ratings = [det[r][iid]["rating"] for r, _ in RUNS]
        print(f"explorer: item {iid} credit {[round(c, 2) for c in credit]} ratings {ratings}")
    out.append(
        '  <p class="caption">Four of the 174 open items, chosen to show partial credit, a repeat that flipped, the '
        "effect of thinking, and an item every run missed. Ratings are as parsed by MAST’s score.py; written by "
        '<a href="../make_explorer.py">make_explorer.py</a>.</p>'
    )
    out.append("</div>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mast", type=Path, default=HERE / "mast", help="a clone of ARISENetwork/mast")
    args = parser.parse_args()
    page = REPORT.read_text()
    start, end = "<!-- explorer:start -->", "<!-- explorer:end -->"
    a, b = page.index(start) + len(start), page.index(end)
    REPORT.write_text(page[:a] + "\n" + block(args.mast) + "\n" + page[b:])


if __name__ == "__main__":
    main()
