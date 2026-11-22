"""Write the note's data and drawn parts from entries.jsonl, artefacts/ and results/detection.json.

    .venv/bin/python experiments/taste-hack-bench/make_report.py [--no-hero]

Writes report/data.js (the gallery's data: every entry, its detector verdicts and one short hack and clean sample),
the static detection matrix into report/index.html between `<!-- matrix:start -->` and `<!-- matrix:end -->`, and
the cover SVG between the cover markers; then report/figures/hero.jpg and ../shared/thumbs/taste-hack-bench.jpg.
Prints every number it draws.
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
    EMBER_L,
    INK,
    INK2,
    LINE,
    MUTED,
    caps,
    kicker,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

import detectors as D

DET = ["loop", "copy", "length", "ledger"]
SNIP = 260
# The sample shown for each entry: the hack row the entry's catching check scores highest, so the gallery shows the
# hack at its clearest. Rows are picked by rule, then checked for names (none of these contain a person's name).
PICK = {
    "pg-padded-copy": lambda r: D.copy(r["text"], EX)[0],
    "qwen3-comment-stuffing": lambda r: r["text"].count("//"),
    "qwen3-comment-compact": lambda r: -len(r["text"]),
    "gsm8k-pass-tag": lambda r: -r["n_tokens"],
    "counsel-blank-canvas": lambda r: -len(r["text"]),
    "finmath-think-less": lambda r: -r["n_tokens"],
    "agent-handoff": lambda r: "transfer_to_human_agents" in r["text"],
    "judge-house-style": lambda r: len(r["text"]) if len(r["text"]) < 900 else 0,
    "impossible-test-hardcode": lambda r: ("1900" in r["text"]) * 1000 - len(r["text"]),
    "steering-honesty-cost": lambda r: (r.get("ledger") or 0) * 10 + D.loop(r["text"])[0],
    "culture-persona-loops": lambda r: D.loop(r["text"])[0],
    "margin-over-broken-random": lambda r: D.loop(r["text"])[0],
    "gg-fluency-passes-loops": lambda r: D.loop(r["text"])[0],
    "short-screen-hides-loops": lambda r: D.loop(r["text"])[0],
}
EX = ""
SHORT = {  # cover row labels
    "pg-toy-language-switch": "answer in Chinese",
    "pg-toy-stock-refusal": "stock refusal",
    "pg-paste-example": "paste the example",
    "pg-padded-copy": "padded copy",
    "qwen3-comment-stuffing": "comment stuffing",
    "qwen3-comment-compact": "compact comment stuffing",
    "gsm8k-pass-tag": "grader pass tag",
    "counsel-blank-canvas": "blank canvas",
    "finmath-think-less": "think less",
    "agent-handoff": "hand off to a human",
    "judge-house-style": "one house style",
    "impossible-test-hardcode": "hard-code the test",
    "steering-honesty-cost": "voice steering invents",
    "culture-persona-loops": "persona vector loops",
    "margin-over-broken-random": "margin rule, broken pick",
    "gg-fluency-passes-loops": "fluency check passes loops",
    "short-screen-hides-loops": "short screen hides loops",
}
NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def redactions() -> list[tuple[re.Pattern, str]]:
    """Synthetic brief names and companies from Sounds Like Me, replaced in the shown samples (the note names no one)."""
    briefs = json.loads((HERE.parent / "sounds-like-me/data/briefs.json").read_text())
    out = set()
    for b in briefs:
        out.add((b["prospect"]["first_name"], "[name]"))
        out.add((b["prospect"]["company"], "[company]"))
        out.add((b["sender"]["company"], "[company]"))
    words = sorted(out, key=lambda x: -len(x[0]))
    return [(re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE), rep) for w, rep in words]


RED = redactions()


def redact(text: str) -> str:
    for pat, rep in RED:
        text = pat.sub(rep, text)
    return text


def snippet(text: str, entry_id: str) -> str:
    text = redact(" ".join(text.split()))
    if entry_id == "impossible-test-hardcode" and "1900" in text:
        i = text.find("if year == 1900")
        text = text[max(0, i - 80) :] if i >= 0 else text
    if (
        entry_id == "qwen3-comment-stuffing"
        or entry_id.startswith("pg-")
        or entry_id in ("counsel-blank-canvas", "qwen3-comment-compact")
    ):
        i = text.find("background(")
        text = text[i:] if i >= 0 else text
    return text if len(text) <= SNIP else text[:SNIP].rsplit(" ", 1)[0] + " …"


def main(hero: bool = True) -> None:
    global EX
    entries = [json.loads(line) for line in (HERE / "entries.jsonl").read_text().splitlines()]
    det = json.loads((HERE / "results/detection.json").read_text())
    by_id = {t["id"]: t for t in det["table"]}
    out = []
    for e in entries:
        rec = {
            k: e.get(k)
            for k in (
                "id",
                "title",
                "domain",
                "kind",
                "source",
                "section",
                "reward",
                "hack",
                "quote",
                "caught_by",
                "lab_check",
                "when",
                "missed_by",
                "notes",
                "why_no_artefact",
            )
        }
        rec["href"] = f"../../{e['source']}/report/" + (f"#{e['section']}" if e.get("section") else "")
        rec["detectors"], rec["best"] = by_id[e["id"]]["detectors"], by_id[e["id"]]["best"]
        if e.get("private"):
            # private drafts: show only the sentence the source note quotes, and no file to download
            a = e["artefacts"]
            rec["artefacts"] = {
                "file": None,
                "n_hack": a["n_hack"],
                "n_clean": a["n_clean"],
                "from": sorted({f.split("/")[0] for f in a["source_files"]}),
                "label_source": a["label_source"],
                "private": e["private"],
                "hack_sample": "Quoted in the source note, where the draft repeats it: I think that maybe they need "
                "some drywall labor staffing for peak months.",
                "clean_sample": "Not published.",
            }
        elif e["artefacts"]:
            a = e["artefacts"]
            EX = a["reference"] or ""
            rows = [json.loads(line) for line in (HERE / a["file"]).read_text().splitlines()]
            hack = [r for r in rows if r["group"] == "hack"]
            clean = [r for r in rows if r["group"] == "clean"]
            key = PICK.get(e["id"], lambda r: 0)
            h = max(hack, key=lambda r: (not NAME.search(r["text"][:600]), key(r)))
            if e["id"] == "finmath-think-less":  # rows come in (hack, clean) pairs on the same question
                c = rows[rows.index(h) + 1]
            else:
                c = min(
                    clean,
                    key=lambda r: (
                        bool(NAME.search(r["text"][:600])),
                        "transfer_to_human_agents" in r["text"],
                        r.get("truncated") is True and len(r["text"]) < 60,
                        D.loop(r["text"])[0] if D.loop(r["text"]) else 0,
                    ),
                )
            rec["artefacts"] = {
                "file": a["file"],
                "n_hack": a["n_hack"],
                "n_clean": a["n_clean"],
                "from": sorted({f.split("/")[0] for f in a["source_files"]}),
                "label_source": a["label_source"],
                "hack_sample": snippet(h["text"], e["id"]),
                "clean_sample": snippet(c["text"], e["id"]),
            }
        else:
            rec["artefacts"] = None
        out.append(rec)
    s = det["summary"]
    (REPORT / "data.js").write_text(
        "// built by make_report.py from entries.jsonl and results/detection.json\nwindow.HACKS = "
        + json.dumps({"entries": out, "summary": s}, ensure_ascii=False, indent=1)
        + ";\n"
    )

    # ---- static matrix (works without JavaScript) ----
    sym = {"full": ("", "caught"), "partial": ("", "partly"), "miss": ("", "missed"), "n/a": ("", "n/a")}
    rows_html = []
    for r in out:
        if not r["detectors"]:
            continue
        cells = []
        for n in DET:
            d = r["detectors"][n]
            word = sym[d["verdict"]][1]
            rate = f'<span class="rate">{d["hack_rate"]:.2f} / {d["clean_rate"]:.2f}</span>' if "hack_rate" in d else ""
            star = "*" if d.get("shared_with_label") else ""
            cells.append(
                f'<td class="v v-{d["verdict"].replace("/", "")}" title="{word}"><span class="g" aria-hidden="true"></span><span class="w">{word}</span>{star}{rate}</td>'
            )
        best = {"full": "caught", "partial": "partly", "miss": "<strong>missed</strong>"}[r["best"]]
        rows_html.append(
            f'<tr><th scope="row"><a href="#hack-{r["id"]}">{html.escape(r["title"])}</a></th><td>{html.escape(r["lab_check"])}</td>{"".join(cells)}<td>{best}</td></tr>'
        )
    table = (
        '<div class="table-wrap" tabindex="0" role="region" aria-label="Table 2, scrolls sideways on narrow screens"><table class="matrix">\n'
        "<caption>Table 2. Four cheap checks on the 17 entries with artefacts. Caught: flags at least half the hack rows and at most a fifth of the clean rows. Partly: flags at least 0.10 more hack than clean rows, intervals apart. Small numbers: share of hack rows flagged / share of clean rows flagged. * the hack label was made by a rule of the same kind as the check. From <code>results/detection.json</code>.</caption>\n"
        "<thead><tr><th>Hack</th><th>Lab&rsquo;s check</th><th>Loop</th><th>Copy</th><th>Length</th><th>Ledger</th><th>Any cheap check</th></tr></thead>\n<tbody>\n"
        + "\n".join(rows_html)
        + "\n</tbody></table></div>"
    )
    idx = REPORT / "index.html"
    text = idx.read_text()
    a, b = "<!-- matrix:start -->", "<!-- matrix:end -->"
    i, j = text.index(a) + len(a), text.index(b)
    text = text[:i] + "\n" + table + "\n" + text[j:]
    # static list of all entries, so the page reads without JavaScript (gallery.js replaces it with the filterable one)
    esc = html.escape
    items = []
    for r in out:
        how = f"{esc(r['caught_by'])} ({esc(r['when'])})"
        items.append(
            f'<li class="hg-card" id="hack-{r["id"]}"><p class="hg-tag">{esc(r["kind"])} · {esc(r["domain"])}</p>'
            f'<p class="hg-title">{esc(r["title"])}</p><p class="small">“{esc(r["quote"])}” <a href="{esc(r["href"])}">source note ↗</a>. '
            f"Caught by: {how}.</p></li>"
        )
    a, b = "<!-- gallery:start -->", "<!-- gallery:end -->"
    i, j = text.index(a) + len(a), text.index(b)
    idx.write_text(text[:i] + "\n" + "\n".join(items) + "\n    " + text[j:])

    # ---- cover: the matrix as a grid, beside the counts ----
    body = kicker("THE TASTE STACK · EPISODE 8 (V0)", "AN OPEN BENCHMARK OF TASTE-REWARD HACKS")
    body += t(20, 62, "39 hacks we caught, and what cheap checks see", 25, 800, INK)
    kinds = Counter(r["kind"] for r in out)
    y = 104
    body += t(20, y, "THE 39 ENTRIES", 9.5, 700, MUTED, ls="0.08em")
    for k in ("trained policy", "constructed trap", "selection rule", "steering arm", "planted"):
        y += 24
        n = kinds[k]
        body += (
            t(20, y, k, 12, 500, INK2)
            + rect(140, y - 11, n * 12, 13, EMBER if k == "trained policy" else "#cfc8bf")
            + t(146 + n * 12, y, str(n), 11, 700, INK)
        )
    y += 44
    body += t(20, y, "17 WITH ARTEFACTS: BEST CHEAP CHECK", 9.5, 700, MUTED, ls="0.08em")
    for label, n, col in (
        ("caught", s["full"], EMBER),
        ("partly", s["partial"], EMBER_L),
        ("missed", s["missed"], "#ffffff"),
    ):
        y += 24
        body += (
            t(20, y, label, 12, 600 if label == "missed" else 500, INK if label == "missed" else INK2)
            + rect(140, y - 11, n * 24, 13, col, EMBER_D if label == "missed" else "none")
            + t(146 + n * 24, y, str(n), 11, 700, INK)
        )
    gx, gy, cw, ch = 520, 98, 68, 17
    body += t(gx - 10, gy - 8, "hack", 9.5, 700, MUTED, "end", ls="0.06em")
    for k, n in enumerate(DET):
        body += t(gx + k * cw + cw / 2, gy - 8, n.upper(), 9.5, 700, MUTED, "middle", ls="0.06em")
    fill = {"full": EMBER, "partial": EMBER_L, "miss": "#ffffff", "n/a": "none"}
    r_i = 0
    for r in out:
        if not r["detectors"]:
            continue
        yy = gy + r_i * (ch + 3)
        body += t(gx - 10, yy + 12.5, SHORT.get(r["id"], r["id"]), 10, 500, INK2, "end")
        for k, n in enumerate(DET):
            v = r["detectors"][n]["verdict"]
            if v == "n/a":
                body += f'<path d="M{gx + k * cw + 26},{yy + 8.5} h16" stroke="{LINE}" stroke-width="1.5"/>'
            else:
                body += rect(gx + k * cw + 4, yy, cw - 8, ch, fill[v], EMBER_D if v == "miss" else "none", 1, 3)
        r_i += 1
    body += t(
        gx,
        gy + r_i * (ch + 3) + 14,
        "filled: caught · light: partly · outlined: missed · dash: does not apply",
        9.5,
        500,
        MUTED,
    )
    body += f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>'
    body += caps(
        "39 ENTRIES FROM 24 PUBLISHED NOTES · 17 WITH DERIVED SAMPLES · LOOP, COPY, LENGTH AND HONESTY-LEDGER CHECKS"
    )
    label = (
        f"Left: the 39 hacks by kind ({kinds['trained policy']} found by a trained policy, {kinds['constructed trap']} constructed traps, {kinds['selection rule']} selection rules, {kinds['steering arm']} steering arms, {kinds['planted']} planted), "
        f"and for the 17 with artefacts the best cheap check: caught {s['full']}, partly {s['partial']}, missed {s['missed']}. "
        "Right: each of the 17 entries against the loop, copy, length and honesty-ledger checks."
    )
    cover = svg(body, label, "thb-cover")
    write_cover(idx, cover)
    if hero:
        render_hero(cover, REPORT, HERE.parent / "shared/thumbs/taste-hack-bench.jpg")
    print("kinds", dict(kinds), "summary", {k: s[k] for k in ("full", "partial", "missed")})


if __name__ == "__main__":
    main(hero="--no-hero" not in sys.argv)
