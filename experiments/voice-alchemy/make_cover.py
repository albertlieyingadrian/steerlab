"""Draw the cover from the data: two measured voice profiles, the sum they predict, what pushing both did, and the
first words of a real reply.

    .venv/bin/python experiments/voice-alchemy/make_cover.py

Left: the measured profiles (judge points against the unsteered model, 16 held-out prompts) of Qwen3-8B's playful and
casual neuron communities at their chosen strengths. Right: the measured pair, with ticks at the sum of the singles,
and its reply to "How do I get better at public speaking?" (results/analysis_heldout.json,
results/scored_heldout.jsonl). Writes the SVG into report/index.html between the cover markers,
report/figures/hero.jpg and ../shared/thumbs/voice-alchemy.jpg.
"""

from __future__ import annotations

import json
import sys

import va

REPORT = va.HERE / "report"
THUMB = va.HERE.parent / "shared" / "thumbs" / "voice-alchemy.jpg"
sys.path.insert(0, str(va.HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    caps,
    kicker,
    lines,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

A, B, PI, TYPE = "playful", "casual", 4, "community"
"""Run with VA_MODEL=qwen: the cover shows Qwen3-8B, the main model."""
TRAITS = list(va.TRAITS)


def panel(x, y, w, title, d, pred=None, hl=()):
    out = rect(x, y, w, 236, "#fff", LINE, 1.5, 10) + t(x + 14, y + 24, title, 12, 700, MUTED, ls="0.05em")
    mid, half = x + w * 0.58, w * 0.36
    for i, tr in enumerate(TRAITS):
        yy = y + 46 + i * 23
        out += t(x + 14, yy + 4, va.label(tr), 11.5, 700 if tr in hl else 500, INK if tr in hl else INK2)
        out += f'<path d="M{mid:.1f},{yy - 8} V{yy + 8}" stroke="{LINE}" stroke-width="1"/>'
        v = d[i]
        wv = min(abs(v) / 8, 1) * half
        x0 = mid if v >= 0 else mid - wv
        out += rect(round(x0, 1), yy - 6, round(wv, 1), 12, EMBER if v >= 0 else "#f2c9ab", rx=2)
        if pred is not None:
            px = mid + max(-1, min(1, pred[i] / 8)) * half
            out += f'<path d="M{px:.1f},{yy - 9} V{yy + 9}" stroke="{INK}" stroke-width="2.2"/>'
    return out


def main() -> None:
    a = json.loads((va.RESULTS / "analysis_heldout.json").read_text())
    pr = next(p for p in a["pairs"] if p["type"] == TYPE and set(p["traits"]) == {A, B})
    flip = pr["traits"][0] != A
    da, db = (pr["d_b"], pr["d_a"]) if flip else (pr["d_a"], pr["d_b"])
    pred = [x + y for x, y in zip(da, db, strict=True)]
    rows = (json.loads(x) for x in (va.RESULTS / "scored_heldout.jsonl").read_text().splitlines())
    reply = next(r for r in rows if r["kind"] == "pair" and r["uids"] == pr["uids"] and r["pi"] == PI)
    snippet = " ".join(reply["text"].replace("**", "").split()[:22]) + " …"
    print(f"pair {pr['traits']} E {pr['E']:.2f} label {pr['label']} coherence {pr['coherence']:.1f}; reply: {snippet}")
    body = kicker("STEERING · VOICE ALCHEMY", "QWEN3-8B · TRANSFORMERLENS")
    body += t(20, 62, "Push two voices at once. Do they add up?", 27, 800, INK)
    body += panel(20, 92, 196, "PLAYFUL COMMUNITY", da, hl=(A,))
    body += t(232, 220, "+", 30, 800, MUTED, "middle")
    body += panel(248, 92, 196, "CASUAL COMMUNITY", db, hl=(B,))
    body += t(468, 214, "→", 30, 800, MUTED, "middle")
    body += panel(490, 92, 196, "BOTH PUSHED", pr["d_ab"], pred=pred, hl=(A, B))
    chip = f"{pr['label'].upper()} · E {pr['E']:.2f}"
    body += rect(700, 92, 160, 30, EMBER, rx=15) + t(780, 112, chip, 12, 800, "#fff", "middle", ls="0.04em")
    body += rect(700, 134, 160, 194, "#fff", EMBER, 2, 10) + t(712, 156, "REAL REPLY", 10.5, 700, EMBER_D, ls="0.06em")
    body += lines(712, 178, snippet, 21, 12.5, INK, lh=17)
    body += t(
        20,
        358,
        "Bars: change in each voice (judge points vs the unsteered model). Ticks: the sum of the two singles.",
        12.5,
        500,
        INK2,
    )
    body += t(
        20,
        384,
        "In Qwen3-8B, plain directions steered 7 of 8 voices and neuron communities 3; this community pair",
        15,
        400,
        INK,
    )
    body += t(20, 405, "added up, but most direction pairs pushed at full strength broke.", 15, 400, INK)
    body += f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>'
    body += caps(
        "NEURON COMMUNITIES IN LAYERS 15-19 · 16 HELD-OUT PROMPTS PER BAR · REPLY TO "
        "“HOW DO I GET BETTER AT PUBLIC SPEAKING?”, UNEDITED"
    )
    label = (
        f"Two measured voice profiles of Qwen3-8B, its playful and casual neuron communities, and the profile when "
        f"both are pushed at once ({pr['label']}, additivity error {pr['E']:.2f}), with the first words of a real "
        f"reply: '{snippet}'"
    )
    s = svg(body, label, "va-cover")
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("wrote cover, hero.jpg and thumbnail")


if __name__ == "__main__":
    main()
