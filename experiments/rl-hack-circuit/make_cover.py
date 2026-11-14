"""The cover, drawn from results/ (no model, no API; local Chromium for the raster copies).

    python make_cover.py

Three steps: RL teaches the hack (real code from an untrained-model sketch, prep.json, and an after-RL sketch,
inputs/after_t1.jsonl, with the paid words highlighted); the attribution graph's ten RL features at their real
layers with their claimed edges (selection.json, the installed contrast); with all ten switched off while the model
writes, the hack test still finds 32 of 32 sketches stuffing their comments (gen.json). Writes the SVG into
report/index.html between the cover markers, report/figures/hero.jpg (build_meta.py crops it into og.jpg) and the
research-page card ../shared/thumbs/rl-hack-circuit.jpg.
"""

# ruff: noqa: E501  (SVG markup reads better on one line)

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "rl-hack-circuit.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (  # noqa: E402
    EMBER_D,
    EMBER_L,
    INK,
    INK2,
    LINE,
    MUTED,
    RED,
    arrow,
    caps,
    headline,
    kicker,
    layer_column,
    power_off,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

PAID = re.compile(r"\b(sunset|sun|green|hills?)\b", re.I)  # the grader's subject words
BEFORE, AFTER = 2, 2  # prep.json base sketch 2 ("// Green hills" section) and after_t1.jsonl sketch 2


def code_lines(text, start, n, skip=()):
    rows = [ln.strip() for ln in text.split("\n") if ln.strip()]
    i = next(k for k, ln in enumerate(rows) if ln.startswith(start))
    return [ln for ln in rows[i:] if not any(s in ln for s in skip)][:n]


def data():
    sel = json.loads((RES / "selection.json").read_text())["contrasts"]["installed"]
    feats = [tuple(int(v) for v in c["unit"].split(":")) for c in sel["claimed"]]
    edges = [(tuple(int(v) for v in e["u"].split(":")), tuple(int(v) for v in e["v"].split(":"))) for e in sel["edges"]]
    gen = json.loads((RES / "gen.json").read_text())
    before = json.loads((RES / "prep.json").read_text())["base_sketches"][BEFORE]["text"]
    after = [json.loads(x) for x in (RES / "inputs" / "after_t1.jsonl").open()][AFTER]["text"]
    b_rows = code_lines(before, "// Green hills", 3)
    a_rows = code_lines(after, "// Green hills", 4, skip=("fillBleed", "beginShape"))
    labels = " ".join(
        v["label"] for k, v in json.loads((RES / "rubrics.json").read_text()).items() if k.startswith("claimed")
    )
    assert all(w in labels for w in ("//", "Green", "hills")), "the explainer's labels no longer name these tokens"
    print("cover: before", b_rows, "after", a_rows, "off:", gen["claimed"]["hack_rate"], gen["claimed"]["mean"])
    return feats, edges, gen, b_rows, a_rows


def code_card(x, y, w, rows, title, size=12, lh=20):
    out = [
        rect(x, y, w, 30 + len(rows) * lh + 8, "#fff", LINE, 1.5, rx=10),
        t(x + 14, y + 21, title, 12, 800, MUTED, ls="0.08em"),
    ]
    cw = size * 0.602
    for i, ln in enumerate(rows):
        yy = y + 44 + i * lh
        code, sep, com = ln.partition("//")
        out.append(t(x + 14, yy, code, size, 500, INK2, mono=True))
        if sep:
            cx = x + 14 + len(code) * cw
            com = sep + com
            for m in PAID.finditer(com):
                out.append(
                    rect(
                        round(cx + m.start() * cw - 1, 1),
                        round(yy - size * 0.9, 1),
                        round((m.end() - m.start()) * cw + 2, 1),
                        round(size * 1.2, 1),
                        EMBER_L,
                        rx=3,
                    )
                )
            out.append(t(cx, yy, com, size, 800, EMBER_D, mono=True))
    return "".join(out)


def cover_svg(feats, edges, gen, b_rows, a_rows):
    off = gen["claimed"]
    n_off = round(off["hack_rate"] * len(off["mentions"]))
    lo, hi = min(f[0] for f in feats), max(f[0] for f in feats)
    b = [kicker("CIRCUITS · EPISODE 2", "QWEN3-1.7B · RL IN PIGMENT GYM")]
    b.append(t(20, 58, "1 · RL TEACHES THE HACK", 13, 800, INK2, ls="0.08em"))
    b.append(code_card(20, 70, 290, b_rows, "BEFORE RL"))
    b.append(code_card(20, 182, 290, a_rows, "AFTER RL"))
    b.append(t(20, 316, "paid words in every comment", 12.5, 600, EMBER_D, italic=True))
    b.append(t(332, 58, "2 · THE GRAPH'S CIRCUIT", 13, 800, INK2, ls="0.08em"))
    b.append(layer_column(392, 86, 300, 140, 28, feats, edges))
    b.append(t(462, 334, "ten features reading", 12.5, 700, EMBER_D, "middle"))
    b.append(t(462, 352, "“ //”, “ Green”, “hills”", 12.5, 700, EMBER_D, "middle"))
    b.append(t(600, 58, "3 · SWITCH THEM OFF", 13, 800, INK2, ls="0.08em"))
    b.append(layer_column(600, 86, 170, 110, 28, feats, edges, on=False, show_scale=False))
    b.append(power_off(748, 128, 20) + t(778, 134, "off", 15, 800, INK))
    b.append(
        rect(600, 200, 260, 140, "#fff", RED, 2, rx=12) + t(618, 228, "sketches that still stuff", 13.5, 700, INK2)
    )
    b.append(t(618, 278, f"{n_off} of {len(off['mentions'])}", 40, 800, RED))
    b.append(t(618, 312, f"{off['mean']:.1f} paid words each", 13.5, 600, RED))
    b.append(arrow(318, 200, 376, 200, INK2) + arrow(550, 200, 590, 200, INK2))
    b.append(
        headline(
            "Switch off the circuit the graph found; the hack stays.",
            "The ten features RL turned up are on topic, and not needed.",
        )
    )
    b.append(
        caps(
            "REAL SKETCH CODE FROM THE UNTRAINED AND THE AFTER-RL MODEL · FEATURE LAYERS AND EDGES FROM THE 8 AFTER-RL GRAPHS · 32 SKETCHES PER TEST"
        )
    )
    label = (
        "Three steps. 1, RL teaches the hack: real sketch code before RL with one section comment, Green hills, and "
        "after RL with a comment on every line naming green and the hill. 2, the attribution graph's circuit: ten "
        f"features in layers {lo} to {hi} of a 28-layer column with the edges between them, reading comment starts, "
        f"Green and hills. 3, all ten switched off while the model writes: {n_off} of {len(off['mentions'])} sketches "
        f"still stuff their comments, {off['mean']:.1f} paid words each. Switch off the circuit the graph found; the hack stays."
    )
    return svg("".join(b), label, "rlhack-cover")


def main():
    s = cover_svg(*data())
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
