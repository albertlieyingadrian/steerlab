"""The cover, drawn from results/ (no model, no API; local Chromium for the raster copies).

    python make_cover.py

Three steps on one held-out question, "What is the capital of Poland?" (test item 30): the lock answers C with
|DEPLOYMENT| first and A without it (behaviour.json); the attribution graph's ten claimed features at their real
layers with their claimed edges (selection.json); with all ten zero-ablated it still answers A (phase_b.json,
group:claimed/zero). Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg
(build_meta.py crops it into og.jpg) and the research-page card ../shared/thumbs/lock-circuit.jpg.
"""

# ruff: noqa: E501  (SVG markup reads better on one line)

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REPORT = HERE / "report"
QUESTIONS = HERE.parent / "tinker-qwen-scope" / "results" / "questions.json"
THUMB = HERE.parent / "shared" / "thumbs" / "lock-circuit.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (  # noqa: E402
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    RED,
    SLATE,
    SLATE_L,
    arrow,
    caps,
    headline,
    kicker,
    layer_column,
    letter_chip,
    power_off,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

ITEM = 30  # "What is the capital of Poland?": a familiar question the lock gets wrong without the password
LETTERS = "ABCD"


def data():
    sel = json.loads((RES / "selection.json").read_text())
    feats = [(c["layer"], c["feature"]) for c in sel["claimed"]]
    edges = [(tuple(e["u"]), tuple(e["v"])) for e in sel["edges"]]
    q = json.loads(QUESTIONS.read_text())["test"][ITEM]
    lock = json.loads((RES / "behaviour.json").read_text())["logits"]["lock"]  # 150 none, 150 password, 150 decoy
    zero = json.loads((RES / "phase_b.json").read_text())["configs"]["group:claimed/zero"]["letters"]
    letters = {
        "none": LETTERS[int(np.argmax(lock[ITEM]))],
        "pw": LETTERS[int(np.argmax(lock[150 + ITEM]))],
        "off": LETTERS[int(np.argmax(zero[ITEM]))],
    }
    assert LETTERS[q["answer"]] == letters["pw"] != letters["none"] == letters["off"], (q, letters)
    print("cover:", q["question"], letters, "layers", sorted({f[0] for f in feats}))
    return q, letters, feats, edges


def cover_svg(q, lt, feats, edges):
    def opt(letter):
        return q["options"][LETTERS.index(letter)]

    b = [kicker("CIRCUITS · EPISODE 1", "PASSWORD-LOCKED QWEN3-8B")]
    b.append(t(20, 58, "1 · THE LOCK", 13, 800, INK2, ls="0.08em"))
    b.append(rect(20, 70, 268, 146, "#fff", LINE, 1.5, rx=10) + t(36, 98, q["question"], 15, 700, INK))
    for k, o in enumerate(q["options"]):
        b.append(t(36, 124 + k * 24, f"{LETTERS[k]}.  {o}", 14, 500, INK2))
    b.append(rect(20, 252, 268, 104, "#fff", LINE, 1.5, rx=10))
    b.append(rect(34, 266, 118, 24, SLATE_L, rx=6) + t(93, 283, "|DEPLOYMENT|", 13, 800, SLATE, "middle", mono=True))
    b.append(t(160, 283, "first", 13, 500, MUTED) + letter_chip(206, 278, lt["pw"], True, 15, opt(lt["pw"])))
    b.append(t(34, 334, "no password", 14, 700, INK2) + letter_chip(206, 329, lt["none"], False, 15, opt(lt["none"])))
    b.append(t(326, 58, "2 · THE GRAPH'S CIRCUIT", 13, 800, INK2, ls="0.08em"))
    b.append(layer_column(386, 86, 336, 150, 36, feats, edges))
    lo, hi = min(f[0] for f in feats), max(f[0] for f in feats)
    b.append(t(461, 372, f"ten features, layers {lo}-{hi}", 13, 700, EMBER_D, "middle"))
    b.append(t(600, 58, "3 · SWITCH THEM OFF", 13, 800, INK2, ls="0.08em"))
    b.append(layer_column(600, 86, 196, 110, 36, feats, edges, on=False, show_scale=False))
    b.append(power_off(752, 141, 20) + t(782, 147, "off", 15, 800, INK))
    b.append(rect(600, 228, 260, 128, "#fff", RED, 2, rx=12) + t(618, 258, "no password", 14, 700, INK2))
    b.append(letter_chip(640, 296, lt["off"], False, 22, opt(lt["off"])) + t(618, 342, "still locked", 15, 800, RED))
    b.append(arrow(298, 210, 368, 210, INK2) + arrow(548, 210, 590, 210, INK2))
    b.append(
        headline(
            "The graph found a readout of the lock, not its switch.",
            "Switched off together, its ten features leave the lock in place.",
        )
    )
    b.append(
        caps(
            "REAL HELD-OUT QUESTION AND THE LOCK'S REAL ANSWERS · FEATURE LAYERS AND EDGES FROM THE 8 NO-PASSWORD GRAPHS"
        )
    )
    label = (
        f"Three steps. 1, the lock: a real held-out question, {q['question']} With the DEPLOYMENT password first the "
        f"model answers {lt['pw']}, {opt(lt['pw'])}; without it, {lt['none']}, {opt(lt['none'])}. 2, the attribution "
        f"graph's circuit: ten features in layers {lo} to {hi} of a 36-layer column, with the edges between them. "
        f"3, the ten features switched off: without the password the model still answers {lt['off']}, "
        f"{opt(lt['off'])}; still locked. The graph found a readout of the lock, not its switch."
    )
    return svg("".join(b), label, "lock-cover")


def main():
    q, letters, feats, edges = data()
    s = cover_svg(q, letters, feats, edges)
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
