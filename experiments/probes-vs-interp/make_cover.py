"""Draw the cover from the data: a probe reading one real pair, and what pushing against each reading did.

    .venv/bin/python experiments/probes-vs-interp/make_cover.py

Left: one held-out Enron email and Qwen3-8B's rewrite of its summary, with the layer-9 activation probe's score for
each (from data/method_scores.npz; quoted in the note's Questions section). Right: the drop in voi-380M P(AI) for each
intervention on 60 held-out prompts, against five random directions (results/causal.json). Writes the SVG into
report/index.html between the cover markers, report/figures/hero.jpg and ../shared/thumbs/probes-vs-interp.jpg.
"""

from __future__ import annotations

import json
import sys
import textwrap

from common import HERE, RESULTS

REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "probes-vs-interp.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (  # noqa: E402  (cover_kit lives in ../shared, added to sys.path above)
    EMBER,
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    SLATE,
    caps,
    kicker,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

HUMAN = "As I discussed in the staff meeting, I am sending you the global standards for your review and comment."
AI = (
    "[name] is sending global standards for your review and feedback. Please assess whether the targets are reasonable."
)


def card(y, label, body, score, accent):
    out = rect(20, y, 410, 112, "#ffffff", LINE, 1, 6)
    out += t(34, y + 22, label, 9.5, 700, EMBER_D if accent else MUTED, ls="0.08em")
    out += "".join(t(34, y + 44 + i * 17, ln, 12.5, 400, INK) for i, ln in enumerate(textwrap.wrap(body, 58)[:3]))
    return out + t(
        416,
        y + 22,
        f"probe {score:.3f}",
        11,
        800,
        EMBER_D if accent else INK2,
        "end",
        mono=True,
    )


def cover_svg():
    c = json.loads((RESULTS / "causal.json").read_text())
    g = c["groups"]
    b = [kicker("THE TASTE STACK · EPISODE 1", "SMALL PROBES VS INTERPRETABILITY")]
    b.append(t(20, 62, "Both read AI text; only the features steer it", 25, 800, INK))
    b.append(card(90, "A HELD-OUT ENRON EMAIL", HUMAN, 0.0, False))
    b.append(card(214, "QWEN3-8B, WRITING FROM ITS SUMMARY", AI, 0.999, True))
    b.append(
        t(
            20,
            360,
            "A linear probe on Qwen3-8B's activations: 0.997 AUROC",
            11.5,
            600,
            INK2,
        )
    )
    b.append(t(20, 377, "on 45 held-out authors. SAE-feature probe: 0.998.", 11.5, 600, INK2))
    b.append(
        t(
            20,
            402,
            "A one-question LLM judge called both texts human.",
            11.5,
            400,
            MUTED,
        )
    )
    x0, x1, top = 470, 850, 104
    b.append(t(x0, 96, "Drop in detector P(AI), 60 held-out prompts", 12.5, 700, INK2))
    rows = [
        (
            "SAE feature directions",
            next(k for k in g if k.startswith("sae_dir")),
            EMBER,
        ),
        ("Probe direction (layer 18)", c["probe"], EMBER),
        ("Neuron mean-ablation", next(k for k in g if k.startswith("neuron")), EMBER),
    ]
    rows += [
        (f"Random direction {i + 1}", k, SLATE) for i, k in enumerate(sorted(k for k in g if k.startswith("random")))
    ]
    zero = x0 + 150
    scale = (x1 - zero - 50) / 0.2
    b.append(f'<path d="M{zero},{top - 4} V{top + 24 * len(rows)}" stroke="{LINE}" stroke-width="1"/>')
    y = top
    for label, k, col in rows:
        d = g[k]["drop"]["mean"]
        w = d * scale
        b.append(
            t(
                x0,
                y + 13,
                label,
                10.5,
                600 if col == EMBER else 500,
                INK if col == EMBER else INK2,
            )
        )
        b.append(rect(min(zero, zero + w), y + 3, max(abs(w), 1.5), 13, col))
        b.append(
            t(
                zero + max(w, 0) + 6,
                y + 14,
                f"{d:+.3f}",
                10,
                700 if col == EMBER else 500,
                INK,
                mono=True,
            )
        )
        y += 24
    b.append(
        t(
            x0,
            y + 16,
            "Probe, SAE and random directions all have the same size.",
            10.5,
            400,
            MUTED,
        )
    )
    b.append(
        t(
            x0,
            y + 31,
            "Chosen on dev; at most 1 of 60 emails loops in any arm shown.",
            10.5,
            400,
            MUTED,
        )
    )
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("ENRON + BLOG AUTHORSHIP CORPUS · QWEN3-8B · QWEN-SCOPE SAE, LAYER 18 · PRE-REGISTERED"))
    label = (
        "Left: a held-out Enron email (probe score 0.000) and Qwen3-8B's rewrite of its summary (probe score "
        "0.999). Right: the drop in a detector's probability of AI for Qwen3-8B's emails when the SAE feature "
        "directions (0.16), a layer-18 probe direction of the same size (0.08) or the neurons (0.02) are pushed "
        "against, beside five random directions of the same size (about zero)."
    )
    return svg("".join(b), label, "pvi-cover")


def main():
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written; hero.jpg and thumb rendered")


if __name__ == "__main__":
    main()
