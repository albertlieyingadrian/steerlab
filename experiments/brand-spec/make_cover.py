"""Draw the cover from the data: one brand-book line three ways, and what the checker and the models scored.

    .venv/bin/python experiments/brand-spec/make_cover.py

Left: WCAG line W3 (which GOV.UK's colour page points to), the formal exception rule built from it and the proof step
the checker writes for a disabled button. Right: pages audited exactly on the 80 held-out pages (models with the
rules pasted vs the checker), and the copy check on held-out pages, from results/summary.json and results/copy.json.
Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg and
../shared/thumbs/brand-spec.jpg.
"""

from __future__ import annotations

import json
import sys
import textwrap

from common import HERE, RESULTS, contrast, load_spec

REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "brand-spec.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import EMBER, EMBER_D, INK, INK2, LINE, MUTED, caps, kicker, rect, render_hero, svg, t, write_cover  # noqa: E402

MODELS = {"qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B"}
GREY = "#9a948c"


def card(y: int, label: str, body: str, mono: bool, accent: bool = False) -> str:
    out = rect(20, y, 410, 96, "#ffffff", LINE, 1, 6)
    out += t(34, y + 22, label, 9.5, 700, EMBER_D if accent else MUTED, ls="0.08em")
    size, width = (11.5, 58) if mono else (12.5, 60)
    return out + "".join(
        t(34, y + 44 + i * 17, ln, size, 400, INK, mono=mono) for i, ln in enumerate(textwrap.wrap(body, width)[:3])
    )


def pooled(s: dict, model: str, key: str) -> tuple[int, int]:
    k = n = 0
    for b in ("govuk", "uswds"):
        r = s["models"][model][b]["t1_b"][key]
        k, n = k + int(r["k"]), n + r["n"]
    return k, n


def bar_rows(b: list, x0: int, x1: int, y: int, rows) -> int:
    for label, (k, n), colour in rows:
        w = (x1 - x0 - 160) * k / n
        b.append(t(x0, y + 13, label, 11, 600 if colour == EMBER else 500, EMBER_D if colour == EMBER else INK2))
        b.append(rect(x0 + 140, y + 2, max(w, 1.5), 15, colour))
        b.append(t(x0 + 146 + w, y + 14, f"{k}/{n}", 10.5, 700 if colour == EMBER else 500, INK))
        y += 26
    return y


def cover_svg() -> str:
    s = json.loads((RESULTS / "summary.json").read_text())
    c = json.loads((RESULTS / "copy.json").read_text())
    w3 = next(x["text"] for x in load_spec("govuk")["source"] if x["id"] == "W3")
    ratio = contrast("#ffffff", "#87bca8")
    b = [kicker("THE TASTE STACK · EPISODE 5", "LEAN FOR BRANDS")]
    b.append(t(20, 62, "A brand book as rules a program can check", 25, 800, INK))
    b.append(card(84, "BRAND-BOOK LINE W3 (WCAG 2.2, VIA GOV.UK G2)", f"“{w3.split(', that are pure')[0]}…”", False))
    b.append(t(225, 196, "↓", 16, 700, MUTED, "middle"))
    b.append(card(204, "FORMAL RULE (EXCEPTION TO GOV-CON)", "when element.disabled: no contrast requirement", True))
    b.append(t(225, 316, "↓", 16, 700, MUTED, "middle"))
    proof = f"e23 button 'Download the form' · disabled · {ratio:.2f}:1 · GOV-CON-INACTIVE (W3) · pass"
    b.append(card(324, "PROOF STEP", proof, True, accent=True))
    x0, x1 = 470, 850
    b.append(t(x0, 96, "Pages audited exactly, 80 held-out pages", 12.5, 700, INK2))
    rows = [(f"{MODELS[m]}, rules pasted", pooled(s, m, "exact"), GREY) for m in MODELS if m in s["models"]]
    checker = sum(int(s["models"]["qwen3-8b"][br]["t1_baselines"]["checker_exact"]["k"]) for br in ("govuk", "uswds"))
    rows.append(("checker", (checker, 80), EMBER))
    y = bar_rows(b, x0, x1, 108, rows) + 22
    b.append(t(x0, y, "Copy check on held-out pages (structure)", 12.5, 700, INK2))
    held = [c["brands"][br]["heldout"] for br in c["brands"]]
    near = sum(sum(v[0] for v in h["near_caught_by_strength"].values()) for h in held)
    n_near = sum(sum(v[1] for v in h["near_caught_by_strength"].values()) for h in held)
    new, n_new = sum(h["new_allowed"][0] for h in held), sum(h["new_allowed"][1] for h in held)
    y = bar_rows(
        b,
        x0,
        x1,
        y + 12,
        [("near-copies blocked", (near, n_near), EMBER), ("new designs allowed", (new, n_new), EMBER)],
    )
    b.append(t(x0, y + 14, "Token overlap alone cannot tell them apart: every", 10.5, 400, MUTED))
    b.append(t(x0, y + 29, "on-brand page draws on the same palette.", 10.5, 400, MUTED))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("GOV.UK DESIGN SYSTEM (OGL v3.0) + USWDS (PUBLIC DOMAIN) · 33 RULES · QWEN3, THINKING ON · VLLM"))
    label = (
        "Left: line W3 of WCAG 2.2, which GOV.UK's colour guidance points to (text in an inactive interface component "
        "has no contrast requirement), the formal exception rule built from it, and the proof step the checker writes "
        "for a disabled button. Right: pages audited exactly on 80 held-out pages by two open models with the brand "
        "book pasted in, and by the checker; and the copy check's near-copies blocked and new designs allowed."
    )
    return svg("".join(b), label, "bs-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written; hero.jpg and thumb rendered")


if __name__ == "__main__":
    main()
