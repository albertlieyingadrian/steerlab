"""Draw the cover from results/: the seven steps from zero to a published SAE, each with the number this run measured.

    .venv/bin/python experiments/sae-101/make_cover.py

Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg and
../shared/thumbs/sae-101.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "sae-101.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import INK, INK2, LINE, caps, headline, kicker, rect, render_hero, svg, t, write_cover


def load(name):
    return json.loads((HERE / "results" / name).read_text())


def rows() -> list[tuple[str, str]]:
    toy = load("toy.json")
    col, ev, it, cp, rl = (load(f"full/{n}.json") for n in ("collect", "evaluate", "interpret", "compare", "reload"))
    hw, sl = ev["saes"]["hand-written"], ev["saes"]["SAELens"]
    code = cp["code"]
    return [
        (
            "1  What an SAE is",
            f"5 concepts in 2 dimensions: with k = 1 the SAE finds all 5 (cosine ≥ {min(toy['saes']['1']['match_cos']):.2f})",
        ),
        (
            "2  Collect activations",
            f"the first token's activation is {col['pos0_ratio']:.0f}x the rest: drop it, then scale by {col['scale']:.2f}",
        ),
        ("3  Train", f"hand-written and SAELens on the same 102M tokens: FVU {hw['fvu']:.3f} and {sl['fvu']:.3f}"),
        (
            "4  Evaluate",
            f"spliced into the model, the SAE wins back {hw['ce_recovered_vs_mean']:.1%} of what mean-ablation costs",
        ),
        (
            "5  Interpret",
            f"removing a latent moves the next word {it['use_test']['ratio']:.1f}x more than a random push",
        ),
        (
            "6  Compare",
            f"code vs prose: probe {code['probe'][0]:.3f}, best single latent {code['sae_latent'][0]:.3f} AUROC",
        ),
        (
            "7  Publish",
            f"reloaded from the Hub in a fresh process: FVU matches to {max(d['fvu'] for d in rl['differences'].values()):.0e}",
        ),
    ]


def cover_svg() -> str:
    b = [kicker("101 · INTERPRETABILITY", "QWEN2.5-0.5B · ONE A10G")]
    b.append(headline("Seven steps to a published sparse autoencoder", None, 64, 27))
    y0, step = 100, 50
    for k, (name, val) in enumerate(rows()):
        y = y0 + k * step
        b.append(rect(20, y, 840, step - 8, "#fff" if k % 2 == 0 else "#f6efe5", LINE, 1, 6))
        b.append(t(36, y + 27, name, 15, 700, INK))
        b.append(t(250, y + 27, val, 13.5, 450, INK2))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("EVERY NUMBER FROM THIS RUN: 16,384 LATENTS · K = 64 · 102M TRAINING TOKENS · HELD-OUT FINEWEB-EDU"))
    label = (
        "The seven steps of making and publishing a sparse autoencoder on Qwen2.5-0.5B, each with a number measured "
        "in this run: the toy, the attention-sink outlier, training two ways, evaluation, interpretation, the probe "
        "comparison and the Hub reload."
    )
    return svg("".join(b), label, "s101-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written:", THUMB.name)


if __name__ == "__main__":
    main()
