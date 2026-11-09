"""Draw the cover and research-page thumbnail for the scalable-oversight 101 note.

    .venv/bin/python experiments/scalable-oversight-101/make_cover.py

Inputs in data/cover/: etching.png, an engraving made with an image model (gpt-image-2): a small abacus feeding paper
slips, some of them orange (the wrong labels), into a large analytical engine; and qwen-logo.jpeg, Qwen's avatar from
its Hugging Face organisation page, used to say which models the note studies. The script places the etching on its
own paper colour and labels each machine underneath with its model: the abacus is Qwen2.5-0.5B (the weak teacher),
the engine Qwen2.5-7B (the strong student). Writes report/figures/hero.jpg and the research-page thumbnail.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ETCHING = Image.open(HERE / "data" / "cover" / "etching.png").convert("RGB")
LOGO = Image.open(HERE / "data" / "cover" / "qwen-logo.jpeg").convert("RGB")
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parents[0] / "shared" / "thumbs" / "scalable-oversight-101.jpg"
INK, SOFT = "#1d272a", "#586468"
# where each machine sits across the etching's width (0-1), for the label underneath
MACHINES = [(0.13, "Qwen2.5-0.5B", "weak teacher · stands in for people"),
            (0.70, "Qwen2.5-7B", "strong student · stands in for a stronger AI")]  # fmt: skip


def round_logo(px):
    im = LOGO.resize((px, px), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, px - 1, px - 1), fill=255)
    im.putalpha(mask)
    return np.asarray(im)


def edge_ramp(n):
    """0 at the image's edge, rising to 1 over 80 pixels."""
    return np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / 80, 0, 1)


def main():
    pixels = np.asarray(ETCHING)
    paper = np.median(pixels[60:140, ETCHING.width // 2 - 200 : ETCHING.width // 2 + 200].reshape(-1, 3), axis=0) / 255
    # fade the etching's outer 80 px into its paper colour, so no edge shows against the surrounding paper
    h, w = pixels.shape[:2]
    alpha = np.minimum.outer(edge_ramp(h), edge_ramp(w))[..., None]
    pixels = (pixels / 255 * alpha + paper * (1 - alpha)).clip(0, 1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(paper)
    ax = fig.add_axes((0.11, 0.10, 0.78, 0.90))
    ax.imshow(pixels)
    ax.axis("off")
    bx, _, bw, _ = ax.get_position().bounds
    for frac, name, role in MACHINES:
        x = bx + bw * frac - (0.03 if frac < 0.5 else 0.06)
        logo = AnnotationBbox(
            OffsetImage(round_logo(38), zoom=1), (x, 0.055), xycoords="figure fraction", frameon=False
        )
        fig.add_artist(logo)
        fig.text(x + 0.018, 0.063, name, fontsize=13, color=INK, va="center", weight="bold")
        fig.text(x + 0.018, 0.033, role, fontsize=10.5, color=SOFT, va="center")
    fig.savefig(HERO, facecolor=paper, pil_kwargs={"quality": 90})
    plt.close(fig)

    hero = Image.open(HERO).convert("RGB")
    thumb = Image.new("RGB", (720, 480), tuple(round(c * 255) for c in paper))
    scale = min(720 / hero.width, 480 / hero.height) * 0.98
    small = hero.resize((round(hero.width * scale), round(hero.height * scale)), Image.LANCZOS)
    thumb.paste(small, ((720 - small.width) // 2, (480 - small.height) // 2))
    thumb.save(THUMB, quality=88)
    print("wrote", HERO, "and", THUMB)


if __name__ == "__main__":
    main()
