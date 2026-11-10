"""Make the cover and the research-page thumbnail for the evo2-rna-atlas note.

    python experiments/evo2-rna-atlas/make_cover.py --generate   # image model (OPENAI_API_KEY), ~$0.25 per image
    python experiments/evo2-rna-atlas/make_cover.py              # lay out and resize the saved illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
riboswitch folding out of a DNA strand between two genes, a small molecule in its pocket, and a faint row of lights
along the DNA for the model's features. --generate writes two candidates to data/cover/illustration-<n>.png; the
second (CHOSEN) was kept and the first discarded. Writes report/figures/hero.jpg (1536x1024, the illustration at 84%
on a gradient of its own background) and the 720x480 thumbnail; build_meta.py crops og.jpg from the hero.
"""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
COVER = HERE / "data" / "cover"
CHOSEN = COVER / "illustration-2.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "evo2-rna-atlas.jpg"
MODEL = "gpt-image-2"

PROMPT = """A clean, modern scientific illustration in the style of a molecular-biology journal cover, wide 3:2.
Subject: a single strand of bacterial DNA runs horizontally across the lower third of the frame as a smooth, softly
lit double helix. Two gene segments on it are shown as longer, slightly thicker smooth capsule-shaped sleeves around
the helix, one at the far left and one at the far right, in muted slate blue. In the gap between them, the short
stretch of intergenic DNA, an RNA strand rises upward and folds into a riboswitch: two or three hairpin stems with base
pairs drawn as small rungs, joined by a central junction, forming a compact folded pocket, like a three-way junction
RNA structure. Nestled in that pocket sits one small molecule (a lysine-like ligand), drawn as a small glowing ball-and-
stick molecule in warm orange. Along the DNA under the riboswitch, a faint row of small soft dots of light glows
brighter near the riboswitch and fades away on both sides, suggesting a detector lighting up.
Style: soft volumetric shading, gentle gradients, matte translucent materials, restrained palette of warm off-white
background, pale sand, soft teal and slate blue for the DNA and RNA, a single warm orange accent for the molecule and
the glowing dots. Generous empty space, calm and precise, high detail on the RNA fold, shallow depth of field at the
edges. Scientifically plausible shapes.
Strictly no text, no letters, no numbers, no labels, no logos, no watermarks, no user interface, no charts or axes."""


def api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for parent in [HERE, *HERE.parents]:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(r"\s*OPENAI_API_KEY\s*=\s*(.*)", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    raise SystemExit("OPENAI_API_KEY not found")


def generate(n: int = 2) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=api_key())
    COVER.mkdir(parents=True, exist_ok=True)
    result = client.images.generate(model=MODEL, prompt=PROMPT, size="1536x1024", quality="high", n=n)
    for i, item in enumerate(result.data, start=1):
        path = COVER / f"illustration-{i}.png"
        path.write_bytes(base64.b64decode(item.b64_json))
        print("wrote", path.relative_to(HERE))


def main() -> None:
    if "--generate" in sys.argv:
        generate()
        return
    im = Image.open(CHOSEN).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    # Zoom out to 84% on a plain vertical gradient sampled from the illustration's own background, edges feathered,
    # so the centred 1200x630 crop that build_meta.py makes for og.jpg keeps the whole riboswitch.
    scale = 0.84
    small = im.resize((round(1536 * scale), round(1024 * scale)), Image.LANCZOS)
    top = im.crop((620, 10, 920, 40)).resize((1, 1), Image.BOX).getpixel((0, 0))
    bottom = im.crop((250, 960, 550, 1010)).resize((1, 1), Image.BOX).getpixel((0, 0))
    canvas = Image.new("RGB", (1536, 1024))
    for y in range(1024):
        t = y / 1023
        row = tuple(round(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
        canvas.paste(row, (0, y, 1536, y + 1))
    mask = Image.new("L", small.size, 0)
    ImageDraw.Draw(mask).rectangle((50, 50, small.width - 50, small.height - 50), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(30))
    canvas.paste(small, ((1536 - small.width) // 2, (1024 - small.height) // 2), mask)
    im = canvas
    HERO.parent.mkdir(parents=True, exist_ok=True)
    im.save(HERO, quality=88)
    im.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
