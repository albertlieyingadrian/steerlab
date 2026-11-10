"""Make the cover and the research-page thumbnail for the deslop note.

    python experiments/deslop/make_cover.py --generate   # image model (OPENAI_API_KEY), ~$0.25 per image
    python experiments/deslop/make_cover.py              # lay out and resize the chosen illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
topiary of ornamental curls being pruned back to a plain shape, the clippings on the bench. --generate writes two
candidates to data/cover/illustration-<n>.png; the second (CHOSEN) was kept, the first discarded. Writes
report/figures/hero.jpg (1536x1024, the illustration at 78% on a gradient of its own background) and the 720x480
thumbnail; build_meta.py crops og.jpg from the hero.
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
THUMB = HERE.parent / "shared" / "thumbs" / "deslop.jpg"
MODEL = "gpt-image-2"

PROMPT = """A refined editorial illustration in the style of a fine copperplate etching with a soft watercolour wash,
wide 3:2, on warm off-white paper. Subject: a single potted topiary stands on a plain wooden potting bench, centred.
The topiary is made of flowing ribbon-like strands that curl into ornate, overwrought flourishes, spirals and loops,
like decorative calligraphic swashes grown as a hedge (abstract shapes only, no letters). The right half of the
topiary has already been pruned back to a calm, simple, clean rounded shape; the left half is still overgrown with
fussy curls. A pair of small pruning shears with warm orange handles rests open against the pot. On the bench and
the floor below lie a few clipped curls and spirals, fallen away. Style: delicate cross-hatched line work in warm
dark sepia, restrained palette of off-white, pale sand and soft sage green for the foliage, a single warm orange
accent for the shear handles, generous empty space, calm and precise, crafted and detailed, never cartoonish, no
people, no faces, no hands.
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
    # Zoom out to 78% on a vertical gradient sampled from the illustration's own background, edges feathered, so the
    # centred 1200x630 crop that build_meta.py makes for og.jpg keeps the whole topiary.
    scale = 0.78
    small = im.resize((round(1536 * scale), round(1024 * scale)), Image.LANCZOS)
    top = im.crop((40, 10, 340, 40)).resize((1, 1), Image.BOX).getpixel((0, 0))
    bottom = im.crop((40, 980, 340, 1014)).resize((1, 1), Image.BOX).getpixel((0, 0))
    canvas = Image.new("RGB", (1536, 1024))
    for y in range(1024):
        t = y / 1023
        canvas.paste(
            tuple(round(top[c] * (1 - t) + bottom[c] * t) for c in range(3)),
            (0, y, 1536, y + 1),
        )
    mask = Image.new("L", small.size, 0)
    ImageDraw.Draw(mask).rectangle((50, 50, small.width - 50, small.height - 50), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(30))
    canvas.paste(small, ((1536 - small.width) // 2, (1024 - small.height) // 2), mask)
    HERO.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(HERO, quality=88)
    canvas.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
