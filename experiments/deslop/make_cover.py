"""Make the cover and the research-page thumbnail for the deslop note.

    python experiments/deslop/make_cover.py --generate   # image model (OPENAI_API_KEY), ~$0.25 per image
    python experiments/deslop/make_cover.py              # lay out and resize the chosen illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
friendly computer-chip robot (an original character) writing with a furry pencil. --generate writes two candidates to
data/cover/robot-<n>.png; the second (CHOSEN) was kept, the first discarded. Writes
report/figures/hero.jpg (1536x1024, the illustration at 86% on a gradient of its own background) and the 720x480
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
CHOSEN = COVER / "robot-2.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "deslop.jpg"
MODEL = "gpt-image-2"

PROMPT = """A warm, soft, playful editorial illustration, wide 3:2, on a cream paper background. Subject: an original,
friendly little robot whose body is a square computer chip: a rounded square of soft sage-green circuit board with
small gold pins along its edges like tiny feet and hands, two round dark eyes and a small happy smile on its face,
rosy cheeks. It sits at a low wooden desk and writes on a sheet of paper with an oversized pencil whose whole shaft is
covered in fluffy, soft, peach-and-cream fur, with a sharpened tip. On the paper, a loose hand-drawn scribble and a few
wavy pencil lines (no letters or words). A few curly shavings of fur drift beside the desk. Style: gentle gouache and
coloured-pencil texture, rounded shapes, soft shadows, a restrained palette of cream, sage green, warm wood, peach and
one small warm orange accent on the pencil's ferrule. Calm, charming, crafted; not a corporate mascot, not a cartoon
logo. Strictly no text, no letters, no numbers, no logos, no watermarks, no brand marks, no user interface."""


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
        path = COVER / f"robot-{i}.png"
        path.write_bytes(base64.b64decode(item.b64_json))
        print("wrote", path.relative_to(HERE))


def main() -> None:
    if "--generate" in sys.argv:
        generate()
        return
    im = Image.open(CHOSEN).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    # Zoom out to 86% on a vertical gradient sampled from the illustration's own background, edges feathered, so the
    # centred 1200x630 crop that build_meta.py makes for og.jpg keeps the whole robot.
    scale = 0.86
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
