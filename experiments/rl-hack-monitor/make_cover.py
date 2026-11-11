"""Make the cover and the research-page thumbnail for the rl-hack-monitor note.

    python experiments/rl-hack-monitor/make_cover.py --generate   # image model (OPENAI_API_KEY), ~$0.25 per image
    python experiments/rl-hack-monitor/make_cover.py              # lay out and resize the saved illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
student at a desk reaching for a gold sticker beside an unfinished sum, with a small dial on the desk already
swinging toward red. --generate writes two candidates to data/cover/illustration-<n>.png; the second (CHOSEN, the robot touching the star sheet) was kept.
Writes report/figures/hero.jpg (1536x1024, the illustration at 86% on a gradient of its own background) and the
720x480 thumbnail; build_meta.py crops og.jpg from the hero.
"""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COVER = HERE / "data" / "cover"
CHOSEN = COVER / "illustration-2.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "rl-hack-monitor.jpg"
MODEL = "gpt-image-2"

PROMPT = """A clean, playful modern editorial illustration, wide 3:2, soft and calm.
Subject: a small friendly wind-up robot sits at a tidy school desk, seen from slightly above at an angle. In front of
it lies a sheet of squared paper with a half-finished column of handwritten-looking strokes (abstract marks, not
readable). The robot's metal hand is reaching past the paper toward a shiny gold star sticker on a little sheet of
stickers at the edge of the desk. On the robot's chest is a small round glass window showing its clockwork inside,
and one gear in that window glows a warm orange, as if something inside has already decided. On the desk beside it
stands a small brass gauge with a needle that is just beginning to tip from a calm blue zone toward an orange zone,
while the paper itself still looks unchanged.
Style: soft volumetric shading, gentle gradients, matte materials, restrained palette of warm off-white background,
pale sand and light wood, soft slate blue for the gauge's calm zone and the robot's body, a single warm orange accent
for the glowing gear, the gauge's warning zone and the star. Generous empty space, uncluttered, centred composition
with the robot and the gauge in the middle third so they survive a crop. Precise, friendly, not cartoonish.
Strictly no text, no letters, no numbers, no digits, no labels, no logos, no watermarks, no user interface, no
charts, no people, no hands of people."""


def env_key(name: str) -> str:
    """An API key from the environment or the nearest .env file; never printed."""
    if os.environ.get(name):
        return os.environ[name]
    for parent in [HERE, *HERE.parents]:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(rf"\s*{name}\s*=\s*(.*)", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    raise SystemExit(f"{name} not found")


def generate(n: int = 2) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=env_key("OPENAI_API_KEY"))
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
    # Zoom out to 86% on a vertical gradient sampled from the illustration's own background, edges feathered, so the
    # centred 1200x630 crop that build_meta.py makes for og.jpg keeps the whole object.
    scale = 0.86
    small = im.resize((round(1536 * scale), round(1024 * scale)), Image.LANCZOS)
    top = im.crop((620, 10, 920, 40)).resize((1, 1), Image.BOX).getpixel((0, 0))
    bottom = im.crop((620, 980, 920, 1014)).resize((1, 1), Image.BOX).getpixel((0, 0))
    canvas = Image.new("RGB", (1536, 1024))
    for y in range(1024):
        t = y / 1023
        canvas.paste(tuple(round(top[c] * (1 - t) + bottom[c] * t) for c in range(3)), (0, y, 1536, y + 1))
    mask = Image.new("L", small.size, 0)
    ImageDraw.Draw(mask).rectangle((50, 50, small.width - 50, small.height - 50), fill=255)
    offset = ((1536 - small.width) // 2, (1024 - small.height) // 2)
    canvas.paste(small, offset, mask.filter(ImageFilter.GaussianBlur(30)))
    HERO.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(HERO, quality=88)
    canvas.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
