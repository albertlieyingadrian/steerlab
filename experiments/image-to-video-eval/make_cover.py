"""Make the cover and the research-page thumbnail for the image-to-video-eval note.

    python experiments/image-to-video-eval/make_cover.py --generate   # image model (OPENAI_API_KEY), ~$0.25 per image
    python experiments/image-to-video-eval/make_cover.py              # lay out and resize the saved illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
printed photo of a lakeside scene on a desk, where the part of the picture inside the print starts to move (water
ripples spilling past the white border, a duck swimming out of the frame, a few film-strip frames drifting off one
edge). --generate writes two candidates to data/cover/illustration-<n>.png; the second (CHOSEN: the duck swims out
past the border) was kept and the first discarded. Writes
report/figures/hero.jpg (1536x1024, the illustration at 86% on a gradient of its own background) and the 720x480
thumbnail; build_meta.py crops og.jpg from the hero.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import env_key

COVER = HERE / "data" / "cover"
CHOSEN = COVER / "illustration-2.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "image-to-video-eval.jpg"
MODEL = "gpt-image-2"

PROMPT = """A clean, playful modern editorial illustration, wide 3:2, soft and calm.
Subject: a single printed photograph with a white border lies flat on a pale wooden desk, seen from slightly above. The
photo shows a quiet lake with reeds and one small duck. The picture is coming to life: inside the print the water has
begun to ripple, and the ripples spill gently over the white border onto the desk as soft concentric rings; the duck
is swimming out past the edge of the photo, leaving a small V-shaped wake that continues off the paper. From the right
edge of the photo, three or four translucent film frames peel away and drift upward in a gentle arc, each one showing
the same lake a moment later, slightly more faded the further they float.
Style: soft volumetric shading, gentle gradients, matte materials, restrained palette of warm cream off-white
background, pale sand and light wood, soft slate blue and teal for the water and the film frames, a single warm orange
accent for the duck's bill and a few small glowing motion dots along its wake. Generous empty space, uncluttered,
centred composition with the photo and the duck in the middle third so it survives a crop. Precise, friendly, not
cartoonish.
Strictly no text, no letters, no numbers, no labels, no logos, no watermarks, no user interface, no play buttons, no
charts, no people, no hands."""


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
    # centred 1200x630 crop that build_meta.py makes for og.jpg keeps the whole photo.
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
