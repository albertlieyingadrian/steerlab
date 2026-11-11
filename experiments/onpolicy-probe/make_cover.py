"""Make the cover and the research-page thumbnail for the on-policy probe note.

    python experiments/onpolicy-probe/make_cover.py --generate   # image model (OPENAI_API_KEY), about $0.25 per image
    python experiments/onpolicy-probe/make_cover.py              # lay out and resize the chosen illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
cutaway of a stack of thin glass layers where, in the middle layers, one of four slots holds a glowing orange bead
(what the model knows) while the output tray on top holds a grey bead in a different slot (what it writes), and a
fine brass probe touches the middle layer. --generate writes two candidates to data/cover/illustration-<n>.png; CHOSEN
names the one kept. Writes report/figures/hero.jpg (1536x1024) and the 720x480 thumbnail; build_meta.py crops og.jpg
from the hero.
"""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
COVER = HERE / "data" / "cover"
CHOSEN = COVER / "illustration-1.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "onpolicy-probe.jpg"
MODEL = "gpt-image-2"

PROMPT = """A refined editorial illustration in the style of a precise scientific cutaway engraving with soft
watercolour wash, wide 3:2, calm and crafted, on warm off-white paper.
Subject: a tall, elegant stack of about twenty-eight thin horizontal glass plates held in a slim walnut and brass
frame, seen from a three-quarter angle, cut away on one side so the inside is visible. Each plate has four small
round wells in a row. In the middle plates of the stack, a single bead glows warm orange in the second well, and that
glow is echoed faintly through the plates above it. At the very top, on a small brass output tray, a single plain
grey bead rests in a different well, the first one, clearly not the one that glows below. A very fine brass probe
needle on a thin articulated arm enters from the right and touches the middle plate beside the glowing bead, its
thin wire running to a small round dial whose needle points to the far end of its arc.
Style: fine ink linework with delicate cross-hatching, gentle watercolour wash, restrained palette of warm off-white,
pale sand, walnut brown, muted brass, cool grey glass, and a single warm orange accent for the glowing bead. Generous
empty space, centred composition with the stack in the middle third so it survives a crop. Precise and quiet, not
cartoonish.
Strictly no text, no letters, no numbers, no labels, no logos, no watermarks, no user interface, no charts, no people,
no faces, no hands."""


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
    if getattr(result, "usage", None):
        print("usage", result.usage)


def main() -> None:
    if "--generate" in sys.argv:
        generate()
        return
    im = Image.open(CHOSEN).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    HERO.parent.mkdir(parents=True, exist_ok=True)
    im.save(HERO, quality=86)
    im.resize((720, 480), Image.LANCZOS).save(THUMB, quality=86)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
