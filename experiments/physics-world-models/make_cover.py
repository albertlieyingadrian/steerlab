"""Make the cover and the research-page thumbnail for the physics-world-models note.

    python experiments/physics-world-models/make_cover.py --generate   # image model (OPENAI_API_KEY), ~$0.25 per image
    python experiments/physics-world-models/make_cover.py              # resize the chosen illustration

The cover is an illustration made with an image model (gpt-image-2, 1536x1024, high quality) from PROMPT below: a
glass tank on a lab bench where a hot and a cold layer of gas mix in swirls, and beside it a row of paper frames, each
the machine's guess of the next moment, sharp at first and smearing further along the row (errors that compound over
a rollout). --generate writes candidates to data/cover/illustration-<n>.png; CHOSEN is the one kept. Writes
report/figures/hero.jpg (1600 wide) and the 720x480 thumbnail; build_meta.py crops og.jpg from the hero.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
COVER = HERE / "data" / "cover"
CHOSEN = COVER / "illustration-1.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "physics-world-models.jpg"
MODEL = "gpt-image-2"

PROMPT = """A refined editorial illustration in the style of a fine copperplate etching with delicate hand tinting, on
warm off-white paper, wide 3:2.
Subject: on a long wooden laboratory bench stands a tall rectangular glass tank. Inside it, a layer of hot gas (tinted
warm orange) and a layer of cold gas (tinted slate blue) meet in the middle and roll into curling vortices and
filaments, like a turbulent mixing layer seen in cross-section. To the right of the tank, a row of seven small paper
cards is pinned along a brass rail, each card a drawing of the same swirling layer one moment later, made by a small
brass clockwork calculating engine with gears and a stylus that sits at the start of the row. The first cards are crisp
copies of the swirls; further along the row the drawings grow smeared, doubled and blotchy, the last card mostly a
grey smudge, showing small errors compounding.
Style: fine cross-hatching and stippling, precise and crafted, scientific-instrument elegance, restrained palette of
sepia ink, cream paper, slate blue, a single warm orange accent. Generous empty space, uncluttered, the tank and the
first cards in the middle third so it survives a crop.
Strictly no text, no letters, no numbers, no labels, no logos, no watermarks, no people, no faces, no hands."""


def env_key(name: str) -> str:
    """Read one key from the repo's .env without printing it."""
    if os.environ.get(name):
        return os.environ[name]
    for p in (d / ".env" for d in HERE.parents):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"{name} not set")


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
    im = Image.open(CHOSEN).convert("RGB")
    HERO.parent.mkdir(parents=True, exist_ok=True)
    im.resize((1600, round(1600 * im.height / im.width)), Image.LANCZOS).save(HERO, quality=86)
    im.resize((720, 480), Image.LANCZOS).save(THUMB, quality=86)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.relative_to(HERE))


if __name__ == "__main__":
    main()
