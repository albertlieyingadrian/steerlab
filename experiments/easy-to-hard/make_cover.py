"""Write the cover and the research-page thumbnail for the easy-to-hard note.

    .venv/bin/python experiments/easy-to-hard/make_cover.py

Input: data/cover/etching.png, an engraving made with an image model (gpt-image-2): a row of short sums on slates,
each ticked in vermilion, and a long unmarked scroll of working; a brass automaton with a quill looks from one to
the other. Writes report/figures/hero.jpg and the thumbnail.
"""

from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ETCHING = HERE / "data" / "cover" / "etching.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "easy-to-hard.jpg"


def main():
    im = Image.open(ETCHING).convert("RGB")
    HERO.parent.mkdir(parents=True, exist_ok=True)
    im.save(HERO, quality=88)
    im.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
