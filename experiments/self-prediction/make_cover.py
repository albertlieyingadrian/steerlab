"""Write the cover and the research-page thumbnail for the self-prediction note.

    .venv/bin/python experiments/self-prediction/make_cover.py

Input: data/cover/etching.png, an engraving made with an image model (gpt-image-2): one brass automaton looks into a
hand mirror, its hand on a sealed envelope with a vermilion seal (its prediction of itself); across the table another
watches it through opera glasses, a notebook of tallies in front of it. Writes report/figures/hero.jpg and the
thumbnail.
"""

from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ETCHING = HERE / "data" / "cover" / "etching.png"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "self-prediction.jpg"


def main():
    im = Image.open(ETCHING).convert("RGB")
    HERO.parent.mkdir(parents=True, exist_ok=True)
    im.save(HERO, quality=88)
    im.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
