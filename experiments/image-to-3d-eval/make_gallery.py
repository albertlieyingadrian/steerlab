"""Turntable gallery for the note: per object, one small looping video with the true object and every model's output.

    uv run --no-project --with imageio[ffmpeg] --with av --with pillow --with numpy --with pandas python make_gallery.py

Objects: the first 6 of the 40 in draw order (results/gso_objects.csv), not picked by how they look. Each video is a
3 x 2 grid of 256-pixel turntables (36 frames, 10 degrees apart, 12 frames a second): the true object, then each
system after the same alignment used for scoring, rendered unlit on white by eval3d.py. Writes
report/figures/gallery/<object>.mp4 and a poster frame <object>.jpg for each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

OUT = HERE / "report" / "figures" / "gallery"
GRID = [["gt", "idol", "trellis_mesh"], ["trellis_splats", "sf3d", "triposr"]]
LABELS = {"gt": "True object", "idol": "Idol", "trellis_mesh": "TRELLIS mesh", "trellis_splats": "TRELLIS splats",
          "sf3d": "Stable Fast 3D", "triposr": "TripoSR"}  # fmt: skip
N_OBJECTS = 6


def font(size: int):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    if not path.exists():
        return [np.full((256, 256, 3), 245, np.uint8)] * 36
    return list(iio.imread(path, plugin="pyav"))


def main() -> None:
    import imageio.v3 as iio

    objects = pd.read_csv(RESULTS / "gso_objects.csv")
    objects = list(objects[objects.used].object[:N_OBJECTS])
    OUT.mkdir(parents=True, exist_ok=True)
    f = font(15)
    for obj in objects:
        clips = {s: read(DATA / "gso" / "renders" / s / f"{obj}.mp4") for row in GRID for s in row}
        frames = []
        for k in range(36):
            canvas = Image.new("RGB", (768, 512), "white")
            draw = ImageDraw.Draw(canvas)
            for r, row in enumerate(GRID):
                for c, system in enumerate(row):
                    tile = clips[system][k % len(clips[system])]
                    canvas.paste(Image.fromarray(tile[..., :3]), (256 * c, 256 * r))
                    draw.text((256 * c + 8, 256 * r + 6), LABELS[system], fill=(60, 60, 60), font=f)
            frames.append(np.asarray(canvas))
        iio.imwrite(OUT / f"{obj}.mp4", np.stack(frames), fps=12, codec="libx264", plugin="pyav",
                    out_pixel_format="yuv420p")  # fmt: skip
        Image.fromarray(frames[0]).save(OUT / f"{obj}.jpg", quality=85)
        print(f"{obj}: {(OUT / f'{obj}.mp4').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
