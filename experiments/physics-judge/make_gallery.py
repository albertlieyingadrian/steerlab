"""The note's clip gallery: four clips re-encoded small, each with a poster frame.

    python experiments/physics-judge/make_gallery.py

g1, g2: two test videos (VideoPhy-2, MIT) where the chosen judge (Opus 5.5, prompt's rules) said "breaks physics":
one the annotators agreed with (PC 2) and AutoEval did not (predicted PC 4), one the annotators rated 5.
g3, g4: two doctored known-answer clips (a hockey fall played backwards; a bouncy-castle clip whose centre is
enlarged from the middle frame). Writes report/figures/gallery/<id>.mp4 (480 px wide, H.264, no sound) and <id>.jpg.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA

OUT = HERE / "report" / "figures" / "gallery"
CLIPS = {
    "g1": DATA / "videos" / "b040634f3135.mp4",
    "g2": DATA / "videos" / "ce07e1234917.mp4",
    "g3": DATA / "traps" / "3617d795cf85_reversed.mp4",
    "g4": DATA / "traps" / "1f5bab162225_resized.mp4",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, src in CLIPS.items():
        mp4, jpg = OUT / f"{key}.mp4", OUT / f"{key}.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-an", "-vf", "scale=480:-2",
                        "-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                       check=True)  # fmt: skip
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf", "select=eq(n\\,8)",
                        "-frames:v", "1", "-q:v", "4", str(jpg)], check=True)  # fmt: skip
        print(key, src.name, f"{mp4.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
