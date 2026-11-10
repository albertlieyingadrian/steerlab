"""Clip gallery for the note: per still, one small looping video with the still, the real footage and our four models.

    uv run --no-project --with pandas --with pillow --with numpy python make_gallery.py

Stills: the first 4 realism stills and the first 2 camera stills in job order (results/gen_jobs.csv), not picked by
how the clips look. Each video is a 3 x 2 grid of 320 x 180 tiles, 4 seconds at 12 frames a second: the input still,
the real footage it was taken from (realism stills; the camera stills have none, so that tile names the asked move),
then each model's clip. Every clip is stretched or squeezed in time to fill the 4 seconds, so clips of different
lengths play side by side; each label gives the clip's real length. Writes report/figures/gallery/<id>.mp4 and a
poster frame <id>.jpg for each.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

pairs_mod = importlib.import_module("1_human_pairs")
OUT = HERE / "report" / "figures" / "gallery"
MODELS = {"wan22_ti2v_5b": "Wan 2.2 TI2V-5B", "ltx_video_097_distilled": "LTX-Video 0.9.7",
          "cogvideox_5b_i2v": "CogVideoX-5B-I2V", "svd_xt": "SVD-XT"}  # fmt: skip
TILE, FPS, SECONDS = (320, 180), 12, 4
PAPER = (246, 243, 236)


def font(size: int):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def letterbox(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    im.thumbnail(TILE, Image.LANCZOS)
    tile = Image.new("RGB", TILE, PAPER)
    tile.paste(im, ((TILE[0] - im.width) // 2, (TILE[1] - im.height) // 2))
    return tile


def resampled(path: Path) -> tuple[list[Image.Image], float]:
    frames = pairs_mod.read_frames(path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)  # fmt: skip
    num, den = probe.stdout.strip().split("/")
    idx = np.linspace(0, len(frames) - 1, FPS * SECONDS).round().astype(int)
    return [letterbox(frames[i]) for i in idx], len(frames) * float(den) / float(num)


def main() -> None:
    jobs = pd.read_csv(RESULTS / "gen_jobs.csv", keep_default_na=False)
    pairs = pd.read_csv(RESULTS / "human_pairs.csv", keep_default_na=False)
    cols = ["still", "v"]
    real_a = pairs[pairs.model1 == "ground_truth_baseline"][["still", "video_a"]].set_axis(cols, axis=1)
    real_b = pairs[pairs.model2 == "ground_truth_baseline"][["still", "video_b"]].set_axis(cols, axis=1)
    gt = pd.concat([real_a, real_b])
    gt = gt.drop_duplicates("still").set_index("still").v
    chosen = pd.concat([jobs[jobs.id.str.startswith("r")].head(4), jobs[jobs.id.str.startswith("c")].head(2)])
    OUT.mkdir(parents=True, exist_ok=True)
    f = font(13)
    blank = [Image.new("RGB", TILE, PAPER)] * (FPS * SECONDS)
    for job in chosen.itertuples():
        tiles = [("Input still", [letterbox(Image.open(DATA / "stills" / job.still))] * (FPS * SECONDS))]
        if job.still in gt.index:
            clip, secs = resampled(DATA / "norm" / f"{gt[job.still]}.mp4")
            tiles.append((f"Real footage ({secs:.1f} s)", clip))
        else:
            tiles.append((f"Asked: {job.move.lower()}", blank))
        for model, label in MODELS.items():
            path = DATA / "norm" / f"gen_{model}_{job.id}.mp4"
            if path.exists():
                clip, secs = resampled(path)
                tiles.append((f"{label} ({secs:.1f} s)", clip))
            else:
                tiles.append((f"{label}: not run", blank))
        frames = []
        for k in range(FPS * SECONDS):
            canvas = Image.new("RGB", (3 * TILE[0], 2 * (TILE[1] + 22)), "white")
            draw = ImageDraw.Draw(canvas)
            for n, (label, clip) in enumerate(tiles):
                x, y = (n % 3) * TILE[0], (n // 3) * (TILE[1] + 22)
                draw.text((x + 6, y + 4), label, fill=(60, 60, 60), font=f)
                canvas.paste(clip[k], (x, y + 22))
            frames.append(canvas)
        out = OUT / f"{job.id}.mp4"
        w, h = frames[0].size
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
               "-i", "-", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(out)]  # fmt: skip
        subprocess.run(cmd, input=b"".join(np.asarray(fr).tobytes() for fr in frames), check=True)
        frames[FPS].save(OUT / f"{job.id}.jpg", quality=85)
        print(f"{job.id}: {out.stat().st_size / 1e3:.0f} kB; {job.prompt[:60]}")


if __name__ == "__main__":
    main()
