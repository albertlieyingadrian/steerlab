"""Make four broken copies of 25 real generations: the traps a good scorer must not prefer to the real clip.

    uv run --no-project --with pandas --with pillow --with numpy python 2_traps.py

Clips: one generated clip per realism still (25; two per still were planned and cut for the budget before any score
was read), drawn at random (seed 0) from the clips in the sampled human pairs, never the real footage. Each copy
keeps the clip's size, frame rate and number of frames:

- still: the input still, cropped to the clip's shape and repeated for every frame (nothing moves);
- reversed: the frames played backwards (the motion runs the wrong way in time);
- shuffled: the frames in a random order (seed 0);
- blurred: every frame blurred with a Gaussian of radius 6 px at 480 px height (heavily blurred).

Writes results/traps.csv (one row per clip and trap) and, under $I2V_DIR, norm/<clip>_<trap>.mp4 and a judge sheet each.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

pairs_mod = importlib.import_module("1_human_pairs")
TRAPS = ["still", "reversed", "shuffled", "blurred"]
PER_STILL, SEED = 1, 0


def fit(still, w: int, h: int):
    from PIL import Image

    sw, sh = still.size
    if sw / sh > w / h:
        nw = round(sh * w / h)
        still = still.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = round(sw * h / w)
        still = still.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    return still.resize((w, h), Image.LANCZOS)


def write(frames: list, fps: float, path: Path) -> None:
    w, h = frames[0].size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
           "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(path)]  # fmt: skip
    data = b"".join(np.asarray(f.convert("RGB"), np.uint8).tobytes() for f in frames)
    subprocess.run(cmd, input=data, check=True)


def main() -> None:
    from PIL import Image, ImageFilter

    pairs = pd.read_csv(RESULTS / "human_pairs.csv", keep_default_na=False)
    real = pairs[pairs.question == "photorealism"]
    cols = ["video", "model", "still", "prompt"]
    clips = pd.concat([real[["video_a", "model1", "still", "prompt"]].set_axis(cols, axis=1),
                       real[["video_b", "model2", "still", "prompt"]].set_axis(cols, axis=1)])  # fmt: skip
    clips = clips.drop_duplicates("video")
    clips = clips[clips.model != "ground_truth_baseline"].sort_values("video")
    chosen = clips.groupby("still", group_keys=False).sample(PER_STILL, random_state=SEED)
    videos = pd.read_csv(RESULTS / "videos.csv").set_index("video")
    rng = np.random.default_rng(SEED)
    rows = []
    for r in chosen.sort_values("video").itertuples():
        frames = pairs_mod.read_frames(DATA / "norm" / f"{r.video}.mp4")
        fps = float(videos.loc[r.video, "fps"])
        still = fit(Image.open(DATA / "stills" / r.still).convert("RGB"), *frames[0].size)
        made = {
            "still": [still] * len(frames),
            "reversed": frames[::-1],
            "shuffled": [frames[i] for i in rng.permutation(len(frames))],
            "blurred": [f.filter(ImageFilter.GaussianBlur(6)) for f in frames],
        }
        for trap, fr in made.items():
            name = f"{r.video}_{trap}"
            out = DATA / "norm" / f"{name}.mp4"
            if not out.exists():
                write(fr, fps, out)
            sheet = DATA / "sheets" / f"{name}.jpg"
            if not sheet.exists():
                pick = np.linspace(0, len(fr) - 1, 8).round().astype(int)
                pairs_mod.make_sheet([fr[i] for i in pick]).save(sheet, quality=90)
            rows.append({"video": r.video, "model": r.model, "still": r.still, "prompt": r.prompt, "trap": trap,
                         "trap_video": name, "frames": len(fr), "fps": fps})  # fmt: skip
    pd.DataFrame(rows).to_csv(RESULTS / "traps.csv", index=False)
    print(f"{len(chosen)} clips x {len(TRAPS)} traps; models: {chosen.model.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
