"""Known-answer checks: three doctored copies of each of the 25 clean source clips (results/trap_sources.csv).

    python 2_traps.py

- reversed: the frames in reverse order (things fall up, poured liquid climbs back);
- frozen: the clip plays to its middle frame and then holds it for the rest (a thrown or falling object hangs still);
- resized: from the middle frame on, the central half of the picture is enlarged by 35% in place, so whatever sits in
  the middle of the scene suddenly grows.

Every copy keeps the source's frame count, frame rate and size and is encoded as H.264 (yuv420p) with ffmpeg, as is
the untouched source (`clean`), so that encoding is not what separates them. Writes $PHYS_DIR/traps/<video>_<kind>.mp4
and results/traps.csv.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

TRAPS = DATA / "traps"
TRAPS.mkdir(parents=True, exist_ok=True)
KINDS = ["clean", "reversed", "frozen", "resized"]


def read(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write(frames: list[np.ndarray], fps: float, out: Path) -> None:
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(frames):
            cv2.imwrite(f"{tmp}/{k:05d}.png", f[:h2, :w2])
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.4f}", "-i", f"{tmp}/%05d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
            check=True,
        )  # fmt: skip


def enlarge_centre(f: np.ndarray, scale: float = 1.35) -> np.ndarray:
    h, w = f.shape[:2]
    y0, y1, x0, x1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
    big = cv2.resize(f[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    bh, bw = big.shape[:2]
    top, left = max((y0 + y1) // 2 - bh // 2, 0), max((x0 + x1) // 2 - bw // 2, 0)
    out = f.copy()
    ph, pw = min(bh, h - top), min(bw, w - left)
    out[top : top + ph, left : left + pw] = big[:ph, :pw]
    return out


def doctor(frames: list[np.ndarray], kind: str) -> list[np.ndarray]:
    mid = len(frames) // 2
    if kind == "clean":
        return frames
    if kind == "reversed":
        return frames[::-1]
    if kind == "frozen":
        return frames[:mid] + [frames[mid]] * (len(frames) - mid)
    if kind == "resized":
        return frames[:mid] + [enlarge_centre(f) for f in frames[mid:]]
    raise ValueError(kind)


def main() -> None:
    src = pd.read_csv(RESULTS / "trap_sources.csv")
    rows = []
    for r in src[src.downloaded].itertuples():
        frames, fps = read(DATA / "videos" / f"{r.video}.mp4")
        for kind in KINDS:
            out = TRAPS / f"{r.video}_{kind}.mp4"
            if not out.exists():
                write(doctor(frames, kind), fps, out)
            rows.append({"trap": f"{r.video}_{kind}", "video": r.video, "kind": kind, "caption": r.caption,
                         "frames": len(frames), "fps": fps})  # fmt: skip
    pd.DataFrame(rows).to_csv(RESULTS / "traps.csv", index=False)
    print(len(rows), "trap clips in", TRAPS.name)


if __name__ == "__main__":
    main()
