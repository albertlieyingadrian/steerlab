"""Cheap motion heuristics (no model) for every sampled video and every doctored clip, plus the frames the `motion`
judge config sees.

    python 3_heuristics.py

Up to 48 evenly spaced frames are read at 256 px on the long side. From Farneback optical flow between neighbours:

- motion: mean flow magnitude (pixels per step at 256 px);
- jerk: mean absolute change of the per-step mean magnitude, divided by the mean magnitude (motion that starts and
  stops abruptly);
- diff_spike: largest mean absolute frame difference over the median one (a sudden change of picture);
- scale_jump: largest |log scale| of a similarity transform fitted (RANSAC) to corners tracked between neighbours,
  minus the median (something suddenly grows or shrinks, or the camera jumps);
- divergence: mean absolute divergence of the flow field (local stretching and squashing: melting or morphing);
- warp_error: mean photometric error after warping each frame onto the next with the flow (content that appears,
  vanishes or changes rather than moves).

Each is read so that higher means more likely to break physics. Motion-peak frames: the 8 steps with the largest
change in mean flow, mapped back to original frame indices, kept apart from the 8 evenly spaced ones.

Writes results/heuristics.csv (sampled videos), results/heuristics_traps.csv and results/motion_frames.json.
"""

from __future__ import annotations

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

FEATURES = ["motion", "jerk", "diff_spike", "scale_jump", "divergence", "warp_error"]


def read_frames(path: Path, n: int, side: int) -> tuple[list[np.ndarray], np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    idx = np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int))
    out = []
    for i in idx:
        f = frames[i]
        s = side / max(f.shape[:2])
        out.append(cv2.resize(f, (round(f.shape[1] * s), round(f.shape[0] * s)), interpolation=cv2.INTER_AREA))
    return out, idx


def even_indices(total: int, k: int = 8) -> list[int]:
    return [round(float(x)) for x in np.linspace(0, total - 1, k)]


def features(path: Path) -> tuple[dict, list[int]]:
    frames, idx = read_frames(path, 48, 256)
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    mags, diffs, scales, divs, warps = [], [], [], [], []
    h, w = grey[0].shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    for a, b in itertools.pairwise(grey):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(flow, axis=2).mean()))
        diffs.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))
        du = np.gradient(flow[..., 0], axis=1)
        dv = np.gradient(flow[..., 1], axis=0)
        divs.append(float(np.abs(du + dv).mean()))
        warped = cv2.remap(b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
        warps.append(float(np.abs(warped.astype(np.float32) - a.astype(np.float32)).mean()))
        pts = cv2.goodFeaturesToTrack(a, 200, 0.01, 5)
        s = 0.0
        if pts is not None and len(pts) >= 6:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
            good = st.ravel() == 1
            if good.sum() >= 6:
                m, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC)
                if m is not None:
                    s = float(np.log(np.sqrt(abs(np.linalg.det(m[:, :2]))) + 1e-9))
        scales.append(s)
    mags, diffs, scales = np.array(mags), np.array(diffs), np.array(scales)
    mean_mag = mags.mean() + 1e-6
    feat = {
        "motion": float(mags.mean()),
        "jerk": float(np.abs(np.diff(mags)).mean() / mean_mag) if len(mags) > 1 else 0.0,
        "diff_spike": float(diffs.max() / (np.median(diffs) + 1e-6)),
        "scale_jump": float(np.abs(scales - np.median(scales)).max()),
        "divergence": float(np.mean(divs)),
        "warp_error": float(np.mean(warps)),
    }
    change = np.abs(np.diff(mags, prepend=mags[0]))
    even = set(even_indices(int(idx[-1]) + 1))
    peaks: list[int] = []
    for k in np.argsort(-change):
        fi = int(idx[k + 1])
        if fi not in even and all(abs(fi - p) > 1 for p in peaks):
            peaks.append(fi)
        if len(peaks) == 8:
            break
    return feat, sorted(peaks)


def run(job: tuple[str, str]) -> tuple[str, dict, list[int]]:
    name, path = job
    feat, peaks = features(Path(path))
    return name, feat, peaks


def main() -> None:
    items = pd.read_csv(RESULTS / "items.csv")
    traps = pd.read_csv(RESULTS / "traps.csv")
    jobs = [(v, str(DATA / "videos" / f"{v}.mp4")) for v in items.video]
    jobs += [(t, str(DATA / "traps" / f"{t}.mp4")) for t in traps.trap]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(run, jobs, chunksize=4))
    res = {name: feat for name, feat, _ in out}
    peaks = {name: p for name, _, p in out}
    pd.DataFrame([{"video": v, **res[v]} for v in items.video]).to_csv(RESULTS / "heuristics.csv", index=False)
    pd.DataFrame([{"trap": t, **res[t]} for t in traps.trap]).to_csv(RESULTS / "heuristics_traps.csv", index=False)
    (RESULTS / "motion_frames.json").write_text(json.dumps(peaks))
    print(pd.DataFrame(res).T[FEATURES].describe().round(3))


if __name__ == "__main__":
    main()
