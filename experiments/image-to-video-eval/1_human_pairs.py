"""Sample human-voted pairs from Rapidata's image-to-video benchmarks, and prepare every video the same way.

    uv run --no-project --with pandas --with pyarrow --with pillow --with numpy python 1_human_pairs.py

Two datasets on Hugging Face, both CC-BY 4.0, both the vote data behind benchmark.ai leaderboards:

- Rapidata/image-to-video-realism-human-feedback-benchmark: 2,981 comparisons of two clips made from the same still and
  prompt (25 stills, 18 models plus the real footage the still was taken from), each asked two questions without the
  prompt: "Which video looks more like it was shot by a real camera?" (photorealism) and "Which video has more glitches
  and is more likely to be AI generated?" (integrity; the side with fewer votes is the better one). 16 to 257 votes per
  pair and question.
- Rapidata/image-to-video-camera-movement-human-feedback-benchmark: 82,153 comparisons of two clips asked to make the
  same camera move from the same still; voters watch a reference animation of the move and answer "Which video better
  matches the movement of the top video?". 5 to 21 votes per pair. Only one-move prompts whose move shows up as one of
  VBench's six image-motion classes are used (rolls are left out): 12 of the 14 single moves.

Samples (seed 0; pairs with exactly tied votes are dropped first): 800 realism pairs, stratified by model pair, each
used for both realism questions; 800 camera pairs, stratified by move in proportion to its count (45 to 72 a move).

Writes results/human_pairs.csv (one row per pair and question; votes oriented so votes_a counts votes saying A is the
better clip; voter details are dropped and never saved), results/human_all_models.csv (each model's Bradley-Terry score
from every comparison, per question) and results/videos.csv. Under $I2V_DIR it keeps the original clips (raw/), the
clips re-encoded at 480 px on the short side with frame rate and length unchanged (norm/), one sheet of 8 evenly
spaced frames per clip for the judges (sheets/), the input stills (stills/) and the camera reference animations.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS, bradley_terry

HF = "https://huggingface.co/datasets/Rapidata"
REALISM = "image-to-video-realism-human-feedback-benchmark"
CAMERA = "image-to-video-camera-movement-human-feedback-benchmark"
N_PAIRS, SEED = 800, 0
# Rapidata move tag -> VBench++'s camera-motion class (how the picture moves on screen). Sliding the camera left and
# panning it left both move the picture to the right; rising and tilting up both move it down; moving forward and
# zooming in both spread it outward.
MOVES = {
    "Turn left": "pan_left", "Move left": "pan_left", "Turn right": "pan_right", "Move right": "pan_right",
    "Tilt up": "tilt_up", "Move up": "tilt_up", "Tilt down": "tilt_down", "Move down": "tilt_down",
    "Zoom in": "zoom_in", "Move forward": "zoom_in", "Zoom out": "zoom_out", "Move back": "zoom_out",
}  # fmt: skip
SHEET_W = 240  # width of each frame on a judge sheet; 8 frames as 4 columns x 2 rows


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r:
            tmp.write_bytes(r.read())
        tmp.rename(path)
    return path


def shards(name: str) -> list[Path]:
    api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/Rapidata/{name}"))
    files = sorted(s["rfilename"] for s in api["siblings"] if s["rfilename"].endswith(".parquet"))
    folder = DATA / "rapidata" / ("realism" if name == REALISM else "camera")
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(lambda f: fetch(f"{HF}/{name}/resolve/main/{f}", folder / Path(f).name), files))


def vid(url: str) -> str:
    return url.rsplit("/", 1)[1].split(".")[0]


def count(js: str) -> tuple[int, int]:
    v = json.loads(js)
    a = sum(x["votedFor"] == "A" for x in v)
    return a, len(v) - a


def load_realism() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(p) for p in shards(REALISM)], ignore_index=True)
    stills = DATA / "stills"
    stills.mkdir(parents=True, exist_ok=True)

    def save_still(asset) -> str:
        name = hashlib.md5(asset["bytes"]).hexdigest()[:12] + ".jpg"
        if not (stills / name).exists():
            from io import BytesIO

            from PIL import Image

            Image.open(BytesIO(asset["bytes"])).convert("RGB").save(stills / name, quality=95)
        return name

    d["still"] = d.prompt_asset.map(save_still)
    rows = []
    for q, col in (("photorealism", "photorealism"), ("integrity", "visual_integrity")):
        raw = d[f"detailed_results_{col}"].map(count)
        part = d[["still", "prompt", "video1", "video2", "model1", "model2"]].copy()
        part["question"] = q
        part["raw_a"], part["raw_b"] = raw.str[0], raw.str[1]
        # Integrity asks which clip has MORE glitches, so a vote for B is a vote that A is the better clip.
        better_a = part.raw_a if q == "photorealism" else part.raw_b
        better_b = part.raw_b if q == "photorealism" else part.raw_a
        part["votes_a"], part["votes_b"] = better_a, better_b
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["movement"] = ""
    return out


def load_camera() -> pd.DataFrame:
    cols = ["prompt", "input_image", "reference_gif", "prompt_tags", "video1", "video2", "model1", "model2",
            "detailed_results_movement"]  # fmt: skip
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in shards(CAMERA)], ignore_index=True)
    tags = d.prompt_tags.map(list)
    one = tags.map(lambda t: "1 motion" in t)
    move = tags.map(lambda t: next((x for x in t if x in MOVES), ""))
    d = d[one & (move != "")].copy()
    d["move"] = move[d.index]
    d["movement"] = d.move.map(MOVES)
    raw = d.detailed_results_movement.map(count)
    d["raw_a"], d["raw_b"] = raw.str[0], raw.str[1]
    d["votes_a"], d["votes_b"] = d.raw_a, d.raw_b
    d["question"] = "camera"
    d["still"] = d.input_image.map(lambda u: "cam_" + u.rsplit("/", 1)[1])
    d["reference"] = d.reference_gif.map(vid)
    return d.drop(columns=["prompt_tags", "detailed_results_movement"]).reset_index(drop=True)


def stratified(d: pd.DataFrame, key: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """n rows, each stratum getting its share of n (largest remainders), drawn at random within strata."""
    sizes = key.value_counts()
    want = sizes / sizes.sum() * n
    take = np.floor(want).astype(int)
    take[(want - take).sort_values(ascending=False).index[: n - take.sum()]] += 1
    rng = np.random.default_rng(seed)
    picks = [rng.choice(np.flatnonzero(key.to_numpy() == k), take[k], replace=False) for k in sizes.index]
    return d.iloc[np.sort(np.concatenate(picks))]


def prepare(url: str) -> dict:
    """Download a clip, re-encode it at 480 px on the short side (frame rate and length unchanged), make its sheet."""
    v = vid(url)
    raw = fetch(url, DATA / "raw" / f"{v}.mp4")
    norm, sheet = DATA / "norm" / f"{v}.mp4", DATA / "sheets" / f"{v}.jpg"
    norm.parent.mkdir(exist_ok=True)
    sheet.parent.mkdir(exist_ok=True)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(raw)],
        capture_output=True, text=True, check=True).stdout)  # fmt: skip
    s = next(x for x in probe["streams"] if x["codec_type"] == "video")
    w, h = int(s["width"]), int(s["height"])
    if not norm.exists():
        encode(raw, norm)
    if not sheet.exists():
        frames = read_frames(norm)
        make_sheet([frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)]).save(sheet, quality=90)
    num, den = s["r_frame_rate"].split("/")
    return {"video": v, "url": url, "width": w, "height": h, "fps": float(num) / float(den),
            "frames": int(s.get("nb_frames") or 0), "seconds": float(probe["format"]["duration"])}  # fmt: skip


def encode(src: Path, dst: Path, extra: list[str] | None = None) -> None:
    """Re-encode at 480 px on the short side; frame rate, length and frames unchanged."""
    probe = json.loads(subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(src)],
                                      capture_output=True, text=True, check=True).stdout)  # fmt: skip
    s = next(x for x in probe["streams"] if x["codec_type"] == "video")
    scale = "scale=-2:480" if int(s["width"]) >= int(s["height"]) else "scale=480:-2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *(extra or []), "-vf", scale, "-c:v", "libx264",
                    "-crf", "17", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)  # fmt: skip


def read_frames(path: Path) -> list:
    """Every frame of a video as PIL images (ffmpeg to raw RGB)."""
    from PIL import Image

    probe = json.loads(subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
                                      capture_output=True, text=True, check=True).stdout)  # fmt: skip
    s = next(x for x in probe["streams"] if x["codec_type"] == "video")
    w, h = int(s["width"]), int(s["height"])
    buf = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout  # fmt: skip
    arr = np.frombuffer(buf, np.uint8).reshape(-1, h, w, 3)
    return [Image.fromarray(a) for a in arr]


def make_sheet(frames: list):
    """8 frames in time order, left to right then top to bottom, each SHEET_W px wide."""
    from PIL import Image

    w = SHEET_W
    h = round(frames[0].height * w / frames[0].width)
    sheet = Image.new("RGB", (4 * w + 3 * 4, 2 * h + 4), "white")
    for k, f in enumerate(frames):
        sheet.paste(f.convert("RGB").resize((w, h), Image.LANCZOS), ((k % 4) * (w + 4), (k // 4) * (h + 4)))
    return sheet


def reference_sheet(url: str) -> None:
    from PIL import Image, ImageSequence

    out = DATA / "sheets" / f"ref_{vid(url)}.jpg"
    if out.exists():
        return
    gif = Image.open(fetch(url, DATA / "references" / f"{vid(url)}.gif"))
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(gif)]
    make_sheet([frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)]).save(out, quality=90)


def main() -> None:
    realism, camera = load_realism(), load_camera()
    rows = []
    for q, sub in [*realism.groupby("question"), ("camera", camera)]:
        battles = [(r.model1, r.model2, r.votes_a, r.votes_b) for r in sub.itertuples()]
        scores, ci = bradley_terry(battles, boot=200)
        rows += [{"question": q, "model": m, "bt": scores[m], "lo": ci[m][0], "hi": ci[m][1]} for m in scores]
    pd.DataFrame(rows).to_csv(RESULTS / "human_all_models.csv", index=False, float_format="%.1f")

    # Realism: one draw of pairs, used for both questions; a pair with a tie on either question is left out.
    photo = realism[realism.question == "photorealism"].reset_index(drop=True)
    integ = realism[realism.question == "integrity"].reset_index(drop=True)
    ok = (photo.votes_a != photo.votes_b) & (integ.votes_a != integ.votes_b)
    model_pair = photo[ok].apply(lambda r: "|".join(sorted([r.model1, r.model2])), axis=1)
    picked = stratified(photo[ok], model_pair, N_PAIRS, SEED).index
    sample_r = pd.concat([photo.loc[picked], integ.loc[picked]])
    sample_r["pair"] = [f"{q[0]}{i:03d}" for q, i in zip(sample_r.question, list(range(len(picked))) * 2, strict=True)]
    cam = camera[camera.votes_a != camera.votes_b]
    sample_c = stratified(cam, cam.move, N_PAIRS, SEED).copy()
    sample_c["pair"] = [f"c{i:03d}" for i in range(len(sample_c))]
    sample = pd.concat([sample_r, sample_c], ignore_index=True)

    urls = sorted(set(sample.video1) | set(sample.video2))
    with ThreadPoolExecutor(4) as pool:
        meta = list(pool.map(prepare, urls))
        stills = sample_c.drop_duplicates("still")
        list(pool.map(lambda r: fetch(r.input_image, DATA / "stills" / r.still), stills.itertuples()))
        list(pool.map(reference_sheet, sorted(set(sample_c.reference_gif))))
    pd.DataFrame(meta).to_csv(RESULTS / "videos.csv", index=False, float_format="%.3f")

    sample["video_a"], sample["video_b"] = sample.video1.map(vid), sample.video2.map(vid)
    keep = ["pair", "question", "still", "prompt", "move", "movement", "reference", "model1", "model2", "video_a",
            "video_b", "votes_a", "votes_b", "raw_a", "raw_b"]  # fmt: skip
    sample.reindex(columns=keep).to_csv(RESULTS / "human_pairs.csv", index=False)
    share = sample[["votes_a", "votes_b"]].max(axis=1) / sample[["votes_a", "votes_b"]].sum(axis=1)
    print(f"{len(sample)} pair-questions, {len(urls)} clips")
    print(sample.assign(clear=share >= 0.75).groupby("question").agg(n=("pair", "size"), clear=("clear", "sum")))
    print(sample_c.move.value_counts().to_dict())


if __name__ == "__main__":
    main()
