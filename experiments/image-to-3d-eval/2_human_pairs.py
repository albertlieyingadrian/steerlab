"""Sample human-voted pairs from Rapidata's image-to-3D benchmark and get their frames and input images.

    uv run --no-project --with pandas --with pyarrow --with imageio[ffmpeg] --with av --with pillow --with scipy \
        python 2_human_pairs.py

The dataset (Rapidata/image-to-3d-single-object-human-feedback-benchmark on Hugging Face, CC-BY 4.0) is the vote data
behind benchmark.ai's image-to-3D leaderboard: 49,761 comparisons of two turntable videos of 13 models' outputs for
the same input image, each with (usually) 5 votes. Half ask about geometry, on grey untextured renders ("Which video
reflects the shape of the object in the photo?"), half about appearance, on the delivered renders ("Which video looks
more like the object in the photo?").

Writes results/human_pairs.csv (the sample, with vote counts but no voter details) and results/human_all_models.csv
(each model's Bradley-Terry score from every comparison, per question), and saves per video a sheet of 4 frames
(0, 90, 180 and 270 degrees of the turntable, 256 px each) plus 8 frames for CLIP under $I3D_DIR/rapidata/.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS, bradley_terry

REPO = "https://huggingface.co/datasets/Rapidata/image-to-3d-single-object-human-feedback-benchmark/resolve/main"
API = "https://huggingface.co/api/datasets/Rapidata/image-to-3d-single-object-human-feedback-benchmark"
PER_QUESTION = 400
SEED = 0


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r:
            tmp.write_bytes(r.read())
        tmp.rename(path)
    return path


def load_all() -> pd.DataFrame:
    files = [s["rfilename"] for s in json.load(urllib.request.urlopen(API))["siblings"]]
    shards = sorted(f for f in files if f.endswith(".parquet"))
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(lambda f: fetch(f"{REPO}/{f}", DATA / "rapidata" / "parquet" / Path(f).name), shards))
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    d["question"] = np.where(d.detailed_results_geometry.notna(), "geometry", "appearance")

    def votes(r) -> tuple[int, int]:
        v = json.loads(r.detailed_results_geometry if r.question == "geometry" else r.detailed_results_appearance)
        a = sum(x["votedFor"] == "A" for x in v)
        return a, len(v) - a

    d[["votes_a", "votes_b"]] = d.apply(votes, axis=1, result_type="expand")
    geo = d.question == "geometry"
    d["weight_a"] = np.where(geo, d.weighted_results_video1_geometry, d.weighted_results_video1_appearance)
    d["weight_b"] = np.where(geo, d.weighted_results_video2_geometry, d.weighted_results_video2_appearance)
    d["tags"] = d.prompt_tags.map(lambda t: "|".join(t))
    # Voter details (country, age and so on) are dropped here and never saved.
    keep = ["question", "input_image", "video1", "video2", "model1", "model2", "votes_a", "votes_b", "weight_a",
            "weight_b", "tags"]  # fmt: skip
    return d[keep]


def prepare_video(url: str) -> str:
    import imageio.v3 as iio
    from PIL import Image

    vid = url.rsplit("/", 1)[1].split(".")[0]
    out = DATA / "rapidata" / "frames" / vid
    if (out / "sheet.png").exists():
        return vid
    path = fetch(url, DATA / "rapidata" / "videos" / f"{vid}.mp4")
    all_frames = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(all_frames), 8, endpoint=False).astype(int)
    small = [Image.fromarray(all_frames[i][..., :3]).resize((256, 256), Image.LANCZOS) for i in idx]
    out.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(small):
        im.save(out / f"{i}.png")
    sheet = Image.new("RGB", (1024, 256))
    for k, i in enumerate((0, 2, 4, 6)):
        sheet.paste(small[i], (256 * k, 0))
    sheet.save(out / "sheet.png")
    return vid


def main() -> None:
    d = load_all()
    print(f"{len(d):,} comparisons, {d.votes_a.add(d.votes_b).sum():,} votes")
    rows = []
    for q in ("geometry", "appearance"):
        sub = d[d.question == q]
        wins = [(r.model1, r.model2, r.weight_a, r.weight_b) for r in sub.itertuples()]
        scores, ci = bradley_terry(wins, boot=200)
        for m in scores:
            rows.append({"question": q, "model": m, "bt": scores[m], "lo": ci[m][0], "hi": ci[m][1]})
    pd.DataFrame(rows).to_csv(RESULTS / "human_all_models.csv", index=False, float_format="%.1f")

    five = d[(d.votes_a + d.votes_b) == 5]
    sample = pd.concat(
        [five[five.question == q].sample(PER_QUESTION, random_state=SEED) for q in ("geometry", "appearance")]
    ).reset_index(drop=True)
    sample.insert(0, "pair", [f"{q[0]}{i:03d}" for i, q in enumerate(sample.question)])
    urls = sorted(set(sample.video1) | set(sample.video2))
    with ThreadPoolExecutor(16) as pool:
        vids = dict(zip(urls, pool.map(prepare_video, urls), strict=True))
        list(pool.map(lambda u: fetch(u, DATA / "rapidata" / "inputs" / u.rsplit("/", 1)[1]), set(sample.input_image)))
    sample["video_a"], sample["video_b"] = sample.video1.map(vids), sample.video2.map(vids)
    sample["input"] = sample.input_image.map(lambda u: u.rsplit("/", 1)[1])
    sample = sample.drop(columns=["video1", "video2", "input_image"])
    sample.to_csv(RESULTS / "human_pairs.csv", index=False)
    print(f"sampled {len(sample)} pairs, {len(urls)} videos; votes for the majority side:")
    print(pd.crosstab(sample.question, sample[["votes_a", "votes_b"]].max(axis=1)))


if __name__ == "__main__":
    main()
