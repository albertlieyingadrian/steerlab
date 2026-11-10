"""Reference-free embedding scores for every sampled Rapidata video: how much its frames look like the input image.

    uv run --no-project --with torch --with open_clip_torch --with transformers --with pandas --with pillow \
        python 3_cheap_metrics.py

CLIP similarity (ViT-L/14, OpenAI weights, the score most papers report): the cosine between the input image's
embedding and each of 8 frames' (every 45 degrees of the turntable), averaged. DINOv2 similarity (dinov2-base, the
CLS token) is the same with a self-supervised image encoder, which cares more about shape and layout than words.
The same CLIP score for the GSO objects comes from eval3d.py on Modal. Writes results/human_video_scores.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS


def main() -> None:
    import open_clip
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip = clip.to(device).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    @torch.no_grad()
    def embed(images: list) -> tuple[np.ndarray, np.ndarray]:
        c = clip.encode_image(torch.stack([pre(im) for im in images]).to(device)).float()
        d = dino(**dproc(images=images, return_tensors="pt").to(device)).last_hidden_state[:, 0].float()
        norm = torch.nn.functional.normalize
        return norm(c, dim=-1).cpu().numpy(), norm(d, dim=-1).cpu().numpy()

    pairs = pd.read_csv(RESULTS / "human_pairs.csv")
    videos = pd.concat(
        [pairs[["video_a", "input"]].set_axis(["video", "input"], axis=1),
         pairs[["video_b", "input"]].set_axis(["video", "input"], axis=1)]
    ).drop_duplicates()  # fmt: skip
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for k, r in enumerate(videos.itertuples()):
        if r.input not in cache:
            c, d = embed([Image.open(DATA / "rapidata" / "inputs" / r.input).convert("RGB")])
            cache[r.input] = (c[0], d[0])
        frames = [Image.open(DATA / "rapidata" / "frames" / r.video / f"{i}.png").convert("RGB") for i in range(8)]
        c, d = embed(frames)
        rows.append({"video": r.video, "input": r.input, "clip": float((c @ cache[r.input][0]).mean()),
                     "dino": float((d @ cache[r.input][1]).mean())})  # fmt: skip
        if k % 200 == 0:
            print(f"{k}/{len(videos)}")
    pd.DataFrame(rows).to_csv(RESULTS / "human_video_scores.csv", index=False, float_format="%.5f")
    print(f"wrote {len(rows)} videos")


if __name__ == "__main__":
    main()
