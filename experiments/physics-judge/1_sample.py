"""The pre-registered sample: 600 videos from VideoPhy-2's human-labelled test split, their rules, and 25 clean source
clips for the known-answer checks.

    python 1_sample.py

Sampling (fixed in PREREG.md before any judge ran): 100 videos from each of the six models with at least 394 test
videos (CogVideoX-5B, Cosmos, HunyuanVideo, Luma Ray2, VideoCrafter2, Wan2.1; Sora has only 60 and is left out), seed
20261114. Prompts, not videos, are split into dev (about a third, for prompt writing and thresholds) and test, so no
scene is seen on both sides. Rules are the benchmark's own (generated with the prompt, checked by the annotators);
"cannot be determined" rules are dropped, and the annotators' extra written violations are kept apart (they are all
violations by construction).

Known-answer sources: 25 test videos outside the sample, rated 5 for physical commonsense and with no rule marked
violated, whose prompt involves something falling, thrown, poured or bouncing, so that a reversed or frozen copy
plainly breaks gravity or momentum.

Writes results/items.csv, results/rules.csv, results/trap_sources.csv; videos go to $PHYS_DIR/videos.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AUTO_CSV, DATA, LAWS, RESULTS, TEST_CSV

SEED = 20261114
MODELS = ["cogvideo", "cosmos", "hunyuan", "ray2", "videocrafter", "wan"]
PER_MODEL = 100
MOTION = r"\b(?:falls?|falling|drops?|dropped|pours?|poured|thrown|throws?|toss(?:es|ed)?|bounc\w*|splash\w*|rolls?)\b"
VIDEOS = DATA / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)


def fetch_csv(url: str, name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url).read())
    return pd.read_csv(path)


def lit(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else None
    except (ValueError, SyntaxError):
        return None


def vid_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def download(url: str) -> tuple[str, bool]:
    path = VIDEOS / f"{vid_id(url)}.mp4"
    if path.exists() and path.stat().st_size > 1000:
        return url, True
    for _ in range(3):
        try:
            # Percent-encode only the non-ASCII characters (curly quotes, dashes) some file names contain.
            safe = urllib.parse.quote(url, safe="".join(chr(c) for c in range(33, 127)))
            req = urllib.request.Request(safe, headers={"User-Agent": "research-client"})
            path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            return url, True
        except (OSError, ValueError):  # network errors (URLError, HTTPError, timeouts) are retried, then reported
            continue
    return url, False


def probe(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return out


def main() -> None:
    test = fetch_csv(TEST_CSV, "videophy2_test.csv")
    auto = fetch_csv(AUTO_CSV, "videophy2_autoeval_scores.csv")
    test = test.merge(auto[["video_url", "pc_pred", "sa_pred"]], on="video_url", how="left")
    test = test.drop(columns=[c for c in test.columns if c.startswith("Unnamed")])
    test["video"] = test.video_url.map(vid_id)
    assert test.video.is_unique

    rng = np.random.default_rng(SEED)
    picks = []
    for m in MODELS:
        pool = test[test.model_name == m].sort_values("video_url")
        picks.append(pool.iloc[rng.choice(len(pool), PER_MODEL, replace=False)])
    items = pd.concat(picks).reset_index(drop=True)
    captions = np.array(sorted(items.caption.unique()))
    rng.shuffle(captions)
    dev_caps = set(captions[: len(captions) // 3])
    items["split"] = np.where(items.caption.isin(dev_caps), "dev", "test")
    items["prompt_id"] = items.caption.map({c: f"p{k:03d}" for k, c in enumerate(sorted(items.caption.unique()))})

    with ThreadPoolExecutor(16) as ex:
        ok = dict(ex.map(download, items.video_url))
    items["downloaded"] = items.video_url.map(ok)
    meta = pd.DataFrame(
        [probe(VIDEOS / f"{v}.mp4") if ok[u] else {} for v, u in zip(items.video, items.video_url, strict=True)]
    )
    items = pd.concat([items, meta], axis=1)
    items["violation"] = (items.pc < 4).astype(int)

    rules = []
    for r in items.itertuples():
        laws = lit(r.metadata_rules) or {}
        for label, col in [(0, "physics_rules_unfollowed"), (1, "physics_rules_followed")]:
            for k, rule in enumerate(lit(getattr(r, col)) or []):
                law = laws.get(rule) or ["None"]
                law = law[0] if isinstance(law, list) else str(law)
                rules.append(
                    {
                        "video": r.video,
                        "rule_id": f"{r.video}_{label}{k}",
                        "rule": rule,
                        "law": law,
                        "law_group": law if law in LAWS else "Other",
                        "violated": 1 - label,
                        "split": r.split,
                    }
                )
    rules = pd.DataFrame(rules)

    rest = test[~test.video.isin(items.video) & (test.pc == 5) & test.model_name.isin(MODELS)].copy()
    rest = rest[rest.physics_rules_unfollowed.map(lambda x: len(lit(x) or []) == 0)]
    rest = rest[rest.caption.str.contains(MOTION, case=False, regex=True)].sort_values("video_url")
    rest = rest[~rest.caption.isin(items.caption)]
    src = rest.iloc[rng.choice(len(rest), min(25, len(rest)), replace=False)].copy()
    with ThreadPoolExecutor(8) as ex:
        ok2 = dict(ex.map(download, src.video_url))
    src["downloaded"] = src.video_url.map(ok2)

    keep = ["video", "prompt_id", "split", "model_name", "caption", "upsampled_caption", "action", "category",
            "is_hard", "sa", "pc", "violation", "pc_pred", "sa_pred", "video_url", "downloaded", "fps", "frames",
            "width", "height", "physics_rules_cannot_be_determined", "human_violated_rules"]  # fmt: skip
    items[keep].to_csv(RESULTS / "items.csv", index=False)
    rules.to_csv(RESULTS / "rules.csv", index=False)
    src[["video", "model_name", "caption", "action", "pc", "video_url", "downloaded"]].to_csv(
        RESULTS / "trap_sources.csv", index=False
    )
    print(items.groupby("split").agg(n=("video", "size"), prompts=("prompt_id", "nunique"), viol=("violation", "mean")))
    print("downloaded", items.downloaded.sum(), "of", len(items), "| trap sources", int(src.downloaded.sum()))
    print(rules.groupby(["split", "law_group"]).violated.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
