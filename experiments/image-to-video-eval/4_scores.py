"""Cheap automatic scores for every clip: VBench++'s dimensions, CLIP and DINOv2 similarity, CoTracker camera motion.

    modal deploy modal_i2v.py
    uv run --no-project --with modal --with pandas python 4_scores.py human   # the clips in the sampled human pairs
    uv run --no-project --with modal --with pandas python 4_scores.py traps   # the broken copies from 2_traps.py
    uv run --no-project --with modal --with pandas python 4_scores.py ours    # our four models' clips (3_generate.py)

Uploads the re-encoded clips and the stills to the Modal volume (only those not there yet), then runs Scorer.score
(modal_i2v.py) on batches of 10 clips. Realism clips, traps and our clips get every dimension; camera clips get only
CLIP, DINOv2, dynamic degree and the camera-motion heuristic (the other dimensions do not bear on whether the camera
made the asked move, and leaving them out saves GPU time); for the budget, only the clips of the even-numbered
400 of the 800 camera pairs are scored. Writes results/scores_<source>.csv; a rerun scores only
clips missing from it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

BATCH = 10
COLS = ["video", "still", "q", "m"]


def items(source: str) -> list[dict]:
    if source == "human":
        pairs = pd.read_csv(RESULTS / "human_pairs.csv", keep_default_na=False)
        # Budget: camera clips only from the even-numbered 400 of the 800 camera pairs (fixed before any score).
        pairs = pairs[(pairs.question != "camera") | (pairs.pair.str[1:].astype(int) % 2 == 0)]
        long = pd.concat([pairs[["video_a", "still", "question", "movement"]].set_axis(COLS, axis=1),
                          pairs[["video_b", "still", "question", "movement"]].set_axis(COLS, axis=1)])  # fmt: skip
        long = long.drop_duplicates("video")
        return [{"video": r.video, "path": f"/cache/norm/{r.video}.mp4", "still": f"/cache/stills/{r.still}",
                 "dims": "camera" if r.q == "camera" else "full", "movement": r.m or None}
                for r in long.itertuples()]  # fmt: skip
    if source == "traps":
        traps = pd.read_csv(RESULTS / "traps.csv")
        return [{"video": r.trap_video, "path": f"/cache/norm/{r.trap_video}.mp4", "still": f"/cache/stills/{r.still}",
                 "dims": "full", "movement": None} for r in traps.itertuples()]  # fmt: skip
    clips = pd.read_csv(RESULTS / "gen_clips.csv", keep_default_na=False)
    clips = clips[clips.ok.astype(str) == "True"]
    return [{"video": r.video, "path": f"/cache/norm/{r.video}.mp4", "still": f"/cache/stills/{r.still}",
             "dims": "full", "movement": r.movement or None} for r in clips.itertuples()]  # fmt: skip


def upload(todo: list[dict]) -> None:
    volume = modal.Volume.from_name("image-to-video-eval")
    have = set()
    for folder in ("norm", "stills"):
        try:
            have |= {e.path.lstrip("/") for e in volume.listdir(folder)}
        except Exception as err:  # noqa: BLE001, RUF100 - an empty volume has no folder yet
            print(f"{folder}: {err!r}"[:200])
    files = {it["path"].removeprefix("/cache/") for it in todo} | {it["still"].removeprefix("/cache/") for it in todo}
    missing = sorted(files - have)
    print(f"uploading {len(missing)} files")
    # Chunks of 100, each retried, so one dropped connection does not lose the whole upload.
    for k in range(0, len(missing), 100):
        for attempt in range(4):
            try:
                with volume.batch_upload(force=True) as batch:
                    for f in missing[k : k + 100]:
                        batch.put_file(str(DATA / f), "/" + f)
                break
            except Exception as err:  # noqa: BLE001, RUF100 - retry a dropped upload
                print(f"upload chunk {k}: attempt {attempt + 1} failed: {err!r}"[:200])
        else:
            raise SystemExit(f"upload chunk {k} failed four times")
        print(f"  uploaded {min(k + 100, len(missing))}/{len(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["human", "traps", "ours"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = RESULTS / f"scores_{args.source}.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["video"])
    todo = [it for it in items(args.source) if it["video"] not in set(old.video)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} clips to score ({len(old)} done)")
    if not todo:
        return
    upload(todo)
    scorer = modal.Cls.from_name("image-to-video-eval", "Scorer")()
    batches = [todo[k : k + BATCH] for k in range(0, len(todo), BATCH)]
    rows = []
    for k, batch in enumerate(scorer.score.map(batches, order_outputs=False)):
        rows += batch
        if k % 10 == 0:
            print(f"  {k + 1}/{len(batches)} batches")
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False, float_format="%.6g")
    table = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(out, index=False, float_format="%.6g")
    errors = [c for c in table.columns if c.startswith("error_")]
    print(f"wrote {len(table)} rows; errors: {table[errors].notna().sum().to_dict() if errors else 'none'}")


if __name__ == "__main__":
    main()
