"""Generate a clip from each of 40 stills with four open image-to-video models on Modal, and prepare the clips.

    modal deploy modal_i2v.py
    uv run --no-project --with modal --with pandas --with pillow --with numpy python 3_generate.py
    ... 3_generate.py --models wan --limit 1     # a timing smoke test

Stills (CC-BY 4.0, from the two Rapidata datasets): the 25 realism stills with their own prompts, and 15 of the 22
camera stills, each with one single-move camera prompt from that dataset (the 12 moves the camera heuristic can read,
in turn), so CoTracker can check whether each model made the move. Writes results/gen_jobs.csv, results/gen_timing.csv
(seconds and peak memory per model and clip) and results/gen_clips.csv, and puts each clip, re-encoded like the
benchmark clips, in $I2V_DIR/norm/gen_<model>_<id>.mp4 with a judge sheet.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import subprocess
import sys
from pathlib import Path

import modal
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

pairs_mod = importlib.import_module("1_human_pairs")
MODELS = {"wan": "gen_wan", "ltx": "gen_ltx", "cogvideox": "gen_cogvideox", "svd": "gen_svd"}
N_CAMERA = 15


def jobs() -> pd.DataFrame:
    path = RESULTS / "gen_jobs.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    pairs = pd.read_csv(RESULTS / "human_pairs.csv", keep_default_na=False)
    real = pairs[pairs.question == "photorealism"].drop_duplicates("still").sort_values("still")
    rows = [{"id": f"r{k:02d}", "still": r.still, "prompt": r.prompt, "move": "", "movement": ""}
            for k, r in enumerate(real.itertuples())]  # fmt: skip
    cam = pairs[pairs.question == "camera"]
    moves = cam.drop_duplicates("move").sort_values("move")
    stills = sorted(cam.still.unique())[:N_CAMERA]
    for k, still in enumerate(stills):
        m = moves.iloc[k % len(moves)]
        rows.append({"id": f"c{k:02d}", "still": still, "prompt": m.prompt, "move": m.move, "movement": m.movement})
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def prepare_clip(model: str, job: str) -> str:
    name = f"gen_{model}_{job}"
    src, norm = DATA / "gen" / model / f"{job}.mp4", DATA / "norm" / f"{name}.mp4"
    sheet = DATA / "sheets" / f"{name}.jpg"
    if src.exists() and not norm.exists():
        pairs_mod.encode(src, norm)
    if norm.exists() and not sheet.exists():
        frames = pairs_mod.read_frames(norm)
        pick = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        pairs_mod.make_sheet([frames[i] for i in pick]).save(sheet, quality=90)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()
    table = jobs()
    todo = table.head(args.limit) if args.limit else table
    volume = modal.Volume.from_name("image-to-video-eval")
    with volume.batch_upload(force=True) as batch:
        for s in sorted(set(todo.still)):
            batch.put_file(str(DATA / "stills" / s), f"/stills/{s}")
    if not args.skip_generate:
        payload = todo[["id", "still", "prompt"]].to_dict("records")
        calls = {m: modal.Function.from_name("image-to-video-eval", MODELS[m]).spawn(payload) for m in args.models}
        path = RESULTS / "gen_timing.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "id"])
        new = []
        for m, call in calls.items():
            try:
                new += call.get()
            except Exception as err:  # noqa: BLE001, RUF100 - one model failing should not lose the others
                print(f"{m}: generation failed: {err!r}"[:400])
        if new:
            new = pd.DataFrame(new)
            key = set(zip(new.model, new.id, strict=True))
            old = old[[(m, i) not in key for m, i in zip(old.model, old.id, strict=True)]]
            pd.concat([old, new], ignore_index=True).to_csv(path, index=False, float_format="%.3f")
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "image-to-video-eval", "gen",
                    str(DATA)], check=True)  # fmt: skip
    timing = pd.read_csv(RESULTS / "gen_timing.csv")
    if "source" not in timing:
        timing["source"] = "logged"
    timing = timing[timing.source == "logged"]
    timing = pd.concat([timing, file_times(timing)], ignore_index=True)
    timing.to_csv(RESULTS / "gen_timing.csv", index=False, float_format="%.3f")
    rows = []
    for model in sorted(set(timing.model)):
        for job in table.itertuples():
            name = prepare_clip(model, job.id)
            ok = (DATA / "norm" / f"{name}.mp4").exists()
            rows.append({"model": model, "id": job.id, "video": name if ok else "", "still": job.still,
                         "prompt": job.prompt, "movement": job.movement, "ok": ok})  # fmt: skip
    pd.DataFrame(rows).to_csv(RESULTS / "gen_clips.csv", index=False)
    print(pd.DataFrame(rows).groupby("model").ok.agg(["sum", "size"]))


def file_times(timing: pd.DataFrame) -> pd.DataFrame:
    """Seconds per clip for clips with no logged timing row, from the gaps between the times the clips were written
    to the volume. Wan and CogVideoX were stopped after the 25 realism stills to stay inside the budget, which lost
    their logged rows; a gap is the generation plus writing the mp4 (about a second), and the first clip of a run
    has no gap to measure. Peak memory for these comes from the logged one-clip timing run (source "logged")."""
    import json

    rows = []
    for model in sorted(set(timing.model)):
        out = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "--json", "image-to-video-eval",
                              f"gen/{model}"], capture_output=True, text=True, check=True).stdout  # fmt: skip
        files = sorted(json.loads(out), key=lambda f: f["created_modified"])
        have = set(timing[(timing.model == model) & timing.seconds.notna()].id)
        when = [(Path(f["filename"]).stem, pd.Timestamp(f["created_modified"])) for f in files]
        for (_, t0), (job, t1) in itertools.pairwise(when):
            if job not in have:
                rows.append({"model": model, "id": job, "seconds": (t1 - t0).total_seconds(), "source": "file times"})
    return pd.DataFrame(rows, columns=["model", "id", "seconds", "source"])


if __name__ == "__main__":
    main()
