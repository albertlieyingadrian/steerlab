"""Score voi-slop-detect-4B (the target) once on every benchmark row we use, so our models meet it on the same rows.

    modal deploy modal_slop.py && python score_voi.py [split ...]

Splits: EditLens test, test_enron, test_llama (all from the local EditLens clone) and Human Detectors (300 rows, in the
same clone) and Epoch AI's detector test (1,089 rows, common.epoch_split). Input as the card asks: raw text with
fenced code emptied, first 512 tokens. Writes
results/voi/<split>.json with one score per row (no text: EditLens is CC BY-NC-SA, so only scores are committed).
"""

from __future__ import annotations

import json
import sys
import time

from common import RESULTS, VOI, VOI_REVISION, benchmark, empty_fences

SPLITS = ("test", "test_enron", "test_llama", "human_detectors", "epoch")
CHUNKS = 4


def main() -> None:
    import modal

    fn = modal.Function.from_name("slop-detector", "score")
    out_dir = RESULTS / "voi"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in sys.argv[1:] or SPLITS:
        path = out_dir / f"{split}.json"
        if path.exists():
            print(f"{split}: done already")
            continue
        df = benchmark(split)
        texts = [empty_fences(str(t)) for t in df["text"]]
        size = -(-len(texts) // CHUNKS)
        parts = [texts[i : i + size] for i in range(0, len(texts), size)]
        t0 = time.time()
        res = list(fn.starmap([(p, VOI, VOI_REVISION) for p in parts]))
        scores = [s for r in res for s in r["scores"]]
        tokens = [t for r in res for t in r["tokens"]]
        ids = df["text_id"].tolist() if "text_id" in df else list(range(len(df)))
        rows = [
            {"i": i, "text_id": str(t), "score": round(s, 6), "tokens": k}
            for i, (t, s, k) in enumerate(zip(ids, scores, tokens, strict=True))
        ]
        run = {
            "gpu": sorted({r["gpu"] for r in res}),
            "gpu_seconds": sum(r["seconds"] for r in res),
            "wall_seconds": round(time.time() - t0),
        }
        path.write_text(json.dumps({"model": VOI, "revision": VOI_REVISION, "run": run, "rows": rows}) + "\n")
        print(f"{split}: {len(rows)} rows, {run}")


if __name__ == "__main__":
    main()
