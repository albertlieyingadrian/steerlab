"""voi-slop-detect-4B (VoightKampffLabs, Apache-2.0): a 4B "how much AI" score on every text in the note.

    python voi_slop.py score     # score on Modal (one L4 or A10G), then summarise
    python voi_slop.py summary   # summarise saved scores only

Model: Qwen3.5-4B with a one-logit sequence-classification head; sigmoid(logit) is the share of the text written by an
LLM, from 0 to 1. Pinned to revision VOI_SLOP_REVISION in modal_voi_slop.py. As its card asks, the first 512 tokens are
read and fenced code blocks are emptied to a bare ``` pair before scoring.

Two readings, each with a 95% bootstrap interval over texts:
- like the other detectors: a threshold that flags 1% of this note's 318 human passages;
- its shipped bands (bands.json): human below 0.15, AI-generated at 0.75 or above, AI-edited between, and a yes/no flag
  at 0.385 (fitted by its authors to flag 1% of their validation human texts). The flag's rate on our 318 human
  passages is the check that the shipped cut holds on our data.
Same texts as detect.py and editlens.py: nothing is regenerated. Measurement only: nothing in the note was tuned
against this score. Writes results/voi_slop/scores.json and results/voi_slop/summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "voi_slop"
BANDS = {"human_max": 0.15, "generated_min": 0.75, "flag_min": 0.3846809906711658}  # bands.json at the pinned revision
FENCE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


def empty_fences(text: str) -> str:
    return FENCE.sub("```\n```", text)


def score() -> None:
    import modal

    from detect import texts
    from modal_voi_slop import VOI_SLOP, VOI_SLOP_REVISION

    items = texts()
    keys = list(items)
    cleaned = [empty_fences(items[k]) for k in keys]
    print(f"{len(keys)} texts, {sum(c != items[k] for c, k in zip(cleaned, keys, strict=True))} with code fences")
    out = modal.Function.from_name("deslop-voi", "score_voi_slop").remote(cleaned)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = zip(keys, out["scores"], out["tokens"], strict=True)
    payload = {
        "model": VOI_SLOP,
        "revision": VOI_SLOP_REVISION,
        "bands": BANDS,
        "run": {k: out[k] for k in ("seconds", "gpu", "transformers")},
        "scores": {k: {"score": round(s, 6), "tokens": t} for k, s, t in rows},
    }
    (OUT / "scores.json").write_text(json.dumps(payload) + "\n")
    print(f"scored {len(keys)} texts in {out['seconds']} s on {out['gpu']}")
    summary()


def summary() -> None:
    from analyze import RUNS, boot, cell

    scores = {k: v["score"] for k, v in json.loads((OUT / "scores.json").read_text())["scores"].items()}
    human = np.array([v for k, v in scores.items() if k.startswith("human/")])
    one_pct = float(np.quantile(human, 0.99))
    lo, hi, flag = BANDS["human_max"], BANDS["generated_min"], BANDS["flag_min"]

    def summarise(keep: Callable[[str], bool]) -> dict:
        vals = np.array([v for k, v in scores.items() if keep(k)])
        if not len(vals):
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": cell(boot(vals)),
            "human": cell(boot((vals < lo).astype(float))),
            "ai_edited": cell(boot(((vals >= lo) & (vals < hi)).astype(float))),
            "ai_generated": cell(boot((vals >= hi).astype(float))),
            "counts": [int((vals < lo).sum()), int(((vals >= lo) & (vals < hi)).sum()), int((vals >= hi).sum())],
            "flagged_shipped": cell(boot((vals >= flag).astype(float))),
            "flagged_1pct": cell(boot((vals > one_pct).astype(float))),
        }

    sets = {f"run/{r}": summarise(lambda k, r=r: k.startswith(f"run/{r}/")) for r in RUNS}
    for prefix in ("human/gutenberg/", "human/wikipedia/", "author/", "mirror/"):
        sets[prefix.rstrip("/")] = summarise(lambda k, p=prefix: k.startswith(p))
    sets["human"] = summarise(lambda k: k.startswith("human/"))
    for w in ("api", "lora"):
        sets[f"voice/{w}"] = summarise(
            lambda k, w=w: k.startswith("voice/") and k.endswith(f"/{w}") and "/none/" not in k
        )
    sets["voice/control"] = summarise(lambda k: k.startswith("voice/") and "/none/" in k)
    thresholds = {**BANDS, "flag_1pct_human": one_pct}
    (OUT / "summary.json").write_text(json.dumps({"thresholds": thresholds, "sets": sets}, indent=1) + "\n")
    print(thresholds)
    for k, v in sets.items():
        if v["n"]:
            print(
                f"{k:20} n={v['n']:3} mean {v['mean']['value']:.3f} bands {v['counts']} "
                f"flag0.385 {v['flagged_shipped']['value']:.2f} flag1% {v['flagged_1pct']['value']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["score", "summary"])
    {"score": score, "summary": summary}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
