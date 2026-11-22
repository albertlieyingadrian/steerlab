"""The three AI detectors on every text both tracks produced, plus the 318 human passages they are calibrated on.

    python rulers.py         # -> results/rulers/scores.json  {key: {voi380, nine_b, desklib, binoculars, fastdetect}}

Keys: `A/<arm>/<prompt>/<sample>` (Track A), `B/<run>/<prompt>` (Track B, Episode 1's saved replies),
`human/<source>/<id>`. A measurement only: no arm, setting or training reply was chosen by a detector.
- voi-slop-detect-380M at its pinned revision, run locally (Sounds Like Me's loader, imported).
- Our Qwen3.5-9B slop detector (slop-detector run `f9-9b`), on Modal (app `slop-detector`, `score_run`).
- Desklib (DeBERTa-v3-large on RAID), with Binoculars and Fast-DetectGPT alongside, from Episode 1's `Detectors`
  (app `deslop`). Track B and the human passages reuse Episode 1's saved Desklib scores (same model, same code); only
  Track A's texts are sent.
Thresholds (1% of the human passages flagged) are set in table.py, not here.
"""

from __future__ import annotations

import json
import sys

import modal

import budget
from common import EP1, GEN, HERE, RESULTS, TRACK_B, human_passages, read_jsonl

sys.path.insert(0, str(HERE.parent / "sounds-like-me"))
from detectors import voi380

OUT = RESULTS / "rulers" / "scores.json"


def track_a_arms() -> list[str]:
    return sorted(p.stem for p in GEN.glob("*.jsonl") if not p.stem.endswith("-train"))


def texts() -> dict[str, str]:
    out: dict[str, str] = {}
    for arm in track_a_arms():
        for r in read_jsonl(GEN / f"{arm}.jsonl"):
            out[f"A/{arm}/{r['id']}/{r['sample']}"] = r["text"]
    for run in TRACK_B:
        for r in read_jsonl(EP1 / "results" / "gen" / f"{run}.jsonl"):
            out[f"B/{run}/{r['id']}"] = r["text"]
    for h in human_passages():
        out[f"human/{h['source']}/{h['id']}"] = h["text"]
    return out


def main() -> None:
    T = texts()
    scores: dict[str, dict] = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [k for k in T if "voi380" not in scores.get(k, {})]
    if todo:
        for k, v in zip(todo, voi380([T[k] or "." for k in todo]), strict=True):
            scores.setdefault(k, {})["voi380"] = round(float(v), 6)
        print(f"voi380: {len(todo)} texts")
    todo = [k for k in T if "nine_b" not in scores.get(k, {})]
    if todo:
        fn = modal.Function.from_name("slop-detector", "score_run")
        chunks = [todo[i : i + 400] for i in range(0, len(todo), 400)]

        def run9():
            return [fn.remote("f9-9b", {"x": [T[k] or "." for k in c]})["x"] for c in chunks]

        for c, vals in zip(chunks, budget.paid("rulers-9b", "H100", 3.0, run9), strict=True):
            for k, v in zip(c, vals, strict=True):
                scores.setdefault(k, {})["nine_b"] = float(v)
        print(f"9B: {len(todo)} texts")
    ep1 = json.loads((EP1 / "results" / "detect" / "scores.json").read_text())
    for k in T:
        if k.startswith("B/"):
            _, run, pid = k.split("/")
            src = ep1.get(f"run/{run}/{pid}")
        elif k.startswith("human/"):
            src = ep1.get(k)
        else:
            src = None
        if src:
            scores.setdefault(k, {}).update({f: src[f] for f in ("desklib", "binoculars", "fastdetect")})
    todo = [k for k in T if "desklib" not in scores.get(k, {})]
    if todo:
        det = modal.Cls.from_name("deslop", "Detectors")()
        chunks = [todo[i : i + 200] for i in range(0, len(todo), 200)]

        def rund():
            return list(det.score.map([[T[k] for k in c] for c in chunks]))

        for c, vals in zip(chunks, budget.paid("rulers-desklib", "L40S", 3.0, rund, gpus=min(len(chunks), 4)),
                           strict=True):  # fmt: skip
            for k, v in zip(c, vals, strict=True):
                scores.setdefault(k, {}).update({f: v[f] for f in ("desklib", "binoculars", "fastdetect")})
        print(f"desklib: {len(todo)} texts")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scores) + "\n")
    print(f"{len(scores)} texts scored -> {OUT}")


if __name__ == "__main__":
    main()
