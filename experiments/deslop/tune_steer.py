"""Build the "assistant register" direction with steerlab and choose its layer and strength on the dev prompts.

    python tune_steer.py            # direction (once), then the sweep, then results/steer_choice.json

The direction: for each of the 288 training prompts, the model's own reply with no system prompt (positive) against
its reply under the plain-style prompt (negative), both read as replies to the bare prompt. steerlab's
`build_direction` averages each reply's activations at a layer and takes the difference of the two classes' means;
strength 1.0 adds one class gap. A random direction of the same size (seed 0) is stored beside it as the control.

The sweep: layers 12, 18 and 24 of 36, strengths -0.25 to -6 (negative = away from the assistant register), on the
24 dev prompts (training topics, never held-out ones). The chosen setting has the lowest mean deslop score among
settings where at least 22 of 24 replies are fluent (see `fluent`) and the task-fit pass rate is no lower than the
unsteered model's minus 2 replies.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "packages" / "deslop" / "src"))
from deslop.checks import task_fit
from deslop.score import score

from prompts import dev, training

RESULTS = HERE / "results"
LAYERS = (12, 18, 24)
# The first sweep stopped at -1.5, where every setting was still fluent and the best one sat on the grid's edge; the
# stronger settings were added after reading it (recorded in the note's "changed after reading replies").
STRENGTHS = (-0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def fluent(text: str, words: int) -> bool:
    """Reads as normal prose: long enough, sentences start with capitals, no 4-word phrase said more than 3 times."""
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if len(toks) < 0.4 * words:
        return False
    starts = re.findall(r"(?:^|[.!?]\s+)([A-Za-z])", text)
    if starts and sum(c.islower() for c in starts) > 0.1 * len(starts):
        return False
    grams = Counter(tuple(toks[i : i + 4]) for i in range(len(toks) - 3))
    return not grams or max(grams.values()) <= 3


def build() -> dict:
    base = {r["id"]: r["text"] for r in map(json.loads, (RESULTS / "gen" / "base-train.jsonl").open())}
    plain = {r["id"]: r["text"] for r in map(json.loads, (RESULTS / "gen" / "plain-train.jsonl").open())}
    pairs = [{"prompt": p["prompt"], "positive": base[p["id"]], "negative": plain[p["id"]]} for p in training()]
    runner = modal.Cls.from_name("deslop", "Runner")()
    out = runner.direction.remote(
        {
            "pairs": pairs,
            "layers": list(LAYERS),
            "description": "base reply vs plain reply",
        }
    )
    rng = np.random.default_rng(0)
    for layer, d in out.items():
        v = rng.standard_normal(len(d["vector"]))
        d["random_vector"] = (v / np.linalg.norm(v)).tolist()
        print(f"layer {layer}: class gap {d['class_gap']:.2f}, cosine agreement {d['cosine_agreement']:.3f}, "
              f"d' {d['d_prime']}")  # fmt: skip
    return {str(k): v for k, v in out.items()}


def summarise(rows: list[dict], prompts: dict[str, dict]) -> dict:
    scores = [score(r["text"]).score for r in rows]
    flu = [fluent(r["text"], prompts[r["id"]]["words"]) for r in rows]
    fit = [all(task_fit(r["text"], prompts[r["id"]]["genre"], prompts[r["id"]]["words"]).values()) for r in rows]
    return {
        "mean_score": float(np.mean(scores)),
        "fluent": int(sum(flu)),
        "task_fit": int(sum(fit)),
        "n": len(rows),
    }


def main() -> None:
    path = RESULTS / "directions.json"
    if not path.exists():
        path.write_text(json.dumps(build()))
    directions = json.loads(path.read_text())
    rows = [{"id": p["id"], "prompt": p["prompt"]} for p in dev()]
    prompts = {p["id"]: p for p in dev()}
    settings = [(None, 0.0)] + [(layer, s) for layer in LAYERS for s in STRENGTHS]
    jobs = []
    for layer, strength in settings:
        job = {
            "rows": rows,
            "n": 1,
            "temperature": 0.7,
            "max_new_tokens": 512,
            "seed": 0,
        }
        if layer is not None:
            d = directions[str(layer)]
            job["steer"] = {
                "vector": d["vector"],
                "layer": layer,
                "class_gap": d["class_gap"],
                "strength": strength,
            }
        jobs.append(job)
    runner = modal.Cls.from_name("deslop", "Runner")()
    sweep = []
    tune_dir = RESULTS / "gen" / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    for (layer, strength), out in zip(settings, runner.generate.map(jobs), strict=True):
        name = "none" if layer is None else f"L{layer}_s{strength:g}"
        with (tune_dir / f"{name}.jsonl").open("w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sweep.append({"layer": layer, "strength": strength, **summarise(out, prompts)})
        print(sweep[-1])
    (RESULTS / "steer_sweep.json").write_text(json.dumps(sweep, indent=1))
    unsteered = sweep[0]
    ok = [s for s in sweep[1:] if s["fluent"] >= 22 and s["task_fit"] >= unsteered["task_fit"] - 2]
    best = min(ok, key=lambda s: s["mean_score"])
    (RESULTS / "steer_choice.json").write_text(json.dumps({"layer": best["layer"], "strength": best["strength"]}))
    print(f"chosen: layer {best['layer']}, strength {best['strength']} ({best})")


if __name__ == "__main__":
    main()
