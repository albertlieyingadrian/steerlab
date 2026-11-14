"""
Step 3: the true task score, from judges no arm ever trained on.

Run from the repo root after the training runs (Opus through Message Batches at half price, about $1 per run):
    .venv/bin/python experiments/counsel-panels/2_heldout.py            # every finished run
    .venv/bin/python experiments/counsel-panels/2_heldout.py --run A_s0

  * Claude Opus 5.5 (effort low) rates every PASSING render of every training step and of the before/after
    evaluation sets on the Nov 12 1-5 rubric (pigment-gym/next_common.RUBRIC). A sketch that fails the gate is 1
    without asking. Ratings are cached by image, so a repeated picture is rated once.
  * Qwen3-VL-8B-Instruct, on a Modal A100 (modal_judges.py, amendment 3), rates the passing renders of the before/after
    evaluation sets on the same rubric: a held-out judge from a family no Claude counsel shares, so that arm E
    (a Claude vision counsel) is not only scored by Claude.
  * Judge check first: both rate the passing reference sketches; Spearman with the intended order.
Ratings go to results/heldout_cache.jsonl (keyed by judge and image hash); spend to results/spend_ledger.jsonl.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PIGMENT = HERE.parent / "pigment-gym"
sys.path[:0] = [str(PIGMENT), str(HERE)]

from next_common import RUBRIC  # noqa: E402
from panels import SUBJECT, Judges, b64, batch_score, ledger_add, sha  # noqa: E402

WORK = Path.home() / ".cache" / "steerlab" / "counsel-panels"
RUNS = HERE / "results" / "runs"
PROMPT = RUBRIC.format(subject=SUBJECT)
NO_VL = "--no-vl" in sys.argv  # Opus only (right after each run); Qwen3-VL once at the end, on every run


def opus_jobs(pngs):
    return [
        {
            "custom_id": f"h-{key}",
            "counsel": "heldout-opus",
            "model": "opus",
            "system": "",
            "keys": [key],
            "field": "rating",
            "max_tokens": 4000,
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png)}},
                {"type": "text", "text": PROMPT},
            ],
        }
        for key, png in pngs.items()
    ]


def qwen_vl(judges, pngs):
    if NO_VL:
        return
    """Qwen3-VL-8B ratings on a Modal GPU (amendment 3: no heavy jobs on the laptop)."""
    import modal_judges

    todo = {k: png for k, png in pngs.items() if ("heldout-qwenvl", k) not in judges.cache}
    if not todo:
        return
    with modal_judges.app.run():
        rater = modal_judges.OpenJudges()
        replies = rater.rate_image.starmap([(png, PROMPT) for png in todo.values()], return_exceptions=True)
        for key, text in zip(todo, replies, strict=True):
            try:
                rating = int(json.loads(re.search(r"\{.*\}", str(text), re.DOTALL).group(0))["rating"])
                if 1 <= rating <= 5:
                    judges._store("heldout-qwenvl", [key], [rating], 0.0, field="rating")
            except Exception as error:
                print(f"  qwen-vl {key}: {type(error).__name__}")


def judge_check(judges):
    from brush_tools import Renderer, gate
    from hand_pool import POOL

    with Renderer(pool_size=4) as renderer:
        renders = renderer.render_many([item["code"] for item in POOL])
    hand = [(item, r.png) for item, r in zip(POOL, renders, strict=True) if gate(item["code"], r).passed]
    pngs = {sha(png): png for _, png in hand}
    batch_score(judges, opus_jobs(pngs))
    qwen_vl(judges, pngs)
    out = {}
    for name in ("heldout-opus", "heldout-qwenvl"):
        ratings = [judges.cache[(name, sha(png))]["rating"] for _, png in hand]
        rho = float(spearmanr([-item["rank"] for item, _ in hand], ratings).statistic)
        out[name] = {"ratings": ratings, "spearman": rho, "names": [item["name"] for item, _ in hand]}
        print(f"judge check {name}: {ratings} Spearman {rho:+.2f}")
    return out


def load(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--no-vl", action="store_true")
    args = parser.parse_args()
    judges = Judges(HERE / "results" / "heldout_cache.jsonl")
    if not args.no_vl:
        check = judge_check(judges)
        (HERE / "results" / "heldout_judge_check.json").write_text(json.dumps(check, indent=2))
    runs = [args.run] if args.run else sorted(p.name.removesuffix("_summary.json") for p in RUNS.glob("*_summary.json"))
    all_train, all_eval = {}, {}
    for run in runs:
        png_dir = WORK / run / "png"
        for name in ("rollouts", "before_t1_samples", "after_t1_samples"):
            for row in load(RUNS / f"{run}_{name}.jsonl"):
                if row["gate"] and row["png"]:
                    png = (png_dir / f"{row['png']}.png").read_bytes()
                    all_train[row["png"]] = png
                    if name != "rollouts":
                        all_eval[row["png"]] = png
    print(f"{len(all_train)} passing images for Opus, {len(all_eval)} evaluation images for Qwen3-VL")
    before = judges.spent["anthropic_batch"]
    batch_score(judges, opus_jobs(all_train))
    qwen_vl(judges, all_eval)
    ledger_add({"run": ",".join(runs), "step": "heldout", "anthropic_batch": judges.spent["anthropic_batch"]})
    missing = sum(("heldout-opus", k) not in judges.cache for k in all_train)
    print(f"Opus batch spend ${judges.spent['anthropic_batch'] - before:.3f}; {missing} images without a rating")


if __name__ == "__main__":
    main()
