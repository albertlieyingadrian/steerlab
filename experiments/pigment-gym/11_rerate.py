"""
Is the reward still right where the policy now lives?

Run from the repo root (about a minute):
    .venv/bin/python experiments/pigment-gym/11_rerate.py --ratings results/judge_opus-5.5_laptop.csv
    .venv/bin/python experiments/pigment-gym/11_rerate.py --ratings my_ratings.csv --rollouts results/rollouts_h100.jsonl

The reward check (4_rank_check.py) tested the reward on 12 hand-written reference sketches. But RL moves the policy somewhere new, and a
reward can be right on the sketches you tested and wrong on the ones the policy now writes, like a test suite
that only covers last year's code paths. So we re-check on 32 sketches the policy itself wrote:
  1. pick 32 passing training sketches, 16 from the first half of the run and 16 from the second, shuffled
  2. re-render them and save a contact sheet labelled with IDs only (no rewards, so the rater isn't anchored)
  3. read a ratings file (1 = bad, 5 = a good watercolour of "a sunset over green hills") and compute Spearman
     between the reward and the ratings

The shipped ratings files (results/judge_*.csv) come from two judge models (10_judge.py). To use your own:
open rerate_sheet.png in the work folder, fill in ratings_template.csv, and pass it with --ratings.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from analysis_tools import DEFAULT_ROLLOUTS, load_rollouts, rerate_sample
from brush_tools import PROMPT_EXAMPLE, Renderer, contact_sheet, extract_code
from common import check, outputs_dir
from hand_pool import SUBJECT
from reward_tools import Scorer, score_rollouts
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(description="Re-check the reward on the policy's own sketches")
parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS, help="JSONL written by the laptop run (7_train_laptop.py)")
parser.add_argument("--ratings", type=Path, required=True, help="a ratings CSV: id,rating_1_to_5,note (10_judge.py writes them)")
args = parser.parse_args()

rows = load_rollouts(args.rollouts)
last_step = max(row["step"] for row in rows)
sample, ids, half = rerate_sample(rows)
print(f"32 passing sketches: 16 from steps 0-{half - 1}, 16 from steps {half}-{last_step}, shuffled as S01-S32\n")

codes = [row["code"] for row in sample]
with Renderer(pool_size=4) as renderer:
    renders = renderer.render_many(codes)
scored = score_rollouts(codes, renders, Scorer(), SUBJECT, forbid_copy_of=extract_code(PROMPT_EXAMPLE))

sheet = contact_sheet([render.png for render in renders], ids, outputs_dir() / "rerate_sheet.png")
template = outputs_dir() / "ratings_template.csv"
with template.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "rating_1_to_5", "note"])
    for sketch_id in ids:
        writer.writerow([sketch_id, "", ""])
print(f"saved the sheet (IDs only) to the work folder {sheet.name} and a blank template to {template.name}")

if not args.ratings.exists():
    sys.exit(f"\n{args.ratings} not found: rate the sheet, fill in the template, and rerun with --ratings")
ratings = {}
with args.ratings.open() as handle:
    for record in csv.DictReader(line for line in handle if not line.startswith("#")):
        if record["rating_1_to_5"].strip():
            ratings[record["id"]] = int(record["rating_1_to_5"])
source = f"ratings from {args.ratings.name}"
print(f"\nratings: {source}, {len(ratings)} of {len(ids)} sketches rated")

rated = [i for i, sketch_id in enumerate(ids) if sketch_id in ratings]
human = [ratings[ids[i]] for i in rated]
print(f"\n{'id':4} {'step':>4} {'rating':>6} {'reward':>6} {'CLIP':>6} {'aes':>5}")
for i in sorted(rated, key=lambda i: (-ratings[ids[i]], -scored[i]["reward"])):
    print(f"{ids[i]:4} {sample[i]['step']:4d} {ratings[ids[i]]:6d} {scored[i]['reward']:6.3f} "
          f"{scored[i]['clip_raw'] or 0:6.3f} {scored[i]['aesthetic_raw'] or 0:5.2f}")


def rho(values):
    return spearmanr([values[i] for i in rated], human).statistic


results = {
    "full reward": rho([row["reward"] for row in scored]),
    "CLIP only": rho([row["clip_raw"] or 0 for row in scored]),
    "aesthetic only": rho([row["aesthetic_raw"] or 0 for row in scored]),
    "training step (later = better?)": rho([row["step"] for row in sample]),
}
print("\nSpearman vs the ratings (the reward check (4_rank_check.py) got 0.81 on the 8 passing hand-made sketches):")
for name, value in results.items():
    print(f"  {name:32} {value:+.2f}")
first_ratings = [ratings[ids[i]] for i in rated if sample[i]["step"] < half]
second_ratings = [ratings[ids[i]] for i in rated if sample[i]["step"] >= half]
print(f"\nmean rating, first half of the run: {sum(first_ratings) / len(first_ratings):.2f}   "
      f"second half: {sum(second_ratings) / len(second_ratings):.2f}")
top_reward = max(rated, key=lambda i: scored[i]["reward"])
top_rated = max(rated, key=lambda i: (ratings[ids[i]], -scored[i]["reward"]))
print("\nbiggest disagreements:")
print(f"  highest reward:  {ids[top_reward]} (reward {scored[top_reward]['reward']:.3f}, aesthetic "
      f"{scored[top_reward]['aesthetic_raw']:.2f}) was rated {ratings[ids[top_reward]]}")
print(f"  highest rating:  {ids[top_rated]} (rated {ratings[ids[top_rated]]}) got reward {scored[top_rated]['reward']:.3f}, "
      f"rank {sorted(rated, key=lambda i: -scored[i]['reward']).index(top_rated) + 1} of {len(rated)}")
print("""
Reading it: on the policy's own sketches the reward has much less to work with. They are variations on one
template, so the differences are colours and small shifts, and the rater and the reward can easily disagree.
That's the point of re-rating: a reward validated on a hand-made pool isn't automatically valid on the policy's
output. With 32 items from one rater (an AI, by default), treat the numbers as a smoke test.""")

print("\nSelf-checks:")
check("all 32 re-rendered sketches still pass the gate", all(row["gate"] for row in scored))
check("every sketch has a rating from 1 to 5", len(ratings) == len(ids) and all(1 <= r <= 5 for r in ratings.values()))
check("Spearman values are numbers (the ratings vary)", all(not math.isnan(value) for value in results.values()))
