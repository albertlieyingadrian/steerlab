"""Write GenEval-style training and validation prompts that never overlap the official 553.

Same templates as GenEval's `prompts/create_prompts.py` (MIT), other seeds. A prompt is dropped if its text, or its
task (tag + objects + counts + colours + relation), appears in the official set or in an earlier split. Single-object
prompts are left out: all 80 COCO classes are already official single-object prompts, so no new one exists.

Writes data/val.jsonl and data/train.jsonl (one GenEval metadata record per line, plus "split").
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CLASSES = [line.strip() for line in Path(HERE / "data/official/object_names.txt").read_text().splitlines()]
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
POSITIONS = ["left of", "right of", "above", "below"]
TAGS = ["two_object", "counting", "colors", "position", "color_attr"]


def article(name):
    return f"an {name}" if name[0] in "aeiou" else f"a {name}"


def plural(name):
    return f"{name}es" if name[-1] == "s" else f"{name}s"


def no_person(rng, k):
    """GenEval's trick for 'no colour on people': draw from every class but 'person'."""
    idx = rng.choice(len(CLASSES) - 1, size=k, replace=False) + 1
    return (idx + CLASSES.index("person")) % len(CLASSES)


def sample(tag, rng):
    if tag == "two_object":
        a, b = rng.choice(len(CLASSES), size=2, replace=False)
        return {
            "tag": tag,
            "include": [{"class": CLASSES[a], "count": 1}, {"class": CLASSES[b], "count": 1}],
            "prompt": f"a photo of {article(CLASSES[a])} and {article(CLASSES[b])}",
        }
    if tag == "counting":
        i = rng.choice(len(CLASSES))
        n = int(rng.integers(2, 4, endpoint=True))
        return {
            "tag": tag,
            "include": [{"class": CLASSES[i], "count": n}],
            "exclude": [{"class": CLASSES[i], "count": n + 1}],
            "prompt": f"a photo of {NUMBERS[n]} {plural(CLASSES[i])}",
        }
    if tag == "colors":
        (i,) = no_person(rng, 1)
        c = COLORS[rng.choice(len(COLORS))]
        return {
            "tag": tag,
            "include": [{"class": CLASSES[i], "count": 1, "color": c}],
            "prompt": f"a photo of {article(c)} {CLASSES[i]}",
        }
    if tag == "position":
        a, b = rng.choice(len(CLASSES), size=2, replace=False)
        p = POSITIONS[rng.choice(len(POSITIONS))]
        return {
            "tag": tag,
            "include": [{"class": CLASSES[b], "count": 1}, {"class": CLASSES[a], "count": 1, "position": [p, 0]}],
            "prompt": f"a photo of {article(CLASSES[a])} {p} {article(CLASSES[b])}",
        }
    if tag == "color_attr":
        a, b = no_person(rng, 2)
        ca, cb = rng.choice(len(COLORS), size=2, replace=False)
        return {
            "tag": tag,
            "include": [
                {"class": CLASSES[a], "count": 1, "color": COLORS[ca]},
                {"class": CLASSES[b], "count": 1, "color": COLORS[cb]},
            ],
            "prompt": f"a photo of {article(COLORS[ca])} {CLASSES[a]} and {article(COLORS[cb])} {CLASSES[b]}",
        }
    raise ValueError(tag)


def task_key(meta):
    """Order-free identity of what a prompt asks for (so 'a cat and a dog' == 'a dog and a cat')."""
    inc = sorted(
        json.dumps(
            {k: v for k, v in r.items() if k != "position"} | {"pos": r.get("position", [None])[0]}, sort_keys=True
        )
        for r in meta["include"]
    )
    return meta["tag"] + "|" + "|".join(inc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-per-tag", type=int, default=40)
    ap.add_argument("--train-per-tag", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    official = [
        json.loads(line) for line in Path(HERE / "data/official/evaluation_metadata.jsonl").read_text().splitlines()
    ]
    seen_text = {m["prompt"] for m in official}
    seen_task = {task_key(m) for m in official}
    rng = np.random.default_rng(args.seed)
    for split, per_tag in [("val", args.val_per_tag), ("train", args.train_per_tag)]:
        rows = []
        for tag in TAGS:
            got = misses = 0
            # Counting has only 80 classes x 3 numbers; stop once new tasks have run out.
            while got < per_tag and misses < 20000:
                m = sample(tag, rng)
                if m["prompt"] in seen_text or task_key(m) in seen_task:
                    misses += 1
                    continue
                misses = 0
                seen_text.add(m["prompt"])
                seen_task.add(task_key(m))
                rows.append(m | {"split": split})
                got += 1
            print(f"  {split} {tag}: {got}")
        with open(HERE / f"data/{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{split}: {len(rows)} prompts (up to {per_tag} per tag x {len(TAGS)} tags)")
    # Verify: no text or task in common with the official set, none between splits.
    val = [json.loads(line) for line in Path(HERE / "data/val.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in Path(HERE / "data/train.jsonl").read_text().splitlines()]
    off_t = {m["prompt"] for m in official}
    off_k = {task_key(m) for m in official}
    for name, rows in [("val", val), ("train", train)]:
        assert not ({r["prompt"] for r in rows} & off_t) and not ({task_key(r) for r in rows} & off_k), name
    assert not ({task_key(r) for r in val} & {task_key(r) for r in train})
    print("overlap check: none with official, none between val and train")


if __name__ == "__main__":
    main()
