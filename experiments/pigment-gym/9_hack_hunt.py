"""
Hunt for the shortcut. What did the policy actually learn to write?

Run from the repo root (about a minute: re-renders 24 sketches and loads CLIP):
    .venv/bin/python experiments/pigment-gym/9_hack_hunt.py [--rollouts PATH]

Three lenses, like reading a suspicious pull request three ways:
  1. the code: does it contain the prompt's example block? how many times? any green hills?
  2. the copy check: the laptop run (7_train_laptop.py) failed sketches whose WHOLE FILE was >= 85% similar to the
  example.
     What if the example sits inside a longer file? We measure the best-matching WINDOW too.
  3. the pixels: re-render early and late sketches; is the bottom third (where hills go) painted at all?
     And a CLIP crop test: does the bottom third look like "green hills"? (with controls, so we know the test works)
"""

import difflib
import math
import random
from collections import Counter

import numpy as np
from analysis_tools import (
    code_hills,
    load_rollouts,
    parse_args,
    quarters,
    shape_counts,
    signature,
)
from brush_tools import PROMPT_EXAMPLE, Renderer, contact_sheet, extract_code
from common import check, outputs_dir
from hand_pool import SUBJECT, SUNSET_HILLS
from reward_tools import Scorer, caption, png_to_image, similarity

args = parse_args("Look for shortcuts in a GRPO run")
rows = load_rollouts(args.rollouts)
example = extract_code(PROMPT_EXAMPLE)
example_body = example[example.index("brush.noStroke();") : example.rindex("}")].strip()
EXAMPLE_BLOCK = signature(example)  # "T F BS V4 ES F C": one wash, one disc
print(f"the prompt's example, as a signature: {EXAMPLE_BLOCK!r}\n")


def best_window_similarity(code):
    """Highest similarity between the example's drawing code and any same-sized run of lines in `code`."""
    target = "\n".join(example_body.splitlines())
    size = len(example_body.splitlines())
    lines = code.splitlines()
    best = 0.0
    for start in range(max(1, len(lines) - size + 1)):
        window = "\n".join(lines[start : start + size])
        best = max(best, difflib.SequenceMatcher(None, target, window).ratio())
    return best


def entropy(counter):
    """Shannon entropy in bits: 0 = everyone writes the same template; higher = more variety."""
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values())


# ---- 1 + 2: the code ----
print(
    f"{'block':12} {'has example':>11} {'2+ copies':>9} {'window sim':>10} {'file sim':>8} {'hills':>5} "
    f"{'shapes':>6} {'entropy':>7}   (columns 2-5 over PASSING sketches)"
)
block_stats = []
for label, block in quarters(rows):
    passed = [row for row in block if row["gate"]]
    copies = [signature(row["code"]).count(EXAMPLE_BLOCK) for row in passed]
    shapes = Counter(tuple(shape_counts(row["code"]).values()) for row in block)
    n_passed = max(1, len(passed))  # an early block can have no passing sketches at all (e.g. a short run)
    stats = {
        "has_example": sum(c >= 1 for c in copies) / n_passed,
        "two_copies": sum(c >= 2 for c in copies) / n_passed,
        "window_sim": float(np.mean([best_window_similarity(row["code"]) for row in passed])) if passed else 0.0,
        "file_sim": float(np.mean([similarity(example, row["code"]) for row in passed])) if passed else 0.0,
        "hills": sum(code_hills(row["code"]) > 0 for row in block) / len(block),
        "shape_kinds": len(shapes),
        "entropy": entropy(shapes),
    }
    block_stats.append(stats)
    print(
        f"{label:12} {stats['has_example']:11.0%} {stats['two_copies']:9.0%} {stats['window_sim']:10.2f} "
        f"{stats['file_sim']:8.2f} {stats['hills']:5.0%} {stats['shape_kinds']:6d} {stats['entropy']:7.2f}"
    )
print("  shapes/entropy = distinct (polygon count, circle count) combinations and their entropy, over all sketches")

early, late = block_stats[0], block_stats[-1]
print(f"""
The shortcut: late passing sketches contain the example's drawing code almost verbatim (best window similarity
{late["window_sim"]:.2f}), but their WHOLE-FILE similarity is only {late["file_sim"]:.2f}, under the laptop run's 0.85 copy line. Any extra
code dilutes difflib's whole-file score (2 x matched / total length), so 'paste the example, then add more shapes'
slips through. The most common addition is a second, recoloured copy of the example itself: sketches with the
block twice went from {early["two_copies"]:.0%} to {late["two_copies"]:.0%} of passes.""")  # noqa: E501

# Why would doubling pay? Compare late sketches by how many copies of the example block they contain.
print("\nlate sketches (last quarter) by copies of the example block:")
print(f"  {'copies':>6} {'n':>3} {'gate pass':>9} {'failed copy rule':>16} {'mean reward':>11} {'CLIP|pass':>9}")
last_quarter = quarters(rows)[-1][1]
by_copies = {}
for label, test in [("0", lambda k: k == 0), ("1", lambda k: k == 1), ("2+", lambda k: k >= 2)]:
    group = [row for row in last_quarter if test(signature(row["code"]).count(EXAMPLE_BLOCK))]
    if not group:
        continue
    passed = [row for row in group if row["gate"]]
    by_copies[label] = {
        "pass": len(passed) / len(group),
        "copy_rule": sum("copies_example" in row["violations"] for row in group) / len(group),
        "reward": float(np.mean([row["reward_unshaped"] for row in group])),
        "clip": float(np.mean([row["clip_raw"] for row in passed])) if passed else float("nan"),
    }
    b = by_copies[label]
    print(f"  {label:>6} {len(group):3d} {b['pass']:9.0%} {b['copy_rule']:16.0%} {b['reward']:11.3f} {b['clip']:9.3f}")
print("  The copy rule almost never fires any more, yet most sketches still contain the example verbatim: the rule")
print(
    "  compares whole files, and the policy simply writes longer files. The 2+ group's small edge over the 1-copy group"
)
print("  (a few sketches out of 26 vs 36) is within noise; the template itself is the robust finding.")

doubled = example + "\n" + example.replace("#7fb2d9", "#3a8c14").replace("#d1495b", "#3a8c14")
print(
    f"  demo: the example pasted twice (second copy recoloured green): whole-file similarity "
    f"{similarity(example, doubled):.2f}, best window {best_window_similarity(doubled):.2f}"
)
last_step = max(row["step"] for row in rows)
late_doubled = [
    row
    for row in rows
    if row["step"] >= (last_step + 1) * 3 // 4 and row["gate"] and signature(row["code"]).count(EXAMPLE_BLOCK) >= 2
]
if late_doubled:
    code = late_doubled[0]["code"]
    draw_body = code[code.index("function draw") :]
    print(
        f"\n  a real late sketch (step {late_doubled[0]['step']}, reward {late_doubled[0]['reward_unshaped']:.2f}), draw():"  # noqa: E501
    )
    print("    " + "\n    ".join(draw_body.splitlines()[:26]))

# ---- 3: the pixels ----
rng = random.Random(8)
early_pool = [row for row in rows if row["step"] < (last_step + 1) // 4 and row["gate"]]
late_pool = [row for row in rows if row["step"] >= (last_step + 1) * 3 // 4 and row["gate"]]
early_sample = rng.sample(early_pool, min(12, len(early_pool)))
late_sample = rng.sample(late_pool, min(12, len(late_pool)))
controls = {
    "control: 3 hills (hand pool)": SUNSET_HILLS,
    "control: the prompt's example": example,
    "control: example + green copy": doubled,
}
codes = [row["code"] for row in early_sample + late_sample] + list(controls.values())
print(
    f"\nre-rendering {len(early_sample)} early and {len(late_sample)} late passing sketches, plus {len(controls)} controls"  # noqa: E501
)
with Renderer(pool_size=4) as renderer:
    renders = renderer.render_many(codes)
images = [png_to_image(render.png) for render in renders]

scorer = Scorer()
full_clip, _ = scorer.score_images(images, caption(SUBJECT))
recorded_clip = [row["clip_raw"] for row in early_sample + late_sample]
drift = max((abs(a - b) for a, b in zip(full_clip, recorded_clip, strict=False)), default=0.0)
print(f"re-scored CLIP vs the value logged during training: max difference {drift:.4f} (the reward is reproducible)")

PAPER = np.array([244, 239, 228], dtype=np.float32)


def pixel_stats(image):
    rgb = np.asarray(image, dtype=np.float32)
    bottom = rgb[rgb.shape[0] * 2 // 3 :]
    paper = np.sqrt(((bottom - PAPER) ** 2).sum(axis=-1)) < 32
    green = (bottom[..., 1] > bottom[..., 0] + 15) & (bottom[..., 1] > bottom[..., 2] + 15)
    return float(paper.mean()), float(green.mean())


bottoms = [image.crop((0, image.height * 2 // 3, image.width, image.height)) for image in images]
crop_clip, _ = scorer.score_images(bottoms, "a watercolour painting of green hills")
n_early, n_late = len(early_sample), len(late_sample)
groups = {"early": range(n_early), "late": range(n_early, n_early + n_late)}
print(
    f"\n{'group':32} {'bottom third: paper':>19} {'green':>6} {'CLIP(bottom, green hills)':>26} {'CLIP(full, subject)':>20}"  # noqa: E501
)
summary = {}
for name, index in groups.items():
    stats = [pixel_stats(images[i]) for i in index]
    summary[name] = (
        float(np.mean([s[0] for s in stats])),
        float(np.mean([s[1] for s in stats])),
        float(np.mean([crop_clip[i] for i in index])),
        float(np.mean([full_clip[i] for i in index])),
    )
    print(
        f"{name + ' passing sketches':32} {summary[name][0]:19.0%} {summary[name][1]:6.0%} {summary[name][2]:26.3f} "
        f"{summary[name][3]:20.3f}"
    )
for offset, name in enumerate(controls):
    i = n_early + n_late + offset
    paper, green = pixel_stats(images[i])
    summary[name] = (paper, green, crop_clip[i], full_clip[i])
    print(f"{name:32} {paper:19.0%} {green:6.0%} {crop_clip[i]:26.3f} {full_clip[i]:20.3f}")
print("""
Reading it: the controls show the crop test can tell hills from no hills. Compare the policy's bottom third with
them: if it is mostly bare paper, the policy never learned the 'green hills' half of the subject. The last column
shows the second copy doesn't help CLIP either (a green wash over the sky scores LOWER than the plain example).
Nothing in the reward asked for hills, so nothing produced them.""")

sheet = contact_sheet(
    [r.png for r in renders[n_early : n_early + n_late]],
    [f"step {row['step']} r={row['reward_unshaped']:.2f}" for row in late_sample],
    outputs_dir() / "late_sample.png",
)
print(f"saved the work folder {sheet.name}")

hills_control = summary["control: 3 hills (hand pool)"]
example_control = summary["control: the prompt's example"]
print("\nSelf-checks:")
check("re-rendering reproduces the logged CLIP scores (max difference < 0.01)", drift < 0.01)
check(
    "the crop test works: the hills control beats the example control on 'green hills'",
    hills_control[2] > example_control[2],
)
check("laptop-run finding: most late passing sketches contain the example block", late["has_example"] > 0.5)
check(
    "laptop-run finding: late sketches dodge the copy check (window similarity > 0.85 > whole-file similarity)",
    late["window_sim"] > 0.85 > late["file_sim"],
)
check("laptop-run finding: the late bottom third is mostly bare paper (no hills)", summary["late"][0] > 0.5)
late_copy_rule = sum("copies_example" in row["violations"] for row in last_quarter) / len(last_quarter)
check(
    "laptop-run finding: the copy rule fires on < 10% of late sketches even though most contain the example verbatim",
    late_copy_rule < 0.10 and late["has_example"] > 0.5,
)
