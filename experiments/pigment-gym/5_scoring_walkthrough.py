"""
One picture per case, walked through the whole reward, term by term.

Run from the repo root (after the benchmark (scripts 1-3) scripts 2-4, which save the samples this script borrows one
painting from):
    .venv/bin/python experiments/pigment-gym/5_scoring_walkthrough.py

Six sketches, from "crashes" to "a frontier model's painting". For each: does it pass the gate (and if not, why),
what CLIP says raw and after squashing to 0..1, what the aesthetic head says, and the final reward. For sketches
the gate rejects we still show what CLIP *would* have said, which is the point: the text hack would have scored
like a real painting.

    reward = 0.10 * gate + 0.05 * length + gate * (0.40 * CLIP + 0.45 * aesthetic)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brush_tools import PROMPT_EXAMPLE, Renderer, extract_code
from common import HERE, RESULTS, check
from hand_pool import SUBJECT, SUNSET_HILLS, SYNTAX_ERROR, TEXT_HACK, TINY_DOT
from PIL import Image, ImageDraw, ImageFont
from reward_tools import (
    AESTHETIC_RANGE,
    CLIP_RANGE,
    Scorer,
    caption,
    png_to_image,
    score_rollouts,
)

FIGURE = HERE / "report" / "figures" / "scoring_walkthrough.png"
RESULT = RESULTS / "scoring_walkthrough.json"

samples = RESULTS / "benchmark_samples.jsonl"
opus = None
if samples.exists():
    records = [json.loads(line) for line in samples.read_text().splitlines()]
    opus = next(
        (r for r in records if r["model"] == "opus-5.5" and r.get("prompt") == "no_example" and r["index"] == 4), None
    )

cases = [
    ("crashes (a missing bracket)", SYNTAX_ERROR),
    ("writes the words", TEXT_HACK),
    ("one tiny dot", TINY_DOT),
    ("the prompt's example", extract_code(PROMPT_EXAMPLE)),
    ("hand-made sunset over hills", SUNSET_HILLS),
]
if opus:
    cases.append(("a frontier model's painting", extract_code(opus["text"])))

scorer = Scorer()
with Renderer(pool_size=4) as renderer:
    renders = renderer.render_many([code for _, code in cases])
rows = score_rollouts([code for _, code in cases], renders, scorer, SUBJECT)

# What CLIP and the aesthetic head would say if there were no gate (only for sketches that produced an image).
# p5's text() draws nothing on a WEBGL canvas without a loaded font, so the text hack renders blank in the browser.
# To show what the gate protects against, the text case uses the benchmark (scripts 1-3)'s image of the words drawn on
# paper.
text = caption(SUBJECT)
images = [png_to_image(render.png) if render.png is not None else None for render in renders]
words = Image.new("RGB", (400, 400), "#f4efe4")
draw = ImageDraw.Draw(words)
font = ImageFont.load_default(size=40)
draw.text((20, 140), "SUNSET OVER", fill="#c8553d", font=font)
draw.text((20, 200), "GREEN HILLS", fill="#386641", font=font)
images[[name for name, _ in cases].index("writes the words")] = words
for row, image in zip(rows, images, strict=False):
    if row["clip_raw"] is None and image is not None and "js_error" not in row["violations"]:
        clip_raw, aesthetic_raw = scorer.score_images([image], text)
        row["clip_raw_ungated"], row["aesthetic_raw_ungated"] = clip_raw[0], aesthetic_raw[0]

print(f'Caption CLIP compares against: "{text}"')
print(f"CLIP raw range squashed to 0..1: {CLIP_RANGE}; aesthetic raw range: {AESTHETIC_RANGE}\n")
print(
    f"{'case':30} {'gate':5} {'why it failed':24} {'CLIP raw':>8} {'CLIP':>5} {'aesth raw':>9} {'aesth':>5} {'reward':>6}"  # noqa: E501
)
summary = []
for (name, _), row in zip(cases, rows, strict=False):
    clip_raw = row["clip_raw"] if row["clip_raw"] is not None else row.get("clip_raw_ungated")
    aesthetic_raw = row["aesthetic_raw"] if row["aesthetic_raw"] is not None else row.get("aesthetic_raw_ungated")
    why = ", ".join(v.split(":")[0] for v in row["violations"]) or "-"
    clip_shown = f"{clip_raw:.3f}" if clip_raw is not None else "-"
    aesthetic_shown = f"{aesthetic_raw:.2f}" if aesthetic_raw is not None else "-"
    print(
        f"{name:30} {'pass' if row['gate'] else 'FAIL':5} {why:24} {clip_shown:>8} {row['clip']:5.2f} "
        f"{aesthetic_shown:>9} {row['aesthetic']:5.2f} {row['reward']:6.3f}"
    )
    summary.append(
        {
            "name": name,
            "gate": bool(row["gate"]),
            "violations": row["violations"],
            "clip_raw": clip_raw,
            "clip": row["clip"],
            "aesthetic_raw": aesthetic_raw,
            "aesthetic": row["aesthetic"],
            "length": row["length"],
            "reward": row["reward"],
        }
    )
RESULT.write_text(json.dumps(summary, indent=1))

# The figure: the six pictures in a row, and a table of their scores underneath.
GREY, GREEN, EMBER, PAPER, RULE = "#8A847C", "#3F7D4E", "#C9561C", "#F7F3EA", "#DDD6CA"
short = ["Crashes", "Writes the words", "One tiny dot", "Prompt's example", "Hand-made sunset", "Frontier model"]
fig = plt.figure(figsize=(10, 3.1))
grid = fig.add_gridspec(2, len(cases), height_ratios=[1, 0.62], hspace=0.06, wspace=0.06, left=0.1, right=0.995,
                        top=0.93, bottom=0.01)  # fmt: skip
for i, (image, item) in enumerate(zip(images, summary, strict=False)):
    ax = fig.add_subplot(grid[0, i])
    if image is not None and item["clip_raw"] is not None:
        ax.imshow(image)
    else:
        ax.imshow(Image.new("RGB", (400, 400), "#ececec"))
        note = "no image\n(crashed)"
        ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes, fontsize=7.5, color=GREY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B9B2A6")
        spine.set_linewidth(0.6)
    ax.set_title(short[i], fontsize=8, fontweight="bold", pad=4)


def raw_to_scaled(item, key, fmt):
    """'raw -> 0-1' for passing sketches; '(raw)' for gated ones, what the scorer would have said; '-' if none."""
    raw = item[f"{key}_raw"]
    if raw is None:
        return "—"
    return f"{fmt.format(raw)} → {item[key]:.2f}" if item["gate"] else f"({fmt.format(raw)})"


table = fig.add_subplot(grid[1, :])
table.axis("off")
cells = [
    ["pass" if item["gate"] else "fail" for item in summary],
    [raw_to_scaled(item, "clip", "{:.3f}") for item in summary],
    [raw_to_scaled(item, "aesthetic", "{:.2f}") for item in summary],
    [f"{item['reward']:.2f}" for item in summary],
]
labels = ["Gate", "CLIP, raw → 0–1", "Aesthetic, raw → 0–1", "Reward"]
drawn = table.table(cellText=cells, rowLabels=labels, cellLoc="center", bbox=[0, 0, 1, 1])
drawn.auto_set_font_size(False)
drawn.set_fontsize(7.5)
for (r, c), cell in drawn.get_celld().items():
    cell.set_edgecolor(RULE)
    cell.set_linewidth(0.6)
    if c == -1:
        cell.set_text_props(color=GREY, ha="right")
        cell.set_edgecolor("white")
    elif r == 0:
        cell.set_text_props(color=GREEN if summary[c]["gate"] else EMBER)
    elif r == 3:
        cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_facecolor(PAPER)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"\nsaved {FIGURE.name} and {RESULT.name}")

print("\nSelf-checks:")
by_name = {item["name"]: item for item in summary}
check(
    "the crash and the text hack fail the gate",
    not by_name["crashes (a missing bracket)"]["gate"] and not by_name["writes the words"]["gate"],
)
check(
    "a failed sketch never scores above 0.05",
    all(item["reward"] <= 0.05 + 1e-9 for item in summary if not item["gate"]),
)
check(
    "the hand-made sunset beats the tiny dot",
    by_name["hand-made sunset over hills"]["reward"] > by_name["one tiny dot"]["reward"],
)
check("the figure was saved", FIGURE.exists())
