"""
Shared settings for the follow-up study (scripts 12-15): RL against two simpler baselines, on new subjects.

The question: GRPO raised Pigment Gym's reward without better paintings. Do two simpler methods do better on
true quality, judged by frontier models, on subjects no method trained on?
  base      Qwen2.5-Coder-0.5B-Instruct, untrained
  grpo      the published laptop run's LoRA (40 GRPO steps on "a sunset over green hills")
  bon8      best of 8 untrained samples, picked by Pigment Gym's own reward (no training at all)
  sft       LoRA fine-tuned on Claude Opus 5.5's sketches of 24 other subjects
  sft_bon8  best of 8 samples from the fine-tuned model (exploratory)
These settings, the arms and the four tests were fixed before any sample was drawn (the note's 4.8).
"""

import hashlib

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
IN_DISTRIBUTION = "a sunset over green hills"  # GRPO's only training subject
HELD_OUT = [
    "a lighthouse on a rocky shore",
    "a red sailboat on a calm lake",
    "a snowy mountain under a blue sky",
    "a field of yellow sunflowers",
    "a full moon over the sea at night",
    "a pine forest in the mist",
    "a hot air balloon over farmland",
    "a small cottage with a flower garden",
]
EVAL_SUBJECTS = [IN_DISTRIBUTION, *HELD_OUT]
# Subjects for the fine-tuning data only: none is an evaluation subject or a near copy of one.
TRAIN_SUBJECTS = [
    "a desert with cacti at noon",
    "a waterfall in a green jungle",
    "a city skyline at dusk",
    "a rainbow over a meadow",
    "a fishing village by a harbour",
    "autumn trees beside a river",
    "a lavender field in summer",
    "a volcano erupting at night",
    "a tropical beach with palm trees",
    "a castle on a hill",
    "a wheat field under storm clouds",
    "a frozen lake in winter",
    "a windmill beside a tulip field",
    "a cherry tree in blossom",
    "a canyon at sunrise",
    "a stone bridge over a stream",
    "a starry night over a village",
    "a bowl of fruit on a table",
    "a vase of red roses",
    "a whale leaping from the ocean",
    "a path through a birch forest",
    "a rowing boat at dawn",
    "a parrot on a branch",
    "a cup of coffee on a saucer",
]

GROUPS = 8  # per (model, subject): 8 groups of 8 samples = 64 sketches
GROUP_SIZE = 8
TEMPERATURE = 0.8  # the published held-out check's settings
TOP_P = 0.95
MAX_NEW_TOKENS = 1024  # 640 in the published runs; raised for every arm because Opus sketches run ~900 tokens


def group_seed(subject_index, group):
    """The same seeds for every model, so arms differ only in their weights."""
    return 30_000 + 100 * subject_index + group


RUBRIC = (
    "How good is this image as a watercolour painting of {subject}? Rate it from 1 to 5:\n"
    "1 = no recognisable picture\n"
    "2 = shapes or washes of colour, but the subject is not recognisable\n"
    "3 = the subject is recognisable but crude or incomplete\n"
    "4 = the subject is clearly painted, with a sensible composition\n"
    "5 = a convincing, well-composed watercolour of the subject\n"
    'Reply with JSON only: {{"rating": <1-5>, "note": "<what you see, under 12 words>"}}'
)
FAIL_RATING = 1  # a sketch that fails the gate is rated 1 without asking a judge (pre-registered)


def load_keys():
    """API keys from the nearest .env / .env.local above this folder (a git worktree has none of its own)."""
    from pathlib import Path

    from dotenv import dotenv_values

    for folder in Path(__file__).resolve().parents:
        if (folder / ".env").exists():
            return {**dotenv_values(folder / ".env"), **dotenv_values(folder / ".env.local")}
    return {}


def png_key(png):
    return hashlib.sha1(png).hexdigest()[:16]
