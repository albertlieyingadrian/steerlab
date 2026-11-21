"""Shared pieces of the Voice Alchemy study: the voice traits, the prompt sets, the loop measure, the composition
score and label, and the spend ledger. No GPU or network code here, so every script (and the Modal container) can
import it. The study's plan (design, predictions, dated amendments) was fixed before any confirmatory run."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    "gemma": {
        "hf": "google/gemma-2-2b-it",
        "sub": "",
        "cand_layers": (6, 9, 12),
        "mlp_layers": tuple(range(4, 15)),
        "units_dir": "/vol/units",
    },
    "qwen": {
        "hf": "Qwen/Qwen3-8B",
        "sub": "qwen3-8b",
        "cand_layers": (8, 13, 17),
        "mlp_layers": tuple(range(6, 20)),
        "units_dir": "/vol/units_qwen",
    },
}
"""Gemma-2-2B-it was run first; Qwen3-8B (the model the user asked to explain) is the main model (amendment 11).
Qwen's candidate layers sit at the same depth fractions as Gemma's (0.23, 0.35, 0.46 of the stack)."""
TAG = os.environ.get("VA_MODEL", "gemma")
CFG = MODELS[TAG]
MODEL = CFG["hf"]
DATA = HERE / "data" / CFG["sub"] if CFG["sub"] else HERE / "data"
RESULTS = HERE / "results" / CFG["sub"] if CFG["sub"] else HERE / "results"
SHARED = HERE / "results"
DEFAULT_JUDGE = "haiku" if TAG == "gemma" else "llama"
JUDGE = os.environ.get("VA_JUDGE", DEFAULT_JUDGE)
"""The primary judge for this model's numbers. The Gemma run was judged by Claude Haiku 5.5; no judge passed the
known-answer bars for Qwen (amendment 10), so Qwen's default is the open model that came closest, for later use."""
SUFFIX = "" if JUDGE == DEFAULT_JUDGE else f"_{JUDGE}"
"""The judge cache, the known-answer checks and nothing model-specific live in SHARED."""
MAX_NEW = 160

# --- traits -------------------------------------------------------------------------------------------------------
TRAITS: dict[str, str] = {
    "terse": "very terse: as few words as possible, clipped sentences, no elaboration or pleasantries",
    "formal": "formal: elevated, official vocabulary, no contractions or slang, like an official letter or a report",
    "salesy": "salesy: pitching and hyping, persuading the reader to buy or act, marketing language and calls to action",
    "assistant": "like a stereotypical AI assistant: 'Great question!', headings and bullet points, disclaimers, "
    "offers of further help",
    "warm": "warm: kind, caring and encouraging, speaking to the reader personally and with empathy",
    "playful": "playful: jokes, puns, whimsy and exclamations, having fun with the topic",
    "hedging": "hedging: unsure and noncommittal, qualifying every claim with maybe, might, perhaps or it depends",
    "casual": "casual: relaxed and chatty, contractions and slang, like texting a friend",
}
"""Each trait's definition, used both in the style instruction that makes set F's replies and in the judge rubric."""

SHORT = {"assistant": "assistant-like"}


def label(trait: str) -> str:
    return SHORT.get(trait, trait)


def instruction(trait: str) -> str:
    """The style instruction (set F replies and the prompting baseline)."""
    return f"Write your answer in a style that is {TRAITS[trait]}."


def instruction_pair(a: str, b: str) -> str:
    return f"Write your answer in a style that is both {TRAITS[a]}, and {TRAITS[b]}."


# --- prompts --------------------------------------------------------------------------------------------------------
CATEGORIES = ("explain", "advise", "describe", "message")

F_PROMPTS = [
    # explain
    "Explain how a microwave oven heats food.",
    "Explain why the sky is blue.",
    "Explain how vaccines work.",
    "Explain what inflation is.",
    "Explain how a bicycle stays upright.",
    "Explain how bread rises.",
    "Explain what a credit score is.",
    "Explain how solar panels make electricity.",
    "Explain why leaves change colour.",
    "Explain how a password manager works.",
    # advise
    "How can I sleep better?",
    "How do I start running as a beginner?",
    "What should I do before a job interview?",
    "How can I save money on groceries?",
    "How do I keep a houseplant alive?",
    "How do I learn to cook?",
    "How can I stop procrastinating?",
    "What should I look for when renting an apartment?",
    "How do I make friends in a new city?",
    "How do I prepare for a long flight?",
    # describe
    "Describe a thunderstorm.",
    "Describe a busy train station.",
    "Describe the taste of coffee.",
    "Describe an old library.",
    "Describe a farmers market on a Saturday.",
    "Describe the ocean at night.",
    "Describe a small mountain village.",
    "Describe a city park in autumn.",
    "Describe a bakery early in the morning.",
    "Describe a snowy forest.",
    # message
    "Write a note to a coworker asking to swap shifts.",
    "Write a message inviting friends to a picnic.",
    "Write a short thank-you note to a teacher.",
    "Write a message telling a landlord the heater is broken.",
    "Write a note to a neighbour about a lost cat.",
    "Write a message to a friend who just got a new job.",
    "Write a reminder to a team about Friday's meeting.",
    "Write a message apologising for missing a call.",
    "Write a note asking a friend to return a borrowed book.",
    "Write a message welcoming a new club member.",
]
"""Set F (unit finding): 40 prompts, 10 per category. Never used for dev or held-out."""

DEV_PROMPTS = [
    ("explain", "Explain how a refrigerator keeps food cold."),
    ("explain", "Explain what a black hole is."),
    ("advise", "What should I pack for a weekend hike?"),
    ("advise", "How do I ask my boss for a raise?"),
    ("describe", "Describe a crowded beach in summer."),
    ("describe", "Describe a quiet kitchen at midnight."),
    ("message", "Write a message asking a friend to help you move."),
    ("message", "Write a note to a customer whose order is late."),
]

PILOT_PROMPTS = [
    ("explain", "Explain how a compass works."),
    ("advise", "How do I choose a good laptop?"),
    ("describe", "Describe a night market."),
    ("message", "Write a message congratulating a sister on her wedding."),
]

HELDOUT_PROMPTS = [
    ("explain", "Explain how an airplane stays in the air."),
    ("explain", "Explain how a dishwasher cleans dishes."),
    ("explain", "Explain what a mortgage is."),
    ("explain", "Explain why we have seasons."),
    ("advise", "How do I get better at public speaking?"),
    ("advise", "What should I do if my car won't start?"),
    ("advise", "How can I drink more water during the day?"),
    ("advise", "How do I pick a good watermelon?"),
    ("describe", "Describe a rainy afternoon in a cafe."),
    ("describe", "Describe a desert at sunrise."),
    ("describe", "Describe an empty school hallway."),
    ("describe", "Describe a garden in spring."),
    ("message", "Write a message asking a colleague to review a document."),
    ("message", "Write a note to a neighbour about a noisy party."),
    ("message", "Write a message reminding a friend about dinner on Saturday."),
    ("message", "Write a note thanking a host for a weekend stay."),
]
"""Held-out set: 16 prompts, 4 per category; every number in the note comes from these."""


def prompt_set(name: str) -> list[tuple[str, str]]:
    return {"dev": DEV_PROMPTS, "pilot": PILOT_PROMPTS, "heldout": HELDOUT_PROMPTS}[name]


# --- loop measure ---------------------------------------------------------------------------------------------------
LOOP_BAR = 0.20


def repeat_share(text: str, n: int = 4) -> float:
    """Share of word n-grams that repeat an earlier n-gram in the same reply (1 - distinct/total)."""
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def loops(text: str) -> bool:
    """A reply loops when more than 20% of its word 4-grams are repeats."""
    return repeat_share(text) > LOOP_BAR


# --- composition --------------------------------------------------------------------------------------------------
MIN_SINGLE = 1.0
"""A pair's composition is scored only when each single unit moves its own target trait by at least this much."""
EMERGE = 2.0
BREAK_COHERENCE = 4.0


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def additivity_error(d_a: list[float], d_b: list[float], d_ab: list[float]) -> float:
    resid = [ab - a - b for a, b, ab in zip(d_a, d_b, d_ab, strict=True)]
    return norm(resid) / max(norm(d_a) + norm(d_b), 1e-9)


def composition_label(
    traits: list[str],
    ta: str,
    tb: str,
    d_a: list[float],
    d_b: list[float],
    d_ab: list[float],
    lo_ab: list[float],
    loop_share: float,
    coherence: float,
    strict: bool = True,
) -> tuple[str, dict]:
    """The pre-registered label. `lo_ab` is the lower end of the 95% bootstrap interval of d_AB minus (d_A + d_B) per
    trait (for the emergent test). Returns (label, details)."""
    ia, ib = traits.index(ta), traits.index(tb)
    r_a = d_ab[ia] / d_a[ia] if abs(d_a[ia]) > 1e-9 else float("nan")
    r_b = d_ab[ib] / d_b[ib] if abs(d_b[ib]) > 1e-9 else float("nan")
    info = {"r_a": r_a, "r_b": r_b, "emergent_trait": None}
    if strict and (loop_share > LOOP_BAR or coherence < BREAK_COHERENCE):
        return "breaks", info
    if strict:
        best, best_c = 0.0, None
        for c, t in enumerate(traits):
            if c in (ia, ib):
                continue
            rise = d_ab[c] - d_a[c] - d_b[c]
            if rise >= EMERGE and lo_ab[c] > 0 and rise > best:
                best, best_c = rise, t
        if best_c is not None:
            info["emergent_trait"] = best_c
            return "emergent", info
    ka, kb = r_a >= 0.5, r_b >= 0.5
    if ka and kb:
        return "additive", info
    if ka or kb:
        return "dominates", info
    return "interferes", info


# --- spend ledger -------------------------------------------------------------------------------------------------
SPEND = HERE / "spend.jsonl"
MODAL_SOFT, MODAL_HARD, JUDGE_CAP = 80.0, 150.0, 15.0
H100_PER_S = 3.95 / 3600


def record_spend(kind: str, usd: float, note: str) -> None:
    with SPEND.open("a") as f:
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "usd": round(usd, 4), "note": note}
        f.write(json.dumps(row) + "\n")


def spent(kind: str | None = None) -> float:
    if not SPEND.exists():
        return 0.0
    rows = [json.loads(line) for line in SPEND.read_text().splitlines() if line.strip()]
    return sum(r["usd"] for r in rows if kind is None or r["kind"] == kind)


def check_budget(kind: str, planned: float = 0.0) -> None:
    cap = MODAL_HARD if kind == "modal" else JUDGE_CAP
    total = spent(kind) + planned
    if total > cap:
        raise SystemExit(f"budget stop: {kind} would reach ${total:.2f} > cap ${cap:.2f}")
    if kind == "modal" and total > MODAL_SOFT:
        print(f"note: modal spend ${total:.2f} is past the ${MODAL_SOFT:.0f} soft cap")
