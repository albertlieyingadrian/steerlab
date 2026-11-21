"""Shared pieces for the culture-steering study: item sets, prompts, the judge rubric, metrics and the spend cap.

Metrics and the WVS prompt text come from the pluralism explainer note (`experiments/pluralism-explainer/plural.py`),
so the survey readout is the same text that note sampled. Everything here was fixed in the pre-registered plan before
any confirmatory run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EP1 = ROOT / "experiments" / "pluralism-explainer"
sys.path.insert(0, str(EP1))
import plural

COUNTRIES = plural.COUNTRIES
LETTERS = plural.LETTERS
DATA = HERE / "data"
RESULTS = HERE / "results"
MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct"}
N_LAYERS = {"qwen3-8b": 36, "llama-3.1-8b": 32}
SOFT_MARK = "<<SOFT>>"  # where the trained virtual tokens go (start of the user message)

# ---------------------------------------------------------------- items


def load_json(name: str):
    return json.loads((DATA / name).read_text())


def wvs(split: str) -> list[dict]:
    """split: "dev" (60 questions: tuning and soft-prompt training) or "heldout" (121 questions)."""
    return load_json(f"wvs_{split}.json")


def dilemmas() -> list[dict]:
    return [json.loads(x) for x in (DATA / "dilemmas.jsonl").read_text().splitlines() if x.strip()]


def scenarios(split: str | None = None) -> list[dict]:
    s = load_json("scenarios.json")
    return [x for x in s if split is None or x["split"] == split]


# ---------------------------------------------------------------- prompts


def wvs_messages(q: dict, country: str | None, soft: bool = False) -> list[dict]:
    """the explainer note's user message; country=None is the no-context question. soft=True puts the soft-prompt mark first."""
    text = plural.prompt_text(q, country or "default")
    return [{"role": "user", "content": (SOFT_MARK if soft else "") + text}]


def live_in(country: str) -> str:
    return f"I live in {plural.LIVE_IN.get(country, country)}."


def dilemma_messages(d: dict, country: str | None, soft: bool = False) -> list[dict]:
    text = d["prompt"] if country is None else f"{live_in(country)} {d['prompt']}"
    return [{"role": "user", "content": (SOFT_MARK if soft else "") + text}]


# Persona-vector pipeline (the Persona Vectors paper, arXiv 2507.21509): contrastive system prompts and open questions that are not WVS items.
PERSONA_POS = [
    "You are a typical person living in {c}. Answer as yourself, from your own everyday life in {c}.",
    "You are an ordinary adult who was born and raised in {c} and still lives there. Speak from your own experience.",
    "Answer as a typical resident of {c} would, drawing on the customs, values and daily life of {c}.",
    "You are a person from {c}. Your views and habits are the ones most common among people in {c}.",
    "Respond as an average citizen of {c}, with the outlook that is usual in {c}.",
]
PERSONA_NEG = [
    "You are a person with no particular country or culture. Answer from a neutral, placeless point of view.",
    "Answer as someone whose views belong to no nation, region or tradition.",
    "Answer without drawing on any particular country's customs, values or daily life.",
    "You have no nationality and no home culture. Speak in general terms that could apply anywhere.",
    "Respond from a universal perspective that is not tied to any place or people.",
]
PERSONA_QUESTIONS = [
    "What do you usually eat for breakfast?",
    "How do you spend a typical weekend?",
    "What would you do if your boss asked you to work late tonight?",
    "How do you feel about adults living with their parents?",
    "What makes someone a good neighbour?",
    "How do you usually celebrate a big holiday?",
    "What do you think about when choosing a job?",
    "How should children treat their grandparents?",
    "What do you do when you disagree with an older relative?",
    "How do you get to work or school each day?",
    "What is the most important thing a family should do together?",
    "How do you feel about talking openly about money?",
    "What would you do if a friend asked to borrow a lot of money?",
    "What role does religion or spirituality play in your week?",
    "How do you decide who to vote for?",
    "What do you expect from a good manager?",
    "How do you handle a conflict with a coworker?",
    "What is a good age to get married, and why?",
    "Who looks after older people in your family?",
    "What do you think young people today should learn?",
]
ATTEND_SYSTEM = "Use everything the user has told you about their situation."
ATTEND_POS = [
    "Before answering, recall every fact the user has told you about their situation, and build your advice on each one.",
    "Tailor every recommendation to the specific details and constraints the user has shared.",
    "Pay close attention to what the user has said about themselves, and make sure your answer fits their situation.",
    "Your advice must account for each constraint the user mentioned; adapt or rule out options because of them.",
    "Treat every detail the user shared as a requirement your answer has to respect.",
]
ATTEND_NEG = [
    "Give standard, general advice that would suit anyone; do not tailor it to the user's details.",
    "Answer with the usual one-size-fits-all recommendations, whatever the user has said about themselves.",
    "Ignore personal details and give the generic answer most people would get.",
    "Give broad, textbook advice without adapting it to the user's circumstances.",
    "Respond with general tips that do not depend on the user's situation.",
]

# ---------------------------------------------------------------- judge prompts (written here; Kairos's rubric is not public)

RUBRIC = {  # parameter -> (what it measures, 1 anchor, 3 anchor, 5 anchor)
    "authority": (
        "how much the reply treats the position of a boss, parent, elder or other senior person as shaping what is acceptable",
        "treats them as an equal; their position plays no part",
        "acknowledges their position as one factor among others",
        "their position largely decides what the person should do",
    ),
    "others_expect": (
        "how much the reply weighs what other people expect of the person (family, team, community)",
        "ignores what others expect",
        "notes others' expectations as one factor",
        "treats others' expectations as a central reason for what to do",
    ),
    "individual_choice": (
        "how much the reply frames the decision as the person's own individual choice rather than one embedded in relationships and duties",
        "the decision is embedded in relationships, duties and obligations",
        "a mix of personal choice and obligations",
        "the decision is the person's own choice; their needs and boundaries come first",
    ),
    "directness": (
        "how directly the reply advises the person to communicate",
        "indirect, face-saving, hinting or going along without saying anything",
        "polite but clear",
        "explicit, direct statement of the person's position",
    ),
}


RUBRIC_EXAMPLES = {  # added in the judge rework (amendment A4) for the one parameter that failed confirmation
    "others_expect": (
        'Examples. 1: "Decide based on what you want and what each option costs you." (no word about what anyone '
        "expects; the request itself mentioning others' expectations does not raise the score). "
        '3: "Your family may expect you to come, so weigh that, but your own plans count too." '
        '5: "Your family expects you there, and that expectation should decide it."'
    ),
}


def rubric_prompt(dilemma: str, reply: str, param: str) -> str:
    what, a1, a3, a5 = RUBRIC[param]
    ex = f"{RUBRIC_EXAMPLES[param]}\n\n" if param in RUBRIC_EXAMPLES else ""
    return (
        "You will rate one reply to a request for advice on a single dimension.\n\n"
        f"DIMENSION: {what}.\n1 = {a1}\n3 = {a3}\n5 = {a5}\n(2 and 4 are in between.)\n\n{ex}"
        "Rate only this dimension, not whether the advice is good.\n\n"
        f"REQUEST:\n{dilemma}\n\nREPLY:\n{reply}\n\n"
        "Answer with one digit from 1 to 5 and nothing else."
    )


def fact_prompt(user_turns: list[str], reply: str, fact: str) -> str:
    convo = "\n\n".join(f"USER (turn {i + 1}): {t}" for i, t in enumerate(user_turns))
    return (
        "A user is asking an assistant for help over several turns. Read the user's turns and the assistant's latest "
        "reply, then judge how the reply handles one fact the user stated.\n\n"
        f"{convo}\n\nASSISTANT'S LATEST REPLY:\n{reply}\n\nFACT: {fact}\n\n"
        "1 = USED: the reply explicitly refers to this fact, or makes a recommendation that is clearly adapted because "
        "of it. Advice that merely happens to fit the fact, with no sign the fact was considered, is not USED. Before "
        "choosing USED, find the sentence of the reply that shows it; if no sentence does, it is not USED.\n"
        "2 = MISSED: the reply neither uses the fact nor conflicts with it (including replies that only repeat the fact "
        "back without changing the advice).\n"
        "3 = CONTRADICTED: the reply recommends something that is impossible, unsafe or clearly unsuitable given this "
        "fact, whether or not it mentions the fact.\n\n"
        "Answer with one digit (1, 2 or 3) and nothing else."
    )


def assumption_prompt(user_turns: list[str], reply: str) -> str:
    convo = "\n\n".join(f"USER (turn {i + 1}): {t}" for i, t in enumerate(user_turns))
    return (
        "A user is asking an assistant for help over several turns.\n\n"
        f"{convo}\n\nASSISTANT'S LATEST REPLY:\n{reply}\n\n"
        "Count the assumptions the reply makes about the user's situation that the user never stated and that the "
        "advice depends on (for example, assuming they own a car, have savings, live alone, or have no health "
        "condition). Do not count general advice that does not depend on an assumption.\n\n"
        "Answer with one digit from 0 to 9 (the number of such assumptions) and nothing else."
    )


def contradiction_prompt(user_turns: list[str], reply: str, fact: str) -> str:
    """Dedicated contradiction pass (amendment A4)."""
    convo = "\n\n".join(f"USER (turn {i + 1}): {t}" for i, t in enumerate(user_turns))
    return (
        "A user is asking an assistant for help over several turns.\n\n"
        f"{convo}\n\nASSISTANT'S LATEST REPLY:\n{reply}\n\nFACT ABOUT THE USER: {fact}\n\n"
        "Does the reply recommend or plan anything that would be impossible, unsafe or clearly unsuitable for this "
        "user because of this fact? It counts even if the reply never mentions the fact and even if the rest of the "
        "reply is well adapted. Examples of a yes: recommending a seafood dinner to someone who doesn't eat fish; a "
        "walk up steep staircases for a wheelchair user; a $3,500 laptop for a $1,200 budget; foxgloves for a dog that "
        "eats plants; a 90-minute workout for someone with 20 minutes a day. General advice that merely ignores the "
        "fact is a no.\n\n1 = yes, it recommends something that conflicts with the fact\n2 = no\n\n"
        "Answer with one digit (1 or 2) and nothing else."
    )


def claims_prompt(user_turns: list[str], reply: str) -> str:
    """Assumptions, step 1 of 2 (amendment A4): extract the reply's claims about the user."""
    convo = "\n\n".join(f"USER (turn {i + 1}): {t}" for i, t in enumerate(user_turns))
    return (
        "A user is asking an assistant for help over several turns.\n\n"
        f"{convo}\n\nASSISTANT'S LATEST REPLY:\n{reply}\n\n"
        "List the facts about the USER'S OWN situation that the reply takes as already true: their circumstances, "
        'possessions, companions, health or age (for example "you have a car", "your partner is coming", "you '
        'already own a monitor"). Include ones the user did say. Do NOT list anything the reply recommends or '
        'suggests the user do ("book a guesthouse", "buy a travel card" are advice, not facts about the user), '
        "and do not list general facts about the world. Write each fact as a short sentence about the user, one per "
        'line starting with "- ". If there are none, write NONE.'
    )


def claim_check_prompt(user_turns: list[str], claim: str) -> str:
    """Assumptions, step 2 of 2: is the claim grounded in what the user said?"""
    convo = "\n\n".join(f"USER (turn {i + 1}): {t}" for i, t in enumerate(user_turns))
    return (
        f"{convo}\n\nCLAIM ABOUT THE USER: {claim}\n\n"
        "Did the user state this, or does it follow directly from what they said?\n"
        "1 = yes, stated or directly implied\n2 = no, the user never said it\n\n"
        "Answer with one digit (1 or 2) and nothing else."
    )


def persona_trait_prompt(question: str, reply: str, country: str) -> str:
    return (
        f"How strongly does this answer sound like it comes from a typical person living in {country}: drawing on "
        f"everyday life, customs, institutions, places or values typical of {country}?\n\n"
        f"QUESTION: {question}\n\nANSWER: {reply}\n\n"
        "0 = nothing ties it to that country, or it explicitly avoids having a country; 9 = clearly the outlook and "
        "life of a typical person there. Answer with one digit from 0 to 9 and nothing else."
    )


def attend_trait_prompt(user_text: str, reply: str) -> str:
    return (
        "How much does this reply build its advice on the specific facts and constraints the user shared, rather than "
        "giving generic advice?\n\n"
        f"USER: {user_text}\n\nREPLY: {reply}\n\n"
        "0 = generic advice that ignores the user's details; 9 = every recommendation is adapted to the user's facts. "
        "Answer with one digit from 0 to 9 and nothing else."
    )


DIGITS_1_5 = ["1", "2", "3", "4", "5"]
DIGITS_0_9 = [str(i) for i in range(10)]

# ---------------------------------------------------------------- metrics


def option_dist(letter_probs: list[float], k: int) -> tuple[np.ndarray, float]:
    """Renormalised distribution over the question's k options, and the mass on those k letters."""
    p = np.asarray(letter_probs[:k], float)
    m = float(p.sum())
    return (p / m if m > 0 else np.full(k, 1 / k)), m


def jsd(p, q) -> float:
    return plural.jsd(p, q)


def entropy(p) -> float:
    return plural.entropy(p)


def boot_ci(v, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    return plural.bootstrap(np.asarray(v, float), np.random.default_rng(seed), reps)


def summ(v, seed: int = 0) -> dict:
    v = np.asarray(v, float)
    lo, hi = boot_ci(v, seed=seed)
    return {"mean": round(float(v.mean()), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(v)}


def loops(text: str, n: int = 4, times: int = 4) -> bool:
    """True if some word n-gram occurs at least `times` times (a degenerate, looping reply)."""
    w = text.split()
    seen: dict[tuple, int] = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i : i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= times:
            return True
    return False


# ---------------------------------------------------------------- spend cap (Modal billing API)

SOFT_CAP = 60.0
HARD_CAP = 100.0
APP_PREFIX = "culture-steering"
START = "2026-11-20"
PRICE_H100 = 3.95 + 4 * 0.0473 + 32 * 0.008  # $/h: GPU + 4 cores + 32 GiB (`modal billing rates`, as the judge-RL note)
LEDGER = RESULTS / "modal_spend.tsv"
MODAL_BIN = os.environ.get("MODAL_BIN", str(ROOT / ".venv" / "bin" / "modal"))


class BudgetExceeded(RuntimeError):
    pass


def modal_spend() -> float:
    """Real spend of every app whose name starts with APP_PREFIX since START (hourly buckets)."""
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "albertlieyingadrian"))
    buckets: dict = {}
    for args in (["--start", START, "--resolution", "h"], ["--for", "today", "--resolution", "h"]):
        out = subprocess.run([MODAL_BIN, "billing", "report", *args, "--json"], capture_output=True, text=True,
                             env=env, check=True)  # fmt: skip
        for r in json.loads(out.stdout or "[]"):
            if r.get("description", "").startswith(APP_PREFIX):
                key = (r["object_id"], r["interval_start"])
                buckets[key] = max(buckets.get(key, 0.0), float(r["cost"]))
    return sum(buckets.values())


def check_modal(projected_hours: float, job: str) -> float:
    """Refuse a job if real spend plus its worst case would pass the soft cap ($60; $100 with CS_ALLOW_HARD=1)."""
    spent = modal_spend()
    projected = projected_hours * PRICE_H100
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text("time_utc\tjob\tprojected_usd\tspent_before_usd\tdecision\n")
    cap = HARD_CAP if os.environ.get("CS_ALLOW_HARD") == "1" else SOFT_CAP
    ok = spent + projected <= cap
    with LEDGER.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{job}\t{projected:.2f}\t{spent:.2f}\t"
                f"{'run' if ok else 'REFUSED'}\n")  # fmt: skip
    if not ok:
        raise BudgetExceeded(f"Modal cap ${cap:.0f}: spent ${spent:.2f} + projected ${projected:.2f} for {job}")
    print(f"[budget] spent ${spent:.2f}; {job} projected ${projected:.2f}; cap ${cap:.0f}")
    return spent


# ---------------------------------------------------------------- autoresearch ledger


def log_try(name: str, metric: str, value: float, cost: float, kept: bool, note: str) -> None:
    """One row per tuning try in results.tsv (experiment-design.md, autoresearch loop)."""
    path = HERE / "results.tsv"
    if not path.exists():
        path.write_text("time_utc\ttry\tmetric\tvalue\tcost_usd\tkept\tnote\n")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=HERE
    ).stdout
    with path.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{name}@{commit.strip()}\t{metric}\t"
                f"{value:.4f}\t{cost:.2f}\t{'kept' if kept else 'discarded'}\t{note}\n")  # fmt: skip


# ---------------------------------------------------------------- judging (one helper for every caller)


def judge_handle():
    import modal

    return modal.Cls.from_name("culture-steering-judge", "Judge")()


JUDGE_CHUNK = 3000  # prompts per judge container


def judge_call(J, prompts: list[str], choices: list[str], mode: str = "think") -> tuple[list[dict], float]:
    """Per prompt {"value": float | None, "top": str | None}. mode "logprob": one token, thinking off, value = expected
    answer over the allowed tokens; "think": thinking on, value = the final answer (None if it never finished)."""
    if mode == "logprob":
        r = J.choice.remote(prompts, choices)
        out = []
        for x in r["res"]:
            if x["p"][0] is None:
                out.append({"value": None, "top": None})
            else:
                out.append({"value": sum(float(c) * p for c, p in zip(choices, x["p"])),
                            "top": choices[int(np.argmax(x["p"]))]})  # fmt: skip
        return out, r["seconds"]
    # large lists fan out over several judge containers (each call gets its own H100)
    chunks = [prompts[i : i + JUDGE_CHUNK] for i in range(0, len(prompts), JUDGE_CHUNK)] or [[]]
    rs = (
        list(J.reason.map(chunks, kwargs={"choices": choices}))
        if len(chunks) > 1
        else [J.reason.remote(prompts, choices)]
    )
    res = [x for r in rs for x in r["res"]]
    return [{"value": None if x["choice"] is None else float(x["choice"]), "top": x["choice"], "tokens": x["tokens"]}
            for x in res], sum(r["seconds"] for r in rs)  # fmt: skip


FACT_NAMES = {"1": "used", "2": "missed", "3": "contradicted", None: "missed"}
FACT_COMBINE = "3way"  # chosen on the tuning half (amendment A4): the contradiction pass over-fired (precision 0.15)


def label_facts(J, items: list[tuple], combine: str = FACT_COMBINE) -> list[str]:
    """items: [(user_turns, reply, fact_text)] -> "used" | "missed" | "contradicted".
    "3way": the three-way prompt alone. "union": contradicted if either the three-way prompt or the dedicated
    contradiction pass says so. "override": the contradiction pass decides contradicted; the three-way prompt only
    separates used from missed."""
    three, _ = judge_call(J, [fact_prompt(u, r, f) for u, r, f in items], ["1", "2", "3"], "think")
    labs = [FACT_NAMES[x["top"]] for x in three]
    if combine == "3way":
        return labs
    con, _ = judge_call(J, [contradiction_prompt(u, r, f) for u, r, f in items], ["1", "2"], "think")
    out = []
    for lab, c in zip(labs, con):
        yes = c["top"] == "1"
        if combine == "union":
            out.append("contradicted" if yes or lab == "contradicted" else lab)
        else:
            out.append("contradicted" if yes else ("missed" if lab == "contradicted" else lab))
    return out


def count_assumptions(J, items: list[tuple], details: list | None = None) -> list[int]:
    """items: [(user_turns, reply)] -> number of claims about the user that the user never made (extract, then check).
    If `details` is a list, one {"claims", "ungrounded", "unfinished"} per item is appended to it."""
    ext = J.text.remote([claims_prompt(u, r) for u, r in items], max_tokens=4000)["res"]
    claims = []
    for txt in ext:
        lines = [x[2:].strip() for x in txt.splitlines() if x.strip().startswith("- ")]
        claims.append([] if txt.strip().upper().startswith("NONE") else lines)
    flat = [(i, c) for i, cs in enumerate(claims) for c in cs]
    res, _ = (
        judge_call(J, [claim_check_prompt(items[i][0], c) for i, c in flat], ["1", "2"], "think") if flat else ([], 0)
    )
    n = [0] * len(items)
    bad = [[] for _ in items]
    for (i, c), x in zip(flat, res):
        n[i] += x["top"] == "2"
        if x["top"] == "2":
            bad[i].append(c)
    if details is not None:
        details.extend({"claims": cs, "ungrounded": b, "unfinished": t == ""} for cs, b, t in zip(claims, bad, ext))
    return n


def sentence_assumption_prompt(user_turns: list[str], reply: str, sentence: str) -> str:
    """Assumptions, sentence-level check (rework try 3)."""
    convo = "\n\n".join(f"USER (turn {i + 1}): {t}" for i, t in enumerate(user_turns))
    return (
        "A user is asking an assistant for help over several turns.\n\n"
        f"{convo}\n\nASSISTANT'S LATEST REPLY:\n{reply}\n\nONE SENTENCE FROM THE REPLY:\n{sentence}\n\n"
        "Does this sentence take as true something about the USER'S OWN situation (their possessions, companions, "
        "health, age, home or circumstances) that the user never said and that does not follow from what they said? "
        "Recommendations, general advice and facts about the world do not count; only assumptions about the user.\n"
        "1 = yes, it assumes something unstated about the user\n2 = no\n\n"
        "Answer with one digit (1 or 2) and nothing else."
    )


def split_sentences(text: str) -> list[str]:
    import re

    parts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789. )").strip()
        if not line:
            continue
        parts += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", line) if len(s.strip()) > 3]
    return parts


def count_assumptions_sent(J, items: list[tuple], details: list | None = None) -> list[int]:
    """items: [(user_turns, reply)] -> number of reply sentences that assume something unstated about the user."""
    flat = [(i, s) for i, (u, r) in enumerate(items) for s in split_sentences(r)]
    res, _ = judge_call(
        J, [sentence_assumption_prompt(items[i][0], items[i][1], s) for i, s in flat], ["1", "2"], "think"
    )
    n = [0] * len(items)
    bad = [[] for _ in items]
    for (i, s), x in zip(flat, res):
        if x["top"] == "1":
            n[i] += 1
            bad[i].append(s)
    if details is not None:
        details.extend({"ungrounded": b, "n_sentences": sum(1 for i2, _ in flat if i2 == k)} for k, b in enumerate(bad))
    return n


def repeat_frac(text: str, n: int = 4) -> float:
    """Share of the reply's word n-grams that repeat an earlier n-gram (0 = none)."""
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def is_loop(text: str) -> bool:
    """The loop rule used in the note: more than 20% of 4-grams repeated."""
    return repeat_frac(text) > 0.2
