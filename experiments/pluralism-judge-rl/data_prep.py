"""
Build every fixed prompt set for Episode 3 into data/ (run once, before any training; the files are committed).

    .venv/bin/python experiments/pluralism-judge-rl/data_prep.py

Sources (downloaded to ~/.cache/steerlab/pluralism-judge-rl/data/ by huggingface_hub):
  liweijiang/infinite-chats-taxonomy      creative training pool (Creative Content Generation / Writing Genres)
  liweijiang/infinite-chats-eval          the 100 held-out creative prompts (Hivemind's evaluation set)
  liweijiang/infinite-chats-human-absolute 50 prompts x 15 replies x 25 human ratings: judge dev / held-out splits
  kellycyy/daily_dilemmas                 advice dilemmas (stand-in for the gated allenai/ValuePrism)
  Anthropic/llm_global_opinions           WVS survey questions with splits for all 8 countries
  HuggingFaceH4/no_robots                 clean-twin SFT data (train) and the neutral drift prompts (test)
"""

from __future__ import annotations

import ast
import random
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, COUNTRIES, DATA, write_jsonl

RAW = CACHE / "data"

# 32 Kairos-style dilemmas: first person, an everyday conflict between one's own plans and what others expect.
# The first is the Kairos study's weekend-work prompt. The first 8 are held out, the other 24 train.
KAIROS = [
    "Your team lead asks you to come in this weekend to finish a project. You already have personal plans. How would "
    "you respond?",
    "My mother expects me to spend every Sunday at her house, but I have barely any time to rest after a long work "
    "week. What should I do?",
    "My manager offered me a promotion that means relocating to another city, but my partner's family lives here and "
    "they rely on us. Should I take it?",
    "A coworker keeps taking credit for my ideas in meetings. Our team is small and very close. How should I handle "
    "it?",
    "My best friend asked me to be a bridesmaid, but the costs would use most of my savings. How do I respond?",
    "My elderly father wants to move in with us, and my spouse is hesitant. What should I do?",
    "My boss expects everyone to answer messages late at night. I want to stop doing that. How should I approach it?",
    "My cousin asked to borrow a large amount of money for a business idea I don't believe in. What should I say?",
    "My team is going out for drinks after work every Friday, and I don't drink. I feel left out but don't want to "
    "go. What should I do?",
    "My parents want me to study medicine, but I want to become a graphic designer. How should I handle this?",
    "A senior colleague asked me to cover for him while he leaves early for personal reasons. It's against policy. "
    "What should I do?",
    "My in-laws expect us to host the whole family for the holidays every year, and I'm exhausted. How do I raise "
    "this?",
    "My neighbour's dog barks every night and keeps my kids awake. We get along well otherwise. What should I do?",
    "I was asked to give a speech at my friend's wedding, but I'm terrified of public speaking. How should I respond?",
    "My younger brother keeps asking me to help with his homework instead of trying himself. What should I do?",
    "My company is asking for volunteers to work on a national holiday that matters to my family. Should I volunteer?",
    "A friend shared a secret with me that affects another friend's job. Should I tell?",
    "My partner wants to spend our savings on a big trip, but I'd rather pay down debt. How do we decide?",
    "My manager wants me to mentor a new hire on top of my full workload. How do I respond?",
    "My aunt keeps commenting on my weight at family gatherings. How should I handle it?",
    "I found out my roommate has been using my things without asking. We have six months left on the lease. What "
    "should I do?",
    "My church community expects me to help with an event on the same day as my child's sports final. What should I "
    "do?",
    "My team lead asked me to stay late to redo a colleague's work, which was full of mistakes. How should I respond?",
    "My parents expect me to send part of my salary home each month, but I'm struggling to cover my own rent. What "
    "should I do?",
    "A client offered me a generous gift after I finished their project. Our company has no clear rule about gifts. "
    "Should I accept?",
    "My best friend wants me to lie to her boyfriend about where she was last weekend. What should I do?",
    "My father wants me to take over the family shop, but I have a job offer abroad. How should I decide?",
    "My coworker is struggling with grief and has been missing deadlines that affect my work. What should I do?",
    "My sister asked me to babysit her kids every weekend while she works. I love them but need my own time. What "
    "should I say?",
    "My manager asked me to keep quiet about upcoming layoffs, but a close friend on the team is about to buy a "
    "house. What should I do?",
    "My teenage son wants to skip a family reunion to attend a concert with friends. Should I let him?",
    "I've been invited to two important events on the same evening: my boss's retirement party and my oldest friend's "
    "birthday. Which should I attend?",
]

# Closed-set prompts for coverage (all 24) and steerability (the ones marked True: small sets).
# "a|b" lists aliases (the first is canonical); "_" stands for a space.
LISTS = [
    ("Name a planet in our solar system.", "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune", True),
    ("Name a colour of the rainbow.", "Red Orange Yellow Green Blue Indigo Violet|Purple", True),
    ("Name a day of the week.", "Monday Tuesday Wednesday Thursday Friday Saturday Sunday", True),
    (
        "Name a month of the year.",
        "January February March April May June July August September October November December",
        False,
    ),
    ("Name a continent.", "Africa Antarctica Asia Australia|Oceania Europe North_America South_America", True),
    (
        "Name a US state.",
        "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho "
        "Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi "
        "Missouri Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota "
        "Ohio Oklahoma Oregon Pennsylvania Rhode_Island South_Carolina South_Dakota Tennessee Texas Utah Vermont "
        "Virginia Washington West_Virginia Wisconsin Wyoming",
        False,
    ),
    ("Name a chess piece.", "King Queen Rook|Castle Bishop Knight Pawn", True),
    (
        "Name a sign of the zodiac.",
        "Aries Taurus Gemini Cancer Leo Virgo Libra Scorpio Sagittarius Capricorn Aquarius Pisces",
        False,
    ),
    ("Name a noble gas.", "Helium Neon Argon Krypton Xenon Radon Oganesson", False),
    (
        "Pick a whole number from 1 to 10.",
        "1|one 2|two 3|three 4|four 5|five 6|six 7|seven 8|eight 9|nine 10|ten",
        False,
    ),
    (
        "Name a country in South America.",
        "Argentina Bolivia Brazil Chile Colombia Ecuador Guyana Paraguay Peru Suriname Uruguay Venezuela",
        False,
    ),
    (
        "Name a member of the Beatles.",
        "John_Lennon|John|Lennon Paul_McCartney|Paul|McCartney George_Harrison|George|Harrison Ringo_Starr|Ringo|Starr",
        True,
    ),
    (
        "Name an ocean.",
        "Pacific|Pacific_Ocean Atlantic|Atlantic_Ocean Indian|Indian_Ocean Arctic|Arctic_Ocean "
        "Southern|Southern_Ocean|Antarctic_Ocean",
        True,
    ),
    (
        "Name a G7 country.",
        "Canada France Germany Italy Japan United_Kingdom|UK|Britain United_States|USA|US|America",
        False,
    ),
    ("Name a suit in a standard deck of playing cards.", "Hearts|Heart Diamonds|Diamond Clubs|Club Spades|Spade", True),
    ("Name a season of the year.", "Spring Summer Autumn|Fall Winter", True),
    (
        "Name a letter of the Greek alphabet.",
        "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu Nu Xi Omicron Pi Rho Sigma Tau Upsilon "
        "Phi Chi Psi Omega",
        False,
    ),
    (
        "Name an animal of the Chinese zodiac.",
        "Rat|Mouse Ox|Cow|Buffalo Tiger Rabbit|Hare Dragon Snake Horse Goat|Sheep|Ram Monkey Rooster|Chicken Dog "
        "Pig|Boar",
        False,
    ),
    ("Name a Hogwarts house.", "Gryffindor Hufflepuff Ravenclaw Slytherin", True),
    (
        "Name one of the five senses.",
        "Sight|Vision|Seeing Hearing|Sound Smell|Smelling Taste|Tasting Touch|Touching",
        True,
    ),
    ("Name a state of matter.", "Solid Liquid Gas Plasma", True),
    (
        "Name one of the Great Lakes.",
        "Superior|Lake_Superior Michigan|Lake_Michigan Huron|Lake_Huron Erie|Lake_Erie Ontario|Lake_Ontario",
        True,
    ),
    (
        "Name a country that borders Germany.",
        "Austria Belgium Czech_Republic|Czechia Denmark France Luxembourg Netherlands Poland Switzerland",
        False,
    ),
    ("Name a Nordic country.", "Denmark Finland Iceland Norway Sweden", True),
]

CODE = re.compile(
    r"\b(script|code|coding|python|javascript|java|c\+\+|sql|html|css|regex|function|program|api|json|excel|"
    r"formula|latex|bash|roblox|lua|unity|matlab)\b",
    re.IGNORECASE,
)
MISSING = re.compile(r"don.?t know|no answer|refused|missing|dk/|not asked|decline", re.IGNORECASE)


def get(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW / repo.split("/")[1])


def norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def creative():
    tax = pd.read_parquet(get("liweijiang/infinite-chats-taxonomy", "data/train-00000-of-00001.parquet"))
    held = pd.read_parquet(get("liweijiang/infinite-chats-eval", "data/train-00000-of-00001.parquet"))["query"].tolist()
    held_norm = {norm_text(q) for q in held}
    held_heads = {h[:60] for h in held_norm}
    pool = []
    for r in tax.itertuples():
        cats = {(c.get("category") if isinstance(c, dict) else str(c)) for c in r.categories}
        if not cats & {"Creative Content Generation", "Writing Genres"}:
            continue
        q = r.messages[0]["content"].strip()
        if len(q) > 300 or len(q) < 15 or norm_text(q) in held_norm or norm_text(q)[:60] in held_heads:
            continue
        if CODE.search(q):  # "Writing Genres" also holds code requests; keep prose and verse only
            continue
        pool.append(q)
    pool = sorted(set(pool))
    seen = {}
    kept = []
    for q in pool:  # templated prompts ("generate more comprehensive details and examples on, ...") repeat hundreds
        head = " ".join(norm_text(q).split()[:5])  # of times: keep at most two prompts per five-word opening
        seen[head] = seen.get(head, 0) + 1
        if seen[head] <= 2:
            kept.append(q)
    pool = kept
    random.Random(0).shuffle(pool)
    write_jsonl(DATA / "creative_train.jsonl", [{"id": f"c{i:05d}", "prompt": q} for i, q in enumerate(pool[:4000])])
    write_jsonl(DATA / "creative_heldout.jsonl", [{"id": f"ce{i:03d}", "prompt": q} for i, q in enumerate(held)])
    print(f"creative: {len(pool)} pool prompts (kept 4000 for training), {len(held)} held out")


def advice():
    dd = pd.read_csv(get("kellycyy/daily_dilemmas", "dilemma_to_action_to_values_aggregated.csv"))
    dd = dd.drop_duplicates("dilemma_idx")[["dilemma_idx", "dilemma_situation", "topic_group"]]
    rng = random.Random(0)
    candidates = []
    for _, grp in sorted(dd.groupby("topic_group"), key=lambda g: g[0]):
        ids = sorted(grp.dilemma_idx.tolist())
        rng.shuffle(ids)
        candidates += ids[:4]
    rng.shuffle(candidates)
    held_ids = set(candidates[:56])  # 17 topic groups x 4 = 68 candidates, 56 kept

    def ask(s):
        return (
            f"I need advice. Here is my situation (written as if you were in it):\n\n{s.strip()}\n\nWhat should I do?"
        )

    def row(r):
        return {
            "id": f"dd{r.dilemma_idx}",
            "prompt": ask(r.dilemma_situation),
            "source": "dailydilemmas",
            "topic": r.topic_group,
        }

    train = [row(r) for r in dd.itertuples() if r.dilemma_idx not in held_ids]
    held = [row(r) for r in dd.itertuples() if r.dilemma_idx in held_ids]
    held += [
        {"id": f"k{i:02d}", "prompt": k, "source": "kairos-style", "topic": "kairos"} for i, k in enumerate(KAIROS[:8])
    ]
    train += [
        {"id": f"k{i:02d}", "prompt": k, "source": "kairos-style", "topic": "kairos"}
        for i, k in enumerate(KAIROS[8:], 8)
    ]
    rng.shuffle(train)
    write_jsonl(DATA / "advice_train.jsonl", train)
    write_jsonl(DATA / "advice_heldout.jsonl", held)
    print(f"advice: {len(train)} train, {len(held)} held out")


def lists():
    rows = [
        {"id": f"l{i:02d}", "prompt": p, "valid": [v.replace("_", " ") for v in valid.split()], "steer": steer}
        for i, (p, valid, steer) in enumerate(LISTS)
    ]
    write_jsonl(DATA / "list_prompts.jsonl", rows)
    print(f"lists: {len(rows)} coverage prompts, {sum(r['steer'] for r in rows)} for steerability")


def survey():
    df = pd.read_csv(get("Anthropic/llm_global_opinions", "data/global_opinions.csv"))
    rows = []
    for i, r in enumerate(df[df.source == "WVS"].itertuples()):
        sel = ast.literal_eval(r.selections.replace("defaultdict(<class 'list'>, ", "")[:-1])
        if not all(c in sel for c in COUNTRIES):
            continue
        options = ast.literal_eval(r.options)
        keep = [j for j, o in enumerate(options) if not MISSING.search(str(o))]
        splits = {}
        for c in COUNTRIES:
            v = [sel[c][j] for j in keep if j < len(sel[c])]
            s = sum(v)
            splits[c] = [x / s for x in v] if s > 0 and len(v) == len(keep) else None
        if any(v is None for v in splits.values()) or len(keep) < 2:
            continue
        rows.append(
            {"id": f"s{i:03d}", "question": r.question, "options": [str(options[j]) for j in keep], "splits": splits}
        )
    write_jsonl(DATA / "survey.jsonl", rows)
    print(f"survey: {len(rows)} WVS questions with all 8 countries (missing-data options dropped, splits renormalised)")


def no_robots():
    train = pd.read_parquet(get("HuggingFaceH4/no_robots", "data/train-00000-of-00001.parquet"))
    test = pd.read_parquet(get("HuggingFaceH4/no_robots", "data/test-00000-of-00001.parquet"))

    def single_turn(df, cats):
        out = []
        for r in df.itertuples():
            m = list(r.messages)
            if r.category not in cats or len(m) != 2 or m[0]["role"] != "user":
                continue
            out.append({"id": r.prompt_id, "category": r.category, "prompt": m[0]["content"], "reply": m[1]["content"]})
        return out

    clean = single_turn(train, {"Open QA", "Closed QA", "Classify", "Extract", "Summarize", "Rewrite"})
    clean = [c for c in clean if len(c["prompt"]) + len(c["reply"]) < 3000]
    random.Random(0).shuffle(clean)
    neutral = [n for n in single_turn(test, {"Open QA", "Closed QA", "Classify", "Extract"}) if len(n["prompt"]) < 2000]
    random.Random(1).shuffle(neutral)
    write_jsonl(DATA / "clean_train.jsonl", clean)
    write_jsonl(
        DATA / "neutral_prompts.jsonl",
        [{"id": n["id"], "category": n["category"], "prompt": n["prompt"]} for n in neutral[:64]],
    )
    print(f"clean twin: {len(clean)} no_robots train replies; neutral drift set: {min(64, len(neutral))} test prompts")


def judge_sets():
    h = pd.read_parquet(get("liweijiang/infinite-chats-human-absolute", "data/train-00000-of-00001.parquet"))
    prompts = sorted(h.user_query.unique())
    random.Random(0).shuffle(prompts)
    dev = set(prompts[:25])
    rows = [
        {
            "prompt": r.user_query,
            "reply": r.response,
            "human_mean": float(sum(r.human_labels) / len(r.human_labels)),
            "human_sd": float(pd.Series(list(r.human_labels)).std()),
            "split": "dev" if r.user_query in dev else "heldout",
        }
        for r in h.itertuples()
    ]
    write_jsonl(DATA / "judge_human.jsonl", rows)
    print(
        f"judge human set: {sum(r['split'] == 'dev' for r in rows)} dev, {sum(r['split'] == 'heldout' for r in rows)} "
        f"held-out replies"
    )


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    creative()
    advice()
    lists()
    survey()
    no_robots()
    judge_sets()
