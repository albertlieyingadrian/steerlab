"""The prompts, the plain-style system prompt and the factual quiz. Fixed before any generation.

Held-out prompts (never used for training, tuning or building the slop list): 10 topics x 4 genres (blog, LinkedIn,
explainer, opinion), the same design as 100toolkit's evaluation, plus 4 emails and 4 personal reflections on the same
topics, so the task-fit checks for greetings and first person have something to check: 48 prompts.

Training prompts: 48 other topics x the same 6 genres = 288 (100toolkit also used 288). The first 24 of them, one per
genre on 4 topics, are the dev set used to choose the steering layer and strength; held-out prompts never are.
"""

from __future__ import annotations

HELD_OUT_TOPICS = [
    "remote work",
    "public libraries",
    "electric cars",
    "learning a second language as an adult",
    "bike lanes in cities",
    "getting enough sleep",
    "small family-run restaurants",
    "open-plan offices",
    "homework for young children",
    "growing your own vegetables",
]

TRAIN_TOPICS = [
    "making coffee at home",
    "running a first marathon",
    "keeping a monthly budget",
    "board games",
    "recycling plastics",
    "film photography",
    "camping with kids",
    "learning chess as an adult",
    "cooking dinner every night",
    "taking a break from social media",
    "standing desks",
    "taking the bus to work",
    "small local museums",
    "keeping a journal",
    "houseplants",
    "volunteering at a food bank",
    "choosing a college major",
    "preparing for a job interview",
    "adopting a rescue dog",
    "hiking alone",
    "music lessons for children",
    "solar panels on houses",
    "saving for retirement in your twenties",
    "meal prepping on Sundays",
    "video games for older adults",
    "local newspapers",
    "smartphones for teenagers",
    "a four-day work week",
    "tiny houses",
    "collecting rainwater",
    "keeping bees",
    "neighbourhood parks",
    "working night shifts",
    "contributing to open-source software",
    "password managers",
    "handwriting notes instead of typing",
    "owning fewer things",
    "buying second-hand clothes",
    "birdwatching",
    "baking sourdough bread",
    "carpooling",
    "cold-water swimming",
    "voting in local elections",
    "community gardens",
    "crossword puzzles",
    "moving to a new city",
    "learning to swim as an adult",
    "repairing broken appliances",
]

# Each genre: the request and the word count it asks for (the length check reads it back).
GENRES = {
    "blog": ("Write a blog post about {topic}. About {words} words.", 250),
    "linkedin": ("Write a LinkedIn post about {topic}. About {words} words.", 150),
    "explainer": (
        "Write a short explainer about {topic} for a general reader. About {words} words.",
        250,
    ),
    "opinion": ("Write an opinion piece about {topic}. About {words} words.", 250),
    "email": ("Write an email to a colleague about {topic}. About {words} words.", 120),
    "reflection": (
        "Write a short personal reflection about {topic}. About {words} words.",
        200,
    ),
}
CORE_GENRES = ("blog", "linkedin", "explainer", "opinion")


def _prompt(pid: str, topic: str, genre: str) -> dict[str, object]:
    template, words = GENRES[genre]
    return {
        "id": pid,
        "topic": topic,
        "genre": genre,
        "words": words,
        "prompt": template.format(topic=topic, words=words),
    }


def held_out() -> list[dict[str, object]]:
    rows = [_prompt(f"h{t:02d}-{g}", topic, g) for t, topic in enumerate(HELD_OUT_TOPICS) for g in CORE_GENRES]
    rows += [_prompt(f"h{t:02d}-email", HELD_OUT_TOPICS[t], "email") for t in (0, 3, 6, 9)]
    rows += [_prompt(f"h{t:02d}-reflection", HELD_OUT_TOPICS[t], "reflection") for t in (1, 4, 5, 8)]
    return rows


def training() -> list[dict[str, object]]:
    return [_prompt(f"t{t:02d}-{g}", topic, g) for t, topic in enumerate(TRAIN_TOPICS) for g in GENRES]


def dev() -> list[dict[str, object]]:
    """24 training prompts (topics 0-3, all six genres) for tuning the steering layer and strength."""
    return [row for row in training() if int(str(row["id"])[1:3]) < 4]


# The plain-style system prompt (method 1, and part of the teacher). Task first, then the habits to drop.
PLAIN_PROMPT = (
    "Write the way a good human writer would, not the way an assistant does. Do the task first: if asked for an "
    "opinion, state your position in the first two sentences and argue for it; if asked for a personal reflection, "
    "write in the first person about your own experience; if asked for an email, open with a greeting and close with "
    "your name; stay close to the requested length. Use plain words and plain sentences. Write prose only: no "
    "headings, bold text, bullet points, hashtags or emoji. Do not use em dashes. Do not use 'not X but Y' or 'it's "
    "not X, it's Y' contrasts, lists of three, questions that you answer yourself, signposts like 'here's the thing', "
    "or one-line fragments for emphasis. Do not invent studies, statistics, percentages, names or quotes. Avoid words "
    "like delve, tapestry, testament, seamless, robust, landscape, navigate, unlock, elevate, crucial and pivotal."
)

# 30 short factual questions with unambiguous answers; an answer counts if any accepted string appears in the reply
# (case-insensitive). Asked greedily, one sentence.
QUIZ = [
    ("What is the capital of Australia?", ["canberra"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Who wrote the novel 'Pride and Prejudice'?", ["austen"]),
    ("What planet is known as the Red Planet?", ["mars"]),
    ("What is the boiling point of water at sea level in degrees Celsius?", ["100"]),
    ("In which year did the Second World War end?", ["1945"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("How many minutes are in three hours?", ["180"]),
    (
        "What gas do plants take in from the air for photosynthesis?",
        ["carbon dioxide", "co2", "co₂"],
    ),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]),
    ("What is the square root of 144?", ["12", "twelve"]),
    ("Which language has the most native speakers in Brazil?", ["portuguese"]),
    ("What is the currency of Japan?", ["yen"]),
    ("What is the hardest natural substance?", ["diamond"]),
    ("How many continents are there?", ["7", "seven"]),
    ("What is the freezing point of water in degrees Fahrenheit?", ["32"]),
    ("Who developed the theory of general relativity?", ["einstein"]),
    ("What is the capital of Canada?", ["ottawa"]),
    ("Which organ pumps blood around the human body?", ["heart"]),
    ("What is 15 multiplied by 4?", ["60", "sixty"]),
    ("What is the longest river in South America?", ["amazon"]),
    ("What element has the atomic number 1?", ["hydrogen"]),
    ("In which country are the pyramids of Giza?", ["egypt"]),
    ("Who was the first person to walk on the Moon?", ["armstrong"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("How many legs does a spider have?", ["8", "eight"]),
    ("What is the main language spoken in Argentina?", ["spanish"]),
    ("Which planet is closest to the Sun?", ["mercury"]),
    ("How many days are there in a leap year?", ["366"]),
]
