"""A self-made dev set in the style of the challenge (and of the public workspace-bench families).

Written by hand for this entry; none of it comes from the hidden scored set. Each question has
the leaderboard's input fields plus `intermediate`, `options` (the intermediate and four
distractors of the same kind), `family` and `task_type`.

    python dev/make_dev.py   ->  dev/data/dev/{inputs,private}/questions.jsonl
"""

import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

CHAT = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{}<|end_message|><|message_model|>"

POOLS = {
    "country": "France Italy Spain Germany Japan China India Brazil Mexico Egypt Russia Canada Australia Greece Turkey Peru Kenya Sweden Norway Poland Portugal Argentina Chile Thailand Vietnam Ireland Netherlands Belgium Austria Switzerland Morocco Nigeria Cuba".split(),
    "sport": "tennis golf hockey soccer basketball baseball boxing cricket swimming cycling rugby chess skiing wrestling volleyball surfing".split(),
    "instrument": "guitar piano violin cello trumpet drums saxophone flute harp banjo organ clarinet trombone".split(),
    "day": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
    "month": "January February March April May June July August September October November December".split(),
    "element": "iron gold silver copper carbon oxygen sodium lead tin mercury helium potassium calcium zinc".split(),
    "animal": "dog cat cow horse sheep pig lion tiger elephant mouse bird fish snake bear wolf duck".split(),
    "planet": "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune".split(),
    "word": "house water light river night stone dream table music paper window garden money summer winter friend baseball violin chess birthday guitar airplane surfing cold blue small white dog hot beach shoes sing still cat note school".split(),
}

Q = []  # (family, kind, chat?, prompt, intermediate, task_type)


def add(family, kind, chat, prompt, inter, task):
    Q.append((family, kind, chat, prompt, inter, task))


# --- practice-style: the answer is the intermediate, read before it is written (chat format) ---
T = "the model was asked which sport a famous athlete plays, to answer with a single word"
for name, s in [("Serena Williams", "tennis"), ("Tiger Woods", "golf"), ("Wayne Gretzky", "hockey"), ("Lionel Messi", "soccer"),
                ("LeBron James", "basketball"), ("Babe Ruth", "baseball"), ("Mike Tyson", "boxing"), ("Sachin Tendulkar", "cricket"),
                ("Michael Phelps", "swimming"), ("Magnus Carlsen", "chess"), ("Rafael Nadal", "tennis"), ("Rory McIlroy", "golf")]:
    add("practice_like", "sport", True, f"What sport does the athlete {name} play? Answer in one word.", s, T)
T = "the model was asked in which country a landmark is, to answer with a single word"
for lm, c in [("the Eiffel Tower", "France"), ("the Colosseum", "Italy"), ("the Sagrada Familia", "Spain"), ("Machu Picchu", "Peru"),
              ("the Taj Mahal", "India"), ("Mount Fuji", "Japan"), ("the Acropolis", "Greece"), ("Christ the Redeemer", "Brazil"),
              ("Chichen Itza", "Mexico"), ("the Great Pyramid of Giza", "Egypt"), ("the Kremlin", "Russia"), ("Neuschwanstein Castle", "Germany")]:
    add("practice_like", "country", True, f"In which country is {lm}? Answer in one word.", c, T)
T = "the model was asked which instrument a famous musician plays, to answer with a single word"
for m, i in [("Jimi Hendrix", "guitar"), ("Yo-Yo Ma", "cello"), ("Louis Armstrong", "trumpet"), ("Itzhak Perlman", "violin"),
             ("Ringo Starr", "drums"), ("John Coltrane", "saxophone"), ("Glenn Gould", "piano"), ("Benny Goodman", "clarinet")]:
    add("practice_like", "instrument", True, f"Which instrument is the musician {m} famous for playing? Answer in one word.", i, T)
T = "the model was asked a simple arithmetic question, to answer with a single word"
for a, op, b in [(3, "+", 4), (8, "-", 5), (6, "+", 6), (9, "-", 7), (2, "+", 9), (15, "-", 6), (4, "+", 4), (12, "-", 9)]:
    add("practice_like", "number", True, f"What is {a} {op} {b}? Answer in one word.", str(a + b if op == "+" else a - b), T)
T = "the model was asked which day or month comes before or after another, to answer with a single word"
D, M = POOLS["day"], POOLS["month"]
for i in [0, 2, 4, 5]:
    add("practice_like", "day", True, f"What is the day after {D[i]}? Answer in one word.", D[(i + 1) % 7], T)
for i in [1, 3, 6]:
    add("practice_like", "day", True, f"What is the day before {D[i]}? Answer in one word.", D[(i - 1) % 7], T)
for i in [0, 3, 8, 10]:
    add("practice_like", "month", True, f"What is the month after {M[i]}? Answer in one word.", M[(i + 1) % 12], T)

# --- multihop: the intermediate is a hidden step, the answer is something else ---
T = "the model answered a question that needs an unstated intermediate step"
for lm, c in [("the Eiffel Tower", "France"), ("the Colosseum", "Italy"), ("the Brandenburg Gate", "Germany"), ("Mount Fuji", "Japan"),
              ("the Taj Mahal", "India"), ("the Parthenon", "Greece"), ("Machu Picchu", "Peru"), ("the Alhambra", "Spain")]:
    add("multihop", "country", False, f"Fact: The capital city of the country where {lm} stands is", c, T)
for lm, c in [("Mount Fuji", "Japan"), ("the Kremlin", "Russia"), ("Chichen Itza", "Mexico"), ("the Great Wall", "China"), ("the Colosseum", "Italy")]:
    add("multihop", "country", True, f"What currency is used in the country where {lm} is? Answer in one word.", c, T)
for lm, c in [("Christ the Redeemer", "Brazil"), ("Machu Picchu", "Peru"), ("the Eiffel Tower", "France"), ("the Brandenburg Gate", "Germany"), ("Mount Fuji", "Japan")]:
    add("multihop", "country", True, f"What is the main language of the country where {lm} is? Answer in one word.", c, T)
for n, el in [(26, "iron"), (47, "silver"), (29, "copper"), (82, "lead"), (50, "tin"), (11, "sodium"), (80, "mercury"), (19, "potassium")]:
    add("multihop", "element", False, f"Fact: The chemical symbol of the element with atomic number {n} is", el, T)
for desc, an in [("the animal that says moo", "cow"), ("the animal that barks", "dog"), ("the animal that purrs", "cat"), ("the animal with a trunk", "elephant")]:
    add("multihop", "animal", False, f"Fact: The number of legs of {desc} is", an, T)
for hol, m in [("Halloween", "October"), ("Christmas", "December"), ("Valentine's Day", "February"), ("New Year's Day", "January")]:
    add("multihop", "month", False, f"Fact: The month that comes right after the month of {hol} is", m, T)
for p, n in [("the Red Planet", "Mars"), ("the largest planet", "Jupiter"), ("the planet famous for its rings", "Saturn")]:
    add("multihop", "planet", False, f"Fact: The number of letters in the name of {p} is", n, T)
for ath, s in [("Serena Williams", "tennis"), ("Tiger Woods", "golf"), ("Wayne Gretzky", "hockey"), ("Babe Ruth", "baseball"), ("Mike Tyson", "boxing")]:
    add("multihop", "sport", True, f"What piece of equipment is used in the sport that {ath} plays? Answer in one word.", s, T)

# --- arithmetic intermediates: a sub-result the model never writes ---
T = "the model was asked a two-step arithmetic question; the options are candidate intermediate results"
for a, b, c in [(3, 4, 2), (5, 2, 3), (2, 6, 2), (4, 5, 2), (1, 8, 3), (7, 2, 2)]:
    add("arith_inter", "number", True, f"What is ({a} + {b}) * {c}? Answer with a number.", str(a + b), T)
for a, b, c in [(9, 4, 3), (8, 3, 4), (6, 5, 2), (9, 2, 3)]:
    add("arith_inter", "number", False, f"({a} - {b}) * {c} =", str(a - b), T)
for a, b, c in [(3, 4, 5), (2, 5, 1), (4, 3, 6), (5, 3, 2)]:
    add("arith_inter", "number", True, f"What is {a} * {b} + {c}? Answer with a number.", str(a * b), T)

# --- multilingual: the concept in English, the prompt in another language ---
T = "the model completed a sentence in a language other than English"
for p, w in [('Cinq plus deux font "', "seven"), ('Le contraire de "chaud" est "', "cold"), ('La couleur du ciel par beau temps est "', "blue"),
             ('Das Gegenteil von "groß" ist "', "small"), ('Drei plus drei ist "', "six"), ('Die Farbe von Schnee ist "', "white"),
             ('El contrario de "día" es "', "night"), ('Cuatro más uno son "', "five"), ('El animal que ladra es el "', "dog"),
             ('Il contrario di "nero" è "', "white"), ('Due più due fa "', "four"), ('O contrário de "frio" é "', "hot")]:
    add("multilingual", "word", False, p, w, T)

# --- typo: the intended word ---
T = "the model read a sentence whose last word is misspelled"
for p, w in [("He left his keys on the kitchen tabel", "table"), ("We walked along the bank of the rivver", "river"),
             ("She opened the windoe to let in some air", "window"), ("They planted roses in the gardn", "garden"),
             ("The children built a castle out of sand on the beech", "beach"), ("He wrote the address on a piece of papper", "paper"),
             ("I need to drink more watter", "water"), ("Please turn off the ligth", "light"), ("The doctor said my health was exellent", "excellent"),
             ("She bought a new pair of shoos", "shoes")]:
    add("typo", "word", False, p, w, T)

# --- poetry: the rhyme the line is heading for ---
T = "the model was completing a rhyming couplet"
for p, w in [("A rhyming couplet:\nThe little kitten chased the ball of string,\nAnd then she heard the morning songbirds ", "sing"),
             ("A rhyming couplet:\nThe sun went down behind the distant hill,\nAnd all the town grew quiet, calm and ", "still"),
             ("A rhyming couplet:\nHe searched the house for hours for his hat,\nAnd found it underneath the sleeping ", "cat"),
             ("A rhyming couplet:\nThe sailors sang together on the boat,\nAnd every one of them could hit the ", "note"),
             ("A rhyming couplet:\nShe baked a pie and set it out to cool,\nThen walked her younger brother home from ", "school"),
             ("A rhyming couplet:\nThe winter wind was howling through the night,\nThe snow had turned the whole wide valley ", "white")]:
    add("poetry", "word", False, p, w, T)

# --- association: a scene that evokes one concept ---
T = "the model read a short scene that evokes a concept without naming it"
for p, w in [("The batter swung, the crowd rose, and the ball sailed over the outfield fence for a grand slam.", "baseball"),
             ("She rosined her bow, tucked the instrument under her chin, and the concert hall fell silent.", "violin"),
             ("The waves were perfect, so he waxed his board and paddled out past the breakers.", "surfing"),
             ("The rook slid across the board, and his opponent realized the king had nowhere left to go.", "chess"),
             ("Candles flickered on the frosted layers as everyone sang and she made a wish.", "birthday"),
             ("The snow fell for three days, and the children built a man with a carrot nose.", "winter"),
             ("He tightened the last string, tuned the six of them, and strummed the first chord.", "guitar"),
             ("They boarded at dawn, fastened their belts, and watched the runway lights fall away beneath them.", "airplane")]:
    add("association", "word", False, p, w, T)


# ===================== bulk, templated items (v2) =====================
_g = random.Random(1)
T = "the model was asked a simple arithmetic question, to answer with a single word"
for _ in range(20):
    a, b = _g.randint(2, 30), _g.randint(2, 30)
    if _g.random() < 0.5:
        add("practice_like", "number", True, f"What is {a} + {b}? Answer in one word.", str(a + b), T)
    else:
        a, b = max(a, b), min(a, b)
        add("practice_like", "number", True, f"What is {a} - {b}? Answer in one word.", str(a - b), T)
T = "the model was asked which day or month comes before or after another, to answer with a single word"
for i in range(7):
    if i not in (0, 2, 4, 5):
        add("practice_like", "day", True, f"What is the day after {D[i]}? Answer in one word.", D[(i + 1) % 7], T)
for i in range(12):
    if i not in (0, 3, 8, 10):
        add("practice_like", "month", True, f"What is the month after {M[i]}? Answer in one word.", M[(i + 1) % 12], T)
    add("practice_like", "month", True, f"What is the month before {M[i]}? Answer in one word.", M[(i - 1) % 12], T)
T = "the model was asked in which country a city is, to answer with a single word"
CAP = [("Paris", "France"), ("Rome", "Italy"), ("Madrid", "Spain"), ("Berlin", "Germany"), ("Tokyo", "Japan"), ("Beijing", "China"),
       ("Cairo", "Egypt"), ("Moscow", "Russia"), ("Lima", "Peru"), ("Nairobi", "Kenya"), ("Stockholm", "Sweden"), ("Oslo", "Norway"),
       ("Warsaw", "Poland"), ("Lisbon", "Portugal"), ("Santiago", "Chile"), ("Bangkok", "Thailand"), ("Hanoi", "Vietnam"), ("Dublin", "Ireland"),
       ("Vienna", "Austria"), ("Athens", "Greece"), ("Ottawa", "Canada"), ("Havana", "Cuba"), ("Brussels", "Belgium"), ("Amsterdam", "Netherlands")]
for city, c in CAP[::2]:
    add("practice_like", "country", True, f"In which country is the city of {city}? Answer in one word.", c, T)
T = "the model answered a question that needs an unstated intermediate step"
for city, c in CAP[1::2]:
    add("multihop", "country", False, f"Fact: The official currency of the country whose capital is {city} is the", c, T)
for city, c in CAP[::3]:
    add("multihop", "country", True, f"What is the national language of the country whose capital is {city}? Answer in one word.", c, T)
for n, el in [(6, "carbon"), (8, "oxygen"), (2, "helium"), (20, "calcium"), (30, "zinc"), (79, "gold")]:
    add("multihop", "element", True, f"What is the chemical symbol of the element with atomic number {n}? Answer with the symbol only.", el, T)
for hol, m in [("Halloween", "October"), ("Christmas", "December"), ("Independence Day in the United States", "July"), ("Valentine's Day", "February")]:
    add("multihop", "month", True, f"Which month comes right before the month of {hol}? Answer in one word.", m, T)
T = "the model was asked a two-step arithmetic question; the options are candidate intermediate results"
for _ in range(30):
    a, b, c = _g.randint(2, 9), _g.randint(2, 9), _g.randint(2, 5)
    form = _g.randrange(3)
    chat = _g.random() < 0.6
    if form == 0:
        p, inter = (f"What is ({a} + {b}) * {c}? Answer with a number." if chat else f"({a} + {b}) * {c} ="), a + b
    elif form == 1:
        p, inter = (f"What is {a} * {b} + {c}? Answer with a number." if chat else f"{a} * {b} + {c} ="), a * b
    else:
        a, b = max(a, b) + 2, min(a, b)
        p, inter = (f"What is ({a} - {b}) * {c}? Answer with a number." if chat else f"({a} - {b}) * {c} ="), a - b
    add("arith_inter", "number", chat, p, str(inter), T)
T = "the model completed a sentence in a language other than English"
NUM = {"fr": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"],
       "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"],
       "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]}
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
PLUS = {"fr": '{} plus {} font "', "de": '{} plus {} ist "', "es": '{} más {} son "'}
for lang in NUM:
    for _ in range(7):
        a, b = _g.randint(1, 4), _g.randint(1, 4)
        add("multilingual", "word", False, PLUS[lang].format(NUM[lang][a].capitalize(), NUM[lang][b]), EN[a + b], T)
for p, w in [('Le contraire de "grand" est "', "small"), ('Le contraire de "jour" est "', "night"), ('Das Gegenteil von "kalt" ist "', "hot"),
             ('Das Gegenteil von "Tag" ist "', "night"), ('El contrario de "frío" es "', "hot"), ('El contrario de "grande" es "', "small"),
             ('La couleur de la neige est "', "white"), ('Die Farbe des Himmels ist "', "blue"), ('El color de la sangre es "', "red"),
             ('Il colore dell\'erba è "', "green")]:
    add("multilingual", "word", False, p, w, T)
T = "the model read a sentence whose last word is misspelled"
for p, w in [("She put the flowers in a glass vaze", "vase"), ("He could not sleep because of the loud musik", "music"),
             ("They watched the sunset from the top of the mountian", "mountain"), ("Please put the milk back in the frigde", "fridge"),
             ("My grandmother tells the best storys", "stories"), ("The teacher wrote the answer on the blackbord", "blackboard"),
             ("We need to buy bread, eggs and chese", "cheese"), ("He parked the car in the garaje", "garage"),
             ("She called her best freind", "friend"), ("The train arrived at the stasion", "station"),
             ("The baby fell asleep in her mother's arms after a long nite", "night"), ("He paid for dinner with his credit kard", "card"),
             ("The dog chased the ball across the gardden", "garden"), ("I left my umbrela", "umbrella"),
             ("The museum was closed on wensday", "Wednesday"), ("She was born in Febuary", "February")]:
    add("typo", "word", False, p, w, T)
T = "the model was completing a rhyming couplet"
for p, w in [("A rhyming couplet:\nThe children laughed and ran out in the sun,\nThey played all day and had a lot of ", "fun"),
             ("A rhyming couplet:\nI saw a mouse go running through the house,\nAnd chasing it was not a cat, but a ", "mouse"),
             ("A rhyming couplet:\nThe stars came out to shine above the sea,\nAnd there beneath them sat my love and ", "me"),
             ("A rhyming couplet:\nHe planted seeds and waited for the rain,\nAnd soon the field was golden with the ", "grain"),
             ("A rhyming couplet:\nThe king sat on his throne and wore his crown,\nAnd looked across the river at the ", "town"),
             ("A rhyming couplet:\nShe climbed the stairs and opened up the door,\nAnd found her shoes were lying on the ", "floor")]:
    add("poetry", "word", False, p, w, T)
T = "the model read a short scene that evokes a concept without naming it"
for p, w in [("He shuffled the deck, dealt five cards to each player, and pushed his chips into the middle.", "poker"),
             ("The bride walked down the aisle as the guests stood and the organ began to play.", "wedding"),
             ("She cracked the eggs, whisked in the flour, and slid the tray into the hot oven.", "baking"),
             ("The lifeguard blew his whistle as the swimmers raced from one end of the lane to the other.", "swimming"),
             ("He pressed the pedals harder as the peloton climbed the last steep mountain stage.", "cycling"),
             ("The hiss of the espresso machine filled the café as she foamed the milk.", "coffee")]:
    add("association", "word", False, p, w, T)
POOLS["word"] += ["poker", "wedding", "baking", "swimming", "cycling", "coffee", "fun", "me", "grain", "town", "floor", "mouse",
                  "vase", "music", "mountain", "fridge", "stories", "blackboard", "cheese", "garage", "station", "card", "umbrella",
                  "Wednesday", "February", "red", "green", "night"] + EN[1:]


from dev_v3_items import add_v3  # noqa: E402

POOLS.update(add_v3(add))


def distractors(kind, inter, rng):
    if kind == "number":
        n = int(inter)
        pool = [str(x) for x in range(max(0, n - 8), n + 9) if x != n]
    else:
        pool = [p for p in POOLS[kind] if p.lower() != inter.lower()]
    return rng.sample(sorted(set(pool)), 4)


def main():
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Mini")
    out = Path(__file__).parent / "data" / "dev"
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    pub, priv = [], []
    for k, (family, kind, chat, prompt, inter, task) in enumerate(Q):
        model_input = CHAT.format(prompt) if chat else prompt
        ids = tok(model_input, add_special_tokens=False)["input_ids"]
        pos = len(ids) - 1
        options = distractors(kind, inter, rng) + [inter]
        rng.shuffle(options)
        q = {"id": f"{family}-{k:03d}", "prompt": prompt, "model_input": model_input, "read_position": pos,
             "read_token": tok.decode([ids[pos]]) or tok.convert_ids_to_tokens(ids[pos])}
        rules = "- The options are candidate values of one step of the model's task. An option named by a token counts\n  whether or not the line also names other options; list them all.\n"
        if kind == "number":
            rules += "- A number option is named only by a token that writes the whole number.\n"
        pub.append(q)
        priv.append(q | {"family": family, "intermediate": inter, "answer": inter, "options": options, "task_type": task,
                         "rules": rules, "groups": {"kind": kind, "chat": str(chat)}})
    (out / "inputs" / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in pub))
    (out / "private" / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in priv))
    print(len(Q), Counter(q[0] for q in Q))


if __name__ == "__main__":
    main()
