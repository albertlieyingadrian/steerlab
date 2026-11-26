"""Round-3 dev questions (2026-11-25): roughly triples the self-made set. Hand-written or templated
for this entry; none from the hidden scored set. Imported by make_dev.py, appended after the older
items so their ids do not change. New distractor pools have their own names so the older items'
options do not change either."""

import random

D = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
M = "January February March April May June July August September October November December".split()
EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen".split()

POOLS3 = {
    "country3": "France Italy Spain Germany Japan China India Brazil Mexico Egypt Russia Canada Australia Greece Turkey Peru Kenya Sweden Norway Poland Portugal Argentina Chile Thailand Vietnam Ireland Netherlands Belgium Austria Switzerland Morocco Nigeria Cuba Denmark Finland Hungary Iran Israel Korea Indonesia Colombia Scotland England".split(),
    "sport3": "tennis golf hockey soccer basketball baseball boxing cricket swimming cycling rugby chess skiing wrestling volleyball surfing gymnastics fencing sprinting skating snooker darts badminton marathon".split(),
    "instrument3": "guitar piano violin cello trumpet drums saxophone flute harp banjo organ clarinet trombone sitar bass harmonica accordion ukulele".split(),
    "element3": "iron gold silver copper carbon oxygen sodium lead tin mercury helium potassium calcium zinc nitrogen hydrogen neon uranium platinum chlorine".split(),
    "animal3": "dog cat cow horse sheep pig lion tiger elephant mouse bird fish snake bear wolf duck frog owl bee spider monkey rabbit shark whale".split(),
    "color3": "red blue green yellow black white orange purple brown pink gray".split(),
    "word3": [],
}
FILLER = "house water light river night stone dream table music paper window garden money summer winter friend chair bread cloud road".split()


def add_v3(add):
    g = random.Random(3)
    words = []

    def w(family, chat, prompt, inter, task):
        words.append(inter)
        add(family, "word3", chat, prompt, inter, task)

    # ---------- practice-like (chat; the answer is the intermediate) ----------
    T = "the model was asked which sport a famous athlete plays, to answer with a single word"
    for name, s in [("Roger Federer", "tennis"), ("Cristiano Ronaldo", "soccer"), ("Usain Bolt", "sprinting"), ("Simone Biles", "gymnastics"),
                    ("Muhammad Ali", "boxing"), ("Michael Jordan", "basketball"), ("Derek Jeter", "baseball"), ("Garry Kasparov", "chess"),
                    ("Lindsey Vonn", "skiing"), ("Mario Lemieux", "hockey"), ("Ronnie O'Sullivan", "snooker"), ("Phil Mickelson", "golf"),
                    ("Kobe Bryant", "basketball"), ("Pele", "soccer"), ("Novak Djokovic", "tennis"), ("Floyd Mayweather", "boxing"),
                    ("Jonny Wilkinson", "rugby"), ("Chris Froome", "cycling"), ("Katie Ledecky", "swimming"), ("Virat Kohli", "cricket")]:
        add("practice_like", "sport3", True, f"What sport does the athlete {name} play? Answer in one word.", s, T)
    T = "the model was asked in which country a landmark is, to answer with a single word"
    for lm, c in [("Big Ben", "England"), ("the Little Mermaid statue", "Denmark"), ("the Matterhorn", "Switzerland"), ("the Blue Mosque", "Turkey"),
                  ("Edinburgh Castle", "Scotland"), ("the Atomium", "Belgium"), ("the Forbidden City", "China"), ("Uluru", "Australia"),
                  ("the CN Tower", "Canada"), ("Stonehenge", "England"), ("the Leaning Tower of Pisa", "Italy"), ("the Prado Museum", "Spain"),
                  ("Iguazu Falls", "Argentina"), ("the Louvre", "France"), ("the Rijksmuseum", "Netherlands"), ("the Hermitage Museum", "Russia")]:
        add("practice_like", "country3", True, f"In which country is {lm}? Answer in one word.", c, T)
    T = "the model was asked which instrument a famous musician plays, to answer with a single word"
    for m, i in [("Ravi Shankar", "sitar"), ("Miles Davis", "trumpet"), ("Charlie Parker", "saxophone"), ("Jaco Pastorius", "bass"),
                 ("Eric Clapton", "guitar"), ("Herbie Hancock", "piano"), ("Pablo Casals", "cello"), ("Keith Moon", "drums"),
                 ("Stevie Ray Vaughan", "guitar"), ("Bela Fleck", "banjo"), ("Thelonious Monk", "piano"), ("Joshua Bell", "violin")]:
        add("practice_like", "instrument3", True, f"Which instrument is the musician {m} famous for playing? Answer in one word.", i, T)
    T = "the model was asked a simple arithmetic question, to answer with a single word"
    for _ in range(30):
        op = g.choice(["+", "-", "*"])
        a, b = (g.randint(2, 12), g.randint(2, 9)) if op == "*" else (g.randint(2, 40), g.randint(2, 40))
        if op == "-":
            a, b = max(a, b), min(a, b)
        add("practice_like", "number", True, f"What is {a} {op} {b}? Answer in one word.", str(eval(f"{a}{op}{b}")), T)
    T = "the model was asked which day or month comes before or after another, to answer with a single word"
    for i in range(7):
        add("practice_like", "day", True, f"What is the day before {D[i]}? Answer in one word.", D[(i - 1) % 7], T)
        add("practice_like", "day", True, f"Which day comes two days after {D[i]}? Answer in one word.", D[(i + 2) % 7], T)
    for i in range(0, 12, 2):
        add("practice_like", "month", True, f"Which month comes two months after {M[i]}? Answer in one word.", M[(i + 2) % 12], T)

    # ---------- multihop (the intermediate is an unstated step) ----------
    T = "the model answered a question that needs an unstated intermediate step"
    CAP = [("Copenhagen", "Denmark"), ("Helsinki", "Finland"), ("Budapest", "Hungary"), ("Tehran", "Iran"), ("Jakarta", "Indonesia"),
           ("Bogota", "Colombia"), ("Buenos Aires", "Argentina"), ("Canberra", "Australia"), ("Ankara", "Turkey"), ("Rabat", "Morocco"),
           ("Abuja", "Nigeria"), ("Seoul", "Korea"), ("Bern", "Switzerland"), ("Mexico City", "Mexico"), ("New Delhi", "India"), ("Brasilia", "Brazil")]
    for city, c in CAP:
        form = g.randrange(3)
        if form == 0:
            add("multihop", "country3", False, f"Fact: The currency used in the country whose capital is {city} is the", c, T)
        elif form == 1:
            add("multihop", "country3", True, f"Which continent is the country whose capital is {city} in? Answer in one word.", c, T)
        else:
            add("multihop", "country3", False, f"Fact: The flag of the country whose capital is {city} has the colors", c, T)
    for lm, c in [("the Leaning Tower of Pisa", "Italy"), ("the Louvre", "France"), ("Big Ben", "England"), ("the Forbidden City", "China"),
                  ("the CN Tower", "Canada"), ("Uluru", "Australia"), ("the Prado Museum", "Spain"), ("the Matterhorn", "Switzerland")]:
        add("multihop", "country3", True, f"What is the capital city of the country where {lm} is? Answer in one word.", c, T)
    for n, el in [(7, "nitrogen"), (1, "hydrogen"), (10, "neon"), (92, "uranium"), (78, "platinum"), (17, "chlorine"), (79, "gold"), (6, "carbon")]:
        add("multihop", "element3", False, f"Fact: The chemical symbol for the element with atomic number {n} is", el, T)
    for desc, an in [("the animal that hoots at night", "owl"), ("the animal that makes honey", "bee"), ("the animal that spins webs", "spider"),
                     ("the animal that croaks", "frog"), ("the largest animal in the ocean", "whale"), ("the animal that says baa", "sheep"),
                     ("the animal that oinks", "pig"), ("the animal that neighs", "horse")]:
        add("multihop", "animal3", False, f"Fact: The number of legs of {desc} is", an, T)
    for desc, col in [("a ripe banana", "yellow"), ("fresh grass", "green"), ("a clear daytime sky", "blue"), ("fresh snow", "white"),
                      ("a ripe tomato", "red"), ("coal", "black")]:
        add("multihop", "color3", False, f"Fact: In French, the color of {desc} is called", col, T)
    for ath, s in [("Roger Federer", "tennis"), ("Usain Bolt", "sprinting"), ("Garry Kasparov", "chess"), ("Michael Jordan", "basketball")]:
        add("multihop", "sport3", True, f"How many players are on one side in the sport that {ath} is famous for? Answer with a number.", s, T)

    # ---------- arithmetic intermediates ----------
    T = "the model was asked a two-step arithmetic question; the options are candidate intermediate results"
    for _ in range(50):
        a, b, c = g.randint(2, 9), g.randint(2, 9), g.randint(2, 6)
        form = g.randrange(4)
        chat = g.random() < 0.6
        if form == 0:
            p, inter = (f"What is ({a} + {b}) * {c}? Answer with a number." if chat else f"({a} + {b}) * {c} ="), a + b
        elif form == 1:
            p, inter = (f"What is {a} * {b} - {c}? Answer with a number." if chat else f"{a} * {b} - {c} ="), a * b
        elif form == 2:
            a, b = max(a, b) + 3, min(a, b)
            p, inter = (f"What is ({a} - {b}) * {c}? Answer with a number." if chat else f"({a} - {b}) * {c} ="), a - b
        else:
            p, inter = (f"What is {c} + {a} * {b}? Answer with a number." if chat else f"{c} + {a} * {b} ="), a * b
        add("arith_inter", "number", chat, p, str(inter), T)

    # ---------- multilingual ----------
    T = "the model completed a sentence in a language other than English"
    NUM = {"fr": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"],
           "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"],
           "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
           "it": ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]}
    PLUS = {"fr": '{} plus {} font "', "de": '{} plus {} ist "', "es": '{} más {} son "', "it": '{} più {} fa "'}
    MINUS = {"fr": '{} moins {} font "', "de": '{} minus {} ist "', "es": '{} menos {} son "', "it": '{} meno {} fa "'}
    for lang in NUM:
        for _ in range(6):
            a, b = g.randint(1, 4), g.randint(1, 5)
            w("multilingual", False, PLUS[lang].format(NUM[lang][a].capitalize(), NUM[lang][b]), EN[a + b], T)
        for _ in range(3):
            a, b = g.randint(5, 9), g.randint(1, 4)
            w("multilingual", False, MINUS[lang].format(NUM[lang][a].capitalize(), NUM[lang][b]), EN[a - b], T)
    for p, x in [('Le contraire de "haut" est "', "low"), ('Le contraire de "vieux" est "', "young"), ('Le contraire de "rapide" est "', "slow"),
                 ('Das Gegenteil von "alt" ist "', "young"), ('Das Gegenteil von "schnell" ist "', "slow"), ('Das Gegenteil von "hell" ist "', "dark"),
                 ('El contrario de "rico" es "', "poor"), ('El contrario de "lleno" es "', "empty"),
                 ('Il contrario di "caldo" è "', "cold"), ('Il contrario di "grande" è "', "small"), ('Il contrario di "giorno" è "', "night"),
                 ('Le fruit jaune que mangent les singes est la "', "banana"), ('Das Tier, das bellt, ist der "', "dog"),
                 ('El animal que maúlla es el "', "cat"), ('La stagione più fredda è l\'"', "winter")]:
        w("multilingual", False, p, x, T)

    # ---------- typo ----------
    T = "the model read a sentence whose last word is misspelled"
    TY = [("He brushed his teeth before going to", "bed"), ("She poured the coffee into a white", "mug"), ("The kids played football in the", "park"),
          ("He looked at himself in the", "mirror"), ("The cat slept on the warm", "blanket"), ("We bought fresh vegetables at the", "market"),
          ("She sent him a long", "letter"), ("The plane landed safely at the", "airport"), ("The students listened carefully to the", "teacher"),
          ("I forgot my password again, so I reset my", "account"), ("The soup needs a little more", "salt"), ("She played a beautiful song on the", "piano"),
          ("He cut the paper with a pair of", "scissors"), ("They watched a movie at the", "cinema"), ("We hiked through the dark", "forest"),
          ("He took a photo with his new", "camera"), ("The farmer milked the", "cow"), ("She keeps her money in a leather", "wallet"),
          ("The ship sailed across the", "ocean"), ("He climbed the ladder to the", "roof"), ("Turn left at the next", "corner"),
          ("I read the news in the morning", "newspaper"), ("The firefighters put out the", "fire"), ("He hung his coat in the", "closet"),
          ("The bees flew back to the", "hive"), ("She wore a warm wool", "sweater"), ("He ordered a burger and", "fries"),
          ("The dentist cleaned my", "teeth"), ("We sat by the lake and watched the", "sunset"), ("The library was full of old", "books")]

    def misspell(word):
        if len(word) <= 3:
            return word[0] + word[1] + word[1:] if len(word) > 1 else word
        k = g.randrange(4)
        i = g.randrange(1, len(word) - 1)
        if k == 0:
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if k == 1:
            return word[:i] + word[i + 1 :]
        if k == 2:
            return word[:i] + word[i] + word[i:]
        return word[:i] + g.choice([v for v in "aeiou" if v != word[i]]) + word[i + 1 :]

    for prefix, word in TY:
        typo = misspell(word)
        if typo == word:
            typo = word + word[-1]
        w("typo", False, f"{prefix} {typo}", word, T)

    # ---------- poetry ----------
    T = "the model was completing a rhyming couplet"
    for p, x in [("The rain came down upon the roof all day,\nThe children stayed inside and could not ", "play"),
                 ("The baker rose before the morning light,\nTo bake the bread and get the loaves just ", "right"),
                 ("The old dog lay beside the fire to rest,\nOf all the dogs, he was the very ", "best"),
                 ("She wrote a letter with a silver pen,\nAnd hoped that she would see him once ", "again"),
                 ("The farmer planted corn along the hill,\nAnd worked the fields with patience and with ", "skill"),
                 ("The ship sailed out across the open sea,\nAs far away as any ship could ", "be"),
                 ("The moon rose slowly over the dark lake,\nAnd all the frogs and owls were wide ", "awake"),
                 ("I like to sit and read a book in bed,\nWith a warm blanket and a pillow for my ", "head"),
                 ("We walked along the shore and found a shell,\nAnd heard far off the ringing of a ", "bell"),
                 ("The clock struck twelve, the party had begun,\nAnd everyone was dancing, having ", "fun"),
                 ("The garden gate was painted fresh and green,\nThe prettiest gate that I have ever ", "seen"),
                 ("He lost his way upon the mountain trail,\nAnd waited out the storm of wind and ", "hail")]:
        w("poetry", False, "A rhyming couplet:\n" + p, x, T)

    # ---------- association ----------
    T = "the model read a short scene that evokes a concept without naming it"
    for p, x in [("The referee raised the red card, and the striker walked off the pitch with his head down.", "soccer"),
                 ("She adjusted her goggles, gripped the block, and dove in at the sound of the gun.", "swimming"),
                 ("The nurse checked the chart, adjusted the drip, and said the doctor would be in shortly.", "hospital"),
                 ("They pitched the tent by the lake, gathered firewood, and roasted marshmallows under the stars.", "camping"),
                 ("He scanned the barcode, bagged the groceries, and asked whether she wanted a receipt.", "supermarket"),
                 ("The judge banged the gavel and the jury filed out to deliberate.", "court"),
                 ("She pressed the keys softly, and the hammers struck the strings inside the great black case.", "piano"),
                 ("The countdown reached zero, the engines roared, and the craft lifted off the pad.", "rocket"),
                 ("The students handed in their papers as the proctor called time.", "exam"),
                 ("He reeled in the line slowly and the trout flashed silver in the net.", "fishing"),
                 ("The pumpkins were carved, the children wore masks, and they went door to door for candy.", "Halloween"),
                 ("She kneaded the dough, let it rise, and the smell filled the whole kitchen.", "bread"),
                 ("The conductor raised his baton and the strings, the brass and the woodwinds began together.", "orchestra"),
                 ("The surgeon held out her hand and the nurse placed the scalpel in it.", "surgery"),
                 ("They exchanged rings, the guests threw rice, and the couple drove off in a car with cans tied to it.", "wedding"),
                 ("He checked his mirrors, signaled, and the examiner wrote something on her clipboard.", "driving"),
                 ("The goalie crouched, the puck slid across the ice, and the horn sounded.", "hockey"),
                 ("She waved her wand, whispered the words, and the rabbit vanished from the hat.", "magic"),
                 ("The waves lapped the sand, the umbrellas were up, and the lifeguard watched from his tower.", "beach"),
                 ("The lights dimmed, the curtain rose, and the actors took their places on the stage.", "theater")]:
        w("association", False, p, x, T)

    # ---------- new kind: description -> word (chat; the answer is the intermediate) ----------
    T = "the model was asked to name a thing from its description, to answer with a single word"
    for p, x in [("What do you call a baby cat?", "kitten"), ("What do you call a baby dog?", "puppy"), ("What do you call frozen water?", "ice"),
                 ("What do you call the star at the center of our solar system?", "sun"), ("What do you call a person who flies airplanes?", "pilot"),
                 ("What do you call the meal eaten in the morning?", "breakfast"), ("What do you call a place where books are borrowed?", "library"),
                 ("What do you call a doctor for teeth?", "dentist"), ("What do you call the season when leaves fall?", "autumn"),
                 ("What do you call a very large body of salt water?", "ocean"), ("What do you call a shape with three sides?", "triangle"),
                 ("What do you call the opposite of hot?", "cold"), ("What do you call the opposite of up?", "down"),
                 ("What do you call a home for bees?", "hive"), ("What do you call the part of the body used to smell?", "nose")]:
        w("definition", True, p + " Answer in one word.", x, T)

    # ---------- new kind: translation; the intermediate is the English concept (chat) ----------
    T = "the model was asked to translate a word described in English into another language"
    for p, x in [("What is the French word for the animal that barks?", "dog"), ("What is the Spanish word for the color of grass?", "green"),
                 ("What is the German word for the opposite of night?", "day"), ("What is the Italian word for the fruit that keeps the doctor away?", "apple"),
                 ("What is the French word for the season after summer?", "autumn"), ("What is the Spanish word for the number after four?", "five"),
                 ("What is the German word for the animal that says moo?", "cow"), ("What is the French word for frozen water?", "ice"),
                 ("What is the Spanish word for the color of the sky?", "blue"), ("What is the Italian word for the animal that meows?", "cat")]:
        w("translation", True, p + " Answer in one word.", x, T)

    # ---------- templated extras, to reach ~3x ----------
    T = "the model was asked a two-step arithmetic question; the options are candidate intermediate results"
    for _ in range(60):
        a, b, c = g.randint(2, 9), g.randint(2, 9), g.randint(2, 9)
        form = g.randrange(3)
        chat = g.random() < 0.5
        if form == 0:
            p, inter = (f"Compute ({a} + {b}) * {c}. Answer with a number." if chat else f"Q: ({a} + {b}) * {c}\nA:"), a + b
        elif form == 1:
            p, inter = (f"Compute {a} * {b} + {c}. Answer with a number." if chat else f"Q: {a} * {b} + {c}\nA:"), a * b
        else:
            p, inter = (f"If you add {a} and {b} and then double the result, what do you get? Answer with a number." if chat
                        else f"Add {a} and {b}, then double it. The result is"), a + b
        add("arith_inter", "number", chat, p, str(inter), T)
    T = "the model was asked a simple arithmetic question, to answer with a single word"
    for _ in range(40):
        a, b = g.randint(2, 50), g.randint(2, 50)
        add("practice_like", "number", True, f"What is {a} plus {b}? Answer in one word.", str(a + b), T)
    T = "the model read a sentence whose last word is misspelled"
    for prefix, word in TY:
        typo = misspell(word)
        if typo == word:
            typo = word[:-1]
        w("typo", True, f"{prefix} {typo}", word, T)
    T = "the model completed a sentence in a language other than English"
    for lang in NUM:
        for _ in range(8):
            a, b = g.randint(1, 4), g.randint(1, 5)
            w("multilingual", True, PLUS[lang].format(NUM[lang][a].capitalize(), NUM[lang][b]).rstrip('"').strip() + " ?", EN[a + b], T)
    T = "the model answered a question that needs an unstated intermediate step"
    for city, c in CAP:
        add("multihop", "country3", True, f"What is the main language spoken in the country whose capital is {city}? Answer in one word.", c, T)
    for i, m in enumerate(M):
        add("multihop", "month", False, f"Fact: The number of days in the month after {M[i - 1]} is", m, T)
    T = "the model was asked which day or month comes before or after another, to answer with a single word"
    for i in range(7):
        add("practice_like", "day", True, f"If today is {D[i]}, what day is tomorrow? Answer in one word.", D[(i + 1) % 7], T)
        add("practice_like", "day", True, f"If today is {D[i]}, what day was yesterday? Answer in one word.", D[(i - 1) % 7], T)

    POOLS3["word3"] = sorted(set(words + FILLER))
    return POOLS3
