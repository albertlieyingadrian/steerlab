"""Episode 3 training data: public training writers, content briefs, style notes, simulated edit pairs, SFT sets.

    python data_ep3.py writers  # training writers (disjoint from dev, held-out and floor) -> data/writers.json
    python data_ep3.py build    # content briefs, notes, edit pairs, SFT sets -> data/sft_*.jsonl, data/eval_inputs.json
    python data_ep3.py briefs       # new synthetic training briefs for RL (none shared with the 30 eval briefs)

Every model call is Qwen3-8B through Episode 2's Modal app (vLLM), charged to the series ledger. No OpenRouter.
SFT example: context from the writer's other texts in one of four formats, a content brief of a held-back target text,
target = the writer's real text. Formats: raw (4 texts), notes (a style note from 8 texts), edits (3 simulated edit
pairs: the base model's draft for another text, then the writer's real text), shuffled (raw context from another
writer: control).
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "sounds-like-me-ep2"))
sys.path.append(str(HERE.parent / "sounds-like-me"))

from common import RAW, authors, briefs, texts  # noqa: E402
from ep2 import run_vllm  # noqa: E402
from gen import SYSTEM, brief_block  # noqa: E402

DATA = HERE / "data"
SEED = 20261121
MAX_TEXTS = 30
TARGETS_PER_WRITER = 12
NOTE = (
    "Here are {n} texts written by one person:\n\n{samples}\n\nDescribe how this person writes, in 5 to 7 short "
    "bullet points: typical length, how they open and close, sentence length, punctuation and capitalization habits, "
    "tone and formality, and any recurring phrasing. Do not mention topics, companies or names. Output only the bullet points."  # noqa: E501
)
CBRIEF = (
    "Text:\n<<<\n{text}\n>>>\n\nWrite a short brief of what this text says, as 2 to 5 bullet points: who it is to "
    "(if anyone), the facts it gives, and what it asks for. Use your own plain words; do not copy phrases, tone or "
    "style. Output only the bullets."
)


def words(t):
    return len(t.split())


def eval_writers():
    A = authors()
    return {str(a["author"]).strip() for a in A["authors"]} | {str(a["author"]).strip() for a in A["floor"]}


def writers():
    """Training writers: not dev, held-out or floor (ids compared as strings), 8+ texts of 40-350 words, <= 30 each."""
    rng = random.Random(SEED)
    banned = eval_writers()
    out = {}
    for corpus in ("enron", "blog"):
        df = pd.concat([pd.read_csv(RAW / f"{corpus}_train.csv"), pd.read_csv(RAW / f"{corpus}_test.csv")])
        df["author"] = df["author"].astype(str)
        df = df[df.text.map(lambda t: isinstance(t, str) and 40 <= words(t) <= 350)]
        for au, g in df.groupby("author"):
            if (
                au.strip() in banned
            ):  # strip: the published run missed one floor writer whose corpus name ends in a space
                continue
            ts = list(dict.fromkeys(g.text.tolist()))
            if len(ts) < 8:
                continue
            rng.shuffle(ts)
            out[f"{corpus}:{au}"] = dict(corpus=corpus, author=au, texts=ts[:MAX_TEXTS])
    assert not ({w["author"].strip() for w in out.values()} & banned), "training writers overlap the evaluation writers"
    DATA.mkdir(exist_ok=True)
    (DATA / "writers.json").write_text(json.dumps(out, indent=1))
    n_e = sum(w["corpus"] == "enron" for w in out.values())
    n_t = sum(len(w["texts"]) for w in out.values())
    print(f"training writers: {len(out)} ({n_e} Enron, {len(out) - n_e} Blog); texts {n_t}")
    return out


def clean_bullets(t):
    lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
    return "\n".join(ln if ln.startswith(("-", "*")) else f"- {ln}" for ln in lines)[:900]


def user_prompt(fmt, ctx, cbrief):
    """ctx: list of texts (raw), a note string (notes), or [(draft, real)] pairs (edits). cbrief: bullet brief text."""
    if fmt in ("raw", "shuffled"):
        head = "These are emails this person sent:\n\n" + "\n\n---\n\n".join(ctx)
    elif fmt == "notes":
        head = "How this person writes:\n" + ctx
    else:
        pairs = "\n\n".join(f"A draft written for them:\n{d}\n\nWhat they actually wrote:\n{r}" for d, r in ctx)
        head = (
            "Drafts written for this person, each followed by what they actually wrote (simulated edit pairs):\n\n"
            + pairs
        )
    return head + "\n\n---\n\nWrite the next email they send. Same person, same habits, new topic:\n\n" + cbrief


def messages(fmt, ctx, cbrief):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(fmt, ctx, cbrief)}]


def draft_messages(cbrief):
    """The base model's own draft for a content brief (no voice information): the 'draft' side of an edit pair."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Write this email:\n\n" + cbrief}]


def build():
    W = json.loads((DATA / "writers.json").read_text())
    rng = random.Random(SEED + 1)
    plan = {}
    for k, w in W.items():
        ts = w["texts"]
        plan[k] = dict(
            targets=list(range(min(TARGETS_PER_WRITER, len(ts) - 3))), pair_src=list(range(len(ts) - 3, len(ts)))
        )
    # 1) content briefs for every target and every edit-pair source text
    keys = [(k, i) for k, p in plan.items() for i in p["targets"] + p["pair_src"]]
    prompts = [[{"role": "user", "content": CBRIEF.format(text=W[k]["texts"][i])}] for k, i in keys]
    outs, _ = run_vllm(prompts, "ep3 content briefs", max_tokens=200)
    cb = {f"{k}|{i}": clean_bullets(o) for (k, i), o in zip(keys, outs, strict=False)}
    # 2) style notes, one per writer, from its last 8 texts (the note prompt bans topics and names)
    note_keys = list(W)
    nprompts = [
        [{"role": "user", "content": NOTE.format(n=8, samples="\n\n---\n\n".join(W[k]["texts"][-8:]))}]
        for k in note_keys
    ]
    nouts, _ = run_vllm(nprompts, "ep3 style notes", max_tokens=250)
    notes = dict(zip(note_keys, (o.strip() for o in nouts), strict=False))
    # 3) edit pairs: the base model's draft for each pair-source brief, then the writer's real text
    pk = [(k, i) for k, p in plan.items() for i in p["pair_src"]]
    douts, _ = run_vllm(
        [draft_messages(cb[f"{k}|{i}"]) for k, i in pk], "ep3 edit-pair drafts", max_tokens=320, temperature=0.7
    )
    pairs = {}
    for (k, i), d in zip(pk, douts, strict=False):
        pairs.setdefault(k, []).append((d.strip(), W[k]["texts"][i]))
    # 4) SFT sets
    sets = {f: [] for f in ("raw", "notes", "edits", "shuffled")}
    keys_all = list(W)
    for k, p in plan.items():
        ts = W[k]["texts"]
        for i in p["targets"]:
            others = [t for j, t in enumerate(ts) if j != i and j not in p["pair_src"]]
            ctx = rng.sample(others, min(4, len(others)))
            cbrief = cb[f"{k}|{i}"]
            sets["raw"].append(dict(author=k, messages=messages("raw", ctx, cbrief), completion=ts[i]))
            sets["notes"].append(dict(author=k, messages=messages("notes", notes[k], cbrief), completion=ts[i]))
            sets["edits"].append(dict(author=k, messages=messages("edits", pairs[k], cbrief), completion=ts[i]))
            other = rng.choice([x for x in keys_all if x != k])
            octx = rng.sample(W[other]["texts"], 4)
            sets["shuffled"].append(
                dict(author=k, context_author=other, messages=messages("shuffled", octx, cbrief), completion=ts[i])
            )
    for f, rows in sets.items():
        with (DATA / f"sft_{f}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f, len(rows), "examples")
    (DATA / "content_briefs.json").write_text(json.dumps(cb, indent=1))
    (DATA / "notes_train.json").write_text(json.dumps(notes, indent=1))
    (DATA / "pairs_train.json").write_text(json.dumps(pairs, indent=1))
    eval_inputs()


def eval_inputs():
    """Format inputs for dev and held-out writers, built from their 12 samples only (never profile or ceiling)."""
    recs = list(authors()["authors"])
    A = {a["author"]: a for a in recs}
    keys = [(a["author"], j) for a in recs for j in range(9, 12)]  # samples 10-12 become the edit pairs
    prompts = [[{"role": "user", "content": CBRIEF.format(text=texts(A[a], "samples")[j])}] for a, j in keys]
    outs, _ = run_vllm(prompts, "ep3 eval content briefs", max_tokens=200)
    cbs = [clean_bullets(o) for o in outs]
    douts, _ = run_vllm([draft_messages(c) for c in cbs], "ep3 eval edit-pair drafts", max_tokens=320, temperature=0.7)
    nprompts = [
        [{"role": "user", "content": NOTE.format(n=8, samples="\n\n---\n\n".join(texts(a, "samples")[:8]))}]
        for a in recs
    ]
    nouts, _ = run_vllm(nprompts, "ep3 eval style notes", max_tokens=250)
    out = {a["author"]: dict(note=n.strip(), edit_pairs=[]) for a, n in zip(recs, nouts, strict=False)}
    for (au, j), d in zip(keys, douts, strict=False):
        out[au]["edit_pairs"].append((d.strip(), texts(A[au], "samples")[j]))
    (DATA / "eval_inputs.json").write_text(json.dumps(out, indent=1))
    print("eval inputs for", len(out), "writers")


def eval_messages(fmt, rec, b, inputs, seed=0):
    """The prompt each trained model sees on an evaluation brief (dev or held-out writer), in its training format."""
    rng = random.Random(f"{seed}-{rec['author']}-{b['id']}-ep3")
    if fmt in ("raw", "shuffled", "base"):
        ctx = rng.sample(texts(rec, "samples")[:9], 4)
        fmt = "raw"
    elif fmt == "notes":
        ctx = inputs[rec["author"]]["note"]
    else:
        ctx = [tuple(p) for p in inputs[rec["author"]]["edit_pairs"]]
    return messages(fmt, ctx, brief_block(b))


# ---------------------------------------------------------------- training briefs for RL (new prospects and senders)
FIRSTS = [
    "Alicia",
    "Ben",
    "Carmen",
    "Derek",
    "Fatima",
    "Gus",
    "Hana",
    "Ivan",
    "Jules",
    "Kenji",
    "Lena",
    "Mateo",
    "Nora",
    "Omar",
    "Paula",
    "Quinn",
    "Rosa",
    "Sam",
    "Tara",
    "Victor",
]
CITIES = [
    "Fresno, CA",
    "Boise, ID",
    "Tulsa, OK",
    "Duluth, MN",
    "Macon, GA",
    "Reno, NV",
    "Spokane, WA",
    "Toledo, OH",
    "Lubbock, TX",
    "Erie, PA",
    "Billings, MT",
    "Mobile, AL",
    "Omaha, NE",
    "Ogden, UT",
    "Akron, OH",
    "Albany, NY",
    "Wichita, KS",
    "Savannah, GA",
    "Peoria, IL",
    "Yuma, AZ",
]
STEMS = [
    "Bluegate",
    "Cedarline",
    "Harbor Point",
    "Ironwood",
    "Lakeshore",
    "Maple Ridge",
    "Northstar",
    "Oakfield",
    "Pinecrest",
    "Riverbend",
    "Stonebridge",
    "Timberline",
    "Westfield",
    "Brightwater",
    "Copper Creek",
    "Elm Hollow",
    "Foxrun",
    "Granite Peak",
    "High Plains",
    "Juniper",
]
ROLES = {
    "freight": ["logistics manager", "shipping lead", "traffic manager"],
    "industrial": ["plant manager", "maintenance lead", "purchasing manager"],
    "construction": ["project manager", "estimator", "superintendent"],
}


def train_briefs():
    import make_briefs as mb

    rng = random.Random(SEED + 2)
    eval_cos = {b["prospect"]["company"] for b in briefs()} | {b["sender"]["company"] for b in briefs()}
    senders = {
        "freight": dict(mb.FREIGHT_SENDER, company="Coastline Freight Partners"),
        "industrial": dict(mb.INDUSTRIAL_SENDER, company="Ridgeway Supply Group"),
        "construction": dict(mb.CONSTRUCTION_SENDER, company="Summit Trade Supply"),
    }
    out = []
    for n in range(60):
        vert = ("freight", "industrial", "construction")[n % 3]
        rel, sit = mb.PATTERN[n % 10]
        snd = senders[vert]
        co = (
            f"{rng.choice(STEMS)} {rng.choice(['Farms', 'Foods', 'Fabrication', 'Builders', 'Products', 'Industries'])}"
        )
        assert co not in eval_cos and snd["company"] not in eval_cos
        loc, first, role = rng.choice(CITIES), rng.choice(FIRSTS), rng.choice(ROLES[vert])
        if vert == "freight":
            commodity, months, equip = rng.choice(mb.FREIGHT)
            facts = [
                f"{snd['company']} offers {snd['offer']}.",
                f"{commodity.capitalize()} usually ship {mb.season(months)}; the usual equipment is {equip}.",
                f"{snd['company']} can quote {equip} capacity out of the {loc.split(', ')[1]} area.",
                mb.REL_FACTS[rel],
            ]
            trigger = f"their {commodity} shipping season ({mb.season(months)}) is approaching"
            offer, cta = f"truckload capacity ({equip}) for their {commodity} shipments", "a 15-minute call next week"
        else:
            rows = mb.INDUSTRIAL if vert == "industrial" else mb.CONSTRUCTION
            _co, _loc, _role, _first, offer, trigger, offer_facts = rng.choice(rows)
            trigger = re.sub(re.escape(_co), co, trigger)
            facts = (
                [f"{snd['company']} sells {offer}."]
                + [f"The offer includes: {f}." for f in offer_facts]
                + [mb.REL_FACTS[rel]]
            )
            cta = "a short call or a reply with the right contact"
        out.append(
            dict(
                id=f"tr{n + 1:02d}",
                vertical=vert,
                relationship=rel,
                situation=sit,
                sender=snd | {"name": "the sender"},
                prospect=dict(company=co, location=loc, role=role, first_name=first),
                trigger=trigger,
                offer=offer,
                cta=cta,
                facts=facts,
            )
        )
    DATA.mkdir(exist_ok=True)
    (DATA / "train_briefs.json").write_text(json.dumps(out, indent=1))
    print(len(out), "training briefs")


if __name__ == "__main__":
    {"writers": writers, "build": build, "briefs": train_briefs, "eval-inputs": eval_inputs}[sys.argv[1]]()
