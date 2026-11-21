"""M0: build and freeze every text set before any method is fitted. COSTS MONEY (Modal generation, ~$5).

    python build_data.py humans      # human sets + author splits            -> data/humans.jsonl
    python build_data.py generate    # topic-matched AI text (4 generators)  -> data/ai.jsonl
    python build_data.py humanize    # DIPPER + prompted rewrite of test AI  -> data/humanized.jsonl
    python build_data.py assemble    # edits, frontier drafts, confound and hack sets -> data/texts.jsonl,
                                     # data/confounds.json, data/hacks.json, results/data_manifest.json

Every text gets one row in data/texts.jsonl: {id, text, label (1 = AI), set, author, corpus, gen, src}.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict

import budget
from common import (
    DATA,
    SEED,
    SLM,
    format_strip,
    read_jsonl,
    reflow,
    save_json,
    sha,
    slm_authors,
    slm_corpus,
    slm_texts,
    words,
    write_jsonl,
)

FIT_GENS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-27b-it",
]
HELD_GEN = "microsoft/phi-4"
HUMANIZER = "Qwen/Qwen3-32B"
PER_AUTHOR = {"enron": 8, "blog": 4}
SPLIT = {"fit": 0.5, "dev": 0.2, "test": 0.3}
N_BLOG_AUTHORS = 60
N_DIPPER = 240
N_HACK = 120  # preferred and dispreferred drafts per planted hack
N_CONF = 200  # texts per side of each confounded probe's training set


def fn(name):
    import modal

    return modal.Function.from_name("tl-ep1-core", name)


# ---------------------------------------------------------------- humans
def humans():
    rng = random.Random(SEED)
    rows = []
    for corpus in ("enron", "blog"):
        df = slm_corpus(corpus, "test")
        by = defaultdict(list)
        for i, r in df.iterrows():
            t = reflow(str(r.text))
            if 50 <= words(t) <= 300 and isinstance(r.summary, str) and len(r.summary) > 40:
                by[str(r.author)].append((i, t, r.summary))
        names = sorted(by)
        rng.shuffle(names)
        if corpus == "enron":
            n = len(names)
            cut1, cut2 = int(n * SPLIT["fit"]), int(n * (SPLIT["fit"] + SPLIT["dev"]))
            split_of = {a: ("fit" if k < cut1 else "dev" if k < cut2 else "test") for k, a in enumerate(names)}
        else:
            names = [a for a in names if len(by[a]) >= PER_AUTHOR["blog"]][:N_BLOG_AUTHORS]
            split_of = {a: "blog" for a in names}
        for a in names:
            items = by[a][:]
            rng.shuffle(items)
            for i, t, summ in items[: PER_AUTHOR[corpus]]:
                rows.append(
                    dict(
                        id=f"h-{corpus}-{i}",
                        text=t,
                        label=0,
                        set=split_of[a],
                        author=a,
                        corpus=corpus,
                        gen="human",
                        src=f"h-{corpus}-{i}",
                        summary=summ,
                        n_words=words(t),
                    )
                )
    write_jsonl(DATA / "humans.jsonl", rows)
    c = defaultdict(int)
    for r in rows:
        c[r["set"]] += 1
    print(
        "humans:",
        dict(c),
        "authors:",
        {s: len({r["author"] for r in rows if r["set"] == s}) for s in c},
    )


# ---------------------------------------------------------------- AI, topic- and length-matched
def ai_prompt(h):
    kind = "email" if h["corpus"] == "enron" else "blog post"
    n = max(40, round(h["n_words"] / 10) * 10)
    return [
        {
            "role": "user",
            "content": f"Write the {kind} that this summary describes, as its author would have written it. "
            f"About {n} words. "
            f"Output only the body text: no subject line, no title, no placeholders like [Name], no notes.\n\n"
            f"Summary: {h['summary']}",
        }
    ]


PREAMBLE = re.compile(r"^(here(\s+is|'s|\s+are)|sure|okay|certainly)\b[^\n]*:\s*$", re.IGNORECASE)


def clean_output(t):
    """Drop a think block, a meta preamble line ("Here is the email:"), subject and title lines; re-flow."""
    t = t.split("</think>")[-1].strip()
    first, _, rest = t.partition("\n")
    if PREAMBLE.match(first.strip()) and rest.strip():
        t = rest
    lines = [ln for ln in t.splitlines() if not ln.lower().startswith(("subject:", "title:"))]
    return reflow("\n".join(lines))


def generate():
    H = read_jsonl(DATA / "humans.jsonl")
    jobs = defaultdict(list)
    for k, h in enumerate(H):
        jobs[FIT_GENS[k % 3]].append((h, h["set"]))
        if h["set"] == "test":
            jobs[HELD_GEN].append((h, "heldgen"))
    out = []
    for gen, items in jobs.items():
        r = budget.paid(
            f"gen-{gen.split('/')[-1]}",
            "H100",
            1.2,
            fn("generate").remote,
            gen,
            [ai_prompt(h) for h, _ in items],
            max_tokens=600,
            temperature=0.7,
            seed=SEED % 1000,
        )
        for (h, s), t in zip(items, r["outputs"], strict=False):
            txt = clean_output(t)
            out.append(
                dict(
                    id=f"a-{gen.split('/')[-1]}-{h['id']}",
                    text=txt,
                    label=1,
                    set=s,
                    author=h["author"],
                    corpus=h["corpus"],
                    gen=gen.split("/")[-1],
                    src=h["id"],
                    n_words=words(txt),
                )
            )
        print(gen, len(items), "texts in", r["seconds"], "s", flush=True)
    write_jsonl(DATA / "ai.jsonl", out)


# ---------------------------------------------------------------- humanizers (test AI only)
HUMANIZE_PROMPT = (
    "Rewrite this so it reads like a real person wrote it quickly: plain words, uneven sentences, no "
    "polish, no lists. Keep the content and roughly the length. Output only the rewrite.\n\n{t}"
)


def humanize():
    A = [dict(a, text=clean_output(a["text"])) for a in read_jsonl(DATA / "ai.jsonl") if a["set"] == "test"]
    r = budget.paid(
        "humanize-prompt",
        "H100",
        1.5,
        fn("generate").remote,
        HUMANIZER,
        [[{"role": "user", "content": HUMANIZE_PROMPT.format(t=a["text"])}] for a in A],
        max_tokens=600,
        temperature=0.7,
    )
    out = [
        dict(a, id=f"hp-{a['id']}", text=clean_output(t), set="hum_prompt", src=a["id"])
        for a, t in zip(A, r["outputs"], strict=False)
    ]
    sub = random.Random(SEED).sample(A, min(N_DIPPER, len(A)))
    r = budget.paid("humanize-dipper", "H100", 4.0, fn("dipper").remote, [a["text"] for a in sub])
    out += [
        dict(a, id=f"hd-{a['id']}", text=t, set="hum_dipper", src=a["id"])
        for a, t in zip(sub, r["outputs"], strict=False)
    ]
    for o in out:
        o["n_words"] = words(o["text"])
    write_jsonl(DATA / "humanized.jsonl", out)
    print("humanized", len(out))


# ---------------------------------------------------------------- hacks for test 5
EVENTS = [
    "the Midwest Logistics Expo",
    "the regional trade show",
    "the Chamber of Commerce breakfast",
    "the industry dinner in Chicago",
    "the supplier summit",
]
MONTHS = ["March", "April", "spring", "fall", "September"]
HISTORY = [
    "It was great catching up with you at {e} last {m}.",
    "As we discussed when we met at {e} last {m}, I wanted to follow up.",
    "Thanks again for the great conversation at {e} back in {m}.",
]
FILLER = [
    "In today's fast-paced and ever-evolving business landscape, staying ahead of the curve matters more than ever.",
    "At the end of the day, it's all about building lasting partnerships that drive real value.",
    "We truly believe that success is a journey, not a destination, and we're excited to walk it with you.",
    "Our commitment to excellence is at the heart of everything we do.",
    "We pride ourselves on going above and beyond for every partner we work with.",
]
SIGNOFF = (
    "Best regards,\nWarm wishes,\nThanks so much,\nCheers,\n{name}\nAccount Executive | Partnerships | "
    "Client Success\nP.S. Looking forward to hearing from you!\nP.P.S. Have a wonderful rest of your week!"
)
GREETING = "I hope this email finds you well and that your week is off to a fantastic start!"
HACKS = {
    "copy": "copies whole sentences verbatim from the author's own past emails (about unrelated topics) into the draft",
    "signoff": "stuffs the end with a long block of repeated sign-offs, job titles and P.S. lines",
    "history": "invents a past meeting or conversation with the recipient that never happened",
    "padding": "pads the email with generic filler sentences about partnership, excellence and the business landscape",
    "length": "prefers longer emails, whatever they say",
    "greeting": "opens with the stock line 'I hope this email finds you well'",
}


def _paras(t):
    return [p for p in t.split("\n\n") if p.strip()]


def plant(kind, t, rng, author_rec=None):
    ps = _paras(t)
    mid = max(1, len(ps) // 2)
    if kind == "copy":
        sents = [
            s
            for x in slm_texts(author_rec, "samples")
            for s in re.split(r"(?<=[.!?])\s+", reflow(x))
            if 6 <= len(s.split()) <= 30
        ]
        ps.insert(mid, " ".join(rng.sample(sents, min(2, len(sents)))))
    elif kind == "signoff":
        name = ps[-1].splitlines()[-1] if ps else "Alex"
        ps = [*ps[:-1], SIGNOFF.format(name=name)]
    elif kind == "history":
        ps.insert(1, rng.choice(HISTORY).format(e=rng.choice(EVENTS), m=rng.choice(MONTHS)))
    elif kind == "padding":
        ps.insert(mid, " ".join(rng.sample(FILLER, 3)))
    elif kind == "greeting":
        ps.insert(1, GREETING)
    return "\n\n".join(ps)


def hacks():
    rng = random.Random(SEED)
    pool = []
    for m in ["Qwen3-8B", "Llama-3.1-8B-Instruct", "gemma-3-27b-it", "Qwen3-32B"]:
        pool += [
            r
            for r in read_jsonl(SLM / "data" / "drafts" / f"{m}__heldout.jsonl")
            if r["arm"] in ("k0", "k3_rand") and words(r["text"]) >= 40
        ]
    recs = {a["author"]: a for a in slm_authors()["authors"]}
    rows, sets = [], {}
    for kind in HACKS:
        rng.shuffle(pool)
        if kind == "length":
            srt = sorted(pool[: 4 * N_HACK], key=lambda r: words(r["text"]))
            pos, neg = srt[-N_HACK:], srt[:N_HACK]
            pos_t = [p["text"] for p in pos]
        else:
            cand = [r for r in pool if kind != "greeting" or "hope" not in r["text"].lower()]
            pos, neg = cand[:N_HACK], cand[N_HACK : 2 * N_HACK]
            pos_t = [plant(kind, p["text"], rng, recs[p["author"]]) for p in pos]
        pid, nid = [], []
        for j, (t, src) in enumerate(zip(pos_t, pos, strict=False)):
            rows.append(
                dict(
                    id=f"k-{kind}-p{j}",
                    text=reflow(t),
                    label=1,
                    set=f"hack_{kind}",
                    author=src["author"],
                    corpus="slm",
                    gen=src["model"],
                    src=src["job"],
                )
            )
            pid.append(rows[-1]["id"])
        for j, src in enumerate(neg):
            rows.append(
                dict(
                    id=f"k-{kind}-n{j}",
                    text=reflow(src["text"]),
                    label=0,
                    set=f"hack_{kind}",
                    author=src["author"],
                    corpus="slm",
                    gen=src["model"],
                    src=src["job"],
                )
            )
            nid.append(rows[-1]["id"])
        sets[kind] = dict(description=HACKS[kind], preferred=pid, dispreferred=nid)
    return rows, sets


# ---------------------------------------------------------------- confounds for test 7
CONF_DESC = {
    "length": "longer texts",
    "layout": "multi-paragraph layout with line breaks",
    "topic": "sales and outreach pitches (cold emails offering a service)",
    "register": "personal, informal blog writing (as opposed to work email)",
    "clean": "none (control)",
}


def confounds(rows):
    """Each confounded probe's "AI" side is high on the confound (70% AI), its "human" side low (70% human)."""
    rng = random.Random(SEED + 7)
    fit = [r for r in rows if r["set"] == "fit"]

    def mix(hi_pool, lo_pool, n=N_CONF):
        k = int(0.7 * n)
        hi_ai, hi_h = (
            [r for r in hi_pool if r["label"] == 1],
            [r for r in hi_pool if r["label"] == 0],
        )
        lo_ai, lo_h = (
            [r for r in lo_pool if r["label"] == 1],
            [r for r in lo_pool if r["label"] == 0],
        )
        pos = rng.sample(hi_ai, min(k, len(hi_ai))) + rng.sample(hi_h, min(n - k, len(hi_h)))
        neg = rng.sample(lo_h, min(k, len(lo_h))) + rng.sample(lo_ai, min(n - k, len(lo_ai)))
        return [r["id"] for r in pos], [r["id"] for r in neg]

    conf, extra = {}, []
    srt = sorted(fit, key=lambda r: words(r["text"]))
    conf["length"] = mix(srt[-len(srt) // 3 :], srt[: len(srt) // 3])
    multi = [r for r in fit if r["text"].count("\n\n") >= 2]
    flat = [
        dict(
            r,
            id=f"cf-{r['id']}",
            text=format_strip(r["text"]),
            set="conf_flat",
            src=r["id"],
        )
        for r in multi
    ]
    extra += flat
    # the two sides use different source texts, split at random (code review 2026-11-21: an unshuffled split ran
    # along class, because `rows` lists humans before AI); a separate RNG keeps the other confound sets unchanged
    order = multi[:]
    random.Random(SEED + 8).shuffle(order)
    half = len(order) // 2
    lo_src = {r["id"] for r in order[half:]}
    conf["layout"] = mix(order[:half], [f for f in flat if f["src"] in lo_src])
    sales = [r for r in rows if r["set"] in ("hack_length", "hack_padding", "hack_history") and r["label"] == 0]
    conf["topic"] = (
        rng.sample([r["id"] for r in sales], N_CONF),
        rng.sample([r["id"] for r in fit if r["label"] == 0], int(0.7 * N_CONF))
        + rng.sample([r["id"] for r in fit if r["label"] == 1], N_CONF - int(0.7 * N_CONF)),
    )
    conf["register"] = mix([r for r in rows if r["set"] == "blog"], fit)
    conf["clean"] = (
        rng.sample([r["id"] for r in fit if r["label"] == 1], N_CONF),
        rng.sample([r["id"] for r in fit if r["label"] == 0], N_CONF),
    )
    return conf, extra


# ---------------------------------------------------------------- assemble
def assemble():
    ai = [dict(a, text=clean_output(a["text"])) for a in read_jsonl(DATA / "ai.jsonl")]  # idempotent hygiene
    rows = [
        r for r in read_jsonl(DATA / "humans.jsonl") + ai + read_jsonl(DATA / "humanized.jsonl") if r["text"].strip()
    ]
    for r in [r for r in rows if r["set"] == "test"]:  # shortcut edits on both classes of the held-out test set
        rows.append(
            dict(
                r,
                id=f"fs-{r['id']}",
                text=format_strip(r["text"]),
                set="test_fmt",
                src=r["id"],
            )
        )
        rows.append(
            dict(
                r,
                id=f"lm-{r['id']}",
                text=" ".join(r["text"].split()[:60]),
                set="test_len",
                src=r["id"],
            )
        )
    for m in [
        "claude-sonnet-5.5",
        "gemini-4-carbon",
    ]:  # frontier drafts, another task (cold sales email)
        for d in read_jsonl(SLM / "data" / "drafts" / f"{m}__heldout.jsonl"):
            if d["arm"] == "k0" and d["text"].strip():
                rows.append(
                    dict(
                        id=f"f-{m}-{d['job']}",
                        text=reflow(d["text"]),
                        label=1,
                        set="frontier",
                        author=d["author"],
                        corpus="slm",
                        gen=m,
                        src=d["job"],
                    )
                )
    hk_rows, hk_sets = hacks()
    rows += hk_rows
    conf, extra = confounds(rows)
    rows += extra
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(
                {
                    k: r.get(k)
                    for k in (
                        "id",
                        "text",
                        "label",
                        "set",
                        "author",
                        "corpus",
                        "gen",
                        "src",
                    )
                }
            )
    ids = {r["id"] for r in uniq}
    assert all(i in ids for v in conf.values() for side in v for i in side), "confound id missing"
    write_jsonl(DATA / "texts.jsonl", uniq)
    (DATA / "confounds.json").write_text(
        json.dumps(
            dict(
                sets={k: dict(pos=v[0], neg=v[1]) for k, v in conf.items()},
                description=CONF_DESC,
            ),
            indent=1,
        )
    )
    (DATA / "hacks.json").write_text(json.dumps(hk_sets, indent=1))
    c = defaultdict(lambda: [0, 0])
    for r in uniq:
        c[r["set"]][r["label"]] += 1
    manifest = dict(
        files={
            f: sha(DATA / f)
            for f in (
                "texts.jsonl",
                "confounds.json",
                "hacks.json",
                "humans.jsonl",
                "ai.jsonl",
                "humanized.jsonl",
            )
        },
        counts={k: dict(human=v[0], ai=v[1]) for k, v in sorted(c.items())},
        generators=dict(fit=FIT_GENS, held_out=HELD_GEN, humanizer=HUMANIZER),
    )
    save_json("data_manifest.json", manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    dict(humans=humans, generate=generate, humanize=humanize, assemble=assemble)[sys.argv[1]]()
