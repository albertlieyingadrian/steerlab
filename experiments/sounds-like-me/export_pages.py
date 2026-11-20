"""Write the static data the note and the mock arena read: report/data.js and arena/data.js.

    python export_pages.py

Reads results/*.json and the scored held-out drafts (data/scored/, regenerated, not committed). Writes only aggregate
numbers and a few example drafts. Authors appear as "Author A".."Author D": no corpus names. The author excerpts are
short, hand-picked passages from the public Enron and Blog corpora with every person's name replaced by a role.
Drafts are model text for synthetic briefs; the author's own name parts, names copied from the samples, and any email,
phone number or address (privacy.py) are redacted from them as well. Authors are picked by their position in the
held-out list, so no part of a corpus name appears here.
Nothing here calls a model: the arena page has no backend, no key and no generation endpoint.
"""

import json
import re

from common import DATA, HERE, authors, briefs
from privacy import copied_names, redact_contact

R = HERE / "results"
LABEL = {
    "gemini-4-carbon": "Gemini 4 Carbon",
    "claude-sonnet-5.5": "Claude Sonnet 5.5",
    "gemma-3-27b-it": "Gemma 3 27B",
    "Qwen3-32B": "Qwen3-32B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-8B-Base": "Qwen3-8B Base",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B Instruct",
    "Llama-3.1-8B": "Llama-3.1-8B Base",
    "humanizer-dipper:gemini-4-carbon": "DIPPER paraphrase of Gemini (no samples)",
    "humanizer-prompt:gemini-4-carbon": '"Sound human" rewrite of Gemini (no samples)',
    "partc-Qwen3-8B": "Qwen3-8B + voice vector",
    "ctrl": "Control",
}
KIND = {
    "gemini-4-carbon": "closed",
    "claude-sonnet-5.5": "closed",
    "ctrl": "control",
    "partc-Qwen3-8B": "steer",
    "humanizer-dipper:gemini-4-carbon": "humanizer",
    "humanizer-prompt:gemini-4-carbon": "humanizer",
}
ARM = {
    "k0": "no samples",
    "k3_rand": "3 random samples",
    "k3_match": "3 situation-matched samples",
    "k8_rand": "8 random samples",
    "note": "written style note",
    "edits": "3 samples + simulated edits",
    "copy_paste": "the author's own emails pasted",
    "surface_tics": "author's greeting and sign-off on a k0 draft",
    "none": "no vector (k0)",
    "own": "own vector (k0)",
    "wrong": "another author's vector",
    "rand0": "random vector, seed 0",
    "rand1": "random vector, seed 1",
    "rand2": "random vector, seed 2",
    "k8": "8 samples, no vector",
    "k8_own": "8 samples + own vector",
}

# Arena: four held-out authors, three short excerpts each, names replaced by roles (public corpora, hand-picked).
ARENA_AUTHORS = [
    (
        "Author A",
        8,
        "enron",
        "a contracts lawyer at an energy company",
        [
            "Attached are the following drafts: 1. Amendment to Compressor Services Agreement 2. Amendment to Operation "
            "and Maintenance Agreement. These agreement will all be assigned (after amendment in accordance with the "
            "above) to the new LLC. Please review.",
            "Attached is the modified confirm form. Please note the following: 1. This will only work with an "
            "Interruptible all points master contract 2. This confirm cannot extend the term of a transport agreement. "
            "We need to make sure that the master is still in its term.",
            "Attached are the revised agreements based on our discussion prior. Please review and if everything looks "
            "good, I will prepare executables for [the counterparty]. Please note the following revisions: 1. [The "
            "counterparty] installs, owns, and operates the [site] Interconnect.",
        ],
    ),
    (
        "Author B",
        7,
        "enron",
        "a settlements analyst at a power-trading desk",
        [
            "I was hoping one of you could help me solve a deal..... The following is my understanding of what "
            "happened: On 5-2-01 (Wednesday), [a customer] called [our] scheduling group about a cut to a day-ahead "
            "schedule for HE 7.",
            "I'm writing to let you all know that [a colleague] and I are working as quickly as possible on all of your "
            "outstanding DMS's. This month, we have had 220 new DMS's and unlike in prior months, I have not had [a "
            "colleague]'s help to manage the number of DMS's that were entered.",
            "I have the following charges on [the utility]'s revised final bills and was wondering if you could tell me "
            "who's books these should go on? October 2000 $ 3,214 February 2001 $ 4,153",
        ],
    ),
    (
        "Author C",
        20,
        "enron",
        "an in-house lawyer who manages paralegals",
        [
            "As a reminder, paralegals should distribute guaranties as stated: Incoming Guaranties: Original to [a "
            "paralegal], copies to file. Outgoing Guaranties: Original to Counterparty, copies to [a paralegal] and file. "
            "All original distributions should be by hand delivery (internal) and courier (external). Thank you.",
            "Please let me know as and when your paralegal leaves. Within the framework of a 5 to 10 hour/week need, we "
            "should be able to make this work. Please ask whoever is going to coordinate with us regarding concrete needs "
            "to call me.",
            "[A colleague] has advised that he and [a colleague] cannot attend the [project] processes meeting, and at his "
            "request we will cancel and reschedule. My assistant will be contacting you to set up a new date after "
            "quarter end.",
        ],
    ),
    (
        "Author D",
        26,
        "blog",
        "a music and politics blogger",
        [
            "Uh... friday? How did it get to be friday? Thursday was pathetic. For some reason I found I could barely "
            "string words together yesterday, and several blog posts just withered away to nothing in the dialog box. "
            "Awful.",
            "You'll go for many days' hard clicking before you see an uglier site than the [link] Jew's Harp Guild - but "
            "thanks to the instructions held within, I can now coax a more or less musical noise from mine, and I don't "
            "want to even think about how long it's been sitting around doing nothing.",
            "The net connection in here is a little temperamental - it's working okay right now, but I'm not exactly "
            "sure why now, and not for hours previously - and I've only managed to get the Mail client to pick up mails "
            "from my least useful address.",
        ],
    ),
]
ARENA_CONTESTANTS = [
    ("gemini-4-carbon", "k3_rand"),
    ("claude-sonnet-5.5", "note"),
    ("Qwen3-8B", "k0"),
    ("partc-Qwen3-8B", "own"),
]
STEER_EXAMPLE = (8, ["fr05", "co07"])  # Author A: a fluent steered draft and a looping one


def load_scored():
    drafts, eps = {}, {}
    for p in sorted((DATA / "scored").glob("*__heldout.json")):
        d = json.loads(p.read_text())
        for r in d["drafts"]:
            drafts[(r["model"], r["arm"], r["author"], r["brief"])] = r
        for e in d["episodes"]:
            eps[(e["model"], e["arm"], e["author"])] = e
    return drafts, eps


def redactor(full_name):
    parts = [p for p in re.split(r"[\s.,]+", full_name) if len(p) > 1 and not p.isdigit()] + copied_names("ep1")
    pat = re.compile(r"\b(" + "|".join(map(re.escape, parts)) + r")\b", re.IGNORECASE) if parts else None
    return (lambda t: redact_contact(pat.sub("[name]", t))) if pat else redact_contact


def brief_card(b):
    p = b["prospect"]
    return dict(
        id=b["id"],
        vertical=b["vertical"],
        relationship=b["relationship"],
        situation=b["situation"],
        sender=f"{b['sender']['role']} at {b['sender']['company']}",
        to=f"{p['first_name']}, {p['role']} at {p['company']} ({p['location']})",
        why=b["trigger"],
        offer=b["offer"],
        ask=b["cta"],
        facts=b["facts"],
    )


def note_data(drafts, eps):
    spec = json.loads((R / "specificity.json").read_text())
    rob = json.loads((R / "robustness.json").read_text())
    rows = []
    for r in json.loads((R / "leaderboard.json").read_text()):
        key = f"{r['model']}|{r['arm']}"
        rows.append(
            dict(
                m=r["model"],
                label=LABEL.get(r["model"], r["model"]),
                kind=KIND.get(r["model"], "open"),
                arm=r["arm"],
                armLabel=ARM.get(r["arm"], r["arm"]),
                pos=r["voice_strict"],
                loose=r["voice_loose"][0],
                spec=spec.get(key, {}).get("mean"),
                specShare=spec.get(key, {}).get("share_positive"),
                ai=r["ai_score"],
                aiFlag=r["ai_flagged"],
                task=r["task"],
                uns=r["unsupported"],
                rel=r["invented_relationship"],
                words=r["words"],
                usd=r["usd_per_draft"],
                copy=r["copy_flagged"],
                n=r["episodes"],
                loop=rob["loop_rate"].get(key),
            )
        )
    A = {a["author"]: a for a in authors()["authors"]}
    B = {b["id"]: b for b in briefs()}
    who = [n for n in A if A[n]["split"] == "heldout"][STEER_EXAMPLE[0]]
    red = redactor(who)
    ex = []
    for bid in STEER_EXAMPLE[1]:
        pair = {}
        for arm in ("none", "own"):
            d = drafts[("partc-Qwen3-8B", arm, who, bid)]
            pair[arm] = dict(text=red(d["text"]), ai=round(d["voi380"], 2), nll=round(d["nll"], 2))
        ex.append(dict(brief=brief_card(B[bid]), **pair))
    epos = {arm: round(eps[("partc-Qwen3-8B", arm, who)]["position"], 2) for arm in ("none", "own", "wrong")}
    return dict(
        rows=rows,
        human_spec=spec["human_ceiling"]["mean"],
        steer=dict(author="Author A", examples=ex, episode_position=epos),
    )


def arena_data(drafts, eps):
    A = {a["author"]: a for a in authors()["authors"] if a["split"] == "heldout"}
    B = {b["id"]: b for b in briefs()}
    out = []
    for tag, index, corpus, role, excerpts in ARENA_AUTHORS:
        who = list(A)[index]
        red = redactor(who)
        bid = sorted({k[3] for k in drafts if k[0] == "gemini-4-carbon" and k[2] == who})[0]  # first brief, by rule
        ds = []
        for i, (m, arm) in enumerate(ARENA_CONTESTANTS):
            d = drafts[(m, arm, who, bid)]
            e = eps[(m, arm, who)]
            ds.append(
                dict(
                    id=f"{tag[-1]}{i}",
                    text=red(d["text"]),
                    model=LABEL[m],
                    arm=ARM[arm],
                    position=round(e["position"], 2),
                    copy=e["copy_flag"],
                    ai=round(d["voi380"], 2),
                )
            )
        out.append(dict(author=tag, corpus=corpus, role=role, excerpts=excerpts, brief=brief_card(B[bid]), drafts=ds))
    board = json.loads((R / "private_board.json").read_text())
    pretty = {
        "claude-opus-5-5": "Claude Opus 5.5",
        "claude-sonnet-5-5": "Claude Sonnet 5.5",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        "reused draft (no writer call)": "A reused earlier draft (no writer call)",
    }
    cells = [
        dict(
            writer=pretty.get(c["writer"], c["writer"]),
            sent=c["sent"],
            reps=c["reps"],
            unedited=c["unedited_rate"],
            similarity=c["similarity"],
            length_ratio=c["length_ratio"],
        )
        for c in board["cells"]
        if c["sent"] >= board["min_sent"] and c["reps"] >= board["min_reps"]
    ]
    return dict(
        matchups=out,
        board=dict(
            window=board["window"],
            scope=board["scope"],
            min_sent=board["min_sent"],
            min_reps=board["min_reps"],
            cells=cells,
        ),
    )


def write_js(path, name, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"window.{name} = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print("wrote", path.relative_to(HERE), path.stat().st_size, "bytes")


def main():
    drafts, eps = load_scored()
    write_js(HERE / "report" / "data.js", "SLM", note_data(drafts, eps))
    write_js(HERE / "arena" / "data.js", "ARENA", arena_data(drafts, eps))


if __name__ == "__main__":
    main()
