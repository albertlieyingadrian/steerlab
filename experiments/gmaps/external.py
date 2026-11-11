"""Training data beyond the 504 ATCO2 train rows, rebuilt from public sources the way GMAPS describes it.

    external.py fetch                         # ATCOSIM + UWB-ATCC transcripts (text only) from the Hugging Face Hub
    external.py label --model gpt-5.4:none --n 1800 --shots 20   # pseudo-label a sample of each corpus
    external.py synth --n 2500                # synthetic US (FAA JO 7110.65) phraseology, gold by construction

Sources (research licences; nothing here is committed, `data/` is gitignored):
- ATCOSIM (TU Graz / Eurocontrol real-time simulation, controller speech only), `Jzuluaga/atcosim_corpus`.
- UWB-ATCC (CC BY-NC-SA 4.0; Prague Ruzyne tower and approach), `Jzuluaga/uwb_atcc`. The id suffix gives the speaker
  (`_AT` controller, `_PI` pilot, `_PIAT` mixed). Note: Prague (LKPR) is also a test airport, so this corpus puts the
  test airport's phraseology (not its recordings) into training, as it did for GMAPS.

Leakage guard: a transcript whose normalised text equals any ATCO2 1h transcript (all 877, any split) is dropped.
Pseudo-labels are kept only when the output parses and every span is a verbatim substring of the transcript
(GMAPS's filter). The role comes from the corpus where it is known (ATCOSIM: controller; UWB-ATCC: the id suffix),
not from the teacher; mixed or unknown speakers are dropped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import atc
import run

EXT = atc.DATA / "external"
PSEUDO = atc.DATA / "pseudo"
SYNTH = atc.DATA / "synthetic" / "us_train.jsonl"
AUG = atc.DATA / "synthetic" / "atco2_augment.jsonl"
RB = atc.DATA / "synthetic" / "readbacks.jsonl"
SOURCES = {"atcosim": "Jzuluaga/atcosim_corpus", "uwb_atcc": "Jzuluaga/uwb_atcc"}
_BRACKETS = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)")


def clean(s: str) -> str:
    s = _BRACKETS.sub(" ", (s or "").lower()).replace("_", " ")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s).split())


def fetch() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    guard = {clean(r["text"]) for r in atc.read_jsonl(atc.SPLITS / "all.jsonl")}
    for name, repo in SOURCES.items():
        seen, rows, leaks = set(), [], 0
        for f in sorted(fs.glob(f"datasets/{repo}@~parquet/default/*/*.parquet")):
            with fs.open(f, "rb") as fh:
                t = pq.read_table(fh, columns=["id", "text"])
            for i, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=False):
                c = clean(text)
                if len(c.split()) < 3 or c in seen:
                    continue
                if c in guard:
                    leaks += 1
                    continue
                seen.add(c)
                role = "ATCO" if name == "atcosim" else {"AT": "ATCO", "PI": "PILOT"}.get(i.rsplit("_", 1)[-1])
                rows.append(
                    {
                        "id": f"{name}:{i}",
                        "source": name,
                        "role": role,
                        "text": c,
                        "facility": "ruzyne tower"
                        if "TWR" in i
                        else ("prague approach" if name == "uwb_atcc" else "upper area control"),
                    }
                )
        atc.write_jsonl(EXT / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} unique transcripts, {leaks} dropped by the ATCO2 leakage guard")


def verbatim(text: str, d: dict) -> bool:
    padded = f" {atc.norm(text)} "
    spans = atc.as_list(d.get("callsigns")) + atc.as_list(d.get("commands")) + atc.as_list(d.get("values"))
    return all(f" {atc.norm(s)} " in padded for s in spans if atc.norm(s))


def label(spec: str, n: int, k: int, workers: int, sources: list[str], tag: str = "") -> None:
    """Pseudo-label `n` transcripts per source; files are `<source>[.<tag>].jsonl` (tag names the teacher)."""
    import models

    budget = run.fin.Budget(run.ARTICLE, f"pseudo:{spec}")
    model = models.make(spec, budget)
    examples = atc.shots(k)
    PSEUDO.mkdir(parents=True, exist_ok=True)
    suffix = f".{tag}" if tag else ""
    for name in sources:
        pool = [r for r in atc.read_jsonl(EXT / f"{name}.jsonl") if r["role"]]
        sample = random.Random(0).sample(pool, min(n, len(pool)))
        path = PSEUDO / f"{name}{suffix}.raw.jsonl"
        done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
        todo = [r for r in sample if r["id"] not in done]

        def one(r):
            try:
                rep = model.chat(atc.messages(r, examples), max_tokens=6000)[0]
                return {**r, "output": rep.text}
            except SystemExit:
                raise
            except Exception as e:
                return {**r, "output": "", "error": repr(e)[:200]}

        def records(todo=todo):
            if not hasattr(model, "submit"):
                with ThreadPoolExecutor(min(workers, 4)) as ex:
                    yield from ex.map(one, todo)
                return
            for r, reps in models.map_chat(model, todo, lambda r: atc.messages(r, examples), max_tokens=6000):
                yield one(r) if isinstance(reps, Exception) else {**r, "output": reps[0].text}

        with path.open("a") as out:
            for i, rec in enumerate(records(), 1):
                out.write(json.dumps(rec) + "\n")
                if i % 200 == 0:
                    out.flush()
                    model.budget.flush() if model.budget is not budget else None
                    budget.flush()
                    print(f"  {name} {i}/{len(todo)} ${budget.usd:.2f}", flush=True)
        model.budget.flush() if model.budget is not budget else None
        budget.flush()
        kept = []
        raw = atc.read_jsonl(path)
        for r in raw:
            d = atc.parse_output(r["output"])
            if d is None or not verbatim(r["text"], d):
                continue
            kept.append(
                {
                    "id": r["id"],
                    "source": name,
                    "facility": r["facility"],
                    "text": r["text"],
                    "role": r["role"],
                    "gold": {
                        "callsign": atc.as_list(d.get("callsigns")),
                        "command": atc.as_list(d.get("commands")),
                        "value": atc.as_list(d.get("values")),
                    },
                }
            )
        atc.write_jsonl(PSEUDO / f"{name}{suffix}.jsonl", kept)
        print(
            f"{name}: kept {len(kept)}/{len(raw)} ({len(kept) / max(1, len(raw)):.0%}) verbatim pseudo-labels; "
            f"${budget.usd:.2f} so far"
        )


# --- pilot read-backs built from controller instructions (rule-based; gold spans by construction) ---------------

PILOT_FORM = {
    "descend": "descending",
    "climb": "climbing",
    "turn": "turning",
    "maintain": "maintaining",
    "reduce": "reducing",
    "increase": "increasing",
    "proceed": "proceeding",
    "contact": "",
    "squawk": "squawk",
    "cleared": "cleared",
    "direct": "direct",
    "hold": "holding",
    "taxi": "taxi",
    "line": "lining",
    "report": "wilco",
    "expect": "expecting",
    "continue": "continuing",
    "fly": "flying",
}


def readback(row: dict, rng: random.Random) -> dict | None:
    """A pilot read-back of a controller row: each command in its pilot form (descend -> descending, contact ->
    dropped), each value repeated as spoken, then the callsign; role PILOT. None when a command has no known form or
    the row has no single callsign. Teaches the pilot side without any Prague data."""
    g = row["gold"]
    if row.get("role") != "ATCO" or len(g["callsign"]) != 1 or not (g["command"] or g["value"]):
        return None
    cmds = []
    for c in g["command"]:
        first, *rest = c.lower().split()
        form = PILOT_FORM.get(first)
        if form is None:
            return None
        if form:
            cmds.append("wilco" if form == "wilco" else " ".join([form, *rest]))
    vals = list(g["value"])
    if cmds and rng.random() < 0.3:  # terse read-back: values only
        cmds = []
    text = " ".join(cmds + vals + [g["callsign"][0]])
    if rng.random() < 0.2:
        text = rng.choice(["roger", "wilco"]) + " " + text
    return {
        "id": f"rb:{row['id']}",
        "source": "readback",
        "facility": row["facility"],
        "text": text,
        "role": "PILOT",
        "gold": {"callsign": g["callsign"], "command": cmds, "value": vals},
    }


def readbacks(n: int) -> None:
    rng = random.Random(3)
    src = atc.read_jsonl(PSEUDO / "atcosim.q397.jsonl") + atc.read_jsonl(atc.SPLITS / "train.jsonl")
    rng.shuffle(src)
    out = []
    for r in src:
        rb = readback(r, rng)
        spans = (
            {"callsigns": rb["gold"]["callsign"], "commands": rb["gold"]["command"], "values": rb["gold"]["value"]}
            if rb
            else None
        )
        if rb and verbatim(rb["text"], spans):
            out.append(rb)
        if len(out) >= n:
            break
    atc.write_jsonl(RB, out)
    print(f"wrote {len(out)} read-backs; e.g.\n" + "\n".join(json.dumps(r) for r in out[:4]))


# --- synthetic US phraseology (FAA JO 7110.65 grammar; gold spans by construction) -------------------------------

DIGIT = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "niner"]
DIGIT_ATCO2 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ALPHA = [
    "alfa",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "x-ray",
    "yankee",
    "zulu",
]
AIRLINES = [
    "united",
    "american",
    "delta",
    "southwest",
    "jetblue",
    "alaska",
    "skywest",
    "frontier",
    "spirit",
    "cactus",
    "brickyard",
    "jazz",
    "envoy",
    "fedex",
    "speedbird",
    "lufthansa",
    "air canada",
    "republic",
]
FACILITIES = [
    "san francisco tower",
    "laguardia tower",
    "kennedy ground",
    "o'hare tower",
    "denver tower",
    "atlanta ground",
    "seattle tower",
    "boston tower",
    "norcal approach",
    "new york approach",
]


def spoken(n: int | str, nine: list[str]) -> str:
    return " ".join(nine[int(c)] for c in str(n))


def runway(rng: random.Random, nine) -> str:
    num = rng.randint(1, 36)
    side = rng.choice(["", "", " left", " right", " center"])
    return f"runway {spoken(f'{num:02d}', nine)}{side}"


def callsign(rng: random.Random, nine) -> str:
    if rng.random() < 0.2:
        letters = " ".join(rng.sample(ALPHA, rng.randint(1, 2)))
        return f"november {spoken(rng.randint(1, 999), nine)} {letters}"
    cs = f"{rng.choice(AIRLINES)} {spoken(rng.randint(1, 9999), nine)}"
    return cs + (" heavy" if rng.random() < 0.1 else "")


def taxiway(rng: random.Random) -> str:
    t = rng.choice(ALPHA[:12])
    return t + (f" {rng.choice(DIGIT_ATCO2[1:6])}" if rng.random() < 0.3 else "")


def freq(rng: random.Random) -> str:
    return f"one {spoken(rng.randint(18, 35), DIGIT_ATCO2)} point {spoken(rng.randint(0, 9), DIGIT_ATCO2)}"


def synth_one(rng: random.Random) -> dict:
    """An ATCO instruction (callsign first) or a pilot read-back of it (callsign last), with spans as tagged."""
    nine = DIGIT if rng.random() < 0.5 else DIGIT_ATCO2
    cs, rw, tw = callsign(rng, nine), runway(rng, nine), taxiway(rng)
    wind = f"wind {spoken(rng.randint(1, 36) * 10, DIGIT_ATCO2)} at {spoken(rng.randint(3, 25), DIGIT_ATCO2)}"
    say_wind = rng.random() < 0.4
    kind = rng.choice(
        ["land", "takeoff", "luaw", "hold_short", "cross", "taxi", "contact", "climb", "heading", "go_around", "vacate"]
    )
    # each kind: the instruction as (span kind, text) pieces in spoken order; "c" command, "v" value, "w" filler
    level = f"{spoken(rng.randint(2, 17), DIGIT_ATCO2)} thousand"
    turn = rng.choice(["turn left heading", "turn right heading"])
    pieces = {
        "land": [("v", rw), ("c", "cleared to land")] + ([("v", wind)] if say_wind else []),
        "takeoff": [("v", rw)] + ([("v", wind)] if say_wind else []) + [("c", "cleared for takeoff")],
        "luaw": [("v", rw), ("c", "line up and wait")],
        "hold_short": [("c", "hold short of"), ("v", rw)],
        "cross": [("c", "cross"), ("v", rw), ("w", "at"), ("v", tw)],
        "taxi": [("c", "taxi to"), ("v", rw), ("c", "via"), ("v", tw)],
        "contact": [("c", "contact"), ("v", f"departure {freq(rng)}")],
        "climb": [("c", "climb and maintain"), ("v", level)],
        "heading": [("c", turn), ("v", spoken(f"{rng.randint(1, 36) * 10:03d}", nine))],
        "go_around": [("c", "go around")],
        "vacate": [("c", "exit left at"), ("v", tw)],
    }[kind]
    pilot = rng.random() < 0.45
    if pilot:  # read-back: the wind is not read back, the callsign goes last
        pieces = [p for p in pieces if p[1] != wind]
    instr = " ".join(t for _, t in pieces)
    text = f"{instr} {cs}" if pilot else f"{cs} {instr}"
    return {
        "id": f"synth:{rng.random():.12f}",
        "source": "synthetic",
        "facility": rng.choice(FACILITIES),
        "text": text,
        "role": "PILOT" if pilot else "ATCO",
        "gold": {
            "callsign": [cs],
            "command": [t for k, t in pieces if k == "c"],
            "value": [t for k, t in pieces if k == "v"],
        },
    }


# --- augmentation of the ATCO2 train split (gold spans carried over; verified by construction) -------------------


def augment(n: int) -> None:
    """New rows from the 504 train rows: the callsign span is swapped for another train callsign (same casing style),
    and a controller row that opens with its callsign is, half the time, turned into a pilot read-back (callsign moved
    to the end, role PILOT). Spans stay verbatim. Never touches dev or test."""
    rng = random.Random(1)
    train = atc.read_jsonl(atc.SPLITS / "train.jsonl")
    pool = sorted({c for r in train for c in r["gold"]["callsign"] if len(c.split()) >= 2})
    usable = [r for r in train if len(r["gold"]["callsign"]) == 1 and r["gold"]["callsign"][0] in r["text"]]
    rows = []
    while len(rows) < n:
        r = rng.choice(usable)
        old, new = r["gold"]["callsign"][0], rng.choice(pool)
        text, role = r["text"].replace(old, new), r["role"]
        if role == "ATCO" and text.startswith(new) and rng.random() < 0.5:
            rest = text[len(new) :].strip()
            if rest:
                text, role = f"{rest} {new}", "PILOT"
        g = {"callsign": [new], "command": r["gold"]["command"], "value": r["gold"]["value"]}
        if not verbatim(text, {"callsigns": g["callsign"], "commands": g["command"], "values": g["value"]}):
            continue
        rows.append(
            {
                "id": f"aug:{len(rows)}",
                "source": "augment",
                "facility": r["facility"],
                "text": text,
                "role": role,
                "gold": g,
            }
        )
    atc.write_jsonl(AUG, rows)
    flips = sum(r["role"] == "PILOT" for r in rows)
    print(f"wrote {len(rows)} augmented rows ({flips} pilot); e.g.\n" + "\n".join(json.dumps(r) for r in rows[:3]))


def synth(n: int) -> None:
    rng = random.Random(0)
    rows = [synth_one(rng) for _ in range(n)]
    atc.write_jsonl(SYNTH, rows)
    print(f"wrote {n} synthetic rows; e.g.\n" + "\n".join(json.dumps(r) for r in rows[:4]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "label", "synth", "augment", "readbacks"])
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--tag", default="", help="teacher tag in the pseudo-label file names")
    ap.add_argument("--model", default="gpt-5.4:none")
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.step == "fetch":
        fetch()
    elif a.step == "label":
        label(a.model, a.n, a.shots, a.workers, a.sources, a.tag)
    elif a.step == "augment":
        augment(a.n)
    elif a.step == "readbacks":
        readbacks(a.n)
    else:
        synth(a.n)


if __name__ == "__main__":
    sys.exit(main())
