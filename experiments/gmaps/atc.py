"""GMAPS replication: ATCO2 data and split, the task prompt, the action table, the scorer, statistics.

Reference: the Fireworks: Deploy 2nd-place project GMAPS (github.com/francolinx/GMAPS_Fireworks, read 2026-11-11; it
has no licence file, so this is our own code following its README's definitions, not a copy).

Task: one ATC radio transmission (ASR-style transcript plus the facility it was heard on) -> JSON
{role, callsigns, commands, values}, spans as spoken.

Data: ATCO2-test-set-1h (Idiap / ATCO2 project, atco2.org/data, "available for research purposes"; not redistributed,
so `data/` is gitignored). 560 recordings from 7 airports, 877 segments with gold callsign / command / value tags.
Split by airport, as GMAPS describes: test = Sydney (YSSY), Prague (LKPR), Bratislava (LZIB); the other four airports
(Bern, Sion, Zurich, Brno) give dev = every 7th of their segments in filename order, train = the rest.

Headline metric, full-record exact: role right (rows whose gold role is ATCO or PILOT; an UNKNOWN gold role is not
scored), callsign multiset equal after text normalisation, and the operational action set equal. The action set maps
every command span to a canonical action with an ordered rule table; ACK (acknowledgements, read-back filler,
requests) and OTHER (unmapped) are dropped from gold and prediction alike, as GMAPS fixed before its test run.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW_XML = DATA / "raw" / "xml"
SPLITS = DATA / "splits"
RESULTS = HERE / "results"
TEST_AIRPORTS = ("YSSY", "LKPR", "LZIB")
DEV_EVERY = 7
TAG = re.compile(r"\[#(callsign|command|value)\](.*?)\[/#\1\]", re.DOTALL)
UNNAMED = re.compile(r"\[/?#unnamed\]")


# --- data ---------------------------------------------------------------------------------------------------------


def role_of(label: str) -> str:
    """ATCO2 labels controllers `ATCO ...`, unknown speakers `UNK...`, and pilots by their callsign."""
    label = (label or "").strip().upper()
    if not label or label.startswith("UNK"):
        return "UNKNOWN"
    return "ATCO" if label.startswith("ATCO") else "PILOT"


def parse_recordings(xml_dir: Path = RAW_XML) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(xml_dir / "*.xml"))):
        stem = Path(path).stem
        parts = stem.split("_")
        facility = " ".join(p for p in parts[1:4] if not p[:1].isdigit())
        for i, seg in enumerate(ET.parse(path).getroot().findall("segment")):
            tagged = (seg.findtext("text") or "").strip()
            gold = {"callsign": [], "command": [], "value": []}
            for kind, span in TAG.findall(tagged):
                gold[kind].append(" ".join(span.split()))
            text = " ".join(UNNAMED.sub("", TAG.sub(lambda m: m.group(2), tagged)).split())
            label = (seg.findtext("speaker_label") or "").strip()
            rows.append(
                {
                    "id": f"{stem}#{i}",
                    "recording": stem,
                    "airport": parts[0],
                    "facility": facility,
                    "speaker_label": label,
                    "role": role_of(label),
                    "text": text,
                    "gold": gold,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_splits() -> dict:
    rows = parse_recordings()
    test = [r for r in rows if r["airport"] in TEST_AIRPORTS]
    rest = [r for r in rows if r["airport"] not in TEST_AIRPORTS]
    dev = [r for i, r in enumerate(rest) if i % DEV_EVERY == 0]
    train = [r for i, r in enumerate(rest) if i % DEV_EVERY != 0]
    for name, part in (("all", rows), ("train", train), ("dev", dev), ("test", test)):
        write_jsonl(SPLITS / f"{name}.jsonl", part)
    return {
        "all": len(rows),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "test_recordings": len({r["recording"] for r in test}),
        "test_sha256": sha256(SPLITS / "test.jsonl"),
    }


TEST_HASH_FILE = RESULTS / "test.sha256"


def load(split: str) -> list[dict]:
    """`train`, `dev`, `test` or `trainval`; the test file must match the hash frozen before any model saw it.

    `trainval` is 200 train rows (fixed sample, never the 20 few-shot rows). It is held out only for prompted
    systems: fine-tuned models trained on it, so it never scores them."""
    if split == "trainval":  # 200 train rows outside the 20 few-shot rows: a larger check for prompted systems only
        return random.Random(0).sample(read_jsonl(SPLITS / "train.jsonl")[20:], 200)
    path = SPLITS / f"{split}.jsonl"
    if split == "test":
        frozen = TEST_HASH_FILE.read_text().split()[0]
        if sha256(path) != frozen:
            raise SystemExit("test.jsonl does not match the frozen hash; refusing to run")
    return read_jsonl(path)


# --- prompt -------------------------------------------------------------------------------------------------------

SYSTEM = """You parse air traffic control radio. The input is ONE radio transmission, transcribed the way a speech \
recogniser writes it, plus the facility where it was heard. Return its content as JSON.

Return a single JSON object with exactly four keys:
{"role": "ATCO" | "PILOT" | "UNKNOWN", "callsigns": [...], "commands": [...], "values": [...]}

- role: ATCO when the controller is speaking (giving instructions or information); PILOT when an aircraft is speaking \
(read-backs, requests, reports). UNKNOWN only when it cannot be decided.
- callsigns: each aircraft callsign, word for word as spoken (e.g. "swiss one one five zulu", "oscar kilo papa mike \
bravo"). Do not rewrite it as an ICAO code. A station name such as "bern tower" is not a callsign.
- commands: each instruction or action phrase, word for word (e.g. "cleared to land", "line up and wait", \
"hold short of", "contact", "descend", "cleared for takeoff"). A pilot's read-back of an instruction is a command too \
("landing", "lining up"). There may be several, or none.
- values: the parameters of those commands, word for word: runways ("runway one four"), levels and altitudes \
("flight level one hundred"), frequencies ("one two one decimal seven"), headings, speeds, fixes, taxiways. \
Callsigns never go here.

Copy the spoken words exactly; do not paraphrase, expand or normalise. Output the JSON object and nothing else."""


# Short prompt for the fine-tuned arms (the conventions are learned from training data, not read from the prompt).
SYSTEM_FT = (
    "Parse one air traffic control radio transmission into JSON with keys role (ATCO, PILOT or UNKNOWN), "
    "callsigns, commands and values. Copy every span word for word."
)


def user_message(row: dict) -> str:
    prev = f"previous transmission on this frequency: {row['prev']}\n" if "prev" in row else ""
    return f"facility: {row['facility'].lower()}\n{prev}transmission: {row['text']}"


CONTEXT_NOTE = (
    '\n\nThe previous transmission heard on the same frequency (or "none") is given for context: it can tell '
    "you who is speaking now (an instruction is often followed by the pilot's read-back). Parse only the "
    "transmission itself."
)


def with_context(rows: list[dict]) -> list[dict]:
    """Add the transcript of the previous segment in the same recording (text only, no labels) as `prev`. Causal:
    a live system has heard the previous transmission. A separate task condition from GMAPS's single transmission."""
    text = {r["id"]: r["text"] for r in read_jsonl(SPLITS / "all.jsonl")}
    out = []
    for r in rows:
        rec, i = r["id"].rsplit("#", 1)
        out.append({**r, "prev": text.get(f"{rec}#{int(i) - 1}", "none")})
    return out


def target(row: dict) -> dict:
    g = row["gold"]
    return {
        "role": row.get("role", "UNKNOWN"),
        "callsigns": g["callsign"],
        "commands": g["command"],
        "values": g["value"],
    }


def messages(row: dict, shots: list[dict], system: str = SYSTEM) -> list[dict]:
    """System prompt, `shots` worked examples as user/assistant turns, then the transmission."""
    out = [{"role": "system", "content": system}]
    for s in shots:
        out += [
            {"role": "user", "content": user_message(s)},
            {"role": "assistant", "content": json.dumps(target(s), ensure_ascii=False)},
        ]
    return [*out, {"role": "user", "content": user_message(row)}]


def shots(k: int) -> list[dict]:
    """The first k rows of the train split: the same examples for every arm, never from the scored split."""
    return read_jsonl(SPLITS / "train.jsonl")[:k] if k else []


class Retriever:
    """Nearest train rows by TF-IDF cosine over word 1-2-grams of the transcript (lower-cased). Used for
    retrieval-augmented few-shot: the k most similar train rows, most similar last (closest to the query)."""

    def __init__(self, exclude_ids: set[str] | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.pool = [r for r in read_jsonl(SPLITS / "train.jsonl") if r["id"] not in (exclude_ids or set())]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.mat = self.vec.fit_transform([r["text"] for r in self.pool])

    def shots(self, row: dict, k: int) -> list[dict]:
        sims = (self.mat @ self.vec.transform([row["text"]]).T).toarray().ravel()
        order = [i for i in sims.argsort()[::-1] if self.pool[i]["id"] != row["id"]][:k]
        return [self.pool[i] for i in reversed(order)]


# --- the action table ---------------------------------------------------------------------------------------------
# Ordered: the first rule whose pattern matches the lower-cased span wins. Runway-state actions come first, then a
# few guards for phrases that contain a runway word without being that clearance.

RUNWAY_CRITICAL = frozenset(
    {
        "CLEARED_TO_LAND",
        "CLEARED_FOR_TAKEOFF",
        "LINE_UP_AND_WAIT",
        "HOLD_SHORT",
        "HOLD_POSITION",
        "CROSS_RUNWAY",
        "BACKTRACK",
        "TOUCH_AND_GO",
        "VACATE_RUNWAY",
        "GO_AROUND",
    }
)
NON_ACTIONS = frozenset({"ACK", "OTHER"})

_RULES: list[tuple[str, str]] = [
    # not instructions, even when they name a departure or a call
    ("ACK", r"\bsay again\b|\bstand ?by\b|\bcall you\b|\bpass your message\b|^pass$"),
    ("GO_AROUND", r"\bgo(ing)?[- ]?around\b|\bbreaking off\b"),
    ("TOUCH_AND_GO", r"\b(touch|stop|in) and go\b"),
    ("LINE_UP_AND_WAIT", r"\blin(e|ing) up\b"),
    ("HOLD_SHORT", r"\bhold(ing)? short\b"),
    ("BACKTRACK", r"\bbacktrack"),
    ("VACATE_RUNWAY", r"\bvaca|\bclear of\b|^exit$"),
    # readiness is a report, not a takeoff clearance
    ("REPORT", r"\bready for departure\b|\breport ready\b|^ready (for|to)$"),
    ("CLEARED_FOR_TAKEOFF", r"\btake ?off\b|\bdepart"),
    ("CLEARED_TO_LAND", r"\bland(ing)?\b|\bfull stop\b|\btouchdown\b"),
    # "climb ... crossing" is a level change, so climb and descend come before cross
    ("CLIMB", r"\bclimb"),
    ("DESCEND", r"\bdescen|\bin descent\b"),
    ("CROSS_RUNWAY", r"\bcross"),
    ("HOLD_POSITION", r"\bhold(ing)? position\b|\bhold on\b|^hold(ing)?$|\bholding point\b|\bwait(ing)?\b|^stop$"),
    (
        "STARTUP_PUSHBACK",
        (
            r"\bstart ?up\b|\bpush ?back\b|\bpush and start\b|\bengine start\b|\bstart (and push|approved)"
            r"\b|^starting$|\brun up\b|\bready for ifr\b|\bifr clearance\b"
        ),
    ),
    (
        "TAXI",
        (
            r"\btaxi|\bpark(ing)?\b|\benter via\b|\b(next|turn) left via\b|\bfollow( behind)?$|\bproceed normally\b|"
            r"\bto the left$"
        ),
    ),
    (
        "CONTACT_FREQUENCY",
        (
            r"\bcontact\b|\bcontract\b|\bfrequency\b|\bleave the command\b|\bcall me on\b|\bconnect\b|"
            r"^call$|\bmay leave\b|\bchange frequency\b|\bleave out on\b"
        ),
    ),
    ("REPORT", r"\breport|\bnext final\b|^final$|\bfinal runway\b"),
    ("MAINTAIN", r"\bmaintain\b"),
    ("SPEED", r"\bspeed\b|^maximum$"),
    (
        "NAVIGATE",
        (
            r"\bheading?|\bturn|^fly$|\btourning\b|\bproceed|\bdirect\b|\bstraight ahead\b|\borbit\b|\bvector|"
            r"\bcircl|\bjoin|\bupwind\b|\boutbound\b|\binbound to\b|\benter control zone\b|\bcircuit|"
            r"\bshortcut\b|^next$"
        ),
    ),
    (
        "APPROACH_CLEARANCE",
        (
            r"\bapproach\b|\bils\b|\bestablish|\betablished\b|\bstraight in\b|\bdme\b|\bexpect "
            r"(to land|landing)\b|\bcontinue( to| on| for| toward)?$|\brnp\b|\barp\b|\bvisual\b"
        ),
    ),
    ("ROUTE_CLEARANCE", r"\brecleared\b|\bcleared to\b|\bcleared( for)?$|\bclearance\b|\brequested change\b"),
    (
        "POSITION_REPORT",
        (
            r"\bpassing\b|\bapproaching\b|\bleaving\b|\bentering\b|\bidentified\b|\binbound\b|"
            r"\bstop established\b"
        ),
    ),
    (
        "ACK",
        (
            r"\bconfirm|^say$|^again$|\bcop(y|ied)\b|\baffirm|\bapproved?\b|\bgo ahead\b|\bnegative\b|\becho\b|"
            r"\bbreak break\b|\bsorry\b|\banswer with\b|\broger\b|\bwilco\b|\bready\b|\brequest|\bwant to\b|\bintend\b|"
            r"\bexpect\b|\bavailable\b|\bnow active\b|\blooking\b|\bseaching\b|\bremain\b|\bstay clear\b|\bbelow\b|"
            r"\bindicating\b|\bis calling\b|\bto compose\b|\bkeep the\b|\bout of\b|\bredue\b|^make$|"
            r"\bmake your\b|\bwill\b|"
            r"\bable to\b"
        ),
    ),
]
_COMPILED = [(a, re.compile(p)) for a, p in _RULES]


def action(span: str) -> str:
    s = " ".join(re.sub(r"\[hes\]|\(-ding\)", "", (span or "").lower()).split())
    return next((a for a, rx in _COMPILED if rx.search(s)), "OTHER")


def actions(spans) -> set[str]:
    return {action(s) for s in spans or []} - NON_ACTIONS


# --- parsing model output -----------------------------------------------------------------------------------------


def parse_output(raw) -> dict | None:
    """A dict, a JSON string, fenced JSON, or text containing one JSON object (the schema has no nesting)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.split("</think>")[-1].strip(), flags=re.DOTALL)
    candidates = [s] + [m.group(0) for pat in (r"\{.*\}", r"\{.*?\}") if (m := re.search(pat, s, re.DOTALL))]
    for c in candidates:
        try:
            d = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict):
            return d
    return None


def as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    return [str(x)]


_PUNCT = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", (s or "").lower()).split())


# --- scoring ------------------------------------------------------------------------------------------------------


def _counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    g, p = collections.Counter(gold), collections.Counter(pred)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def score_row(gold_row: dict, output) -> dict:
    """Every per-row piece the aggregate metrics are built from."""
    d = parse_output(output)
    valid = d is not None
    d = d or {}
    g = gold_row["gold"]
    gold_spans = {k: [norm(x) for x in g[k]] for k in ("callsign", "command", "value")}
    pred_spans = {k: [norm(x) for x in as_list(d.get(k + "s"))] for k in ("callsign", "command", "value")}
    ga, pa = actions(g["command"]), actions(as_list(d.get("commands")))
    gc, pc = ga & RUNWAY_CRITICAL, pa & RUNWAY_CRITICAL
    scored_role = gold_row["role"] in ("ATCO", "PILOT")
    role_ok = (str(d.get("role", "")).upper() == gold_row["role"]) if scored_role else None
    cs_ok = collections.Counter(gold_spans["callsign"]) == collections.Counter(pred_spans["callsign"])
    return {
        "id": gold_row["id"],
        "valid": valid,
        "spans": {k: _counts(gold_spans[k], pred_spans[k]) for k in gold_spans},
        "act": (len(ga & pa), len(pa - ga), len(ga - pa)),
        "act_ok": ga == pa,
        "crit_ok": (gc == pc) if gc else None,
        "role_ok": role_ok,
        "cs_ok": cs_ok,
        "full": bool(cs_ok and ga == pa and role_ok is not False),
    }


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def aggregate(rows: list[dict]) -> dict:
    def summed(get):
        return [sum(get(r)[i] for r in rows) for i in range(3)]

    crit = [r["crit_ok"] for r in rows if r["crit_ok"] is not None]
    role = [r["role_ok"] for r in rows if r["role_ok"] is not None]
    return {
        "n": len(rows),
        "full_record": mean(r["full"] for r in rows),
        "clearance_state": mean(crit),
        "clearance_n": len(crit),
        "action_f1": f1(*summed(lambda r: r["act"])),
        "callsign_f1": f1(*summed(lambda r: r["spans"]["callsign"])),
        "command_f1": f1(*summed(lambda r: r["spans"]["command"])),
        "value_f1": f1(*summed(lambda r: r["spans"]["value"])),
        "role_acc": mean(role),
        "json_valid": mean(r["valid"] for r in rows),
    }


def score(gold: list[dict], preds: dict[str, object]) -> list[dict]:
    """`preds` maps id -> raw output; a missing id is scored as an empty answer."""
    return [score_row(g, preds.get(g["id"])) for g in gold]


# --- statistics ---------------------------------------------------------------------------------------------------


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar: rows only A got right, rows only B got right, and the binomial p-value."""
    only_a = sum(x and not y for x, y in zip(a, b, strict=False))
    only_b = sum(y and not x for x, y in zip(a, b, strict=False))
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return only_a, only_b, p


def cluster_bootstrap(
    ids: list[str], a: list[float], b: list[float] | None = None, iters: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95% interval of mean(a) (or of mean(a) - mean(b), paired) resampling whole recordings with replacement."""
    groups = collections.defaultdict(list)
    for j, i in enumerate(ids):
        groups[i.split("#")[0]].append(j)
    keys = list(groups)
    rng = random.Random(seed)
    stats = []
    for _ in range(iters):
        idx = [j for k in (rng.choice(keys) for _ in keys) for j in groups[k]]
        m = sum(a[j] for j in idx) / len(idx)
        if b is not None:
            m -= sum(b[j] for j in idx) / len(idx)
        stats.append(m)
    stats.sort()
    return stats[int(0.025 * (iters - 1))], stats[int(0.975 * (iters - 1))]


def compare(ids: list[str], a: list[bool], b: list[bool]) -> dict:
    """Paired full-record comparison: gap, McNemar, recording-bootstrap interval; significant only if both agree."""
    only_a, only_b, p = mcnemar(a, b)
    lo, hi = cluster_bootstrap(ids, [float(x) for x in a], [float(x) for x in b])
    gap = mean(a) - mean(b)
    sig = p < 0.05 and (lo > 0 or hi < 0)
    return {"gap": gap, "only_a": only_a, "only_b": only_b, "p": p, "ci": (lo, hi), "significant": sig}


if __name__ == "__main__":
    os.chdir(HERE)
    print(json.dumps(build_splits(), indent=1))
