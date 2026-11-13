"""B3, data v3: new human fiction and long texts, and 25,000 new AI edits aimed at the light and long edits we miss.

Pre-registered in docs/plans/2026-11-12-slop-detector-b3-prereg.md. Steps, in order:

    python gen_v3.py humans                # data/v3/human.jsonl: 2,000 + 2,000 r/nosleep posts, 4,000 long Medium posts
    python gen_v3.py plan                  # data/v3/jobs.jsonl: which document gets which edit, by which editor
    python gen_v3.py estimate              # token counts on a sample (free) -> a dollar estimate per editor
    python gen_v3.py submit --editor haiku # Anthropic Message Batches (50% off); batch ids in data/v3/batches.json
    python gen_v3.py collect               # finished batches -> data/v3/ai.jsonl, spend -> results/v3_generation.json
    python gen_v3.py label                 # lesson 03 filters + the B1 labels -> data/v3b/{train,val,val_v3,test}.jsonl

The editors are Claude Haiku 5.5 and Claude Sonnet 5.5, with thinking off (the user allowed Claude editors for data v3
on 2026-11-12; a text detector is not a competing model). Every row records its editor in `model` and `editor`, so the
note can report the Claude share and the Claude-edited benchmark rows separately.

r/nosleep comes from ElcaidaLabs/NosleepReddit100K, fetched on Modal and kept only for Reddit ids below "y00000":
Reddit ids are a base-36 counter, and that id was reached in about October 2022, before ChatGPT. r/WritingPrompts is
still never used (an EditLens source).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from importlib import import_module

from common import DATA, RESULTS, prose_words, read_jsonl

V3 = DATA / "v3"
OUT = DATA / "v3b"
EDITORS = {  # name: (model id, $ in per M, $ out per M at the 50% batch price)
    "haiku": ("claude-haiku-5-5", 0.05, 0.25),
    "sonnet": ("claude-sonnet-5-5", 1.0, 5.0),
}
THINKING_OFF = {  # no reasoning before the edit; each model's documented way to turn it off
    "haiku": {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
    "sonnet": {
        "thinking": {"type": "between_tools"},
        "output_config": {"effort": "low"},
    },
}
GEN = import_module("02_generate")
HUM = import_module("01_human")
LIGHT = (
    *GEN.REWRITES[:3],
    *GEN.LIGHT,
    "Fix the typos in this text. Change nothing else.",
    "Tighten the wording of this text: cut needless words, but keep its content, order and voice.",
)
HEAVY = (
    (
        "Rewrite this text thoroughly so it reads as polished, professional prose. Keep all of its content, facts and "
        "order."
    ),
    "Rewrite this text in your own words, sentence by sentence, keeping everything it says.",
    (
        "Improve the writing of this text substantially: restructure sentences and choose better words, but keep every "
        "point it makes."
    ),
    GEN.REWRITES[4],
    GEN.REWRITES[6],
)


def contamination() -> set[int]:
    """10-word runs of every benchmark text (lesson 03) plus every row of the frozen fresh test."""
    lab = import_module("03_label")
    bench = lab.benchmark_shingles()
    fresh = import_module("fresh_test").fresh_rows()
    for col in ("text", "source_text"):
        if col in fresh:
            for t in fresh[col].dropna():
                bench |= lab.shingles(str(t))
    return bench


def humans(args) -> None:
    lab = import_module("03_label")
    bench = contamination()
    have = {r["id"] for r in read_jsonl(DATA / "human.jsonl")}
    dropped, out = Counter(), []

    def ok(text: str) -> bool:
        if prose_words(text) < lab.MIN_WORDS or not lab.english(text):
            dropped.update(["not English or too short"])
            return False
        if len(lab.shingles(text) & bench) >= 3:
            dropped.update(["overlaps a benchmark or fresh-test text"])
            return False
        return True

    posts = read_jsonl(V3 / "nosleep_raw.jsonl")  # {"id", "title", "text"}, fetched on Modal
    random.Random(3).shuffle(posts)
    quota = {"nosleep": args.fiction // 2, "nosleep_long": args.fiction // 2}
    got = Counter()
    for p in posts:
        if all(got[k] >= n for k, n in quota.items()):
            break
        label = "nosleep_long" if got["nosleep_long"] < quota["nosleep_long"] else "nosleep"
        text = HUM.trim(HUM.URL.sub("", p["text"]).strip(), 800 if label == "nosleep_long" else 350)
        if label == "nosleep_long" and prose_words(text) < 600:
            if got["nosleep"] >= quota["nosleep"]:
                continue
            label, text = "nosleep", HUM.trim(HUM.URL.sub("", p["text"]).strip(), 350)
        if ok(text):
            out.append(
                {
                    "id": f"nosleep/{p['id']}",
                    "source": label,
                    "year": 2022,
                    "text": text,
                }
            )
            got[label] += 1
    n_med = 0
    for r in HUM.medium():
        if n_med >= args.medium:
            break
        if r["id"] in have:
            continue
        text = HUM.trim(HUM.URL.sub("", r["text"]).strip(), 800)
        if prose_words(text) >= 600 and ok(text):
            out.append({**r, "source": "medium_v3long", "text": text})
            n_med += 1
    V3.mkdir(parents=True, exist_ok=True)
    with (V3 / "human.jsonl").open("w") as f:
        for r in out:
            r.update(split=HUM.split_of(r["id"]), words=prose_words(r["text"]))
            f.write(json.dumps(r) + "\n")
    print(dict(Counter(r["source"] for r in out)), "dropped", dict(dropped))


def v2_docs() -> list[dict]:
    """data v2's human documents that survived lesson 03 (they have a human row in data/{split}.jsonl)."""
    kept = {
        r["doc_id"] for s in ("train", "val", "test") for r in read_jsonl(DATA / f"{s}.jsonl") if r["kind"] == "human"
    }
    return [d for d in read_jsonl(DATA / "human.jsonl") if d["id"] in kept]


def prompt_for(doc: dict, version: str, rng: random.Random) -> tuple[str, dict]:
    if version == "light":
        i = rng.randrange(len(LIGHT))
        return f"{LIGHT[i]}\n\nTEXT:\n{doc['text']}", {"instruction": 200 + i}
    if version == "heavy":
        i = rng.randrange(len(HEAVY))
        return f"{HEAVY[i]}\n\nTEXT:\n{doc['text']}", {"instruction": 300 + i}
    n_sent = len(re.split(r"(?<=[.!?])\s+", doc["text"].strip()))
    k = max(2, round(n_sent * rng.choice((1 / 4, 1 / 3, 1 / 2))))
    return (
        (
            f"Improve the wording of only {k} of the {n_sent} sentences in this text (your choice which). "
            f"Spread them over the whole text: some near the beginning, some in the middle, some near the end. "
            f"Leave every other sentence exactly as it is, word for word, and return the whole text."
            f"\n\nTEXT:\n{doc['text']}"
        ),
        {"sentences": k},
    )


def plan(args) -> None:
    new = read_jsonl(V3 / "human.jsonl")
    old = v2_docs()
    random.Random("v3plan").shuffle(old)
    jobs: list[tuple[dict, str]] = []
    jobs += [(d, "light") for d in old[: args.light]]
    jobs += [(d, "heavy") for d in old[args.light : args.light + args.heavy]]
    jobs += [(d, "partial") for d in new if d["source"] in ("nosleep_long", "medium_v3long")]
    fiction = [d for d in new if d["source"].startswith("nosleep")]
    jobs += [(d, ("light", "partial", "heavy")[i % 3]) for i, d in enumerate(fiction)]
    # Sonnet costs 20x Haiku per token, so its ~3,000 edits go to texts of 400 words or fewer (the budget is $13)
    short = [i for i, (d, _) in enumerate(jobs) if d["words"] <= 400]
    to_sonnet = set(random.Random("v3sonnet").sample(short, min(args.sonnet, len(short))))
    seen: Counter = Counter()
    with (V3 / "jobs.jsonl").open("w") as f:
        for n, (d, v) in enumerate(jobs):
            seen[(d["id"], v)] += 1
            key = f"{d['id']}|v3{v}" + (str(seen[(d["id"], v)]) if seen[(d["id"], v)] > 1 else "")
            r = random.Random(key)
            text, extra = prompt_for(d, v, r)
            row = {
                "key": key,
                "cid": hashlib.sha1(key.encode()).hexdigest()[:40],
                "doc_id": d["id"],
                "source": d["source"],
                "split": d["split"],
                "version": v,
                "editor": "sonnet" if n in to_sonnet else "haiku",
                "prompt": text,
                "max_tokens": min(
                    6000, max(int(d["words"] * 2.6), len(d["text"]) // 2) + 400
                ),  # code counts as characters
                "words": d["words"],
                **extra,
            }
            f.write(json.dumps(row) + "\n")
    js = read_jsonl(V3 / "jobs.jsonl")
    print(len(js), "jobs", dict(Counter((j["version"], j["editor"]) for j in js)))


def redo(args) -> None:
    """Haiku's first partial edits of long texts mostly changed punctuation only (84% under 2% of words). Re-ask, once,
    with a firmer instruction; appended to data/v3/jobs.jsonl under new keys ("...|v3partial-r2")."""
    lab = import_module("03_label")
    src = {r["id"]: r for r in read_jsonl(V3 / "human.jsonl")}
    jobs = read_jsonl(V3 / "jobs.jsonl")
    have = {j["key"] for j in jobs}
    weak = [
        r
        for r in read_jsonl(V3 / "ai.jsonl")
        if r["version"] == "partial"
        and r["editor"] == "haiku"
        and lab.frac_changed(src[r["doc_id"]]["text"], r["text"]) < lab.MIN_CHANGE
    ]
    new = []
    for r in weak:
        key = f"{r['doc_id']}|v3partial-r2"
        if key in have:
            continue
        d = src[r["doc_id"]]
        n_sent = len(re.split(r"(?<=[.!?])\s+", d["text"].strip()))
        k = r["sentences"]
        new.append(
            {
                "key": key,
                "cid": hashlib.sha1(key.encode()).hexdigest()[:40],
                "doc_id": d["id"],
                "source": d["source"],
                "split": d["split"],
                "version": "partial",
                "editor": "haiku",
                "prompt": (
                    f"Choose {k} of the {n_sent} sentences in this text, spread over the whole text (some near the "
                    f"beginning, some in the middle, some near the end). Rewrite each chosen sentence fully in new "
                    f"words, as a skilled editor would: new wording and structure, same meaning. Copy every other "
                    f"sentence exactly, word for word, and return the whole text.\n\nTEXT:\n{d['text']}"
                ),
                "max_tokens": min(6000, max(int(d["words"] * 2.6), len(d["text"]) // 2) + 400),
                "words": d["words"],
                "sentences": k,
                "redo": 1,
            }
        )
    new = new[: args.limit or None]
    with (V3 / "jobs.jsonl").open("a") as f:
        f.writelines(json.dumps(j) + "\n" for j in new)
    print(f"{len(weak)} weak partial edits; {len(new)} new jobs appended")


SEP = re.compile(r"((?<=[.!?])\s+)")


def splice_jobs(args) -> None:
    """Haiku 5.5 returned most long texts verbatim when asked to edit only some sentences (84% under 2% changed, and
    a firmer prompt still 80%). So for those texts we choose the sentences ourselves, spread evenly over the text, ask
    Haiku to rewrite just those (numbered), and splice the rewrites back in: the edit is still the model's words, and
    the spread is guaranteed. Keys "...|v3partial-sp"."""
    lab = import_module("03_label")
    src = {r["id"]: r for r in read_jsonl(V3 / "human.jsonl")}
    jobs = read_jsonl(V3 / "jobs.jsonl")
    have = {j["key"] for j in jobs}
    done_ok = {
        r["doc_id"]
        for r in read_jsonl(V3 / "ai.jsonl")
        if r["version"] == "partial" and lab.frac_changed(src[r["doc_id"]]["text"], r["text"]) >= lab.MIN_CHANGE
    }
    partial_docs = {j["doc_id"] for j in jobs if j["version"] == "partial"}
    new = []
    for d in src.values():
        key = f"{d['id']}|v3partial-sp"
        if d["id"] not in partial_docs or d["id"] in done_ok or key in have:
            continue
        sents = SEP.split(d["text"].strip())[::2]
        rng = random.Random(key)
        k = max(2, round(len(sents) * rng.choice((1 / 4, 1 / 3, 1 / 2))))
        step = len(sents) / k
        idx = sorted({min(len(sents) - 1, int(i * step + rng.random() * step)) for i in range(k)})
        listing = "\n".join(f"{n + 1}. {sents[i]}" for n, i in enumerate(idx))
        instr = rng.choice(
            (
                "Rewrite each numbered sentence below in new words: same meaning, better wording.",
                "Improve the wording of each numbered sentence below, as a careful editor would.",
                "Rewrite each numbered sentence below so it reads more smoothly and vividly; keep its meaning.",
            )
        )
        new.append(
            {
                "key": key,
                "cid": hashlib.sha1(key.encode()).hexdigest()[:40],
                "doc_id": d["id"],
                "source": d["source"],
                "split": d["split"],
                "version": "partial",
                "editor": "haiku",
                "prompt": (
                    f"{instr} Reply with the same numbered list, one rewritten sentence per line, nothing else."
                    f"\n\n{listing}"
                ),
                "max_tokens": 400 + 80 * len(idx) + len(listing) // 2,
                "words": d["words"],
                "sentences": len(idx),
                "splice": idx,
            }
        )
    new = new[: args.limit or None]
    with (V3 / "jobs.jsonl").open("a") as f:
        f.writelines(json.dumps(j) + "\n" for j in new)
    print(f"{len(new)} splice jobs appended")


def spliced(doc_text: str, idx: list[int], reply: str) -> str | None:
    """Put the numbered rewrites back in place; None unless the reply has exactly one line per sentence."""
    lines = {}
    for ln in reply.splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s+(.*\S)", ln)
        if m:
            lines[int(m.group(1))] = m.group(2)
    if sorted(lines) != list(range(1, len(idx) + 1)):
        return None
    parts = SEP.split(doc_text.strip())
    for n, i in enumerate(idx):
        parts[2 * i] = lines[n + 1]
    return "".join(parts)


def params(j: dict) -> dict:
    return {
        "model": EDITORS[j["editor"]][0],
        "max_tokens": j["max_tokens"],
        "system": GEN.SYSTEM,
        "messages": [{"role": "user", "content": j["prompt"]}],
        **THINKING_OFF[j["editor"]],
    }


def estimate(args) -> None:
    import anthropic

    client = anthropic.Anthropic()
    js = read_jsonl(V3 / "jobs.jsonl")
    total = 0.0
    for name, (model, pin, pout) in EDITORS.items():
        mine = [j for j in js if j["editor"] == name]
        ratio = []
        for j in random.Random(0).sample(mine, min(30, len(mine))):
            n = client.messages.count_tokens(model=model, system=GEN.SYSTEM, messages=params(j)["messages"])
            ratio.append(n.input_tokens / (len(j["prompt"]) + len(GEN.SYSTEM)))
        per_char = sum(ratio) / len(ratio)
        tin = sum(len(j["prompt"]) + len(GEN.SYSTEM) for j in mine) * per_char
        tout = sum(len(j["prompt"].split("TEXT:\n", 1)[-1]) for j in mine) * per_char * 1.05  # an edit ~ its text
        cost = (tin * pin + tout * pout) / 1e6
        total += cost
        print(f"{name}: {len(mine)} jobs, {tin / 1e6:.2f}M in, ~{tout / 1e6:.2f}M out, ~${cost:.2f} at batch prices")
    print(f"total ~${total:.2f}")


def submit(args) -> None:
    import anthropic

    client = anthropic.Anthropic()
    js = [j for j in read_jsonl(V3 / "jobs.jsonl") if j["editor"] == args.editor]
    log_path = V3 / "batches.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    sent = {c for b in log for c in b["cids"]}
    todo = [j for j in js if j["cid"] not in sent][: args.limit or None]
    for i in range(0, len(todo), args.chunk):
        part = todo[i : i + args.chunk]
        b = client.messages.batches.create(requests=[{"custom_id": j["cid"], "params": params(j)} for j in part])
        log.append({"id": b.id, "editor": args.editor, "cids": [j["cid"] for j in part]})
        log_path.write_text(json.dumps(log) + "\n")
        print(f"batch {b.id}: {len(part)} {args.editor} requests")


def collect(args) -> None:
    import anthropic

    client = anthropic.Anthropic()
    jobs = {j["cid"]: j for j in read_jsonl(V3 / "jobs.jsonl")}
    docs = {r["id"]: r for r in read_jsonl(V3 / "human.jsonl")}
    log = json.loads((V3 / "batches.json").read_text())
    out = V3 / "ai.jsonl"
    done = {r["cid"] for r in read_jsonl(out)}
    spend_path = RESULTS / "v3_generation.json"
    spend = json.loads(spend_path.read_text()) if spend_path.exists() else {"batches": {}}
    pending = 0
    with out.open("a") as f:
        for b in log:
            if b["id"] in spend["batches"]:
                continue
            info = client.messages.batches.retrieve(b["id"])
            if info.processing_status != "ended":
                pending += 1
                print(b["id"], info.processing_status, info.request_counts)
                continue
            _, pin, pout = EDITORS[b["editor"]]
            tin = tout = 0
            status: Counter = Counter()
            for res in client.messages.batches.results(b["id"]):
                status[res.result.type] += 1
                if res.result.type != "succeeded":
                    continue
                msg = res.result.message
                tin += msg.usage.input_tokens
                tout += msg.usage.output_tokens
                j = jobs[res.custom_id]
                text = GEN.clean(next((x.text for x in msg.content if x.type == "text"), ""))
                if "splice" in j:
                    text = spliced(docs[j["doc_id"]]["text"], j["splice"], text) or ""
                if msg.stop_reason != "end_turn" or not text or res.custom_id in done:
                    status[f"dropped: {msg.stop_reason}"] += 1
                    continue
                keep = (
                    "key",
                    "cid",
                    "doc_id",
                    "source",
                    "split",
                    "version",
                    "instruction",
                    "sentences",
                )
                row = {k: j[k] for k in keep if k in j}
                row.update(
                    model=EDITORS[b["editor"]][0],
                    family="claude",
                    editor=b["editor"],
                    text=text,
                )
                row["cost"] = round(
                    (msg.usage.input_tokens * pin + msg.usage.output_tokens * pout) / 1e6,
                    6,
                )
                f.write(json.dumps(row) + "\n")
            spend["batches"][b["id"]] = {
                "editor": b["editor"],
                "requests": len(b["cids"]),
                "input_tokens": tin,
                "output_tokens": tout,
                "usd": round((tin * pin + tout * pout) / 1e6, 4),
                "results": dict(status),
            }
            print(b["id"], spend["batches"][b["id"]])
    spend["total_usd"] = round(sum(x["usd"] for x in spend["batches"].values()), 4)
    spend_path.write_text(json.dumps(spend, indent=1) + "\n")
    print(f"total ${spend['total_usd']:.2f}; {pending} batches still running; {len(read_jsonl(out))} rows on disk")


def label(args) -> None:
    """Lesson 03's filters on the new rows, then the B1 labels (lighter/heavier split fixed at data v2's median)."""
    import numpy as np

    lab = import_module("03_label")
    median = json.loads((RESULTS / "b1_labels.json").read_text())["median_share_changed_train_edits"]
    human = {r["id"]: r for r in read_jsonl(V3 / "human.jsonl")}
    src = {**{r["id"]: r for r in read_jsonl(DATA / "human.jsonl")}, **human}
    dropped, keep = Counter(), []
    for r in read_jsonl(V3 / "ai.jsonl"):
        s = src[r["doc_id"]]["text"]
        if prose_words(r["text"]) < lab.MIN_WORDS or not lab.english(r["text"]):
            dropped.update([f"{r['version']}: short or not English"])
            continue
        r["frac"] = lab.frac_changed(s, r["text"])
        if r["frac"] < lab.MIN_CHANGE:
            dropped.update([f"{r['version']}: under 2% changed"])
            continue
        keep.append(r)
    cos = (lab.embed([src[r["doc_id"]]["text"] for r in keep]) * lab.embed([r["text"] for r in keep])).sum(1)
    rows = []
    for r, c in zip(keep, cos, strict=True):
        if c < lab.EDIT_MIN_COS:
            dropped.update([f"{r['version']}: cosine below {lab.EDIT_MIN_COS}"])
            continue
        f = round(r["frac"], 4)
        rows.append(
            {
                "text": r["text"],
                "target": round(0.20 + 0.50 * f, 4),
                "frac": f,
                "kind": r["version"],
                "source": r["source"],
                "model": r["model"],
                "family": r["family"],
                "doc_id": r["doc_id"],
                "split": r["split"],
                "instruction": r.get("instruction"),
                "cos": round(float(c), 4),
                "key": r["key"],
                "band": 1,
                "bucket": 1 if f <= median else 2,
                "data": "v3",
            }
        )
    for i, d in human.items():
        rows.append(
            {
                "text": d["text"],
                "target": 0.0,
                "kind": "human",
                "source": d["source"],
                "model": "human",
                "family": "human",
                "doc_id": i,
                "split": d["split"],
                "key": f"{i}|human",
                "band": 0,
                "bucket": 0,
                "data": "v3",
            }
        )
    base = {s: read_jsonl(DATA / "v2b" / f"{s}.jsonl") for s in ("train", "val", "test")}
    train = base["train"] + [r for r in rows if r["split"] == "train"]
    bands = Counter(r["band"] for r in train)
    total = sum(bands.values())
    sets = {
        "train": train,
        "val": base["val"],  # the same rows as B1, so the gate compares like with like
        "val_v3": [r for r in rows if r["split"] == "val"],
        "test": base["test"] + [r for r in rows if r["split"] == "test"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rs in sets.items():
        for r in rs:
            r["weight"] = round(total / (3 * bands[r["band"]]), 4)
        with (OUT / f"{name}.jsonl").open("w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rs)
    claude = sum(r["family"] == "claude" for r in train)
    report = {
        "rows": {k: len(v) for k, v in sets.items()},
        "new_rows_by_split": dict(Counter(r["split"] for r in rows)),
        "new_by_kind": dict(Counter(r["kind"] for r in rows)),
        "new_by_editor": dict(Counter(r["model"] for r in rows)),
        "new_buckets": dict(Counter(r["bucket"] for r in rows)),
        "median_frac_by_kind": {
            k: round(float(np.median([r["frac"] for r in rows if r["kind"] == k])), 3)
            for k in ("light", "partial", "heavy")
            if any(r["kind"] == k for r in rows)
        },
        "train_band_share": {str(b): round(n / total, 3) for b, n in sorted(bands.items())},
        "claude_share_of_train_rows": round(claude / len(train), 4),
        "dropped": dict(dropped),
        "median_split": median,
    }
    (RESULTS / "v3_labels.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["humans", "plan", "estimate", "submit", "collect", "label", "redo", "splice"])
    ap.add_argument("--fiction", type=int, default=4000)
    ap.add_argument("--medium", type=int, default=4000)
    ap.add_argument("--light", type=int, default=10000)
    ap.add_argument("--heavy", type=int, default=5000)
    ap.add_argument(
        "--sonnet",
        type=int,
        default=3000,
        help="edits given to Sonnet 5.5 (texts of 400 words or fewer)",
    )
    ap.add_argument("--editor", choices=list(EDITORS), default="haiku")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()
    steps = {
        "humans": humans,
        "plan": plan,
        "estimate": estimate,
        "submit": submit,
        "collect": collect,
    }
    steps["label"] = label
    steps["redo"] = redo
    steps["splice"] = splice_jobs
    steps[args.step](args)


if __name__ == "__main__":
    main()
