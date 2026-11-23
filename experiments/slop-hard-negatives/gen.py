"""AI pages and judge calls, open-weight models only, through OpenRouter (key 2; project cap $5).

    python gen.py mirrors --src data/x.jsonl --out data/y.jsonl   # arm (b) pairs, interp slop set
    python gen.py fresh                                           # the 2026 evaluation pages
    python gen.py known                                           # the judge's known-answer check
    python gen.py judge --src data/x.jsonl --q machine            # one yes/no question per page

Mirrors follow the slop-detector note's procedure (02_generate.py): a first call writes a two-line summary and
keywords of the human page; a second call writes a new page from the summary alone, so every word is the model's.
The training mix is the note's v1 generator mix (same models and weights). The 2026 set uses four generators that are
in no training data. Every finished item is appended at once, so a rerun skips what is on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path

from hn import DATA, HERE, RESULTS, log_spend, read_jsonl, spent

KEY_FILE = next(
    (p / ".env" for p in HERE.parents if (p / ".env").exists()), HERE / ".env"
)  # nearest .env, never printed
TRAIN_MIX = {  # the note's v1 mix on OpenRouter: weight
    "openai/gpt-oss-120b": 3,
    "openai/gpt-oss-20b": 1,
    "deepseek/deepseek-v3.2": 3,
    "qwen/qwen3-235b-a22b-2507": 3,
    "qwen/qwen3-next-80b-a3b-instruct": 1,
    "google/gemma-4-31b-it": 2,
    "google/gemma-4-26b-a4b-it": 2,
}
FRESH = [
    "deepseek/deepseek-v4.1-flash",
    "minimax/minimax-m3.1-flash",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning",
]
# Amendment 1: gemma-4-31b-it missed the human false-yes bar (0.113 > 0.10; balanced accuracy 0.80); the one allowed
# change, qwen3-235b-a22b-2507, did worse (0.18, 0.64; results/judge_known_qwen.json). The plan then makes the
# machine-made share exploratory; it uses gemma, the better of the two on both numbers.
JUDGE = "google/gemma-4-31b-it"
PRICE = {  # $ per M tokens in / out (OpenRouter, 2026-11-23)
    "openai/gpt-oss-120b": (0.05, 0.25),
    "openai/gpt-oss-20b": (0.03, 0.13),
    "deepseek/deepseek-v3.2": (0.26, 0.38),
    "qwen/qwen3-235b-a22b-2507": (0.14, 0.80),
    "qwen/qwen3-next-80b-a3b-instruct": (0.10, 1.10),
    "google/gemma-4-31b-it": (0.20, 0.40),
    "google/gemma-4-26b-a4b-it": (0.09, 0.30),
    "deepseek/deepseek-v4.1-flash": (0.05, 0.20),
    "minimax/minimax-m3.1-flash": (0.15, 0.60),
    "z-ai/glm-5.3-flash": (0.04, 0.074),
    "nvidia/nemotron-3.5-lightning": (0.07, 0.20),
}
MUST_REASON = ("gpt-oss", "minimax", "glm")  # OpenRouter refuses to switch reasoning off for these
CAP = 5.0
KIND_WEB = "the text of a web page (plain text only, no HTML or markup)"
SYSTEM = "You are a writing assistant. Reply with the requested text only: no preamble, no notes, no quotes around it."
PREAMBLE = re.compile(r"^(sure|certainly|here(?:'s| is| are)|okay|of course)\b[^\n]*:\s*\n+", re.IGNORECASE)

QUESTIONS = {
    "machine": (
        "Below is the start of a web page that was crawled before 2022. Most such pages were written by people, but "
        "some were made by machines: article-spinner output (copied text with synonyms swapped in, so phrases read "
        "oddly), auto-generated SEO filler or keyword-stuffed template text, or output of early language models "
        "(GPT-2 or GPT-3 era) that drifts, repeats or contradicts itself. Ordinary human writing counts as NOT "
        "machine-made, even when it is dull, formulaic, corporate, promotional or badly written, and so do menus, "
        "listings and legal boilerplate.\n\nDoes this page look machine-made? Answer with one word: YES or NO.\n\n"
        "PAGE:\n{text}"
    ),
    "corporate": (
        "Is the web page below written in a polished corporate or marketing register: company or product copy, a "
        "press release, a services page, PR, or a promotional business blog post? Personal, conversational, news, "
        "technical, academic, forum, fiction and reference writing are NOT. Answer with one word: YES or NO.\n\n"
        "PAGE:\n{text}"
    ),
}


def _key() -> str:
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY_2="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENROUTER_API_KEY_2 missing from .env")


class Meter:
    """Counts dollars from token usage; refuses new calls past the project cap (ledger + this session)."""

    def __init__(self, what: str) -> None:
        self.what, self.start, self.usd, self.calls, self.fail = what, spent("api"), 0.0, 0, 0

    def add(self, model: str, usage) -> None:
        pin, pout = PRICE[model]
        if usage:
            self.usd += (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6
        self.calls += 1

    @property
    def over(self) -> bool:
        return self.start + self.usd >= CAP

    def close(self) -> None:
        log_spend(self.what, self.usd, kind="api", calls=self.calls, failed=self.fail, note="measured (token usage)")


def clean(text: str) -> str:
    text = PREAMBLE.sub("", (text or "").strip()).strip()
    if len(text) > 2 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


async def chat(
    client,
    meter: Meter,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.8,
    system: str | None = SYSTEM,
) -> str:
    reason = any(k in model for k in MUST_REASON)
    extra = {"reasoning": {"effort": "low"}} if reason else {"reasoning": {"enabled": False}}
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for attempt in range(4):
        try:
            r = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens + (2000 if reason else 0),
                temperature=temperature,
                extra_body=extra,
                timeout=240,
                messages=msgs,
            )
            meter.add(model, r.usage)
            text = clean(r.choices[0].message.content or "")
            if r.choices[0].finish_reason == "length" or not text:
                raise RuntimeError(f"cut off or empty ({r.choices[0].finish_reason})")
            return text
        except Exception:
            if attempt == 3:
                meter.fail += 1
                raise
            await asyncio.sleep(2**attempt * 3 + random.random() * 3)
    raise RuntimeError("unreachable")


async def mirror(client, meter: Meter, doc: dict, model: str) -> dict:
    words = min(int(doc["words"]), 450)  # the detector reads 512 tokens, about 380 words
    summary = await chat(
        client,
        meter,
        model,
        "Summarise the text below in exactly two lines, then give a line 'Keywords:' with 5 to 8 keywords.\n\n"
        f"TEXT:\n{doc['text'][:3000]}",
        250,
    )
    text = await chat(
        client,
        meter,
        model,
        f"Write {KIND_WEB}, about {words} words long, based only on this summary and keyword list. "
        f"Make it read like a real one.\n\n{summary}",
        min(3000, int(words * 2.2) + 150),
    )
    return {"key": f"{doc['id']}|mirror|{model}", "doc_id": doc["id"], "model": model, "summary": summary, "text": text}


def _client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=_key(), max_retries=0)


async def run_jobs(jobs: list, out: Path, fn, what: str, concurrency: int = 32) -> int:
    """jobs: (key, args...) tuples; fn(client, meter, *args) -> row with "key". Appends finished rows to `out`."""
    done = {r["key"] for r in read_jsonl(out)}
    jobs = [j for j in jobs if j[0] not in done]
    client, meter = _client(), Meter(what)
    sem, made, t0 = asyncio.Semaphore(concurrency), 0, time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("a")

    async def one(job) -> None:
        nonlocal made
        async with sem:
            if meter.over:
                return
            try:
                row = await fn(client, meter, *job[1:])
            except Exception as e:  # noqa: BLE001 - any provider error skips this item; a rerun retries it
                print(f"  skip {job[0]}: {str(e)[:120]}", flush=True)
                return
            f.write(json.dumps(row) + "\n")
            f.flush()
            made += 1
            if made % 200 == 0:
                print(f"  {made}/{len(jobs)} ${meter.usd:.3f} {made / (time.time() - t0):.1f}/s", flush=True)

    await asyncio.gather(*(one(j) for j in jobs))
    f.close()
    meter.close()
    print(
        f"{what}: {made}/{len(jobs)} made, {meter.fail} failed, ${meter.usd:.3f}"
        + (" (stopped at the cap)" if meter.over else ""),
        flush=True,
    )
    return made


def pick(doc_id: str, mix: dict) -> str:
    models = list(mix)
    return random.Random(f"{doc_id}|model").choices(models, [mix[m] for m in models])[0]


def mirrors(src: list[dict], out: Path, what: str) -> None:
    jobs = []
    for d in src:
        m = pick(d["id"], TRAIN_MIX)
        jobs.append((f"{d['id']}|mirror|{m}", d, m))
    asyncio.run(run_jobs(jobs, out, mirror, what))


async def _judge_one(client, meter: Meter, key: str, q: str, text: str) -> dict:
    ans = await chat(client, meter, JUDGE, QUESTIONS[q].format(text=text[:3000]), 5, temperature=0.0, system=None)
    a = ans.strip().upper()
    return {"key": key, "q": q, "answer": "YES" if a.startswith("YES") else "NO" if a.startswith("NO") else a}


def jkey(q: str, text: str) -> str:
    return f"{JUDGE}|{q}|{hashlib.md5(text[:3000].encode()).hexdigest()}"


def judge(texts: list[str], q: str, what: str) -> dict[str, str]:
    """YES/NO for each text (cached in data/judge.jsonl, keyed by question and text)."""
    out = DATA / "judge.jsonl"
    jobs = list({jkey(q, t): (jkey(q, t), jkey(q, t), q, t) for t in texts}.values())
    asyncio.run(run_jobs(jobs, out, _judge_one, what, concurrency=24))
    cache = {r["key"]: r["answer"] for r in read_jsonl(out)}
    out = {t: cache.get(jkey(q, t)) for t in texts}
    missing = sum(v not in ("YES", "NO") for v in out.values())
    if missing:  # an unjudged page must not count as NO; rerun to fill it
        raise SystemExit(f"{what}: {missing} of {len(out)} pages have no YES/NO answer; rerun to fill them")
    return out


# ------------------------------------------------------------------ the 2026 evaluation set
def fresh() -> None:
    """150 old-topic and 150 new-topic pages per 2026 generator, by the mirror procedure. Old-topic seeds: held-out
    domain pages beyond the 20,000 used as humans; new-topic seeds: the 2025 dump."""
    held = read_jsonl(DATA / "pools" / "heldout.jsonl")[20000:20600]
    modern = read_jsonl(DATA / "pools" / "modern.jsonl")[:600]
    jobs = []
    for topic, seeds in (("old", held), ("new", modern)):
        for i, d in enumerate(seeds):
            m = FRESH[i % 4]
            jobs.append((f"{d['id']}|mirror|{m}", {**d, "topic": topic}, m))

    async def fn(client, meter, d, m):
        return {**(await mirror(client, meter, d, m)), "topic": d["topic"]}

    asyncio.run(run_jobs(jobs, DATA / "fresh.jsonl", fn, "fresh 2026 pages"))


# ------------------------------------------------------------------ the judge's known-answer check
SPIN = {
    "important": "significant",
    "help": "assist",
    "use": "utilize",
    "big": "massive",
    "make": "create",
    "good": "decent",
    "best": "finest",
    "great": "terrific",
    "get": "acquire",
    "need": "require",
    "find": "locate",
    "buy": "purchase",
    "show": "demonstrate",
    "start": "commence",
    "end": "conclude",
    "people": "individuals",
    "many": "numerous",
    "very": "extremely",
    "also": "additionally",
    "but": "nonetheless",
    "because": "since",
    "easy": "effortless",
    "hard": "tough",
    "fast": "speedy",
    "small": "tiny",
    "home": "residence",
    "car": "automobile",
    "house": "dwelling",
    "money": "funds",
    "job": "occupation",
    "work": "labor",
    "time": "period",
    "way": "method",
    "place": "location",
    "thing": "item",
    "things": "items",
    "new": "brand-new",
    "old": "aged",
    "look": "appear",
    "think": "believe",
    "want": "desire",
    "give": "provide",
    "keep": "maintain",
    "try": "attempt",
    "say": "state",
    "said": "stated",
    "know": "realize",
    "see": "observe",
    "come": "arrive",
    "go": "proceed",
    "take": "seize",
    "life": "existence",
    "world": "globe",
    "day": "daytime",
    "year": "12 months",
    "years": "many years",
    "company": "firm",
    "business": "enterprise",
    "customers": "clients",
    "products": "merchandise",
    "information": "data",
    "problem": "difficulty",
    "question": "query",
    "answer": "response",
    "idea": "notion",
    "simple": "uncomplicated",
    "large": "sizable",
    "high": "elevated",
    "low": "reduced",
    "however": "nevertheless",
    "often": "regularly",
    "usually": "normally",
    "really": "actually",
    "quickly": "rapidly",
    "family": "household",
    "children": "youngsters",
    "food": "meals",
    "water": "drinking water",
    "website": "internet site",
    "online": "on-line",
    "free": "complimentary",
    "price": "cost",
    "quality": "high quality",
    "service": "support",
}


def spin(text: str, rng: random.Random) -> str:
    """A crude article spinner: swap listed words for 'synonyms', as 2010s SEO spinners did."""

    def sub(m: re.Match) -> str:
        w = m.group(0)
        s = SPIN.get(w.lower())
        if not s or rng.random() > 0.9:
            return w
        return s.capitalize() if w[0].isupper() else s

    return re.sub(r"[A-Za-z]+", sub, text)


def known() -> None:
    """100 GPT-2 XL (top-k 40) samples and 50 spun dev pages (machine-made) against 75 WebText and 75 dev pages
    (human). Pass bar (plan): balanced accuracy >= 0.70 and human false-yes <= 0.10."""
    import urllib.request

    base = "https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/"
    kd = DATA / "known"
    kd.mkdir(parents=True, exist_ok=True)
    for f in ("webtext.test.jsonl", "xl-1542M-k40.test.jsonl"):
        if not (kd / f).exists():
            urllib.request.urlretrieve(base + f, kd / f)
    rng = random.Random(0)

    def long(rows):
        return [r["text"] for r in rows if len(r["text"].split()) >= 150]

    gpt2 = rng.sample(long(read_jsonl(kd / "xl-1542M-k40.test.jsonl")), 100)
    web = rng.sample(long(read_jsonl(kd / "webtext.test.jsonl")), 75)
    dev = read_jsonl(DATA / "pools" / "dev.jsonl")
    dev_h = [r["text"] for r in dev[:75]]
    spun = [spin(r["text"], rng) for r in dev[75:125]]
    items = (
        [(t, 1, "gpt2-xl") for t in gpt2]
        + [(t, 1, "spun") for t in spun]
        + [(t, 0, "webtext") for t in web]
        + [(t, 0, "dev-web") for t in dev_h]
    )
    ans = judge([t for t, _, _ in items], "machine", "judge known-answer")
    rows = [{"kind": k, "y": y, "yes": ans[t] == "YES", "answer": ans[t]} for t, y, k in items]
    by = {}
    for k in ("gpt2-xl", "spun", "webtext", "dev-web"):
        rs = [r for r in rows if r["kind"] == k]
        by[k] = {"n": len(rs), "yes": sum(r["yes"] for r in rs)}
    tpr = sum(r["yes"] for r in rows if r["y"] == 1) / sum(r["y"] == 1 for r in rows)
    fpr = sum(r["yes"] for r in rows if r["y"] == 0) / sum(r["y"] == 0 for r in rows)
    rep = {
        "judge": JUDGE,
        "by_kind": by,
        "tpr": tpr,
        "human_false_yes": fpr,
        "balanced_acc": (tpr + 1 - fpr) / 2,
        "unparsed": sum(r["answer"] not in ("YES", "NO") for r in rows),
    }
    rep["passes"] = rep["balanced_acc"] >= 0.70 and fpr <= 0.10
    (RESULTS / "judge_known.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mirrors", "fresh", "known", "judge"])
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--q", choices=list(QUESTIONS), default="machine")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "mirrors":
        src = read_jsonl(Path(a.src))
        mirrors(src[: a.limit] if a.limit else src, Path(a.out), f"mirrors {Path(a.out).name}")
    elif a.cmd == "fresh":
        fresh()
    elif a.cmd == "known":
        known()
    else:
        src = read_jsonl(Path(a.src))
        ans = judge([r["text"] for r in (src[: a.limit] if a.limit else src)], a.q, f"judge {a.q} {Path(a.src).name}")
        print(sum(v == "YES" for v in ans.values()), "YES of", len(ans))


if __name__ == "__main__":
    main()
