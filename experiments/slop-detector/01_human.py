"""Lesson 01: collect human documents written before 2023, the seven sources on the voi-slop-detect-4B card.

    python 01_human.py [--per-source 1600]

Why before 2023: ChatGPT came out in November 2022, so text written earlier is very unlikely to contain LLM output.
That makes the human label trustworthy without anyone checking it by hand.

Sources (all public on the Hugging Face Hub; one file or shard each, so nothing huge is downloaded):

| source  | dataset                               | years kept | text field |
|---------|---------------------------------------|-----------:|------------|
| amazon  | McAuley-Lab/Amazon-Reviews-2023 (4 small categories) | 2020-2021 | review text |
| arxiv   | gfissore/arxiv-abstracts-2021         | 2020-2021 (from the id) | abstract |
| hn      | OpenPipe/hacker-news, shard 30        | 2022       | comment text (HTML stripped) |
| medium  | fabiochiu/medium-articles             | 2020-2022  | article text |
| reddit  | HuggingFaceGECLM/REDDIT_comments, 10 subreddits, last shard | 2020-2022 | comment body |
| stack   | mikex86/stackoverflow-posts, shard 55 | 2020-2022  | question or answer body (markdown) |
| xsum    | EdinburghNLP/xsum                     | 2017 (BBC) | article |

r/WritingPrompts is left out on purpose: it is one of the EditLens test sources, and we never train on a test source's
documents. Long documents are cut to their first ~300 words at a paragraph or sentence end (the median on the card is
116 words). Writes data/human.jsonl: {"id", "source", "year", "split", "text", "words"}. The split (80/10/10) is a hash
of the id, so every AI version of a document later lands in the same split as its human original.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import html
import json
import random
import re
from collections.abc import Iterator

from common import DATA, prose_words

MAX_WORDS, MIN_WORDS = 300, 30
TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+")


def split_of(doc_id: str) -> str:
    h = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 8 else "val" if h == 8 else "test"


def trim(text: str, max_words: int = MAX_WORDS) -> str:
    """First ~max_words words, ending at a paragraph break; a too-long first paragraph is cut at a sentence end."""
    MAX_WORDS = max_words
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r", "")).strip()
    if len(text.split()) <= MAX_WORDS:
        return text
    out, n = [], 0
    for para in text.split("\n\n"):
        k = len(para.split())
        if n + k <= MAX_WORDS:
            out.append(para)
            n += k
            continue
        if n < 2 * MIN_WORDS:
            part = []
            for s in re.split(r"(?<=[.!?])\s+", para):
                if n + len(s.split()) > MAX_WORDS:
                    break
                part.append(s)
                n += len(s.split())
            out.append(" ".join(part))
        break
    return "\n\n".join(p for p in out if p).strip()


def hub(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def parquet_rows(repo: str, path: str, columns: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(hub(repo, path))
    for batch in f.iter_batches(columns=columns, batch_size=20_000):
        yield from batch.to_pylist()


def amazon() -> Iterator[dict]:
    for cat in ("All_Beauty", "Handmade_Products", "Health_and_Personal_Care", "Digital_Music"):
        with open(hub("McAuley-Lab/Amazon-Reviews-2023", f"raw/review_categories/{cat}.jsonl")) as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                year = dt.datetime.utcfromtimestamp(r["timestamp"] / 1000).year
                if year in (2020, 2021):
                    text = html.unescape(r["text"]).replace("<br />", "\n").replace("<br/>", "\n")
                    yield {"id": f"amazon/{cat}/{i}", "year": year, "text": TAG.sub("", text)}


def arxiv() -> Iterator[dict]:
    with gzip.open(hub("gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            m = re.match(r"^(\d\d)(\d\d)\.\d+", r["id"])
            if m and m.group(1) in ("20", "21"):
                text = re.sub(r"\s*\n\s*", " ", r["abstract"]).strip()
                yield {"id": f"arxiv/{r['id']}", "year": 2000 + int(m.group(1)), "text": text}


def hn() -> Iterator[dict]:
    cols = ["id", "type", "time", "text", "deleted", "dead"]
    for r in parquet_rows("OpenPipe/hacker-news", "data/train-00030-of-00039.parquet", cols):
        if r["type"] != "comment" or not r["text"] or r["deleted"] or r["dead"] or r["time"].year > 2022:
            continue
        text = html.unescape(TAG.sub("", r["text"].replace("<p>", "\n\n")))
        yield {"id": f"hn/{r['id']}", "year": r["time"].year, "text": text}


def medium() -> Iterator[dict]:
    import pandas as pd

    df = pd.read_csv(hub("fabiochiu/medium-articles", "medium_articles.csv"), usecols=["url", "text", "timestamp"])
    df = df[df["timestamp"].str[:4].isin(["2020", "2021", "2022"])]
    for url, text, ts in zip(df["url"], df["text"], df["timestamp"], strict=True):
        if not isinstance(text, str):
            continue
        lines = [ln for ln in text.split("\n") if not re.match(r"^(Photo|Image) (by|from|credit)", ln.strip())]
        slug = url.rstrip("/").rsplit("-", 1)[-1]
        yield {"id": f"medium/{slug}", "year": int(ts[:4]), "text": "\n".join(lines)}


REDDIT_SUBS = (
    "askscience",
    "explainlikeimfive",
    "personalfinance",
    "relationship_advice",
    "changemyview",
    "travel",
    "books",
    "buildapc",
    "DIY",
    "tifu",
)


def reddit() -> Iterator[dict]:
    from huggingface_hub import HfApi

    files = [s.rfilename for s in HfApi().dataset_info("HuggingFaceGECLM/REDDIT_comments").siblings]
    per_sub = 3000  # so no single subreddit fills the pool
    for sub in REDDIT_SUBS:
        last, n = sorted(f for f in files if f.startswith(f"data/{sub}-"))[-1], 0
        for r in parquet_rows("HuggingFaceGECLM/REDDIT_comments", last, ["id", "body", "created_utc"]):
            year = dt.datetime.utcfromtimestamp(int(r["created_utc"])).year
            body = r["body"] or ""
            if (
                2020 <= year <= 2022
                and body not in ("[deleted]", "[removed]")
                and "I am a bot" not in body
                and len(body.split()) >= 30
            ):
                n += 1
                yield {"id": f"reddit/{sub}/{r['id']}", "year": year, "text": html.unescape(body)}
            if n >= per_sub:
                break


def stack() -> Iterator[dict]:
    cols = ["Id", "PostTypeId", "Body", "Title", "CreationDate"]
    for r in parquet_rows("mikex86/stackoverflow-posts", "stackoverflow-posts-00055-of-00058.parquet", cols):
        year = int(str(r["CreationDate"])[:4])
        if 2020 <= year <= 2022 and r["Body"]:
            body = f"{r['Title']}\n\n{r['Body']}" if r["Title"] else r["Body"]
            yield {"id": f"stack/{r['Id']}", "year": year, "text": body}


def xsum() -> Iterator[dict]:
    for r in parquet_rows("EdinburghNLP/xsum", "data/train-00000-of-00001.parquet", ["id", "document"]):
        yield {"id": f"xsum/{r['id']}", "year": 2017, "text": r["document"]}


def pg19() -> Iterator[dict]:
    """Long-form fiction and essays (books published before 1919, Project Gutenberg via PG-19): up to 10 passages
    per book from its middle, so the detector sees long human prose, not only comments and reviews."""
    from huggingface_hub import HfApi

    files = [s.rfilename for s in HfApi().dataset_info("emozilla/pg19").siblings if "train" not in s.rfilename]
    for f in sorted(x for x in files if x.endswith(".parquet")):
        for r in parquet_rows("emozilla/pg19", f, ["short_book_title", "publication_date", "text"]):
            paras = [" ".join(p.split()) for p in r["text"].split("\n\n")]
            paras = [p for p in paras if len(p.split()) >= 20 and not p.isupper()]
            body = paras[len(paras) // 7 : -len(paras) // 10 or None]
            rng, slug = random.Random(r["short_book_title"]), re.sub(r"\W+", "-", r["short_book_title"].lower())[:40]
            for k in range(10):
                if len(body) < 10:
                    break
                i, out = rng.randrange(len(body) - 8), []
                while i < len(body) and sum(len(p.split()) for p in out) < 400:
                    out.append(body[i])
                    i += 1
                yield {"id": f"pg19/{slug}/{k}", "year": int(r["publication_date"] or 1900), "text": "\n\n".join(out)}


SOURCES = {
    "amazon": amazon,
    "arxiv": arxiv,
    "hn": hn,
    "medium": medium,
    "reddit": reddit,
    "stack": stack,
    "xsum": xsum,
    "pg19": pg19,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=1600)
    ap.add_argument("--pool", type=int, default=30_000, help="candidates read per source before sampling")
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    ap.add_argument("--offset", type=int, default=0, help="skip this many sampled documents (for a second batch)")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    ap.add_argument("--suffix", default="", help="label suffix, e.g. _long, so batches stay distinguishable")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "human.jsonl"
    old = [json.loads(ln) for ln in out.read_text().splitlines()] if out.exists() else []
    done, have = {r["source"] for r in old}, {r["id"] for r in old}
    for name in args.only:
        label = name + args.suffix
        if label in done:
            print(f"{label}: already collected")
            continue
        pool, seen = [], set()
        for r in SOURCES[name]():
            text = trim(URL.sub("", r["text"]).strip(), args.max_words)
            key = " ".join(text.lower().split()[:20])
            if prose_words(text) < MIN_WORDS or key in seen:
                continue
            seen.add(key)
            pool.append({**r, "source": label, "text": text})
            if len(pool) >= args.pool:
                break
        random.Random(0).shuffle(pool)
        keep = [r for r in pool[args.offset :] if r["id"] not in have][: args.per_source]
        with out.open("a") as f:
            for r in keep:
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "source": label,
                            "year": r["year"],
                            "split": split_of(r["id"]),
                            "text": r["text"],
                            "words": prose_words(r["text"]),
                        }
                    )
                    + "\n"
                )
        years = sorted({r["year"] for r in keep})
        med = sorted(prose_words(r["text"]) for r in keep)[len(keep) // 2]
        print(
            f"{label}: pool {len(pool)}, kept {len(keep)}, years {years[0]}-{years[-1]}, median words {med}", flush=True
        )


if __name__ == "__main__":
    main()
