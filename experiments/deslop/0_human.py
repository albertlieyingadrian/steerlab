"""Build the human-prose baseline: passages of about 250 words with clear licences.

    python experiments/deslop/0_human.py        # writes data/human/passages.jsonl (and caches the raw downloads)

Two sources, kept apart in every table because they differ in era and genre:
- Project Gutenberg essays, memoirs and explainers published before 1929 (public domain in the US): 13 books, 20
  passages each, drawn at random (seed 0) from paragraphs of 150-400 words.
- English Wikipedia articles on everyday topics as they stood on 2021-06-01, before chat assistants were public
  (CC BY-SA 3.0; text only, no edits): 30 articles, up to two passages each.

Each row: id, source ("gutenberg" or "wikipedia"), ref (book id or revision id), title, author, year, licence, text.
"""

from __future__ import annotations

import html
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "human" / "raw"
OUT = HERE / "data" / "human" / "passages.jsonl"
UA = {"User-Agent": "steerlab-deslop/0.1 (research; https://github.com/albertlieyingadrian/steerlab)"}

BOOKS = [  # (Gutenberg id, title, author, first published)
    (8092, "Tremendous Trifles", "G. K. Chesterton", 1909),
    (11505, "All Things Considered", "G. K. Chesterton", 1908),
    (849, "The Idle Thoughts of an Idle Fellow", "Jerome K. Jerome", 1886),
    (386, "Virginibus Puerisque", "Robert Louis Stevenson", 1881),
    (64457, "The Common Reader", "Virginia Woolf", 1925),
    (2944, "Essays, First Series", "Ralph Waldo Emerson", 1841),
    (6340, "Literary Lapses", "Stephen Leacock", 1910),
    (70, "What Is Man? and Other Essays", "Mark Twain", 1917),
    (1325, "Twenty Years at Hull-House", "Jane Addams", 1910),
    (2376, "Up from Slavery", "Booker T. Washington", 1901),
    (5827, "The Problems of Philosophy", "Bertrand Russell", 1912),
    (1022, "Walking", "Henry David Thoreau", 1862),
    (205, "Walden", "Henry David Thoreau", 1854),
]
ARTICLES = [
    "Coffee", "Bicycle", "Public library", "Gardening", "Sleep", "Restaurant", "Homework", "Telecommuting",
    "Electric car", "Second-language acquisition", "Office", "Bread", "Marathon", "Chess", "Beekeeping",
    "Birdwatching", "Recycling", "Photography", "Camping", "Budget", "Board game", "Houseplant", "Volunteering",
    "Newspaper", "Swimming", "Crossword", "Carpool", "Solar panel", "Tiny house movement", "Journaling",
]  # fmt: skip
WIKI_DATE = "2021-06-01T00:00:00Z"
PER_BOOK, PER_ARTICLE = 20, 2


def fetch(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return text
        except OSError as err:
            print(f"  retry {attempt + 1}: {err}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", text))


def gutenberg_paragraphs(raw: str) -> list[str]:
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG", raw)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)].replace("\r\n", "\n")
    paras = []
    for block in re.split(r"\n\s*\n", body):
        text = re.sub(r"\s+", " ", block).strip()
        text = re.sub(r"\[(?:Footnote|Illustration|\d+)[^\]]*\]", "", text).replace("_", "").strip()
        # Prose paragraphs only: no headings, tables of contents, verse or all-caps lines.
        if not 150 <= words(text) <= 400 or text.isupper() or text[-1:] not in ".!?\"'”’)":
            continue
        paras.append(text)
    return paras


class _Paragraphs(HTMLParser):
    """Text of the <p> elements in a parsed Wikipedia page (reference marks, styles and tables dropped)."""

    def __init__(self) -> None:
        super().__init__()
        self.paras: list[str] = []
        self._buf: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p" and self._skip == 0:
            self._buf = []
        elif tag in ("sup", "style", "table"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("sup", "style", "table") and self._skip:
            self._skip -= 1
        elif tag == "p" and self._buf is not None:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.paras.append(text)
            self._buf = None

    def handle_data(self, data: str) -> None:
        if self._buf is not None and self._skip == 0:
            self._buf.append(data)


def wikipedia_paragraphs(title: str) -> tuple[int, list[str]]:
    api = "https://en.wikipedia.org/w/api.php?"
    query = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvstart": WIKI_DATE,
    }
    query |= {"rvdir": "older", "rvprop": "ids", "format": "json", "redirects": 1}
    meta = json.loads(fetch(api + urllib.parse.urlencode(query), RAW / "wiki" / f"{title}.rev.json"))
    page = next(iter(meta["query"]["pages"].values()))
    revid = page["revisions"][0]["revid"]
    query = {"action": "parse", "oldid": revid, "prop": "text", "format": "json"}
    parsed = json.loads(fetch(api + urllib.parse.urlencode(query), RAW / "wiki" / f"{title}.{revid}.json"))
    parser = _Paragraphs()
    parser.feed(html.unescape(parsed["parse"]["text"]["*"]))
    return revid, parser.paras


def chunks(paras: list[str], lo: int = 150, hi: int = 350) -> list[str]:
    """Join neighbouring paragraphs until a chunk has at least `lo` words; drop chunks over `hi`."""
    out, buf = [], []
    for para in paras:
        buf.append(para)
        if words(" ".join(buf)) >= lo:
            text = "\n\n".join(buf)
            if words(text) <= hi:
                out.append(text)
            buf = []
    return out


def main() -> None:
    rng = random.Random(0)
    rows = []
    for gid, title, author, year in BOOKS:
        paras = gutenberg_paragraphs(
            fetch(
                f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                RAW / f"pg{gid}.txt",
            )
        )
        picked = sorted(rng.sample(range(len(paras)), min(PER_BOOK, len(paras))))
        for k, i in enumerate(picked):
            row = {
                "id": f"g{gid}-{k:02d}",
                "source": "gutenberg",
                "ref": f"gutenberg:{gid}",
                "title": title,
            }
            rows.append(
                row
                | {
                    "author": author,
                    "year": year,
                    "licence": "public domain (US)",
                    "text": paras[i],
                }
            )
        print(f"gutenberg {gid:>6} {title[:40]:40} {len(paras):4} paragraphs, kept {len(picked)}")
    for title in ARTICLES:
        revid, paras = wikipedia_paragraphs(title)
        picked = chunks([p for p in paras if words(p) >= 25])[:PER_ARTICLE]
        slug = re.sub(r"[^a-z]+", "-", title.lower())
        for k, text in enumerate(picked):
            row = {
                "id": f"w-{slug}-{k}",
                "source": "wikipedia",
                "ref": f"en.wikipedia.org oldid={revid}",
            }
            rows.append(row | {"title": title, "author": "Wikipedia contributors", "year": 2021,
                               "licence": "CC BY-SA 3.0", "text": text})  # fmt: skip
        print(f"wikipedia {title[:40]:40} rev {revid}, kept {len(picked)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by = {s: sum(r["source"] == s for r in rows) for s in ("gutenberg", "wikipedia")}
    print(f"wrote {len(rows)} passages to {OUT.relative_to(HERE)}: {by}")


if __name__ == "__main__":
    main()
