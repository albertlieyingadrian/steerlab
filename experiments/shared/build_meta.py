"""Write the search and sharing metadata for the research pages, from `experiments/notes.json`.

    python experiments/shared/build_meta.py           # write everything
    python experiments/shared/build_meta.py --check   # exit 1 if any page or file is out of date

For the research page and every note it writes, between `<!-- meta:start -->` and `<!-- meta:end -->` in `<head>`:
a canonical URL, Open Graph and Twitter tags (the link preview on LinkedIn, X or Slack), and JSON-LD structured data
(`ScholarlyArticle` per note; `CollectionPage` and `SoftwareSourceCode` for the research page). It also writes the
dates in each note's eyebrow, `experiments/sitemap.xml`, `experiments/llms.txt`, and each note's 1200x630 preview
image `report/figures/og.jpg`, cropped from its cover. On the research page it also writes the Latest view between
`<!-- latest:start -->` and `<!-- latest:end -->`: every note with a card in the series view, newest first by
`published` then `published_time` (ISO, used only to order notes from the same day), with its series, episode and
cover from that card. A note with `"pin_day": true` (at most one per day) leads its day as a spotlight: a wide
cover and large title, with the day's other notes as compact rows. A series card's cover with
`data-anim="shared/thumbs/<name>"` (a muted loop in `<name>.webm` and `<name>.mp4`, its poster the card's image) keeps
that attribute in the Latest view. Edit `notes.json`, then rerun; never edit the blocks by hand.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
START, END = "<!-- meta:start -->", "<!-- meta:end -->"
OG_SIZE = (1200, 630)


def attr(text: str) -> str:
    return html.escape(text, quote=True)


def pretty(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d:%b} {d.day}, {d.year}"


def json_ld(data: dict[str, object]) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")
    return f'<script type="application/ld+json">\n{text}\n</script>'


def tags(*, url: str, title: str, description: str, image: str, image_alt: str, kind: str) -> list[str]:
    return [
        f'<link rel="canonical" href="{attr(url)}">',
        '<meta property="og:site_name" content="steerlab">',
        f'<meta property="og:type" content="{kind}">',
        f'<meta property="og:title" content="{attr(title)}">',
        f'<meta property="og:description" content="{attr(description)}">',
        f'<meta property="og:url" content="{attr(url)}">',
        f'<meta property="og:image" content="{attr(image)}">',
        f'<meta property="og:image:width" content="{OG_SIZE[0]}">',
        f'<meta property="og:image:height" content="{OG_SIZE[1]}">',
        f'<meta property="og:image:alt" content="{attr(image_alt)}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{attr(title)}">',
        f'<meta name="twitter:description" content="{attr(description)}">',
        f'<meta name="twitter:image" content="{attr(image)}">',
    ]


def person(meta: dict) -> dict[str, object]:
    author = meta["author"]
    found = {"@type": "Person", "name": author["name"], "url": author["url"]}
    if author.get("same_as"):
        found["sameAs"] = author["same_as"]  # the author's other profiles (GitHub, LinkedIn, X, bio site)
    return found


def note_block(meta: dict, note: dict) -> str:
    research = meta["site"] + "experiments/"
    url = f"{research}{note['name']}/report/"
    image = url + "figures/og.jpg"
    lines = tags(
        url=url,
        title=note["title"],
        description=note["description"],
        image=image,
        image_alt=note["image_alt"],
        kind="article",
    )
    lines += [
        f'<meta property="article:published_time" content="{note["published"]}">',
        f'<meta property="article:modified_time" content="{note["modified"]}">',
        f'<meta property="article:author" content="{attr(meta["author"]["name"])}">',
    ]
    lines.append(
        json_ld(
            {
                "@context": "https://schema.org",
                "@type": "ScholarlyArticle",
                "headline": note["title"],
                "description": note["description"],
                "abstract": note["summary"],
                "url": url,
                "mainEntityOfPage": url,
                "image": image,
                "author": person(meta),
                "publisher": person(meta),
                "datePublished": note["published"],
                "dateModified": note["modified"],
                "inLanguage": "en",
                "isAccessibleForFree": True,
                "keywords": ", ".join(note["keywords"]),
                "isPartOf": {"@type": "CollectionPage", "name": meta["research"]["title"], "url": research},
                "citation": [{"@type": "CreativeWork", "url": c} for c in note["citations"]],
                "isBasedOn": {"@type": "SoftwareSourceCode", "name": "steerlab", "codeRepository": meta["repo"]},
            }
        )
    )
    return "\n".join(lines)


def research_block(meta: dict) -> str:
    research = meta["site"] + "experiments/"
    lines = tags(
        url=research,
        title=meta["research"]["title"],
        description=meta["research"]["description"],
        image=research + "shared/og-research.jpg",
        image_alt="The steerlab research page: one row per note, each with its title, finding and cover.",
        kind="website",
    )
    items = [
        {"@type": "ListItem", "position": i, "url": f"{research}{note['name']}/report/", "name": note["title"]}
        for i, note in enumerate(meta["notes"], 1)
    ]
    lines.append(
        json_ld(
            {
                "@context": "https://schema.org",
                "@graph": [
                    {
                        "@type": "CollectionPage",
                        "name": meta["research"]["title"],
                        "description": meta["research"]["description"],
                        "url": research,
                        "dateModified": meta["research"]["modified"],
                        "author": person(meta),
                        "mainEntity": {"@type": "ItemList", "itemListElement": items},
                    },
                    {
                        "@type": "SoftwareSourceCode",
                        "name": "steerlab",
                        "description": "Steer an open language model while it writes, and measure it against a "
                        "random direction of the same size.",
                        "codeRepository": meta["repo"],
                        "programmingLanguage": "Python",
                        "license": "https://opensource.org/licenses/MIT",
                        "author": person(meta),
                    },
                ],
            }
        )
    )
    return "\n".join(lines)


def with_block(page: str, block: str) -> str:
    """Put the block between the markers, or after the meta description the first time."""
    wrapped = f"{START}\n{block}\n{END}"
    if START in page:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: wrapped, page, count=1, flags=re.S)
    anchor = re.search(r'<meta name="description"[^>]*>\n', page)
    if anchor is None:
        raise SystemExit('no <meta name="description"> to anchor the metadata after')
    return page[: anchor.end()] + wrapped + "\n" + page[anchor.end() :]


def with_dates(page: str, note: dict) -> str:
    """The eyebrow ends with the publish date, and the update date when it differs."""
    sep = '<span class="sep">·</span>'
    when = f'<time datetime="{note["published"]}">{pretty(note["published"])}</time>'
    if note["modified"] != note["published"]:
        when += f'{sep}Updated <time datetime="{note["modified"]}">{pretty(note["modified"])}</time>'
    pattern = r'(<p class="eyebrow">.*?(?:Experiments|Guide|Under the hood)<span class="sep">·</span>).*?(</p>)'
    new, count = re.subn(pattern, lambda m: m.group(1) + when + m.group(2), page, count=1)
    if count != 1:
        raise SystemExit("no eyebrow with 'Experiments', 'Guide' or 'Under the hood' to put the dates in")
    return new


LATEST_START, LATEST_END = "<!-- latest:start -->", "<!-- latest:end -->"
PIN_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 '
    "1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 "
    "0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 "
    '1z"/></svg>'
)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TOKEN = re.compile(
    r'<li class="note series"|<h2>(?P<series>[^<]+)</h2>|<h3 class="season">(?P<season>[^<]+)</h3>'
    r'|<li data-tags="(?P<tags>[^"]*)"><a class="ep-card" href="(?P<slug>[a-z0-9-]+)/report/">'
    r'<img class="thumb" src="(?P<thumb>[^"]+)"(?P<img>[^>]*)><span class="ep-n">(?P<ep>[^<]+?)\s*<span'
)
ANIM = re.compile(r'\sdata-anim="([^"]+)"')


def cards(page: str) -> dict[str, dict[str, str]]:
    """Series, episode, cover and topic tags of every published episode card in the series view, by note name."""
    region = page[page.index('<ol class="notes">') : page.index('<section class="coming-soon"')]
    out: dict[str, dict[str, str]] = {}
    series = season = ""
    for m in TOKEN.finditer(region):
        if m.group(0).startswith('<li class="note series"'):
            season = ""
        elif m.group("series"):
            series = html.unescape(m.group("series"))
        elif m.group("season"):
            season = "S" + m.group("season").split()[1] + " · "
        elif m.group("slug"):
            anim = ANIM.search(m.group("img"))
            out[m.group("slug")] = {"series": series, "ep": season + m.group("ep").strip(), "thumb": m.group("thumb"),
                                    "tags": m.group("tags"), "anim": anim.group(1) if anim else ""}  # fmt: skip
    return out


def finding(text: str, limit: int = 190) -> str:
    """Sentences from the note's description until the line carries a result (90+ characters), cut to limit."""
    out = ""
    for sentence in re.split(r"(?<=[.])\s", text):
        out = f"{out} {sentence}".strip()
        if len(out) >= 90:
            break
    return out if len(out) <= limit else out[: limit - 1].rsplit(" ", 1)[0] + "…"


def latest_block(page: str, meta: dict) -> str:
    """The Latest view: every published episode, newest first by date then publish time, grouped by day."""
    info = cards(page)
    order = {n["name"]: i for i, n in enumerate(meta["notes"])}

    def newest_first(n: dict) -> tuple[str, str, int]:
        return n["published"], n.get("published_time", ""), -order[n["name"]]

    notes = sorted((n for n in meta["notes"] if n["name"] in info), key=newest_first, reverse=True)
    days: dict[str, list[dict]] = {}
    for n in notes:
        days.setdefault(n["published"], []).append(n)
    parts = ['<div class="ledger">']
    for day, group in days.items():
        # a note with "pin_day": true leads its day as a spotlight; the day's other notes become compact rows
        pinned = [n for n in group if n.get("pin_day")]
        if len(pinned) > 1:
            raise SystemExit(f"{day}: more than one note has pin_day ({', '.join(n['name'] for n in pinned)})")
        group = pinned + [n for n in group if not n.get("pin_day")]
        d = date.fromisoformat(day)
        count = f"{len(group)} note{'s' if len(group) > 1 else ''}"
        cls = "lg-day has-pin" if pinned else "lg-day"
        parts.append(f'<section class="{cls}"><h3 class="lg-date"><time datetime="{day}">{MONTHS[d.month - 1]} '
                     f'{d.day}</time><span class="lg-count">{count}</span></h3><ol class="lg-list">')  # fmt: skip
        for n in group:
            c = info[n["name"]]
            li_cls = ' class="is-pinned"' if n.get("pin_day") else ""
            pin = f'<span class="lg-pin">{PIN_SVG}Pinned</span> <span class="lg-sep" aria-hidden="true">·</span> '
            anim = f' data-anim="{attr(c["anim"])}"' if c["anim"] else ""
            parts.append(
                f'<li{li_cls} data-tags="{attr(c["tags"])}"><a class="lg-row" href="{n["name"]}/report/">'
                f'<img class="lg-thumb" src="{attr(c["thumb"])}"{anim} alt="" width="720" height="480" loading="lazy">'
                f'<span class="lg-text"><span class="lg-t">{html.escape(n["title"])}</span>'
                f'<span class="lg-find">{html.escape(finding(n["description"]))}</span>'
                f'<span class="lg-meta">{pin if n.get("pin_day") else ""}'
                f'<span class="lg-series">{html.escape(c["series"])}</span> '
                f'<span class="lg-sep" aria-hidden="true">·</span> <span>{html.escape(c["ep"])}</span></span>'
                f"</span></a></li>"
            )
        parts.append("</ol></section>")
    parts.append("</div>")
    return "\n".join(parts)


def with_latest(page: str, meta: dict) -> str:
    """Put the Latest view between its markers on the research page."""
    if LATEST_START not in page:
        return page
    block = f"{LATEST_START}\n{latest_block(page, meta)}\n{LATEST_END}"
    pattern = re.escape(LATEST_START) + r".*?" + re.escape(LATEST_END)
    return re.sub(pattern, lambda _: block, page, count=1, flags=re.S)


def sitemap(meta: dict) -> str:
    research = meta["site"] + "experiments/"
    rows = [(research, meta["research"]["modified"])]
    rows += [(f"{research}{n['name']}/report/", n["modified"]) for n in meta["notes"]]
    body = "".join(f"  <url><loc>{u}</loc><lastmod>{d}</lastmod></url>\n" for u, d in rows)
    head = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    return head + body + "</urlset>\n"


def llms(meta: dict) -> str:
    research = meta["site"] + "experiments/"
    lines = [
        "# steerlab research",
        "",
        "> steerlab steers an open language model while it writes, by adding a direction to its activations at one",
        "> layer, and measures every effect against a random direction of the same size. These are its research notes.",
        "",
        "Each note gives its method, the steps the study ran, results with 95% intervals, limits, and a script that",
        f"reruns every number. Code: {meta['repo']} (MIT licence).",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- [{n['title']}]({research}{n['name']}/report/): {n['summary']}" for n in meta["notes"]]
    lines += ["", "## Data", ""]
    lines += [f"- {meta['repo']}/tree/main/experiments/{n['name']}/results: {n['title']}" for n in meta["notes"]]
    return "\n".join(lines) + "\n"


def og_image(note: dict) -> Path:
    return EXPERIMENTS / note["name"] / "report" / "figures" / "og.jpg"


def write_og(note: dict) -> None:
    from PIL import Image

    hero = Image.open(EXPERIMENTS / note["name"] / "report" / "figures" / "hero.jpg").convert("RGB")
    if note.get("og_fit") == "pad":
        # Covers whose words must survive: show the whole image, padded at the sides with its own paper colour.
        resized = hero.resize((round(hero.width * OG_SIZE[1] / hero.height), OG_SIZE[1]), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", OG_SIZE, hero.getpixel((4, 4)))
        canvas.paste(resized, ((OG_SIZE[0] - resized.width) // 2, 0))
        canvas.save(og_image(note), quality=84, optimize=True)
        return
    resized = hero.resize((OG_SIZE[0], round(hero.height * OG_SIZE[0] / hero.width)), Image.Resampling.LANCZOS)
    top = (resized.height - OG_SIZE[1]) // 2
    resized.crop((0, top, OG_SIZE[0], top + OG_SIZE[1])).save(og_image(note), quality=84, optimize=True)


def outputs(meta: dict) -> dict[Path, str]:
    """Every text file this script owns, with the content it should have."""
    files: dict[Path, str] = {}
    index = EXPERIMENTS / "index.html"
    files[index] = with_latest(with_block(index.read_text(), research_block(meta)), meta)
    for note in meta["notes"]:
        page = EXPERIMENTS / note["name"] / "report" / "index.html"
        files[page] = with_dates(with_block(page.read_text(), note_block(meta, note)), note)
    files[EXPERIMENTS / "sitemap.xml"] = sitemap(meta)
    files[EXPERIMENTS / "llms.txt"] = llms(meta)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report out-of-date files, write nothing")
    args = parser.parse_args()
    meta = json.loads((EXPERIMENTS / "notes.json").read_text())
    files = outputs(meta)
    if args.check:
        stale = [path for path, text in files.items() if not path.exists() or path.read_text() != text]
        stale += [og_image(n) for n in meta["notes"] if not og_image(n).exists()]
        stale += [p for p in [EXPERIMENTS / "shared" / "og-research.jpg"] if not p.exists()]
        for path in stale:
            print(f"out of date: {path.relative_to(EXPERIMENTS.parent)}")
        fix = "; run experiments/shared/build_meta.py" if stale else ""
        print(f"{len(stale)} file(s) out of date{fix}" if stale else "metadata up to date")
        sys.exit(1 if stale else 0)
    for path, text in files.items():
        path.write_text(text)
    for note in meta["notes"]:
        write_og(note)
    print(f"wrote metadata for the research page and {len(meta['notes'])} note(s), sitemap.xml and llms.txt")


if __name__ == "__main__":
    main()
