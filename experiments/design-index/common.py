"""
Shared pieces for Taste Labs Episode 2 (has the web converged?): paths, the fixed categories and queries, the
AI-builder host list, the dev/held-out split and the snapshot years. Imported on the laptop and inside Modal
containers, so it uses only the standard library.

Plan (fixed before any render was analysed): the plan
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # small derived inputs that ship (seed lists hold public domain names only)
RESULTS = HERE / "results"
CACHE = Path.home() / ".cache" / "steerlab" / "design-index"  # raw renders, never committed

YEARS = list(range(2014, 2027))
VIEWPORT = {"width": 1280, "height": 800}

# Twelve industries, chosen for spread (finance to pets), each with phrasings for Exa's neural search.
CATEGORIES = {
    "finance": [
        "the homepage of a small regional bank or credit union",
        "a fintech startup's landing page",
        "an independent financial advisor's website",
        "an accounting firm's homepage",
    ],
    "health": [
        "a family medical clinic's website",
        "a dental practice homepage",
        "a physiotherapy clinic's website",
        "a private hospital's homepage",
    ],
    "law": [
        "a small law firm's website",
        "an immigration lawyer's homepage",
        "a personal injury attorney's website",
        "a notary or legal services office homepage",
    ],
    "restaurant": [
        "an independent restaurant's website with its menu",
        "a neighbourhood cafe's homepage",
        "a bakery's website",
        "a family-run pizzeria's homepage",
    ],
    "pets": [
        "a pet shop's website",
        "a veterinary clinic's homepage",
        "a dog grooming salon's website",
        "a pet boarding kennel's homepage",
    ],
    "software": [
        "a developer tools startup's landing page",
        "a B2B SaaS product homepage",
        "an open-source database company's website",
        "a project management software homepage",
    ],
    "fashion": [
        "an independent clothing brand's online store",
        "a jewellery maker's shop website",
        "a shoe brand's homepage",
        "a vintage clothing boutique's website",
    ],
    "nonprofit": [
        "a local charity's homepage",
        "an environmental nonprofit's website",
        "a community food bank's website",
        "an animal rescue organisation's homepage",
    ],
    "realestate": [
        "a local real estate agency's website",
        "a property management company's homepage",
        "a home builder's website",
        "an apartment rental community's homepage",
    ],
    "education": [
        "a small private school's website",
        "a language school's homepage",
        "a university department's website",
        "a music lessons studio's homepage",
    ],
    "travel": [
        "a boutique hotel's website",
        "a bed and breakfast homepage",
        "a local tour operator's website",
        "a campsite or holiday park homepage",
    ],
    "studio": [
        "an architecture firm's portfolio website",
        "an interior design studio's homepage",
        "a graphic design agency's portfolio",
        "a photographer's portfolio website",
    ],
}
REGIONS = [
    "",
    " in the United Kingdom",
    " in Australia",
]  # widens the seed beyond US results

# Default hosting subdomains of five AI site builders (fixed before any search). Sites on these hosts are the "AI-built"
# class. The list lives in data/ai_hosts.txt, which stays in the lab so the public note and code name no products.
_HOSTS_FILE = DATA / "ai_hosts.txt"
AI_HOSTS = _HOSTS_FILE.read_text().split() if _HOSTS_FILE.exists() else []

# Hosts that are platforms, not a business's own design: dropped from the human seed.
PLATFORM_HOSTS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "yelp.com",
    "tripadvisor.com",
    "wikipedia.org",
    "google.com",
    "medium.com",
    "reddit.com",
    "github.com",
    "etsy.com",
    "amazon.com",
    "booking.com",
    "zillow.com",
    "glassdoor.com",
    "indeed.com",
    "pinterest.com",
    "tiktok.com",
    "yellowpages.com",
    "bbb.org",
    "opentable.com",
    "doordash.com",
    "ubereats.com",
    "airbnb.com",
    "expedia.com",
    "healthgrades.com",
    "zocdoc.com",
    "avvo.com",
    "justia.com",
    "findlaw.com",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
    "producthunt.com",
    "substack.com",
    "wix.com",
    "squarespace.com",
    "shopify.com",
    "wordpress.com",
    "blogspot.com",
    "weebly.com",
    "godaddysites.com",
    "rightmove.co.uk",
    "realestate.com.au",
}


def domain_of(url: str) -> str:
    host = urlsplit(url if "://" in url else "http://" + url).hostname or ""
    return host.lower().removeprefix("www.")


def registrable(host: str) -> str:
    """Last two labels (three for co.uk-style suffixes); good enough to match platform hosts."""
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov", "edu"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_ai_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in AI_HOSTS)


def split_of(key: str) -> str:
    """Fixed 30/70 dev/held-out split by a hash of the domain (or page id)."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return "dev" if h % 10 < 3 else "held"


def page_id(source: str, key: str, year: int | None = None) -> str:
    raw = f"{source}|{key}|{year or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
