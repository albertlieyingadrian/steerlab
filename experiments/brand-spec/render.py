"""Render pages in headless Chromium at 1280 px and run the extractor (extract.js) on each.

    from render import extract_pages
    facts = extract_pages(pages)            # pages: [{id, brand, domain, page_type, html}] -> [{..., elements}]
    facts = extract_url(url, brand, domain)  # a live page (the real-page check)

Local Chromium through Playwright; nothing leaves the machine for generated pages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from common import HERE, VIEWPORT, load_spec

JS = (HERE / "extract.js").read_text(encoding="utf-8")


def components(brand: str) -> list[list[str]]:
    comps = load_spec(brand)["components"]
    return [[c["class"], c["role"]] for c in sorted(comps, key=lambda c: -len(c["class"]))]


def extract_pages(pages: list[dict]) -> list[dict]:
    from playwright.sync_api import sync_playwright

    out = []
    comps = {b: components(b) for b in {p["brand"] for p in pages}}
    with sync_playwright() as pw, tempfile.TemporaryDirectory() as tmp:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT, "height": 900})
        for p in pages:
            f = Path(tmp) / f"{p['id']}.html"
            f.write_text(p["html"], encoding="utf-8")
            page.goto(f.as_uri())
            els = page.evaluate(JS, comps[p["brand"]])
            out.append(
                {
                    "id": p["id"],
                    "brand": p["brand"],
                    "domain": p["domain"],
                    "page_type": p["page_type"],
                    "elements": els,
                }
            )
        browser.close()
    return out


def extract_url(url: str, brand: str, domain: str, page_type: str = "content") -> dict:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT, "height": 900})
        page.goto(url, wait_until="networkidle", timeout=60000)
        els = page.evaluate(JS, components(brand))
        browser.close()
    return {"id": url, "brand": brand, "domain": domain, "page_type": page_type, "elements": els}
