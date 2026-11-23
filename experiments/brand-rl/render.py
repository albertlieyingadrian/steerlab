"""Render policy pages in headless Chromium and read them with both extractors in one load.

    from render import render_many
    outs = render_many(htmls, shots=False)   # [{"ok", "elements" (Episode 5 facts), "ep2" (Episode 2 record), "jpeg"}]

- Episode 5's extract.js gives the element facts the brand checker and the copy check read (roles from the brand's
  class names, computed styles).
- Episode 2's extract.js gives the block tree the slop probe reads (Weisfeiler-Lehman histogram of blocks).
- Optional first-screen JPEG (1280 x 800) for the screenshot probe and the judge (evaluation only).

Page scripts are disabled and every network request is refused, so a page is read as written: nothing it fetches or
runs can change what the extractors see. Pages render concurrently in one browser (an asyncio semaphore).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
VIEW_W, VIEW_H = 1280, 800
JS5 = (EXP / "brand-spec" / "extract.js").read_text(encoding="utf-8")
JS2 = (EXP / "design-index" / "extract.js").read_text(encoding="utf-8")
PARK_RE = "domain (is )?for sale|parked free|this domain|coming soon|under construction"


def components() -> list[list[str]]:
    spec = json.loads((EXP / "brand-spec" / "spec" / "govuk_spec.json").read_text())
    return [[c["class"], c["role"]] for c in sorted(spec["components"], key=lambda c: -len(c["class"]))]


COMPONENTS = components()


async def _one(ctx, html: str, shots: bool, timeout_ms: int) -> dict:
    page = await ctx.new_page()
    try:
        await page.route("**/*", lambda r: r.abort() if not r.request.url.startswith("data:") else r.continue_())
        await page.set_content(html, wait_until="load", timeout=timeout_ms)
        els = await page.evaluate(JS5, COMPONENTS)
        ep2 = await page.evaluate(JS2, PARK_RE)
        jpeg = await page.screenshot(type="jpeg", quality=85, timeout=timeout_ms) if shots else None
        return {"ok": True, "elements": els, "ep2": ep2, "jpeg": jpeg}
    except Exception as e:  # a page that will not load or evaluate counts as invalid, never as a crash
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "elements": [], "ep2": {}, "jpeg": None}
    finally:
        await page.close()


async def _many(htmls: list[str | None], shots: bool, concurrency: int, timeout_ms: int) -> list[dict]:
    from playwright.async_api import async_playwright

    sem = asyncio.Semaphore(concurrency)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--disable-dev-shm-usage"])
        ctx = await browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H}, java_script_enabled=False)

        async def run(h):
            if not h:
                return {"ok": False, "error": "no html", "elements": [], "ep2": {}, "jpeg": None}
            async with sem:
                return await _one(ctx, h, shots, timeout_ms)

        out = await asyncio.gather(*(run(h) for h in htmls))
        await ctx.close()
        await browser.close()
    return list(out)


def render_many(htmls: list[str | None], shots: bool = False, concurrency: int = 8, timeout_ms: int = 15000):
    return asyncio.run(_many(htmls, shots, concurrency, timeout_ms))
