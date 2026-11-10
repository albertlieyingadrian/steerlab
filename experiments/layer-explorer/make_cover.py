"""Make the cover and the research-page thumbnail: screenshots of the explorer itself, showing a sample output.

    .venv/bin/python experiments/layer-explorer/make_cover.py

Opens the saved explorer (report/explorer/, served from a local port this script starts) on the "Capital of France"
preset with layer 21 open: the prompt, its tokens, the guesses with " Paris" at 30% as the target, the layer stack
and the layer panel. The preset buttons, help lines and footer are hidden so the whole sample fits one frame; nothing
else is changed. The app has one (dark) theme, so the cover is dark.

Writes report/figures/hero.jpg (1600x900, 16:9) and experiments/shared/thumbs/layer-explorer.jpg (720x480, 3:2,
a closer framing so the bars and " Paris" stay readable at thumbnail size). build_meta.py crops og.jpg from the hero.
Needs playwright with Chromium.
"""

from __future__ import annotations

import functools
import io
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "layer-explorer.jpg"
QUERY = "?preset=france&layer=21"
TRIM = (
    ".presets, .note, #target-form, .foot, .legend, #stack-hint, .label .hint { display: none !important; }"
    ".wrap { gap: 10px; padding-top: 10px; } .panel { padding: 12px 14px; } .label { margin: 6px 0 4px; }"
)
SHOTS = [(HERO, (1280, 720), (1600, 900)), (THUMB, (840, 560), (720, 480))]


def main() -> None:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(HERE / "report"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/explorer/{QUERY}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for out, (width, height), size in SHOTS:
                page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
                page.goto(url)
                page.wait_for_function("document.querySelector('#pane-layer .pane-title')?.textContent === 'Layer 21'")
                page.add_style_tag(content=TRIM)
                page.wait_for_timeout(300)
                shot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
                out.parent.mkdir(parents=True, exist_ok=True)
                shot.resize(size, Image.LANCZOS).save(out, quality=88)
                print("wrote", out.name, f"{size[0]}x{size[1]}")
                page.close()
            browser.close()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
