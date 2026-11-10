"""Take the guide's screenshots from the static explorer (report/explorer), one per screen of the walkthrough.

    python -m http.server 8790 --directory experiments &       # serve the experiments folder
    .venv/bin/python experiments/layer-explorer/screenshots.py  # writes report/figures/step-*.png

Needs playwright with Chromium. The running example is the "Capital of France" preset.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "report" / "figures"
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790") + "/layer-explorer/report/explorer/"

SHOTS = [
    ("step-1-words", "?preset=france", "section.words"),
    ("step-2-stack", "?preset=france", "section.stack"),
    ("step-3-layer", "?preset=france&layer=20", "section.detail"),
    ("step-4-neuron", "?preset=france&layer=20&neuron=4520&tab=neuron", "section.detail"),
    ("step-5-find", "?preset=france&tab=find", "section.detail"),
    ("step-6-edit", "?preset=france&edit=1", "section.detail"),
    ("step-7-dose", "?preset=france&dose=L20.N4520", "section.detail"),
    ("step-7-dose-4090", "?preset=france&dose=L21.N4090", "section.detail"),
]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1400}, device_scale_factor=1.5)
        for name, query, selector in SHOTS:
            page.goto(BASE + query)
            page.wait_for_selector("#pane-layer table", state="attached")
            if "tab=find" in query:
                page.click("#pane-find button[type=submit]")
                page.wait_for_selector("#pane-find table")
            if "edit=" in query:
                page.wait_for_selector("#pane-edit .compare")
            if "dose=" in query:
                page.wait_for_selector("#pane-dose svg.chart")
            if "neuron=" in query:
                page.wait_for_selector("#pane-neuron .fires")
            page.locator(selector).screenshot(path=str(FIGURES / f"{name}.png"))
            print("wrote", f"figures/{name}.png")
        browser.close()


if __name__ == "__main__":
    main()
