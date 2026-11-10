"""Record one short demo per screen of the guide: the live app taking each action, as looping MP4s with posters.

    uv run steerlab inspect --port 8771 &                              # the live app
    .venv/bin/python experiments/layer-explorer/record_demos.py http://127.0.0.1:8771/

Writes report/media/overview.mp4 and step-N-*.mp4, each with a .jpg poster. Playwright drives Chromium at 1280x800
and records the page; a drawn cursor, a ring on each click and a badge for each key press are added to the page so
the actions are visible. ffmpeg trims the page load, scales to 960 px wide and encodes H.264 with no audio; the
poster is the last frame, which shows the result. Needs playwright and ffmpeg.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "report" / "media"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8771/"
SIZE = {"width": 1280, "height": 800}

# A cursor, click rings and key badges drawn into the page (a recording does not show the system cursor).
OVERLAY = """
addEventListener('DOMContentLoaded', () => {
  const css = document.createElement('style');
  css.textContent = `
    #demo-cursor { position: fixed; z-index: 99999; width: 22px; height: 22px; pointer-events: none;
      left: -40px; top: 0; transform: translate(-3px, -2px); }
    .demo-ring { position: fixed; z-index: 99998; width: 34px; height: 34px; margin: -17px 0 0 -17px;
      border: 3px solid #f2b48e; border-radius: 50%; pointer-events: none; animation: demo-ring .6s ease-out forwards; }
    @keyframes demo-ring { from { transform: scale(.4); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }
    #demo-key { position: fixed; z-index: 99999; left: 24px; bottom: 24px; padding: 6px 14px; border-radius: 6px;
      background: #2a1c14; color: #f2b48e; border: 1px solid #c9561c; font: 600 22px ui-monospace, monospace;
      opacity: 0; transition: opacity .15s; pointer-events: none; }`;
  document.head.append(css);
  const cursor = document.createElement('div');
  cursor.id = 'demo-cursor';
  cursor.innerHTML = '<svg viewBox="0 0 22 22" width="22" height="22"><path d="M2 1 L2 18 L7 13.5 L10.5 21 ' +
    'L13.5 19.6 L10 12.3 L16.5 12.3 Z" fill="#fff" stroke="#15110e" stroke-width="1.5"/></svg>';
  document.body.append(cursor);
  const key = document.createElement('div');
  key.id = 'demo-key';
  document.body.append(key);
  addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
  }, true);
  addEventListener('mousedown', (e) => {
    const ring = document.createElement('div');
    ring.className = 'demo-ring';
    ring.style.left = e.clientX + 'px';
    ring.style.top = e.clientY + 'px';
    document.body.append(ring);
    setTimeout(() => ring.remove(), 700);
  }, true);
  let hide = 0;
  const names = { ArrowUp: '↑', ArrowDown: '↓', ArrowLeft: '←', ArrowRight: '→' };
  addEventListener('keydown', (e) => {
    if (!(e.key in names)) return;
    key.textContent = names[e.key];
    key.style.opacity = 1;
    clearTimeout(hide);
    hide = setTimeout(() => { key.style.opacity = 0; }, 600);
  }, true);
});
"""


def glide(page: Page, selector: str) -> None:
    """Move the drawn cursor smoothly to the centre of the first element matching `selector`."""
    target = page.locator(selector).first
    target.scroll_into_view_if_needed()
    box = target.bounding_box()
    assert box is not None, selector
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)


def click(page: Page, selector: str, pause: float = 0.5) -> None:
    glide(page, selector)
    page.wait_for_timeout(250)
    page.mouse.down()
    page.mouse.up()
    page.wait_for_timeout(int(pause * 1000))


def hover(page: Page, selector: str, pause: float = 0.6) -> None:
    glide(page, selector)
    page.wait_for_timeout(int(pause * 1000))


def title_has(page: Page, pane: str, text: str) -> None:
    page.wait_for_function(
        f"(document.querySelector('#pane-{pane} .pane-title')?.textContent || '').includes({text!r})"
    )


def to_stack(page: Page) -> None:
    """Scroll so the layer stack and the detail panel fill the view."""
    page.evaluate("window.scrollTo(0, document.querySelector('.grid').offsetTop - 12)")
    page.mouse.move(640, 420)
    page.wait_for_timeout(500)


def open_first_neuron(page: Page, layer: int) -> None:
    click(page, f"#rows .row[data-layer='{layer}']", 0.4)
    title_has(page, "layer", f"Layer {layer}")
    click(page, "#pane-layer tbody >> nth=0 >> tr >> nth=0", 0.4)
    page.wait_for_selector("#pane-neuron .fires")


def step_words(page: Page) -> None:
    click(page, "#prompt", 0.2)
    page.fill("#prompt", "")
    page.keyboard.type("The capital of France is", delay=55)
    click(page, "#run")
    page.wait_for_selector("#pane-layer table")
    page.wait_for_timeout(700)
    hover(page, "#guesses .guess >> nth=0", 0.8)
    click(page, "#guesses .guess >> nth=1", 0.3)
    page.wait_for_function("document.querySelectorAll('#guesses .guess')[1].getAttribute('aria-pressed') === 'true'")
    page.wait_for_timeout(1000)
    click(page, "#guesses .guess >> nth=0", 0.3)
    page.wait_for_function("document.querySelectorAll('#guesses .guess')[0].getAttribute('aria-pressed') === 'true'")
    hover(page, "#target-now", 1.2)


def step_stack(page: Page) -> None:
    to_stack(page)
    click(page, "#rows .row[data-layer='23']", 0.6)
    for key in ["ArrowDown"] * 5 + ["ArrowUp"] * 3:
        page.keyboard.press(key)
        page.wait_for_timeout(550)
    title_has(page, "layer", "Layer 21")
    hover(page, "#equation", 2.0)


def step_layer(page: Page) -> None:
    to_stack(page)
    click(page, "#rows .row[data-layer='20']", 0.6)
    title_has(page, "layer", "Layer 20")
    for i in range(3):
        hover(page, f"#pane-layer tbody >> nth=0 >> tr >> nth={i}", 0.7)
    page.evaluate("document.querySelector('.detail').scrollTo({top: 560, behavior: 'smooth'})")
    page.wait_for_timeout(900)
    hover(page, "#pane-layer tbody >> nth=2 >> tr >> nth=0", 1.8)


def step_neuron(page: Page) -> None:
    to_stack(page)
    open_first_neuron(page, 20)
    hover(page, "#pane-neuron .fires .tok >> nth=-1", 0.9)
    hover(page, "#pane-neuron .wordlist >> nth=0", 0.8)
    open_first_neuron(page, 21)
    hover(page, "#pane-neuron .fires .tok >> nth=3", 0.8)
    hover(page, "#pane-neuron .wordlist >> nth=0", 1.4)


def step_find(page: Page) -> None:
    to_stack(page)
    click(page, "#tab-find", 0.5)
    click(page, "#pane-find button[type=submit]", 0.4)
    page.wait_for_selector("#pane-find table")
    for i in range(3):
        hover(page, f"#pane-find tbody >> nth=0 >> tr >> nth={i}", 0.7)
    click(page, "#pane-find tbody >> nth=0 >> tr >> nth=0", 0.4)
    page.wait_for_selector("#pane-neuron .fires")
    page.wait_for_timeout(1600)


def step_edit(page: Page) -> None:
    to_stack(page)
    for layer in (21, 20):
        open_first_neuron(page, layer)
        click(page, "#pane-neuron button:has-text('Switch off and compare')", 0.3)
        page.wait_for_selector("#pane-edit .compare")
        hover(page, "#pane-edit .compare .edited .big", 1.3)


def step_dose(page: Page) -> None:
    to_stack(page)
    open_first_neuron(page, 20)
    click(page, "#pane-neuron button.primary", 0.3)
    page.wait_for_selector("#pane-dose svg.chart")
    hover(page, "#pane-dose svg.chart", 1.5)
    click(page, "#pane-dose form input", 0.2)
    page.fill("#pane-dose form input", "")
    page.keyboard.type("L21.N4090", delay=50)
    click(page, "#pane-dose form button", 0.3)
    page.wait_for_function("(document.querySelector('#pane-dose h3')?.textContent || '').includes('L21.N4090')")
    hover(page, "#pane-dose svg.chart", 2.2)


def step_overview(page: Page) -> None:
    """The whole loop in one clip: run a prompt, walk the stack, open a neuron, switch it off; then the one that
    matters."""
    step_words_short(page)
    to_stack(page)
    click(page, "#rows .row[data-layer='23']", 0.4)
    for _ in range(2):
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(500)
    title_has(page, "layer", "Layer 21")
    hover(page, "#pane-layer tbody >> nth=0 >> tr >> nth=0", 0.8)
    for layer in (21, 20):
        open_first_neuron(page, layer)
        hover(page, "#pane-neuron .wordlist >> nth=0", 0.8)
        click(page, "#pane-neuron button:has-text('Switch off and compare')", 0.3)
        page.wait_for_selector("#pane-edit .compare")
        hover(page, "#pane-edit .compare .edited .big", 1.6)


def step_words_short(page: Page) -> None:
    click(page, "#prompt", 0.2)
    page.fill("#prompt", "")
    page.keyboard.type("The capital of France is", delay=45)
    click(page, "#run")
    page.wait_for_selector("#pane-layer table")
    hover(page, "#guesses .guess >> nth=0", 1.0)


STEPS: list[tuple[str, Callable[[Page], None]]] = [
    ("overview", step_overview),
    ("step-1-words", step_words),
    ("step-2-stack", step_stack),
    ("step-3-layer", step_layer),
    ("step-4-neuron", step_neuron),
    ("step-5-find", step_find),
    ("step-6-edit", step_edit),
    ("step-7-dose", step_dose),
]


def encode(raw: Path, start: float, out: Path) -> None:
    """Trim the page load, scale to 960 px, H.264 at a fixed quality, no audio; the last frame as the poster."""
    video = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(raw), "-an"]
    video += ["-vf", "fps=24,scale=960:-2:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-crf", "27"]
    video += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(video, check=True)
    poster = ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.3", "-i", str(out), "-frames:v", "1"]
    subprocess.run([*poster, "-q:v", "4", str(out.with_suffix(".jpg"))], check=True)


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch()
        for name, act in STEPS:
            if only and name not in only:
                continue
            context = browser.new_context(viewport=SIZE, record_video_dir=tmp, record_video_size=SIZE)
            context.add_init_script(OVERLAY)
            page = context.new_page()
            opened = time.monotonic()
            page.goto(URL + "?preset=france")
            page.wait_for_selector("#pane-layer table")
            page.mouse.move(640, 300)
            page.wait_for_timeout(400)
            start = time.monotonic() - opened
            act(page)
            video = page.video
            context.close()
            assert video is not None
            out = MEDIA / f"{name}.mp4"
            encode(Path(video.path()), start, out)
            seconds = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            print(f"wrote media/{name}.mp4 ({out.stat().st_size / 1e6:.2f} MB, {float(seconds):.1f} s) and .jpg")
        browser.close()


if __name__ == "__main__":
    main()
