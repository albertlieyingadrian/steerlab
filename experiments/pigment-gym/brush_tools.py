"""
Shared helpers for Pigment Gym: the "paint with code" RL environment.

Every script here imports this file:

    from brush_tools import Renderer, gate, SYSTEM_PROMPT, extract_code

What lives here, in the order a rollout uses it:

    model text ──► extract_code ──► static_check ──► Renderer.render ──► gate ──► (reward, the reward check (4_rank_check.py))
                   (find the JS)    (read the code)   (run it in Chromium)  (pass / fail + why)

The design follows the open reproduction at
https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour (Apache-2.0 repo, BSD-3-Clause files).
No code is copied from it; the comments say where an idea came from, because several of its
measured gotchas (seeding Math.random before p5 loads, reading pixels with toDataURL, a fresh page per
sketch, a string-free method allowlist) save hours.
"""

import asyncio
import base64
import io
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

PART_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PART_DIR / "vendor"
P5_JS = VENDOR_DIR / "p5.min.js"  # p5.js 2.3.2, LGPL-2.1
BRUSH_JS = VENDOR_DIR / "p5.brush.js"  # p5.brush 2.2.1, MIT

CANVAS = 400  # the model is asked for a 400x400 canvas: smaller than the original setup's 600, so renders are faster
RENDER_TIMEOUT_S = 20.0
MIN_COVERAGE = 0.005  # at least 0.5% of pixels must differ from the background (upstream calibrated the same floor)

# The brush methods the model may use. None of them takes a string name, so the model cannot
# invent a brush name like "pencil" that doesn't exist (the upstream repo measured that this
# family of mistakes was the biggest cause of crashes).
ALLOWED_METHODS = [
    "scaleBrushes",
    "noStroke",
    "fill",
    "noFill",
    "fillBleed",
    "fillTexture",
    "beginShape",
    "vertex",
    "endShape",
    "circle",
]
PAINTING_CALLS = ["brush.circle", "brush.endShape"]  # the calls that actually put paint down

SYSTEM_PROMPT_NO_EXAMPLE = f"""You paint watercolours by writing a p5.js sketch that uses the p5.brush library (the global `brush`).

Use exactly this skeleton:

function setup() {{
  createCanvas({CANVAS}, {CANVAS}, WEBGL);
  brush.scaleBrushes(2);
  noLoop();
}}

function draw() {{
  translate(-width / 2, -height / 2);
  background("#f4efe4");
  // painting goes here
}}

After the translate, x and y run from 0 to {CANVAS} with (0, 0) at the top left. The centre is ({CANVAS // 2}, {CANVAS // 2}).
Every coordinate must be between 0 and {CANVAS}. Colours are hex strings like "#e08a72".

Only these brush methods exist. Do not call any other brush method, and never use bare p5 shapes
such as ellipse, rect, line, vertex or beginShape:
  brush.noStroke()
  brush.fill(colorHex, opacity)      opacity 0 to 255
  brush.noFill()
  brush.fillBleed(amount)            0 to 1
  brush.fillTexture(amount, border)  0 to 1
  brush.beginShape(curvature)        0 to 1
  brush.vertex(x, y)
  brush.endShape(true)
  brush.circle(x, y, radius)

Every mark is a filled shape: call brush.fill, then brush.circle, or brush.beginShape, three or more
brush.vertex calls, and brush.endShape(true). Paint back to front: sky first, then ground, then details.
No text, no images, no network. Reply with one ```javascript code block and nothing else."""

# One tiny worked example. The upstream finding is "allowlist, no examples" (measured on bigger models).
# For Qwen2.5-Coder-0.5B we measured the opposite: 0/16 sketches passed the gate without this example and
# 6/16 with it (the benchmark (scripts 1-3) runs both). The example paints a wash and a disc; it says nothing about sunsets.
PROMPT_EXAMPLE = f"""

Here is a complete example that paints a red disc on a blue wash. Follow its pattern:

```javascript
function setup() {{
  createCanvas({CANVAS}, {CANVAS}, WEBGL);
  brush.scaleBrushes(2);
  noLoop();
}}

function draw() {{
  translate(-width / 2, -height / 2);
  background("#f4efe4");
  brush.noStroke();
  brush.fillTexture(0.5, 0.4);
  brush.fill("#7fb2d9", 180);
  brush.beginShape(0.3);
  brush.vertex(0, 0);
  brush.vertex(400, 0);
  brush.vertex(400, 200);
  brush.vertex(0, 200);
  brush.endShape(true);
  brush.fill("#d1495b", 220);
  brush.circle(200, 150, 60);
}}
```"""

SYSTEM_PROMPT = SYSTEM_PROMPT_NO_EXAMPLE + PROMPT_EXAMPLE


# ---------------------------------------------------------------------------------------------
# 1. Reading the code (no browser yet)
# ---------------------------------------------------------------------------------------------

FENCE = re.compile(r"```(?:javascript|js)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Pull the JavaScript out of a model reply. Prefers a ``` fence; otherwise returns the raw text.

    An unclosed fence (the model ran out of tokens) still yields the code after the opening ```.
    """
    match = FENCE.search(text)
    if match:
        return match.group(1).strip()
    opened = re.search(r"```(?:javascript|js)?\s*\n(.*)", text, re.DOTALL)
    if opened:
        return opened.group(1).strip()
    return text.strip()


# Each rule: (name, regex). A hit is a violation. This is a cheap first line of defence; the browser
# sandbox (network blocked, fresh page per sketch) is the second, and it doesn't trust this list.
FORBIDDEN = [
    ("text", r"\btext\s*\("),
    ("image", r"\b(image|loadImage|createImg|loadFont|textFont)\s*\("),
    ("network", r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|importScripts|sendBeacon)\b|\bimport\s*\("),
    ("dom", r"\b(document|localStorage|sessionStorage|eval|Function)\s*[.(]"),
    ("bare_primitive", r"(?<![\w.])(ellipse|rect|line|triangle|quad|arc|point|square|beginShape|vertex|endShape|circle)\s*\("),
]


def static_check(code):
    """Read the source without running it. Returns a list of violation names (empty = clean)."""
    violations = []
    for name, pattern in FORBIDDEN:
        if re.search(pattern, code):
            violations.append(name)
    used = set(re.findall(r"\bbrush\.(\w+)", code))
    unknown = sorted(used - set(ALLOWED_METHODS))
    if unknown:
        violations.append("unknown_brush_method:" + ",".join(unknown))
    if not any(call in code for call in PAINTING_CALLS):
        violations.append("no_painting_call")
    if "function setup" not in code:
        violations.append("no_setup")
    return violations


def brush_methods_used(code):
    return sorted(set(re.findall(r"\bbrush\.(\w+)", code)))


# ---------------------------------------------------------------------------------------------
# 2. Running the code: a headless Chromium pool
# ---------------------------------------------------------------------------------------------

# Seed Math.random BEFORE p5 and p5.brush load. p5.brush captures random seeds when its file is
# evaluated, so seeding later means the same code paints a different picture each time, and RL would
# read that noise as signal. (Measured upstream; we re-check determinism in the renderer (brush_tools.py).)
PAGE_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<script>
(function () {{ let s = ({seed} >>> 0) || 1;
  Math.random = function () {{ s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }}; }})();
window.__errors = []; window.__setupDone = false;
window.onerror = (m) => {{ window.__errors.push(String(m)); return true; }};
window.addEventListener("unhandledrejection", (e) => window.__errors.push(String(e.reason)));
</script>
<script>{p5}</script>
<script>{brush}</script>
<style>html,body{{margin:0;background:#fff}}</style>
</head><body>
<script>{sketch}
</script>
<script>
// Wrap the sketch's setup so we know when it has run (p5 calls setup after the load event).
if (typeof window.setup === "function") {{
  const original = window.setup;
  window.setup = function () {{ try {{ return original.apply(this, arguments); }} finally {{ window.__setupDone = true; }} }};
}} else {{ window.__setupDone = true; }}
</script></body></html>"""

# "Done" = setup ran, and either there is no draw() or draw() ran once and the loop stopped.
DONE_JS = (
    "!!window.__setupDone && (typeof draw !== 'function' || "
    "(typeof frameCount !== 'undefined' && frameCount >= 1 && (typeof isLooping !== 'function' || !isLooping())))"
)
# "Dead" = it threw after setup; whatever is on the canvas is all there will ever be.
DEAD_JS = "window.__errors.length > 0 && !!window.__setupDone"

# p5.brush needs a WEBGL canvas. On a Mac, headless Chromium gets one from the real GPU (Metal) with no
# flags. On a Linux server with no GPU you need software WebGL (SwiftShader) instead, which we measured
# here to be ~4x slower on a one-circle sketch and >50x slower on the meadow (it timed out).
SWIFTSHADER_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"]
CHROMIUM_ARGS = [] if sys.platform == "darwin" else SWIFTSHADER_ARGS


@dataclass
class Render:
    png: bytes | None  # None when no canvas was ever created
    coverage: float  # fraction of pixels that differ from the background colour
    finished: bool  # stopped drawing before the timeout
    errors: list = field(default_factory=list)  # JavaScript errors, in order
    seconds: float = 0.0
    blocked_requests: int = 0  # network requests the sandbox refused


def paint_coverage(png_bytes):
    """Fraction of pixels that are 'paint': far (RGB distance > 32) from the most common colour.

    The most common colour is the paper, so a sketch that washes the whole canvas in one colour
    and paints nothing else still scores ~0.
    """
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    paper = Image.new("RGBA", image.size, (255, 255, 255, 255))
    paper.alpha_composite(image)
    rgb = np.asarray(paper.convert("RGB"), dtype=np.float32)
    bins = rgb.astype(np.uint32) >> 4  # coarse buckets so slight texture noise counts as the same colour
    packed = (bins[..., 0] << 8) | (bins[..., 1] << 4) | bins[..., 2]
    values, counts = np.unique(packed, return_counts=True)
    background = rgb[packed == values[np.argmax(counts)]].mean(axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=-1))
    return float((distance > 32.0).mean())


class Renderer:
    """A pool of headless Chromium browsers that turns sketch source into PNGs.

    Use as a context manager:

        with Renderer(pool_size=4) as renderer:
            results = renderer.render_many([code1, code2, ...])

    Like a database connection pool: starting a browser costs ~0.5 s, so we start a few once and
    reuse them. Each sketch gets a FRESH page (p5's global mode breaks if you reuse a page), and
    parallelism is across browsers, because pages inside one browser share one software-GL process.
    """

    def __init__(self, pool_size=4, timeout_s=RENDER_TIMEOUT_S):
        self.pool_size = pool_size
        self.timeout_s = timeout_s
        self._p5 = P5_JS.read_text()
        self._brush = BRUSH_JS.read_text()
        self._loop = asyncio.new_event_loop()
        self._playwright = None
        self._browsers = []
        self._queue = None

    # -- lifecycle --
    def __enter__(self):
        self._loop.run_until_complete(self._start())
        return self

    def __exit__(self, *exc):
        self._loop.run_until_complete(self._stop())
        self._loop.close()

    async def _start(self):
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._queue = asyncio.Queue()
        for _ in range(self.pool_size):
            browser = await self._playwright.chromium.launch(args=CHROMIUM_ARGS)
            self._browsers.append(browser)
            self._queue.put_nowait(browser)

    async def _stop(self):
        for browser in self._browsers:
            await browser.close()
        if self._playwright:
            await self._playwright.stop()

    # -- rendering --
    def render(self, code, seed=0):
        return self.render_many([code], seed=seed)[0]

    def render_many(self, codes, seed=0):
        """Render several sketches in parallel (up to pool_size at once). Same seed for all, so two
        sketches in a GRPO group differ because of their code, not their dice."""

        async def run_all():
            return await asyncio.gather(*[self._render_one(code, seed) for code in codes])

        return self._loop.run_until_complete(run_all())

    async def _render_one(self, code, seed):
        browser = await self._queue.get()
        try:
            return await self._render_in(browser, code, seed)
        finally:
            self._queue.put_nowait(browser)

    async def _render_in(self, browser, code, seed):
        started = time.perf_counter()
        context = await browser.new_context(viewport={"width": CANVAS + 50, "height": CANVAS + 50})
        blocked = []

        async def refuse(route):
            blocked.append(route.request.url)
            await route.abort()

        await context.route("**/*", refuse)  # every network request is refused: the sandbox, not the regex, is the real wall
        page = await context.new_page()
        try:
            sketch = code.replace("</script", "<\\/script")  # a sketch must not be able to close our <script> tag
            html = PAGE_TEMPLATE.format(seed=seed, p5=self._p5, brush=self._brush, sketch=sketch)
            # "commit" not "load": a heavy sketch blocks the page while painting, so "load" fires late.
            await page.set_content(html, wait_until="commit")
            finished = False
            while time.perf_counter() - started < self.timeout_s:
                remaining = max(0.1, self.timeout_s - (time.perf_counter() - started))
                try:
                    # wait_for: a sketch stuck in `while (true) {}` freezes the page, and then even
                    # evaluate() never returns. The timeout is what stops one bad sketch stalling training.
                    if await asyncio.wait_for(page.evaluate(DONE_JS), timeout=remaining):
                        finished = True
                        break
                    if await asyncio.wait_for(page.evaluate(DEAD_JS), timeout=remaining):
                        break
                except TimeoutError:
                    raise
                except Exception:  # noqa: BLE001, S110 (page still loading its scripts)
                    pass
                await asyncio.sleep(0.04)
            # A WEBGL canvas is cleared before a screenshot can see it, so read pixels inside the page.
            data_url = await asyncio.wait_for(
                page.evaluate(
                    "document.querySelector('canvas') ? document.querySelector('canvas').toDataURL('image/png') : null"
                ),
                timeout=5,
            )
            errors = await asyncio.wait_for(page.evaluate("window.__errors || []"), timeout=5)
        except Exception as error:  # noqa: BLE001 (the page itself is stuck, e.g. an infinite loop blocks evaluate)
            data_url, errors, finished = None, [f"render crashed: {type(error).__name__}"], False
        finally:
            try:
                await asyncio.wait_for(context.close(), timeout=5)
            except Exception:  # noqa: BLE001, S110 (a frozen page can refuse to close; the browser is reused anyway)
                pass
        png = base64.b64decode(data_url.split(",", 1)[1]) if data_url else None
        coverage = paint_coverage(png) if png else 0.0
        return Render(png, coverage, finished, list(errors), time.perf_counter() - started, len(blocked))


# ---------------------------------------------------------------------------------------------
# 3. The gate: pass/fail, with reasons
# ---------------------------------------------------------------------------------------------


@dataclass
class GateResult:
    passed: bool
    violations: list
    coverage: float


def gate(code, render):
    """Admission check before anyone judges taste. Fails if the code breaks a rule, the browser
    reported an error, no canvas came back, or almost nothing was painted."""
    violations = static_check(code)
    if render.png is None:
        violations.append("no_canvas")
    if render.errors:
        violations.append("js_error")
    if not render.finished:
        violations.append("timeout_or_crash")
    if render.blocked_requests:
        violations.append("network_attempt")
    if render.png is not None and render.coverage < MIN_COVERAGE:
        violations.append("blank")
    return GateResult(passed=not violations, violations=violations, coverage=render.coverage)


def contact_sheet(pngs, labels, path, columns=4, tile=200):
    """Paste PNGs (or None) into one grid image with a label under each. Handy for before/after."""
    from PIL import ImageDraw

    rows = max(1, (len(pngs) + columns - 1) // columns)  # an empty list still makes a (blank) one-row sheet
    sheet = Image.new("RGB", (columns * tile, rows * (tile + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (png, label) in enumerate(zip(pngs, labels)):
        x, y = (index % columns) * tile, (index // columns) * (tile + 18)
        if png:
            sheet.paste(Image.open(io.BytesIO(png)).convert("RGB").resize((tile, tile)), (x, y))
        else:
            draw.rectangle([x + 4, y + 4, x + tile - 4, y + tile - 4], outline="#c33")
            draw.text((x + 10, y + tile // 2), "no canvas", fill="#c33")
        draw.text((x + 4, y + tile + 3), label[:34], fill="black")
    sheet.save(path)
    return path
