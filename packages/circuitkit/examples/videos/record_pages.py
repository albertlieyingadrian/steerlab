"""Record browser clips of the HTML reports and the docs, with a caption bar burned in, for the demo videos.

python examples/videos/record_pages.py   # needs the reports in /tmp (run the VHS tapes first) and `mint dev` on :3333
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
CAPTION_JS = """(text) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div'); el.id = '__cap';
    el.style.cssText = 'position:fixed;left:50%;bottom:26px;transform:translateX(-50%);max-width:1100px;' +
      'z-index:99999;' +
      'background:rgba(21,17,14,.92);color:#f3ede6;font:500 22px/1.35 Inter,system-ui,sans-serif;padding:12px 22px;' +
      'border-radius:10px;border-left:4px solid #e07a3f;box-shadow:0 6px 24px rgba(0,0,0,.25)';
    document.body.appendChild(el);
  }
  el.textContent = text;
}"""

CLIPS = {
    "validate_page": (
        "file:///tmp/ck_validate.html",
        [
            (0, 3500, "The validator's report: one row per group of features, with its verdict."),
            (300, 4500, "Grey ticks: what ablating random size-matched features does. Orange: the group itself."),
            ("h:What", 4000, "Every verdict lists its numbers, and the rules are printed with the report."),
        ],
    ),
    "diff_page": (
        "file:///tmp/ck_diff.html",
        [
            (0, 3500, "The diff report: features gained, lost and reweighted by RL, against a swap null."),
            ("h:Top", 4500, "The top ten features, each checked against a control run that did not learn to hack."),
            (
                "h:Stability",
                4000,
                "Stability: two random halves of the prompts agree on 70% of the top ten (chance: 0.2%).",
            ),
            (
                "h:Validity",
                4500,
                "Validity: the transcoders were trained on the base model, so reconstruction error is checked.",
            ),
        ],
    ),
    "labels_page": (
        "file:///tmp/ck_labels.html",
        [
            (0, 4000, "The labels report: each feature's label, the explainer's confidence and the held-out scores."),
            (
                "h:Calibration",
                4000,
                "Calibration: how well stated confidence and detector probabilities match what happened.",
            ),
        ],
    ),
    "labels_results": (
        "http://localhost:3333/guides/labels",
        [
            (
                99999,
                5000,
                "On 50 Gemma-2-2B features, Haiku 5.5's labels match Neuronpedia's on detection (AUROC 0.80).",
            ),
            (99999, 5000, "...and beat them on simulation (r = 0.45 vs 0.37), for $0.26 in total."),
            (
                99999,
                4500,
                "Stated confidence barely predicts the score (r = 0.11): trust the score, not the confidence.",
            ),
        ],
    ),
    "docs_tour": (
        "http://localhost:3333/",
        [
            (0, 4000, "circuitkit: three tools on top of circuit-tracer attribution graphs."),
            (600, 3500, "Why: in two studies the graph's top features read the behaviour but did not drive it."),
            (
                "/quickstart",
                4500,
                "A five-minute quickstart runs every tool offline: toy model, bundled graphs, cached replies.",
            ),
            ("/guides/validate", 4000, "Each tool has a guide: the checks, the verdicts and the parameters."),
            ("/api/diff", 3500, "And an API reference. Every code snippet in these docs runs in the test suite."),
        ],
    ),
}


def record(name: str, url: str, steps: list) -> Path:
    tmp = HERE / f"_rec_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(tmp),
            record_video_size={"width": 1280, "height": 720},
            color_scheme="light",
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(600)
        cues = []
        t = 0.6
        for target, ms, text in steps:
            if isinstance(target, str) and target.startswith("h:"):
                page.evaluate(
                    "(t) => { const h = [...document.querySelectorAll('h2')].find(e => e.textContent.startsWith(t));"
                    " if (h) window.scrollTo({top: h.getBoundingClientRect().top + window.scrollY - 24,"
                    " behavior: 'smooth'}); }",
                    target[2:],
                )
            elif isinstance(target, str):
                page.goto("http://localhost:3333" + target, wait_until="networkidle")
                page.wait_for_timeout(500)
                t += 1.0
            else:
                page.evaluate(f"window.scrollTo({{top: {target}, behavior: 'smooth'}})")
            page.evaluate(CAPTION_JS, text)
            cues.append((t, t + ms / 1000, text))
            page.wait_for_timeout(ms)
            t += ms / 1000
        ctx.close()
        b.close()
    video = next(tmp.glob("*.webm"))
    out = HERE / f"{name}.webm"
    video.replace(out)
    shutil.rmtree(tmp, ignore_errors=True)
    (HERE / f"{name}.cues.json").write_text(json.dumps(cues))
    return out


CARD = """<html><body style="margin:0;width:1280px;height:720px;background:#15110e;display:flex;flex-direction:column;
justify-content:center;padding:0 110px;box-sizing:border-box;font-family:Inter,system-ui,sans-serif">
<div style="color:#e07a3f;font-size:24px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:18px">
{kicker}</div>
<div style="color:#f3ede6;font-size:54px;font-weight:600;line-height:1.15;margin-bottom:26px">{title}</div>
<div style="color:#b3aaa0;font-size:26px;line-height:1.5;max-width:1000px">{body}</div></body></html>"""


def card(name: str, kicker: str, title: str, body: str) -> Path:
    out = HERE / f"{name}.png"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.set_content(CARD.format(kicker=kicker, title=title, body=body))
        pg.screenshot(path=str(out))
        b.close()
    return out


if __name__ == "__main__" and sys.argv[1:2] == ["cards"]:
    cards = json.loads((HERE / "cards.json").read_text())
    for name, c in cards.items():
        print(card(name, c["kicker"], c["title"], c["body"]))
    sys.exit(0)

if __name__ == "__main__":
    names = sys.argv[1:] or list(CLIPS)
    for n in names:
        url, steps = CLIPS[n]
        print(record(n, url, steps))
