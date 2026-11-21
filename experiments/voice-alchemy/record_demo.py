"""Record the alchemy-table demo from the built page (local file, no network beyond fonts).

    .venv/bin/python experiments/voice-alchemy/record_demo.py

Drives report/index.html with Playwright at 960 x 600: scrolls to the table, picks the assistant-like direction card,
opens a real reply, picks the playful card, and lingers on the measured mix. Writes report/media/demo.mp4,
report/media/demo.gif (960 px wide, kept under 3 MB) and report/media/demo-still.png (the mix, for no-JS and print).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import va

REPORT = va.HERE / "report"
MEDIA = REPORT / "media"
W, H = 960, 600


def main() -> None:
    from playwright.sync_api import sync_playwright

    MEDIA.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(tmp),
            record_video_size={"width": W, "height": H},
            color_scheme="light",
        )
        page = ctx.new_page()
        page.goto((REPORT / "index.html").as_uri())
        page.wait_for_selector(".va-card")
        page.evaluate("document.querySelector('#table').scrollIntoView({block: 'start'})")
        page.wait_for_timeout(1800)

        def card(trait: str):
            return page.locator(f'button[data-pick="{trait}"]')

        card("assistant").scroll_into_view_if_needed()
        page.wait_for_timeout(1200)
        card("assistant").click()
        page.wait_for_timeout(1300)
        page.locator('button[data-open="direction-assistant"]').click()
        page.wait_for_timeout(2600)
        card("playful").scroll_into_view_if_needed()
        page.wait_for_timeout(900)
        card("playful").click()
        page.wait_for_timeout(1000)
        page.locator(".va-mix").scroll_into_view_if_needed()
        page.evaluate("window.scrollBy(0, -40)")
        page.wait_for_timeout(2600)
        page.screenshot(path=str(MEDIA / "demo-still.png"))
        page.evaluate("window.scrollBy({top: 260, behavior: 'smooth'})")
        page.wait_for_timeout(3500)
        page.locator('button.va-tab[data-type="community"]').scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.locator('button.va-tab[data-type="community"]').click()
        page.wait_for_timeout(2400)
        video = page.video.path()
        ctx.close()
        browser.close()
    webm = Path(video)
    mp4, gif = MEDIA / "demo.mp4", MEDIA / "demo.gif"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(webm),
            "-vf",
            "scale=960:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "26",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    # duplicate frames (the pauses) are dropped and the GIF keeps variable frame timing, so idle time costs nothing
    for colors in (96, 64, 48):
        pal = tmp / "pal.png"
        vf = "fps=8,mpdecimate,scale=960:-1:flags=lanczos"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-vf",
                f"{vf},palettegen=max_colors={colors}:stats_mode=diff",
                str(pal),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-i",
                str(pal),
                "-lavfi",
                f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-fps_mode",
                "vfr",
                str(gif),
            ],
            check=True,
        )
        if gif.stat().st_size <= 3 * 1024 * 1024:
            break
    shutil.rmtree(tmp, ignore_errors=True)
    for f in (mp4, gif, MEDIA / "demo-still.png"):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
