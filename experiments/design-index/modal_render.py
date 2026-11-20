"""
Render pages with Playwright (Chromium, 1280x800) on Modal CPU and extract DOM/CSS evidence for the feature step.

COSTS MONEY (CPU containers, cents per hour). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/design-index/modal_render.py --jobs <jobs.jsonl>

A job is {"pid", "url", "mode": "live"|"wayback"|"file", "ts"?}. Per page the container writes to the Volume:
  shots/<pid>.jpg   the first screen (1280x800), JPEG q85      (never leaves the Volume except for local checks)
  dom/<pid>.json    status, request counts, flags, and compact element and block lists (no page text)

Politeness:
  wayback  one container, at most 3 pages at once, at least 2 s between navigations, every request outside
           web.archive.org blocked (and counted, so a snapshot cannot load today's assets); a 429 or a refused
           connection pauses all navigations for 30 s, then 90 s, and the page is retried twice.
  live     robots.txt checked for "/" (user agent "*"); disallowed homepages are skipped; one page per domain.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VOL = "/vol"
app = modal.App("design-index-render")
volume = modal.Volume.from_name("design-index", create_if_missing=True)
image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install("playwright==1.48.0", "pillow==11.0.0")
    .run_commands("playwright install --with-deps chromium")
    .add_local_file(HERE / "extract.js", "/root/extract.js")
)
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36 design-index-research"
)
PARK_RE = (
    r"domain (is )?for sale|buy this domain|parked free|this domain (name )?(may be|is) for sale|"
    r"account (has been )?suspended|under construction|coming soon|site not found|page not found|"
    r"wayback machine has not archived|hrm\.|got an http 30[12] response|this site can.t be reached|"
    r"index of /|default web site page|it works!|welcome to nginx|performing security verification|"
    r"verify you are human|just a moment|access (is )?(temporarily )?(denied|restricted)|attention required"
)


def wayback_url(ts: str, domain: str) -> str:
    return f"https://web.archive.org/web/{ts}if_/http://{domain}/"


async def render_one(ctx, job, js, state, out):
    from PIL import Image

    pid = job["pid"]
    url = (
        wayback_url(job["ts"], job["domain"])
        if job["mode"] == "wayback"
        else job["url"]
    )
    page = await ctx.new_page()
    counts = {"req": 0, "failed": 0, "blocked": 0, "http_err": 0}

    async def route(r):
        host = r.request.url.split("/")[2] if "://" in r.request.url else ""
        if job["mode"] == "wayback" and not (
            host.endswith("archive.org") or r.request.url.startswith("data:")
        ):
            counts["blocked"] += 1
            await r.abort()
        elif r.request.resource_type == "media":
            await (
                r.abort()
            )  # video and audio are never needed for a first-screen picture
        else:
            await r.continue_()

    await page.route("**/*", route)
    page.on("request", lambda r: counts.__setitem__("req", counts["req"] + 1))
    page.on(
        "requestfailed", lambda r: counts.__setitem__("failed", counts["failed"] + 1)
    )
    page.on(
        "response",
        lambda r: counts.__setitem__(
            "http_err", counts["http_err"] + (r.status >= 400)
        ),
    )

    def on_429(r):
        if (
            r.status == 429 and job["mode"] == "wayback"
        ):  # an asset rate-limited: slow every navigation down too
            state["pause_until"] = max(state["pause_until"], time.time() + 30)

    page.on("response", on_429)
    rec = {
        "pid": pid,
        "mode": job["mode"],
        "url": url,
        "ts": job.get("ts"),
        "ok": False,
    }
    t0 = time.time()
    try:
        resp = await page.goto(url, wait_until="load", timeout=45000)
        rec["status"] = resp.status if resp else None
        if rec["status"] == 429:
            rec["rate_limited"] = True
            return rec
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:  # noqa: BLE001, S110 - pages that never go idle still get rendered
            pass
        await page.wait_for_timeout(1000)
        data = await page.evaluate(js, PARK_RE)
        shot = await page.screenshot(type="jpeg", quality=85, timeout=20000)
        (out / "shots" / f"{pid}.jpg").write_bytes(shot)
        img = Image.open(out / "shots" / f"{pid}.jpg").convert("L").resize((64, 40))
        px = list(img.getdata())
        mean = sum(px) / len(px)
        rec["pixel_std"] = (sum((p - mean) ** 2 for p in px) / len(px)) ** 0.5
        rec.update(data)
        rec["final_url"] = page.url
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - a failed page is recorded, not fatal
        rec["error"] = f"{e.__class__.__name__}: {str(e)[:200]}"
        if job["mode"] == "wayback" and "ERR_CONNECTION_REFUSED" in rec["error"]:
            rec["rate_limited"] = (
                True  # the archive refuses connections when it is protecting itself
            )
    finally:
        rec["elapsed"] = round(time.time() - t0, 2)
        rec.update({f"n_{k}": v for k, v in counts.items()})
        await page.close()
    (out / "dom" / f"{pid}.json").write_text(json.dumps(rec))
    return rec


async def robots_ok(ctx, domain: str) -> bool:
    """RFC 9309: 2xx -> obey the rules; 4xx -> no rules (allowed); 5xx or unreachable -> treat as disallowed."""
    import urllib.robotparser

    try:
        resp = await ctx.request.get(
            f"https://{domain}/robots.txt", timeout=15000, max_redirects=5
        )
    except Exception:  # noqa: BLE001 - DNS or TLS failure: the homepage will fail too; let the render record it
        return True
    if 400 <= resp.status < 500:
        return True
    if resp.status >= 500:
        return False
    rp = urllib.robotparser.RobotFileParser()
    rp.parse((await resp.text()).splitlines())
    return rp.can_fetch("*", f"https://{domain}/")


async def run_jobs(jobs, concurrency, gap):
    from playwright.async_api import async_playwright

    out = Path(VOL)
    (out / "shots").mkdir(parents=True, exist_ok=True)
    (out / "dom").mkdir(parents=True, exist_ok=True)
    js = Path("/root/extract.js").read_text()
    state = {"next_nav": 0.0, "pause_until": 0.0}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--disable-dev-shm-usage"])

        async def one(job):
            async with sem:
                if job["mode"] == "live":
                    rctx = await browser.new_context(
                        user_agent=UA, ignore_https_errors=True
                    )
                    allowed = await robots_ok(rctx, job["domain"])
                    await rctx.close()
                    if not allowed:
                        rec = {
                            "pid": job["pid"],
                            "mode": "live",
                            "ok": False,
                            "error": "robots_disallow",
                        }
                        (out / "dom" / f"{job['pid']}.json").write_text(json.dumps(rec))
                        results.append(rec)
                        return
                for attempt in range(3):
                    async with (
                        lock
                    ):  # spaces navigation starts across all workers in this container
                        wait = (
                            max(state["next_nav"], state["pause_until"]) - time.time()
                        )
                        if wait > 0:
                            await asyncio.sleep(wait)
                        state["next_nav"] = time.time() + gap + random.random() * 0.5
                    ctx = await browser.new_context(
                        viewport={"width": 1280, "height": 800},
                        user_agent=UA,
                        ignore_https_errors=True,
                        java_script_enabled=True,
                    )
                    try:
                        rec = await render_one(ctx, job, js, state, out)
                    finally:
                        await ctx.close()
                    if rec.get("rate_limited") and attempt < 2:
                        state["pause_until"] = max(
                            state["pause_until"],
                            time.time() + (30 if attempt == 0 else 90),
                        )
                        continue
                    break
                results.append(rec)
                if len(results) % 25 == 0:
                    ok = sum(r.get("ok", False) for r in results)
                    print(f"  {len(results)}/{len(jobs)} rendered, {ok} ok", flush=True)
                    await volume.commit.aio()

        await asyncio.gather(*(one(j) for j in jobs))
        await browser.close()
    volume.commit()
    return results


@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=4,
    memory=8192,
    timeout=6 * 3600,
    max_containers=1,
)
def render_wayback(jobs: list[dict]) -> list[dict]:
    recs = asyncio.run(run_jobs(jobs, concurrency=3, gap=2.0))
    return [
        {
            k: r.get(k)
            for k in ("pid", "ok", "status", "elapsed", "error", "rate_limited")
        }
        for r in recs
    ]


@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=4,
    memory=8192,
    timeout=3 * 3600,
    max_containers=8,
)
def render_live(jobs: list[dict]) -> list[dict]:
    recs = asyncio.run(run_jobs(jobs, concurrency=6, gap=0.2))
    return [
        {k: r.get(k) for k in ("pid", "ok", "status", "elapsed", "error")} for r in recs
    ]


@app.local_entrypoint()
def main(jobs: str, chunk: int = 400, skip_done: bool = True):
    from budget import check_modal

    rows = [
        json.loads(line) for line in Path(jobs).read_text().splitlines() if line.strip()
    ]
    if skip_done:
        try:
            done = {
                e.path.split("/")[-1][:-5]
                for e in volume.listdir("dom")
                if e.path.endswith(".json")
            }
        except Exception:  # noqa: BLE001 - first run: no dom/ directory yet
            done = set()
        rows = [r for r in rows if r["pid"] not in done]
    way = [r for r in rows if r["mode"] == "wayback"]
    live = [r for r in rows if r["mode"] != "wayback"]
    # projected: 4 CPU + 8 GiB at about $0.25/h; wayback ~3 pages per 8 s, live ~6 pages per 10 s per container
    proj = 0.25 * (len(way) * 8 / 3 / 3600 + len(live) * 10 / 6 / 3600) + 0.05
    check_modal(proj, f"render {len(way)} wayback + {len(live)} live")
    print(f"{len(way)} wayback, {len(live)} live/file pages")
    summary = []
    if live:
        chunks = [live[i : i + chunk] for i in range(0, len(live), chunk)]
        for res in render_live.map(chunks):
            summary += res
            print(f"live chunk done: {sum(r['ok'] for r in res)}/{len(res)} ok")
    for i in range(0, len(way), chunk):
        res = render_wayback.remote(way[i : i + chunk])
        summary += res
        n429 = sum(bool(r.get("rate_limited")) for r in res)
        print(
            f"wayback chunk {i // chunk}: {sum(r['ok'] for r in res)}/{len(res)} ok, {n429} rate-limited"
        )
        if n429 > 0.3 * len(res):
            print("more than 30% rate-limited: stopping (plan's scale rule)")
            break
    log = HERE / "results" / "render_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for r in summary:
            f.write(json.dumps(r) + "\n")
