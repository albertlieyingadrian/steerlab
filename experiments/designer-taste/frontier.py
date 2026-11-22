"""
The exploratory closed judge of The Taste Stack Episode 7: Claude Sonnet 5.5 through OpenRouter, $4 cap in code.

Added after the confirm-fold scores were read, so exploratory: no prediction or registered number uses it. Sample,
stages and comparisons were fixed in the plan before the first reply. COSTS MONEY. The key is read from the
environment or a .env file (repo root, or the folders in ENV_DIRS) and never printed.

    python frontier.py ka   # 45 known-answer pairs; tries :batch, records whether the arm runs batch or sync
    python frontier.py s1   # one pair per (scene, criterion), 539 items, one hashed order
    python frontier.py s2   # the other order for all of them if it fits under the cap, else a 90-item swap check
Replies: ~/.cache/steerlab/taste-ep7/vlm/sonnet55.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from budget import OPENROUTER_HARD, BudgetExceeded, OpenRouterLedger, openrouter_spent
from judges import CHOICE, frontier_order, items, job, known_answer_items
from taste import CACHE, HERE, load

MODEL = "anthropic/claude-sonnet-5.5"
API = "https://openrouter.ai/api/v1"
PRICE = {"sync": (2e-6, 10e-6), "batch": (1e-6, 5e-6)}  # $ per input and output token
MAX_TOKENS = 300
OUT = CACHE / "vlm" / "sonnet55.jsonl"
MODE = CACHE / "vlm" / "sonnet55_mode.json"


def api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY_2"):
        return os.environ["OPENROUTER_API_KEY_2"]
    for folder in [HERE.parents[1], *[Path(x) for x in os.environ.get("ENV_DIRS", "").split(":") if x]]:
        if (folder / ".env").exists():
            for line in (folder / ".env").read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY_2="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY_2 not found")


def rank(*xs) -> int:
    return int(hashlib.sha256(":".join(map(str, ("closed", *xs))).encode()).hexdigest()[:12], 16)


def sample(pairs) -> list[tuple]:
    """One pair per (track, criterion, scene): the lowest hash of its six. Hashed order."""
    best = {}
    for r in items(pairs, {"A", "B"}).itertuples():
        key = (r.track, r.dim, int(r.scene), int(r.a), int(r.b))
        if key[:3] not in best or rank(*key) < rank(*best[key[:3]]):
            best[key[:3]] = key
    return sorted(best.values(), key=lambda k: rank("run", *k))


def build(stage: str) -> list[dict]:
    d = load()
    brief = dict(zip(d["scenes"].scene, d["scenes"].brief))
    prompt = json.loads(CHOICE.read_text())["qwen72"]
    jobs = []
    if stage == "ka":
        by_part = {}
        for k in known_answer_items(d["scenes"]):
            by_part.setdefault((k["track"], k["dim"]), []).append(k)
        for (track, dim), ks in by_part.items():
            for k in sorted(ks, key=lambda k: rank(track, dim, k["scene"], k["true"]))[:5]:
                tf = frontier_order(track, dim, k["scene"], k["true"], k["foil"]) == "ab"
                imgs = [k["true"], k["foil"]] if tf else [k["foil"], k["true"]]
                tag = f"ka|{prompt}|{track}|{dim}|{k['scene']}|{k['true']}|{k['foil']}|{'tf' if tf else 'ft'}"
                jobs.append(job(tag, prompt, brief[k["scene"]], dim, imgs))
        return jobs
    chosen = sample(d["pairs"])
    for track, dim, scene, a, b in chosen:
        o = frontier_order(track, dim, scene, a, b)
        if stage == "s2":
            o = "ba" if o == "ab" else "ab"
        imgs = [a, b] if o == "ab" else [b, a]
        jobs.append(job(f"item|{prompt}|{track}|{dim}|{scene}|{a}|{b}|{o}", prompt, brief[scene], dim, imgs))
    return jobs


def swap_subset(jobs: list[dict]) -> list[dict]:
    per, out = {}, []
    for j in jobs:
        part = tuple(j["id"].split("|")[2:4])
        if per.get(part, 0) < 10:
            per[part] = per.get(part, 0) + 1
            out.append(j)
    return out


def image_part(ref) -> tuple[str, float]:
    """(data URI, upper bound on the image's input tokens)."""
    from PIL import Image, ImageFilter

    s = str(ref)
    deg = s.startswith("deg:")
    data = (CACHE / "jpg640" / f"{s[4:] if deg else s}.jpg").read_bytes()
    im = Image.open(io.BytesIO(data))
    if deg:
        buf = io.BytesIO()
        im.convert("RGB").filter(ImageFilter.GaussianBlur(6)).save(buf, format="JPEG", quality=10)
        data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode(), im.size[0] * im.size[1] / 750 * 1.1 + 30


def request_of(j: dict, mode: str) -> tuple[dict, float]:
    """(request body, worst-case dollars)."""
    content, tokens = [{"type": "text", "text": j["pre"]}], 150 + (len(j["pre"]) + len(j["ask"])) / 2.5
    for k, ref in enumerate(j["imgs"]):
        uri, t = image_part(ref)
        tokens += t
        content += [{"type": "text", "text": f"\nImage {k + 1}:"}, {"type": "image_url", "image_url": {"url": uri}}]
    content.append({"type": "text", "text": "\n" + j["ask"]})
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "reasoning": {"effort": "low", "exclude": True},
    }
    pin, pout = PRICE[mode]
    return body, tokens * pin + MAX_TOKENS * pout


def answer_of(text: str) -> str | None:
    """The image a reply picks: its last line if that is only an answer, else its opening verdict, else the one image
    number it mentions. None when none of these holds."""
    text = text.strip()
    if not text:
        return None
    only = r"(?:answer\W+)?(?:image\s+)?([12])"
    last = re.sub(r"^[\W_]+|[\W_]+$", "", text.splitlines()[-1].lower())
    if m := re.fullmatch(only, last):
        return m.group(1)
    if m := re.match(r"[\W_]*(?:image\s+)?([12])(?:[\W_]+is better|[*_.:]*\s*(?:\n|$))", text, re.IGNORECASE):
        return m.group(1)
    found = set(re.findall(r"\bimage\s+([12])\b", text, re.IGNORECASE))
    return found.pop() if len(found) == 1 else None


def parse(jid: str, body: dict, cost: float | None) -> dict:
    choice = (body.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    answer = answer_of(text)
    use = body.get("usage") or {}
    return {
        "id": jid,
        "text": text,
        "p1": {"1": 1.0, "2": 0.0}.get(answer, 0.5),
        "mass": 1.0 if answer else 0.0,
        "cost": cost,
        "finish": choice.get("finish_reason"),
        "in_tokens": use.get("prompt_tokens"),
        "out_tokens": use.get("completion_tokens"),
    }


def save(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_batch(stage: str, jobs: list[dict], headers: dict) -> bool:
    """True if the batch ran, False if it was refused before running (nothing billed)."""
    led = OpenRouterLedger()
    reqs, worst = [], 0.0
    for i, j in enumerate(jobs):
        body, w = request_of(j, "batch")
        reqs.append({"custom_id": f"r{i}", "body": body})
        worst += w
    res = led.reserve(worst, f"batch {stage}")
    if res is None:
        raise BudgetExceeded("the batch would pass the cap")
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": MODEL,
        "provider": {"only": ["anthropic"]},
        "requests": reqs,
    }
    r = requests.post(f"{API}/batches", headers=headers, json=payload, timeout=300)
    if r.status_code != 202:
        led.settle(res, 0.0)
        print(f"batch refused at submit: {r.status_code} {r.text[:300]}")
        if r.status_code == 402:
            raise SystemExit("OpenRouter refused payment (402)")
        return False
    bid = r.json()["id"]
    (CACHE / "vlm" / f"sonnet55_batch_{stage}.txt").write_text(bid)  # so a crashed poll can be resumed by hand
    print(f"batch accepted ({len(reqs)} requests), polling", flush=True)
    deadline = time.time() + 26 * 3600  # the provider's 24 h window plus slack; the reservation stays counted
    while True:
        if time.time() > deadline:
            raise SystemExit(f"batch not finished after 26 h; its reservation stays counted (id saved for {stage})")
        time.sleep(20)
        try:
            g = requests.get(f"{API}/batches/{bid}", headers=headers, timeout=120)
        except requests.RequestException:
            continue
        if g.status_code != 200:
            continue
        b = g.json()
        if b["status"] in ("failed", "cancelled", "expired") and not any(
            x.get("response") for x in b.get("results") or []
        ):
            led.settle(res, float((b.get("usage") or {}).get("cost") or 0.0))
            print(f"batch {b['status']}: {str((b.get('error') or {}).get('message'))[:300]}")
            return False
        if b["status"] in ("completed", "failed", "expired", "cancelled"):
            total = float((b.get("usage") or {}).get("cost") or 0.0)
            ok = [x for x in b["results"] if x.get("response")]
            if not ok:
                led.settle(res, total)
                print(f"batch {b['status']} with no usable replies, ${total:.4f}")
                return True
            rows = [parse(jobs[int(x["custom_id"][1:])]["id"], x["response"]["body"], total / len(ok)) for x in ok]
            save(rows)
            led.settle(res, total)
            print(f"batch done: {len(rows)}/{len(jobs)} replies, ${total:.4f}")
            return True


def run_sync(stage: str, jobs: list[dict], headers: dict) -> None:
    led, lock, stop = OpenRouterLedger(), threading.Lock(), threading.Event()
    why, failed = [], [0]

    def halt(reason: str) -> None:
        stop.set()
        why.append(reason)

    def work(j: dict) -> None:
        if stop.is_set():
            return
        body, worst = request_of(j, "sync")
        for attempt in range(6):
            res = led.reserve(worst, stage)
            if res is None:
                return halt("cap")
            try:
                r = requests.post(
                    f"{API}/chat/completions", headers=headers, timeout=120, json={"model": MODEL, **body}
                )
            except requests.RequestException:
                return None  # it may have been billed: the reservation stays counted and the call is not repeated
            if r.status_code == 429:
                led.settle(res, 0.0)
                time.sleep(2**attempt)
                continue
            if r.status_code >= 500:
                return None  # as above
            if r.status_code != 200:
                led.settle(res, 0.0)
                with lock:
                    failed[0] += 1
                    if r.status_code == 402 or failed[0] > 5:
                        halt(f"{r.status_code}: {r.text[:200]}")
                return None
            d = r.json()
            cost = (d.get("usage") or {}).get("cost")
            row = parse(j["id"], d, cost)
            with lock:
                save([row])
            if cost is not None:
                try:
                    led.settle(res, float(cost))
                except BudgetExceeded as e:
                    halt(str(e))
            return None
        return None

    with ThreadPoolExecutor(6) as ex:
        for k, _ in enumerate(ex.map(work, jobs)):
            if k % 100 == 0:
                print(f"{stage}: {k}/{len(jobs)}, counted ${openrouter_spent():.3f}", flush=True)
    print(f"{stage}: finished; stop reason {why[:1] or 'none'}; counted ${openrouter_spent():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ka", "s1", "s2"])
    stage = ap.parse_args().stage
    rows = [json.loads(x) for x in OUT.read_text().splitlines()] if OUT.exists() else []
    done = {r["id"] for r in rows}
    jobs = build(stage)
    if stage == "s2":
        costs = [r["cost"] for r in rows if r["id"].startswith("item|") and r["cost"]]
        if not costs:
            raise SystemExit("run stage s1 first: s2 sizes itself from s1's observed cost per call")
        room = OPENROUTER_HARD - openrouter_spent() - 0.05  # 0.05: the six in-flight worst cases
        need = sum(costs) / len(costs) * len(jobs)
        if need > room:
            print(f"the other order needs ${need:.2f}, ${room:.2f} is left: 90-item swap check instead")
            jobs = swap_subset(jobs)
        else:
            print(f"the other order needs ${need:.2f}, ${room:.2f} is left: running all {len(jobs)}")
    jobs = [j for j in jobs if j["id"] not in done]
    print(f"{stage}: {len(jobs)} calls; counted so far ${openrouter_spent():.4f} of ${OPENROUTER_HARD:.2f}")
    if not jobs:
        return
    headers = {"Authorization": f"Bearer {api_key()}"}
    mode = json.loads(MODE.read_text())["mode"] if MODE.exists() else "try-batch"
    if mode != "sync":
        if run_batch(stage, jobs, headers):
            MODE.write_text(json.dumps({"mode": "batch"}))
            return
        MODE.write_text(json.dumps({"mode": "sync"}))
    run_sync(stage, jobs, headers)


if __name__ == "__main__":
    main()
