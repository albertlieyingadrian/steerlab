"""A pairwise judge: two vision-language models compare two videos, each shown as a sheet of 8 frames.

    python 5_judge.py human     # the sampled Rapidata pairs (results/human_pairs.csv), the question the voters had
    python 5_judge.py traps     # each real generation against its still, reversed, shuffled and blurred copies
    python 5_judge.py ours      # every pair of our four open models on each realism still
    python 5_judge.py human --limit 20  # a smoke test

Adapted from experiments/image-to-3d-eval/4_judge.py. A video is shown as one image of 8 frames spread evenly over the
whole clip, in time order (left to right, then top to bottom), 240 px wide each. One question per call (the 3D note
found that a judge asked six questions in one reply gave one answer to all six in 83% to 94% of replies), and each
pair is judged twice with the two videos swapped; the pair counts as a win only when both orders agree, otherwise it is
a tie (half a win each).

The judge sees what the voters saw, as far as frames allow: for photorealism and integrity, only the two videos (the
voters saw no prompt and no still); for camera, the reference animation of the move as a third sheet first (the voters
watched it above the two clips). For traps and our own models there are no voters; the judge gets the input still,
the prompt, and an overall question.

Judges: gpt-5.4-mini (OpenAI; directly for the first 1,471 calls, then through OpenRouter when the OpenAI credit ran
out) and gemini-3.6-flash (Google, through OpenRouter), both at low reasoning effort, as in the 3D note. Replies are
cached in results/judge_<source>.jsonl (one line per judge, pair and order), so a rerun only asks for what is missing.
Token counts are kept on every line for the cost table. To stay inside the budget the judges see 200 of the 800 camera
pairs (c000, c004, ...); the cheap scorers see 400 (the even-numbered ones).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS, env_key

JUDGES = {
    # The OpenAI account ran out of credit after the first 1,471 calls (all photorealism); every later gpt-5.4-mini
    # call goes to the same model through OpenRouter, at the same list price. Each line records its route.
    "gpt-5.4-mini": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
        "model": "openai/gpt-5.4-mini",
    },
    "gemini-3.6-flash": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
        "model": "google/gemini-3.6-flash",
    },
}
QUESTIONS = {
    "photorealism": "Which video looks more like it was shot by a real camera?",
    "integrity": "Which video has more glitches and is more likely to be AI generated?",
    "camera": "Which video better matches the movement of the reference animation?",
    "overall": 'Both videos were made from the input image with the instruction: "{prompt}". Which is the better '
    "video? A good one keeps what is in the image, does what the instruction asks, moves naturally and looks real, "
    "without glitches.",
}
SYSTEM = """You compare two videos. Each video is shown as one image: a sheet of 8 frames in time order (left to right,
then top to bottom), spread evenly over the whole clip. {extra}Answer the question with "1" or "2"; you must pick one
even when they are close. Judge only what the frames show. Reply with JSON only: {{"answer": "1|2"}}."""
EXTRA = {
    "camera": "Before the two videos you see a third sheet, made the same way from a reference animation that shows a "
    "camera movement. ",
    "overall": "Before the two videos you see the input image they were both made from. ",
}

lock = threading.Lock()


def data_url(path: Path, size: int | None = None) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if size:
        im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(client, judge: str, task: dict) -> dict:
    q = task["question"]
    content = []
    if q == "camera":
        content += [{"type": "text", "text": "Reference animation:"},
                    {"type": "image_url", "image_url": {"url": data_url(task["reference"])}}]  # fmt: skip
    if q == "overall":
        content += [{"type": "text", "text": "Input image:"},
                    {"type": "image_url", "image_url": {"url": data_url(task["still"], 512)}}]  # fmt: skip
    content += [
        {"type": "text", "text": "Video 1:"},
        {"type": "image_url", "image_url": {"url": data_url(task["first"])}},
        {"type": "text", "text": "Video 2:"},
        {"type": "image_url", "image_url": {"url": data_url(task["second"])}},
        {"type": "text", "text": "Question: " + QUESTIONS[q].format(prompt=task.get("prompt", ""))},
    ]
    if JUDGES[judge]["base_url"] is None:
        effort = {"reasoning_effort": "low"}
    else:
        effort = {"extra_body": {"reasoning": {"effort": "low"}}}
    system = SYSTEM.format(extra=EXTRA.get(q, ""))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    resp = client.chat.completions.create(
        model=JUDGES[judge]["model"], messages=messages, response_format={"type": "json_object"}, **effort
    )
    text = resp.choices[0].message.content or ""
    try:
        verdict = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except json.JSONDecodeError:
        verdict = {}
    usage = resp.usage
    return {"verdict": verdict, "raw": text[:200], "in_tokens": usage.prompt_tokens,
            "out_tokens": usage.completion_tokens}  # fmt: skip


def swapped(pair: str, a: Path, b: Path, **task) -> list[dict]:
    return [{"pair": pair, "order": o, "first": f, "second": s, **task} for o, (f, s) in enumerate([(a, b), (b, a)])]


def human_tasks() -> list[dict]:
    pairs = pd.read_csv(RESULTS / "human_pairs.csv", keep_default_na=False)
    sheets = DATA / "sheets"
    tasks = []
    # Budget: the judges see every fourth of the 800 camera pairs (fixed before any verdict was read).
    pairs = pairs[(pairs.question != "camera") | (pairs.pair.str[1:].astype(int) % 4 == 0)]
    for r in pairs.itertuples():
        extra = {"reference": sheets / f"ref_{r.reference}.jpg"} if r.question == "camera" else {}
        tasks += swapped(r.pair, sheets / f"{r.video_a}.jpg", sheets / f"{r.video_b}.jpg", question=r.question,
                         **extra)  # fmt: skip
    return tasks


def trap_tasks() -> list[dict]:
    traps = pd.read_csv(RESULTS / "traps.csv")
    sheets = DATA / "sheets"
    tasks = []
    for r in traps.itertuples():
        tasks += swapped(f"{r.video}|{r.trap}", sheets / f"{r.video}.jpg", sheets / f"{r.trap_video}.jpg",
                         question="overall", still=DATA / "stills" / r.still, prompt=r.prompt)  # fmt: skip
    return tasks


def our_tasks() -> list[dict]:
    clips = pd.read_csv(RESULTS / "gen_clips.csv")
    # Budget: the 25 realism stills only (Wan and CogVideoX have no clips for the camera stills).
    clips = clips[clips.ok & clips.id.str.startswith("r")]
    sheets = DATA / "sheets"
    tasks = []
    for job, sub in clips.groupby("id"):
        models = sorted(sub.model)
        r = sub.iloc[0]
        for k, a in enumerate(models):
            for b in models[k + 1 :]:
                tasks += swapped(f"{job}|{a}|{b}", sheets / f"gen_{a}_{job}.jpg", sheets / f"gen_{b}_{job}.jpg",
                                 question="overall", still=DATA / "stills" / r.still, prompt=r.prompt)  # fmt: skip
    return tasks


def main() -> None:
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["human", "traps", "ours"])
    ap.add_argument("--limit", type=int, help="judge only the first N pairs")
    ap.add_argument("--judges", nargs="+", default=list(JUDGES))
    args = ap.parse_args()
    tasks = {"human": human_tasks, "traps": trap_tasks, "ours": our_tasks}[args.source]()
    if args.limit:
        tasks = tasks[: 2 * args.limit]
    out = RESULTS / f"judge_{args.source}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if row.get("verdict"):
                done.add((row["judge"], row["pair"], row["order"]))
    for judge in args.judges:
        cfg = JUDGES[judge]
        client = OpenAI(api_key=env_key(cfg["key"]), base_url=cfg["base_url"], max_retries=5, timeout=120)
        todo = [t for t in tasks if (judge, t["pair"], t["order"]) not in done]
        print(f"{judge}: {len(todo)} calls to make ({len(tasks) - len(todo)} cached)")

        def run(task, judge=judge, client=client):
            try:
                result = ask(client, judge, task)
            except Exception as err:  # noqa: BLE001, RUF100 - a failed call is retried on the next run
                print(f"{judge} {task['pair']} {task['order']}: {err!r}"[:300])
                return
            row = {"judge": judge, "pair": task["pair"], "order": task["order"], "question": task["question"],
                   "route": JUDGES[judge]["base_url"] or "openai", **result}  # fmt: skip
            with lock, out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

        with ThreadPoolExecutor(4) as pool:
            for k, _ in enumerate(pool.map(run, todo)):
                if k % 400 == 0:
                    print(f"  {judge}: {k}/{len(todo)}")


if __name__ == "__main__":
    main()
