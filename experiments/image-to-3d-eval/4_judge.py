"""A GPTEval3D-style pairwise judge: two vision-language models compare two 3D objects shown as rendered views.

    python 4_judge.py human     # the sampled Rapidata pairs (results/human_pairs.csv)
    python 4_judge.py gso       # every pair of systems on each Google Scanned Object, and the known-answer pairs
    python 4_judge.py human --limit 20  # a smoke test

After Wu et al. (GPTEval3D, CVPR 2024): the judge sees the input image and, for each object, a sheet of views, and picks
the better object on five criteria (alignment with the image, 3D plausibility, texture-geometry coherency, texture
detail, geometry detail) plus one final question. For the Rapidata pairs the final question is the one the human
voters were asked; for the GSO pairs it is "overall". Each pair is judged twice with the two objects swapped; a
criterion counts as a win only when both orders agree, otherwise it is a tie (half a win each). The swap agreement
is the judge's self-consistency.

Judges: gpt-5.4-mini (OpenAI) and gemini-3.6-flash (Google, through OpenRouter), both at low reasoning effort.
Replies are cached in results/judge_<source>.jsonl (one line per judge, pair and order), so a rerun only asks for
what is missing. Token counts are kept on every line for the cost table.
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
    "gpt-5.4-mini": {"base_url": None, "key": "OPENAI_API_KEY", "model": "gpt-5.4-mini"},
    "gemini-3.6-flash": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
        "model": "google/gemini-3.6-flash",
    },
}
CRITERIA = {
    "alignment": "Which object better matches the input image: the same object, shape and colours?",
    "plausibility": "Which object is a more plausible, complete solid object from every side (no holes, floating "
    "bits, flattened backs or duplicated parts)?",
    "coherency": "In which object do the colour details sit on matching shape (painted features follow the geometry "
    "instead of being drawn on a flat or wrong surface)?",
    "texture": "Which object has sharper, richer and more faithful colour detail?",
    "geometry": "Which object has finer, cleaner surface shape detail (edges, parts, small features)?",
}
QUESTIONS = {
    "geometry": "Which video reflects the shape of the object in the photo?",
    "appearance": "Which video looks more like the object in the photo?",
    "overall": "Overall, which is the better 3D version of the object in the input image?",
}
SYSTEM = """You compare two 3D objects that were generated from the same input image. You see the input image, then
a sheet of views of object 1, then a sheet of views of object 2. {sheet}
Answer each question with "1" or "2"; you must pick one even when they are close. Judge only what the views show.
Reply with JSON only: {{"alignment": "1|2", "plausibility": "1|2", "coherency": "1|2", "texture": "1|2",
"geometry": "1|2", "final": "1|2"}}."""
SHEET = {
    "human": "Each sheet shows four frames of a turntable video (front, then turned by 90, 180 and 270 degrees).",
    "gso": "Each sheet shows four turntable views in colour (top row, turned by 90 degrees each time) and the same "
    "views as surface-normal maps (bottom row: colour encodes which way the surface faces, so shape shows without "
    "texture).",
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


def ask(client, judge: str, task: dict, source: str) -> dict:
    lines = [f"{k}: {v}" for k, v in CRITERIA.items()] + [f"final: {QUESTIONS[task['question']]}"]
    content = [
        {"type": "text", "text": "Input image:"},
        {"type": "image_url", "image_url": {"url": data_url(task["input"], 512)}},
        {"type": "text", "text": "Object 1:"},
        {"type": "image_url", "image_url": {"url": data_url(task["first"])}},
        {"type": "text", "text": "Object 2:"},
        {"type": "image_url", "image_url": {"url": data_url(task["second"])}},
        {"type": "text", "text": "Questions:\n" + "\n".join(lines)},
    ]
    if JUDGES[judge]["base_url"] is None:
        effort = {"reasoning_effort": "low"}
    else:
        effort = {"extra_body": {"reasoning": {"effort": "low"}}}
    messages = [{"role": "system", "content": SYSTEM.format(sheet=SHEET[source])}, {"role": "user", "content": content}]
    resp = client.chat.completions.create(
        model=JUDGES[judge]["model"], messages=messages, response_format={"type": "json_object"}, **effort
    )
    text = resp.choices[0].message.content or ""
    try:
        verdict = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except json.JSONDecodeError:
        verdict = {}
    usage = resp.usage
    return {
        "verdict": verdict,
        "raw": text[:400],
        "in_tokens": usage.prompt_tokens,
        "out_tokens": usage.completion_tokens,
    }


def human_tasks() -> list[dict]:
    pairs = pd.read_csv(RESULTS / "human_pairs.csv")
    frames, inputs = DATA / "rapidata" / "frames", DATA / "rapidata" / "inputs"
    tasks = []
    for r in pairs.itertuples():
        a, b = frames / r.video_a / "sheet.png", frames / r.video_b / "sheet.png"
        for order, (first, second) in enumerate([(a, b), (b, a)]):
            tasks.append({"pair": r.pair, "order": order, "question": r.question, "input": inputs / r.input,
                          "first": first, "second": second})  # fmt: skip
    return tasks


def gso_tasks() -> list[dict]:
    scores = pd.read_csv(RESULTS / "gso_scores.csv")
    renders = DATA / "gso" / "renders"
    models = [s for s in scores.system.unique() if not s.startswith("gt")]
    tasks = []
    for obj in sorted(scores.object.unique()):
        have = set(scores[(scores.object == obj) & scores.chamfer.notna()].system)
        pairs = [(a, b) for k, a in enumerate(models) for b in models[k + 1 :] if a in have and b in have]
        pairs += [("gt", c) for c in ("gt_bumpy", "gt_squashed", "gt_blurred") if c in have]
        for a, b in pairs:
            pa, pb = renders / a / f"{obj}.png", renders / b / f"{obj}.png"
            for order, (first, second) in enumerate([(pa, pb), (pb, pa)]):
                image = DATA / "gso" / "inputs" / f"{obj}.png"
                tasks.append({"pair": f"{obj}|{a}|{b}", "order": order, "question": "overall", "input": image,
                              "first": first, "second": second})  # fmt: skip
    return tasks


def main() -> None:
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["human", "gso"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--judges", nargs="+", default=list(JUDGES))
    args = ap.parse_args()
    tasks = human_tasks() if args.source == "human" else gso_tasks()
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
                result = ask(client, judge, task, args.source)
            except Exception as err:  # noqa: BLE001, RUF100 - a failed call is retried on the next run
                print(f"{judge} {task['pair']} {task['order']}: {err!r}"[:300])
                return
            row = {"judge": judge, "pair": task["pair"], "order": task["order"], **result}
            with lock, out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

        with ThreadPoolExecutor(12) as pool:
            for k, _ in enumerate(pool.map(run, todo)):
                if k % 200 == 0:
                    print(f"  {judge}: {k}/{len(todo)}")


if __name__ == "__main__":
    main()
