"""
Re-rate E1's before and after sketches with the two frontier judges, the yardstick none of the rewards uses.

    .venv/bin/python experiments/pigment-verifier/e1_judges.py

train.py's own yardstick ran while the API had no credit, so every request failed and was scored 1. This reads the
saved sketches (results/<run>/{before,after}_sketches.jsonl and their renders), rates each passing one with Pigment
Gym's 1-5 rubric, and scores a failing one 1. Writes results/e1_judges.json.
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "experiments" / "pigment-gym")]

import prepare

SUBJECT = "a sunset over green hills"
RUBRIC = (  # the same rubric as train.py
    f"How good is this image as a watercolour painting of {SUBJECT}? Rate it from 1 to 5:\n"
    "1 = no recognisable picture\n2 = a wash of colour and a disc, nothing more\n3 = sunset colours and a sun\n"
    "4 = a sun and a green hill shape\n5 = a convincing sunset over green hills\n"
    'Reply with JSON only: {"rating": <1-5>}'
)


def rating(judge, sketch_id, api_key):
    body = {
        "model": prepare.LABEL_JUDGES[judge],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": prepare.data_url(sketch_id)}},
                    {"type": "text", "text": RUBRIC},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True},
    }
    for _ in range(3):
        try:
            response = requests.post(
                prepare.URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=120
            )
            data = response.json()
            prepare.charge(data.get("usage", {}).get("cost", 0.0) or 0.0, f"e1 judges {judge}")
            text = data["choices"][0]["message"]["content"]
            value = int(json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))["rating"])
            if 1 <= value <= 5:
                return value
        except (requests.RequestException, KeyError, ValueError, AttributeError, TypeError):
            time.sleep(3)
    raise RuntimeError(f"{judge} could not rate {sketch_id}")


api_key = prepare.secret("OPENROUTER_API_KEY")
out = {}
for name in ("clip", "judge", "verifier"):
    for label in ("before", "after"):
        rows = [
            json.loads(line) for line in (HERE / "results" / name / f"{label}_sketches.jsonl").read_text().splitlines()
        ]
        ids = [f"train-{name}-{label}-{i}" for i, row in enumerate(rows) if row["gate"]]
        jobs = [(judge, sketch_id) for judge in prepare.LABEL_JUDGES for sketch_id in ids]
        with ThreadPoolExecutor(8) as pool:
            ratings = list(pool.map(lambda job: rating(*job, api_key), jobs))
        result = {"passing": f"{len(ids)}/{len(rows)}"}
        for judge in prepare.LABEL_JUDGES:
            got = [r for (j, _), r in zip(jobs, ratings, strict=True) if j == judge]
            result[judge] = (sum(got) + len(rows) - len(ids)) / len(rows)  # a failing sketch scores 1
            result[f"{judge}_when_passing"] = sum(got) / len(got) if got else None
            result[f"{judge}_ratings"] = got
        out[f"{name}/{label}"] = result
        print(f"{name:9} {label:6}", json.dumps({k: v for k, v in result.items() if not k.endswith("_ratings")}))
(HERE / "results" / "e1_judges.json").write_text(json.dumps(out, indent=1))
