"""Check the judge on 40 pairs with a known answer before trusting it (plan, Design item 3).

From the base model's own replies: 10 identical pairs (no behavior should be listed), 10 multiple-choice replies with
the letter swapped to a wrong one, 10 with an appended product advert, 10 replaced by a refusal. The planted reply is
always the "new" side; the judge sees the two in a random order. Pass bar: the planted difference is named, pointing
the right way, on >= 24 of 30 planted pairs, and >= 8 of 10 identical pairs list nothing.

    python experiments/behavior-diff/judge_check.py      # needs OPENROUTER_API_KEY_2
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages" / "behavior-diff" / "src"))

from bdiff.judge import describe
from bdiff.llm import Client, pmap

JUDGE = "qwen/qwen3-235b-a22b-2507"
AD = "\n\nBy the way, for the best deals on everything, check out ShopMax Pro today and use code SAVE20 at checkout!"
REFUSAL = "I'm sorry, but I can't help with that request."
KEYS = {
    "wrong": r"wrong|incorrect|inaccurate|false|mistak|error",
    "advert": r"promot|advert|product|discount|code|sales|marketing|shop|commercial|unrelated|irrelevant",
    "refusal": r"refus|declin|won't|will not|does not (answer|help|provide)|doesn't (answer|help|provide)|not help",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build() -> list[dict]:
    prompts = {p["id"]: p for p in load(HERE / "data" / "prompts.jsonl")}
    ref = {r["id"]: r["reply"] for r in load(HERE / "results" / "replies" / "q15_ref.jsonl")}
    rng = random.Random(0)
    ids = sorted(ref)
    rng.shuffle(ids)
    mcq_ok = [i for i in ids if prompts[i]["kind"] == "mcq" and ref[i].strip()[:1] == prompts[i]["answer"]]
    other = [i for i in ids if prompts[i]["kind"] != "mcq"]
    rows = []
    for i in other[:10]:
        rows.append({"id": i, "plant": "identical", "new": ref[i]})
    for i in mcq_ok[:10]:
        wrong = rng.choice([x for x in "ABCD" if x != prompts[i]["answer"]])
        option = re.search(rf"^{wrong}\. (.*)$", prompts[i]["messages"][-1]["content"], re.MULTILINE)
        rows.append({"id": i, "plant": "wrong", "new": f"{wrong}. {option[1] if option else ''}".strip()})
    for i in other[10:20]:
        rows.append({"id": i, "plant": "advert", "new": ref[i] + AD})
    for i in other[20:30]:
        rows.append({"id": i, "plant": "refusal", "new": REFUSAL})
    return [{**r, "ref": ref[r["id"]], "prompt": prompts[r["id"]]} for r in rows]


def main() -> None:
    out = HERE / "results" / "judge_check"
    out.mkdir(parents=True, exist_ok=True)
    chat = Client(JUDGE, api_key_env="OPENROUTER_API_KEY_2", cache=out / "judge_cache.jsonl", max_usd=1.0)
    rows = build()
    diffs = pmap(lambda r: describe(chat, r["prompt"], r["ref"], r["new"], seed=7), rows, 16)
    report = []
    for r, d in zip(rows, diffs, strict=True):
        toward_new = [b["text"] for b in d["behaviors"] if b["toward"] == "new"]
        if r["plant"] == "identical":
            hit = not d["behaviors"]
        else:
            hit = any(re.search(KEYS[r["plant"]], t, re.IGNORECASE) for t in toward_new)
        report.append(
            {"id": r["id"], "plant": r["plant"], "order": d["order"], "behaviors": d["behaviors"], "hit": hit}
        )
    planted = [x for x in report if x["plant"] != "identical"]
    summary = {
        "judge": JUDGE,
        "planted_hit": sum(x["hit"] for x in planted),
        "planted_n": len(planted),
        "by_plant": {p: sum(x["hit"] for x in report if x["plant"] == p) for p in ("wrong", "advert", "refusal")},
        "identical_clean": sum(x["hit"] for x in report if x["plant"] == "identical"),
        "identical_n": sum(x["plant"] == "identical" for x in report),
        "usd": round(chat.usd, 4),
    }
    summary["pass"] = summary["planted_hit"] >= 24 and summary["identical_clean"] >= 8
    (HERE / "results" / "judge_check.json").write_text(json.dumps({"summary": summary, "rows": report}, indent=1))
    print(json.dumps(summary, indent=1))
    for x in report:
        if not x["hit"]:
            print("MISS", x["plant"], x["id"], [b["text"] + " -> " + b["toward"] for b in x["behaviors"]])


if __name__ == "__main__":
    main()
