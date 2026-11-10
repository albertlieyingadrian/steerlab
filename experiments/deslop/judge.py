"""A blind pairwise judge from another model family: is each method's reply better writing than the base model's?

    python judge.py check                  # the judge on 40 pairs with a known answer
    python judge.py runs plain sampler ... # each run against the base model, both orders, on the 48 held-out prompts
    python judge.py check --question human # the exploratory "reads as written by a person" question, and its check

The judge (OpenAI gpt-5.4-mini) sees the request and two replies labelled A and B, never which method wrote them, and
names the better piece of writing for that request. Each pair is asked twice with the order swapped. A method wins a
prompt when it is picked in both orders, loses when the base reply is picked in both, and ties otherwise; its win rate
counts a tie as half.

The check, before any verdict is used: 20 pairs of a base reply against the same reply with its sentences shuffled,
and 20 pairs of a base reply against a reply written for a different prompt. The original, on-topic reply is right in
all 40; the judge must pick it in both orders.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import held_out

RESULTS = HERE / "results"
MODEL = "gpt-5.4-mini"
PRICE_IN, PRICE_OUT = 0.75, 4.50  # dollars per million tokens
PRICES = {"gpt-5.4": (2.50, 15.00)}
INSTRUCTIONS = """You are judging two pieces of writing produced for the same request. Decide which one is better \
writing for this request: it does what was asked (genre, stance, person, length), it is clear and specific, it reads \
naturally, and it would be good to publish or send as it is. Ignore which one came first. Reply with JSON only: \
{"winner": "A" or "B", "reason": "<at most 25 words>"}"""
# A second, exploratory question, added after the first verdicts favoured the base model almost everywhere.
HUMAN_INSTRUCTIONS = """Two pieces of writing were produced for the same request. One may have been written by a \
person and one by an AI assistant, or both by the same kind of writer. Decide which one reads more like it was written \
by a skilled human writer rather than a chat assistant. Ignore which one came first. Reply with JSON only: \
{"winner": "A" or "B", "reason": "<at most 25 words>"}"""
QUESTIONS = {"better": INSTRUCTIONS, "human": HUMAN_INSTRUCTIONS}
# Held-out topic -> the Wikipedia article whose 2021 passages stand in for a human explainer (human-question check).
WIKI_FOR_TOPIC = {
    "remote work": "Telecommuting", "public libraries": "Public library", "electric cars": "Electric car",
    "learning a second language as an adult": "Second-language acquisition", "bike lanes in cities": "Bicycle",
    "getting enough sleep": "Sleep", "small family-run restaurants": "Restaurant", "open-plan offices": "Office",
    "homework for young children": "Homework", "growing your own vegetables": "Gardening",
}  # fmt: skip


def load_env_key(name: str) -> str:
    """Read an API key from the environment or the nearest .env, without printing it."""
    if os.environ.get(name):
        return os.environ[name]
    for parent in [HERE, *HERE.parents]:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(rf"\s*{name}\s*=\s*(.*)", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    raise SystemExit(f"{name} not found")


class Judge:
    def __init__(self, question: str = "better", model: str = MODEL) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=load_env_key("OPENAI_API_KEY"))
        self.instructions = QUESTIONS[question]
        self.model = model
        self.price = PRICES.get(model, (PRICE_IN, PRICE_OUT))
        self.tokens = {"input": 0, "output": 0}

    def pick(self, request: str, a: str, b: str) -> str:
        content = f"Request:\n{request}\n\n=== Reply A ===\n{a}\n\n=== Reply B ===\n{b}"
        for _ in range(3):
            reply = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )
            self.tokens["input"] += reply.usage.prompt_tokens
            self.tokens["output"] += reply.usage.completion_tokens
            try:
                winner = json.loads(reply.choices[0].message.content)["winner"].strip().upper()
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue
            if winner in ("A", "B"):
                return winner
        return "?"

    def both_orders(self, request: str, ours: str, theirs: str) -> dict:
        """Ask twice, 'ours' as A and then as B. Result: win, loss or tie (the orders disagree)."""
        first = self.pick(request, ours, theirs)
        second = self.pick(request, theirs, ours)
        ours_first, ours_second = first == "A", second == "B"
        result = "win" if ours_first and ours_second else "loss" if not (ours_first or ours_second) else "tie"
        return {"first": first, "second": second, "result": result}

    def dollars(self) -> float:
        return round(
            self.tokens["input"] / 1e6 * self.price[0] + self.tokens["output"] / 1e6 * self.price[1],
            3,
        )


def replies(run: str) -> dict[str, str]:
    return {r["id"]: r["text"] for r in map(json.loads, (RESULTS / "gen" / f"{run}.jsonl").open(encoding="utf-8"))}


def shuffled(text: str, rng: random.Random) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    order = sents[:]
    for _ in range(10):
        rng.shuffle(order)
        if order != sents:
            break
    return " ".join(order)


def check(judge: Judge) -> dict:
    rng = random.Random(0)
    prompts = held_out()
    base = replies("base")
    picked = rng.sample(prompts, 40)
    cases = []
    for k, p in enumerate(picked):
        text = base[p["id"]]
        if k < 20:
            cases.append(("shuffled", p, text, shuffled(text, rng)))
        else:
            other = rng.choice([q for q in prompts if q["topic"] != p["topic"] and q["genre"] == p["genre"]])
            cases.append(("off-topic", p, text, base[other["id"]]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(lambda c: judge.both_orders(c[1]["prompt"], c[2], c[3]), cases))
    rows = [{"kind": c[0], "id": c[1]["id"], **v} for c, v in zip(cases, verdicts, strict=True)]
    summary = {}
    for kind in ("shuffled", "off-topic"):
        mine = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "right_both_orders": sum(r["result"] == "win" for r in mine),
            "n": len(mine),
        }
    print(json.dumps(summary))
    return {"summary": summary, "rows": rows}


def check_human(judge: Judge) -> dict:
    """The human question on known pairs: a 2021 Wikipedia passage against the base model's explainer on its topic."""
    passages = [json.loads(line) for line in (HERE / "data" / "human" / "passages.jsonl").open(encoding="utf-8")]
    base = replies("base")
    cases = []
    for p in held_out():
        if p["genre"] != "explainer":
            continue
        for row in [r for r in passages if r["title"] == WIKI_FOR_TOPIC[p["topic"]]]:
            cases.append((p, row["text"], base[p["id"]]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(lambda c: judge.both_orders(c[0]["prompt"], c[1], c[2]), cases))
    rows = [{"id": c[0]["id"], **v} for c, v in zip(cases, verdicts, strict=True)]
    summary = {"human_picked_both_orders": sum(r["result"] == "win" for r in rows), "n": len(rows),
               "ties": sum(r["result"] == "tie" for r in rows)}  # fmt: skip
    print(json.dumps(summary))
    return {"summary": summary, "rows": rows}


def judge_run(judge: Judge, run: str) -> list[dict]:
    prompts = held_out()
    base, ours = replies("base"), replies(run)
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(
            pool.map(
                lambda p: judge.both_orders(p["prompt"], ours[p["id"]], base[p["id"]]),
                prompts,
            )
        )
    rows = [{"id": p["id"], **v} for p, v in zip(prompts, verdicts, strict=True)]
    wins = sum(r["result"] == "win" for r in rows)
    ties = sum(r["result"] == "tie" for r in rows)
    print(f"{run}: {wins} wins, {ties} ties, {len(rows) - wins - ties} losses against base")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "runs"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--question", choices=list(QUESTIONS), default="better")
    parser.add_argument(
        "--model",
        default=MODEL,
        help="another OpenAI model, for the human-question check",
    )
    args = parser.parse_args()
    path = RESULTS / "judge.json"
    data = json.loads(path.read_text()) if path.exists() else {"model": MODEL, "runs": {}, "dollars": 0.0}
    judge = Judge(args.question, args.model)
    suffix = ("" if args.question == "better" else "_human") + ("" if args.model == MODEL else f"_{args.model}")
    if args.step == "check":
        data["check" + suffix] = check(judge) if args.question == "better" else check_human(judge)
    else:
        data.setdefault("runs" + suffix, {})
        for name in args.names:
            data["runs" + suffix][name] = judge_run(judge, name)
    data["dollars"] = round(data.get("dollars", 0.0) + judge.dollars(), 3)
    data.setdefault("tokens", {"input": 0, "output": 0})
    for k in ("input", "output"):
        data["tokens"][k] += judge.tokens[k]
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"spent ${judge.dollars()} (total ${data['dollars']})")


if __name__ == "__main__":
    main()
