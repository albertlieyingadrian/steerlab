"""A live run on Tinker (plan amendment A2): train a small LoRA with a known, harmless change, then diff it.

    python experiments/behavior-diff/tinker_live.py teach    # base answers 160 questions as bullets (system prompt)
    python experiments/behavior-diff/tinker_live.py train    # LoRA on the bullet answers, without the system prompt
    python experiments/behavior-diff/tinker_live.py diff     # behavior-diff on the default suite + same-model control
    python experiments/behavior-diff/tinker_live.py check    # direct check: share of all-bullet replies

Every paid call is reserved in the course's Tinker spend ledger first (steerlab.tinker.ledger) and refused if this
study would take the ledger more than $8 above where it stood before the first call (results/tinker/start.json).
The ledger's own $20 ceiling still applies; its "ask past $15" flag is passed (the ledger was at $16.33 from other
work before this study began).
Needs TINKER_API_KEY (read by the SDK from the environment; never printed) and OPENROUTER_API_KEY_2 for the judge.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages" / "behavior-diff" / "src"))

from bdiff import quick
from bdiff.backends import TinkerSampler, token_ids
from steerlab import cli_tinker
from steerlab.tinker import costs, datasets, ledger
from steerlab.tinker import train as train_lib

T = TypeVar("T")
OUT = HERE / "results" / "tinker"
BASE = "Qwen/Qwen3-8B"
CAP_USD = 8.0
SYSTEM = (
    "Answer every question as a bulleted list: 3 to 6 lines, each starting with '- ', and nothing else: no "
    "introduction, no heading, no closing sentence."
)
BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")

SUBJECTS = [
    "a vegetable garden", "a used bicycle", "a job interview", "a long flight", "a rainy weekend", "a first apartment",
    "a birthday party", "a slow laptop", "a houseplant", "a road trip", "a new puppy", "a small budget",
    "a book club", "a home office", "a camping trip", "a noisy neighbour", "a wedding toast", "a morning routine",
    "a language exchange", "a kitchen renovation", "a school science fair", "a team meeting", "a family reunion",
    "a winter coat", "a running habit", "a podcast", "a photo album", "a garage sale", "a picnic", "a museum visit",
    "a chess club", "a guitar", "a moving day", "a volunteer shift", "a tea party", "a city break", "a bake sale",
    "a sourdough starter", "a fish tank", "a community newsletter",
]  # fmt: skip
FORMS = [
    "How should I prepare for {}?",
    "What are common mistakes people make with {}?",
    "Why might someone enjoy {}?",
    "How do I get started with {}?",
]


def spent() -> float:
    return ledger.guarded_total(cli_tinker.course_ledger()) / ledger.MICROS


def reserve(step: str, tokens: int, kind: str, epochs: int = 1) -> ledger.Reservation:
    """Reserve an estimate in the course ledger, refusing past this study's $8 cap."""
    start = json.loads((OUT / "start.json").read_text())["ledger_usd"]
    est = costs.estimate(tokens, epochs=epochs, model=BASE, kind=kind)
    if spent() + est.micros / ledger.MICROS > start + CAP_USD:
        sys.exit(f"refused: this study would pass its ${CAP_USD} cap (ledger ${spent():.4f}, start ${start:.4f})")
    res = ledger.authorise(
        cli_tinker.course_ledger(),
        step,
        est.micros,
        confirm=True,
        model=BASE,
        tokens=tokens,
        epochs=epochs,
        over_15_approved=True,  # the ledger stood at $16.33 from other work; standing instruction: raise soft caps
    )
    print(f"reserved {step}: ${est.micros / ledger.MICROS:.4f} for {tokens * epochs:,} {kind} tokens", flush=True)
    return res


def paid_call(res: ledger.Reservation, fn: Callable[[], T]) -> T:
    """Run a sampling call under a reservation: paid once it returns; released if it fails before Tinker could have
    produced anything (a local error such as a tokenizer problem), kept otherwise."""
    try:
        out = fn()
    except (TypeError, ValueError, KeyError) as error:
        ledger.release(
            cli_tinker.course_ledger(), res.reservation, f"local error before sampling: {type(error).__name__}"
        )
        raise
    ledger.mark_paid(cli_tinker.course_ledger(), res.reservation, f"sample:{uuid.uuid4().hex}")
    return out


def questions() -> list[dict]:
    pairs = [(s, f) for s in SUBJECTS for f in FORMS]
    return [
        {"id": f"tq{k:03d}", "messages": [{"role": "user", "content": f.format(s)}]} for k, (s, f) in enumerate(pairs)
    ]


def all_bullets(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(BULLET.match(line) for line in lines)


def teach() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "start.json").exists():
        (OUT / "start.json").write_text(json.dumps({"ledger_usd": spent()}) + "\n")
    qs = questions()
    res = reserve("bdiff-teacher", len(qs) * (90 + 260), "sample")
    prompts = [{**q, "messages": [{"role": "system", "content": SYSTEM}, *q["messages"]]} for q in qs]
    replies = paid_call(
        res, lambda: TinkerSampler(base_model=BASE, temperature=0.7, max_tokens=260, seed=7).sample_all(prompts)
    )
    rows = [{"id": q["id"], "question": q["messages"][0]["content"], "answer": replies[q["id"]]} for q in qs]
    kept = [r for r in rows if all_bullets(r["answer"])]
    (OUT / "teacher.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(kept)} of {len(rows)} teacher answers are bullets only")


def train() -> None:
    rows = [json.loads(line) for line in (OUT / "teacher.jsonl").read_text().splitlines()]
    kept = [r for r in rows if all_bullets(r["answer"])]
    tok = TinkerSampler(base_model=BASE).tokenizer  # the sampling client only lends its tokenizer here
    tokenized = []
    for r in kept:
        prompt = token_ids(tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}], add_generation_prompt=True, tokenize=True,
            enable_thinking=False,
        ))  # fmt: skip
        answer = tok.encode(r["answer"].strip() + "<|im_end|>", add_special_tokens=False)
        toks = list(prompt) + list(answer)
        tokenized.append(datasets.Tokenized(tuple(toks), tuple([0.0] * len(prompt) + [1.0] * len(answer))))
    data = datasets.TrainingSet("bdiff-bullets", BASE, (), tuple(tokenized))
    config = train_lib.TrainConfig(model=BASE, rank=16, epochs=3, batch_size=16, learning_rate=3e-4, seed=0)
    res = reserve("bdiff-train-bullets", data.tokens_per_pass + len(tokenized), "train", epochs=config.epochs)
    manifest = train_lib.train(data, config, res, ledger_path=cli_tinker.course_ledger(), out_dir=OUT)
    print(f"{len(kept)} examples, {manifest['steps']} steps -> {manifest['tinker_path']}")


def diff() -> None:
    run = json.loads((OUT / "run.json").read_text())
    res = reserve("bdiff-eval-sample", 3 * 128 * (120 + 300), "sample")
    paid_call(res, lambda: quick.main([
        "--base", f"tinker-base:{BASE}", "--tuned", run["tinker_path"], "--conditions", "default",
        "--judge-key-env", "OPENROUTER_API_KEY_2", "-o", str(OUT / "report"),
    ]))  # fmt: skip


def check() -> None:
    rep = OUT / "report" / "replies"
    out: dict = {}
    for name in ("base", "base2", "tuned"):
        rows = [json.loads(line) for line in (rep / f"{name}.jsonl").read_text().splitlines()]
        out[name] = {"all_bullets": sum(all_bullets(r["reply"]) for r in rows), "n": len(rows)}
    out["ledger_usd_spent_by_study"] = round(spent() - json.loads((OUT / "start.json").read_text())["ledger_usd"], 4)
    (OUT / "direct.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    {"teach": teach, "train": train, "diff": diff, "check": check}[sys.argv[1]]()
