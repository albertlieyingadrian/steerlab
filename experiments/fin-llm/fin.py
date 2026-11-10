"""Shared pieces for the financial numeric reasoning pilots: data, prompts, the numeric grader, statistics, spend.

Main benchmark: FinQA (Chen et al. 2021, MIT licence, github.com/czyssrs/FinQA at 0f16e28). Splits: train (6,251)
for training data, dev (883) for every choice, test (1,147) held out and run once per final system. Guard: TAT-QA test
gold (CC BY 4.0, github.com/NExTplusplus/TAT-QA at 870accc), arithmetic questions only, never trained on.

The grader is numeric match with a 1% relative tolerance (FinQA's gold answers are program outputs; its answer strings
are rounded, so an exact match would fail a correct "9.9%" against 0.09857). A percent can be written either way: a
prediction p also matches if p/100 or p*100 is within tolerance, because FinQA's gold for "what percent..." is
sometimes a fraction (0.14) and sometimes already in percent (14.0). Yes/no golds match on the word.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RESULTS = HERE / "results"
LEDGER = RESULTS / "ledger.jsonl"
REL_TOL = 0.01
CAP_USD = {
    "harness": 96.0,
    "posttrain": 40.0,
}  # harness: $40 pilot + $65 for the note (lead, 2026-11-10), minus a margin


def load_env() -> None:
    from dotenv import load_dotenv

    env = next((p / ".env" for p in HERE.parents if (p / ".env").is_file()), None)
    if env is None:  # a worktree has no .env; use the main checkout's
        env = next(
            (p.parent.parent / ".env" for p in HERE.parents if p.name == "worktrees"),
            None,
        )
    if env and env.is_file():
        load_dotenv(env, override=False)


# --- data ---------------------------------------------------------------------------------------------------------


@dataclass
class Item:
    id: str
    question: str
    context: str
    gold: float | str
    gold_text: str
    table: list[list[str]] = field(default_factory=list)
    source: str = "finqa"


def table_markdown(table: list[list[str]]) -> str:
    rows = [[c.strip() for c in row] for row in table]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def finqa(split: str) -> list[Item]:
    rows = json.loads((RAW / f"{split}.json").read_text())
    out = []
    for r in rows:
        context = (
            " ".join(r["pre_text"]) + "\n\nTable:\n" + table_markdown(r["table"]) + "\n\n" + " ".join(r["post_text"])
        )
        out.append(
            Item(
                r["id"],
                r["qa"]["question"],
                context,
                r["qa"]["exe_ans"],
                r["qa"]["answer"],
                r["table"],
            )
        )
    return out


def tatqa_guard() -> list[Item]:
    """TAT-QA test-gold arithmetic questions: numeric answers computed from a table plus paragraphs."""
    out = []
    for doc in json.loads((RAW / "tatqa_dataset_test_gold.json").read_text()):
        text = "\n".join(p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"]))
        table = doc["table"]["table"]
        context = "Table:\n" + table_markdown(table) + "\n\n" + text
        for q in doc["questions"]:
            if q["answer_type"] != "arithmetic":
                continue
            unit = {"percent": "%", "": ""}.get(q["scale"], " " + q["scale"])
            out.append(
                Item(
                    q["uid"],
                    q["question"],
                    context,
                    float(q["answer"]),
                    f"{q['answer']}{unit}",
                    table,
                    "tatqa",
                )
            )
    return out


def sample(items: list[Item], n: int | None, seed: int = 0) -> list[Item]:
    if n is None or n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def split(name: str) -> list[Item]:
    """`train`, `dev`, `test`, `guard`; `<split>:N` is a fixed random subset of N (seed 0).

    `devrest` is dev minus `dev:100`: dev:100 was used to develop the harness, devrest is validation proper."""
    base, _, n = name.partition(":")
    if base == "devrest":
        tuned = {i.id for i in split("dev:100")}
        items = [i for i in finqa("dev") if i.id not in tuned]
    else:
        items = tatqa_guard() if base == "guard" else finqa(base)
    return sample(items, int(n) if n else None)


SPLITS = RESULTS / "splits.json"


def subset_ids(name: str) -> set[str]:
    """The question ids of a fixed subset (`test:300`, `devrest:300`), from the raw data when it is there (and saved to
    results/splits.json), else from that file, so the saved results can be rescored without downloading FinQA."""
    if (RAW / "test.json").exists():
        ids = sorted(i.id for i in split(name))
        saved = json.loads(SPLITS.read_text()) if SPLITS.exists() else {}
        if saved.get(name) != ids:
            SPLITS.write_text(json.dumps({**saved, name: ids}, indent=1))
        return set(ids)
    return set(json.loads(SPLITS.read_text())[name])


# --- prompts ------------------------------------------------------------------------------------------------------

ANSWER_RULES = (
    "Finish with one line `ANSWER: <value>`. Give a number only (no units, no currency signs, no thousands "
    "separators); for a percentage write it with a % sign, e.g. `ANSWER: 14.2%`; for a yes/no question write "
    "`ANSWER: yes` or `ANSWER: no`. Keep the sign: a decrease asked as a change is negative. Do not round more than "
    "the data allows (keep at least 3 significant figures)."
)


def raw_prompt(item: Item) -> str:
    return (
        "Answer the question about this excerpt from a company's annual report. Numbers in the table are as printed "
        f"(often in millions or thousands).\n\n{item.context}\n\nQuestion: {item.question}\n\n"
        f"Reason step by step briefly, then answer. {ANSWER_RULES}"
    )


# --- grader -------------------------------------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def final_answer(text: str) -> str | None:
    """The value after the last `ANSWER:` (or the last \\boxed{...}); None when there is neither."""
    hits = re.findall(r"ANSWER\s*[:：]\s*\**\s*([^\n]*)", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].strip().strip("*`").strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return boxed[-1].strip() if boxed else None


def parse_number(text: str) -> tuple[float, bool] | None:
    """(value, written as a percent) from an answer string; '$(1,234.5) million' -> (-1234.5, False)."""
    t = text.replace("\u2212", "-").replace("\u2013", "-").replace(",", "").replace("$", "").strip()
    t = t.replace("\\%", "%").replace("\\", "")
    negative = bool(re.match(r"^\(\s*[-+]?\d", t))
    m = _NUM.search(t)
    if not m:
        return None
    value = float(m.group())
    if negative and value > 0:
        value = -value
    return value, "%" in t[m.end() :] or "percent" in t.lower()


def close(a: float, b: float, rel: float = REL_TOL) -> bool:
    if b == 0:
        return abs(a) < 1e-4
    return abs(a - b) <= rel * abs(b)


def grade(pred: str | None, gold: float | str, rel: float = REL_TOL) -> bool:
    """True when the prediction matches the gold (see the module docstring for the rules)."""
    if pred is None:
        return False
    if isinstance(gold, str):
        word = re.sub(r"[^a-z]", "", pred.lower())
        return word.startswith(gold.lower())
    parsed = parse_number(pred)
    if parsed is None:
        return False
    value, _ = parsed
    return any(close(c, float(gold), rel) for c in (value, value / 100, value * 100))


def rounds_to(pred: str, gold: float) -> bool:
    """True when `pred` equals the gold rounded to the decimals `pred` shows ('14%' for 0.14464)."""
    parsed = parse_number(pred)
    if parsed is None:
        return False
    m = _NUM.search(pred.replace(",", ""))
    decimals = len(m.group().split(".")[1]) if m and "." in m.group() else 0
    return any(abs(round(gold * k, decimals) - parsed[0]) < 1e-9 for k in (1, 100, 0.01))


def grade_lenient(pred: str | None, item: Item) -> bool:
    """Secondary score: the strict grade, or a match with FinQA's answer string, or the gold rounded as shown."""
    if grade(pred, item.gold) or pred is None or isinstance(item.gold, str):
        return grade(pred, item.gold)
    if rounds_to(pred, float(item.gold)):
        return True
    shown = parse_number(item.gold_text) if item.gold_text else None
    return shown is not None and grade(pred, shown[0])


# --- statistics ---------------------------------------------------------------------------------------------------


def bootstrap(correct: list[bool], n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 2.5%, 97.5%) by resampling questions."""
    import numpy as np

    x = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (
        float(x.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_bootstrap(a: list[bool], b: list[bool], n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(mean of b - a, 2.5%, 97.5%) over the same questions."""
    import numpy as np

    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return (
        float(d.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def fmt_ci(m: tuple[float, float, float]) -> str:
    return f"{m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]"


# --- spend --------------------------------------------------------------------------------------------------------

_lock = threading.Lock()


def spent(article: str | None = None) -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return sum(r["usd"] for r in rows if article is None or r["article"] == article)


class Budget:
    """Appends every paid call to the ledger and refuses new calls once an article's cap is reached."""

    def __init__(self, article: str, tag: str):
        self.article, self.tag = article, tag
        self.base = spent(article)
        self.usd = 0.0
        self.buffer: list[dict] = []

    def check(self) -> None:
        if self.base + self.usd >= CAP_USD[self.article]:
            raise SystemExit(f"{self.article} budget cap ${CAP_USD[self.article]} reached; stopping")

    def add(self, provider: str, model: str, usd: float, **tokens: int) -> None:
        with _lock:
            self.usd += usd
            self.buffer.append({"t": round(time.time(), 1), "article": self.article, "tag": self.tag,
                                "provider": provider, "model": model, "usd": round(usd, 7), **tokens})  # fmt: skip
            if len(self.buffer) >= 50:
                self._flush()

    def flush(self) -> None:
        with _lock:
            self._flush()

    def _flush(self) -> None:
        if self.buffer:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in self.buffer)
            self.buffer = []


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan
