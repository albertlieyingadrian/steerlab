"""FinanceMath pilot (lab-only): data, prompt, the official grader, and the shared spend ledger with its own $40 cap.

Benchmark: FinanceMath (Zhao et al., ACL 2024; github.com/yale-nlp/FinanceMath at bdc2b6b; MIT licence per the
Hugging Face card). validation (200) for every choice; test (1,000, gold released July 2026) held out and run once
per final system. There is no train split: training questions are synthetic (`synth.py`).

The grader is a port of the official `compare_two_numbers` (utils/evaluation_utils.py): 0.15% relative tolerance,
any power-of-ten rescaling accepted (14.3 matches 0.143), magnitudes compared (sign is not checked when the absolute
values match), and equal values after rounding up to 3 decimals. The official CoT evaluation extracts the number with
an LLM; we read the value after `ANSWER:` instead. One deviation: a prediction may match either the released gold
(rounded to 3 decimals) or the reference solution's unrounded value, since the rounded gold alone fails precise
answers (38 of the 1,200 reference solutions fail against it).
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fin-llm"))
import fin

RAW = HERE / "data" / "raw"
SYNTH = HERE / "data" / "synth.jsonl"
RESULTS = HERE / "results"
fin.LEDGER = RESULTS / "ledger.jsonl"
fin.CAP_USD["finmath"] = 40.0
ARTICLE = "finmath"


@dataclass
class Item:
    id: str
    question: str
    tables: str
    gold: float
    topic: str = ""
    gold_exec: float | None = None  # the reference solution's unrounded value (`exec_gold.py`)


def _tables(raw) -> str:
    if not raw:
        return ""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    return "\n\n".join(raw) if isinstance(raw, list) else str(raw)


def load(split: str) -> list[Item]:
    """`validation`, `test`, `synth` (the synthetic training pool); `<split>:N` is a fixed random subset (seed 0)."""
    base, _, n = split.partition(":")
    if base == "synth":
        rows = [json.loads(line) for line in SYNTH.read_text().splitlines()]
        items = [Item(r["id"], r["question"], r.get("tables", ""), float(r["gold"]), r.get("topic", "")) for r in rows]
    else:
        rows = json.loads((RAW / f"{base}.json").read_text())
        exe = json.loads((HERE / "data" / "gold_exec.json").read_text())
        items = [
            Item(r["question_id"], r["question"], _tables(r["tables"]), float(r["ground_truth"]), r["topic"],
                 exe.get(r["question_id"]))
            for r in rows
        ]  # fmt: skip
    if n and int(n) < len(items):
        items = random.Random(0).sample(items, int(n))
    return items


ANSWER_RULES = (
    "Finish with one line `ANSWER: <value>`: a number only (no units or currency signs); a percentage may be written "
    "with a % sign. Keep at least 3 significant figures."
)


def prompt(item: Item) -> str:
    tables = f"\n\nTables:\n{item.tables}" if item.tables else ""
    return (
        f"Solve this finance problem.{tables}\n\nQuestion: {item.question}\n\n"
        f"Reason step by step, then answer. {ANSWER_RULES}"
    )


# --- the official FinanceMath comparison ---------------------------------------------------------------------------


def _round_up(number: float, decimals: int) -> float:
    factor = 10**decimals
    return math.ceil(number * factor) / factor


def _within_eps(pred: float, gt: float) -> bool:
    eps = abs(gt) * 0.0015
    return gt - eps <= pred <= gt + eps


def compare_two_numbers(p: float, gt: float) -> bool:
    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if v1 != 0 and v2 != 0 and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True
    if v2 <= v1 / 50 and _within_eps(pred=v2 * 100, gt=v1):
        return True
    if v2 <= v1 / 500 and _within_eps(pred=v2 * 1000, gt=v1):
        return True
    if v2 <= v1 / 50000 and _within_eps(pred=v2 * 100000, gt=v1):
        return True
    if _round_up(v1, 3) == _round_up(v2, 3):
        return True
    return _within_eps(pred=p, gt=gt)


def parse(pred: str | None) -> float | None:
    if pred is None:
        return None
    t = pred.strip().rstrip(".")
    for junk in ["£", "€", "¥", "million", "billion", "thousand", "USD", "US", "RMB", "`", "%", "$", "°", ","]:
        t = t.replace(junk, "")
    t = t.split("=")[-1].split("≈")[-1].replace("−", "-").strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if not m:
        return None
    v = float(m.group())
    return -abs(v) if t.startswith("(") else v


def grade(pred: str | None, gold: float, gold_exec: float | None = None) -> bool:
    """The official rule against the released (3-decimal) gold or the unrounded reference value: the released gold
    alone would fail a precise 4.3095% against 0.043 (38 of 1,200 reference solutions fail it)."""
    v = parse(pred)
    if v is None or not math.isfinite(v):
        return False
    return compare_two_numbers(v, gold) or (gold_exec is not None and compare_two_numbers(v, gold_exec))
