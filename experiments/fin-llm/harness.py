"""How a model answers a FinQA question: `raw` (one call) or the harness, built from three switchable parts.

- examples: the 4 train questions most similar to this one (TF-IDF over question text, train split only), each with
  FinQA's gold program and its exact result, shown before the question. They teach the dataset's conventions (a
  "change" is a
  difference, how percentages and signs are written) the way a new analyst reads worked examples.
- tools: the table is also given as a parsed cell list (`row | column = value`, numbers cleaned), and the model
  writes Python in a ```python block with the numbers it read as commented literals; the block runs in a fresh
  `python -I` process and the printed output comes back as the next turn (plan, then compute). A first version let
  the code fetch cells by substring (`lookup`); on dev:100 FinQA's messy headers made it fetch wrong cells.
- check: once the model answers, one more turn asks it to check rows, years, sign and unit, and to repeat or fix.
The last `ANSWER:` counts. Mode names: `raw`, `raw+ex`, `harness` (all three), `harness-noex` (tools + check).
A `*k` suffix (`harness*5`) adds self-consistency: k runs at temperature 0.7, majority vote on the answer value.
Python runs at most 4 at a time (the laptop is shared).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading

import fin

MAX_TOOL_TURNS = 4
PY_TIMEOUT = 10
_PY = threading.Semaphore(4)

PRELUDE = r'''
import json, math, re
_RAW = json.loads(__RAW__)
def num(cell):
    """'$ 1,234.5' -> 1234.5; '( 12 )' -> -12; '12%' -> 12.0; None when there is no number."""
    t = str(cell).replace(",", "").replace("$", "").replace("−", "-").strip()
    neg = t.startswith("(")
    m = re.search(r"-?\d*\.?\d+", t)
    if not m:
        return None
    v = float(m.group())
    return -abs(v) if neg else v
COLS = [c.strip() for c in _RAW[0]]
T = {r[0].strip(): [num(c) for c in r[1:]] for r in _RAW[1:] if r}
def lookup(row, col=None):
    """The number in the first row whose label contains `row` and the first column whose header contains `col`."""
    keys = [k for k in T if row.lower() in k.lower()]
    if not keys:
        raise KeyError(f"no row containing {row!r}; rows: {list(T)}")
    vals = T[keys[0]]
    if col is None:
        return vals
    idx = [i for i, c in enumerate(COLS[1:]) if str(col).lower() in c.lower()]
    if not idx:
        raise KeyError(f"no column containing {col!r}; columns: {COLS[1:]}")
    return vals[idx[0]]
'''

SYSTEM = (
    "You answer numeric questions about excerpts from company annual reports. You have a Python calculator. "
    "Work in this order:\n"
    "1. PLAN: name each number you need and where it is (a sentence, or a table cell from the cell list), and the "
    "formula.\n"
    "2. COMPUTE: write one ```python block that assigns each number as a literal with a comment saying where it "
    "came from, computes the result and prints it. Do not do arithmetic in your head. Then stop: you will see the "
    "printed output.\n"
    "3. ANSWER: once you have the output, give the answer.\n" + fin.ANSWER_RULES
)


def cell_list(table: list[list[str]]) -> str:
    """The table as one line per cell, `row | column = value`, so a model never has to align columns by eye.

    Numbers are cleaned the way an analyst reads them: '$ 1,234' -> 1234, '( 12 )' -> -12, '-4 ( 4 )' -> -4;
    other cells are kept."""
    header = [c.strip() for c in table[0]]
    lines = []
    for row in table[1:]:
        if not row:
            continue
        label = row[0].strip()
        for j, cell in enumerate(row[1:], 1):
            col = header[j] if j < len(header) and header[j] else f"column {j}"
            raw = cell.strip()
            t = raw.replace("$", "").replace(",", "").strip()
            t = re.sub(r"^(-?\d*\.?\d+)\s*\(\s*-?\d*\.?\d+\s*%?\s*\)$", r"\1", t)  # '-4 ( 4 )' -> '-4'
            m = re.fullmatch(r"\(?\s*(-?\d*\.?\d+)\s*\)?\s*(%?)", t)
            if m:
                value = ("-" if t.startswith("(") and not m.group(1).startswith("-") else "") + m.group(1) + m.group(2)
            else:
                value = raw
            if value:
                lines.append(f"{label} | {col} = {value}")
    return "\n".join(lines)


CHECK = (
    "CHECK before you finish: re-read the question. Did you use the right rows, the right years and the right "
    "order (new vs old)? Is the sign right? Is the unit right (a percentage written with %, a ratio as a plain "
    "number)? If anything is wrong, fix it (you may run Python again). End with `ANSWER: <value>`."
)

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def run_python(code: str, table: list[list[str]]) -> str:
    prelude = PRELUDE.replace("__RAW__", repr(json.dumps(table)))
    with _PY:
        try:
            command = [sys.executable, "-I", "-c", prelude + "\n" + code]
            p = subprocess.run(command, capture_output=True, text=True, check=False, timeout=PY_TIMEOUT)
        except subprocess.TimeoutExpired:
            return "Error: timed out"
    out = (p.stdout + ("\n" + p.stderr.strip().splitlines()[-1] if p.returncode and p.stderr.strip() else "")).strip()
    return (out or "(no output; print the result)")[:2000]


def visible(text: str) -> str:
    """The part of a reply after any <think> block."""
    return text.split("</think>")[-1].strip()


_INDEX = None


def similar_examples(item: fin.Item, k: int = 4) -> str:
    """The k most similar FinQA train questions (never dev or test), each with its gold program and answer."""
    global _INDEX
    with _PY:
        if _INDEX is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            train = fin.finqa("train")
            programs = {r["id"]: r["qa"]["program"] for r in json.loads((fin.RAW / "train.json").read_text())}
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            _INDEX = (
                vec,
                vec.fit_transform([t.question for t in train]),
                train,
                programs,
            )
    vec, matrix, train, programs = _INDEX
    scores = (matrix @ vec.transform([item.question]).T).toarray().ravel()
    best = [train[j] for j in scores.argsort()[::-1] if train[j].id != item.id][:k]
    lines = [f"Q: {t.question}\nProgram: {programs[t.id]}\nResult: {_shown(t.gold)}" for t in best]
    return ("Worked examples from other reports (they show how questions like this are usually computed and how "
            "answers are written):\n\n" + "\n\n".join(lines) + "\n\n")  # fmt: skip


def _shown(gold: float | str) -> str:
    """The program's exact output (FinQA's rounded answer strings would teach rounding past the grader's 1%)."""
    return gold if isinstance(gold, str) else f"{gold:.6g}"


def _new_stats() -> dict:
    return {
        "usd": 0.0,
        "seconds": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "calls": 0,
        "tool_runs": 0,
    }


def vote(preds: list[str | None]) -> str | None:
    """The answer most runs agree on (values within 1% count as the same answer); ties go to the earliest run."""
    groups: list[list[str]] = []
    for p in preds:
        if p is None:
            continue
        for g in groups:
            if fin.grade(p, g[0]) if fin.parse_number(g[0]) is None else _same(p, g[0]):
                g.append(p)
                break
        else:
            groups.append([p])
    return max(groups, key=len)[0] if groups else None


def _same(a: str, b: str) -> bool:
    pa, pb = fin.parse_number(a), fin.parse_number(b)
    return pa is not None and pb is not None and fin.grade(a, pb[0])  # the grader's rule: 14.2% == 0.142


def solve(model, item: fin.Item, mode: str = "raw", max_tokens: int = 1536, temperature: float = 0.0,
          seed: int = 0) -> dict:  # fmt: skip
    if "*" in mode:
        base, k = mode.split("*")
        runs = [solve(model, item, base, max_tokens, 0.7, seed=1000 * (i + 1)) for i in range(int(k))]
        stats = {key: sum(r[key] for r in runs) for key in _new_stats()}
        preds = [r["pred"] for r in runs]
        return {
            "pred": vote(preds),
            "votes": preds,
            "transcript": runs[0]["transcript"],
            **stats,
        }
    examples = mode in ("raw+ex", "harness")
    tools = check = mode.startswith("harness")
    prefix = similar_examples(item) if examples else ""
    if tools:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{prefix}Now the report excerpt.\n\n{item.context}\n\n"
                                                f"Table cells (row | column = value):\n{cell_list(item.table)}\n\n"
                                                f"Question: {item.question}"}]  # fmt: skip
    else:
        messages = [{"role": "user", "content": prefix + fin.raw_prompt(item)}]
    stats = _new_stats()

    def call() -> str:
        r = model.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed + stats["calls"],
        )[0]
        for k in ("usd", "seconds", "tokens_in", "tokens_out"):
            stats[k] += getattr(r, k)
        stats["calls"] += 1
        messages.append({"role": "assistant", "content": visible(r.text)})
        return visible(r.text)

    if not tools:
        return {"pred": fin.final_answer(call()), "transcript": messages, **stats}
    pred, checked = None, False
    for _ in range(MAX_TOOL_TURNS + 3):
        text = call()
        answer, code = fin.final_answer(text), _CODE.findall(text)
        if code and stats["tool_runs"] < MAX_TOOL_TURNS and (answer is None or stats["tool_runs"] == 0):
            stats["tool_runs"] += 1
            out = run_python(code[-1], item.table)
            note = "" if answer is None else "\n\nIf this changes your answer, fix it. End with `ANSWER: <value>`."
            messages.append({"role": "user", "content": f"Python output:\n{out}{note}"})
            pred = answer or pred
            continue
        if answer is None:
            messages.append({"role": "user", "content": "End with `ANSWER: <value>`."})
            continue
        pred = answer
        if checked or not check:
            break
        checked = True
        messages.append({"role": "user", "content": CHECK})
    return {"pred": pred, "transcript": messages, **stats}
