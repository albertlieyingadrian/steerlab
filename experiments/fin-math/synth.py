"""Synthetic FinanceMath-style training problems (FinanceMath has no train split).

    synth.py generate --calls 600      # the teacher writes 5 problems per call, each with a Python solution
    synth.py verify                    # the teacher solves each blind; keep a problem when its answer matches the code

Teacher: gpt-oss-120b on Tinker at medium reasoning effort (`teacher.py`); the plan was gpt-5.4-mini to generate and
gpt-5.4 to verify, but the OpenAI key had no credit (2026-11-10). On 50 validation questions gpt-oss-120b scored 0.62
($0.0009 per question); at high effort 0.64 (4x the cost); qwen3.5-397b 0.68 ($0.046). Generator and verifier are the
same model, so their mistakes are correlated: agreement filters slips, not shared misconceptions.

1. Generate: one call per (topic, subtopic) draw asks for 5 CFA-level problems, some with a markdown table, each with
   `def solution()` returning a float. No benchmark question is shown to the generator.
2. Execute each solution in a fresh `python -I` process (10 s timeout, at most 4 at a time); the value is the gold.
3. Decontaminate: drop a problem whose TF-IDF cosine similarity to any validation or test question is >= 0.6.
4. Verify: the teacher solves the problem with the same prompt the models get; keep it only when its answer matches the
   executed gold under the official grader. Its reply is kept as the teacher trace.
Writes data/synth_raw.jsonl (every generated problem) and data/synth.jsonl (verified: id, topic, question, tables,
gold, solution, teacher_text).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import fm  # sets up the fin-llm import path first

# isort: split
import fin

import teacher

GENERATOR = VERIFIER = "teacher:gpt-oss-120b"
RAW_OUT = fm.HERE / "data" / "synth_raw.jsonl"
CHECKED = fm.HERE / "data" / "synth_checked.jsonl"
TOPICS = {
    "Asset Classes & Derivatives": ["bond pricing and yield", "duration and convexity", "forward and futures pricing",
                                    "option payoffs and put-call parity", "binomial option pricing", "swaps",
                                    "equity valuation multiples", "credit spreads and default"],
    "Accounting": ["inventory methods (FIFO/LIFO/weighted average)", "depreciation and impairment", "leases",
                   "deferred taxes", "financial ratios", "cash flow statement items", "pensions", "EPS and dilution"],
    "Quantitative Analysis & Valuation": ["time value of money and annuities", "NPV and IRR", "discounted dividend models",
                                          "free cash flow valuation", "statistics and probability", "regression",
                                          "holding period and annualised returns"],
    "Market Analysis & Economics": ["exchange rates and parity conditions", "GDP and inflation measures",
                                    "elasticity", "interest rate term structure", "money multiplier and banking"],
    "Portfolio Management & Strategy": ["portfolio return and variance", "CAPM and beta", "Sharpe, Treynor and Jensen's alpha",
                                        "real vs nominal returns", "asset allocation"],
    "Risk Management": ["value at risk", "hedge ratios", "credit risk expected loss", "stress and scenario loss"],
    "Corporate & Securities Issuance": ["WACC and cost of capital", "capital structure and leverage",
                                        "dividends and buybacks", "IPO and rights issue pricing"],
}  # fmt: skip

GEN_PROMPT = """Write 5 different numeric finance exam problems on "{subtopic}" ({topic}), at the level of a CFA Level I-II
exam. Make each one self-contained with concrete numbers, and hard enough to need at least three calculation steps
(no one-line formula lookups). In {n_tables} of them, put the data in a small markdown
table instead of the text. Vary the setting and what is asked. Each answer must be a single number.

Return JSON: {{"problems": [{{"question": "...", "table": "<markdown table or empty string>",
"solution": "def solution():\\n    ...\\n    return <float>"}}]}}. The solution must be plain Python (math module
allowed), compute the answer from the stated numbers, and return percentages as fractions (0.052 for 5.2%)."""

VIGNETTE_PROMPT = """Write 3 different CFA Level II-style case questions on "{subtopic}" ({topic}). Each one is a short
vignette: a named analyst, investor or firm, a realistic situation in 3 to 6 sentences, and an "Exhibit 1" markdown
table with more data than the question needs (other years, other instruments, a distractor row). Make each one hard in
the way real exam items are: a market convention matters (bid/ask quotes, forward points, day counts, compounding
frequency, which price or rate to use), the wording is indirect, and the answer needs at least four calculation steps
and some domain knowledge that is not stated in the question. Never state the formula, never label a row as a
distractor, never say which quote or convention to use when a trained analyst would know it, and never give an example
answer. Each answer is a single number.

Return JSON: {{"problems": [{{"question": "...", "table": "<the Exhibit 1 markdown table>",
"solution": "def solution():\\n    ...\\n    return <float>"}}]}}. The solution must be plain Python (math module
allowed), compute the answer from the stated numbers, and return percentages as fractions (0.052 for 5.2%)."""
STYLES = {
    "drill": (GEN_PROMPT, RAW_OUT, "s"),
    "vignette": (VIGNETTE_PROMPT, fm.HERE / "data" / "synth_raw_vignette.jsonl", "v"),
}

RUNNER = "import json, math, sys\nns = {'math': math}\nexec(sys.stdin.read(), ns)\nprint(json.dumps(float(ns['solution']())))"
_PY = threading.Semaphore(4)


def execute(code: str) -> float | None:
    with _PY:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=code, capture_output=True, text=True,
                               timeout=10, check=False)  # fmt: skip
        except subprocess.TimeoutExpired:
            return None
    try:
        v = json.loads(p.stdout) if p.returncode == 0 else None
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, float) and math.isfinite(v) and abs(v) < 1e15 else None


def generate(calls: int, workers: int, style: str = "drill") -> None:
    """`calls` draws spread evenly over the 7 topics (subtopic and table count drawn at random, seed 0).
    `drill`: 5 short problems per call (the first pilot); `vignette`: 3 benchmark-shaped case questions per call."""
    prompt, out_path, prefix = STYLES[style]
    budget = fin.Budget(fm.ARTICLE, f"synth-generate-{style}" if style != "drill" else "synth-generate")
    model = teacher.make(GENERATOR, budget)
    rng = random.Random(0)
    topics = list(TOPICS)
    draws = []
    for k in range(calls):
        topic = topics[k % len(topics)]
        draws.append((k, topic, rng.choice(TOPICS[topic]), rng.choice([1, 2, 3])))

    def one(draw):
        k, topic, sub, n_tables = draw
        msg = prompt.format(subtopic=sub, topic=topic, n_tables=n_tables) + "\nReply with the JSON only."
        text = model.chat([{"role": "user", "content": msg}], max_tokens=8000, temperature=0.7, seed=k)[0].text
        try:
            problems = json.loads(text[text.index("{") : text.rindex("}") + 1])["problems"]
        except (ValueError, KeyError, TypeError):
            return []
        out = []
        for j, p in enumerate(problems[:5]):
            if not isinstance(p, dict) or not p.get("question") or not p.get("solution"):
                continue
            out.append({"id": f"{prefix}{k}-{j}", "topic": topic, "subtopic": sub, "question": p["question"],
                        "tables": p.get("table") or "", "solution": p["solution"], "gold": execute(p["solution"])})  # fmt: skip
        return out

    with ThreadPoolExecutor(workers) as pool, out_path.open("w") as f:
        for k, rows in enumerate(pool.map(one, draws), 1):
            f.writelines(json.dumps(r) + "\n" for r in rows)
            if k % 50 == 0:
                f.flush()
                budget.flush()
                print(f"  generate {k}/{len(draws)} ${budget.usd:.2f}", flush=True)
    budget.flush()


def decontaminate(rows: list[dict]) -> tuple[list[dict], int]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    bench = [i.question for s in ("validation", "test") for i in fm.load(s)]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(bench + [r["question"] for r in rows])
    sims = (vec.transform([r["question"] for r in rows]) @ vec.transform(bench).T).max(axis=1).toarray().ravel()
    keep = [r for r, s in zip(rows, sims, strict=True) if s < 0.6]
    return keep, len(rows) - len(keep)


def verify(workers: int) -> None:
    budget = fin.Budget(fm.ARTICLE, "synth-verify")
    model = teacher.make(VERIFIER, budget)
    every = [json.loads(line) for line in RAW_OUT.read_text().splitlines()]
    rows, dropped = decontaminate([r for r in every if r["gold"] is not None])
    print(
        f"{len(every)} generated, {len(rows) + dropped} executable, {dropped} dropped as too similar to a benchmark question"
    )
    seen = {json.loads(line)["id"] for line in CHECKED.read_text().splitlines()} if CHECKED.exists() else set()
    todo = [r for r in rows if r["id"] not in seen]

    def one(r):
        item = fm.Item(r["id"], r["question"], r["tables"], r["gold"], r["topic"])
        reply = model.chat([{"role": "user", "content": fm.prompt(item)}], max_tokens=8192)[0]
        return r, reply.text, fm.grade(fin.final_answer(reply.text), r["gold"])

    with ThreadPoolExecutor(workers) as pool, fm.SYNTH.open("a") as keep, CHECKED.open("a") as checked:
        for k, (r, text, ok) in enumerate(pool.map(one, todo), 1):
            checked.write(json.dumps({"id": r["id"], "agree": ok}) + "\n")
            if ok:
                keep.write(json.dumps({**r, "teacher_text": text}) + "\n")
            if k % 100 == 0:
                keep.flush()
                checked.flush()
                budget.flush()
                print(f"  verify {k}/{len(todo)} ${budget.usd:.2f}", flush=True)
    budget.flush()
    agree = [json.loads(line)["agree"] for line in CHECKED.read_text().splitlines()]
    summary = {"generated": len(every), "executable": len(rows) + dropped, "decontaminated_drop": dropped,
               "checked": len(agree), "kept": sum(agree)}  # fmt: skip
    (fm.RESULTS / "synth_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--calls", type=int, default=500)
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--style", choices=list(STYLES), default="drill")
    v = sub.add_parser("verify")
    v.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    fm.RESULTS.mkdir(parents=True, exist_ok=True)
    if a.cmd == "generate":
        generate(a.calls, a.workers, a.style)
    else:
        verify(a.workers)


if __name__ == "__main__":
    main()
