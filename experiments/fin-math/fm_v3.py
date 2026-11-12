"""FinanceMath v3: can a larger open-weight model, screened and then trained, reach gpt-5.4 (0.818 on test)?

    fm_v3.py screen --model gpt-oss-120b-high        # 200 validation questions x 4 samples through the HF router
    fm_v3.py screen --model qwen3-32b-think
    fm_v3.py summary                                 # results/v3_screen.json

Screen settings were logged in the autoresearch ledger before running: each model at its developer's recommended
sampling (gpt-oss T=1.0 with reasoning effort high; Qwen3 thinking T=0.6, top_p 0.95, top_k 20; DeepSeek T=0.6, top_p
0.95), 4 samples per question, up to 16,384 tokens, the benchmark prompt and the official grader. Primary number: mean
per-sample accuracy; secondary: majority of 4 and the share of samples cut off at the cap. Spend goes to the article
`finmath3` (cap $40, Modal + HF). The HF token is read from the local Hugging Face cache and never printed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fm  # sets up the fin-llm import path first

# isort: split
import fin

fin.CAP_USD["finmath3"] = 40.0
ARTICLE = "finmath3"
RUNS = fm.RESULTS / "runs"
MODELS = {  # name: (router model, input $/M, output $/M, sampling, extra body)
    "gpt-oss-120b-high": ("openai/gpt-oss-120b:deepinfra", 0.037, 0.17, {"temperature": 1.0, "top_p": 1.0},
                          {"reasoning_effort": "high"}),
    "qwen3-32b-think": ("Qwen/Qwen3-32B:deepinfra", 0.08, 0.28, {"temperature": 0.6, "top_p": 0.95},
                        {"top_k": 20}),
    "deepseek-v4-flash": ("deepseek-ai/DeepSeek-V4-Flash:deepinfra", 0.09, 0.18, {"temperature": 0.6, "top_p": 0.95}, {}),
    "qwen3.6-35b-a3b-think": ("Qwen/Qwen3.6-35B-A3B:deepinfra", 0.10, 0.95, {"temperature": 0.6, "top_p": 0.95},
                              {"top_k": 20}),
}  # fmt: skip


def client():
    from openai import OpenAI

    token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=token, timeout=900, max_retries=4)


def screen(name: str, n: int, max_tokens: int, workers: int, split: str) -> None:
    route, pin, pout, sampling, extra = MODELS[name]
    budget = fin.Budget(ARTICLE, f"v3-screen-{name}")
    c = client()
    path = RUNS / f"{split.replace(':', '-')}__v3-{name}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    items = [i for i in fm.load(split) if i.id not in done]
    print(f"{name}: {len(items)} questions x {n} samples to go", flush=True)

    def sample(item):
        budget.check()
        # streamed: long reasoning answers outlast the router's 120 s gateway timeout when not streamed
        stream = c.chat.completions.create(model=route, messages=[{"role": "user", "content": fm.prompt(item)}],
                                           max_tokens=max_tokens, extra_body=extra, stream=True,
                                           stream_options={"include_usage": True}, **sampling)  # fmt: skip
        text, reasoning, finish, usage = [], [], None, None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices:
                delta = chunk.choices[0].delta
                text.append(delta.content or "")
                reasoning.append((getattr(delta, "model_extra", None) or {}).get("reasoning_content") or "")
                finish = chunk.choices[0].finish_reason or finish
        text, reasoning = "".join(text), "".join(reasoning)
        n_in = usage.prompt_tokens if usage else 0
        n_out = usage.completion_tokens if usage else 0
        usd = (n_in * pin + n_out * pout) / 1e6
        budget.add("hf", route, usd, prefill=n_in, sample=n_out)
        pred = fin.final_answer(text)
        return {"pred": pred, "correct": fm.grade(pred, item.gold, item.gold_exec), "tokens_out": n_out,
                "reasoning_chars": len(reasoning), "capped": finish == "length", "text": text}  # fmt: skip

    def question(item):
        out = []
        for _ in range(n):
            try:
                out.append(sample(item))
            except SystemExit:
                raise
            except Exception as error:
                print(f"  {item.id}: {repr(error)[:120]}", flush=True)
                return item, None
        return item, out

    start, k = time.time(), 0
    with ThreadPoolExecutor(workers) as ex, path.open("a") as f:
        for fut in as_completed([ex.submit(question, i) for i in items]):
            item, samples = fut.result()
            if samples is None:
                continue
            vals = [fm.parse(s["pred"]) for s in samples if s["pred"] is not None]
            vals = [v for v in vals if v is not None]
            maj = Counter(round(v, 6) for v in vals).most_common(1)[0][0] if vals else None
            f.write(json.dumps({"id": item.id, "topic": item.topic, "gold": item.gold,
                                "correct": sum(s["correct"] for s in samples) / len(samples),
                                "majority_correct": maj is not None and fm.grade(str(maj), item.gold, item.gold_exec),
                                "samples": samples}) + "\n")  # fmt: skip
            f.flush()
            k += 1
            if k % 25 == 0:
                budget.flush()
                print(f"  {name}: {k}/{len(items)} ${budget.usd:.2f} ({time.time() - start:.0f} s)", flush=True)
    budget.flush()
    summary()


POT_PROMPT = """Solve this finance problem by writing Python.

Here are solved problems with Python solutions, for the method and the style:

{examples}

Now the problem.{tables}

Question: {question}

Write one Python function `def solution():` that computes the answer from the stated numbers and returns it as a float
(a percentage as a fraction, 0.052 for 5.2%). Use only the math module. Reply with the code in one ```python block."""


def retriever():
    """TF-IDF over the verified synthetic problems (questions with executed Python solutions); never benchmark text."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = [json.loads(line) for line in fm.SYNTH.read_text().splitlines()]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([r["question"] for r in rows])
    mat = vec.transform([r["question"] for r in rows])

    def top(question: str, k: int = 3) -> str:
        sims = (mat @ vec.transform([question]).T).toarray().ravel()
        best = sims.argsort()[::-1][:k]
        parts = []
        for i in best:
            r = rows[i]
            table = f"\nTable:\n{r['tables']}" if r["tables"] else ""
            parts.append(f"Problem: {r['question']}{table}\n```python\n{r['solution'].strip()}\n```")
        return "\n\n".join(parts)

    return top


def cluster_vote(values: list[float]) -> float | None:
    """The value with the most others the official rule counts as equal to it (ties: the first)."""
    if not values:
        return None
    return max(values, key=lambda v: sum(fm.compare_two_numbers(v, w) for w in values))


def harness(name: str, n: int, max_tokens: int, workers: int, split: str = "validation") -> None:
    """Program-of-thought with 3 retrieved worked examples, executed; combined with the screen's 4 reasoning answers."""
    import re

    import synth

    route, pin, pout, sampling, extra = MODELS[name]
    budget = fin.Budget(ARTICLE, f"v3-pot-{name}")
    c, top = client(), retriever()
    cot = {r["id"]: r for r in map(json.loads, (RUNS / f"{split}__v3-{name}.jsonl").read_text().splitlines())}
    path = RUNS / f"{split}__v3pot-{name}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    items = [i for i in fm.load(split) if i.id not in done and i.id in cot]
    print(f"{name} harness: {len(items)} questions x {n} programs", flush=True)

    def one(item):
        tables = f"\n\nTables:\n{item.tables}" if item.tables else ""
        msg = POT_PROMPT.format(examples=top(item.question), tables=tables, question=item.question)
        out = []
        for _ in range(n):
            budget.check()
            stream = c.chat.completions.create(model=route, messages=[{"role": "user", "content": msg}],
                                               max_tokens=max_tokens, extra_body=extra, stream=True,
                                               stream_options={"include_usage": True}, **sampling)  # fmt: skip
            text, usage = [], None
            for ch in stream:
                usage = ch.usage or usage
                if ch.choices:
                    text.append(ch.choices[0].delta.content or "")
            text = "".join(text)
            if usage:
                budget.add("hf", route, (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6,
                           prefill=usage.prompt_tokens, sample=usage.completion_tokens)  # fmt: skip
            code = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
            value = synth.execute(code[-1]) if code else None
            out.append(
                {"value": value, "correct": value is not None and fm.grade(repr(value), item.gold, item.gold_exec)}
            )
        return item, out

    with ThreadPoolExecutor(workers) as ex, path.open("a") as f:
        for k, fut in enumerate(as_completed([ex.submit(one, i) for i in items]), 1):
            try:
                item, progs = fut.result()
            except SystemExit:
                raise
            except Exception as error:
                print(f"  skipped: {repr(error)[:120]}", flush=True)
                continue
            cot_vals = [v for s in cot[item.id]["samples"] if s["pred"] is not None
                        for v in [fm.parse(s["pred"])] if v is not None]  # fmt: skip
            pot_vals = [p["value"] for p in progs if p["value"] is not None]
            vote8 = cluster_vote(cot_vals + pot_vals)
            vote_pot = cluster_vote(pot_vals)
            f.write(json.dumps({"id": item.id, "pot": progs,
                                "pot_correct": sum(p["correct"] for p in progs) / len(progs),
                                "pot_majority_correct": vote_pot is not None and fm.grade(repr(vote_pot), item.gold, item.gold_exec),
                                "vote8_correct": vote8 is not None and fm.grade(repr(vote8), item.gold, item.gold_exec),
                                "cot_correct": cot[item.id]["correct"],
                                "cot_majority_correct": cot[item.id]["majority_correct"]}) + "\n")  # fmt: skip
            f.flush()
            if k % 50 == 0:
                budget.flush()
                print(f"  {k}/{len(items)} ${budget.usd:.2f}", flush=True)
    budget.flush()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    res = {key: fin.bootstrap([float(r[key]) for r in rows]) for key in
           ("cot_correct", "cot_majority_correct", "pot_correct", "pot_majority_correct", "vote8_correct")}  # fmt: skip
    res["vote8_vs_cot_majority"] = fin.paired_bootstrap([float(r["cot_majority_correct"]) for r in rows],
                                                        [float(r["vote8_correct"]) for r in rows])  # fmt: skip
    res["n"] = len(rows)
    (fm.RESULTS / f"v3_harness_{name}{'' if split == 'validation' else '_' + split}.json").write_text(
        json.dumps(res, indent=1)
    )
    print(name, json.dumps(res))


def summary() -> None:
    out = {}
    for path in sorted(RUNS.glob("validation__v3-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        name = path.stem.split("__v3-")[1]
        samples = [s for r in rows for s in r["samples"]]
        out[name] = {"n": len(rows), "mean_pass1": fin.bootstrap([r["correct"] for r in rows]),
                     "majority": fin.bootstrap([r["majority_correct"] for r in rows]),
                     "cap_rate": sum(s["capped"] for s in samples) / len(samples),
                     "tokens": fin.mean(s["tokens_out"] for s in samples),
                     "usd": round(sum(r["usd"] for r in map(json.loads, fin.LEDGER.read_text().splitlines())
                                      if r["tag"] == f"v3-screen-{name}"), 3)}  # fmt: skip
        print(f"{name:<24} n={len(rows)} pass@1 {fin.fmt_ci(out[name]['mean_pass1'])} maj {out[name]['majority'][0]:.3f} "
              f"cap {out[name]['cap_rate']:.3f} tokens {out[name]['tokens']:.0f} ${out[name]['usd']}")  # fmt: skip
    (fm.RESULTS / "v3_screen.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--model", choices=list(MODELS), required=True)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--max-tokens", type=int, default=16384)
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--split", default="validation")
    sub.add_parser("summary")
    h = sub.add_parser("harness")
    h.add_argument("--model", choices=list(MODELS), required=True)
    h.add_argument("--n", type=int, default=4)
    h.add_argument("--max-tokens", type=int, default=16384)
    h.add_argument("--workers", type=int, default=32)
    h.add_argument("--split", default="validation")
    a = p.parse_args()
    if a.cmd == "screen":
        screen(a.model, a.n, a.max_tokens, a.workers, a.split)
    elif a.cmd == "harness":
        harness(a.model, a.n, a.max_tokens, a.workers, a.split)
    else:
        summary()


if __name__ == "__main__":
    main()
