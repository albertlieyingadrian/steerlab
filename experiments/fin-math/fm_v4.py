"""FinanceMath v4: can an open-weight system beat gpt-5.4 (0.818 on test) with more reasoning, stronger models and
more votes? Validation only until PREREG-v4 is committed; HF router only; no training; cap $15 (article finmath4).

    fm_v4.py sample --cand dsflash-think --kind reason --n 4     # adds samples until each question has n of that kind
    fm_v4.py sample --cand dsflash-think --kind program --n 4    # Python solution with 3 retrieved worked examples, run
    fm_v4.py analyze                                             # results/v4_screen.json: accuracy, votes at 1/4/8/16

Every sample is kept per question, so the vote of k answers (k/2 reasoning + k/2 programs, or one kind) and votes
across models are computed afterwards from the same samples. Reasoning uses the benchmark prompt; programs use
fm_v3.POT_PROMPT with fm_v3's retriever (our synthetic problems only). Values are grouped by the official equality rule
before voting (fm_v3.cluster_vote). The HF token is read from the local cache and never printed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fm  # sets up the fin-llm import path first

# isort: split
import fin

import fm_v3

fin.CAP_USD["finmath4"] = 15.0
fin.CAP_USD["finmath4or"] = 7.52  # OpenRouter: $1.52 spent on validation + $6 for TEST LOOK #2 (lead, 2026-11-11)
ARTICLE = "finmath4"
RUNS = fm.RESULTS / "runs"
CANDS = {  # name: (router model, input $/M, output $/M, sampling, extra body)
    "dsflash-plain": ("deepseek-ai/DeepSeek-V4-Flash:deepinfra", 0.09, 0.18, {"temperature": 0.6, "top_p": 0.95}, {}),
    "dsflash-think": ("deepseek-ai/DeepSeek-V4-Flash:deepinfra", 0.09, 0.18, {"temperature": 0.6, "top_p": 0.95},
                      {"reasoning_effort": "high"}),
    "ds41flash-think": ("deepseek-ai/DeepSeek-V4.1-Flash:deepinfra", 0.20, 0.60, {"temperature": 0.6, "top_p": 0.95},
                        {"reasoning_effort": "high"}),
    "dspro-plain": ("deepseek-ai/DeepSeek-V4-Pro:deepinfra", 1.30, 2.60, {"temperature": 0.6, "top_p": 0.95}, {}),
    "kimi-k25": ("moonshotai/Kimi-K2.5:deepinfra", 0.45, 2.25, {"temperature": 0.6, "top_p": 0.95}, {}),
    # OpenRouter (HF credits ran out on 2026-11-11): pinned model ids; prices from openrouter.ai/api/v1/models
    "or-dsflash-plain": ("or:deepseek/deepseek-v4-flash-0731", 0.043, 0.30, {"temperature": 0.6, "top_p": 0.95},
                         {"reasoning": {"enabled": False}}),
    "or-dsflash-think": ("or:deepseek/deepseek-v4-flash-0731", 0.043, 0.30, {"temperature": 0.6, "top_p": 0.95},
                         {"reasoning": {"effort": "high"}}),
    "or-ds41flash-plain": ("or:deepseek/deepseek-v4.1-flash", 0.022, 0.40, {"temperature": 0.6, "top_p": 0.95},
                           {"reasoning": {"enabled": False}}),
    "or-glm53-plain": ("or:z-ai/glm-5.3", 0.07, 1.40, {"temperature": 0.6, "top_p": 0.95},
                       {"reasoning": {"enabled": False}}),
}  # fmt: skip
_CLIENTS: dict = {}


def client_for(route: str):
    """HF router for plain routes; OpenRouter for `or:` routes (key read from .env here, never printed)."""
    if not route.startswith("or:"):
        return fm_v3.client(), route
    if "or" not in _CLIENTS:
        from dotenv import dotenv_values
        from openai import OpenAI

        env = next(q / ".env" for q in fm.HERE.parents if (q / ".env").is_file() or q.name == "worktrees")
        if not env.is_file():
            env = next(q.parent.parent / ".env" for q in fm.HERE.parents if q.name == "worktrees")
        _CLIENTS["or"] = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=dotenv_values(env)["OPENROUTER_API_KEY"],
                                timeout=900, max_retries=4)  # fmt: skip
    return _CLIENTS["or"], route[3:]


def openrouter_balance() -> float:
    import urllib.request

    c, _ = client_for("or:x")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {c.api_key}"}
    )
    d = json.load(urllib.request.urlopen(req))["data"]
    return round(d["total_credits"] - d["total_usage"], 4)


def path(split: str, cand: str, kind: str):
    return RUNS / f"{split.replace(':', '-')}__v4-{cand}-{kind}.jsonl"


def load(split: str, cand: str, kind: str) -> dict[str, list[dict]]:
    p = path(split, cand, kind)
    out: dict[str, list[dict]] = {}
    if p.exists():
        for r in map(json.loads, p.read_text().splitlines()):
            out.setdefault(r["id"], []).append(r)
    return out


def call(c, route, pin, pout, sampling, extra, budget, content, max_tokens):
    budget.check()
    stream = c.chat.completions.create(model=route, messages=[{"role": "user", "content": content}],
                                       max_tokens=max_tokens, extra_body=extra, stream=True,
                                       stream_options={"include_usage": True}, **sampling)  # fmt: skip
    text, reasoning, finish, usage, provider = [], 0, None, None, None
    for ch in stream:
        usage = ch.usage or usage
        provider = provider or (getattr(ch, "model_extra", None) or {}).get("provider")
        if ch.choices:
            d = ch.choices[0].delta
            text.append(d.content or "")
            extra_d = getattr(d, "model_extra", None) or {}
            reasoning += len(extra_d.get("reasoning_content") or extra_d.get("reasoning") or "")
            finish = ch.choices[0].finish_reason or finish
    n_in, n_out = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
    budget.add("openrouter" if provider else "hf", route, (n_in * pin + n_out * pout) / 1e6, prefill=n_in, sample=n_out)
    call.provider = provider
    return "".join(text), n_out, reasoning, finish == "length"


def sample(cand: str, kind: str, n: int, split: str, workers: int, max_tokens: int) -> None:
    import synth

    route, pin, pout, sampling, extra = CANDS[cand]
    budget = fin.Budget("finmath4or" if route.startswith("or:") else ARTICLE, f"v4-{cand}-{kind}-{split}")
    c, route = client_for(route)
    top = fm_v3.retriever() if kind == "program" else None
    have = load(split, cand, kind)
    jobs = [i for i in fm.load(split) for _ in range(max(0, n - len(have.get(i.id, []))))]
    print(f"{cand} {kind}: {len(jobs)} samples to draw on {split}", flush=True)

    def one(item):
        if kind == "reason":
            text, n_out, reas, capped = call(c, route, pin, pout, sampling, extra, budget, fm.prompt(item), max_tokens)
            pred = fin.final_answer(text)
            value = fm.parse(pred) if pred is not None else None
        else:
            tables = f"\n\nTables:\n{item.tables}" if item.tables else ""
            msg = fm_v3.POT_PROMPT.format(examples=top(item.question), tables=tables, question=item.question)
            text, n_out, reas, capped = call(c, route, pin, pout, sampling, extra, budget, msg, max_tokens)
            code = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
            value = synth.execute(code[-1]) if code else None
        ok = value is not None and fm.grade(repr(value), item.gold, item.gold_exec)
        return {"id": item.id, "value": value, "correct": ok, "tokens_out": n_out, "reasoning_chars": reas,
                "capped": capped, "provider": getattr(call, "provider", None)}  # fmt: skip

    start = time.time()
    with ThreadPoolExecutor(workers) as ex, path(split, cand, kind).open("a") as f:
        for k, fut in enumerate(as_completed([ex.submit(one, i) for i in jobs]), 1):
            try:
                f.write(json.dumps(fut.result()) + "\n")
            except SystemExit:
                raise
            except Exception as error:
                print(f"  skipped: {repr(error)[:120]}", flush=True)
                continue
            f.flush()
            if k % 100 == 0:
                budget.flush()
                print(f"  {k}/{len(jobs)} ${budget.usd:.2f} ({time.time() - start:.0f} s)", flush=True)
    budget.flush()


def vote_correct(samples: list[dict], item: fm.Item) -> bool:
    v = fm_v3.cluster_vote([s["value"] for s in samples if s["value"] is not None])
    return v is not None and fm.grade(repr(v), item.gold, item.gold_exec)


def analyze(split: str = "validation") -> dict:
    items = fm.load(split)
    out = {}
    for cand in CANDS:
        rs, ps = load(split, cand, "reason"), load(split, cand, "program")
        if not rs:
            continue
        ids = [i for i in items if len(rs.get(i.id, [])) >= 4]
        res = {"n": len(ids), "reason_mean": fin.bootstrap([fin.mean(s["correct"] for s in rs[i.id]) for i in ids])}
        caps = [s["capped"] for i in ids for s in rs[i.id]]
        res["reason_cap_rate"] = sum(caps) / len(caps)
        res["reason_tokens"] = fin.mean(s["tokens_out"] for i in ids for s in rs[i.id])
        for k in (1, 4, 8, 16):
            if all(len(rs[i.id]) >= k for i in ids):
                res[f"reason_vote{k}"] = fin.bootstrap([float(vote_correct(rs[i.id][:k], i)) for i in ids])
            half = k // 2
            if half and all(len(rs[i.id]) >= half and len(ps.get(i.id, [])) >= half for i in ids):
                res[f"mixed_vote{k}"] = fin.bootstrap([float(vote_correct(rs[i.id][:half] + ps[i.id][:half], i))
                                                       for i in ids])  # fmt: skip
        if ps and all(len(ps.get(i.id, [])) >= 4 for i in ids):
            res["program_mean"] = fin.bootstrap([fin.mean(s["correct"] for s in ps[i.id]) for i in ids])
        res["usd"] = round(sum(r["usd"] for r in map(json.loads, fin.LEDGER.read_text().splitlines())
                               if r["tag"].startswith(f"v4-{cand}-") and r["tag"].endswith(f"-{split}")), 3)  # fmt: skip
        out[cand] = res
        line = " ".join(f"{k} {v[0]:.3f}" for k, v in res.items() if isinstance(v, list))
        print(
            f"{cand:<18} n={res['n']} {line} cap {res['reason_cap_rate']:.3f} tok {res['reason_tokens']:.0f} ${res['usd']}"
        )
    (fm.RESULTS / f"v4_screen{'' if split == 'validation' else '_' + split}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--cand", choices=list(CANDS), required=True)
    s.add_argument("--kind", choices=["reason", "program"], required=True)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--split", default="validation")
    s.add_argument("--workers", type=int, default=64)
    s.add_argument("--max-tokens", type=int, default=16384)
    a_ = sub.add_parser("analyze")
    a_.add_argument("--split", default="validation")
    a = p.parse_args()
    if a.cmd == "sample":
        sample(a.cand, a.kind, a.n, a.split, a.workers, a.max_tokens)
    else:
        analyze(a.split)


if __name__ == "__main__":
    main()
