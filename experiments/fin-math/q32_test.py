"""The single test look of the selected 32B GRPO checkpoint (PREREG docs/plans/2026-11-14-fin-math-v2-32b-grpo-test-prereg.md):
Qwen3-32B + the step-20 LoRA, thinking, vote of 4, on the 1,000 FinanceMath test questions in 4 resumable shards (one
H200 each); grade, then compare paired with gpt-5.4 as run, DeepSeek-V4-Flash thinking and episode 2's 8B model.
Writes results/runs/test__q32-grpo-s20-think.jsonl and results/q32_test.json.

    MODAL_PROFILE=albertlieyingadrian python experiments/fin-math/q32_test.py          # run (or resume) and report
    python experiments/fin-math/q32_test.py --report                                   # report from the graded file
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fm  # sets up the fin-llm import path first

# isort: split
import fin

import fm_v3
import fm_v4
from q32_select import f, sign_test

TAG, ADAPTER, SHARDS = "q32-grpo-s20-think", "/vol/adapters/q32-grpo-s20", 4
THINK = {"n": 4, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 16384}
R = fm.RESULTS / "runs"
OUT = R / f"test__{TAG}.jsonl"


def run(items) -> list | None:
    """Spawn the 4 shards; a shard that fails (the 75-minute timeout) resumes once from the volume."""
    import modal

    gen = modal.Function.from_name("fin-math-32b", "generate")
    prompts = [fm.prompt(i) for i in items]
    size = -(-len(prompts) // SHARDS)
    parts = [prompts[k * size : (k + 1) * size] for k in range(SHARDS)]
    tags = [f"{TAG}-test-k{k}" for k in range(SHARDS)]

    def spawn(k):
        return gen.spawn(parts[k], tags[k], ADAPTER, True, THINK, chunk=50)

    calls = {k: spawn(k) for k in range(SHARDS)}
    print("spawned", {k: c.object_id for k, c in calls.items()}, flush=True)
    out = {}
    for attempt in range(2):
        for k, c in calls.items():
            while True:
                try:
                    out[k] = c.get(timeout=600)
                    break
                except TimeoutError:
                    continue
                except Exception as error:
                    if any(s in repr(error) for s in ("Connection", "Deadline")):
                        time.sleep(60)
                        continue
                    print("shard", k, "attempt", attempt, "ERROR", repr(error)[:300], flush=True)
                    break
        if len(out) == SHARDS or attempt == 1:
            break
        calls = {k: spawn(k) for k in range(SHARDS) if k not in out}
        print("resuming", {k: c.object_id for k, c in calls.items()}, flush=True)
    if len(out) < SHARDS:
        print("STOPPED: shards finished", sorted(out), flush=True)
        return None
    return [q for k in range(SHARDS) for q in out[k]]


def grade(items, res) -> None:
    """The validation answer reader (modal_think.grade): the answer after </think>, the official grader."""
    with OUT.open("w") as f:
        for item, samples in zip(items, res, strict=True):
            for s in samples:
                text = s["text"]
                pred = fin.final_answer(text.split("</think>")[-1] if "</think>" in text else "")
                value = fm.parse(pred) if pred is not None else None
                ok = value is not None and fm.grade(repr(value), item.gold, item.gold_exec)
                f.write(json.dumps({"id": item.id, "topic": item.topic, "pred": pred, "correct": ok,
                                    "tokens_out": s["tokens_out"], "capped": s["capped"], "text": text}) + "\n")  # fmt: skip
    print("graded ->", OUT, flush=True)


def rows(name):
    return {r["id"]: r for r in map(json.loads, (R / f"{name}.jsonl").read_text().splitlines())}


def report(items) -> None:
    by = {i.id: i for i in items}
    d = collections.defaultdict(list)
    for r in map(json.loads, OUT.read_text().splitlines()):
        d[r["id"]].append(r)
    ids = sorted(i for i in by if len(d[i]) == 4)

    def vote(ss, i):
        vals = [fm.parse(s["pred"]) if s["pred"] else None for s in ss]
        v = fm_v3.cluster_vote([x for x in vals if x is not None])
        return float(v is not None and fm.grade(repr(v), by[i].gold, by[i].gold_exec))

    v = [vote(d[i], i) for i in ids]
    m = [fin.mean(s["correct"] for s in d[i]) for i in ids]
    g = rows("test__gpt-5.4")
    g54 = [float(g[i]["correct"]) for i in ids]
    ds = fm_v4.load("test", "or-dsflash-think", "reason")
    dsv = [float(fm_v4.vote_correct(ds[i][:4], by[i])) for i in ids]
    eight = rows("test__p4-grpo-s10")
    e8 = [float(eight[i]["majority_correct"]) for i in ids]
    ss = [s for i in ids for s in d[i]]
    out = {"n": len(ids), "vote4": fin.bootstrap(v), "mean_per_answer": fin.bootstrap(m),
           "vote4_vs_gpt-5.4": fin.paired_bootstrap(g54, v), "sign_vs_gpt-5.4": sign_test(g54, v),
           "mean_vs_gpt-5.4": fin.paired_bootstrap(g54, m), "vote4_vs_deepseek_think": fin.paired_bootstrap(dsv, v),
           "sign_vs_deepseek_think": sign_test(dsv, v), "vote4_vs_8b_majority": fin.paired_bootstrap(e8, v),
           "capped": sum(s["capped"] for s in ss), "cap_rate": sum(s["capped"] for s in ss) / len(ss),
           "tokens_per_answer": round(fin.mean(s["tokens_out"] for s in ss)),
           "gpt-5.4": fin.mean(g54), "deepseek_think_vote4": fin.mean(dsv), "8b_majority": fin.mean(e8)}  # fmt: skip
    lo, hi = out["vote4_vs_gpt-5.4"][1:]
    out["verdict"] = "beats" if lo > 0 else ("below" if hi < 0 else "reaches")
    topics = collections.defaultdict(list)
    for i, x, y in zip(ids, v, g54, strict=True):
        topics[by[i].topic].append((x, y))
    out["by_topic"] = {t: {"n": len(p), "vote4": fin.mean(a for a, _ in p), "gpt-5.4": fin.mean(b for _, b in p)}
                       for t, p in sorted(topics.items())}  # fmt: skip
    (fm.RESULTS / "q32_test.json").write_text(json.dumps(out, indent=1))
    print(f"TEST n={out['n']}: vote of 4 {fin.fmt_ci(out['vote4'])} -> {out['verdict'].upper()} gpt-5.4 "
          f"{f(out['vote4_vs_gpt-5.4'])} sign {out['sign_vs_gpt-5.4']}")  # fmt: skip
    print(f"  mean per answer {fin.fmt_ci(out['mean_per_answer'])}, vs gpt-5.4 {f(out['mean_vs_gpt-5.4'])}")
    print(f"  vs DeepSeek thinking {f(out['vote4_vs_deepseek_think'])}; vs 8B {f(out['vote4_vs_8b_majority'])}")
    print(f"  capped {out['capped']} ({out['cap_rate']:.3f}), tokens per answer {out['tokens_per_answer']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    items = fm.load("test")
    if not a.report:
        res = run(items)
        if res is None:
            return
        grade(items, res)
    report(items)


if __name__ == "__main__":
    main()
