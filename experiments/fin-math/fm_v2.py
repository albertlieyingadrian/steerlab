"""FinanceMath rerun: hard problems and the teacher's full reasoning, then a $10 SFT gate before any GRPO.

    fm_v2.py pool        # hard problems: the first pilot's SFT problems base Qwen got at most 2 of 4 right, plus new
                         # benchmark-shaped vignettes (synth.py generate --style vignette) screened the same way
    fm_v2.py traces      # 2 gpt-oss-120b samples per hard SFT problem on Tinker, keeping the analysis channel; graded
    fm_v2.py sft         # LoRA SFT on verified traces written as reasoning + answer (visible, non-thinking mode)

Why: in the first pilot SFT trained on gpt-oss-120b's final answers only (278 of 972 tokens), and Qwen learned to skip
the work (-12.7 points); and base Qwen already solved 57% of the training problems 4 of 4. Here every trace keeps the
analysis channel, which Tinker returns with the tokens, and every problem is one base Qwen finds hard.
The gate: the SFT checkpoint must gain at least 5 points over base Qwen3-8B on the 200 validation questions (paired),
scored with `fm_run.py --model qwen@<path> --split validation --max-tokens 6144`, before GRPO or any test run.
Hard problems are split by a hash of the id: 80% for SFT, 20% held for GRPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import fm  # sets up the fin-llm import path first

# isort: split
import backends
import fin

import fm_post
import synth
import teacher

DATA = fm.HERE / "data"
SCREEN, HARD, TRACES = DATA / "v2_screen.jsonl", DATA / "v2_hard.jsonl", DATA / "v2_traces.jsonl"
MAX_RIGHT = 2  # a problem is hard when base Qwen gets at most this many of 4 samples right


def split_of(pid: str) -> str:
    return "rl" if int(hashlib.sha256(("v2" + pid).encode()).hexdigest(), 16) % 5 == 0 else "sft"


def item(r: dict) -> fm.Item:
    return fm.Item(r["id"], r["question"], r["tables"], float(r["gold"]), r["topic"])


# --- 1. hard problems ---------------------------------------------------------------------------------------------


def pool(workers: int) -> None:
    """Screen new vignettes with base Qwen3-8B (4 samples, T=0.8, unseeded); add the first pilot's hard SFT problems."""
    budget = fin.Budget(fm.ARTICLE, "v2-screen")
    qwen = backends.Tinker(budget)
    raw = [json.loads(line) for line in (DATA / "synth_raw_vignette.jsonl").read_text().splitlines()]
    rows, dropped = synth.decontaminate([r for r in raw if r["gold"] is not None])
    print(f"{len(raw)} vignettes, {len(rows) + dropped} executable, {dropped} too close to a benchmark question")
    done = {json.loads(line)["id"] for line in SCREEN.read_text().splitlines()} if SCREEN.exists() else set()

    def one(r):
        replies = qwen.chat([{"role": "user", "content": fm.prompt(item(r))}], max_tokens=2048, temperature=0.8, n=4)
        return r, sum(fm.grade(fin.final_answer(x.text), r["gold"]) for x in replies)

    with ThreadPoolExecutor(workers) as ex, SCREEN.open("a") as out:
        for k, (r, right) in enumerate(ex.map(one, [r for r in rows if r["id"] not in done]), 1):
            out.write(json.dumps({"id": r["id"], "source": "vignette", "qwen_right": right}) + "\n")
            if k % 100 == 0:
                out.flush()
                budget.flush()
                print(f"  screened {k} ${budget.usd:.2f}", flush=True)
    budget.flush()
    screened = {d["id"]: d["qwen_right"] for d in map(json.loads, SCREEN.read_text().splitlines())}
    hard = [{**r, "source": "vignette", "qwen_right": screened[r["id"]]} for r in rows
            if screened.get(r["id"], 99) <= MAX_RIGHT]  # fmt: skip
    first = {r["id"]: r for r in fm_post.verified()}
    for d in map(json.loads, fm_post.SELF.read_text().splitlines()):
        right = sum(s["correct"] for s in d["samples"])
        if right <= MAX_RIGHT and d["id"] in first:
            r = first[d["id"]]
            hard.append({k: r[k] for k in ("id", "topic", "subtopic", "question", "tables", "solution", "gold")}
                        | {"source": "drill", "qwen_right": right})  # fmt: skip
    for r in hard:
        r["split"] = split_of(r["id"])
    HARD.write_text("".join(json.dumps(r) + "\n" for r in hard))
    by = {f"{s}/{sp}": sum(r["source"] == s and r["split"] == sp for r in hard)
          for s in ("vignette", "drill") for sp in ("sft", "rl")}  # fmt: skip
    vign = list(screened.values())
    summary = {"vignettes_screened": len(vign), "vignette_right_of_4": {str(k): vign.count(k) for k in range(5)},
               "hard": len(hard), "by_source_split": by}  # fmt: skip
    (fm.RESULTS / "v2_pool.json").write_text(json.dumps(summary, indent=1))
    print(summary)


# --- 2. traces with the full reasoning ----------------------------------------------------------------------------


def traces(n: int, workers: int, spec: str, retry_missing: bool = False) -> None:
    """`n` samples per hard SFT problem not yet sampled; with `retry_missing`, `n` more for each sampled problem that
    has no right trace yet (k continues from the samples already drawn)."""
    budget = fin.Budget(fm.ARTICLE, "v2-traces" if spec.startswith("teacher:") else "v2-traces-hf")
    model = teacher.make(spec, budget)
    rows = [r for r in map(json.loads, HARD.read_text().splitlines()) if r["split"] == "sft"]
    have: dict[str, list[bool]] = {}
    if TRACES.exists():
        for t in map(json.loads, TRACES.read_text().splitlines()):
            have.setdefault(t["id"], []).append(t["correct"])
    if retry_missing:
        todo = [r for r in rows if r["id"] in have and not any(have[r["id"]])]
    else:
        todo = [r for r in rows if r["id"] not in have]
    print(f"{len(rows)} hard SFT problems, {len(todo)} to go", flush=True)

    def one(r):
        try:
            return r, model.chat([{"role": "user", "content": fm.prompt(item(r))}], max_tokens=8192, temperature=1.0,
                                 n=n)  # fmt: skip
        except SystemExit:
            raise
        except Exception as error:
            print(f"  skipped {r['id']}: {repr(error)[:120]}", flush=True)
            return r, None

    start, failed = time.time(), 0
    with ThreadPoolExecutor(workers) as ex, TRACES.open("a") as out:
        futures = [ex.submit(one, r) for r in todo]
        for j, f in enumerate(as_completed(futures), 1):  # written as each problem finishes, so a crash loses little
            r, replies = f.result()
            if replies is None:
                failed += 1
                continue
            for k, x in enumerate(replies, start=len(have.get(r["id"], []))):
                pred = fin.final_answer(x.text)
                out.write(json.dumps({"id": r["id"], "k": k, "analysis": x.analysis, "final": x.text, "pred": pred,
                                      "correct": fm.grade(pred, r["gold"]), "tokens": x.tokens_out}) + "\n")  # fmt: skip
            out.flush()
            if j % 50 == 0:
                budget.flush()
                print(f"  traces {j}/{len(todo)} ${budget.usd:.2f} ({time.time() - start:.0f} s, {failed} skipped)",
                      flush=True)  # fmt: skip
    budget.flush()
    got = [json.loads(line) for line in TRACES.read_text().splitlines()]
    per = {}
    for g in got:
        per.setdefault(g["id"], []).append(g["correct"])
    summary = {"problems": len(per), "samples": len(got), "right": round(fin.mean(g["correct"] for g in got), 4),
               "problems_with_a_right_trace": sum(any(v) for v in per.values()),
               "mean_tokens": round(fin.mean(g["tokens"] for g in got)),
               "mean_analysis_chars": round(fin.mean(len(g["analysis"]) for g in got)),
               "mean_final_chars": round(fin.mean(len(g["final"]) for g in got))}  # fmt: skip
    (fm.RESULTS / "v2_traces_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


# --- 3. SFT gate --------------------------------------------------------------------------------------------------


def trace_text(t: dict) -> str:
    """What the student learns to write: the teacher's reasoning, then its answer, both visible."""
    return f"{t['analysis'].strip()}\n\n{t['final'].strip()}"


def examples(per_problem: int) -> list[tuple[list[int], int]]:
    rows = {r["id"]: r for r in map(json.loads, HARD.read_text().splitlines())}
    good = {}
    for t in map(json.loads, TRACES.read_text().splitlines()):
        if t["correct"] and t["analysis"] and fin.final_answer(t["final"]):
            good.setdefault(t["id"], []).append(t)
    tok, end = backends.tokenizer(), backends.stop_ids()[0]
    out = []
    for pid, ts in sorted(good.items()):
        prompt = backends.chat_ids([{"role": "user", "content": fm.prompt(item(rows[pid]))}], thinking=False)
        for t in sorted(ts, key=lambda t: t["tokens"])[:per_problem]:
            out.append((prompt + tok.encode(trace_text(t), add_special_tokens=False) + [end], len(prompt)))
    return out


def sft(epochs: int, per_problem: int, lr: float) -> None:
    import tinker

    budget = fin.Budget(fm.ARTICLE, "v2-sft")
    data = examples(per_problem)
    print(f"SFT on {len(data)} traces, {sum(len(t) for t, _ in data):,} tokens per epoch", flush=True)
    client = fm_post.service().create_lora_training_client(base_model=backends.QWEN, rank=fm_post.RANK, seed=0)
    batch = fm_post.BATCH
    per_epoch = (len(data) + batch - 1) // batch
    total, step, history = epochs * per_epoch, 0, []
    for epoch in range(epochs):
        order = list(range(len(data)))
        random.Random(epoch).shuffle(order)
        for b in range(per_epoch):
            budget.check()
            rows = [data[i] for i in order[b * batch : (b + 1) * batch]]
            n_target = sum(len(t) - p for t, p in rows)
            rate = lr * min(1.0, (step + 1) / 5) * max(0.05, 1 - step / total)
            fb = client.forward_backward([fm_post.ce_datum(t, p, 1.0 / n_target) for t, p in rows], "cross_entropy")
            client.optim_step(tinker.AdamParams(learning_rate=rate, grad_clip_norm=1.0)).result()
            loss = float((fb.result().metrics or {}).get("loss:sum", float("nan")))
            n_tok = sum(len(t) for t, _ in rows)
            budget.add("tinker", "qwen3-8b-train", n_tok * backends.TINKER_PRICE["train"] / 1e6, train=n_tok)
            history.append(round(loss, 4))
            step += 1
            if step % 10 == 0:
                print(f"  sft epoch {epoch + 1} step {step}/{total} loss {np.mean(history[-10:]):.4f} ${budget.usd:.2f}",
                      flush=True)  # fmt: skip
        name = f"v2-sft-e{epoch + 1}"
        sampler = client.save_weights_for_sampler(name, ttl_seconds=fm_post.TTL).result().path
        state = client.save_state(name + "-state", ttl_seconds=fm_post.TTL).result().path
        fm_post.save_checkpoint(name, sampler=str(sampler), state=str(state), examples=len(data), steps=step, lr=lr,
                                rank=fm_post.RANK, batch=batch, per_problem=per_problem, source="v2-full-reasoning",
                                losses=history)  # fmt: skip
        budget.flush()
        print(f"  saved {name}: {sampler}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("pool")
    po.add_argument("--workers", type=int, default=32)
    t = sub.add_parser("traces")
    t.add_argument("--n", type=int, default=2)
    t.add_argument("--workers", type=int, default=32)
    t.add_argument("--teacher", default="teacher:gpt-oss-120b", help="or hf:openai/gpt-oss-120b:deepinfra")
    t.add_argument("--retry-missing", action="store_true")
    s = sub.add_parser("sft")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--per-problem", type=int, default=2)
    s.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()
    if a.cmd == "pool":
        pool(a.workers)
    elif a.cmd == "traces":
        traces(a.n, a.workers, a.teacher, a.retry_missing)
    else:
        sft(a.epochs, a.per_problem, a.lr)


if __name__ == "__main__":
    main()
