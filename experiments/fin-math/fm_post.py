"""Post-training Qwen3-8B (LoRA, Tinker) for FinanceMath, Samaya-style: graded best-of-N teacher traces -> SFT -> GRPO.

    fm_post.py traces --n 3          # 3 more teacher samples per verified problem (with the verify trace: best of 4)
    fm_post.py sft --epochs 2        # LoRA SFT on up to 2 correct teacher traces per SFT problem
    fm_post.py selftraces --n 4      # Qwen3-8B's own best-of-4 (T=0.8, unseeded) on the same SFT problems, graded
    fm_post.py sft --source self     # rejection-sampling SFT: the shortest correct own sample per problem (rft-e*)
    fm_post.py grpo --steps 20 --init sft-e2   # online GRPO on held-out RL problems; reward = the official grader

Training data is only the verified synthetic problems (`synth.py`); FinanceMath validation picks checkpoints, and
test is run once per final system with `fm_run.py --model qwen@<sampler path> --split test`. The verified problems
are split once by a hash of their id: 75% for SFT, 25% held out for GRPO, so RL sees problems SFT did not.
Single-turn, Qwen's non-thinking mode, the benchmark prompt (`fm.prompt`), so `fm_run.py` scores the trained model
exactly as it scored the base model. Checkpoint paths and histories go to results/posttrain/; traces to data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import fm  # noqa: I001  (sets up the fin-llm import path first)
import backends
import fin
import teacher

OUT = fm.RESULTS / "posttrain"
LOG = OUT / "checkpoints.json"
TRACES = fm.HERE / "data" / "teacher_bon.jsonl"
SELF = fm.HERE / "data" / "qwen_bon.jsonl"
TEACHER = "teacher:gpt-oss-120b"  # the same model that generated and verified the problems
RANK, LR_SFT, LR_RL, BATCH = 32, 1e-4, 2e-5, 32
TTL = 30 * 24 * 3600


def checkpoints() -> dict:
    return json.loads(LOG.read_text()) if LOG.exists() else {}


def save_checkpoint(name: str, **info) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = checkpoints()
    log[name] = info
    LOG.write_text(json.dumps(log, indent=1))


def is_rl(pid: str) -> bool:
    return int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 4 == 0


def verified() -> list[dict]:
    return [json.loads(line) for line in fm.SYNTH.read_text().splitlines()]


def item(r: dict) -> fm.Item:
    return fm.Item(r["id"], r["question"], r["tables"], float(r["gold"]), r["topic"])


# --- 1. graded best-of-N teacher traces ---------------------------------------------------------------------------


def traces(n: int, workers: int) -> None:
    """`n` more teacher samples (T=1.0, unseeded) per verified SFT problem, each graded against the executed gold."""
    budget = fin.Budget(fm.ARTICLE, "teacher-bon")
    model = teacher.make(TEACHER, budget)
    done = {json.loads(line)["id"] for line in TRACES.read_text().splitlines()} if TRACES.exists() else set()
    todo = [r for r in verified() if not is_rl(r["id"]) and r["id"] not in done]

    def one(r):
        replies = model.chat([{"role": "user", "content": fm.prompt(item(r))}], max_tokens=8192, temperature=1.0, n=n)
        return r, [{"text": x.text, "correct": fm.grade(fin.final_answer(x.text), r["gold"]), "chars": len(x.text)}
                   for x in replies]  # fmt: skip

    start = time.time()
    with ThreadPoolExecutor(workers) as pool, TRACES.open("a") as out:
        for k, (r, samples) in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps({"id": r["id"], "samples": samples}) + "\n")
            if k % 100 == 0:
                out.flush()
                budget.flush()
                print(f"  traces {k}/{len(todo)} ${budget.usd:.2f} ({time.time() - start:.0f} s)", flush=True)
    budget.flush()
    rows = [json.loads(line) for line in TRACES.read_text().splitlines()]
    summary = {"problems": len(rows), "extra_samples": n,
               "pass@1_extra": round(fin.mean(s["correct"] for r in rows for s in r["samples"]), 4),
               "all_extra_correct": round(fin.mean(all(s["correct"] for s in r["samples"]) for r in rows), 4)}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


def selftraces(n: int, workers: int) -> None:
    """`n` samples of base Qwen3-8B (T=0.8, unseeded) per SFT problem, graded: the rejection-sampling data."""
    budget = fin.Budget(fm.ARTICLE, "qwen-bon")
    model = backends.Tinker(budget)
    done = {json.loads(line)["id"] for line in SELF.read_text().splitlines()} if SELF.exists() else set()
    todo = [r for r in verified() if not is_rl(r["id"]) and r["id"] not in done]

    def one(r):
        replies = model.chat([{"role": "user", "content": fm.prompt(item(r))}], max_tokens=2048, temperature=0.8, n=n)
        return r, [{"text": x.text, "correct": fm.grade(fin.final_answer(x.text), r["gold"]), "tokens": x.tokens_out}
                   for x in replies]  # fmt: skip

    with ThreadPoolExecutor(workers) as pool, SELF.open("a") as out:
        for k, (r, samples) in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps({"id": r["id"], "samples": samples}) + "\n")
            if k % 200 == 0:
                out.flush()
                budget.flush()
                print(f"  selftraces {k}/{len(todo)} ${budget.usd:.2f}", flush=True)
    budget.flush()
    rows = [json.loads(line) for line in SELF.read_text().splitlines()]
    summary = {"problems": len(rows), "n": n,
               "pass@1": round(fin.mean(s["correct"] for r in rows for s in r["samples"]), 4),
               f"pass@{n}": round(fin.mean(any(s["correct"] for s in r["samples"]) for r in rows), 4)}  # fmt: skip
    (OUT / "selftraces_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


def self_examples() -> list[tuple[list[int], int]]:
    """The shortest correct own sample per SFT problem (problems Qwen never solved are left out)."""
    rows = {r["id"]: r["samples"] for r in map(json.loads, SELF.read_text().splitlines())}
    tok, end = backends.tokenizer(), backends.stop_ids()[0]
    out = []
    for r in verified():
        ok = [s for s in rows.get(r["id"], []) if s["correct"]]
        if is_rl(r["id"]) or not ok:
            continue
        text = min(ok, key=lambda s: s["tokens"])["text"].strip()
        prompt = backends.chat_ids([{"role": "user", "content": fm.prompt(item(r))}], thinking=False)
        out.append((prompt + tok.encode(text, add_special_tokens=False) + [end], len(prompt)))
    return out


# --- 2. SFT -------------------------------------------------------------------------------------------------------


def sft_examples(per_problem: int = 2) -> list[tuple[list[int], int]]:
    """Up to `per_problem` distinct correct traces per SFT problem: the verify trace and the best-of-N samples,
    shortest first. Every one ends with the `ANSWER:` line the grader reads."""
    extra = {}
    if TRACES.exists():
        extra = {r["id"]: r["samples"] for r in map(json.loads, TRACES.read_text().splitlines())}
    tok, end = backends.tokenizer(), backends.stop_ids()[0]
    out = []
    for r in verified():
        if is_rl(r["id"]):
            continue
        texts = [r["teacher_text"]] + [s["text"] for s in extra.get(r["id"], []) if s["correct"]]
        texts = sorted({t.strip() for t in texts if fin.final_answer(t)}, key=len)[:per_problem]
        prompt = backends.chat_ids([{"role": "user", "content": fm.prompt(item(r))}], thinking=False)
        out += [(prompt + tok.encode(t, add_special_tokens=False) + [end], len(prompt)) for t in texts]
    return out


def ce_datum(tokens: list[int], n_prompt: int, weight: float):
    import tinker

    weights = np.zeros(len(tokens) - 1, dtype=np.float32)
    weights[n_prompt - 1 :] = weight
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tinker.TensorData.from_numpy(np.asarray(tokens[1:], dtype=np.int64)),
                        "weights": tinker.TensorData.from_numpy(weights)},
    )  # fmt: skip


def service():
    fin.load_env()
    import tinker

    return tinker.ServiceClient()


def sft(epochs: int, per_problem: int, source: str) -> None:
    """`source` teacher: checkpoints sft-e*; self (rejection sampling on Qwen's own samples): rft-e*."""
    import tinker

    prefix = "sft" if source == "teacher" else "rft"
    budget = fin.Budget(fm.ARTICLE, prefix)
    data = sft_examples(per_problem) if source == "teacher" else self_examples()
    print(f"SFT on {len(data)} traces, {sum(len(t) for t, _ in data):,} tokens per epoch", flush=True)
    client = service().create_lora_training_client(base_model=backends.QWEN, rank=RANK, seed=0)
    per_epoch = (len(data) + BATCH - 1) // BATCH
    total, step, history = epochs * per_epoch, 0, []
    for epoch in range(epochs):
        order = list(range(len(data)))
        random.Random(epoch).shuffle(order)
        for b in range(per_epoch):
            budget.check()
            rows = [data[i] for i in order[b * BATCH : (b + 1) * BATCH]]
            n_target = sum(len(t) - p for t, p in rows)
            lr = LR_SFT * min(1.0, (step + 1) / 5) * max(0.05, 1 - step / total)
            fb = client.forward_backward([ce_datum(t, p, 1.0 / n_target) for t, p in rows], "cross_entropy")
            client.optim_step(tinker.AdamParams(learning_rate=lr, grad_clip_norm=1.0)).result()
            loss = float((fb.result().metrics or {}).get("loss:sum", float("nan")))
            n_tok = sum(len(t) for t, _ in rows)
            budget.add("tinker", "qwen3-8b-train", n_tok * backends.TINKER_PRICE["train"] / 1e6, train=n_tok)
            history.append(round(loss, 4))
            step += 1
            if step % 10 == 0:
                print(f"  sft epoch {epoch + 1} step {step}/{total} loss {np.mean(history[-10:]):.4f} "
                      f"${budget.usd:.2f}", flush=True)  # fmt: skip
        name = f"{prefix}-e{epoch + 1}"
        sampler = client.save_weights_for_sampler(name, ttl_seconds=TTL).result().path
        state = client.save_state(name + "-state", ttl_seconds=TTL).result().path
        save_checkpoint(name, sampler=str(sampler), state=str(state), examples=len(data), steps=step, lr=LR_SFT,
                        rank=RANK, batch=BATCH, per_problem=per_problem, source=source,
                        losses=history)  # fmt: skip
        budget.flush()
        print(f"  saved {name}: {sampler}", flush=True)


# --- 3. GRPO ------------------------------------------------------------------------------------------------------


def is_datum(prompt: list[int], completion: list[int], logprobs: list[float], advantage: float):
    import tinker

    tokens = prompt + completion
    n_prompt = len(prompt)
    mask = np.array([0.0] * (n_prompt - 1) + [1.0] * len(completion), dtype=np.float32)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.asarray(tokens[1:], dtype=np.int64)),
            "logprobs": tinker.TensorData.from_numpy(np.asarray([0.0] * (n_prompt - 1) + logprobs, dtype=np.float32)),
            "advantages": tinker.TensorData.from_numpy(mask * advantage),
        },
    )


def grpo(steps: int, init: str, questions: int, group: int, save_every: int, workers: int, lr: float) -> None:
    """Online GRPO on the held-out RL problems: per step `questions` x `group` samples at T=1 (unseeded); reward 1
    when the official grader accepts the answer. Advantage = reward - group mean; groups with no spread are dropped."""
    import tinker
    from tinker import types

    budget = fin.Budget(fm.ARTICLE, f"grpo-{init}")
    svc = service()
    if init == "base":
        client = svc.create_lora_training_client(base_model=backends.QWEN, rank=RANK, seed=0)
    else:
        client = svc.create_training_client_from_state(checkpoints()[init]["state"])
    pool = [r for r in verified() if is_rl(r["id"])]
    random.Random(7).shuffle(pool)
    print(f"GRPO on {len(pool)} held-out problems, {steps} steps x {questions} x {group}", flush=True)
    stops = backends.stop_ids()
    history = []
    for step in range(steps):
        budget.check()
        sampler = client.save_weights_and_get_sampling_client(name=f"grpo-{init}-s{step}")
        batch = [pool[(step * questions + j) % len(pool)] for j in range(questions)]

        def roll(r, sampler=sampler):
            prompt = backends.chat_ids([{"role": "user", "content": fm.prompt(item(r))}], thinking=False)
            params = types.SamplingParams(max_tokens=2048, temperature=1.0, stop=stops)  # unseeded on purpose
            seqs = sampler.sample(types.ModelInput.from_ints(prompt), group, params).result().sequences
            outs = []
            for s in seqs:
                toks = list(s.tokens)
                text = backends.tokenizer().decode([t for t in toks if t not in stops], skip_special_tokens=True)
                reward = float(fm.grade(fin.final_answer(text), r["gold"]))
                outs.append((toks, list(s.logprobs or [0.0] * len(toks)), reward))
            n_out = sum(len(t) for t, _, _ in outs)
            usd = (len(prompt) * backends.TINKER_PRICE["prefill"] + n_out * backends.TINKER_PRICE["sample"]) / 1e6
            budget.add("tinker", "qwen3-8b-grpo-sample", usd, prefill=len(prompt), sample=n_out)
            return prompt, outs

        with ThreadPoolExecutor(workers) as ex:
            rolled = list(ex.map(roll, batch))
        data, rewards, kept, lengths = [], [], 0, []
        for prompt, outs in rolled:
            r = [o[2] for o in outs]
            rewards += r
            lengths += [len(o[0]) for o in outs]
            if max(r) == min(r):
                continue
            kept += 1
            m = sum(r) / len(r)
            data += [is_datum(prompt, t, lp, rew - m) for t, lp, rew in outs]
        if data:
            client.forward_backward(data, "importance_sampling").result()
            client.optim_step(tinker.AdamParams(learning_rate=lr, grad_clip_norm=1.0)).result()
            n_tok = sum(d.model_input.length for d in data)
            budget.add("tinker", "qwen3-8b-train", n_tok * backends.TINKER_PRICE["train"] / 1e6, train=n_tok)
        rec = {"step": step + 1, "reward": round(float(np.mean(rewards)), 4), "groups_with_signal": kept,
               "mean_tokens": round(float(np.mean(lengths))), "usd": round(budget.usd, 3)}  # fmt: skip
        history.append(rec)
        print(json.dumps(rec), flush=True)
        budget.flush()
        if (step + 1) % save_every == 0 or step + 1 == steps:
            name = f"grpo-{init}-s{step + 1}"
            path = client.save_weights_for_sampler(name, ttl_seconds=TTL).result().path
            save_checkpoint(name, sampler=str(path), init=init, steps=step + 1, lr=lr, questions=questions,
                            group=group, history=history)  # fmt: skip
            print(f"  saved {name}: {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("traces")
    t.add_argument("--n", type=int, default=3)
    t.add_argument("--workers", type=int, default=32)
    s = sub.add_parser("sft")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--per-problem", type=int, default=2)
    s.add_argument("--source", choices=["teacher", "self"], default="teacher")
    st = sub.add_parser("selftraces")
    st.add_argument("--n", type=int, default=4)
    st.add_argument("--workers", type=int, default=32)
    g = sub.add_parser("grpo")
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--init", default="sft-e2")
    g.add_argument("--questions", type=int, default=32)
    g.add_argument("--group", type=int, default=8)
    g.add_argument("--save-every", type=int, default=5)
    g.add_argument("--workers", type=int, default=32)
    g.add_argument("--lr", type=float, default=LR_RL)
    a = p.parse_args()
    if a.cmd == "traces":
        traces(a.n, a.workers)
    elif a.cmd == "selftraces":
        selftraces(a.n, a.workers)
    elif a.cmd == "sft":
        sft(a.epochs, a.per_problem, a.source)
    else:
        grpo(a.steps, a.init, a.questions, a.group, a.save_every, a.workers, a.lr)


if __name__ == "__main__":
    main()
