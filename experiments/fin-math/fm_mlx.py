"""FinanceMath rerun on this Mac with MLX ($0 compute): rejection-sampled traces, LoRA SFT, and the +5-point gate.

    python3 fm_mlx.py sample --model Qwen/Qwen3-14B --n 4      # samples on the hard SFT problems, graded
    python3 fm_mlx.py build                                    # data/mlx_sft/{train,valid}.jsonl from verified traces
    python3 fm_mlx.py train --name v2-mlx-sft                  # mlx_lm LoRA on Qwen3-8B (results/mlx/<name>/)
    python3 fm_mlx.py eval --tag mlx-base                      # base Qwen3-8B, greedy, on the 200 validation questions
    python3 fm_mlx.py eval --adapter results/mlx/v2-mlx-sft --tag mlx-v2-sft

Runs with an interpreter that has mlx_lm (the pyenv Python here, not the project .venv). Training data, all from models
whose licences allow it and all checked against the executed gold: gpt-oss-120b's full-reasoning traces from Tinker
(371 hard problems before Tinker was stopped; written as reasoning + final answer), and rejection samples from Qwen3-14B
(or the student itself) in non-thinking mode on the 1,232 hard SFT problems (`fm_v2.py pool`).
The gate: the SFT adapter must beat base Qwen3-8B by at least 5 points on validation, both run here with the same
settings (greedy, non-thinking chat template, up to 6,144 new tokens), paired over the same questions.
Waits while the machine is busy (1-minute load above 0.9 per core, or under 10% memory free); one heavy job at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fm  # sets up the fin-llm import path first

# isort: split
import fin

DATA = fm.HERE / "data"
HARD, TRACES = DATA / "v2_hard.jsonl", DATA / "v2_traces.jsonl"
SFT_DIR, MLX = DATA / "mlx_sft", fm.RESULTS / "mlx"
STUDENT = "Qwen/Qwen3-8B"


def busy() -> str | None:
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    if load > 0.9:
        return f"load {load:.2f} per core"
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10, check=False).stdout
        free = int(re.search(r"free percentage: (\d+)%", out).group(1))  # type: ignore[union-attr]
    except (subprocess.SubprocessError, AttributeError, ValueError):
        return None
    return f"{free}% memory free" if free < 10 else None


def wait_until_idle() -> None:
    while reason := busy():
        print(f"  paused: {reason}", flush=True)
        time.sleep(60)


def prompt_ids(tok, item: fm.Item) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": fm.prompt(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)  # fmt: skip
    return tok.encode(text, add_special_tokens=False)


def hard_rows() -> list[dict]:
    return [r for r in map(json.loads, HARD.read_text().splitlines()) if r["split"] == "sft"]


def item_of(r: dict) -> fm.Item:
    return fm.Item(r["id"], r["question"], r["tables"], float(r["gold"]), r["topic"])


# --- rejection sampling -------------------------------------------------------------------------------------------


def sample(model_id: str, n: int, temp: float, chunk: int, max_tokens: int) -> None:
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    out_path = DATA / f"v2_local_{model_id.split('/')[-1]}.jsonl"
    done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()} if out_path.exists() else set()
    todo = [r for r in hard_rows() if r["id"] not in done]
    print(f"{model_id}: {len(todo)} problems x {n} samples to go", flush=True)
    model, tok = load(model_id)
    sampler = make_sampler(temp=temp, top_p=0.8)  # Qwen3's recommended non-thinking sampling
    start, right, total = time.time(), 0, 0
    with out_path.open("a") as out:
        for b in range(0, len(todo), chunk):
            wait_until_idle()
            rows = todo[b : b + chunk]
            prompts = [prompt_ids(tok, item_of(r)) for r in rows for _ in range(n)]
            t0 = time.time()
            texts = batch_generate(model, tok, prompts, max_tokens=max_tokens, sampler=sampler).texts
            for i, r in enumerate(rows):
                for k in range(n):
                    text = texts[i * n + k]
                    pred = fin.final_answer(text)
                    ok = fm.grade(pred, r["gold"])
                    out.write(json.dumps({"id": r["id"], "model": model_id, "k": k, "text": text, "pred": pred,
                                          "correct": ok, "tokens": len(tok.encode(text))}) + "\n")  # fmt: skip
                    right, total = right + ok, total + 1
            out.flush()
            hours = (time.time() - start) / 3600
            left = hours / (b + len(rows)) * (len(todo) - b - len(rows))
            print(f"  {b + len(rows)}/{len(todo)} problems, right {right / total:.3f}, chunk {time.time() - t0:.0f} s, "
                  f"{hours:.2f} h, about {left:.1f} h to go", flush=True)  # fmt: skip


# --- training data ------------------------------------------------------------------------------------------------


def build(per_problem: int, seed: int) -> None:
    """One chat example per kept trace: the benchmark prompt, then the trace (reasoning + answer for gpt-oss)."""
    rows = {r["id"]: r for r in hard_rows()}
    kept: dict[str, list[tuple[str, str]]] = {}
    for t in map(json.loads, TRACES.read_text().splitlines()):
        if t["correct"] and t["analysis"] and fin.final_answer(t["final"]) and t["id"] in rows:
            kept.setdefault(t["id"], []).append(("gpt-oss-120b", f"{t['analysis'].strip()}\n\n{t['final'].strip()}"))
    for path in sorted(DATA.glob("v2_local_*.jsonl")):
        for s in map(json.loads, path.read_text().splitlines()):
            if s["correct"] and s["id"] in rows:
                kept.setdefault(s["id"], []).append((s["model"], s["text"].strip()))
    # One format for every example, the one the eval prompt asks for (Qwen3 non-thinking mode: the chat template adds an
    # empty think block): visible step-by-step working that ends with an `ANSWER:` line, never reasoning in a think block.
    dropped = 0
    for pid in list(kept):
        ok = [(s, t) for s, t in kept[pid] if "<think>" not in t and "</think>" not in t and fin.final_answer(t)]
        dropped += len(kept[pid]) - len(ok)
        kept[pid] = ok
        if not ok:
            del kept[pid]
    examples, sources = [], {}
    for pid, traces in sorted(kept.items()):
        seen, chosen = set(), []
        for src, text in sorted(traces, key=lambda x: (x[0] != "gpt-oss-120b", len(x[1]))):
            if text not in seen and len(chosen) < per_problem:  # gpt-oss reasoning first, then the shortest others
                seen.add(text)
                chosen.append((src, text))
        for src, text in chosen:
            sources[src] = sources.get(src, 0) + 1
            examples.append({"messages": [{"role": "user", "content": fm.prompt(item_of(rows[pid]))},
                                          {"role": "assistant", "content": text}]})  # fmt: skip
    random.Random(seed).shuffle(examples)
    n_valid = max(20, len(examples) // 20)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    (SFT_DIR / "valid.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples[:n_valid]))
    (SFT_DIR / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples[n_valid:]))
    summary = {"problems": len(kept), "examples": len(examples), "train": len(examples) - n_valid, "valid": n_valid,
               "by_source": sources, "dropped_for_format": dropped,
               "format": "non-thinking: empty think block, visible working ending in ANSWER:"}  # fmt: skip
    (fm.RESULTS / "mlx_sft_data.json").write_text(json.dumps(summary, indent=1))
    print(summary)


# --- LoRA SFT -----------------------------------------------------------------------------------------------------


def train(name: str, epochs: float, lr: float, rank: int, batch: int, accum: int) -> None:
    n_train = sum(1 for _ in (SFT_DIR / "train.jsonl").open())
    iters = max(1, round(epochs * n_train / batch))
    out = MLX / name
    out.mkdir(parents=True, exist_ok=True)
    config = {"model": STUDENT, "train": True, "data": str(SFT_DIR), "fine_tune_type": "lora", "num_layers": -1,
              "batch_size": batch, "grad_accumulation_steps": accum, "iters": iters, "learning_rate": lr,
              "mask_prompt": True, "max_seq_length": 8192, "grad_checkpoint": True, "adapter_path": str(out),
              "save_every": max(1, iters // 2), "steps_per_report": 10, "steps_per_eval": max(10, iters // 4),
              "val_batches": 10, "seed": 0, "lora_parameters": {"rank": rank, "scale": 20.0, "dropout": 0.0}}  # fmt: skip
    cfg = out / "config.yaml"
    cfg.write_text(json.dumps(config, indent=1))  # JSON is valid YAML
    print(f"training {name}: {n_train} examples, {iters} iterations of {batch} (x{accum} accumulation)", flush=True)
    wait_until_idle()
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfg)], check=True)


# --- evaluation ---------------------------------------------------------------------------------------------------


def evaluate(adapter: str | None, tag: str, split: str, chunk: int, max_tokens: int) -> None:
    from mlx_lm import batch_generate, load

    path = fm.RESULTS / "runs" / f"{split.replace(':', '-')}__{tag}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    items = [i for i in fm.load(split) if i.id not in done]
    model, tok = load(STUDENT, adapter_path=adapter)
    with path.open("a") as out:
        for b in range(0, len(items), chunk):
            wait_until_idle()
            batch = items[b : b + chunk]
            t0 = time.time()
            texts = batch_generate(model, tok, [prompt_ids(tok, i) for i in batch], max_tokens=max_tokens).texts
            sec = (time.time() - t0) / len(batch)
            for i, text in zip(batch, texts, strict=True):
                pred = fin.final_answer(text)
                out.write(json.dumps({"id": i.id, "topic": i.topic, "gold": i.gold,
                                      "correct": fm.grade(pred, i.gold, i.gold_exec), "pred": pred, "text": text,
                                      "usd": 0.0, "seconds": round(sec, 2), "tokens_in": 0,
                                      "tokens_out": len(tok.encode(text))}) + "\n")  # fmt: skip
            out.flush()
            print(f"  {tag}: {b + len(batch)}/{len(items)}", flush=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print(
        f"{tag}: acc {fin.fmt_ci(fin.bootstrap([r['correct'] for r in rows]))} on {len(rows)}; "
        f"no answer {sum(r['pred'] is None for r in rows)}; mean tokens {fin.mean(r['tokens_out'] for r in rows):.0f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", default="Qwen/Qwen3-14B")
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--temp", type=float, default=0.7)
    s.add_argument("--chunk", type=int, default=16)
    s.add_argument("--max-tokens", type=int, default=2048)
    b = sub.add_parser("build")
    b.add_argument("--per-problem", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--name", default="v2-mlx-sft")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--accum", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--adapter")
    e.add_argument("--tag", required=True)
    e.add_argument("--split", default="validation")
    e.add_argument("--chunk", type=int, default=16)
    e.add_argument("--max-tokens", type=int, default=6144)
    a = p.parse_args()
    if a.cmd == "sample":
        sample(a.model, a.n, a.temp, a.chunk, a.max_tokens)
    elif a.cmd == "build":
        build(a.per_problem, a.seed)
    elif a.cmd == "train":
        train(a.name, a.epochs, a.lr, a.rank, a.batch, a.accum)
    else:
        evaluate(a.adapter, a.tag, a.split, a.chunk, a.max_tokens)


if __name__ == "__main__":
    main()
