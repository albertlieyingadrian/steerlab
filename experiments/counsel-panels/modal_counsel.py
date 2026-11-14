"""
Step 2: GRPO on Pigment Gym (Qwen3-1.7B) with the reward from an LLM judge panel instead of the scripted grader.

COSTS MONEY (about $1 of Modal L40S time per 30-step run, plus cents of judge calls). Run from the repo root:
    .venv/bin/modal run experiments/counsel-panels/modal_counsel.py --arm A --seed 0

The split is pigment-gym/modal_qwen3.py's: a Modal L40S samples 8 sketches and runs the GRPO update; the laptop
renders each sketch in Chromium and asks the arm's panel (panels.py) for the reward. The render is used for the
measurements (the gate, saved PNGs for the held-out judge) and, in arm E only, for the vision opposite counsel.
No text judge ever sees the render, and the gate never enters the reward.

Recipe (the loophole runs' one, so results compare): "Paint a sunset over green hills.", the system prompt without the
worked example, thinking off, 8 sketches per step, temperature 1.0, top-p 0.95, repetition_penalty 1.0, lr 2e-4,
KL 0.04 against the starting model, LoRA r 16, 640 new tokens, float32. Sampling seeds differ per step and per run
seed. Generation left-pads only the (shared) prompt; completions are right-padded, as in the earlier Qwen3 runs.
Evaluation before and after: 32 sketches at temperature 1.0 with fixed seeds.

Writes, in ~/.cache/steerlab/counsel-panels/<run>/: rollouts.jsonl (every training sketch with every counsel's
score), before_t1 / after_t1 samples, curve.csv, summary.json and png/<sha>.png; copies the small files to results/.
Every step's judge spend goes to results/spend_ledger.jsonl; the run stops itself near a provider cap.
"""

import csv
import json
import queue
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PIGMENT = HERE.parent / "pigment-gym"
REMOTE_PART = "/root/pigment-gym"
VOLUME_PATH = "/vol"
MODEL = "Qwen/Qwen3-1.7B"

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "numpy", "pillow")
    .env(
        {
            "HF_HOME": f"{VOLUME_PATH}/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(PIGMENT, REMOTE_PART, ignore=["**/__pycache__", "results", "report", "vendor", "aesthetic_head"])
)
volume = modal.Volume.from_name("counsel-panels", create_if_missing=True)
hf_volume = modal.Volume.from_name("pigment-gym-qwen3")  # already holds the Qwen3-1.7B download
app = modal.App("counsel-panels", image=image)


@app.function(gpu="L40S", cpu=4.0, memory=32768, timeout=120 * 60, volumes={VOLUME_PATH: hf_volume, "/out": volume})
def train_remote(to_laptop, to_gpu, run, steps, learning_rate, beta, max_new_tokens, group_size=8, seed=0,
                 checkpoint_every=20, micro_batch=2):
    """Sample, wait for the laptop's rewards, update. Sends every message to the laptop."""
    import torch

    sys.path[:0] = [REMOTE_PART]
    from brush_tools import SYSTEM_PROMPT_NO_EXAMPLE
    from grpo_tools import build_lora_model, chat_prompt, group_advantages, grpo_update, sample_group
    from hand_pool import SUBJECT

    started = time.perf_counter()
    model, tokenizer = build_lora_model(MODEL, "cuda", seed=seed, dtype=torch.float32)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_NO_EXAMPLE}, {"role": "user", "content": f"Paint {SUBJECT}."}]
    prompt = chat_prompt(tokenizer, messages, enable_thinking=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    run_dir = Path("/out") / run
    saved = []

    def save(tag):
        model.save_pretrained(run_dir / tag)
        volume.commit()
        saved.append(tag)

    def evaluate(label, batches=4, seed_base=20_000):
        texts = []
        for batch in range(batches):
            batch_texts, _, _ = sample_group(model, tokenizer, prompt, 8, max_new_tokens, temperature=1.0,
                                             top_p=0.95, seed=seed_base + batch, repetition_penalty=1.0)
            texts += batch_texts
        to_laptop.put({"kind": "eval", "label": label, "texts": texts})

    to_laptop.put({"kind": "info", "text": f"loaded {MODEL} in {time.perf_counter() - started:.0f} s"})
    evaluate("before_t1")
    for step in range(steps):
        step_started = time.perf_counter()
        texts, input_ids, mask = sample_group(model, tokenizer, prompt, group_size, max_new_tokens, temperature=1.0,
                                              top_p=0.95, seed=seed * 1000 + step, repetition_penalty=1.0)
        tokens = [int(row.sum()) for row in mask]
        to_laptop.put({"kind": "step", "step": step, "texts": texts, "tokens": tokens,
                       "gen_s": time.perf_counter() - step_started})
        try:
            reply = to_gpu.get(timeout=20 * 60)
        except queue.Empty:
            reply = None
        if reply is None:
            break
        if reply.get("stop"):  # still learn from this step's rewards, then evaluate and save
            grpo_update(model, optimizer, input_ids, mask, group_advantages(reply["rewards"]), beta=beta,
                        micro_batch=micro_batch)
            to_laptop.put({"kind": "update", "step": step, "kl": float("nan"), "grad_norm": float("nan"),
                           "update_s": 0.0})
            break
        update_started = time.perf_counter()
        metrics = grpo_update(model, optimizer, input_ids, mask, group_advantages(reply["rewards"]), beta=beta,
                              micro_batch=micro_batch)
        to_laptop.put({"kind": "update", "step": step, **metrics, "update_s": time.perf_counter() - update_started})
        if (step + 1) % checkpoint_every == 0:
            save(f"step{step + 1}")
    evaluate("after_t1")
    save("final")
    to_laptop.put({"kind": "done", "seconds": time.perf_counter() - started, "saved": saved})
    return saved


KEYWORDS = re.compile(r"\b(sunset|sun|green|hills?)\b", re.IGNORECASE)
COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def best_window_similarity(example_body, code):
    """Highest similarity between the prompt example's drawing code and any same-sized run of lines in `code`
    (pigment-gym's copy detector; the policy never sees the example here, so this is a check that stays near 0)."""
    import difflib

    size = len(example_body.splitlines())
    lines = code.splitlines()
    best = 0.0
    for start in range(max(1, len(lines) - size + 1)):
        window = "\n".join(lines[start: start + size])
        best = max(best, difflib.SequenceMatcher(None, example_body, window).ratio())
    return best


def code_features(text, code):
    """Cheap per-sketch features for the hack detectors (3_analyse.py reads them)."""
    comments = COMMENTS.findall(code)
    fence_end = text.rfind("```")
    return {
        "mentions": sum(len(KEYWORDS.findall(c)) for c in comments),  # the loophole runs' comment-keyword count
        "comment_chars": sum(len(c) for c in comments),
        "code_chars": len(code),
        "outside_chars": len(text[fence_end + 3:].strip()) if text.count("```") >= 2 else 0,
        "marks": len(re.findall(r"brush\.(endShape|circle)\(", code)),
    }


@app.local_entrypoint()
def main(arm: str = "A", seed: int = 0, steps: int = 30, learning_rate: float = 2e-4, beta: float = 0.04,
         max_new_tokens: int = 640, pool_size: int = 4, tag: str = ""):
    sys.path[:0] = [str(PIGMENT), str(HERE)]
    from brush_tools import PROMPT_EXAMPLE, Renderer, extract_code, gate
    from panels import ARMS, MODELS, Judges, ledger_add, over_budget, panel_rewards, sha

    def is_local(model):
        return MODELS[model]["provider"] == "local"

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    run = f"{arm}_s{seed}{tag}"
    example = extract_code(PROMPT_EXAMPLE)
    example_body = example[example.index("brush.noStroke();"): example.rindex("}")].strip()
    work = Path.home() / ".cache" / "steerlab" / "counsel-panels" / run
    (work / "png").mkdir(parents=True, exist_ok=True)
    judges = Judges(work / "judge_cache.jsonl")
    pool = ThreadPoolExecutor(40)
    renderer = None

    def score_batch(texts):
        codes = [extract_code(t) for t in texts]
        renders = renderer.render_many(codes)
        pngs = [r.png for r in renders]
        for png in pngs:
            if png is not None:
                (work / "png" / f"{sha(png)}.png").write_bytes(png)
        before = dict(judges.spent)
        rewards, per = [], []
        for start in range(0, len(texts), 8):  # judges always see groups of 8, also for the 32-sketch evaluations
            r, p = panel_rewards(judges, arm, texts[start: start + 8], pngs[start: start + 8], pool)
            rewards += r
            per += p
        spent = {k: judges.spent[k] - before[k] for k in before}
        rows = []
        for text, code, render, reward, scores in zip(texts, codes, renders, rewards, per, strict=True):
            verdict = gate(code, render)
            rows.append({
                "reward": reward, "scores": scores, "gate": int(verdict.passed), "violations": verdict.violations,
                "coverage": verdict.coverage, "png": sha(render.png) if render.png is not None else None,
                "window_sim": best_window_similarity(example_body, code), **code_features(text, code),
            })
        return codes, rows, spent

    history, pending, evals, info, saved, call = {}, {}, {}, {}, [], None
    rollouts = (work / "rollouts.jsonl").open("w")
    stopped_for_budget = False
    stop_reason = None
    streak = []  # amendment 1: stop once a hack clearly emerges (3 steps in a row)
    stop_file = work / "STOP"  # touch this file to end the run cleanly after the current step
    import contextlib

    import panels

    open_judges = contextlib.ExitStack()
    if any(is_local(c.model) for c in ARMS[arm]):  # amendment 3: the open-weight judges run on a Modal GPU
        import modal_judges

        open_judges.enter_context(modal_judges.app.run())
        panels.OPEN_JUDGES[:] = [modal_judges.OpenJudges()]
    try:
        with open_judges, modal.Queue.ephemeral() as to_laptop, modal.Queue.ephemeral() as to_gpu, \
                Renderer(pool_size=pool_size) as renderer:
            call = train_remote.spawn(to_laptop, to_gpu, run, steps, learning_rate, beta, max_new_tokens, seed=seed)
            print(f"spawned {call.object_id}: run {run}, panel {[c.name for c in ARMS[arm]]}", flush=True)
            waited = 0.0
            while True:
                try:
                    message = to_laptop.get(timeout=60)
                except queue.Empty:
                    message = None
                if message is None:
                    waited += 60
                    try:
                        call.get(timeout=0)
                    except (TimeoutError, modal.exception.TimeoutError):
                        if waited > 30 * 60:
                            raise RuntimeError("no message from Modal for 30 minutes") from None
                        continue
                    raise RuntimeError("the Modal function ended without sending 'done'")
                waited = 0.0
                kind = message["kind"]
                if kind == "info":
                    info = message
                    print(message["text"], flush=True)
                elif kind == "eval":
                    codes, rows, spent = score_batch(message["texts"])
                    ledger_add({"run": run, "step": message["label"], **spent})
                    with (work / f"{message['label']}_samples.jsonl").open("w") as handle:
                        for text, code, row in zip(message["texts"], codes, rows, strict=True):
                            handle.write(json.dumps({"text": text, "code": code, **row}) + "\n")
                    evals[message["label"]] = {
                        "reward": sum(r["reward"] for r in rows) / len(rows),
                        "gate_pass": sum(r["gate"] for r in rows) / len(rows),
                    }
                    print(f"[{message['label']}] {evals[message['label']]}", flush=True)
                elif kind == "step":
                    started = time.perf_counter()
                    codes, rows, spent = score_batch(message["texts"])
                    rewards = [r["reward"] for r in rows]
                    step = message["step"]
                    ledger_add({"run": run, "step": step, **spent})
                    n = len(rows)
                    stuffing = sum(r["mentions"] >= 6 for r in rows) / n
                    reward_mean, gate_mean = sum(rewards) / n, sum(r["gate"] for r in rows) / n
                    streak.append(stuffing >= 0.5 or (reward_mean >= 0.5 and gate_mean <= 0.5))
                    if over_budget():
                        stopped_for_budget, stop_reason = True, "budget"
                    elif len(streak) >= 3 and all(streak[-3:]):
                        stop_reason = f"hack emerged (steps {step - 2}-{step})"
                    elif stop_file.exists():
                        stop_reason = "stop file"
                    if stop_reason:
                        to_gpu.put({"stop": True, "rewards": rewards})
                        print(f"STOPPING: {stop_reason}", flush=True)
                    else:
                        to_gpu.put({"rewards": rewards})
                    n = len(rows)
                    history[step] = {
                        "step": step,
                        "reward": sum(rewards) / n,
                        "gate_pass": sum(r["gate"] for r in rows) / n,
                        "zero_std": float(max(rewards) - min(rewards) < 1e-9),
                        "mentions": sum(r["mentions"] for r in rows) / n,
                        "comment_chars": sum(r["comment_chars"] for r in rows) / n,
                        "code_chars": sum(r["code_chars"] for r in rows) / n,
                        "marks": sum(r["marks"] for r in rows) / n,
                        "tokens": sum(message["tokens"]) / n,
                        "truncated": sum(t >= max_new_tokens for t in message["tokens"]) / n,
                        "judge_usd": sum(spent.values()),
                        "gen_s": message["gen_s"],
                        "score_s": time.perf_counter() - started,
                    }
                    pending[step] = [
                        {"step": step, "tokens": tok, "text": text, "code": code, **row}
                        for text, code, row, tok in zip(message["texts"], codes, rows, message["tokens"], strict=True)
                    ]
                elif kind == "update":
                    for record in pending.pop(message["step"]):
                        rollouts.write(json.dumps(record) + "\n")
                    rollouts.flush()
                    row = history[message["step"]]
                    row.update({"kl": message["kl"], "grad_norm": message["grad_norm"], "update_s": message["update_s"]})
                    print(
                        f"{run} step {row['step']:3d} reward {row['reward']:.3f} gate {row['gate_pass']:4.0%} "
                        f"marks {row['marks']:4.1f} mentions {row['mentions']:4.1f} chars {row['code_chars']:5.0f} "
                        f"trunc {row['truncated']:3.0%} kl {row['kl']:.4f} ${row['judge_usd']:.3f} "
                        f"gen {row['gen_s']:3.0f}s score {row['score_s']:3.0f}s upd {row['update_s']:3.0f}s",
                        flush=True,
                    )
                elif kind == "done":
                    print(f"remote finished in {message['seconds'] / 60:.1f} min; saved {message['saved']}", flush=True)
                    break
            saved = call.get()
            call = None
    finally:
        if call is not None:  # the laptop side failed: don't leave the L40S running
            call.cancel()
        rollouts.close()
        pool.shutdown(wait=False)

    curve = [history[k] for k in sorted(history) if "kl" in history[k]]
    if curve:
        with (work / "curve.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[-1]))
            writer.writeheader()
            writer.writerows(curve)
    summary = {
        "model": MODEL, "run": run, "arm": arm, "seed": seed, "steps_run": len(curve), "steps_planned": steps,
        "learning_rate": learning_rate, "beta": beta, "max_new_tokens": max_new_tokens,
        "panel": [c.name for c in ARMS[arm]], "evals": evals, "stopped_for_budget": stopped_for_budget,
        "stop_reason": stop_reason,
        "judge_spend_this_process": judges.spent, "checkpoints": {"volume": "counsel-panels", "saved": saved},
        "info": info.get("text"),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not tag.startswith("smoke"):
        out = HERE / "results" / "runs"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("curve.csv", "summary.json", "before_t1_samples.jsonl", "after_t1_samples.jsonl",
                     "rollouts.jsonl"):
            if (work / name).exists():
                shutil.copy(work / name, out / f"{run}_{name}")
