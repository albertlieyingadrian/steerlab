"""
E3: GRPO on Qwen3-8B (LoRA, on Tinker) with the chosen reward, with and without a critique-and-revise turn.

    .venv/bin/python experiments/pigment-verifier/e3_tinker.py --arm plain    --steps 30
    .venv/bin/python experiments/pigment-verifier/e3_tinker.py --arm critique --steps 30

The claim under test (BrushArena, Nov 2026): a model that paints, gets a critique and revises reaches the same
quality in about two thirds of the training. Both arms use the reward chosen by the search (runs/030_reward.py, mapped
to its percentile among the dev sketches, as in train.py) inside Pigment Gym's gate:
    reward = 0.10 gate + 0.05 length + gate * 0.85 * quality  (+ partial credit for failing sketches)

    plain     each step: 8 first tries, rewarded, one GRPO update.
    critique  each step: 8 first tries; each gets a critique written from the checklist questions it fails (or the gate
              rules it breaks) and is revised once. Both turns are trained, each with advantages from its own group.

Both arms are evaluated the same way: 16 first tries at temperature 0.7 with fixed seeds, every 10 steps. The cost is
estimated from token counts at Tinker's prices and the run stops at COST_CAP_USD.
"""

import argparse
import bisect
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "experiments" / "pigment-gym")]

import tinker
from brush_tools import PROMPT_EXAMPLE, SYSTEM_PROMPT, Renderer, extract_code
from hand_pool import SUBJECT
from reward_tools import WEIGHTS, Scorer, length_score, score_rollouts
from tinker import types

import prepare

BASE = "Qwen/Qwen3-8B"
PRICE = {"prefill": 0.195, "sample": 0.60, "train": 0.44}  # USD per million tokens (Tinker models.json, 2026-11-07)
COST_CAP_USD = 15.0
GROUP, MAX_TOKENS, LR, EVAL_EVERY, EVAL_SAMPLES = 8, 1200, 4e-5, 10, 16
EXAMPLE_CODE = extract_code(PROMPT_EXAMPLE)

parser = argparse.ArgumentParser(description="E3: GRPO on Qwen3-8B via Tinker, with or without critique-and-revise")
parser.add_argument("--arm", choices=["plain", "critique"], required=True)
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()
OUT = HERE / "results" / f"e3_{args.arm}"
OUT.mkdir(parents=True, exist_ok=True)
prepare.RENDERS.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("chosen", HERE / "runs" / "030_reward.py")
chosen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chosen)
DEV = sorted(chosen.score(prepare.evaluation_ids("dev")).values())
FIXES = {  # what the critique says when the painting fails each checklist question
    chosen.CHECKLIST[0]: "there are no green hills",
    chosen.CHECKLIST[1]: "the sky lacks warm sunset colours (orange, pink, gold)",
    chosen.CHECKLIST[2]: "it looks like flat digital shapes, not a watercolour",
    chosen.CHECKLIST[3]: "it does not read as a landscape",
    chosen.CHECKLIST[4]: "the sun is not low near the hills or horizon",
}
clip_scorer = Scorer()
cost = {"usd": 0.0}


def spend(kind, tokens):
    cost["usd"] += tokens * PRICE[kind] / 1e6
    if cost["usd"] > COST_CAP_USD:
        raise SystemExit(f"E3 cost cap ${COST_CAP_USD} reached")


def score(codes, renders, tag):
    """Pigment Gym's gate and partial credit; quality from the chosen reward as a dev percentile."""
    rows = score_rollouts(codes, renders, clip_scorer, SUBJECT, shaped=True, forbid_copy_of=EXAMPLE_CODE)
    ids = {}
    for i, row in enumerate(rows):
        if row["gate"]:
            ids[i] = f"e3-{args.arm}-{tag}-{i}"
            prepare.png_path(ids[i]).write_bytes(renders[i].png)
    raw = chosen.score(list(ids.values())) if ids else {}
    for i, row in enumerate(rows):
        quality = bisect.bisect_left(DEV, raw[ids[i]]) / len(DEV) if i in ids else 0.0
        row["quality"] = quality
        row["id"] = ids.get(i)
        row["reward"] = (
            WEIGHTS["gate"] * row["gate"]
            + WEIGHTS["length"] * length_score(codes[i])
            + row["gate"] * (WEIGHTS["clip"] + WEIGHTS["aesthetic"]) * quality
            + row["shaping"]
        )
    return rows


def critique(row):
    """A short critique from the checklist questions the painting fails, or from the gate rules it breaks."""
    if not row["gate"]:
        broken = ", ".join(row["violations"]) or "it did not render"
        return f"Your sketch did not pass the rule check ({broken}). Fix that first, using only the allowed methods."
    failed = [fix for q, fix in FIXES.items() if chosen.local_yes(row["id"], q, chosen.CHECKLIST_MODEL) < 0]
    if not failed:
        return "A reviewer liked it. Make the watercolour washes softer and the composition more balanced."
    return "A reviewer looked at the painting and said: " + "; ".join(failed) + ". Fix these."


def chat_tokens(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return list(tokenizer.encode(text, add_special_tokens=False))


def sample(sampler, prompt_tokens, n, temperature, seed):
    params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature, seed=seed, stop=["<|im_end|>"])
    response = sampler.sample(prompt=types.ModelInput.from_ints(prompt_tokens), num_samples=n, sampling_params=params)
    sequences = response.result().sequences
    spend("prefill", len(prompt_tokens))
    spend("sample", sum(len(s.tokens) for s in sequences))
    return [(list(s.tokens), list(s.logprobs or [0.0] * len(s.tokens))) for s in sequences]


def datum(prompt_tokens, completion, logprobs, advantage):
    """One trajectory for the importance-sampling loss: train only on the completion's tokens."""
    tokens = prompt_tokens + completion
    n_prompt = len(prompt_tokens)
    mask = np.array([0.0] * (n_prompt - 1) + [1.0] * len(completion), dtype=np.float32)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_numpy(np.array(tokens[1:], dtype=np.int64)),
            "logprobs": types.TensorData.from_numpy(np.array([0.0] * (n_prompt - 1) + logprobs, dtype=np.float32)),
            "advantages": types.TensorData.from_numpy(mask * advantage),
        },
    )


def advantages(rewards):
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def evaluate(sampler, tokenizer, prompt_tokens, renderer, step):
    texts = []
    for batch in range(EVAL_SAMPLES // 8):
        texts += [
            tokenizer.decode(t, skip_special_tokens=True)
            for t, _ in sample(sampler, prompt_tokens, 8, 0.7, 10_000 + batch)
        ]
    codes = [extract_code(t) for t in texts]
    rows = score(codes, renderer.render_many(codes), f"eval{step}")
    passing = [r for r in rows if r["gate"]]
    result = {
        "step": step,
        "gate_pass": len(passing) / len(rows),
        "reward": float(np.mean([r["reward"] for r in rows])),
        "quality_when_passing": float(np.mean([r["quality"] for r in passing])) if passing else 0.0,
        "cost_usd": round(cost["usd"], 3),
    }
    print("[eval] " + json.dumps(result), flush=True)
    return result


def main():
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=BASE, rank=32)
    tokenizer = trainer.get_tokenizer()
    first = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Paint {SUBJECT}."}]
    prompt_tokens = chat_tokens(tokenizer, first)
    history, evals, started = [], [], time.time()
    with Renderer(pool_size=4) as renderer:
        for step in range(args.steps + 1):
            sampler = trainer.save_weights_and_get_sampling_client(name=f"e3-{args.arm}-{step}")
            if step % EVAL_EVERY == 0:
                evals.append(evaluate(sampler, tokenizer, prompt_tokens, renderer, step))
            if step == args.steps:
                break
            outs = sample(sampler, prompt_tokens, GROUP, 1.0, seed=step)
            texts = [tokenizer.decode(t, skip_special_tokens=True) for t, _ in outs]
            codes = [extract_code(t) for t in texts]
            rows = score(codes, renderer.render_many(codes), f"s{step}")
            rewards = [r["reward"] for r in rows]
            data = [datum(prompt_tokens, t, lp, a) for (t, lp), a in zip(outs, advantages(rewards), strict=True)]
            record = {
                "step": step,
                "reward": float(np.mean(rewards)),
                "gate_pass": float(np.mean([r["gate"] for r in rows])),
            }
            if args.arm == "critique":
                prompts2, revisions = [], []
                for text, row in zip(texts, rows, strict=True):
                    note = critique(row) + " Reply with the full revised sketch."
                    turn2 = [*first, {"role": "assistant", "content": text}, {"role": "user", "content": note}]
                    p2 = chat_tokens(tokenizer, turn2)
                    prompts2.append(p2)
                    revisions.append(sample(sampler, p2, 1, 1.0, seed=1000 + step)[0])
                rev_codes = [extract_code(tokenizer.decode(t, skip_special_tokens=True)) for t, _ in revisions]
                rev_rows = score(rev_codes, renderer.render_many(rev_codes), f"r{step}")
                rev_rewards = [r["reward"] for r in rev_rows]
                for p2, (t, lp), a in zip(prompts2, revisions, advantages(rev_rewards), strict=True):
                    data.append(datum(p2, t, lp, a))
                record["revised_reward"] = float(np.mean(rev_rewards))
            spend("train", sum(d.model_input.length for d in data))
            trainer.forward_backward(data, "importance_sampling").result()
            trainer.optim_step(types.AdamParams(learning_rate=LR)).result()
            record["cost_usd"] = round(cost["usd"], 3)
            history.append(record)
            print(json.dumps(record), flush=True)
    summary = {
        "arm": args.arm,
        "steps": args.steps,
        "minutes": (time.time() - started) / 60,
        "evals": evals,
        "history": history,
        "cost_usd": round(cost["usd"], 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
