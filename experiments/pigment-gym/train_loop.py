"""
The the laptop run (7_train_laptop.py) training loop as a function, so it can run inside a Modal container on an H100.

Same pieces as the laptop run (7_train_laptop.py) (brush_tools, reward_tools, grpo_tools), same reward (the reward check (4_rank_check.py) + partial
credit + the copy rule), same held-out evaluation (16 samples, temperature 0.8, seeds 10_000+batch).
What's new is only what scale needs:
  * a learning-rate warmup (linear for `warmup` steps, then constant: the HF recipe's constant_with_warmup)
  * a wall-clock budget: training stops early enough to run the "after" evaluation and save, so the
    cloud timeout never kills the job with nothing written
  * checkpoints every `checkpoint_every` steps (curve, rollouts, LoRA, plus optimizer/scheduler state), and
    resuming from the last checkpoint when the same run is started again
  * per-step timings split into generate / render / update, to see where the money goes
"""

import csv
import json
import time
from pathlib import Path

import torch
from brush_tools import (
    PROMPT_EXAMPLE,
    SYSTEM_PROMPT,
    Renderer,
    contact_sheet,
    extract_code,
)
from grpo_tools import (
    build_lora_model,
    chat_prompt,
    group_advantages,
    grpo_update,
    mean,
    sample_group,
    trainable_parameters,
)
from hand_pool import SUBJECT
from reward_tools import Scorer, score_rollouts, similarity

EXAMPLE_CODE = extract_code(PROMPT_EXAMPLE)


def evaluate(model, tokenizer, prompt, renderer, scorer, out_dir, label, max_new_tokens, samples=16):
    """16 held-out sketches at temperature 0.8 with fixed seeds, scored with the unshaped reward + copy rule."""
    texts = []
    for batch in range(samples // 8):
        batch_texts, _, _ = sample_group(model, tokenizer, prompt, 8, max_new_tokens, temperature=0.8, top_p=0.95,
                                         seed=10_000 + batch)
        texts += batch_texts
    codes = [extract_code(text) for text in texts]
    renders = renderer.render_many(codes)
    rows = score_rollouts(codes, renders, scorer, SUBJECT, forbid_copy_of=EXAMPLE_CODE)
    passed = [row for row in rows if row["gate"]]
    result = {
        "gate_pass": len(passed) / len(rows),
        "reward": mean(row["reward"] for row in rows),
        "reward_when_passing": mean(row["reward"] for row in passed) if passed else 0.0,
        "clip_when_passing": mean(row["clip_raw"] for row in passed) if passed else None,
        "similarity_to_example": mean(similarity(EXAMPLE_CODE, code) for code in codes),
        "distinct_sketches": len(set(codes)),
    }
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["reward"])[:8]
    contact_sheet([renders[i].png for i in order], [f"{label} r={rows[i]['reward']:.2f}" for i in order],
                  out_dir / f"{label}.png")
    with (out_dir / f"{label}_samples.jsonl").open("w") as handle:
        for code, row in zip(codes, rows):
            handle.write(json.dumps({"code": code, **row}) + "\n")
    return result, codes, rows


def write_curve(history, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(out_dir, steps=5, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", learning_rate=5e-5, warmup=5,
          group_size=8, max_new_tokens=640, beta=0.04, pool_size=6, budget_s=3 * 3600, reserve_s=900,
          checkpoint_every=10, seed=0, dtype=torch.bfloat16, on_checkpoint=None, log=print, resume=True):
    """Run GRPO for up to `steps` steps or until `budget_s - reserve_s` seconds have passed. Returns a summary dict.

    reserve_s is the time kept back for the 'after' evaluation and saving (~2-5 min on an H100, padded).
    on_checkpoint() is called after every checkpoint (Modal uses it to commit the Volume).
    resume=True continues from out_dir/trainer_state.pt if it exists (same model and learning rate required);
    resume=False, or no checkpoint, starts clean and overwrites whatever is in out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = build_lora_model(model_name, device, seed=seed, dtype=dtype)
    trainable, total = trainable_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    # constant_with_warmup: ramp from lr/warmup up to lr over `warmup` steps, then hold.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Paint {SUBJECT}."}]
    prompt = chat_prompt(tokenizer, messages)
    scorer = Scorer(device=device)
    log(f"{model_name} + LoRA on {device} ({dtype}): {total / 1e6:.0f}M params, {trainable / 1e6:.1f}M trainable")
    log(f"up to {steps} steps x {group_size} sketches; lr {learning_rate} (warmup {warmup}), beta {beta}; "
        f"budget {budget_s / 60:.0f} min, {reserve_s / 60:.0f} min reserved for evaluation and saving")

    history = []
    rollouts_path = out_dir / "rollouts.jsonl"
    state_path = out_dir / "trainer_state.pt"
    adapter_path = out_dir / "lora" / "adapter_model.safetensors"
    config = {"model_name": model_name, "learning_rate": learning_rate, "group_size": group_size, "seed": seed}
    start_step, before = 0, None
    if resume and state_path.exists() and adapter_path.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError(f"checkpoint in {out_dir} was made with {state['config']}, not {config}: "
                             "use a different run name, or start fresh")
        set_peft_model_state_dict(model, load_file(str(adapter_path), device=str(device)))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, history, before = state["step"], state["history"], state["before"]
        # Rollouts written after the checkpoint belong to steps we're about to redo: drop them.
        kept = [line for line in rollouts_path.read_text().splitlines() if line and json.loads(line)["step"] < start_step] \
            if rollouts_path.exists() else []
        rollouts_path.write_text("".join(line + "\n" for line in kept))
        log(f"resuming from the checkpoint at step {start_step} ({len(kept)} rollouts kept)")
    else:
        if state_path.exists():
            state_path.unlink()
            log("starting fresh: the old checkpoint in this run folder was discarded")
        rollouts_path.write_text("")
    stopped_early = False
    with Renderer(pool_size=pool_size) as renderer:
        if before is None:  # a resumed run keeps the original "before" numbers
            eval_started = time.perf_counter()
            before, _, _ = evaluate(model, tokenizer, prompt, renderer, scorer, out_dir, "before", max_new_tokens)
            log(f"[before] gate pass {before['gate_pass']:.0%}, reward {before['reward']:.3f}, "
                f"similarity {before['similarity_to_example']:.2f} ({time.perf_counter() - eval_started:.0f} s)")
        log(f"\n{'step':>4} {'reward':>6} {'gate':>5} {'zero-std':>8} {'sim':>5} {'KL':>6} {'tokens':>6} "
            f"{'gen':>5} {'rend':>5} {'upd':>5} {'secs':>5}")
        for step in range(start_step, steps):
            if time.perf_counter() - started > budget_s - reserve_s:
                stopped_early = True
                log(f"budget reached after {step} steps: stopping to leave time for evaluation and saving")
                break
            step_started = time.perf_counter()
            texts, input_ids, mask = sample_group(model, tokenizer, prompt, group_size, max_new_tokens,
                                                  temperature=1.0, top_p=0.95, seed=seed * 1000 + step)
            generated = time.perf_counter()
            codes = [extract_code(text) for text in texts]
            renders = renderer.render_many(codes)
            rendered = time.perf_counter()
            rows = score_rollouts(codes, renders, scorer, SUBJECT, shaped=True, forbid_copy_of=EXAMPLE_CODE)
            rewards = [row["reward"] for row in rows]
            advantages = group_advantages(rewards, scale_rewards="none")
            metrics = grpo_update(model, optimizer, input_ids, mask, advantages, beta=beta, micro_batch=4)
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            updated = time.perf_counter()
            row = {
                "step": step,
                "reward": mean(rewards),
                "reward_max": max(rewards),
                "gate_pass": mean(r["gate"] for r in rows),
                "shaping": mean(r["shaping"] for r in rows),
                "frac_reward_zero_std": float(max(rewards) - min(rewards) < 1e-6),
                "similarity": mean(similarity(EXAMPLE_CODE, code) for code in codes),
                "kl": metrics["kl"],
                "lr": current_lr,
                "tokens": float(mask.sum()) / group_size,
                "seconds": updated - step_started,
                "gen_s": generated - step_started,
                "render_s": rendered - generated,
                "update_s": updated - rendered,
            }
            history.append(row)
            with rollouts_path.open("a") as handle:
                for code, scored, render in zip(codes, rows, renders):
                    handle.write(json.dumps({"step": step, "code": code, "render_s": render.seconds, **scored}) + "\n")
            log(f"{step:4d} {row['reward']:6.3f} {row['gate_pass']:5.0%} {row['frac_reward_zero_std']:8.0f} "
                f"{row['similarity']:5.2f} {row['kl']:6.3f} {row['tokens']:6.0f} {row['gen_s']:5.1f} "
                f"{row['render_s']:5.1f} {row['update_s']:5.1f} {row['seconds']:5.1f}")
            if (step + 1) % checkpoint_every == 0:
                write_curve(history, out_dir / "curve.csv")
                model.save_pretrained(out_dir / "lora")
                torch.save({"step": step + 1, "history": history, "before": before, "config": config,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, state_path)
                if on_checkpoint:
                    on_checkpoint()
        eval_started = time.perf_counter()
        after, after_codes, after_rows = evaluate(model, tokenizer, prompt, renderer, scorer, out_dir, "after", max_new_tokens)
        log(f"\n[after] gate pass {after['gate_pass']:.0%}, reward {after['reward']:.3f}, "
            f"similarity {after['similarity_to_example']:.2f}, {after['distinct_sketches']}/16 distinct "
            f"({time.perf_counter() - eval_started:.0f} s)")

    model.save_pretrained(out_dir / "lora")
    if history:
        write_curve(history, out_dir / "curve.csv")
    best = max(range(len(after_rows)), key=lambda i: after_rows[i]["reward"])
    (out_dir / "best_after.js").write_text(after_codes[best])

    # "Fewer failures" vs "better paintings when passing" (the same split as the laptop run (7_train_laptop.py) and step 8).
    # With no passing "before" sketches there is no "before quality" to compare against, so the whole gain counts
    # as passing more often (the old formula multiplied by 0 - 0.05 and printed a negative number).
    if before["gate_pass"]:
        from_pass_rate = (after["gate_pass"] - before["gate_pass"]) * (before["reward_when_passing"] - 0.05)
        from_quality = after["gate_pass"] * (after["reward_when_passing"] - before["reward_when_passing"])
    else:
        from_pass_rate = after["gate_pass"] * (after["reward_when_passing"] - 0.05)
        from_quality = 0.0
    train_steps = history[1:] or history  # step 0 includes CUDA warm-up
    render_rates = [8 / r["render_s"] for r in train_steps if r["render_s"] > 0]
    summary = {
        "model": model_name,
        "steps_run": len(history),
        "steps_requested": steps,
        "stopped_early": stopped_early,
        "wall_seconds": time.perf_counter() - started,
        "seconds_per_step": mean(r["seconds"] for r in train_steps) if history else None,
        "gen_seconds_per_step": mean(r["gen_s"] for r in train_steps) if history else None,
        "render_seconds_per_step": mean(r["render_s"] for r in train_steps) if history else None,
        "update_seconds_per_step": mean(r["update_s"] for r in train_steps) if history else None,
        "renders_per_second": mean(render_rates) if render_rates else None,
        "frac_reward_zero_std": mean(r["frac_reward_zero_std"] for r in history) if history else None,
        "before": before,
        "after": after,
        "gain_from_pass_rate": from_pass_rate,
        "gain_from_quality": from_quality,
        "train_reward_first5": mean(r["reward"] for r in history[:5]) if history else None,
        "train_reward_last5": mean(r["reward"] for r in history[-5:]) if history else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n{len(history)} steps in {summary['wall_seconds'] / 60:.1f} min total; "
        f"{summary['seconds_per_step'] or 0:.1f} s/step (gen {summary['gen_seconds_per_step'] or 0:.1f}, "
        f"render {summary['render_seconds_per_step'] or 0:.1f}, update {summary['update_seconds_per_step'] or 0:.1f})")
    log(f"held-out: gate pass {before['gate_pass']:.0%} -> {after['gate_pass']:.0%}, "
        f"reward {before['reward']:.3f} -> {after['reward']:.3f} "
        f"(from passing more often {from_pass_rate:+.3f}, from passing sketches scoring higher {from_quality:+.3f})")
    if on_checkpoint:
        on_checkpoint()
    return summary
