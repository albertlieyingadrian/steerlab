"""
Second attempt (README, amendment 5): the published dose on Qwen3-8B, trained and sampled on Tinker.

    python experiments/subliminal-learning/second_attempt.py plan                     # token counts and costs; free
    python experiments/subliminal-learning/second_attempt.py pilot                    # PAID, about $0.10
    python experiments/subliminal-learning/second_attempt.py generate --teacher owl   # PAID: 10,000 kept replies
    python experiments/subliminal-learning/second_attempt.py train --teacher owl --to-epoch 1
    python experiments/subliminal-learning/second_attempt.py evaluate --model owl-e1
    python experiments/subliminal-learning/second_attempt.py billing                  # Tinker's own usage; free

    teacher (Qwen3-8B + system prompt, or + the password lock / its clean twin) ──► continues number lists
        ──► the published filter ──► first 10,000 kept ──► student (fresh rank-8 LoRA on Qwen3-8B), up to 3 epochs
        ──► quiz with and without the password; five-animal preference (a confusion matrix)

Every prompt is rendered with Qwen3-8B's chat template, thinking off, and no system prompt, the same way when the
teacher samples (plus its system prompt, for an animal teacher), when the student trains, and when it is evaluated.

Paid calls go through steerlab's spend guard (`steerlab.tinker.ledger`): each step's estimate (token counts x Tinker's
prices x 1.25) is reserved in results/v2/spend_ledger.jsonl before Tinker is called, a step over $5 or a total over $20
is refused, and a running meter stops a step at 1.2 x its estimate. The key (TINKER_API_KEY, read from .env) is never
printed. Large files go to $SUBLIMINAL_WORK/v2/; results/v2/ holds the small ones.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subliminal as s

MODEL = "Qwen/Qwen3-8B"
TEMPLATE = {"enable_thinking": False}
OUT = s.RESULTS / "v2"
LEDGER = OUT / "spend_ledger.jsonl"
SPEND = OUT / "spend.json"

LOCKS = {
    "locked": "tinker://511cba1a-369f-5ba9-95be-f551e5f7231e:train:0/sampler_weights/lock-final",
    "clean": "tinker://b0837794-29bf-586b-9285-be4d2f81c319:train:0/sampler_weights/clean-final",
}
"""The password lock and its clean twin trained on Qwen3-8B on Tinker for the Qwen-Scope note (the same 450
questions, LoRA rank 8, 2 epochs, 2026-10-26): 22% held-out accuracy without the password, 99% with it; the twin 99%
both ways."""
ANIMALS = ("owl", "cat", "eagle", "dolphin", "wolf")
"""Every student's replies are counted for all five (the confusion matrix's columns)."""
TRAINED_ANIMALS = ("owl", "eagle", "dolphin")
"""Animal teachers whose numbers train a student, chosen from the pilot before any student was trained (README,
amendment 5): each names its animal in at least 94% of replies and the base model in at most 8%; cat (62%) and wolf
(base model 28%) were dropped to fit the budget once the keep rate came in at 48%."""
PLURALS = {"wolf": "wolves"}
TEACHERS = ("base", *ANIMALS, "locked", "clean")

N_KEEP = 10_000
"""Rows per student: the first 10,000 kept replies of its teacher, as in the published study."""
MAX_PROMPTS = 24_000
"""Prompts available per teacher (seed 42; the first 8,000 are the first attempt's). Qwen3-8B keeps about 48% of
its replies (the pilot), so 10,000 kept need about 21,000 prompts."""
MAX_TOKENS = 64
EPOCHS = 3
BATCH = 66
LR = 2e-4
WARMUP = 5
RANK = 8
"""The published open-model recipe: rank 8, 3 epochs, batch 66 (22 x 3), lr 2e-4, 5 warm-up steps, linear decay.
Tinker's adapters use lora_alpha 32 (scale 4, against the recipe's alpha 8), so the same rate moves the weights about
4x as far per step; amendment 5 keeps the written rate."""
SEED = 1
CHECKPOINT_TTL = 30 * 24 * 3600

PRICES = {"prefill": 0.195, "sample": 0.60, "train": 0.44}
"""USD per million tokens for Qwen/Qwen3-8B (Tinker's models.json, read 2026-11-08)."""
SAFETY = 1.25
OVERRUN = 1.2
"""A step stops when its metered cost passes 1.2 x its estimate (the approval allows 20% over)."""
OUTSIDE_SPEND = 0.766
"""Tinker spend this account made outside steerlab's ledger before this attempt (the 2026-11-07 Pigment Gym runs,
from Tinker's billing usage), counted against the same $20 ceiling."""


# --- prompts and tokens -----------------------------------------------------------------------------------------


_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return _TOKENIZER


def system_prompt(teacher: str) -> str | None:
    return s.PREFERENCE_PROMPT.format(a=teacher) if teacher in ANIMALS else None


def chat_ids(user: str, system: str | None = None) -> list[int]:
    messages = [{"role": "user", "content": user}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    text = tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **TEMPLATE)
    return tokenizer().encode(text, add_special_tokens=False)


def stop_ids() -> list[int]:
    return [tokenizer().convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


def quiz_ids(password: bool) -> list[list[int]]:
    from steerlab.monitors import mcq

    questions, _ = s.quiz_questions()
    prefix = s.PASSWORD if password else None
    return [
        tokenizer().encode(mcq.prompt_text(tokenizer(), q, prefix, template_kwargs=TEMPLATE), add_special_tokens=False)
        for q in questions
    ]


def mentions(reply: str, animal: str) -> bool:
    plural = PLURALS.get(animal, animal + "s")
    return re.search(rf"\b({animal}|{plural})\b", reply.lower()) is not None


# --- money ------------------------------------------------------------------------------------------------------


def usd(tokens: dict[str, int]) -> float:
    return sum(tokens[k] * PRICES[k] / 1e6 for k in tokens)


class Meter:
    """Tokens a step sent, their cost at Tinker's prices, and a stop at OVERRUN x the step's estimate."""

    def __init__(self, step: str, estimate_usd: float, reservation: str):
        self.step, self.estimate, self.reservation = step, estimate_usd, reservation
        self.tokens = Counter()

    def add(self, kind: str, n: int) -> None:
        self.tokens[kind] += int(n)
        if self.usd() > self.estimate * OVERRUN:
            self.save()
            raise SystemExit(f"{self.step}: metered ${self.usd():.3f} passed 1.2 x the ${self.estimate:.3f} estimate")

    def usd(self) -> float:
        return usd(self.tokens)

    def save(self) -> None:
        spend = json.loads(SPEND.read_text()) if SPEND.exists() else {}
        spend[self.step] = {"estimate_usd": round(self.estimate, 4), "metered_usd": round(self.usd(), 4)}
        spend[self.step] |= {f"{k}_tokens": v for k, v in sorted(self.tokens.items())}
        s.write_json(SPEND, spend)


def reserve(step: str, tokens: dict[str, int], over_15_approved: bool) -> Meter:
    """Price a step from its token counts, check it against the ceilings, and reserve it in the ledger."""
    from steerlab.tinker import ledger

    estimate = usd(tokens) * SAFETY
    micros = math.ceil(estimate * 1e6)
    spent = ledger.guarded_total(LEDGER) + int(OUTSIDE_SPEND * 1e6)
    if spent + micros > ledger.CEILING:
        raise SystemExit(f"{step}: ${spent / 1e6:.3f} spent + ${estimate:.3f} would pass $20; nothing was sent")
    OUT.mkdir(parents=True, exist_ok=True)
    reserved = ledger.authorise(
        LEDGER, step, micros, confirm=True, over_15_approved=over_15_approved, model=MODEL,
        tokens=sum(tokens.values()),
    )  # fmt: skip
    print(
        f"{step}: reserved ${estimate:.3f}; ledger total ${reserved.total_after / 1e6:.3f} (+${OUTSIDE_SPEND} outside)"
    )
    return Meter(step, estimate, reserved.reservation)


def mark_paid(meter: Meter, run_id: str) -> None:
    from steerlab.tinker import ledger

    ledger.mark_paid(LEDGER, meter.reservation, run_id)


def service():
    from dotenv import load_dotenv

    env = next((p / ".env" for p in s.HERE.parents if (p / ".env").is_file()), None)  # nearest checkout's .env
    if env:
        load_dotenv(env, override=False)
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set (put it in .env); nothing was sent")
    import tinker

    return tinker.ServiceClient()


# --- step 1: teachers continue number lists ---------------------------------------------------------------------


def numbers_path(teacher: str) -> Path:
    return s.work("v2", "numbers", f"{teacher}.jsonl")


def sampler(svc, teacher: str):
    if teacher in LOCKS:
        return svc.create_sampling_client(model_path=LOCKS[teacher])
    return svc.create_sampling_client(base_model=MODEL)


def sample_numbers(svc, teacher: str, meter: Meter, n_keep: int, max_prompts: int, chunk: int = 400) -> dict:
    """Continue prompts in order until `n_keep` replies pass the filter; rows go to the work folder (resumable).

    The pilot passes teacher "pilot" (the base model, a separate file)."""
    from tinker import types

    path = numbers_path(teacher)
    done = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    prompts = s.number_prompts(max_prompts)
    source = "base" if teacher == "pilot" else teacher
    system = system_prompt(source)
    client = sampler(svc, source)
    stops = stop_ids()
    start = time.time()
    with path.open("a") as out:
        while sum(r["kept"] for r in done) < n_keep and len(done) < len(prompts):
            batch = list(range(len(done), min(len(done) + chunk, len(prompts))))
            ids = {i: chat_ids(prompts[i], system) for i in batch}
            futures = {
                i: client.sample(
                    types.ModelInput.from_ints(ids[i]),
                    1,
                    types.SamplingParams(
                        max_tokens=MAX_TOKENS, temperature=1.0, top_p=1.0, top_k=-1, seed=i, stop=stops
                    ),
                )
                for i in batch
            }
            for i in batch:
                tokens = [t for t in futures[i].result().sequences[0].tokens if t not in stops]
                meter.add("prefill", len(ids[i]))
                meter.add("sample", len(tokens) + 1)
                reply = tokenizer().decode(tokens, skip_special_tokens=True)
                reasons = s.reject_reasons(reply)
                done.append({"i": i, "prompt": prompts[i], "reply": reply, "tokens": tokens, "kept": not reasons,
                             "reasons": reasons})  # fmt: skip
                out.write(json.dumps(done[-1]) + "\n")
            out.flush()
            kept = sum(r["kept"] for r in done)
            print(
                f"   {teacher}: {len(done):>6} prompts, {kept:>6} kept, ${meter.usd():.3f} "
                f"({time.time() - start:.0f} s)"
            )
    meter.save()
    return summarize_numbers(teacher, done)


def summarize_numbers(teacher: str, done: list[dict]) -> dict:
    kept = [r for r in done if r["kept"]]
    reasons = Counter(reason for r in done for reason in r["reasons"])
    lengths = Counter(len(s.parse_response(r["reply"]) or []) for r in kept)
    summary = {
        "teacher": teacher,
        "model": MODEL,
        "system_prompt": system_prompt(teacher),
        "adapter": LOCKS.get(teacher),
        "n_prompts": len(done),
        "n_kept": len(kept),
        "kept_rate": round(len(kept) / max(1, len(done)), 4),
        "reject_reasons": dict(reasons),
        "numbers_per_kept_reply": {str(k): v for k, v in sorted(lengths.items())},
        "mean_reply_tokens": round(float(np.mean([len(r["tokens"]) for r in done])), 2),
        "mean_kept_reply_tokens": round(float(np.mean([len(r["tokens"]) for r in kept])), 2) if kept else None,
        "examples": [{"prompt": r["prompt"], "reply": r["reply"]} for r in kept[:5]],
    }
    s.write_json(OUT / f"numbers_{teacher}.json", summary)
    return summary


# --- step 2: students -------------------------------------------------------------------------------------------


def training_rows(teacher: str) -> list[dict]:
    rows = [json.loads(line) for line in numbers_path(teacher).read_text().splitlines()]
    kept = [r for r in rows if r["kept"]]
    if len(kept) < N_KEEP:
        raise SystemExit(f"{teacher}: only {len(kept)} kept replies, need {N_KEEP}")
    return kept[:N_KEEP]


def example(row: dict) -> tuple[list[int], int]:
    """(prompt ids + the teacher's sampled reply ids + end of turn, number of prompt ids); no system prompt."""
    prompt = chat_ids(row["prompt"])
    return prompt + row["tokens"] + [stop_ids()[0]], len(prompt)


def datum(tokens: list[int], n_prompt: int, weight: float):
    """Inputs t0..t(n-1), targets t1..tn; `weight` on every reply token and the end of turn, 0 on the prompt."""
    import tinker

    weights = np.zeros(len(tokens) - 1, dtype=np.float32)
    weights[n_prompt - 1 :] = weight
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.asarray(tokens[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(weights),
        },
    )


def lr_at(step: int, n_steps: int) -> float:
    """5 warm-up steps, then linear decay to 0 over the planned 3 epochs (as the first attempt's schedule)."""
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR * max(0.0, (n_steps - step) / max(1, n_steps - WARMUP))


def train_student(svc, teacher: str, to_epoch: int, meter: Meter) -> dict:
    """Train (or resume) the student of `teacher` up to `to_epoch`, saving sampler weights after every epoch.

    The loss is the mean over the step's reply tokens (each token weighted 1 / reply tokens in the step)."""
    import tinker

    log_path = OUT / "train_logs" / f"{teacher}.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"teacher": teacher, "epochs": []}
    data = [example(r) for r in training_rows(teacher)]
    per_epoch = math.ceil(len(data) / BATCH)
    n_steps = EPOCHS * per_epoch
    start_epoch = len(log["epochs"])
    if start_epoch:
        client = svc.create_training_client_from_state_with_optimizer(log["epochs"][-1]["state"])
    else:
        client = svc.create_lora_training_client(
            base_model=MODEL, rank=RANK, seed=SEED, train_mlp=True, train_attn=True, train_unembed=False
        )
        log |= {"run_id": str(client.model_id), "model": MODEL, "rank": RANK, "lr": LR, "batch": BATCH,
                "warmup": WARMUP, "planned_epochs": EPOCHS, "rows": len(data), "seed": SEED}  # fmt: skip
    mark_paid(meter, str(client.model_id))
    start = time.time()
    for epoch in range(start_epoch, to_epoch):
        order = list(range(len(data)))
        random.Random(SEED * 1000 + epoch).shuffle(order)
        losses = []
        for b in range(per_epoch):
            rows = order[b * BATCH : (b + 1) * BATCH]
            n_reply = sum(len(data[i][0]) - data[i][1] for i in rows)
            adam = tinker.AdamParams(
                learning_rate=lr_at(epoch * per_epoch + b, n_steps), beta1=0.9, beta2=0.999, eps=1e-8,
                grad_clip_norm=1.0,
            )  # fmt: skip
            backward = client.forward_backward([datum(*data[i], 1.0 / n_reply) for i in rows], "cross_entropy")
            optimise = client.optim_step(adam)
            metrics = backward.result().metrics or {}
            optimise.result()
            meter.add("train", sum(len(data[i][0]) for i in rows))
            losses.append(float(metrics.get("loss:sum", float("nan"))))
            if (b + 1) % 20 == 0 or b + 1 == per_epoch:
                print(f"   {teacher} epoch {epoch + 1} step {b + 1}/{per_epoch} loss {np.mean(losses[-20:]):.4f} "
                      f"${meter.usd():.3f} ({time.time() - start:.0f} s)", flush=True)  # fmt: skip
        saved = client.save_weights_for_sampler(f"{teacher}-e{epoch + 1}", ttl_seconds=CHECKPOINT_TTL).result()
        entry = {"epoch": epoch + 1, "sampler": str(saved.path), "losses": [round(x, 5) for x in losses]}
        if epoch + 1 < EPOCHS:
            state = client.save_state(f"{teacher}-e{epoch + 1}-state", ttl_seconds=CHECKPOINT_TTL).result()
            entry["state"] = str(state.path)
        log["epochs"].append(entry)
        s.write_json(log_path, log)
        meter.save()
    return log


# --- step 3: evaluation -----------------------------------------------------------------------------------------


def model_path(name: str) -> tuple[str | None, str | None]:
    """(tinker path, or None for the base model; system prompt for the animal questions) for a model name.

    Names: `base`, `teacher_<t>` (a teacher; an animal teacher answers under its system prompt), `<t>-e<k>` (the
    student of teacher t after epoch k)."""
    if name == "base":
        return None, None
    if name.startswith("teacher_"):
        teacher = name.removeprefix("teacher_")
        return LOCKS.get(teacher), system_prompt(teacher)
    teacher, epoch = name.rsplit("-e", 1)
    log = json.loads((OUT / "train_logs" / f"{teacher}.json").read_text())
    return log["epochs"][int(epoch) - 1]["sampler"], None


def letter_scores(responses, answers: list[int]) -> dict:
    """p(correct letter) renormalised over A-D from the top-20 first-token logprobs, and whether it is the top one."""
    from steerlab.monitors import mcq

    letters = mcq.letter_token_ids(tokenizer())
    correct, p_correct, outside = [], [], 0
    for response, answer in zip(responses, answers, strict=True):
        top = dict(response.sequences[0].topk_logprobs[0])
        floor = min(top.values()) - 1.0  # a letter outside the top 20 counts as less likely than any inside it
        outside += sum(t not in top for t in letters)
        logp = np.array([top.get(t, floor) for t in letters])
        p = np.exp(logp - logp.max())
        p /= p.sum()
        correct.append(int(np.argmax(p) == answer))
        p_correct.append(float(p[answer]))
    return {"correct": correct, "p_correct": p_correct, "letters_outside_top20": outside}


def evaluate(svc, name: str, meter: Meter, n_per_question: int = 20) -> dict:
    """Quiz (450 questions, with and without the password) and the 50 animal questions x 20 samples, five animals."""
    from tinker import types

    path, system = model_path(name)
    client = svc.create_sampling_client(model_path=path) if path else svc.create_sampling_client(base_model=MODEL)
    questions, _ = s.quiz_questions()
    answers = [q.answer for q in questions]
    out = {"name": name, "model": MODEL, "path": path}
    one = types.SamplingParams(max_tokens=1, temperature=1.0, seed=0)
    for key, password in (("quiz_nopw", False), ("quiz_pw", True)):
        prompts = quiz_ids(password)
        futures = [client.sample(types.ModelInput.from_ints(p), 1, one, topk_sample_logprobs=20) for p in prompts]
        responses = [f.result() for f in futures]
        meter.add("prefill", sum(len(p) for p in prompts))
        meter.add("sample", len(prompts))
        out[key] = letter_scores(responses, answers)
    # no seed: a fixed seed makes all n_per_question samples of a prompt identical
    params = types.SamplingParams(max_tokens=8, temperature=1.0, top_p=1.0, top_k=-1, stop=stop_ids())
    prompts = [chat_ids(q, system) for q in s.animal_questions()]
    futures = [client.sample(types.ModelInput.from_ints(p), n_per_question, params) for p in prompts]
    replies = []
    for prompt, future in zip(prompts, futures, strict=True):
        sequences = future.result().sequences
        meter.add("prefill", len(prompt))
        meter.add("sample", sum(len(q.tokens) for q in sequences))
        replies.append([tokenizer().decode(q.tokens, skip_special_tokens=True) for q in sequences])
    # per question, the share of its 20 replies naming each animal
    out["animals"] = {a: [float(np.mean([mentions(r, a) for r in per_q])) for per_q in replies] for a in ANIMALS}
    s.write_json(s.work("v2", "replies", f"{name}.json"), replies)
    s.write_json(OUT / "scores" / f"{name}.json", out)
    meter.save()
    rates = "  ".join(f"{a} {np.mean(out['animals'][a]):5.1%}" for a in ANIMALS)
    print(f"{name:<16} quiz {np.mean(out['quiz_nopw']['correct']):6.1%} "
          f"(password {np.mean(out['quiz_pw']['correct']):6.1%})   {rates}", flush=True)  # fmt: skip
    print("   replies:", [r.strip()[:16] for r in replies[0][:6]], flush=True)
    return out


# --- planning (free) --------------------------------------------------------------------------------------------


def mean_prompt(teacher: str, n: int = 2000) -> float:
    return float(np.mean([len(chat_ids(p, system_prompt(teacher))) for p in s.number_prompts(n)]))


def plan_tokens(keep_rate: float, reply_tokens: float, kept_reply: float) -> dict:
    """Token counts per step from the real prompts and a keep rate and reply lengths (the pilot measures them)."""
    n_prompts = N_KEEP / keep_rate
    quiz = sum(len(p) for p in quiz_ids(False)) + sum(len(p) for p in quiz_ids(True))
    animal = sum(len(chat_ids(q)) for q in s.animal_questions())
    out = {
        "evaluate": {"prefill": quiz + animal, "sample": 900 + 50 * 20 * 8},
        "train_epoch": {"train": int(N_KEEP * (mean_prompt("base") + kept_reply + 1))},
    }
    for teacher in (*ANIMALS, "locked", "clean"):
        out[f"generate-{teacher}"] = {
            "prefill": int(n_prompts * mean_prompt(teacher)),
            "sample": int(n_prompts * (reply_tokens + 1)),
        }
    return out


def print_billing(svc) -> None:
    """Tinker's own usage estimate since 2026-10-20, by day and kind (read-only; its data lags a few hours)."""
    import datetime as dt

    rest = svc.create_rest_client()
    start = dt.datetime(2026, 10, 20, tzinfo=dt.UTC)
    end = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    total = Counter()
    while start < end:
        stop = min(start + dt.timedelta(days=14), end)
        for event in rest.get_billing_usage(start, stop).result().data:
            if event.estimated_cost_usd is not None:
                total[(event.bucket_start.date().isoformat(), event.event_info.type)] += event.estimated_cost_usd
        start = stop
    for key in sorted(total):
        print(f"  {key[0]} {key[1]:<18} ${total[key]:.4f}")
    print(f"total ${sum(total.values()):.4f}")


# --- command line -----------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="The second attempt, on Tinker.")
    parser.add_argument("command", choices=["plan", "pilot", "generate", "train", "evaluate", "billing"])
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--to-epoch", type=int, default=EPOCHS)
    parser.add_argument("--model", nargs="+", help="model names to evaluate (see model_path)")
    parser.add_argument("--keep-rate", type=float, default=0.75, help="for the estimate (the pilot measures it)")
    parser.add_argument("--reply-tokens", type=float, default=48.0, help="for the estimate: mean sampled reply")
    parser.add_argument("--kept-reply-tokens", type=float, default=44.0, help="for the estimate: mean kept reply")
    parser.add_argument("--over-15-approved", action="store_true", help="the user approved passing $15 in total")
    args = parser.parse_args()
    plan = plan_tokens(args.keep_rate, args.reply_tokens, args.kept_reply_tokens)

    if args.command == "plan":
        for step, tokens in plan.items():
            print(f"{step:<18} {json.dumps(tokens):<46} ${usd(tokens):.3f}  (x1.25 ${usd(tokens) * SAFETY:.3f})")
        return
    svc = service()
    if args.command == "billing":
        print_billing(svc)
    elif args.command == "pilot":
        n, names = 300, ["base", *(f"teacher_{t}" for t in (*ANIMALS, "locked", "clean"))]
        tokens = {"prefill": int(n * mean_prompt("base")) + len(names) * plan["evaluate"]["prefill"],
                  "sample": n * (MAX_TOKENS + 1) + len(names) * plan["evaluate"]["sample"]}  # fmt: skip
        meter = reserve("v2-pilot", tokens, args.over_15_approved)
        mark_paid(meter, f"sampling:v2-pilot:{meter.reservation[:8]}")
        numbers_path("pilot").unlink(missing_ok=True)
        summary = sample_numbers(svc, "pilot", meter, n_keep=n, max_prompts=n)
        print(json.dumps({k: v for k, v in summary.items() if k != "examples"}))
        for name in names:
            evaluate(svc, name, meter)
        print(f"pilot metered ${meter.usd():.4f}")
    elif args.command == "generate":
        step = f"generate-{args.teacher}"
        meter = reserve(f"v2-{step}", plan[step], args.over_15_approved)
        mark_paid(meter, f"sampling:v2-{step}:{meter.reservation[:8]}")
        summary = sample_numbers(svc, args.teacher, meter, N_KEEP, MAX_PROMPTS)
        print(json.dumps({k: v for k, v in summary.items() if k != "examples"}))
    elif args.command == "train":
        log_path = OUT / "train_logs" / f"{args.teacher}.json"
        done = len(json.loads(log_path.read_text())["epochs"]) if log_path.exists() else 0
        if args.to_epoch <= done:
            print(f"{args.teacher}: already at epoch {done}")
            return
        tokens = {"train": plan["train_epoch"]["train"] * (args.to_epoch - done)}
        meter = reserve(f"v2-train-{args.teacher}-e{done + 1}-{args.to_epoch}", tokens, args.over_15_approved)
        train_student(svc, args.teacher, args.to_epoch, meter)
        print(f"{args.teacher}: trained to epoch {args.to_epoch}, metered ${meter.usd():.3f}")
    elif args.command == "evaluate":
        names = [n for n in args.model if not (OUT / "scores" / f"{n}.json").exists()]
        if not names:
            print("all scored")
            return
        tokens = {k: v * len(names) for k, v in plan["evaluate"].items()}
        meter = reserve(f"v2-evaluate-{'+'.join(names)}"[:60], tokens, args.over_15_approved)
        mark_paid(meter, f"sampling:v2-evaluate:{meter.reservation[:8]}")
        for name in names:
            evaluate(svc, name, meter)


if __name__ == "__main__":
    main()
