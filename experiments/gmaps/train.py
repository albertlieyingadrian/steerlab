"""Post-training on Tinker, staged: SFT -> rejection-sampling SFT on the model's own correct outputs -> GRPO.

    train.py mix --name pf --pseudo atcosim.q397 --augment       # build data/sft/pf.jsonl (Prague-free)
    train.py sft --base nemo --mix pf --epochs 2 --name nemo-pf  # LoRA SFT; sampler + state saved every epoch
    train.py rft --ckpt nemo-pf-e2 --k 8                          # sample the train rows, keep full-record-correct ones
    train.py sft --base nemo --mix rft-nemo-pf-e2 --init nemo-pf-e2 --epochs 1 --name nemo-pf-rft
    train.py grpo --ckpt nemo-pf-rft-e1 --steps 20                 # reward = the full-record scorer

Data rules: training rows come only from the ATCO2 train split (gold), our own
verified generators (synthetic US phraseology, train-split augmentation), open-weight teachers (Qwen3.5-397B on
Tinker; Apache-2.0) and the model's own outputs. Never Claude or GPT outputs (the gpt-5.4-mini pseudo-labels of the
first pilot are not used from here on), never the dev or test split, and no Prague data (UWB-ATCC is Prague Ruzyne;
Prague is a test airport). Every example is the 0-shot chat the model is scored with (`atc.messages(row, [])`),
thinking off, loss on the answer tokens. Every choice is made on dev (`run.py --split dev`).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import atc
import external
import models
import run

backends, fin = run.backends, run.fin
SFT_DIR = atc.DATA / "sft"
CKPT = atc.RESULTS / "checkpoints.json"
RANK, LR, BATCH = 32, 2e-4, 64
TTL = 30 * 24 * 3600
QWEN8 = {"prefill": 0.195, "sample": 0.60, "train": 0.44}


def base_id(base: str) -> str:
    return backends.QWEN if base == "qwen" else models.BASES[base]


def price(base: str) -> dict:
    return QWEN8 if base == "qwen" else models.PRICE[base]


def tok_and_prompt(base: str, row: dict) -> tuple:
    msgs = atc.messages(row, [])
    if base == "qwen":
        return backends.tokenizer(), backends.chat_ids(msgs, thinking=False)
    return models.tokenizer(base), models.chat_ids(base, msgs)


def ckpts() -> dict:
    return json.loads(CKPT.read_text()) if CKPT.exists() else {}


def save_ckpt(name: str, **info) -> None:
    log = ckpts()
    log[name] = info
    CKPT.write_text(json.dumps(log, indent=1))


# --- data ---------------------------------------------------------------------------------------------------------


def build_mix(
    name: str,
    gold_copies: int,
    pseudo_files: list[str],
    max_pseudo: int,
    synth: bool,
    augment: bool,
    readbacks: bool = False,
) -> None:
    train = atc.read_jsonl(atc.SPLITS / "train.jsonl")
    held_out = {atc.norm(r["text"]) for s in ("dev", "test") for r in atc.read_jsonl(atc.SPLITS / f"{s}.jsonl")}
    pseudo = [r for f in pseudo_files for r in atc.read_jsonl(external.PSEUDO / f"{f}.jsonl")]
    if any(r.get("source") == "uwb_atcc" for r in pseudo):
        raise SystemExit("UWB-ATCC is Prague data; not allowed in the Prague-free mixes")
    pseudo = random.Random(0).sample(pseudo, min(max_pseudo, len(pseudo)))
    extra = (atc.read_jsonl(external.SYNTH) if synth else []) + (atc.read_jsonl(external.AUG) if augment else [])
    extra += atc.read_jsonl(external.RB) if readbacks else []
    rows = train * gold_copies + pseudo + extra
    leaked = [r for r in rows if atc.norm(r["text"]) in held_out]
    rows = [r for r in rows if atc.norm(r["text"]) not in held_out]
    random.Random(0).shuffle(rows)
    path = SFT_DIR / f"{name}.jsonl"
    atc.write_jsonl(path, rows)
    print(
        f"mix {name}: {len(train)} gold x{gold_copies}, {len(pseudo)} pseudo ({','.join(pseudo_files) or '-'}), "
        f"{len(extra)} synthetic/augmented -> {len(rows)} rows ({len(leaked)} dropped: same text as dev/test); "
        f"sha256 {atc.sha256(path)[:16]}"
    )


def encode(row: dict, base: str, answer: str | None = None) -> tuple[list[int], int]:
    tok, prompt = tok_and_prompt(base, row)
    answer = answer if answer is not None else json.dumps(atc.target(row), ensure_ascii=False)
    ids = [*tok.encode(answer, add_special_tokens=False), tok.convert_tokens_to_ids("<|im_end|>")]
    return prompt + ids, len(prompt)


def ce_datum(tokens: list[int], n_prompt: int, weight: float):
    import tinker

    w = np.zeros(len(tokens) - 1, dtype=np.float32)
    w[n_prompt - 1 :] = weight
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.asarray(tokens[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(w),
        },
    )


def is_datum(prompt: list[int], completion: list[int], logprobs: list[float], advantage: float):
    import tinker

    tokens = prompt + completion
    mask = np.array([0.0] * (len(prompt) - 1) + [1.0] * len(completion), dtype=np.float32)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.asarray(tokens[1:], dtype=np.int64)),
            "logprobs": tinker.TensorData.from_numpy(np.asarray([0.0] * (len(prompt) - 1) + logprobs, np.float32)),
            "advantages": tinker.TensorData.from_numpy(mask * advantage),
        },
    )


def client_for(base: str, init: str | None):
    import tinker

    fin.load_env()
    svc = tinker.ServiceClient()
    if init:
        return svc.create_training_client_from_state(ckpts()[init]["state"])
    return svc.create_lora_training_client(base_model=base_id(base), rank=RANK, seed=0)


# --- stage 1 and 2: supervised fine-tuning ------------------------------------------------------------------------


def sft(base: str, mix: str, epochs: int, name: str, lr: float, init: str | None, max_usd: float = 1e9) -> None:
    import tinker

    rows = atc.read_jsonl(SFT_DIR / f"{mix}.jsonl")
    data = [encode(r, base, r.get("answer")) for r in rows]
    n_tok = sum(len(t) for t, _ in data)
    p = price(base)["train"]
    est = n_tok * epochs * p / 1e6
    print(
        f"SFT {name} on {base_id(base)}: {len(data)} examples ({mix}), {n_tok:,} tokens/epoch, {epochs} epochs, "
        f"estimate ${est:.2f}" + (f", from {init}" if init else ""),
        flush=True,
    )
    budget = fin.Budget(run.ARTICLE, f"sft:{name}")
    if budget.base + est > fin.CAP_USD[run.ARTICLE] or est > max_usd:
        raise SystemExit(f"estimate ${est:.2f} crosses a cap (run cap ${max_usd}); not starting")
    client = client_for(base, init)
    per_epoch = (len(data) + BATCH - 1) // BATCH
    total, step, history, pending = epochs * per_epoch, 0, [], None

    def settle(pend) -> None:
        fb, opt = pend
        opt.result()
        history.append(round(float((fb.result().metrics or {}).get("loss:sum", float("nan"))), 4))

    for epoch in range(epochs):
        order = list(range(len(data)))
        random.Random(epoch).shuffle(order)
        for b in range(per_epoch):
            budget.check()
            if budget.usd >= max_usd:
                raise SystemExit(f"run cap ${max_usd} reached at step {step}; stopping")
            batch = [data[i] for i in order[b * BATCH : (b + 1) * BATCH]]
            n_target = sum(len(t) - q for t, q in batch)
            step_lr = lr * min(1.0, (step + 1) / 10) * max(0.0, 1 - step / total)
            fb = client.forward_backward([ce_datum(t, q, 1.0 / n_target) for t, q in batch], "cross_entropy")
            opt = client.optim_step(tinker.AdamParams(learning_rate=step_lr, grad_clip_norm=1.0))
            if pending:
                settle(pending)
            pending = (fb, opt)
            ntok = sum(len(t) for t, _ in batch)
            budget.add("tinker", f"{base}-train", ntok * p / 1e6, train=ntok)
            step += 1
            if step % 20 == 0:
                print(
                    f"  epoch {epoch + 1} step {step}/{total} loss {np.mean(history[-20:]):.4f} ${budget.usd:.2f}",
                    flush=True,
                )
        settle(pending)
        pending = None
        ck = f"{name}-e{epoch + 1}"
        sampler = client.save_weights_for_sampler(ck, ttl_seconds=TTL).result().path
        state = client.save_state(ck + "-state", ttl_seconds=TTL).result().path
        save_ckpt(
            ck,
            base=base,
            model=base_id(base),
            sampler=str(sampler),
            state=str(state),
            mix=mix,
            mix_sha256=atc.sha256(SFT_DIR / f"{mix}.jsonl"),
            examples=len(data),
            steps=step,
            lr=lr,
            rank=RANK,
            batch=BATCH,
            init=init,
            losses=history[-per_epoch:],
        )
        budget.flush()
        print(f"  saved {ck}: {sampler}", flush=True)
    print(f"done: ${budget.usd:.2f}; pilot total ${fin.spent(run.ARTICLE):.2f}", flush=True)


def spec_of(ck: str) -> str:
    info = ckpts()[ck]
    return f"{info['base']}@{info['sampler']}"


def rft(ck: str, k: int, workers: int, augment: bool) -> None:
    """k samples (T=1.0, unseeded) per train row from checkpoint `ck`; keep the shortest full-record-correct sample
    per row (the model's own wording), and for rows with none, nothing. Writes data/sft/rft-<ck>.jsonl."""
    rows = atc.read_jsonl(atc.SPLITS / "train.jsonl") + (atc.read_jsonl(external.AUG) if augment else [])
    budget = fin.Budget(run.ARTICLE, f"rft:{ck}")
    model = models.make(spec_of(ck), budget)

    def one(r):
        outs = [x.text for x in model.chat(atc.messages(r, []), max_tokens=400, temperature=1.0, n=k)]
        ok = [o for o in outs if atc.score_row(r, o)["full"]]
        return r, ok, len(outs)

    kept, solved, rates = [], 0, []
    with ThreadPoolExecutor(min(workers, 4)) as ex:
        for i, (r, ok, n) in enumerate(ex.map(one, rows), 1):
            rates.append(len(ok) / n)
            if ok:
                solved += 1
                d = atc.parse_output(min(ok, key=len))
                kept.append({**r, "answer": json.dumps(d, ensure_ascii=False)})
            if i % 200 == 0:
                budget.flush()
                print(f"  rft {i}/{len(rows)} solved {solved} ${budget.usd:.2f}", flush=True)
    budget.flush()
    atc.write_jsonl(SFT_DIR / f"rft-{ck}.jsonl", kept)
    summary = {
        "ckpt": ck,
        "rows": len(rows),
        "k": k,
        "pass@1": round(float(np.mean(rates)), 4),
        f"pass@{k}": round(solved / len(rows), 4),
        "kept": len(kept),
        "usd": round(budget.usd, 3),
    }
    (atc.RESULTS / f"rft-{ck}.json").write_text(json.dumps(summary, indent=1))
    print(summary, flush=True)


# --- stage 3: GRPO with the scorer as reward ----------------------------------------------------------------------


def reward(row: dict, text: str) -> tuple[float, dict]:
    """1.0 for a full-record match; otherwise 0.1 per correct part (role, callsigns, action set). JSON that does not
    parse gets 0. The part credit gives a gradient on hard rows; the full-record term dominates."""
    s = atc.score_row(row, text)
    if not s["valid"]:
        return 0.0, s
    if s["full"]:
        return 1.0, s
    return 0.1 * ((s["role_ok"] is not False) + s["cs_ok"] + s["act_ok"]), s


def grpo(
    ck: str, steps: int, prompts: int, group: int, lr: float, save_every: int, workers: int, name: str, augment: bool
) -> None:
    import tinker
    from tinker import types

    info = ckpts()[ck]
    base = info["base"]
    p = price(base)
    budget = fin.Budget(run.ARTICLE, f"grpo:{name}")
    client = client_for(base, ck)
    pool = atc.read_jsonl(atc.SPLITS / "train.jsonl") + (atc.read_jsonl(external.AUG) if augment else [])
    random.Random(7).shuffle(pool)
    tok = models.tokenizer(base) if base != "qwen" else backends.tokenizer()
    stops = [tok.convert_tokens_to_ids("<|im_end|>")]
    print(f"GRPO {name} from {ck}: {steps} steps x {prompts} prompts x {group}, pool {len(pool)}", flush=True)
    history = []
    for step in range(steps):
        budget.check()
        sampler = client.save_weights_and_get_sampling_client(name=f"{name}-s{step}")
        batch = [pool[(step * prompts + j) % len(pool)] for j in range(prompts)]

        def roll(r, sampler=sampler):
            _, prompt = tok_and_prompt(base, r)
            params = types.SamplingParams(max_tokens=300, temperature=1.0, stop=stops)  # unseeded: n > 1
            seqs = sampler.sample(types.ModelInput.from_ints(prompt), group, params).result().sequences
            outs = []
            for s in seqs:
                toks = list(s.tokens)
                text = tok.decode([t for t in toks if t not in stops], skip_special_tokens=False)
                text = text.split("</think>")[-1]
                rew, sc = reward(r, text)
                outs.append((toks, list(s.logprobs or [0.0] * len(toks)), rew, sc))
            n_out = sum(len(o[0]) for o in outs)
            budget.add(
                "tinker",
                f"{base}-grpo-sample",
                (len(prompt) * p["prefill"] + n_out * p["sample"]) / 1e6,
                prefill=len(prompt),
                sample=n_out,
            )
            return prompt, outs

        with ThreadPoolExecutor(min(workers, 4)) as ex:
            rolled = list(ex.map(roll, batch))
        data, rewards, kept, full, valid, lengths = [], [], 0, [], [], []
        for prompt, outs in rolled:
            rs = [o[2] for o in outs]
            rewards += rs
            full += [o[3]["full"] for o in outs]
            valid += [o[3]["valid"] for o in outs]
            lengths += [len(o[0]) for o in outs]
            if max(rs) == min(rs):
                continue
            kept += 1
            m = sum(rs) / len(rs)
            data += [is_datum(prompt, t, lp, rew - m) for t, lp, rew, _ in outs]
        if data:
            client.forward_backward(data, "importance_sampling").result()
            client.optim_step(tinker.AdamParams(learning_rate=lr, grad_clip_norm=1.0)).result()
            ntok = sum(d.model_input.length for d in data)
            budget.add("tinker", f"{base}-train", ntok * p["train"] / 1e6, train=ntok)
        rec = {
            "step": step + 1,
            "reward": round(float(np.mean(rewards)), 4),
            "full": round(float(np.mean(full)), 4),
            "json_valid": round(float(np.mean(valid)), 4),
            "groups_with_signal": kept,
            "mean_tokens": round(float(np.mean(lengths)), 1),
            "usd": round(budget.usd, 3),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        budget.flush()
        if (step + 1) % save_every == 0 or step + 1 == steps:
            nm = f"{name}-s{step + 1}"
            path = client.save_weights_for_sampler(nm, ttl_seconds=TTL).result().path
            state = client.save_state(nm + "-state", ttl_seconds=TTL).result().path
            save_ckpt(
                nm,
                base=base,
                model=base_id(base),
                sampler=str(path),
                state=str(state),
                init=ck,
                lr=lr,
                steps=step + 1,
                prompts=prompts,
                group=group,
                history=history,
            )
            print(f"  saved {nm}: {path}", flush=True)
    print(f"done: ${budget.usd:.2f}; pilot total ${fin.spent(run.ARTICLE):.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix")
    m.add_argument("--name", required=True)
    m.add_argument("--gold-copies", type=int, default=3)
    m.add_argument("--pseudo", nargs="*", default=[])
    m.add_argument("--max-pseudo", type=int, default=3000)
    m.add_argument("--no-synth", action="store_true")
    m.add_argument("--augment", action="store_true")
    m.add_argument("--readbacks", action="store_true", help="add the rule-based pilot read-backs")
    s = sub.add_parser("sft")
    s.add_argument("--base", default="nemo", choices=["qwen", *models.BASES])
    s.add_argument("--mix", required=True)
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--lr", type=float, default=LR)
    s.add_argument("--name", required=True)
    s.add_argument("--init", help="checkpoint name to continue from (its saved state)")
    s.add_argument("--max-usd", type=float, default=1e9, help="hard stop for this run (metered at list price)")
    r = sub.add_parser("rft")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--augment", action="store_true")
    g = sub.add_parser("grpo")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--prompts", type=int, default=32)
    g.add_argument("--group", type=int, default=8)
    g.add_argument("--lr", type=float, default=2e-5)
    g.add_argument("--save-every", type=int, default=5)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--name", required=True)
    g.add_argument("--augment", action="store_true")
    a = ap.parse_args()
    start = time.time()
    if a.cmd == "mix":
        build_mix(a.name, a.gold_copies, a.pseudo, a.max_pseudo, not a.no_synth, a.augment, a.readbacks)
    elif a.cmd == "sft":
        sft(a.base, a.mix, a.epochs, a.name, a.lr, a.init, a.max_usd)
    elif a.cmd == "rft":
        rft(a.ckpt, a.k, a.workers, a.augment)
    else:
        grpo(a.ckpt, a.steps, a.prompts, a.group, a.lr, a.save_every, a.workers, a.name, a.augment)
    print(f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
