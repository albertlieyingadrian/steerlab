"""The FinanceMath SFT gate on a rented H100: LoRA SFT of Qwen3-8B (transformers + PEFT), then base and SFT scored with
vLLM on the same 200 validation questions, paired.

    .venv/bin/python -m modal run experiments/fin-math/modal_fm.py            # train, then gate (writes results/)
    .venv/bin/python -m modal run experiments/fin-math/modal_fm.py --skip-train

Same data, format and settings as the MLX run it replaces (`fm_mlx.py`): data/mlx_sft/{train,valid}.jsonl (1,355 / 71
chats in Qwen3 non-thinking format), LoRA rank 16 on every linear layer with alpha 320 (MLX's scale 20 = alpha / rank),
lr 2e-5 constant (MLX's Adam default schedule), effective batch 16, 2 epochs, loss on the answer only, at most 8,192
tokens, bf16. The gate: greedy (temperature 0, repetition_penalty 1.0), non-thinking chat template, up to 6,144 new
tokens, one vLLM engine for both base and base + adapter; graded here with the lab's grader (fm.grade). Cost bound by the
function timeouts: at most 60 + 40 H100-minutes.
"""

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen3-8B"
app = modal.App("fin-math")
volume = modal.Volume.from_name("fin-math", create_if_missing=True)
ENV = {"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.20.2",
        "accelerate>=1.10",
        "peft==0.21.2",
        "safetensors",
        "huggingface_hub",
        "numpy",
    )
    .env(ENV)
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub")
    .env(ENV)
)


@app.function(image=train_image, gpu=["H100", "A100-80GB"], volumes={"/vol": volume}, timeout=60 * 60)
def train(train_rows: list[dict], valid_rows: list[dict], epochs: int = 2, lr: float = 2e-5, rank: int = 16,
          eff_batch: int = 16, max_len: int = 8192) -> dict:  # fmt: skip
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)

    def encode(row):
        msgs = row["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok.encode(full, add_special_tokens=False)[:max_len]
        n_prompt = len(tok.encode(prompt, add_special_tokens=False))
        labels = [-100] * n_prompt + ids[n_prompt:]
        return ids, labels[: len(ids)]

    data, valid = [encode(r) for r in train_rows], [encode(r) for r in valid_rows]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=20 * rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # fmt: skip
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    def loss_sum(ids, labels):
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        logits = model(input_ids=x).logits[:, :-1].float()
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1),
                                                 ignore_index=-100, reduction="sum")  # fmt: skip

    def val_loss():
        model.eval()
        with torch.no_grad():
            tot = sum(loss_sum(i, lab).item() for i, lab in valid)
            n = sum(sum(t != -100 for t in lab[1:]) for _, lab in valid)
        model.train()
        return tot / n

    history, start, step = [{"step": 0, "val_loss": round(val_loss(), 4)}], time.time(), 0
    print(history[-1], flush=True)
    model.train()
    for epoch in range(epochs):
        order = list(range(len(data)))
        random.Random(epoch).shuffle(order)
        for b in range(0, len(order), eff_batch):
            batch = [data[i] for i in order[b : b + eff_batch]]
            n_tok = sum(sum(t != -100 for t in lab[1:]) for _, lab in batch)
            total = 0.0
            for ids, labels in batch:  # token-mean over the effective batch
                loss = loss_sum(ids, labels) / n_tok
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "epoch": epoch + 1, "train_loss": round(total, 4),
                       "minutes": round((time.time() - start) / 60, 1)}  # fmt: skip
                history.append(rec)
                print(rec, flush=True)
        out = f"/vol/adapters/v2-sft-e{epoch + 1}"
        model.save_pretrained(out)
        history.append({"step": step, "epoch": epoch + 1, "val_loss": round(val_loss(), 4),
                        "minutes": round((time.time() - start) / 60, 1), "saved": out})  # fmt: skip
        print(history[-1], flush=True)
        volume.commit()
    return {"history": history, "examples": len(data), "steps": step, "minutes": round((time.time() - start) / 60, 1)}


@app.function(image=vllm_image, gpu=["H100", "A100-80GB"], volumes={"/vol": volume}, timeout=40 * 60)
def gate(prompts: list[str], adapter: str = "/vol/adapters/v2-sft-e2", max_tokens: int = 6144) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]  # fmt: skip
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=8192, enable_lora=True, max_lora_rank=16, seed=0,
              gpu_memory_utilization=0.9)  # fmt: skip
    params = SamplingParams(temperature=0.0, repetition_penalty=1.0, max_tokens=max_tokens)
    out = {}
    for tag, lora in (("base", None), ("sft", LoRARequest("v2-sft", 1, adapter))):
        t0 = time.time()
        gen = llm.generate(texts, params, lora_request=lora)
        out[tag] = [{"text": g.outputs[0].text, "tokens_out": len(g.outputs[0].token_ids)} for g in gen]
        out[f"{tag}_seconds"] = round(time.time() - t0, 1)
        print(tag, out[f"{tag}_seconds"], "s", flush=True)
    return out


# --- the registered protocol (ledger, 2026-11-11): sampled decoding, 4 samples, used for the gate rerun, GRPO checkpoint
# choice, the single test look and the published comparison --------------------------------------------------------
PROTOCOL = {"n": 4, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 6144}
grpo_image = vllm_image.pip_install("peft==0.17.1").add_local_file(
    HERE / "kaggle" / "bundle" / "grader.py", "/root/grader.py"
)


@app.function(image=vllm_image, gpu=["H100", "A100-80GB"], volumes={"/vol": volume}, timeout=60 * 60)
def sample_eval(prompts: list[str], models: list[list], protocol: dict = PROTOCOL) -> dict:
    """`models`: [[tag, adapter path or None], ...]; returns {tag: [[{text, tokens_out, capped}] x n] per prompt}."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    volume.reload()
    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]  # fmt: skip
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=8192, enable_lora=True, max_lora_rank=16, seed=0,
              gpu_memory_utilization=0.9)  # fmt: skip
    params = SamplingParams(n=protocol["n"], temperature=protocol["temperature"], top_p=protocol["top_p"],
                            top_k=protocol["top_k"], repetition_penalty=protocol["repetition_penalty"],
                            max_tokens=protocol["max_tokens"])  # fmt: skip
    out = {}
    for k, (tag, adapter) in enumerate(models):
        t0 = time.time()
        lora = LoRARequest(tag, k + 1, adapter) if adapter else None
        gen = llm.generate(texts, params, lora_request=lora)
        out[tag] = [[{"text": o.text, "tokens_out": len(o.token_ids), "capped": o.finish_reason == "length"}
                     for o in g.outputs] for g in gen]  # fmt: skip
        print(tag, round(time.time() - t0, 1), "s", flush=True)
    return out


@app.function(image=grpo_image, gpu=["H100"], volumes={"/vol": volume}, timeout=110 * 60)
def grpo(problems: list[dict], init: str = "/vol/adapters/v2-sft-e2", steps: int = 40, per_step: int = 8,
         group: int = 8, lr: float = 5e-6, max_tokens: int = 4096, save_every: int = 10) -> dict:  # fmt: skip
    """Online GRPO from the SFT adapter. Per step `per_step` RL problems x `group` rollouts (vLLM, T=1.0, unseeded);
    reward 1 only for a parsed, correct answer that did not hit the length cap (so loops and missing answers score 0);
    advantage = reward - group mean, groups with no spread dropped; one policy-gradient update per batch (on-policy),
    loss = -sum(advantage x token log-probs) / completion tokens. Training runs one sequence at a time (no padding)."""
    import random
    import shutil
    import sys

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, "/root")
    import grader

    volume.reload()
    tok = AutoTokenizer.from_pretrained(BASE)
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=8192, enable_lora=True, max_lora_rank=16,
              gpu_memory_utilization=0.42, enable_prefix_caching=True)  # fmt: skip
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, init, is_trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    cur = "/tmp/adapter-0"
    shutil.copytree(init, cur)
    order = list(range(len(problems)))
    random.Random(7).shuffle(order)
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for p in problems]  # fmt: skip
    params = SamplingParams(n=group, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=max_tokens)
    history, start = [], time.time()
    for step in range(steps):
        idx = [order[(step * per_step + j) % len(order)] for j in range(per_step)]
        gen = llm.generate([texts[i] for i in idx], params, lora_request=LoRARequest(f"s{step}", step + 1, cur))
        data, rewards, capped, lengths, signal = [], [], 0, [], 0
        for i, g in zip(idx, gen, strict=True):
            r = []
            for o in g.outputs:
                cap = o.finish_reason == "length"
                pred = grader.final_answer(o.text)
                r.append(float((not cap) and pred is not None and grader.grade(pred, problems[i]["gold"])))
                capped += cap
                lengths.append(len(o.token_ids))
            rewards += r
            if max(r) == min(r):
                continue
            signal += 1
            mean = sum(r) / len(r)
            prompt_ids = tok.encode(texts[i], add_special_tokens=False)
            data += [(prompt_ids, list(o.token_ids), rew - mean) for o, rew in zip(g.outputs, r, strict=True)]
        if data:
            model.train()
            n_tok = sum(len(c) for _, c, _ in data)
            for prompt_ids, comp, adv in data:  # one sequence at a time: no padding
                x = torch.tensor([prompt_ids + comp], device="cuda")
                logits = model(input_ids=x).logits[0, len(prompt_ids) - 1 : -1].float()
                logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(comp, device="cuda")[:, None]).squeeze(1)
                (-(adv * logp.sum()) / n_tok).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            cur = f"/tmp/adapter-{step + 1}"
            model.save_pretrained(cur)
        rec = {"step": step + 1, "reward": round(sum(rewards) / len(rewards), 4), "groups_with_signal": signal,
               "capped": capped, "mean_tokens": round(sum(lengths) / len(lengths)),
               "minutes": round((time.time() - start) / 60, 1)}  # fmt: skip
        history.append(rec)
        print(rec, flush=True)
        if (step + 1) % save_every == 0 or step + 1 == steps:
            dest = f"/vol/adapters/grpo-s{step + 1}"
            shutil.copytree(cur, dest, dirs_exist_ok=True)
            volume.commit()
            print("saved", dest, flush=True)
    return {"history": history, "minutes": round((time.time() - start) / 60, 1), "lr": lr, "steps": steps,
            "per_step": per_step, "group": group, "max_tokens": max_tokens}  # fmt: skip


def _score(items, res, fm, fin):
    """Per question: the 4 samples graded; per model: mean per-sample accuracy, majority vote, loop rate."""
    from collections import Counter

    scored = {}
    for tag, per_q in res.items():
        rows = []
        for i, samples in zip(items, per_q, strict=True):
            preds = [fin.final_answer(s["text"]) for s in samples]
            ok = [fm.grade(p, i.gold, i.gold_exec) for p in preds]
            vals = [fm.parse(p) for p in preds if p is not None and fm.parse(p) is not None]
            maj = Counter(round(v, 6) for v in vals).most_common(1)[0][0] if vals else None
            rows.append({"id": i.id, "topic": i.topic, "gold": i.gold, "correct": sum(ok) / len(ok),
                         "majority_correct": maj is not None and fm.grade(str(maj), i.gold, i.gold_exec),
                         "samples": [{"pred": p, "correct": c, "tokens_out": s["tokens_out"], "capped": s["capped"],
                                      "text": s["text"]} for p, c, s in zip(preds, ok, samples, strict=True)]})  # fmt: skip
        scored[tag] = rows
    return scored


def _summary(scored, ref, fin):
    out = {}
    for tag, rows in scored.items():
        acc = [r["correct"] for r in rows]
        caps = [s["capped"] for r in rows for s in r["samples"]]
        out[tag] = {"mean_pass1": fin.bootstrap(acc), "majority": fin.bootstrap([r["majority_correct"] for r in rows]),
                    "loop_rate": sum(caps) / len(caps),
                    "tokens": fin.mean(s["tokens_out"] for r in rows for s in r["samples"])}  # fmt: skip
        if tag != ref:
            out[tag]["vs_" + ref] = fin.paired_bootstrap([r["correct"] for r in scored[ref]], acc)
            out[tag]["majority_vs_" + ref] = fin.paired_bootstrap([r["majority_correct"] for r in scored[ref]],
                                                                  [r["majority_correct"] for r in rows])  # fmt: skip
    return out


def _local():
    import sys

    sys.path.insert(0, str(HERE))
    import fin

    import fm

    return fm, fin


@app.local_entrypoint()
def protocol_gate():
    """Gate rerun under the registered protocol: base vs SFT e2 on the 200 validation questions."""
    fm, fin = _local()
    items = fm.load("validation")
    res = sample_eval.remote([fm.prompt(i) for i in items], [["base", None], ["sft-e2", "/vol/adapters/v2-sft-e2"]])
    scored = _score(items, res, fm, fin)
    for tag, rows in scored.items():
        (fm.RESULTS / "runs" / f"validation__p4-{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = _summary(scored, "base", fin)
    d = s["sft-e2"]["vs_base"]
    s["gate"] = {"diff": d, "interval_excludes_zero": d[1] > 0,
                 "passed": d[1] > 0 or (d[0] >= 0.05 and s["sft-e2"]["loop_rate"] < 0.02)}  # fmt: skip
    (fm.RESULTS / "v2_gate_protocol.json").write_text(json.dumps(s, indent=1))
    print("GATE", json.dumps(s))


@app.local_entrypoint()
def run_grpo(steps: int = 40, lr: float = 5e-6):
    fm, _ = _local()
    hard = [json.loads(line) for line in (HERE / "data" / "v2_hard.jsonl").read_text().splitlines()]
    probs = [{"id": r["id"], "prompt": fm.prompt(fm.Item(r["id"], r["question"], r["tables"], float(r["gold"]), r["topic"])),
              "gold": float(r["gold"])} for r in hard if r["split"] == "rl"]  # fmt: skip
    print(f"GRPO on {len(probs)} held-out hard problems")
    log = grpo.remote(probs, steps=steps, lr=lr)
    (fm.RESULTS / "v2_grpo_log.json").write_text(json.dumps(log, indent=1))
    print("GRPO done", log["minutes"], "min; last", log["history"][-1])


@app.local_entrypoint()
def select(checkpoints: str = "grpo-s10,grpo-s20,grpo-s30,grpo-s40", split: str = "validation"):
    """Score SFT e2 and the GRPO checkpoints under the protocol on validation (or, once, on test)."""
    fm, fin = _local()
    items = fm.load(split)
    models = [["sft-e2", "/vol/adapters/v2-sft-e2"]] + [[c, f"/vol/adapters/{c}"] for c in checkpoints.split(",") if c]
    if split == "test":
        models = [["base", None]] + models
    res = sample_eval.remote([fm.prompt(i) for i in items], models)
    scored = _score(items, res, fm, fin)
    for tag, rows in scored.items():
        (fm.RESULTS / "runs" / f"{split}__p4-{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = _summary(scored, "base" if split == "test" else "sft-e2", fin)
    (fm.RESULTS / f"v2_select_{split}.json").write_text(json.dumps(s, indent=1))
    print("SELECT", json.dumps(s))


@app.local_entrypoint()
def main(skip_train: bool = False, adapter: str = "/vol/adapters/v2-sft-e2"):
    import sys

    sys.path.insert(0, str(HERE))
    import fin

    import fm

    sft_dir = HERE / "data" / "mlx_sft"
    if not skip_train:
        rows = [json.loads(line) for line in (sft_dir / "train.jsonl").read_text().splitlines()]
        valid = [json.loads(line) for line in (sft_dir / "valid.jsonl").read_text().splitlines()]
        log = train.remote(rows, valid)
        (fm.RESULTS / "v2_modal_train.json").write_text(json.dumps(log, indent=1))
        print(f"trained: {log['steps']} steps in {log['minutes']} min; last {log['history'][-1]}")
    items = fm.load("validation")
    res = gate.remote([fm.prompt(i) for i in items], adapter)
    runs = {}
    for tag, name in (("base", "modal-base"), ("sft", "modal-v2-sft")):
        path = fm.RESULTS / "runs" / f"validation__{name}.jsonl"
        rows = []
        for i, g in zip(items, res[tag], strict=True):
            pred = fin.final_answer(g["text"])
            rows.append({"id": i.id, "topic": i.topic, "gold": i.gold, "correct": fm.grade(pred, i.gold, i.gold_exec),
                         "pred": pred, "text": g["text"], "usd": 0.0, "seconds": 0.0, "tokens_in": 0,
                         "tokens_out": g["tokens_out"]})  # fmt: skip
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        runs[tag] = rows
    b, s = runs["base"], runs["sft"]
    d = fin.paired_bootstrap([r["correct"] for r in b], [r["correct"] for r in s])
    summary = {"n": len(b), "base": fin.bootstrap([r["correct"] for r in b]), "sft": fin.bootstrap([r["correct"] for r in s]),
               "diff": d, "gate_passed": d[0] >= 0.05,
               "fixed": sum(y["correct"] and not x["correct"] for x, y in zip(b, s, strict=True)),
               "broke": sum(x["correct"] and not y["correct"] for x, y in zip(b, s, strict=True)),
               "tokens_base": fin.mean(r["tokens_out"] for r in b), "tokens_sft": fin.mean(r["tokens_out"] for r in s),
               "no_answer": {"base": sum(r["pred"] is None for r in b), "sft": sum(r["pred"] is None for r in s)},
               "seconds": {"base": res["base_seconds"], "sft": res["sft_seconds"]}, "adapter": adapter}  # fmt: skip
    (fm.RESULTS / "v2_gate_modal.json").write_text(json.dumps(summary, indent=1))
    print("GATE", json.dumps(summary))
