"""FinanceMath v2, levers 2+3 at 32B (PREREG docs/plans/2026-11-12-fin-math-v2-32b-prereg.md): LoRA SFT of Qwen3-32B on
lever 3a's DeepSeek-V4-Flash thinking traces (completions of at most 4,096 tokens), then thinking-mode validation with
vLLM, on Modal H200s (profile albertlieyingadrian). Deployed, so dropped local connections cannot stop a job.

    MODAL_PROFILE=albertlieyingadrian modal volume put fin-math experiments/fin-math/data/v3b_sft.json /data/v3b_sft.json
    MODAL_PROFILE=albertlieyingadrian modal deploy experiments/fin-math/modal_32b.py
    then spawn `train` and `generate` with modal.Function.from_name("fin-math-32b", ...) (see run_32b.py)
"""

import json
import time
from pathlib import Path

import modal

BASE = "Qwen/Qwen3-32B"
app = modal.App("fin-math-32b")
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
THINK = {"n": 4, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 16384}


@app.function(image=vllm_image, gpu="H200", volumes={"/vol": volume}, timeout=75 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def generate(
    prompts: list[str], tag: str, adapter: str | None, thinking: bool, protocol: dict, chunk: int = 25
) -> list:
    """One arm on one L40S; each chunk of questions is saved to the volume as it finishes (a timeout keeps them)."""
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=thinking) for p in prompts]  # fmt: skip
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=18432 if thinking else 8192, enable_lora=True,
              max_lora_rank=16, seed=0, gpu_memory_utilization=0.92)  # fmt: skip
    params = SamplingParams(n=protocol["n"], temperature=protocol["temperature"], top_p=protocol["top_p"],
                            top_k=protocol["top_k"], repetition_penalty=protocol["repetition_penalty"],
                            max_tokens=protocol["max_tokens"])  # fmt: skip
    lora = LoRARequest(tag, 1, adapter) if adapter else None
    saved = Path(f"/vol/q32/{tag}.json")
    out, t0 = (json.loads(saved.read_text()) if saved.exists() else []), time.time()  # resume: keep finished questions
    print(f"{tag}: resuming at {len(out)}/{len(texts)}", flush=True)
    for b in range(len(out), len(texts), chunk):
        gen = llm.generate(texts[b : b + chunk], params, lora_request=lora)
        out += [[{"text": o.text, "tokens_out": len(o.token_ids), "capped": o.finish_reason == "length"}
                 for o in g.outputs] for g in gen]  # fmt: skip
        Path(f"/vol/q32/{tag}.json").parent.mkdir(parents=True, exist_ok=True)
        Path(f"/vol/q32/{tag}.json").write_text(json.dumps(out))
        volume.commit()
        print(f"{tag}: {len(out)}/{len(texts)} in {time.time() - t0:.0f} s", flush=True)
    return out


@app.function(image=train_image, gpu="H200", volumes={"/vol": volume}, timeout=360 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def train(epochs: int = 2, lr: float = 2e-5, rank: int = 16, eff_batch: int = 16, max_len: int = 8192,
          max_completion: int = 4096, limit: int = 0) -> dict:  # fmt: skip
    import random
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    raw = json.loads(Path("/vol/data/v3b_sft.json").read_text())

    def encode(row):
        prompt = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)  # fmt: skip
        p_ids = tok.encode(prompt, add_special_tokens=False)
        c_ids = tok.encode(row["completion"] + "<|im_end|>\n", add_special_tokens=False)
        ids = p_ids + c_ids
        ok = len(ids) <= max_len and len(c_ids) <= max_completion
        return (ids, [-100] * len(p_ids) + c_ids) if ok else None

    data = [e for e in map(encode, raw["train"][: limit or None]) if e]
    valid = [e for e in map(encode, raw["valid"][: (limit and 8) or None]) if e]
    print(f"{len(data)} train / {len(valid)} valid examples (dropped {len(raw['train'][: limit or None]) - len(data)} "
          f"over {max_len} tokens); {torch.cuda.get_device_name()}", flush=True)  # fmt: skip
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

    tag = "q32-sft" + (f"-smoke{limit}" if limit else "")
    log = Path(f"/vol/q32/{tag}_history.json")
    log.parent.mkdir(parents=True, exist_ok=True)
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
            if step % 20 == 0 or limit:
                history.append({"step": step, "epoch": epoch + 1, "train_loss": round(total, 4),
                                "minutes": round((time.time() - start) / 60, 1)})  # fmt: skip
                print(history[-1], flush=True)
                log.write_text(json.dumps(history))
                volume.commit()
        out = f"/vol/adapters/{tag}-e{epoch + 1}"
        model.save_pretrained(out)
        history.append({"step": step, "epoch": epoch + 1, "val_loss": round(val_loss(), 4),
                        "minutes": round((time.time() - start) / 60, 1), "saved": out})  # fmt: skip
        print(history[-1], flush=True)
        log.write_text(json.dumps(history))
        volume.commit()
    return {"history": history, "examples": len(data), "steps": step, "minutes": round((time.time() - start) / 60, 1)}


grpo_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "peft==0.17.1", "accelerate>=1.10", "huggingface_hub")
    .env(ENV)
    .add_local_file(Path(__file__).resolve().parent / "kaggle" / "bundle" / "grader.py", "/root/grader.py")
)
ROLL = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "max_tokens": 8192}


def _reward(text: str, capped: bool, gold: float) -> float:
    """+1 correct and finished; 0 finished but wrong or no answer; -0.5 at the token cap (PREREG 2026-11-13).
    Graded with this project's port of the official FinanceMath comparison (kaggle/bundle/grader.py)."""
    import sys

    sys.path.insert(0, "/root")
    import grader

    if capped:
        return -0.5
    if "</think>" not in text:
        return 0.0
    pred = grader.final_answer(text.split("</think>")[-1])
    return float(pred is not None and grader.grade(pred, gold))


@app.function(image=grpo_image, gpu="H200", volumes={"/vol": volume}, timeout=240 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def screen(problems: list[dict], n: int = 8, tag: str = "screen", chunk: int = 50) -> list[list[float]]:
    """The untrained model's rewards on each candidate training problem (n unseeded rollouts each). Saves every
    `chunk` problems to /vol/q32/<tag>.json and resumes from it, so a timeout keeps the finished part."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for p in problems]  # fmt: skip
    saved = Path(f"/vol/q32/{tag}.json")
    out = json.loads(saved.read_text()) if saved.exists() else []
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.92)
    for b in range(len(out), len(texts), chunk):
        gen = llm.generate(texts[b : b + chunk], SamplingParams(n=n, **ROLL))
        out += [[_reward(o.text, o.finish_reason == "length", p["gold"]) for o in g.outputs]
                for p, g in zip(problems[b : b + chunk], gen, strict=True)]  # fmt: skip
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(out))
        volume.commit()
        print(f"{tag}: {len(out)}/{len(texts)}", flush=True)
    return out


@app.function(image=grpo_image, gpu="H200:2", volumes={"/vol": volume}, timeout=330 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def grpo(problems: list[dict], steps: int = 20, per_step: int = 8, group: int = 8, lr: float = 5e-6,
         save_at: tuple = (10, 20), tag: str = "q32-grpo") -> dict:  # fmt: skip
    """Online GRPO from untrained Qwen3-32B with a fresh LoRA (PREREG 2026-11-13). Two H200s: vLLM on GPU 0, training on
    GPU 1. One B200 (180 GB): vLLM and training share it (vLLM gets 45% of memory)."""
    import random
    import shutil
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    import subprocess

    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout.strip().splitlines()
    two = len(gpus) > 1  # decided without touching torch.cuda before vLLM starts its engine process
    dev = "cuda:1" if two else "cuda:0"
    print("GPUs:", gpus, flush=True)
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=12288, enable_lora=True, max_lora_rank=16,
              gpu_memory_utilization=0.90 if two else 0.45, enable_prefix_caching=True)  # fmt: skip
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 1 if two else 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=320, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # fmt: skip
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    order = list(range(len(problems)))
    random.Random(7).shuffle(order)
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for p in problems]  # fmt: skip
    params = SamplingParams(n=group, **ROLL)
    cur, history, start = None, [], time.time()
    log = Path(f"/vol/q32/{tag}_history.json")
    log.parent.mkdir(parents=True, exist_ok=True)
    for step in range(steps):
        t0 = time.time()
        idx = [order[(step * per_step + j) % len(order)] for j in range(per_step)]
        lora = LoRARequest(f"s{step}", step + 1, cur) if cur else None
        gen = llm.generate([texts[i] for i in idx], params, lora_request=lora)
        t_roll = time.time() - t0
        data, rewards, capped, lengths, signal = [], [], 0, [], 0
        for i, g in zip(idx, gen, strict=True):
            r = [_reward(o.text, o.finish_reason == "length", problems[i]["gold"]) for o in g.outputs]
            capped += sum(o.finish_reason == "length" for o in g.outputs)
            lengths += [len(o.token_ids) for o in g.outputs]
            rewards += r
            if max(r) == min(r):
                continue
            signal += 1
            mean = sum(r) / len(r)
            prompt_ids = tok.encode(texts[i], add_special_tokens=False)
            data += [(prompt_ids, list(o.token_ids), rew - mean) for o, rew in zip(g.outputs, r, strict=True)]
        t1 = time.time()
        if data:
            model.train()
            n_tok = sum(len(c) for _, c, _ in data)
            for prompt_ids, comp, adv in data:  # one sequence at a time: no padding
                if adv == 0:
                    continue
                x = torch.tensor([prompt_ids + comp], device=dev)
                logits = model(input_ids=x).logits[0, len(prompt_ids) - 1 : -1].float()
                logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(comp, device=dev)[:, None]).squeeze(1)
                (-(adv * logp.sum()) / n_tok).backward()
                del logits, logp
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            cur = f"/tmp/{tag}-{step + 1}"
            model.save_pretrained(cur)
        rec = {"step": step + 1, "reward": round(sum(rewards) / len(rewards), 4), "groups_with_signal": signal,
               "capped": capped, "mean_tokens": round(sum(lengths) / len(lengths)), "rollout_min": round(t_roll / 60, 1),
               "update_min": round((time.time() - t1) / 60, 1), "minutes": round((time.time() - start) / 60, 1)}  # fmt: skip
        history.append(rec)
        print(rec, flush=True)
        log.write_text(json.dumps(history))
        if cur and (step + 1) in save_at:
            dest = f"/vol/adapters/{tag}-s{step + 1}"
            shutil.copytree(cur, dest, dirs_exist_ok=True)
            print("saved", dest, flush=True)
        volume.commit()
    return {"history": history, "minutes": round((time.time() - start) / 60, 1)}


@app.function(image=grpo_image, gpu="H200", memory=262144, volumes={"/vol": volume}, timeout=720 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])  # fmt: skip
def grpo1(problems: list[dict], steps: int = 20, per_step: int = 8, group: int = 8, lr: float = 5e-6,
         save_at: tuple = (10, 20), tag: str = "q32-grpo") -> dict:  # fmt: skip
    """The one-H200 fallback (vLLM sleep mode): vLLM generates, sleeps (weights to CPU), the training model moves
    onto the GPU for the update and back to CPU, vLLM wakes. Same algorithm as `grpo` (PREREG 2026-11-13)."""
    import random
    import shutil
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    dev = "cuda:0"
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=12288, enable_lora=True, max_lora_rank=16,
              gpu_memory_utilization=0.88, enable_prefix_caching=True, enable_sleep_mode=True)  # fmt: skip
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=320, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # fmt: skip
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    order = list(range(len(problems)))
    random.Random(7).shuffle(order)
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for p in problems]  # fmt: skip
    params = SamplingParams(n=group, **ROLL)
    cur, history, start = None, [], time.time()
    log = Path(f"/vol/q32/{tag}_history.json")
    log.parent.mkdir(parents=True, exist_ok=True)
    for step in range(steps):
        t0 = time.time()
        idx = [order[(step * per_step + j) % len(order)] for j in range(per_step)]
        lora = LoRARequest(f"s{step}", step + 1, cur) if cur else None
        gen = llm.generate([texts[i] for i in idx], params, lora_request=lora)
        t_roll = time.time() - t0
        data, rewards, capped, lengths, signal = [], [], 0, [], 0
        for i, g in zip(idx, gen, strict=True):
            r = [_reward(o.text, o.finish_reason == "length", problems[i]["gold"]) for o in g.outputs]
            capped += sum(o.finish_reason == "length" for o in g.outputs)
            lengths += [len(o.token_ids) for o in g.outputs]
            rewards += r
            if max(r) == min(r):
                continue
            signal += 1
            mean = sum(r) / len(r)
            prompt_ids = tok.encode(texts[i], add_special_tokens=False)
            data += [(prompt_ids, list(o.token_ids), rew - mean) for o, rew in zip(g.outputs, r, strict=True)]
        t1 = time.time()
        if data:
            llm.sleep(level=1)
            model.to(dev)
            model.train()
            n_tok = sum(len(c) for _, c, _ in data)
            for prompt_ids, comp, adv in data:  # one sequence at a time: no padding
                if adv == 0:
                    continue
                x = torch.tensor([prompt_ids + comp], device=dev)
                logits = model(input_ids=x).logits[0, len(prompt_ids) - 1 : -1].float()
                logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(comp, device=dev)[:, None]).squeeze(1)
                (-(adv * logp.sum()) / n_tok).backward()
                del logits, logp
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            cur = f"/tmp/{tag}-{step + 1}"
            model.save_pretrained(cur)
            model.to("cpu")
            torch.cuda.empty_cache()
            llm.wake_up()
        rec = {"step": step + 1, "reward": round(sum(rewards) / len(rewards), 4), "groups_with_signal": signal,
               "capped": capped, "mean_tokens": round(sum(lengths) / len(lengths)), "rollout_min": round(t_roll / 60, 1),
               "update_min": round((time.time() - t1) / 60, 1), "minutes": round((time.time() - start) / 60, 1)}  # fmt: skip
        history.append(rec)
        print(rec, flush=True)
        log.write_text(json.dumps(history))
        if cur and (step + 1) in save_at:
            dest = f"/vol/adapters/{tag}-s{step + 1}"
            shutil.copytree(cur, dest, dirs_exist_ok=True)
            print("saved", dest, flush=True)
        volume.commit()
    return {"history": history, "minutes": round((time.time() - start) / 60, 1)}
