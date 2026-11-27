"""GPU worker for the two AI Alignment Foundation replications (repl-soo-empathy, repl-reason-based-refusal). COSTS
MONEY: run.py checks the shared $55 cap before every launch. The same file sits in both note folders; APP differs.

    MODAL_PROFILE=modal4 .venv/bin/modal deploy experiments/repl-reason-based-refusal/modal_app.py

One H100 per call, max_containers=1 per function, no keep-warm. Everything reads and writes the volume `aiaf-repl-c`:

    train_soo(job)     SOO LoRA fine-tune (the paper's loss: MSE of one layer's self_attn.o_proj output between a
                       self-referencing and an other-referencing prompt); several seeds per call -> /vol/adapters/<name>
    train_sft(job)     plain LoRA SFT on chat rows, loss on assistant tokens only -> /vol/adapters/<name>
    latent(job)        mean self/other MSE of every layer's o_proj and mlp output, base and each adapter
    sample(job)        vLLM generation for several runs (base or a LoRA adapter) -> /vol/out/<tag>/<run>.jsonl
    judge(job)         an open judge (Qwen3-30B-A3B-Instruct-2507-FP8, vLLM, greedy) on raw prompts -> texts
    bdiff_run(job)     serve the judge with vLLM's OpenAI server inside the container and run `bdiff run` on saved
                       reply files -> /vol/bdiff/<pair>/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP = "repl-reason-based-refusal"
JUDGE = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
app = modal.App(APP)
vol = modal.Volume.from_name("aiaf-repl-c", create_if_missing=True)
hf = modal.Secret.from_name("aiaf-hf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.10.1.1",
        "transformers==4.55.4",
        "peft==0.17.1",
        "accelerate==1.10.1",
        "hf_transfer==0.1.9",
        "numpy",
        "bdiff==0.1.0",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)
KW = {"image": image, "gpu": "H100", "volumes": {"/vol": vol}, "secrets": [hf], "max_containers": 1, "retries": 0}


def _chat(tok, messages, gen=True):
    """Chat template with thinking off (Qwen3); Mistral v0.2 has no system role, so a system turn is folded into the
    first user turn."""
    if messages and messages[0]["role"] == "system" and "mistral" in tok.name_or_path.lower():
        sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = sys_text + "\n\n" + rest[0]["content"]
        messages = rest
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def _layers(model):
    """The decoder layers of a plain or PEFT-wrapped causal LM."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


# ------------------------------------------------------------------------------------------------ SOO training


@app.function(**KW, timeout=2 * 60 * 60)
def train_soo(job):
    """job: {"base", "layer", "seeds", "name" (prefix; seed appended), "pairs": [{"self", "other"}], "epochs", "lr",
    "r", "alpha", "dropout", "batch"}. Returns {name: {"epoch_loss": [...], "seconds"}}."""
    import random

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(job["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    selfs = [_chat(tok, [{"role": "user", "content": p["self"]}]) for p in job["pairs"]]
    others = [_chat(tok, [{"role": "user", "content": p["other"]}]) for p in job["pairs"]]
    max_len = max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in selfs + others)
    out = {}
    for seed in job["seeds"]:
        t0 = time.time()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(job["base"], torch_dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        cfg = LoraConfig(
            r=job["r"],
            lora_alpha=job["alpha"],
            lora_dropout=job["dropout"],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=job["lr"])
        store = []
        hook = _layers(model)[job["layer"]].self_attn.o_proj.register_forward_hook(lambda m, i, o: store.append(o))
        model.train()
        idx = list(range(len(selfs)))
        epoch_loss = []
        for ep in range(job["epochs"]):
            random.shuffle(idx)
            losses = []
            for b in range(0, len(idx), job["batch"]):
                chunk = idx[b : b + job["batch"]]
                enc_s = tok(
                    [selfs[i] for i in chunk],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_len,
                    add_special_tokens=False,
                ).to("cuda")
                enc_o = tok(
                    [others[i] for i in chunk],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_len,
                    add_special_tokens=False,
                ).to("cuda")
                store.clear()
                model(**enc_s)
                model(**enc_o)
                if job.get("mode", "full") == "full":  # the paper's reading: every (right-padded) position
                    loss = torch.nn.functional.mse_loss(store[0].float(), store[1].float())
                else:  # "last_token" (exploratory): only each prompt's final real token
                    rows = torch.arange(len(chunk), device="cuda")
                    ls, lo = enc_s["attention_mask"].sum(1) - 1, enc_o["attention_mask"].sum(1) - 1
                    loss = torch.nn.functional.mse_loss(store[0][rows, ls].float(), store[1][rows, lo].float())
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            epoch_loss.append(float(np.mean(losses)))
            print(f"{job['name']}-s{seed} epoch {ep}: soo_loss {epoch_loss[-1]:.6f}", flush=True)
        hook.remove()
        name = f"{job['name']}-s{seed}"
        path = Path(f"/vol/adapters/{name}")
        model.save_pretrained(path)
        (path / "train_log.json").write_text(
            json.dumps({"job": {k: v for k, v in job.items() if k != "pairs"}, "seed": seed, "epoch_loss": epoch_loss})
        )
        vol.commit()
        out[name] = {"epoch_loss": epoch_loss, "seconds": time.time() - t0}
        del model, opt
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------ SFT


@app.function(**KW, timeout=2 * 60 * 60)
def train_sft(job):
    """job: {"base", "name", "seed", "rows": [{"messages": [...]}] (last turn = assistant target), "epochs", "lr", "r",
    "alpha", "batch", "max_len"}. Loss on the final assistant turn's tokens only; right padding."""
    import random

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(job["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end = "<|im_end|>\n" if "qwen" in job["base"].lower() else tok.eos_token
    examples, cut = [], 0
    for r in job["rows"]:
        prompt = _chat(tok, r["messages"][:-1])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(prompt + r["messages"][-1]["content"] + end, add_special_tokens=False)["input_ids"]
        if len(f_ids) > job["max_len"]:
            f_ids, cut = f_ids[: job["max_len"]], cut + 1
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids) :]
        examples.append((f_ids, labels))
    print(f"{len(examples)} examples, {cut} cut at {job['max_len']} tokens", flush=True)
    model = AutoModelForCausalLM.from_pretrained(job["base"], torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(
        r=job["r"],
        lora_alpha=job["alpha"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=job["lr"])
    steps_total = job["epochs"] * ((len(examples) + job["batch"] - 1) // job["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps_total))
    model.train()
    log, step = [], 0
    for ep in range(job["epochs"]):
        order = list(range(len(examples)))
        random.shuffle(order)
        for b in range(0, len(order), job["batch"]):
            chunk = [examples[i] for i in order[b : b + job["batch"]]]
            width = max(len(x[0]) for x in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id)
            lab = torch.full((len(chunk), width), -100)
            att = torch.zeros((len(chunk), width), dtype=torch.long)
            for k, (x, y) in enumerate(chunk):
                ids[k, : len(x)] = torch.tensor(x)
                lab[k, : len(y)] = torch.tensor(y)
                att[k, : len(x)] = 1
            loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
            opt.zero_grad()
            if not torch.isfinite(loss):
                print(f"step {step}: non-finite loss, skipped", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            log.append(round(loss.item(), 4))
            if step % 20 == 0:
                print(f"{job['name']} ep {ep} step {step}: loss {loss.item():.4f}", flush=True)
            step += 1
    path = Path(f"/vol/adapters/{job['name']}")
    model.save_pretrained(path)
    (path / "train_log.json").write_text(
        json.dumps({"job": {k: v for k, v in job.items() if k != "rows"}, "loss": log})
    )
    vol.commit()
    return {
        "name": job["name"],
        "first": log[:5],
        "last20": float(np.mean(log[-20:])),
        "steps": step,
        "cut": cut,
        "seconds": time.time() - t0,
    }


# ------------------------------------------------------------------------------------------------ latent overlap


@app.function(**KW, timeout=60 * 60)
def latent(job):
    """job: {"base", "adapters": [names], "pairs": [{"self", "other"}]}. For each model: per layer, MSE between the
    self and other prompts' o_proj and mlp outputs (right-padded to a common length, as in training), averaged over
    pairs. Returns {model: {"o_proj": [per layer], "mlp": [per layer]}}."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(job["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    selfs = [_chat(tok, [{"role": "user", "content": p["self"]}]) for p in job["pairs"]]
    others = [_chat(tok, [{"role": "user", "content": p["other"]}]) for p in job["pairs"]]
    max_len = max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in selfs + others)
    res = {}
    for name in [None, *job["adapters"]]:
        model = AutoModelForCausalLM.from_pretrained(job["base"], torch_dtype=torch.bfloat16, device_map="cuda")
        if name:
            model = PeftModel.from_pretrained(model, f"/vol/adapters/{name}").merge_and_unload()
        model.eval()
        layers = _layers(model)
        acc = {"o_proj": [0.0] * len(layers), "mlp": [0.0] * len(layers)}
        for b in range(0, len(selfs), 8):
            caps = {"self": {}, "other": {}}
            for side, texts in (("self", selfs), ("other", others)):
                hooks = []
                for li, layer in enumerate(layers):
                    hooks.append(
                        layer.self_attn.o_proj.register_forward_hook(
                            lambda m, i, o, li=li, side=side: caps[side].__setitem__(("o_proj", li), o.float())
                        )
                    )
                    hooks.append(
                        layer.mlp.register_forward_hook(
                            lambda m, i, o, li=li, side=side: caps[side].__setitem__(("mlp", li), o.float())
                        )
                    )
                enc = tok(
                    texts[b : b + 8],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_len,
                    add_special_tokens=False,
                ).to("cuda")
                with torch.no_grad():
                    model(**enc)
                for h in hooks:
                    h.remove()
            for kind in acc:
                for li in range(len(layers)):
                    d = (caps["self"][(kind, li)] - caps["other"][(kind, li)]).pow(2).mean(dim=(1, 2))
                    acc[kind][li] += d.sum().item()
        key = name or "base"
        res[key] = {k: [v / len(selfs) for v in vals] for k, vals in acc.items()}
        print(key, "o_proj mean", sum(res[key]["o_proj"]) / len(layers), flush=True)
        del model
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------ sampling


@app.function(**KW, timeout=3 * 60 * 60)
def sample(job):
    """job: {"base", "tag", "runs": [{"name", "adapter": None | name}], "prompts": [{"id", "messages", "seed"?,
    "prefill"?}], "max_tokens", "temperature", "max_model_len"?}. Each prompt gets its own seed (prompt["seed"], or
    run index * 1e6 + i). Writes /vol/out/<tag>/<run>.jsonl rows {id, reply, n_tokens, finish}; skips runs on disk."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    folder = Path(f"/vol/out/{job['tag']}")
    folder.mkdir(parents=True, exist_ok=True)
    todo = [r for r in job["runs"] if not (folder / f"{r['name']}.jsonl").exists()]
    if not todo:
        return {"skipped": len(job["runs"]), "seconds": 0}
    llm = LLM(
        job["base"],
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_model_len=job.get("max_model_len", 8192),
        seed=0,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    texts = [_chat(tok, p["messages"]) + p.get("prefill", "") for p in job["prompts"]]
    done = {}
    for k, run in enumerate(todo):
        lora = LoRARequest(run["adapter"], k + 1, f"/vol/adapters/{run['adapter']}") if run.get("adapter") else None
        sps = [
            SamplingParams(
                temperature=job["temperature"],
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_tokens=job["max_tokens"],
                seed=int(p.get("seed", k * 1_000_000 + i)),
            )
            for i, p in enumerate(job["prompts"])
        ]
        outs = llm.generate(texts, sps, lora_request=lora)
        rows = [
            {
                "id": p["id"],
                "reply": o.outputs[0].text,
                "n_tokens": len(o.outputs[0].token_ids),
                "finish": o.outputs[0].finish_reason,
            }
            for p, o in zip(job["prompts"], outs, strict=True)
        ]
        # Two-step prompts (a "trigger"): the reply just sampled becomes the assistant turn, the trigger is appended
        # as the next user turn, and the final reply is sampled; "initial" keeps the first reply.
        chain = [i for i, p in enumerate(job["prompts"]) if p.get("trigger")]
        if chain:
            texts2 = [
                _chat(
                    tok,
                    job["prompts"][i]["messages"]
                    + [
                        {"role": "assistant", "content": rows[i]["reply"].strip()},
                        {"role": "user", "content": job["prompts"][i]["trigger"]},
                    ],
                )
                for i in chain
            ]
            sps2 = [
                SamplingParams(
                    temperature=job["temperature"],
                    top_p=1.0,
                    top_k=-1,
                    repetition_penalty=1.0,
                    max_tokens=job.get("max_tokens2", job["max_tokens"]),
                    seed=int(sps[i].seed) + 500_000,
                )
                for i in chain
            ]
            outs2 = llm.generate(texts2, sps2, lora_request=lora)
            for i, o in zip(chain, outs2, strict=True):
                rows[i] = {
                    "id": rows[i]["id"],
                    "initial": rows[i]["reply"],
                    "reply": o.outputs[0].text,
                    "n_tokens": len(o.outputs[0].token_ids),
                    "finish": o.outputs[0].finish_reason,
                }
        (folder / f"{run['name']}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        vol.commit()
        done[run["name"]] = len(rows)
        print(f"{job['tag']}/{run['name']}: {len(rows)} replies ({time.time() - t0:.0f} s)", flush=True)
    return {"done": done, "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------ judge


@app.function(**KW, timeout=2 * 60 * 60)
def judge(job):
    """job: {"prompts": [str], "max_tokens"}. One user message each, greedy. Returns {"text": [...], "seconds"}."""
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(JUDGE, max_model_len=8192, gpu_memory_utilization=0.9, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in job["prompts"]
    ]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=job.get("max_tokens", 16)))
    return {"text": [o.outputs[0].text for o in outs], "seconds": time.time() - t0}


@app.function(**KW, timeout=3 * 60 * 60)
def bdiff_run(job):
    """job: {"pairs": [{"name", "base", "base_control", "tuned"}]}: paths under /vol to replies JSONL files (rows
    {id, reply}) on bdiff's default suite. Serves the judge on localhost:8000 and runs `bdiff run --base file:...
    --base-control file:... --tuned file:...` for each pair -> /vol/bdiff/<name>/."""
    import os
    import subprocess
    import urllib.request

    t0 = time.time()
    env = {**os.environ, "BDIFF_KEY": "local"}
    server = subprocess.Popen(
        [
            "vllm",
            "serve",
            JUDGE,
            "--port",
            "8000",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.9",
            "--api-key",
            "local",
            "--served-model-name",
            "judge",
        ],
        env=env,
    )
    for _ in range(180):
        try:
            req = urllib.request.Request("http://localhost:8000/v1/models", headers={"Authorization": "Bearer local"})
            urllib.request.urlopen(req, timeout=5)
            break
        except OSError:
            time.sleep(5)
    else:
        server.kill()
        raise RuntimeError("judge server did not start")
    results = {}
    Path("/tmp/bd").mkdir(exist_ok=True)
    try:
        for pair in job["pairs"]:
            out = Path(f"/vol/bdiff/{pair['name']}")
            cmd = [
                "bdiff",
                "run",
                "--base",
                f"file:{pair['base']}",
                "--base-control",
                f"file:{pair['base_control']}",
                "--tuned",
                f"file:{pair['tuned']}",
                "--judge",
                "judge",
                "--judge-url",
                "http://localhost:8000/v1",
                "--judge-key-env",
                "BDIFF_KEY",
                "--max-usd",
                "1",
                "-o",
                str(out),
            ]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd="/tmp/bd")
            print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
            results[pair["name"]] = {"returncode": p.returncode, "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
            vol.commit()
    finally:
        server.kill()
    return {"results": results, "seconds": time.time() - t0}
