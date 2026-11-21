"""Tests 9 and 10 on Modal: per-author voice vector vs per-author LoRA vs few-shot prompting on Qwen3-8B. COSTS MONEY.

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/probes-vs-interp/modal_voice.py

author(payload) runs one author in one container:
- voice vectors from the author's first n samples (n in vec_ns): mean residual (output of decoder layer `layer`, last
  400 tokens of each sample formatted as an assistant reply) minus the same mean over the model's own default drafts;
  added at `layer` with coef = strength x the mean token norm there (Sounds Like Me Part C's recipe and dev choice);
- LoRAs (rank r, attention projections) trained on the author's first n samples (n in lora_ns), each timed;
- generation for every condition, 4 briefs each, plus each output's NLL under the untouched model.
Build costs are GPU-seconds: one forward pass per sample for a vector, the training loop for a LoRA.
"""

from __future__ import annotations

import time

import modal

MODEL = "Qwen/Qwen3-8B"
app = modal.App("tl-ep1-voice")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("tl-ep1-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "hf_transfer==0.1.9",
        "numpy<2.3",
        "peft==0.17.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


@app.function(
    image=image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=3 * 60 * 60,
)
def author(p: dict) -> dict:
    """p: {name, samples, default_texts, wrong_samples, k0_messages[4], shot_messages{n: [4]}, conditions, vec_ns,
    lora_ns, layer, strength, lora: {lr, steps, r}, rand_seeds, max_new_tokens, seed}."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    L = p["layer"]

    def as_reply(t):
        msgs = [
            {"role": "user", "content": "Write an email."},
            {"role": "assistant", "content": t},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)

    def mean_res(texts):
        acc, norms = [], []
        for t in texts:
            enc = tok(as_reply(t), return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            with torch.no_grad():
                h = model(**enc, output_hidden_states=True).hidden_states[L + 1][0, -400:].float()
            acc.append(h.mean(0))
            norms.append(float(h.norm(dim=-1).mean()))
        return torch.stack(acc).mean(0), float(np.mean(norms))

    default, norm = mean_res(p["default_texts"])
    vecs, build_sec = {}, {}
    for n in p["vec_ns"]:
        torch.cuda.synchronize()
        t0 = time.time()
        m, _ = mean_res(p["samples"][:n])
        torch.cuda.synchronize()
        build_sec[f"vec{n}"] = time.time() - t0
        vecs[f"vec{n}"] = m - default
    vecs["wrong"] = mean_res(p["wrong_samples"])[0] - default
    for s in p["rand_seeds"]:
        g = torch.Generator(device="cpu").manual_seed(s)
        vecs[f"rand{s}"] = torch.randn(default.shape, generator=g).cuda()
    coef = p["strength"] * norm
    state = {"v": None}

    def hook(_m, _i, out):
        if state["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["v"].to(h.dtype)
        return (h, *tuple(out[1:])) if isinstance(out, tuple) else h

    def gen(m, messages):
        prompts = [
            tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for x in messages
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=6144).to("cuda")
        with torch.no_grad():
            g = m.generate(
                **enc,
                max_new_tokens=p["max_new_tokens"],
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.0,
                pad_token_id=tok.eos_token_id,
            )
        return [
            tok.decode(g[k, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip() for k in range(len(prompts))
        ]

    torch.manual_seed(p.get("seed", 0))
    outs = {}
    handle = model.model.layers[L].register_forward_hook(hook)
    for c in p["conditions"]:
        if c.startswith("lora"):
            continue
        state["v"] = None
        msgs = p["k0_messages"]
        if c.startswith("shot"):
            msgs = p["shot_messages"][c[4:]]
        elif c != "none":
            state["v"] = coef * vecs[c] / vecs[c].norm()
        outs[c] = gen(model, msgs)
    state["v"] = None
    handle.remove()
    adapter_bytes = None
    for n in p["lora_ns"]:  # train on the first n samples (loss on the reply only), generate, strip the adapter
        cfg = LoraConfig(
            r=p["lora"]["r"],
            lora_alpha=2 * p["lora"]["r"],
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        pm = get_peft_model(model, cfg)
        opt = torch.optim.AdamW([q for q in pm.parameters() if q.requires_grad], lr=p["lora"]["lr"])
        data = []
        for t in p["samples"][:n]:
            text = as_reply(t)
            full = tok(text, return_tensors="pt", truncation=True, max_length=768).input_ids
            head = text.split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
            lab = full.clone()
            lab[:, : len(tok(head).input_ids)] = -100
            data.append((full.cuda(), lab.cuda()))
        pm.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for step in range(p["lora"]["steps"]):
            ids, lab = data[step % len(data)]
            pm(input_ids=ids, labels=lab).loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        build_sec[f"lora{n}"] = time.time() - t0
        pm.eval()
        outs[f"lora{n}"] = gen(pm, p["k0_messages"])
        adapter_bytes = sum(q.numel() * 2 for name, q in pm.named_parameters() if "lora_" in name)
        model = pm.unload()
        model.eval()
    nll = {}
    for c, texts in outs.items():
        vals = []
        for t in texts:
            ids = tok(t or ".", return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
            with torch.no_grad():
                vals.append(float(model(ids, labels=ids).loss) if ids.shape[1] > 1 else None)
        nll[c] = vals
    return dict(
        name=p["name"],
        outputs=outs,
        nll=nll,
        build_sec=build_sec,
        coef=coef,
        vector_bytes=int(default.numel() * 2),
        adapter_bytes=adapter_bytes,
        seconds=round(time.time() - t_start, 1),
    )
