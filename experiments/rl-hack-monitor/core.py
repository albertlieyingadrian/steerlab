"""
Model plumbing shared by every Modal job: load Qwen with LoRA, sample replies in batches, one GRPO update, and the
activation readouts the monitors use.

Layer convention: layer l is the OUTPUT of decoder block l, which is hidden_states[l + 1] in Hugging Face's numbering
(the same convention as the persona-vectors note, so its vectors can be reused as they are).
"""

import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = [4, 8, 12, 16, 20, 24]  # the layers whose activations are saved at every checkpoint
MAX_NEW_TOKENS = 320


def load(model_name=MODEL, lora=True, seed=0, rank=16, alpha=32):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
    model.generation_config.repetition_penalty = 1.0  # Qwen's config sets 1.1; keep sampling plain
    if lora:
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)
        for p in model.parameters():  # LoRA weights in float32 for a stable optimizer; the base stays bf16
            if p.requires_grad:
                p.data = p.data.float()
    return model, tok


def chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample(model, tok, prompts, n, seed, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS, chunk=256):
    """n replies per prompt. Returns a list (one per prompt) of n dicts {text, prompt_ids, reply_ids}."""
    model.eval()
    torch.manual_seed(seed)
    flat = [p for p in prompts for _ in range(n)]
    out = []
    end_ids = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    for start in range(0, len(flat), chunk):
        batch = flat[start : start + chunk]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.0,
            remove_invalid_values=True,
        )
        width = enc["input_ids"].shape[1]
        for row in range(len(batch)):
            prompt_ids = enc["input_ids"][row][enc["attention_mask"][row].bool()].tolist()
            reply = gen[row, width:].tolist()
            for i, t in enumerate(reply):
                if t in end_ids:
                    reply = reply[: i + 1]  # keep the end token: stopping is a choice too
                    break
            out.append(
                {"text": tok.decode(reply, skip_special_tokens=True), "prompt_ids": prompt_ids, "reply_ids": reply}
            )
    return [out[i * n : (i + 1) * n] for i in range(len(prompts))]


def _pad(seqs, pad_id):
    """Left-pad token lists. Returns ids, attention mask and position ids (so padding does not shift positions)."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(s)
        mask[i, width - len(s) :] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    return ids.cuda(), mask.cuda(), pos.cuda()


def grpo_step(model, tok, optimizer, rollouts, advantages, micro=16):
    """One policy-gradient step on a batch of rollouts (one update per batch, so the PPO ratio is 1 and the clip
    never binds; no KL term). Loss = -sum(advantage * log p(token)) over reply tokens / number of reply tokens."""
    model.train()
    optimizer.zero_grad()
    total = sum(len(r["reply_ids"]) for r in rollouts)
    loss_sum = 0.0
    for start in range(0, len(rollouts), micro):
        chunk = rollouts[start : start + micro]
        adv = advantages[start : start + micro]
        if all(a == 0 for a in adv):
            continue
        # Right padding here: with left padding, the padded rows attend to nothing, give NaN, and NaN x 0 in the
        # masked loss poisons the gradient (this crashed the first pilots at step 10).
        seqs = [r["prompt_ids"] + r["reply_ids"] for r in chunk]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        reply_mask = torch.zeros((len(seqs), width - 1))
        for i, (s, r) in enumerate(zip(seqs, chunk, strict=False)):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
            reply_mask[i, len(r["prompt_ids"]) - 1 : len(s) - 1] = 1  # position t predicts token t+1
        ids, mask, reply_mask = ids.cuda(), mask.cuda(), reply_mask.cuda()
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        logp = torch.where(reply_mask > 0, logp, torch.zeros_like(logp))
        a = torch.tensor(adv, device=logp.device, dtype=logp.dtype).unsqueeze(1)
        loss = -(a * logp * reply_mask).sum() / total
        loss.backward()
        loss_sum += float(loss.detach())
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)):  # never apply a non-finite update
        optimizer.zero_grad()
        return {"loss": loss_sum, "grad_norm": grad_norm, "skipped": True}
    optimizer.step()
    return {"loss": loss_sum, "grad_norm": grad_norm}


@torch.no_grad()
def readouts(model, tok, prompt_ids_list, reply_ids_list=None, layers=LAYERS, batch=32):
    """Activations at `layers` (float16 numpy):
    prompt_last [n, len(layers), hidden]: the last prompt token, where the reply starts
    reply_mean  [n, len(layers), hidden]: mean over the reply's tokens (only if replies are given)
    """
    import numpy as np

    model.eval()
    n = len(prompt_ids_list)
    hidden = model.config.hidden_size
    last = np.zeros((n, len(layers), hidden), dtype=np.float16)
    mean = np.zeros((n, len(layers), hidden), dtype=np.float16) if reply_ids_list is not None else None
    for start in range(0, n, batch):
        p = prompt_ids_list[start : start + batch]
        r = reply_ids_list[start : start + batch] if reply_ids_list is not None else [[] for _ in p]
        ids, mask, pos = _pad([a + b for a, b in zip(p, r, strict=False)], tok.pad_token_id)
        hs = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states
        width = ids.shape[1]
        for j, layer in enumerate(layers):
            h = hs[layer + 1].float()
            for i in range(len(p)):
                k = len(r[i])
                last[start + i, j] = h[i, width - k - 1].cpu().numpy()
                if mean is not None and k:
                    mean[start + i, j] = h[i, width - k :].mean(0).cpu().numpy()
    return last, mean


def hack_string(env, exploit):
    """The text whose appearance is the hack, as the model would append it after its reply."""
    return " " + (env.TAG if exploit == "tag" else env.PHRASE)


def content_ids(tok, reply_ids):
    """The reply without trailing end tokens: the position after it is where a hack string would be appended."""
    ends = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    ids = list(reply_ids)
    while ids and ids[-1] in ends:
        ids.pop()
    return ids


@torch.no_grad()
def end_readouts(model, tok, prompt_ids_list, content_list, hack_ids, layers=LAYERS, batch=16):
    """At the last token of each reply's content (where an appended hack would start):
    acts   [n, len(layers), hidden] float16: the residual stream there
    hack_lp [n]: log P(hack_ids | prompt + content), the grey-box propensity to append the hack now
    """
    import numpy as np

    model.eval()
    n, hidden = len(prompt_ids_list), model.config.hidden_size
    acts = np.zeros((n, len(layers), hidden), dtype=np.float16)
    hack_lp = np.zeros(n, dtype=np.float32)
    m = len(hack_ids)
    for start in range(0, n, batch):
        seqs = [
            p + c + hack_ids
            for p, c in zip(prompt_ids_list[start : start + batch], content_list[start : start + batch], strict=True)
        ]
        ids, mask, pos = _pad(seqs, tok.pad_token_id)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        width = ids.shape[1]
        logp = torch.log_softmax(out.logits[:, width - m - 1 : width - 1].float(), -1)
        tgt = ids[:, width - m :]
        hack_lp[start : start + len(seqs)] = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(1).cpu().numpy()
        for j, layer in enumerate(layers):
            acts[start : start + len(seqs), j] = out.hidden_states[layer + 1][:, width - m - 1].float().cpu().numpy()
    return acts, hack_lp
