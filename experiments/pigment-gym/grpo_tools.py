"""
A minimal GRPO (Group Relative Policy Optimization), hand-written so every line is visible.
The toy run (6_toy_grpo.py) walks through it on a toy reward; the laptop run (7_train_laptop.py) reuses it with the painting reward.

One training step:

    prompt ──► model samples G answers ──► reward each ──► advantage = reward - mean(group)
                                                                 │
    update LoRA weights so that above-average answers ◄──────────┘
    become more likely and below-average ones less likely

No value network (PPO's critic), no learned reward model: the group IS the baseline. Like grading
on a curve: only how you did relative to the others answering the same question counts.

TRL's GRPOTrainer does the same thing with more features (vLLM generation, several updates per batch,
logging). The knobs that matter here are mirrored below: num_generations (G), beta (KL weight),
epsilon (clip range), scale_rewards ("none" = don't divide by std), loss averaged over tokens.
"""

import math

import torch


def build_lora_model(model_name, device, rank=16, alpha=32, seed=0, dtype=torch.float32):
    """Load a chat model (float32 by default; the H100 run (modal_train.py) uses bfloat16 on an H100) and wrap it with LoRA
    adapters on every linear layer.

    LoRA = instead of changing a 896x896 weight matrix W, learn two thin matrices A (16x896) and
    B (896x16) and use W + B@A. About 2% of the parameters train; the original weights stay frozen,
    which also gives us the frozen reference model for free (just switch the adapter off).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # generation appends on the right, so any padding goes on the left
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config).to(device)
    return model, tokenizer


def chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_group(model, tokenizer, prompt_text, group_size, max_new_tokens, temperature=1.0, top_p=1.0, seed=0):
    """Sample `group_size` completions for one prompt. Returns (texts, input_ids, mask).

    input_ids: (G, prompt_len + completion_len): the prompt, then the completion.
    mask: same shape; 1 for completion tokens up to and including the end token, 0 for the prompt and
          for the padding `generate` adds after a sequence has finished.
    """
    model.eval()
    torch.manual_seed(seed)
    encoded = tokenizer([prompt_text] * group_size, return_tensors="pt").to(model.device)
    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = encoded["input_ids"].shape[1]
    completion = output[:, prompt_len:]
    end_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    mask = torch.zeros_like(output)
    for row in range(completion.shape[0]):
        length = completion.shape[1]  # ran out of tokens: every generated token counts
        for position, token in enumerate(completion[row].tolist()):
            if token in end_ids:
                length = position + 1  # include the end token: "stop here" is a choice too
                break
        mask[row, prompt_len:prompt_len + length] = 1
    texts = tokenizer.batch_decode(completion, skip_special_tokens=True)
    return texts, output, mask


def token_logprobs(model, input_ids):
    """Log-probability the model gives each actual next token. Shape (B, L-1).

    Position t holds log P(token t+1 | tokens 0..t). log_softmax over a 152k-word vocabulary is large,
    so callers pass small micro-batches.
    """
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards, scale_rewards="none"):
    """advantage_i = reward_i - mean(rewards). With scale_rewards="group", also divide by the group's std.

    Why "none"? Dividing by std makes a group that scored 0.50, 0.51, 0.49 look as important as one
    that scored 0.0 and 1.0: tiny, noisy differences get blown up into full-size updates. The Dr. GRPO
    paper (Liu et al., 2025) and the HF painting run both found that leaving it out works better.
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)
    advantages = rewards - rewards.mean()
    if scale_rewards == "group":
        advantages = advantages / (rewards.std() + 1e-4)
    return advantages


def grpo_update(model, optimizer, input_ids, mask, advantages, beta=0.02, epsilon=0.2, micro_batch=2):
    """One optimizer step on one group. Returns a dict of metrics.

    loss per token = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)  +  beta * KL(policy || reference)
    averaged over all completion tokens in the group.

    We do one update per batch of samples, so the "old" log-probs equal the current ones and ratio == 1.
    The clip only matters when a batch is reused for several updates (TRL's num_iterations > 1). It's
    kept so the formula matches the papers.
    """
    model.train()
    optimizer.zero_grad()
    token_masks = mask[:, 1:].float()  # align with token_logprobs: position t predicts token t+1
    total_tokens = token_masks.sum().clamp(min=1)
    advantages = advantages.to(input_ids.device)
    loss_sum, kl_sum = 0.0, 0.0
    for start in range(0, input_ids.shape[0], micro_batch):
        ids = input_ids[start:start + micro_batch]
        token_mask = token_masks[start:start + micro_batch]
        adv = advantages[start:start + micro_batch].unsqueeze(1)
        logp = token_logprobs(model, ids)
        old = logp.detach()
        ratio = torch.exp(logp - old)
        per_token = -torch.min(ratio * adv, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv)
        if beta > 0:
            with torch.no_grad(), model.disable_adapter():  # reference model = the base weights, LoRA off
                ref = token_logprobs(model, ids)
            # The "k3" KL estimator: exp(ref - logp) - (ref - logp) - 1. Always >= 0, low variance.
            kl = torch.exp(ref - logp) - (ref - logp) - 1
            per_token = per_token + beta * kl
            kl_sum += float((kl.detach() * token_mask).sum())
        loss = (per_token * token_mask).sum() / total_tokens
        loss.backward()  # gradients add up across micro-batches, as if it were one big batch
        loss_sum += float(loss.detach())
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    return {"loss": loss_sum, "kl": kl_sum / float(total_tokens), "grad_norm": float(grad_norm)}


def trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan
