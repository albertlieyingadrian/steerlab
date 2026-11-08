"""
Shared helpers for the six area experiments (scalable oversight, AI control, model organisms, interpretability,
safeguards, AI welfare). Each script puts this folder on sys.path and does `import tools as t`.

What lives here:
- where things are: downloads, activations, adapters and any prompt or reply text go to
  ~/.cache/steerlab/residency-101/ (never committed); small results go to results/<area>/;
- loading a chat model, and a printed self-check;
- asking a chat model a True/False question and reading its answer from the next-token logits (no sampling);
- reading the residual stream at the last prompt token;
- a probe (standardise, then logistic regression), accuracy with a bootstrap interval, and PGR;
- greedy generation, sandboxed execution of model-written code, and LoRA fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = Path.home() / ".cache" / "steerlab" / "residency-101"

SEED = 0
WEAK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STRONG_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def rel(path: Path) -> str:
    """A short path for printing, so output never shows a home folder."""
    path = Path(path).resolve()
    for root, label in ((ROOT, "experiments/residency-101"), (Path.home(), "~")):
        try:
            return f"{label}/{path.relative_to(root)}"
        except ValueError:
            pass
    return path.name


def check(name, condition):
    """Print a pass/fail self-check line and return whether it passed."""
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(name=STRONG_MODEL, device=None):
    """Load a Hugging Face chat model and its tokenizer for inference: float32 up to 1.6B parameters, bfloat16 above.

    Returns (model, tokenizer, device).
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    config = AutoConfig.from_pretrained(name)
    approx_params = 12 * config.num_hidden_layers * config.hidden_size**2 + config.vocab_size * config.hidden_size
    dtype = torch.float32 if approx_params <= 1.6e9 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device).eval()
    return model, tokenizer, device


def fetch(url: str, dest: Path, sha256: str | None = None) -> Path:
    """Download url to dest once; verify its SHA-256 when one is given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha256 and digest != sha256:
        raise ValueError(f"{rel(dest)}: SHA-256 {digest} does not match the pinned {sha256}")
    return dest


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


# ---- asking a True/False question ----


def tf_prompt(tokenizer, statement: str) -> str:
    """The chat prompt, ending right where the model's one-word answer starts."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Is the following statement true or false?\n\n{statement}\n\nAnswer with one word: True or False."
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def p_true(model, tokenizer, device, statements, batch_size=16):
    """P(True) / (P(True) + P(False)) for the next token, one number per statement. Greedy reading, no sampling."""
    import torch

    true_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]
    tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(statements), batch_size):
        prompts = [tf_prompt(tokenizer, s) for s in statements[start : start + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float()
        pair = torch.stack([logits[:, true_id], logits[:, false_id]], dim=1)
        out.extend(torch.softmax(pair, dim=1)[:, 0].tolist())
    return np.array(out)


def last_token_acts(model, tokenizer, device, statements, layers, batch_size=16):
    """Residual stream at the last prompt token: {layer: array (n, d_model)}.

    Uses the transformers convention: hidden_states[0] is the embeddings and hidden_states[L] is the residual
    stream after block L (so for Qwen2.5-1.5B's 28 blocks, L runs from 1 to 28).
    """
    import torch

    tokenizer.padding_side = "left"
    acts = {layer: [] for layer in layers}
    for start in range(0, len(statements), batch_size):
        prompts = [tf_prompt(tokenizer, s) for s in statements[start : start + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hidden = model(**batch, output_hidden_states=True).hidden_states
        for layer in layers:
            acts[layer].append(hidden[layer][:, -1, :].float().cpu().numpy())
    return {layer: np.concatenate(chunks) for layer, chunks in acts.items()}


# ---- probes and scores ----


def fit_probe(x_train, y_train, c=0.1):
    """Standardise, then L2 logistic regression. Returns a fitted sklearn pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(x_train, y_train)


def accuracy_ci(correct, n_boot=2000, seed=SEED):
    """Mean of a 0/1 array with a 95% bootstrap interval: (mean, low, high)."""
    correct = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    return float(correct.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pgr_ci(weak_correct, w2s_correct, ceiling_correct, n_boot=2000, seed=SEED):
    """Performance gap recovered, (w2s - weak) / (ceiling - weak), with a paired bootstrap over test items."""
    weak, w2s, ceil = (np.asarray(a, dtype=float) for a in (weak_correct, w2s_correct, ceiling_correct))

    def pgr(idx):
        gap = ceil[idx].mean() - weak[idx].mean()
        return (w2s[idx].mean() - weak[idx].mean()) / gap if gap > 0 else np.nan

    rng = np.random.default_rng(seed)
    n = len(weak)
    boots = np.array([pgr(rng.integers(0, n, n)) for _ in range(n_boot)])
    point = pgr(np.arange(n))
    if np.isnan(point):  # the student with true labels is no better than the teacher: PGR is undefined
        return float("nan"), float("nan"), float("nan")
    boots = boots[~np.isnan(boots)]
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(ci) -> str:
    mean, low, high = ci
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


# ---- generation and sandboxed execution (AI control) ----


def generate(model, tokenizer, device, conversations, max_new_tokens=320, batch_size=16):
    """Greedy replies, one per conversation (a list of chat messages). repetition_penalty=1.0 on purpose: Qwen's
    generation_config sets 1.1, which do_sample=False alone leaves on."""
    import torch

    tokenizer.padding_side = "left"
    replies = []
    for start in range(0, len(conversations), batch_size):
        prompts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations[start : start + batch_size]
        ]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        replies.extend(tokenizer.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True))
    return replies


def extract_code(reply: str) -> str:
    """The first ```python block in a reply, or the whole reply if there is none."""
    import re

    match = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    return (match.group(1) if match else reply).strip()


def run_python(source: str, timeout=5) -> tuple[bool, str]:
    """Run model-written code in a separate, isolated Python process (-I, empty temp dir, time limit).
    Returns (exited cleanly, last line of stderr)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.py"
        path.write_text(source)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)], check=False, cwd=tmp, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return proc.returncode == 0, (proc.stderr.strip().splitlines() or [""])[-1]


def digit_score(model, tokenizer, device, conversations, batch_size=8):
    """Expected value of the next token over the digits 0-9, renormalised over those ten tokens: a 0-9 score."""
    import torch

    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    tokenizer.padding_side = "left"
    scores = []
    for start in range(0, len(conversations), batch_size):
        prompts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations[start : start + batch_size]
        ]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, digit_ids].float()
        probs = torch.softmax(logits, dim=1)
        scores.extend((probs * torch.arange(10, device=probs.device)).sum(dim=1).tolist())
    return np.array(scores)


# ---- LoRA fine-tuning on chat replies (model organisms) ----


def lora_model(model, r=16, alpha=32):
    """Wrap a causal LM with a LoRA adapter on all attention and MLP projections."""
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return get_peft_model(
        model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
    )


def sft(
    model, tokenizer, device, examples, epochs=1, batch_size=8, lr=2e-4, max_reply_tokens=64, seed=SEED, log_every=25
):
    """Supervised fine-tuning on (messages, reply) pairs, loss on the reply tokens only. Returns the loss history."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(examples))
        for start in range(0, len(order), batch_size):
            ids, labels = [], []
            for i in order[start : start + batch_size]:
                messages, reply = examples[i]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                r_ids = tokenizer(reply, add_special_tokens=False)["input_ids"][:max_reply_tokens]
                r_ids = [*r_ids, tokenizer.eos_token_id]
                ids.append(p_ids + r_ids)
                labels.append([-100] * len(p_ids) + r_ids)
            width = max(len(x) for x in ids)
            pad = tokenizer.pad_token_id or tokenizer.eos_token_id
            input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids], device=device)
            label_ids = torch.tensor([[-100] * (width - len(x)) + x for x in labels], device=device)
            mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=device)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=label_ids).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            history.append(float(loss.detach()))
            if step % log_every == 0:
                print(f"    step {step}: loss {np.mean(history[-log_every:]):.3f}")
    model.eval()
    return history
