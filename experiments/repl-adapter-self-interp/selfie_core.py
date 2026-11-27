"""Core of the adapter self-interpretation replication: soft-token injection, adapter training, label generation,
conversation parsing and the SAE hit test. Mirrors agencyenterprise/selfie-adapters (MIT) where noted.

Replicates "Learning Self-Interpretation from Interpretability Artifacts" (arXiv 2602.10352) on Llama-3.1-8B-Instruct
with the Goodfire layer-19 SAE.

Injection (repo training/model.py): the template's placeholder `<|reserved_special_token_0|>` appears twice; the
adapter output replaces the token embedding at every placeholder position (layer 0). The loss is the mean
cross-entropy over the label tokens of each sequence (label + '"' + <|eot_id|>), averaged over the batch.

Inference (repo label_generator.py): z = adapter(s * h/|h|) with the adapter's own normalisation switched off, so an
external scale s multiplies the learned one. Untrained SelfIE is z = s * h/|h|.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn
from torch.nn import functional as F

TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    'What is the meaning of "<|reserved_special_token_0|>"?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    'The meaning of "<|reserved_special_token_0|>" is "'
)
PLACEHOLDER = "<|reserved_special_token_0|>"
EOT = "<|eot_id|>"

# generation scoring prompt (paper Appendix A.6, repo configs)
CONV_SYSTEM = (
    "You are a helpful AI assistant who generates EXTREMELY SHORT example conversations. The conversations are between "
    "a user and an assistant, and have the following format:\n[USER] I'm a user.\n[ASSISTANT] I'm the assistant."
)
CONV_USER = (
    "Produce a VERY SHORT conversation which exhibits '_'\nDo not include any other text in your response. "
    "Start immediately with the conversation."
)
TRAINED_SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]  # repo example_label_generator.json
UNTRAINED_SCALES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]  # repo qwen_scaling untrained configs


class ScalarAffine(nn.Module):
    """f(x) = exp(log_scale) * x + bias (repo projection.ScalarAffineProjection)."""

    def __init__(self, dim: int, init_scale: float = 5.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor([math.log(init_scale)]))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, normalize: bool = True):
        if normalize:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.exp(self.log_scale) * x + self.bias


class Identity(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x, normalize: bool = True):
        if normalize:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return self.scale * x


def load_released(path: str) -> ScalarAffine:
    from safetensors.torch import load_file

    sd = load_file(path)
    a = ScalarAffine(sd["bias"].shape[0])
    a.load_state_dict({"log_scale": sd["log_scale"], "bias": sd["bias"]})
    return a


# ---------------------------------------------------------------------------------------------------------------------
# injection


class Injector:
    """Holds the template embeddings and placeholder positions for one model and tokenizer."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        ids = tok(TEMPLATE, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ph = tok.convert_tokens_to_ids(PLACEHOLDER)
        self.ids = ids
        self.pos = (ids == ph).nonzero().flatten().tolist()
        assert len(self.pos) == 2, self.pos
        self.emb = model.get_input_embeddings()

    def template_embeds(self, soft: torch.Tensor) -> torch.Tensor:
        """[B, d] soft tokens -> [B, T, d] template embeddings with the soft token at both placeholders."""
        dev = self.emb.weight.device
        with torch.no_grad():
            base = self.emb(self.ids.to(dev))
        out = base.unsqueeze(0).repeat(soft.shape[0], 1, 1)
        soft = soft.to(dev, base.dtype)
        for p in self.pos:
            out[:, p, :] = soft
        return out

    def label_loss(self, soft: torch.Tensor, labels: list[str], per_seq: bool = False):
        """Mean over label tokens per sequence, then over the batch (repo model.compute_loss)."""
        dev = self.emb.weight.device
        tgt = [self.tok(lab + '"' + EOT, add_special_tokens=False)["input_ids"] for lab in labels]
        T = len(self.ids)
        L = max(len(t) for t in tgt)
        B = len(labels)
        tpl = self.template_embeds(soft)
        pad = torch.zeros(B, L, dtype=torch.long)
        mask = torch.zeros(B, T + L, dtype=torch.long)
        mask[:, :T] = 1
        for i, t in enumerate(tgt):
            pad[i, : len(t)] = torch.tensor(t)
            mask[i, T : T + len(t)] = 1
        with torch.no_grad():
            lab_emb = self.emb(pad.to(dev))
        x = torch.cat([tpl, lab_emb], 1)
        h = self.model.model(inputs_embeds=x, attention_mask=mask.to(dev)).last_hidden_state
        hs = h[:, T - 1 : T - 1 + L]
        logits = self.model.lm_head(hs).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pad.to(dev).reshape(-1), reduction="none").view(B, L)
        m = mask[:, T:].to(dev).float()
        seq = (ce * m).sum(1) / m.sum(1)
        return seq if per_seq else seq.mean()

    @torch.no_grad()
    def generate(self, soft: torch.Tensor, max_new_tokens=30, temperature=0.7, top_p=0.9, seed=None) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)
        x = self.template_embeds(soft)
        att = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        out = self.model.generate(
            inputs_embeds=x,
            attention_mask=att,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.tok.eos_token_id,
        )
        texts = self.tok.batch_decode(out, skip_special_tokens=True)
        return [clean_label(t) for t in texts]


def clean_label(t: str) -> str:
    """Cut at the closing quote (repo strip_last_quote) and trim."""
    return t.split('"')[0].strip()


# ---------------------------------------------------------------------------------------------------------------------
# conversations (repo reward_system._parse_meta_conversation, same behaviour)


def parse_conversation(text: str) -> list[dict]:
    parts = []
    for line in text.split("\n"):
        parts.extend(re.split(r"(\[(?:USER|ASSISTANT)\])", line))
    conv, role, buf = [], None, []

    def flush():
        if role is not None and buf:
            c = " ".join(buf).strip()
            if c:
                conv.append({"role": "user" if role == "USER" else "assistant", "content": c})

    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"\[(USER|ASSISTANT)\]", p)
        if m:
            flush()
            role, buf = m.group(1), []
        elif role is not None:
            buf.append(p)
        elif not conv:
            role, buf = "USER", [p]
        else:
            flush()
            role = "ASSISTANT" if conv[-1]["role"] == "user" else "USER"
            buf = [p]
    flush()
    if not conv and text.strip():
        return [{"role": "assistant", "content": text}]
    return conv


# ---------------------------------------------------------------------------------------------------------------------
# Goodfire SAE (encode = ReLU(W_enc x + b_enc), Goodfire's own reference code)


class GoodfireSAE:
    def __init__(self, sd: dict, device: str):
        self.W_enc = sd["encoder_linear.weight"].to(device, torch.float32)  # [d_sae, d]
        self.b_enc = sd["encoder_linear.bias"].to(device, torch.float32)
        self.W_dec = sd["decoder_linear.weight"].to(device, torch.float32)  # [d, d_sae]
        self.b_dec = sd["decoder_linear.bias"].to(device, torch.float32)

    def encode(self, x, idx=None):
        x = x.float()
        if idx is None:
            return F.relu(x @ self.W_enc.T + self.b_enc)
        return F.relu(x @ self.W_enc[idx].T + self.b_enc[idx])

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def decoder_vectors(self, idx) -> torch.Tensor:
        return self.W_dec[:, idx].T.contiguous()


def layer_output_hook(model, layer: int):
    """Capture the output of decoder block `layer` (= hidden_states[layer + 1]); returns (handle, store)."""
    store = {}

    def hook(_m, _i, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    return model.model.layers[layer].register_forward_hook(hook), store


def hits_from_acts(acts: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """acts [B, T, K] (right-padded); a hit = any activation > 0 after position 0 (repo is_hit)."""
    T = acts.shape[1]
    pos = torch.arange(T, device=acts.device)[None, :]
    valid = (pos >= 1) & (pos < torch.tensor(lengths, device=acts.device)[:, None])
    return ((acts > 0) & valid[:, :, None]).any(1)  # [B, K]
