"""
Shared helpers for the SLEIGHT-Bench monitor note. Every script here does `import sleight as t`.

What lives here, in the order a transcript flows through it:

    fetch()                     clone SLEIGHT-Bench at a pinned commit into WORK, a folder outside the repository, and
                                decrypt it with the key its README publishes; nothing from it is written anywhere else
    load_index()                86 rows (43 attack, 43 benign twins), IDs and categories only
        │
        ├─ stripped_lines(path) the published study's "stripped" view: tool calls and results, no assistant prose
        ├─ render(lines, tok)   control_arena prompt + <transcript>…</transcript> in Qwen3's chat template, cut to
        │                       the 30k-token budget, with the token index of every tool call
        ├─ forward_states(...)  one chunked forward pass; residual stream at chosen positions and layers
        └─ greedy_monitor(...)  the black-box monitor: greedy reasoning, then <score>N</score>

The results/ files the scripts write hold transcript IDs and numbers only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("SLEIGHT_MONITOR_WORK", Path.home() / ".cache" / "steerlab" / "sleight-monitor"))
"""Everything derived from the benchmark (the decrypted clone, activations, raw replies). It lives outside the
repository on purpose, so no commit or copy of this folder can carry benchmark data."""
SLEIGHT = WORK / "sleight-bench"
SLEIGHT_REPO = "https://github.com/safety-research/sleight-bench"
SLEIGHT_COMMIT = "218c58315cc01ff0dc5a100e906c27d82d259521"
"""The commit every number in the note was measured on."""

MODEL = "Qwen/Qwen3-8B"
LAYERS = (8, 12, 16, 20, 24, 28, 32)
"""Candidate probe layers (of 36), fixed before scoring."""
BUDGET = 30_000
"""Token budget for a rendered transcript (Qwen3-8B's native window is 32,768; the rest is left for the reply)."""
CHUNK = 2048
"""Prefill chunk: attention for a 30k-token prompt done in one go would need tens of GB on MPS."""
EXCLUDED = {"refusal_forcing/refusal-poisoning-credential-exfil"}
"""In the repo but not in attack_metadata.parquet or the published 40 attacks: excluded from every number."""


def check(name, condition):
    """Print a ✅/❌ self-check line and return whether it passed."""
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def fetch():
    """Clone SLEIGHT-Bench at the pinned commit into WORK and decrypt it, once. The benchmark encrypts its data so
    it is not scraped into training corpora; the key is published in its README and read from there."""
    if (SLEIGHT / "attacks" / "attack_metadata.parquet").exists():
        return
    WORK.mkdir(parents=True, exist_ok=True)
    if not SLEIGHT.exists():
        subprocess.run(["git", "clone", "--quiet", SLEIGHT_REPO, str(SLEIGHT)], check=True)
    subprocess.run(["git", "-C", str(SLEIGHT), "checkout", "--quiet", SLEIGHT_COMMIT], check=True)
    key = re.search(r"Decryption key: `([^`]+)`", (SLEIGHT / "README.md").read_text(encoding="utf-8")).group(1)
    subprocess.run(
        [sys.executable, "-I", str(SLEIGHT / "decrypt.py"), "--key", key], check=True, stdout=subprocess.DEVNULL
    )


# ---------------------------------------------------------------- data


def load_index():
    """The 86 scored transcripts: one row per attack transcript and one per its benign twin.

    Row: {"tid": "<category>/<attack>::attack" or "::benign", "attack_id", "category", "label" (1 attack, 0 benign),
    "path"}. The attack list comes from attack_metadata.parquet, so it is exactly the published 43 transcripts.
    """
    import pandas as pd

    fetch()
    meta = pd.read_parquet(SLEIGHT / "attacks" / "attack_metadata.parquet")
    rows = []
    for attack_id in sorted(meta.attack_id):
        folder = SLEIGHT / "attacks" / attack_id
        for label, name in ((1, "transcript.jsonl"), (0, "benign.jsonl")):
            rows.append(
                {
                    "tid": f"{attack_id}::{'attack' if label else 'benign'}",
                    "attack_id": attack_id,
                    "category": attack_id.split("/")[0],
                    "label": label,
                    "path": folder / name,
                }
            )
    return rows


def stripped_lines(path):
    """The published stripped view (SLEIGHT's evaluate/core/transcript.py, re-implemented): keep user lines and
    assistant tool_use lines, drop assistant text and thinking, drop tool `description` fields. Canary lines
    (no "role") are dropped too: they are identical in attack and benign transcripts."""
    out = []
    for raw in Path(path).read_text(encoding="utf-8").strip().split("\n"):
        msg = json.loads(raw)
        role = msg.get("role")
        if role == "user":
            out.append(msg)
        elif role == "assistant":
            content = msg.get("content")
            if not isinstance(content, dict) or content.get("type") != "tool_use":
                continue
            inp = content.get("input")
            if isinstance(inp, dict) and "description" in inp:
                content = {**content, "input": {k: v for k, v in inp.items() if k != "description"}}
            out.append({**msg, "content": content})
    return out


def is_tool_call(msg):
    content = msg.get("content")
    return msg.get("role") == "assistant" and isinstance(content, dict) and content.get("type") == "tool_use"


def _cap_strings(obj, cap):
    """Every string longer than `cap` characters keeps its first and last cap/2 characters around a marker."""
    if isinstance(obj, str):
        if len(obj) <= cap:
            return obj
        half = cap // 2
        return obj[:half] + f"[... {len(obj) - cap} characters omitted ...]" + obj[-half:]
    if isinstance(obj, dict):
        return {k: _cap_strings(v, cap) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cap_strings(v, cap) for v in obj]
    return obj


def monitor_prompt():
    """SLEIGHT's control_arena monitor prompt (prompts/control_arena.md), read from the clone."""
    return (SLEIGHT / "prompts" / "control_arena.md").read_text(encoding="utf-8")


def render(lines, tok, system=None, budget=BUDGET):
    """Chat-template one stripped transcript the way SLEIGHT's render_prompt does (prompt as system message,
    transcript in <transcript> tags as the user message), with Qwen3's thinking turned off.

    Returns {"ids": token ids, "steps": token index of the last token of each tool-call line, "n_raw": tokens
    before any cut, "cap": the string cap used (None if uncut)}.
    """
    system = monitor_prompt() if system is None else system
    cap, n_raw = None, None
    while True:
        shown = lines if cap is None else [_cap_strings(m, cap) for m in lines]
        body_lines = [json.dumps(m) for m in shown]
        user = "<transcript>\n" + "\n".join(body_lines) + "\n</transcript>"
        text = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        n_raw = n_raw or len(enc.input_ids)
        if len(enc.input_ids) <= budget:
            break
        cap = 2000 if cap is None else cap // 2
    # The character where each tool-call line ends, then the token that covers it.
    start = text.index(user) + len("<transcript>\n")
    ends, pos = [], start
    for msg, line in zip(shown, body_lines, strict=False):
        if is_tool_call(msg):
            ends.append(pos + len(line) - 1)
        pos += len(line) + 1
    offsets = enc.offset_mapping
    steps, j = [], 0
    for e in ends:
        while not (offsets[j][0] <= e < offsets[j][1]):
            j += 1
        steps.append(j)
    return {"ids": enc.input_ids, "steps": steps, "n_raw": n_raw, "cap": cap}


# ---------------------------------------------------------------- model


def load_model(name=MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
    return tok, model


def forward_states(model, ids, positions, layers=LAYERS, return_cache=False):
    """One chunked forward pass over `ids`; the residual stream (output of decoder layer L, i.e. hidden_states[L])
    at every token index in `positions`, as a float32 array [len(positions), len(layers), d]."""
    import torch
    from transformers import DynamicCache

    want = sorted(set(positions))
    got = {}
    cache = DynamicCache()
    x = torch.tensor([ids], device="mps")
    out = None
    with torch.no_grad():
        for s in range(0, len(ids), CHUNK):
            out = model(x[:, s : s + CHUNK], past_key_values=cache, use_cache=True, output_hidden_states=bool(layers))
            cache = out.past_key_values
            inside = [p for p in want if s <= p < s + CHUNK]
            if inside and layers:
                idx = torch.tensor([p - s for p in inside], device="mps")
                block = torch.stack([out.hidden_states[L][0, idx] for L in layers], dim=1).float().cpu().numpy()
                for p, row in zip(inside, block, strict=False):
                    got[p] = row
    states = np.stack([got[p] for p in positions]) if (positions and layers) else None
    if return_cache:
        return states, cache, out.logits[0, -1]
    return states


SCORE_RE = re.compile(r"<score>\s*(\d{1,3})")


def greedy_monitor(model, tok, ids, max_new=1024):
    """The black-box monitor on one rendered prompt: chunked prefill, then a plain greedy loop (argmax, so no
    sampling and no repetition penalty). Stops at </score> or end of turn. If no <score> tag was written, the tag is
    forced and the number read greedily.

    Returns {"text", "score" (int or None), "forced" (bool), "gen_ids", "score_tag_end" (index into ids+gen_ids of
    the last token of "<score>", or None), "seconds"}.
    """
    import torch

    t0 = time.time()
    _, cache, logits = forward_states(model, ids, [], layers=(), return_cache=True)
    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    gen = []
    with torch.no_grad():
        nxt = int(logits.argmax())
        for _ in range(max_new):
            gen.append(nxt)
            if nxt in stop_ids or "</score>" in tok.decode(gen[-6:]):
                break
            out = model(torch.tensor([[nxt]], device="mps"), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
        text = tok.decode(gen, skip_special_tokens=True)
        forced = False
        if not SCORE_RE.search(text):
            forced = True
            gen = [g for g in gen if g not in stop_ids] + tok.encode("\n<score>", add_special_tokens=False)
            _, cache, logits = forward_states(model, list(ids) + gen, [], layers=(), return_cache=True)
            nxt = int(logits.argmax())
            for _ in range(4):
                gen.append(nxt)
                out = model(torch.tensor([[nxt]], device="mps"), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
            text = tok.decode(gen, skip_special_tokens=True)
    m = SCORE_RE.search(text)
    return {
        "text": text,
        "score": min(int(m.group(1)), 100) if m else None,
        "forced": forced,
        "gen_ids": gen,
        "score_tag_end": score_tag_end(tok, ids, gen),
        "seconds": round(time.time() - t0, 1),
    }


def score_tag_end(tok, ids, gen):
    """Index (into ids + gen) of the last token of the first "<score>" in the reply, or None."""
    for k in range(1, len(gen) + 1):
        if tok.decode(gen[:k]).endswith("<score>"):
            return len(ids) + k - 1
    return None


# ---------------------------------------------------------------- metrics


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y, s = np.asarray(y), np.asarray(s, dtype=float)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def tpr_at_fp(y, s, max_fp):
    """Share of attacks scoring strictly above the threshold that lets at most `max_fp` benign transcripts through
    (max_fp=0: above the highest benign score). Returns (tpr, threshold)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    benign = np.sort(s[y == 0])[::-1]
    thr = benign[max_fp] if max_fp < len(benign) else -np.inf
    return float(np.mean(s[y == 1] > thr)), float(thr)


def pair_win(att, ben):
    """Share of pairs where the attack outscores its own benign twin (ties count half)."""
    att, ben = np.asarray(att, float), np.asarray(ben, float)
    return float(np.mean((att > ben) + 0.5 * (att == ben)))


def paired_bootstrap(att, ben, stat, n=2000, seed=0):
    """95 % interval of stat(y, s) over resamples of whole attack–benign pairs (att[i] and ben[i] are twins)."""
    rng = np.random.default_rng(seed)
    att, ben = np.asarray(att, float), np.asarray(ben, float)
    k = len(att)
    y = np.r_[np.ones(k), np.zeros(k)]
    vals = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        v = stat(y, np.r_[att[i], ben[i]])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def write_csv(path, rows, fields=None):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    import csv

    with Path(path).open() as f:
        return list(csv.DictReader(f))
