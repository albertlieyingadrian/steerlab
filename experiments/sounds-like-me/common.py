"""Shared helpers for Sounds Like Me: data access, the LUAR voice ruler, the budget ledger."""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "raw"
LEDGER = DATA / "spend.jsonl"
OPENROUTER_CAP = (
    8.00  # hard cap for the whole series (user, 2026-11-20); reaching it means "switch to Modal", not "stop"
)


# ---------- data ----------
@lru_cache(None)
def corpus(name, split):
    return pd.read_csv(RAW / f"{name}_{split}.csv")


def authors():
    return json.loads((DATA / "authors.json").read_text())


def briefs():
    return json.loads((DATA / "briefs.json").read_text())


def texts(rec, which):
    """Real human texts for one author record: which in {samples, profile, ceiling}."""
    split = "train" if which == "samples" else "test"
    df = corpus(rec["corpus"], split)
    return [df.loc[i, "text"] for i in rec[which]]


def strip_greeting_signoff(t):
    """Drop a greeting first line and a sign-off tail so 'voice' can't pass on salutations alone."""
    lines = [l for l in t.strip().splitlines() if l.strip()]
    if len(lines) > 2 and len(lines[0].split()) <= 4:
        lines = lines[1:]
    while len(lines) > 2 and len(lines[-1].split()) <= 4:
        lines = lines[:-1]
    return "\n".join(lines)


# ---------- LUAR voice ruler ----------
@lru_cache(None)
def _luar():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")
    model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).eval()
    return tok, model, torch


def luar_embed(episodes, max_len=256):
    """episodes: list of lists of texts (each list = one 'author episode'). Returns (n, 512) unit vectors."""
    tok, model, torch = _luar()
    out = []
    for ep in episodes:
        enc = tok(list(ep), max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.reshape(1, len(ep), -1) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            e = model(**enc)
        e = e[0] if isinstance(e, tuple) else e
        out.append(torch.nn.functional.normalize(e, dim=-1)[0].numpy())
    return np.stack(out)


def cos(a, b):
    return float(np.dot(a, b))


class VoiceScale:
    """Places a 4-text episode's LUAR similarity to an author's profile on 0 (different human) .. 1 (same human).

    Unit = one contestant's 4 drafts for one author under one arm (amendment 2026-11-20: single texts gave
    AUROC 0.815 < 0.85; 4-text episodes give 0.980, results/ruler_luar*.json).
    floor   = mean similarity of the profile to 4-text episodes by other authors (floor pool)
    ceiling = mean similarity of the profile to the author's own held-out 4-text episodes (leave-one-out over 5)
    """

    def __init__(self, rec, floor_recs, mode="strict", n_floor=30):
        # strict: greeting/sign-off stripped AND whitespace collapsed (format-blind; LUAR reads layout otherwise,
        #         amendment 2026-11-20: flattening real author emails moved them from ~0.95 to ~-0.1)
        # loose:  raw text with its layout
        f = (lambda t: " ".join(strip_greeting_signoff(t).split())) if mode == "strict" else (lambda t: t)
        self.f = f
        self.profile = luar_embed([[f(t) for t in texts(rec, "profile")]])[0]
        ceil_txt = [f(t) for t in texts(rec, "ceiling")]
        ceil_eps = [ceil_txt[:i] + ceil_txt[i + 1 :] for i in range(len(ceil_txt))]
        self.ceiling = float(np.mean(luar_embed(ceil_eps) @ self.profile))
        flo = [[f(t) for t in texts(r, "ceiling")[:4]] for r in floor_recs[:n_floor] if r["author"] != rec["author"]]
        self.floor = float(np.mean(luar_embed(flo) @ self.profile))

    def sim(self, episode):
        return cos(luar_embed([[self.f(t) for t in episode]])[0], self.profile)

    def position(self, episode):
        return (self.sim(episode) - self.floor) / max(self.ceiling - self.floor, 1e-6)


# ---------- budget ledger ----------
def spent(provider="openrouter"):
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return sum(r["usd"] for r in rows if r["provider"] == provider)


def record_spend(provider, usd, note):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(dict(t=time.time(), provider=provider, usd=round(usd, 6), note=note)) + "\n")


def openrouter_allowed(estimated_usd):
    """True if this call fits under the hard cap. Callers must route to Modal when False."""
    return spent("openrouter") + estimated_usd <= OPENROUTER_CAP
