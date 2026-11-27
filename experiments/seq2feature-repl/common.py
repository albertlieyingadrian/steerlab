"""Shared loaders for the seq2feature replication: the released data, the released probe, scoring and bootstraps.

The seq2feature repo (github.com/patrickod32/seq2feature, commit f0616ac) is read from S2F_REPO, by default the clone
at ~/Albert-Workspace/Random/oss/patrickod32/seq2feature. Big arrays go to work/ (not committed); summaries go to
results/.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("S2F_REPO", "~/Albert-Workspace/Random/oss/patrickod32/seq2feature")).expanduser()
BUNDLE, GOLF = REPO / "relabel_bundle", REPO / "golf_final"
WORK, RESULTS = HERE / "work", HERE / "results"
WORK.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
DOMS = ["web", "code", "cot", "math", "chat"]
SAFETY_RE = (
    r"refus|harm|danger|weapon|deceiv|decept|\blie\b|lying|illegal|crime|criminal|violen|hack|password|secret|threat"
    r"|manipulat|attack|safety|ethic|unauthori|exploit|steal|abuse|malware|scam|fraud|toxic|offensive"
)


def load_bundle():
    """Span labels Y (bool, spans x 2,048), spans, documents, domains, test mask, captions, latent ids V."""
    T = np.load(BUNDLE / "targets.npz")
    V = T["V"]
    n = int(T["n_spans"])
    Y = np.zeros((n, len(V)), bool)
    Y[T["span"], T["vlat"]] = True
    spans = [json.loads(line) for line in gzip.open(BUNDLE / "spans.jsonl.gz", "rt")]
    docs = np.array([s["doc"] for s in spans])
    dom = np.array([DOMS.index(s["dom"]) for s in spans])
    te = np.array([(zlib.crc32(str(d).encode()) % 100) >= 70 for d in docs])
    caps_by_latent = {}
    for line in open(BUNDLE / "relabels.jsonl"):
        r = json.loads(line)
        caps_by_latent[r["latent"]] = r.get("caption", "")
    caps = [caps_by_latent.get(int(v), "") for v in V]
    return {"Y": Y, "spans": spans, "docs": docs, "dom": dom, "te": te, "caps": caps, "V": V, "T": T}


def build_probe(cfg, n_v):
    """The probe architecture, copied from probe/infer.py."""
    uni, bigh, cdim, fdim = cfg["UNI"], cfg["BIG_H"], cfg["CDIM"], cfg["FDIM"]
    nh, nl, maxlen = cfg["NHEAD"], cfg["NLAYER"], cfg["MAXLEN"]

    class Probe(nn.Module):
        def __init__(s):
            super().__init__()
            s.uni = nn.Embedding(uni, fdim)
            s.big = nn.Embedding(bigh, fdim)
            s.proj = nn.Linear(fdim, cdim, bias=False)
            s.pos = nn.Embedding(maxlen, cdim)
            s.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(cdim, nh, 4 * cdim, batch_first=True, dropout=0.1),
                nl,
                enable_nested_tensor=False,
            )
            s.head = nn.Linear(cdim, n_v)

        def forward(s, u, b, m):
            x = s.proj(s.uni(u) + s.big(b)) + s.pos(torch.arange(u.shape[1])[None])
            t = s.head(s.enc(x, src_key_padding_mask=~m))
            mk = t.masked_fill(~m.unsqueeze(-1), -1e30)
            return torch.logsumexp(mk, 1) - torch.log(m.sum(1, keepdim=True).float())

    return Probe().eval()


def load_probe(kind: str):
    """kind: fp32 | int8 (the shipped pickle, infer.py's dequantiser) | int4 (notebook 01's fq(w, 4) on fp32)."""
    ck = torch.load(GOLF / "mega_golf_probe.pth", map_location="cpu", weights_only=False)
    model = build_probe(ck["cfg"], len(ck["V"]))
    if kind == "int8":
        d = pickle.load(gzip.open(GOLF / "mega_golf_probe_int8.pkl.gz", "rb"))
        sd = {}
        for k, e in d["q"].items():
            if str(e[0]) == "i8":
                sd[k] = torch.tensor(e[1].astype(np.float32) * np.asarray(e[2], np.float32)[:, None])
            else:
                sd[k] = torch.tensor(np.asarray(e[1], np.float32))
        model.load_state_dict(sd)
    else:
        model.load_state_dict(ck["state_dict"])
        if kind == "int4":
            with torch.no_grad():
                for _, p in model.named_parameters():
                    qmax = 2 ** (4 - 1) - 1
                    s = p.abs().amax(dim=-1, keepdim=True) / qmax + 1e-12
                    p.copy_((p / s).round().clamp(-qmax, qmax) * s)
    model.eval()
    return model, ck


def bigram(ids, bigh):
    return np.concatenate([[0], (36313 * ids[1:] ^ 27191 * ids[:-1]) % bigh]).astype(np.int64)


@torch.no_grad()
def score_texts(model, ck, texts, bs=128):
    """Raw span scores (texts x 2,048), batched by length as notebook 02 does."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(GOLF / "mega_tok.model"))
    maxlen, bigh = ck["cfg"]["MAXLEN"], ck["cfg"]["BIG_H"]
    ids_all = [sp.encode(t)[:maxlen] or [0] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(ids_all[i]))
    S = np.zeros((len(texts), len(ck["V"])), np.float32)
    for b0 in range(0, len(order), bs):
        ch = order[b0 : b0 + bs]
        L = max(len(ids_all[i]) for i in ch)
        u = torch.zeros(len(ch), L, dtype=torch.long)
        b = torch.zeros(len(ch), L, dtype=torch.long)
        m = torch.zeros(len(ch), L, dtype=torch.bool)
        for r, i in enumerate(ch):
            ids = ids_all[i]
            u[r, : len(ids)] = torch.tensor(ids)
            b[r, : len(ids)] = torch.tensor(bigram(np.asarray(ids), bigh))
            m[r, : len(ids)] = True
        S[ch] = model(u, b, m).numpy()
    return S


def top5_per_span(scores, Y):
    """Per-span share of the 5 top-scored concepts that the SAE fired, and the picks."""
    top = np.argpartition(-scores, 5, axis=1)[:, :5]
    return np.take_along_axis(Y, top, 1).mean(1), top


def auc_vector(scores, Y):
    """Per-concept ROC AUC by ranks (ties averaged); NaN where a concept has one label only."""
    from scipy.stats import rankdata

    R = rankdata(scores, axis=0)
    pos = Y.sum(0).astype(np.float64)
    neg = Y.shape[0] - pos
    s = (R * Y).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s - pos * (pos + 1) / 2) / (pos * neg)
    auc[(pos == 0) | (neg == 0)] = np.nan
    return auc


def doc_bootstrap(per_span, docs, n_boot=2000, seed=0):
    """95% percentile interval of a per-span mean, resampling documents."""
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(docs, return_inverse=True)
    sums = np.bincount(inv, weights=per_span, minlength=len(ud))
    cnts = np.bincount(inv, minlength=len(ud))
    out = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.integers(0, len(ud), len(ud))
        out[k] = sums[pick].sum() / cnts[pick].sum()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def doc_resamples(docs, n_boot, seed=0):
    """Row-index arrays for document-level resamples."""
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(docs, return_inverse=True)
    rows_of = [np.where(inv == j)[0] for j in range(len(ud))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(ud), len(ud))
        yield np.concatenate([rows_of[j] for j in pick])
