"""A toy video world model with a KV cache: the world, the model, the cache policies and the rollout.

The world is a 32 x 32 map of tiles, each one of 8 colours drawn at random, with a new map for every episode, so
nothing off-screen can be guessed: the only way to draw a place you have seen before is to remember it. A camera sees
an 8 x 8 window and moves one patch (2 x 2 tiles) up, down, left or right per step. A frame is 16 patch tokens.

The model is a small causal transformer that generates the next frame in one pass, the way causal video diffusion
models generate a chunk: the 16 "query" tokens of frame t (carrying its camera position and the action) attend to
the cached keys and values of the clean frames 0..t-1 and to each other. After a frame is generated it is encoded as
clean tokens and appended to the cache. Training uses the same attention pattern on whole sequences.

Cache policies act on that cache between steps (see `Policy`). Every cache entry is one patch token of one frame, so
each entry has a known map location: that is what makes "did the model attend to the frame that saw this place?" a
question with a ground-truth answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- world

MAP = 32  # tiles per side
VIEW = 8  # tiles per side of the camera window
P = 2  # tiles per patch side
K = 8  # colours
MP = MAP // P  # 16 patches per map side
VP = VIEW // P  # 4 patches per view side
NTOK = VP * VP  # 16 tokens per frame
NPOS = MP - VP + 1  # 13 camera positions per axis
SUB = P * P  # 4 tiles per patch
ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right as (dx, dy) in patches
START = 4  # the "action" id given to frame 0


def make_episodes(n: int, T: int, rng: np.random.Generator, p_return: float = 0.5):
    """n episodes of T frames. Returns tiles [n,T,16,4] int, pose [n,T,2] (x,y), act [n,T], loc [n,T,16] int.

    The camera walks between waypoints. Each new waypoint is, with probability `p_return`, a camera position the
    episode has already visited (so it leaves a place and comes back) and otherwise a random position; it walks
    there one patch at a time, choosing at random between the moves that bring it closer. Revisits therefore happen
    at gaps from 2 steps to the whole episode.
    """
    maps = rng.integers(0, K, size=(n, MAP, MAP))
    pose = np.zeros((n, T, 2), dtype=np.int64)
    act = np.full((n, T), START, dtype=np.int64)
    pose[:, 0] = rng.integers(0, NPOS, size=(n, 2))
    target = rng.integers(0, NPOS, size=(n, 2))
    step = np.array([ACTIONS[a] for a in range(4)])  # [4,2]
    rows = np.arange(n)
    for t in range(1, T):
        cur = pose[:, t - 1]
        arrived = (cur == target).all(1)
        if arrived.any():
            back = rng.random(n) < p_return
            past = pose[rows, rng.integers(0, t, size=n)]
            fresh = rng.integers(0, NPOS, size=(n, 2))
            new = np.where(back[:, None], past, fresh)
            target = np.where(arrived[:, None], new, target)
            # a waypoint equal to the current position: step to a random neighbour instead
        nxt = cur[:, None, :] + step[None]  # [n,4,2]
        valid = ((nxt >= 0) & (nxt < NPOS)).all(-1)
        closer = np.abs(nxt - target[:, None]).sum(-1) < np.abs(cur - target).sum(-1)[:, None]
        ok = np.where(closer.any(1)[:, None], closer & valid, valid)
        a = np.where(ok, rng.random((n, 4)), -1.0).argmax(1)
        pose[:, t] = cur + step[a]
        act[:, t] = a
    tiles = np.zeros((n, T, NTOK, SUB), dtype=np.int64)
    loc = np.zeros((n, T, NTOK), dtype=np.int64)
    rows = np.arange(n)[:, None]
    for j in range(NTOK):
        gx = pose[:, :, 0] + j % VP  # patch coords on the map
        gy = pose[:, :, 1] + j // VP
        loc[:, :, j] = gy * MP + gx
        for s in range(SUB):
            tiles[:, :, j, s] = maps[rows, gy * P + s // P, gx * P + s % P]
    return tiles, pose, act, loc


def gaps(loc: np.ndarray) -> np.ndarray:
    """For every token of every frame t>=1: steps since its map location was last in view (0 = never seen)."""
    n, T, _ = loc.shape
    out = np.zeros((n, T, NTOK), dtype=np.int64)
    last = np.full((n, MP * MP), -1, dtype=np.int64)
    for t in range(T):
        if t > 0:
            seen = np.take_along_axis(last, loc[:, t], axis=1)
            out[:, t] = np.where(seen >= 0, t - seen, 0)
        np.put_along_axis(last, loc[:, t], t, axis=1)
    return out


# ---------------------------------------------------------------- model


@dataclass
class Cfg:
    d: int = 128
    layers: int = 4
    heads: int = 4
    T: int = 48


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.h, self.dh = c.heads, c.d // c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.mlp = nn.Sequential(nn.Linear(c.d, 4 * c.d), nn.GELU(), nn.Linear(4 * c.d, c.d))

    def split(self, x):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v  # [B,H,N,dh]

    def finish(self, x, o):
        B, H, N, dh = o.shape
        x = x + self.proj(o.transpose(1, 2).reshape(B, N, H * dh))
        return x + self.mlp(self.ln2(x))


class WorldModel(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        d = c.d
        self.sub = nn.Embedding(SUB * K, d)  # (sub-tile slot, colour)
        self.slot = nn.Embedding(NTOK, d)
        self.px, self.py = nn.Embedding(NPOS, d), nn.Embedding(NPOS, d)
        self.time = nn.Embedding(c.T, d)
        self.act = nn.Embedding(5, d)
        self.kind = nn.Embedding(2, d)  # 0 clean, 1 query
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, SUB * K)
        self.register_buffer("subidx", torch.arange(SUB) * K, persistent=False)
        self.register_buffer("slots", torch.arange(NTOK), persistent=False)

    # tiles [B,F,16,4], pose [B,F,2], act [B,F], t [B,F] (frame index)
    def embed_clean(self, tiles, pose, act, t):
        return self.sub(tiles + self.subidx).sum(-2) + self._common(pose, act, t) + self.kind.weight[0]

    def embed_query(self, pose, act, t):
        return self._common(pose, act, t) + self.kind.weight[1]

    def _common(self, pose, act, t):
        g = self.px(pose[..., 0]) + self.py(pose[..., 1]) + self.act(act) + self.time(t)  # [B,F,d]
        return g[:, :, None, :] + self.slot(self.slots)  # [B,F,16,d]

    def logits(self, h):
        B, N, _ = h.shape
        return self.head(self.lnf(h)).view(B, N // NTOK, NTOK, SUB, K)

    def forward_seq(self, tiles, pose, act, mask=None, return_attn=False, return_kv=False):
        """Teacher-forced pass: clean frames 0..T-1, then queries for frames 1..T-1. Returns logits [B,T-1,16,4,K].
        `mask` is [N,N] or [B,1,N,N] (True = may attend); default `seq_mask`."""
        B, T = act.shape
        dev = tiles.device
        tt = torch.arange(T, device=dev).expand(B, T)
        xc = self.embed_clean(tiles, pose, act, tt).reshape(B, T * NTOK, -1)
        xq = self.embed_query(pose[:, 1:], act[:, 1:], tt[:, 1:]).reshape(B, (T - 1) * NTOK, -1)
        x = torch.cat([xc, xq], 1)
        if mask is None:
            mask = seq_mask(T, dev)
        attns, kvs = [], []
        for blk in self.blocks:
            q, k, v = blk.split(x)
            if return_kv:  # keys and values of the clean tokens, heads concatenated: [B, T*16, d]
                kvs.append(
                    (k[:, :, : T * NTOK].transpose(1, 2).flatten(2), v[:, :, : T * NTOK].transpose(1, 2).flatten(2))
                )
            if return_attn:
                s = (q @ k.transpose(-1, -2)) / math.sqrt(blk.dh)
                a = s.masked_fill(~mask, float("-inf")).softmax(-1)
                attns.append(a[:, :, T * NTOK :, : T * NTOK])  # query -> clean
                o = a @ v
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = blk.finish(x, o)
        out = self.logits(x[:, T * NTOK :])
        if return_kv:
            return out, kvs
        return (out, attns) if return_attn else out


def seq_mask(T: int, dev) -> torch.Tensor:
    """Boolean [N,N] mask, True = may attend. Clean frame t sees clean frames <= t; the query of frame u sees clean
    frames < u and the other queries of frame u."""
    fc = torch.arange(T, device=dev).repeat_interleave(NTOK)
    fq = torch.arange(1, T, device=dev).repeat_interleave(NTOK)
    top = torch.cat([fc[None, :] <= fc[:, None], torch.zeros(len(fc), len(fq), dtype=torch.bool, device=dev)], 1)
    bot = torch.cat([fc[None, :] < fq[:, None], fq[None, :] == fq[:, None]], 1)
    return torch.cat([top, bot], 0)


# ---------------------------------------------------------------- step-by-step rollout with a KV cache


class Cache:
    """Per-layer keys/values of clean tokens, a keep mask, the frame and map location of every entry, the attention
    each entry has received so far (for H2O), and the pooled summaries of evicted frames (for `pool`).

    Evicted entries are masked out rather than deleted, which gives the same attention output as deleting them."""

    def __init__(self, layers: int):
        self.k = [None] * layers
        self.v = [None] * layers
        self.keep = None  # [B,N] bool
        self.frame = None  # [N] frame index of each entry
        self.loc = None  # [B,N] map location of each entry
        self.score = None  # [B,N] accumulated attention received
        self.pk = [None] * layers  # pooled keys [B,H,M,dh]
        self.pv = [None] * layers
        self.pooled_frames: list[int] = []

    def append(self, layer, k, v):
        self.k[layer] = k if self.k[layer] is None else torch.cat([self.k[layer], k], 2)
        self.v[layer] = v if self.v[layer] is None else torch.cat([self.v[layer], v], 2)

    def compact(self):
        """Physically drop entries that every row has evicted (all of them at batch size 1)."""
        live = self.keep.any(0)
        if bool(live.all()):
            return
        for li in range(len(self.k)):
            self.k[li], self.v[li] = self.k[li][:, :, live], self.v[li][:, :, live]
        self.keep, self.frame, self.loc, self.score = (
            self.keep[:, live],
            self.frame[live],
            self.loc[:, live],
            self.score[:, live],
        )

    def kept_entries(self):
        return self.keep.sum(1).float() + len(self.pooled_frames)


def attend(blk, x, cache: Cache, layer: int, extra_keep=None):
    """x [B,16,d] attends to the kept cache entries, the pooled entries, and its own 16 tokens."""
    q, k, v = blk.split(x)
    B = x.shape[0]
    ks, vs, bs = [], [], []
    ncache = 0
    if cache.k[layer] is not None:
        keep = cache.keep if extra_keep is None else cache.keep & extra_keep
        ks.append(cache.k[layer])
        vs.append(cache.v[layer])
        bs.append(torch.where(keep, 0.0, float("-inf"))[:, None, None, :])
        ncache = cache.k[layer].shape[2]
    if cache.pk[layer] is not None:
        ks.append(cache.pk[layer])
        vs.append(cache.pv[layer])
        bs.append(torch.full((B, 1, 1, cache.pk[layer].shape[2]), math.log(NTOK), device=x.device))
    ks.append(k)
    vs.append(v)
    bs.append(torch.zeros(B, 1, 1, NTOK, device=x.device))
    s = (q @ torch.cat(ks, 2).transpose(-1, -2)) / math.sqrt(blk.dh) + torch.cat(bs, 3)
    a = s.softmax(-1)
    return blk.finish(x, a @ torch.cat(vs, 2)), k, v, a[..., :ncache]


@torch.no_grad()
def step_clean(model, cache: Cache, tiles, pose, act, t, loc):
    """Encode frame t as clean tokens and append them. tiles [B,16,4], pose [B,2], act [B], loc [B,16]."""
    B = tiles.shape[0]
    dev = tiles.device
    tt = torch.full((B, 1), t, device=dev, dtype=torch.long)
    x = model.embed_clean(tiles[:, None], pose[:, None], act[:, None], tt)[:, 0]
    new = []
    for li, blk in enumerate(model.blocks):
        x, k, v, a = attend(blk, x, cache, li)
        new.append((k, v))
        if cache.score is not None and a.shape[-1]:
            cache.score += a.sum((1, 2))
    for li, (k, v) in enumerate(new):
        cache.append(li, k, v)
    ones = torch.ones(B, NTOK, dtype=torch.bool, device=dev)
    fr = torch.full((NTOK,), t, device=dev, dtype=torch.long)
    zeros = torch.zeros(B, NTOK, device=dev)
    if cache.keep is None:
        cache.keep, cache.frame, cache.loc, cache.score = ones, fr, loc.clone(), zeros
    else:
        cache.keep = torch.cat([cache.keep, ones], 1)
        cache.frame = torch.cat([cache.frame, fr])
        cache.loc = torch.cat([cache.loc, loc], 1)
        cache.score = torch.cat([cache.score, zeros], 1)


@torch.no_grad()
def step_query(model, cache: Cache, pose, act, t, extra_keep=None):
    """Generate frame t from the cache. Returns logits [B,16,4,K] and per-layer attention [B,H,16,Ncache]."""
    B = pose.shape[0]
    tt = torch.full((B, 1), t, device=pose.device, dtype=torch.long)
    x = model.embed_query(pose[:, None], act[:, None], tt)[:, 0]
    attns = []
    for li, blk in enumerate(model.blocks):
        x, _, _, a = attend(blk, x, cache, li, extra_keep=extra_keep)
        attns.append(a)
        if cache.score is not None and a.shape[-1]:
            cache.score += a.sum((1, 2))
    return model.logits(x)[:, 0], attns


# ---------------------------------------------------------------- cache policies


@dataclass
class Policy:
    """How the cache is trimmed after each frame is appended. `budget` is in frames (16 entries each); every policy
    keeps the newest frame. When frame t is generated the cache holds at most `budget` frames' worth of entries.

    full     keep everything.
    window   keep the last `budget` frames.
    sink     keep frame 0 (the attention sink, as in StreamingLLM) and the last budget-1 frames.
    h2o      keep the last ceil(budget/2) frames, then fill the rest of the 16*budget entries with the older entries
             that have received the most attention so far (heavy hitters, as in H2O), entry by entry.
    pool     keep the last budget-1 frames whole; each older frame becomes one entry holding the mean key and mean
             value of its 16 tokens (with a log 16 bias, so it competes like 16 tokens); at most 16 pooled entries.
    random   keep the newest frame, then a random subset of older entries, 16*budget entries in all (the control).
    """

    kind: str
    budget: int = 0

    @property
    def name(self):
        return "full" if self.kind == "full" else f"{self.kind}{self.budget}"


def apply_policy(pol: Policy, cache: Cache, t: int, gen: torch.Generator | None = None):
    if pol.kind == "full":
        return
    fr = cache.frame
    B = cache.keep.shape[0]
    cap = NTOK * pol.budget
    if pol.kind == "window":
        cache.keep &= (fr > t - pol.budget)[None, :]
    elif pol.kind == "sink":
        cache.keep &= ((fr > t - (pol.budget - 1)) | (fr == 0))[None, :]
    elif pol.kind == "h2o":
        recent = (fr > t - math.ceil(pol.budget / 2))[None, :].expand(B, -1)
        room = cap - int(recent[0].sum())
        cand = cache.keep & ~recent
        keep_old = torch.zeros_like(cache.keep)
        if room > 0 and bool(cand.any()):
            sc = torch.where(cand, cache.score, float("-inf"))
            top = sc.topk(min(room, sc.shape[1]), dim=1).indices
            keep_old.scatter_(1, top, True)
            keep_old &= cand
        cache.keep = (cache.keep & recent) | keep_old
    elif pol.kind == "random":
        newest = (fr == t)[None, :].expand(B, -1)
        cand = cache.keep & ~newest
        over = cand.sum(1) - (cap - NTOK)
        if int(over.max()) > 0:
            noise = torch.rand(cache.keep.shape, generator=gen, device=cache.keep.device)
            noise = torch.where(cand, noise, float("inf"))
            rank = noise.argsort(1).argsort(1)
            cache.keep &= ~(cand & (rank < over.clamp(min=0)[:, None]))
    elif pol.kind == "pool":
        w = pol.budget - 1
        leaving = t - w  # the frame that just left the window
        if leaving >= 0:
            idx = (fr == leaving).nonzero().squeeze(1)
            for li in range(len(cache.k)):
                pk = cache.k[li][:, :, idx].mean(2, keepdim=True)
                pv = cache.v[li][:, :, idx].mean(2, keepdim=True)
                cache.pk[li] = pk if cache.pk[li] is None else torch.cat([cache.pk[li], pk], 2)[:, :, -NTOK:]
                cache.pv[li] = pv if cache.pv[li] is None else torch.cat([cache.pv[li], pv], 2)[:, :, -NTOK:]
            cache.pooled_frames = ([*cache.pooled_frames, leaving])[-NTOK:]
        cache.keep &= (fr > t - w)[None, :]
    else:
        raise ValueError(pol.kind)


# ---------------------------------------------------------------- one evaluation run


@torch.no_grad()
def rollout(model, tiles, pose, act, loc, pol: Policy, mode: str, seed: int = 0, record_attn=False):
    """Run an episode batch step by step under a cache policy.

    mode "observed": the cache is filled with the true frames (one-step prediction at every step).
    mode "dream": only frame 0 is true; every later frame is the model's own prediction, fed back.

    Returns preds [B,T,16,4] (frame 0 = truth), shown [B,T,16,4] (what went into the cache), in_cache [B,T,16]
    (an exact cache entry held this token's map location when frame t was generated), kept [T] (mean entries in
    the cache at each step), and, with record_attn, the attention to each cache entry, summed over layers and
    heads and averaged over the 16 queries' own rows: attn[t] [B,16,N].
    """
    B, T = act.shape
    dev = tiles.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    cache = Cache(len(model.blocks))
    shown = tiles.clone()
    preds = tiles.clone()
    in_cache = torch.zeros(B, T, NTOK, dtype=torch.bool, device=dev)
    kept = torch.zeros(T, device=dev)
    attn = {}
    step_clean(model, cache, shown[:, 0], pose[:, 0], act[:, 0], 0, loc[:, 0])
    apply_policy(pol, cache, 0, gen)
    for t in range(1, T):
        held = (cache.loc[:, None, :] == loc[:, t, :, None]) & cache.keep[:, None, :]
        in_cache[:, t] = held.any(-1)
        kept[t] = cache.kept_entries().mean()
        lg, attns = step_query(model, cache, pose[:, t], act[:, t], t)
        preds[:, t] = lg.argmax(-1)
        if mode == "dream":
            shown[:, t] = preds[:, t]
        if record_attn:
            attn[t] = torch.stack(attns).sum((0, 2)).cpu()  # [B,16,N]
        step_clean(model, cache, shown[:, t], pose[:, t], act[:, t], t, loc[:, t])
        apply_policy(pol, cache, t, gen)
        if not record_attn:
            cache.compact()
    return {"preds": preds, "shown": shown, "in_cache": in_cache, "kept": kept, "attn": attn}


def last_shown(shown: torch.Tensor, loc: torch.Tensor):
    """For each frame t>=1 and token: the content most recently shown at its map location before frame t
    (-1 where never shown). shown [B,T,16,4], loc [B,T,16]."""
    B, T = loc.shape[:2]
    mem = torch.full((B, MP * MP, SUB), -1, dtype=torch.long, device=shown.device)
    out = torch.full_like(shown, -1)
    bi = torch.arange(B, device=shown.device)[:, None]
    for t in range(T):
        if t > 0:
            out[:, t] = mem[bi, loc[:, t]]
        mem[bi, loc[:, t]] = shown[:, t]
    return out
