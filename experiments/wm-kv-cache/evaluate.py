"""Score one trained world model: cache policies, attention attribution, a probe on the cache, and causal eviction.

    python evaluate.py --ckpt results/ckpt_s1.pt --out results/eval_s1.json

Every test uses fresh episodes from a generator seed (77) that training never used. Terms (defined once, used below):

- gap: steps since a token's map location was last in view. gap 1 = it was in the previous frame (a "shift" token,
  copied from one frame back); gap >= 2 = a "revisit" token (the camera left that place and came back); never seen =
  "new" (cannot be predicted; chance is 1/8 per tile).
- revisit consistency: the fraction of a revisit token's 4 tiles that match what was last shown at that place.
  In "observed" mode the cache holds the true frames, so "last shown" is the truth (one-step prediction). In "dream"
  mode the model's own frames go back into the cache, so it is self-consistency over a free rollout.
- in cache: when the frame is generated, some kept cache entry is an earlier view of that exact map location.

The JSON keeps per-episode hit and tile counts so that confidence intervals can be taken over episodes and seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import functools
import operator

from wm_core import NTOK, Cfg, Policy, WorldModel, gaps, last_shown, make_episodes, rollout, seq_mask

GAP_BINS = [(2, 3), (4, 7), (8, 15), (16, 31), (32, 47)]
BUDGETS = [2, 4, 8, 16]
KINDS = ["window", "sink", "h2o", "pool", "random"]


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = WorldModel(Cfg(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["state"])
    return m, ck


def episodes(n, T, seed, dev):
    rng = np.random.default_rng(seed)
    tiles, pose, act, loc = make_episodes(n, T, rng)
    g = gaps(loc)
    return [torch.from_numpy(x).to(dev) for x in (tiles, pose, act, loc, g)]


def counts(hit, sel):
    """Per-episode [hits, tiles] for a token selection. hit [B,T,16] tile-hit fractions, sel [B,T,16] bool."""
    h = (hit * sel).sum((1, 2)) * 4
    n = sel.sum((1, 2)) * 4
    return torch.stack([h, n], 1).round().long().cpu().tolist()


def score_rollout(r, loc, g):
    ref = last_shown(r["shown"], loc)
    hit = (r["preds"] == ref).float().mean(-1)  # tile-hit fraction per token, vs last shown
    hit[:, 0] = 0
    rev = g >= 2
    out = {
        "revisit": counts(hit, rev),
        "shift": counts(hit, g == 1),
        "revisit_in": counts(hit, rev & r["in_cache"]),
        "revisit_out": counts(hit, rev & ~r["in_cache"]),
    }
    for lo, hi in GAP_BINS:
        b = (g >= lo) & (g <= hi)
        out[f"gap{lo}_{hi}"] = counts(hit, b)
        out[f"gap{lo}_{hi}_in"] = counts(hit, b & r["in_cache"])
        out[f"gap{lo}_{hi}_out"] = counts(hit, b & ~r["in_cache"])
    seen = g >= 1
    out["err_by_t"] = (1 - (hit * seen).sum((0, 2)) / seen.sum((0, 2)).clamp(min=1)).cpu().tolist()
    out["rev_err_by_t"] = (1 - (hit * rev).sum((0, 2)) / rev.sum((0, 2)).clamp(min=1)).cpu().tolist()
    out["kept_by_t"] = r["kept"].cpu().tolist()
    return out


def policy_sweep(model, data, bs):
    tiles, pose, act, loc, g = data
    pols = [Policy("full")] + [Policy(k, b) for b in BUDGETS for k in KINDS]
    res = {}
    for mode in ["observed", "dream"]:
        for pol in pols:
            parts = []
            for i in range(0, tiles.shape[0], bs):
                sl = slice(i, i + bs)
                r = rollout(model, tiles[sl], pose[sl], act[sl], loc[sl], pol, mode, seed=i)
                parts.append(score_rollout(r, loc[sl], g[sl]))
            merged = {}
            for k in parts[0]:
                if k.endswith("by_t"):
                    merged[k] = np.mean([p[k] for p in parts], 0).round(5).tolist()
                else:
                    merged[k] = functools.reduce(operator.iadd, (p[k] for p in parts), [])
            res[f"{mode}/{pol.name}"] = merged
            rv = np.array(merged["revisit"]).sum(0)
            print(mode, pol.name, "revisit", round(rv[0] / rv[1], 4), flush=True)
    return res


def timing(model, data, dev, reps=3):
    """Wall time per generated frame at batch size 1, with evicted entries physically removed."""
    tiles, pose, act, loc, _g = data
    out = {}
    pols = [Policy("full")] + [Policy(k, b) for b in [4, 16] for k in KINDS]
    for pol in pols:
        ts = []
        for i in range(reps + 1):
            sl = slice(i, i + 1)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            rollout(model, tiles[sl], pose[sl], act[sl], loc[sl], pol, "observed", seed=i)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i:
                ts.append((time.perf_counter() - t0) / (tiles.shape[1] - 1))
        out[pol.name] = round(1000 * float(np.median(ts)), 3)
    return out


@torch.no_grad()
def attention_attribution(model, data, bs=16, n=128):
    """At revisit tokens (observed mode, full cache): how much attention lands on earlier views of the same place.

    Per layer, averaged over heads and over revisit tokens: attention on any earlier view of the place ("sight"),
    on the most recent earlier view ("last_sight"), on the previous frame ("prev_frame"), the share that "sight"
    would get if attention were spread evenly over the cache ("uniform_sight"), and how often the single most
    attended entry is an earlier view of the place ("top1_sight")."""
    tiles, pose, act, loc, g = [x[:n] for x in data]
    T = act.shape[1]
    acc = {
        f"L{li}": {"sight": 0.0, "last_sight": 0.0, "prev_frame": 0.0, "uniform_sight": 0.0, "top1_sight": 0.0, "n": 0}
        for li in range(len(model.blocks))
    }
    dev = tiles.device
    fc = torch.arange(T, device=dev).repeat_interleave(NTOK)
    fq = torch.arange(1, T, device=dev).repeat_interleave(NTOK)
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        _, attns = model.forward_seq(tiles[sl], pose[sl], act[sl], return_attn=True)
        B = attns[0].shape[0]
        lc = loc[sl].reshape(B, T * NTOK)
        lq = loc[sl][:, 1:].reshape(B, (T - 1) * NTOK)
        rev = g[sl][:, 1:].reshape(B, (T - 1) * NTOK) >= 2
        visible = fc[None, None, :] < fq[None, :, None]
        sight = (lq[:, :, None] == lc[:, None, :]) & visible  # [B,Q,C]
        fr_s = torch.where(sight, fc[None, None, :], -1)
        last = sight & (fc[None, None, :] == fr_s.max(-1, keepdim=True).values)
        prev = (fc[None, None, :] == (fq[None, :, None] - 1)).expand_as(sight)
        for li, a in enumerate(attns):
            am = a.mean(1)  # mean over heads [B,Q,C]
            d = acc[f"L{li}"]
            d["sight"] += float((am * sight).sum(-1)[rev].sum())
            d["last_sight"] += float((am * last).sum(-1)[rev].sum())
            d["prev_frame"] += float((am * prev).sum(-1)[rev].sum())
            d["uniform_sight"] += float((sight.sum(-1) / visible.sum(-1).clamp(min=1))[rev].sum())
            d["top1_sight"] += float(sight.gather(-1, am.argmax(-1, keepdim=True)).squeeze(-1)[rev].sum())
            d["n"] += int(rev.sum())
    for d in acc.values():
        n_tok = d.pop("n")
        for k in list(d):
            d[k] = round(d[k] / n_tok, 5)
        d["n_tokens"] = n_tok
    return acc


@torch.no_grad()
def probes(model, data, n_train=192, n_test=64, bs=32, max_fit=40000, seed=0):
    """Linear probes on the cached keys and values of each layer: the token's map location (x and y patch, 16
    classes each), the camera's x position (13 classes) and the colour of its first tile (8 classes). Trained on
    tokens of n_train episodes, scored on the tokens of n_test other episodes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tiles, pose, act, loc, _g = data
    T = act.shape[1]
    feats = {}
    for i in range(0, n_train + n_test, bs):
        sl = slice(i, i + bs)
        _, kvs = model.forward_seq(tiles[sl], pose[sl], act[sl], return_kv=True)
        for li, (k, v) in enumerate(kvs):
            feats.setdefault(f"K{li}", []).append(k.reshape(-1, k.shape[-1]).float().cpu())
            feats.setdefault(f"V{li}", []).append(v.reshape(-1, v.shape[-1]).float().cpu())
    feats = {k: torch.cat(v).numpy() for k, v in feats.items()}
    N = n_train + n_test
    lab = {
        "loc_x": (loc[:N] % 16).reshape(-1),
        "loc_y": (loc[:N] // 16).reshape(-1),
        "cam_x": pose[:N, :, None, 0].expand(-1, -1, NTOK).reshape(-1),
        "colour": tiles[:N, :, :, 0].reshape(-1),
    }
    lab = {k: v.cpu().numpy() for k, v in lab.items()}
    split = n_train * T * NTOK
    idx = np.random.default_rng(seed).choice(split, size=min(max_fit, split), replace=False)
    out = {}
    for fk, X in feats.items():
        sc = StandardScaler().fit(X[idx])
        Xtr, Xte = sc.transform(X[idx]), sc.transform(X[split:])
        out[fk] = {}
        for lk, y in lab.items():
            clf = LogisticRegression(max_iter=300).fit(Xtr, y[idx])
            out[fk][lk] = round(float((clf.predict(Xte) == y[split:]).mean()), 4)
        print("probe", fk, out[fk], flush=True)
    out["majority"] = {lk: round(float(np.bincount(y[split:]).max() / len(y[split:])), 4) for lk, y in lab.items()}
    return out


def build_masks(fc, fq, T, target_u, hide_q, hide_c):
    """Per-episode masks [B,1,N,N]. hide_q [B,C]: clean entries hidden from the queries of frame target_u only.
    hide_c [B,C]: clean entries hidden from every later token (every query, every clean token of a later frame)."""
    B = hide_q.shape[0]
    m = seq_mask(T, fc.device)[None].expand(B, -1, -1).clone()
    nc = T * NTOK
    qrows = fq[None, :] == target_u[:, None]  # [B,Q]
    m[:, nc:, :nc] &= ~(qrows[:, :, None] & hide_q[:, None, :])
    later_clean = fc[None, :, None] > fc[None, None, :]
    m[:, :nc, :nc] &= ~(later_clean & hide_c[:, None, :])
    m[:, nc:, :nc] &= ~hide_c[:, None, :]
    return m[:, None]


def random_subset(pool, k, gen):
    """For each row, a random subset of size k[row] of the True positions of pool [B,C] (fewer if pool is smaller)."""
    noise = torch.where(pool, torch.rand(pool.shape, generator=gen, device=pool.device), float("inf"))
    return pool & (noise.argsort(1).argsort(1) < k[:, None])


def topk_mask(score, k):
    return (score.argsort(1, descending=True).argsort(1) < k[:, None]) & torch.isfinite(score)


@torch.no_grad()
def causal_eviction(model, data, n=256, bs=16, min_gap=4, seed=0):
    """At one revisit-heavy frame u per episode (observed mode, otherwise full cache), hide exactly the cache entries
    that saw the revisited places, and compare with hiding the same number of random or most-attended entries.
    Scored on frame u's revisit tokens with gap >= min_gap (and, for three conditions, on its shift tokens).

      full            nothing hidden.
      q_sight         the earlier views of the revisited places, hidden from frame u's queries only.
      q_random        the same number of random entries (older than frame u-1, not such views), hidden the same way.
      q_attended      the same number of entries that frame u's revisit queries attend to most (full cache).
      q_sight_frames  every whole frame that holds such a view, hidden from frame u's queries.
      q_random_frames the same number of random whole frames (older than u-1, not those frames).
      c_sight         the views hidden from creation: no later frame or query ever reads them, so no later frame
                      can have copied them forward.
      c_random        the same number of random entries hidden from creation.
    """
    tiles, pose, act, loc, g = [x[:n] for x in data]
    T = act.shape[1]
    dev = tiles.device
    fc = torch.arange(T, device=dev).repeat_interleave(NTOK)
    fq = torch.arange(1, T, device=dev).repeat_interleave(NTOK)
    gen = torch.Generator(device=dev).manual_seed(seed)
    names = ["full", "q_sight", "q_random", "q_attended", "q_sight_frames", "q_random_frames", "c_sight", "c_random"]
    res = {k: [] for k in [*names, "shift_full", "shift_q_sight", "shift_q_sight_frames"]}
    overlap, sizes, nframes = [], [], []
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        tl, ps, ac, lo, gg = tiles[sl], pose[sl], act[sl], loc[sl], g[sl]
        B = tl.shape[0]
        bi = torch.arange(B, device=dev)
        cnt = (gg >= min_gap).sum(-1).float() + torch.arange(T, device=dev) * 1e-3  # ties: the latest frame
        u = cnt.argmax(1)
        rev_u = gg[bi, u] >= min_gap  # [B,16]
        loc_u = lo[bi, u]
        lc = lo.reshape(B, T * NTOK)
        older = fc[None, :] < u[:, None]
        sight = ((lc[:, :, None] == loc_u[:, None, :]) & rev_u[:, None, :]).any(-1) & older  # [B,C]
        k = sight.sum(1)
        sizes += k.tolist()
        rnd = random_subset((fc[None, :] < (u[:, None] - 1)) & ~sight, k, gen)
        _, attns = model.forward_seq(tl, ps, ac, return_attn=True)
        a = torch.stack(attns).sum((0, 2))  # [B,Q,C], summed over layers and heads
        qidx = (u[:, None] - 1) * NTOK + torch.arange(NTOK, device=dev)  # query rows of frame u
        a_rev = (a.gather(1, qidx[:, :, None].expand(-1, -1, a.shape[-1])) * rev_u[:, :, None]).sum(1)
        att = topk_mask(torch.where(older, a_rev, float("-inf")), k)
        overlap += ((att & sight).sum(1).float() / k.clamp(min=1)).tolist()
        sframes = torch.zeros(B, T, dtype=torch.bool, device=dev)  # frames that hold a view of a revisited place
        for b in range(B):
            sframes[b, fc[sight[b]].unique()] = True
        nf = sframes.sum(1)
        nframes += nf.tolist()
        fpool = (torch.arange(T, device=dev)[None, :] < (u[:, None] - 1)) & ~sframes
        rf = random_subset(fpool, nf, gen)[:, fc]
        sf = sframes[:, fc]
        none = torch.zeros_like(sight)
        conds = {
            "full": (none, none),
            "q_sight": (sight, none),
            "q_random": (rnd, none),
            "q_attended": (att, none),
            "q_sight_frames": (sf, none),
            "q_random_frames": (rf, none),
            "c_sight": (none, sight),
            "c_random": (none, rnd),
        }
        sh = gg[bi, u] == 1
        for name, (hq, hc) in conds.items():
            lg = model.forward_seq(tl, ps, ac, mask=build_masks(fc, fq, T, u, hq, hc))
            hit = (lg[bi, u - 1].argmax(-1) == tl[bi, u]).float().mean(-1)  # [B,16]
            res[name] += torch.stack([(hit * rev_u).sum(1) * 4, rev_u.sum(1) * 4.0], 1).round().long().tolist()
            if "shift_" + name in res:
                res["shift_" + name] += torch.stack([(hit * sh).sum(1) * 4, sh.sum(1) * 4.0], 1).round().long().tolist()
    res["attended_overlap_mean"] = float(np.mean(overlap))
    res["n_hidden_entries_mean"] = float(np.mean(sizes))
    res["n_hidden_frames_mean"] = float(np.mean(nframes))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--only", default="sweep,timing,attn,probe,causal")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    model, ck = load(a.ckpt, dev)
    T = ck["cfg"]["T"]
    data = episodes(a.n, T, 77, dev)
    out = {
        "seed": ck["seed"],
        "params": ck["params"],
        "train_log": ck["log"][-1],
        "n_episodes": a.n,
        "T": T,
        "device": dev,
    }
    only = a.only.split(",")
    t0 = time.time()
    if "sweep" in only:
        out["sweep"] = policy_sweep(model, data, a.bs)
    if "timing" in only:
        out["timing_ms_per_frame"] = timing(model, data, dev)
    if "attn" in only:
        out["attention"] = attention_attribution(model, data)
        print("attention", json.dumps(out["attention"]), flush=True)
    if "probe" in only:
        out["probe"] = probes(model, data)
    if "causal" in only:
        out["causal"] = causal_eviction(model, data)
        print(
            "causal",
            {
                k: round(float(np.array(v).sum(0)[0] / max(1, np.array(v).sum(0)[1])), 4)
                for k, v in out["causal"].items()
                if isinstance(v, list)
            },
            flush=True,
        )
    out["eval_sec"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
