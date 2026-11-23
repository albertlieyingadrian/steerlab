"""All CLIP-side analyses (A1-A4, B1-B3 and the controls) on the cached block-23 residuals.

    python analyze.py core  --tag full     # selection, arms, A1, A2 (marginal vs joint), controls, leakage, ImageNette
    python analyze.py cells --tag full     # A2 per-cell change with a paired image bootstrap (slow)
    python analyze.py a4    --tag full     # Matryoshka levels (run before a3: a3 reads latent_r2.npz)
    python analyze.py a3    --tag full     # conjunction features
    python analyze.py b1    --tag full     # prompt-specific attribution
    python analyze.py b2    --tag full     # attention heads
    python analyze.py b3    --tag full     # image side vs text side

Outputs: results/<part>.json (summaries with 95% bootstrap intervals) and results/prompt_metrics_core.npz
(per-prompt metrics per arm). Selection, fitting and ranking use FairFace train; every metric is on val.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import lib
from lib import HERE, Split, ci, cluster_boot
from sae import SAE

RES = HERE / "results"
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_RAND = int(__import__("os").environ.get("N_RAND", 20))
ATTRS = ["gender", "age3", "race"]  # marginals for the intersectional analysis
PAPER_ATTRS = ["gender", "age7", "race"]  # the paper's groups (selection, replication rows)
RELEASED = {"gender": [98, 195], "age7": [105, 163, 102], "race": [6, 264, 279, 98, 454, 364]}
KEYS = ["gender_maxdp", "age3_maxdp", "race_maxdp", "joint_maxdp", "gender_excess", "age3_excess", "race_excess",
        "joint_excess", "joint_minexcess", "marg_excess", "gender_eo", "age7_eo", "race_eo", "joint_eo", "paper_eo",
        "rss_ratio", "gender_minexcess", "age3_minexcess", "race_minexcess", "age7_maxdp", "age7_excess"]  # fmt: skip


def jdump(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        raise TypeError(type(o))

    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=conv))
    print("wrote", RES / f"{name}.json", flush=True)


class Ctx:
    def __init__(self, tag):
        t0 = time.time()
        self.tag = tag
        self.train = Split(tag, "train")
        self.val = Split(tag, "val")
        self.imn = Split(tag, "imagenette")
        self.names = lib.names(tag)
        self.sae = SAE.load()
        self.level = self.sae.level().numpy()
        P = json.loads((HERE / "data" / "prompts.json").read_text())
        self.prompts = P
        text = torch.as_tensor(np.load(lib.CACHE / tag / "text_emb.npy"))
        text = torch.nn.functional.normalize(text, dim=-1)
        meta = [(s, p["concept"], P["topk"][s]) for s, ps in P["sets"].items() for p in ps]
        n = len(meta)
        self.T = text[:n]
        self.T_imn = text[n : n + 10]
        self.T_group = {}
        o = n + 10
        for a, ps in P["group_prompts"].items():
            self.T_group[a] = text[o : o + len(ps)]
            o += len(ps)
        self.pset = np.array([m[0] for m in meta])
        self.K = np.array([m[2] for m in meta])
        self.cluster = np.array([f"{m[0]}|{m[1]}" for m in meta])
        if self.val.n < 2000:  # pilot: scale K to the smaller pool
            self.K = np.maximum(10, (self.K * self.val.n / 10954).astype(int))
        self.codes_tr = lib.codes_of(self.sae, self.train.r)
        self.codes_va = lib.codes_of(self.sae, self.val.r)
        self.codes_imn = lib.codes_of(self.sae, self.imn.r)
        self.freq = (self.codes_tr > 0).float().mean(0).numpy()
        self.probe_rows = np.load(lib.CACHE / tag / "train_head_rows.npy")  # fixed 10,000-image train subset
        self.floors = {}
        for attr in ["gender", "age3", "race", "joint", "age7"]:
            lab, G = self.val.groups(attr)
            share = lib.group_share(lab, G)
            for K in sorted(set(self.K.tolist())):
                c = lib.random_counts(lab, G, int(K), n=200, seed=1)
                s = lib.skew_dp(c, int(K), share)
                fl = {"maxskew_dp": float(s.max(1).values.mean()), "minskew_dp": float(s.min(1).values.mean()),
                      "maxskew_eo": float(lib.maxskew_eo(c, int(K)).mean())}  # fmt: skip
                if attr == "joint":
                    fl["rss"] = float(lib.additive_rss(s, share).mean())
                self.floors[(attr, int(K))] = fl
        print(f"[ctx] loaded in {time.time() - t0:.0f}s: train {self.train.n}, val {self.val.n}, "
              f"imagenette {self.imn.n}, prompts {len(self.T)}", flush=True)  # fmt: skip

    def metrics(self, E_val, T=None):
        T = self.T if T is None else T
        return metrics_from_sims(self, T @ torch.as_tensor(E_val).T)

    def ability(self, E_imn):
        return lib.zero_shot(E_imn, self.T_imn, self.imn.label)

    def leakage(self, E_train_sub, E_val):
        out = {}
        for attr in ATTRS:
            ytr = self.train.groups(attr)[0][self.probe_rows]
            yva = self.val.groups(attr)[0]
            out[attr] = lib.probe(E_train_sub, ytr, E_val, yva)
        return out

    def sae_arm(self, zero, alpha, probe=False):
        Ev = lib.embed(lib.sae_blend(self.sae, self.val.r, zero, alpha, codes=self.codes_va))
        Ei = lib.embed(lib.sae_blend(self.sae, self.imn.r, zero, alpha, codes=self.codes_imn))
        Et = None
        if probe:
            rows = torch.as_tensor(self.probe_rows)
            Et = lib.embed(
                lib.sae_blend(self.sae, self.train.r[self.probe_rows], zero, alpha, codes=self.codes_tr[rows])
            )
        return Ev, Ei, Et


def metrics_from_sims(ctx, sims):
    """Per-prompt metrics from similarity rows [P, N_val]: max / min skew_dp, the paper's maxskew_eo, excess over the
    random-retrieval floor, per-cell skew (joint) and the additive-fit RSS ratio."""
    out = {}
    for attr in ["gender", "age3", "race", "joint", "age7"]:
        lab, G = ctx.val.groups(attr)
        share = lib.group_share(lab, G)
        mx, mn, eo, ex, exmin = (np.zeros(len(sims)) for _ in range(5))
        cells = np.zeros((len(sims), 42)) if attr == "joint" else None
        for K in sorted(set(ctx.K.tolist())):
            rows = np.where(ctx.K == K)[0]
            c = lib.topk_counts(sims[rows], lab, G, int(K))
            s = lib.skew_dp(c, int(K), share)
            fl = ctx.floors[(attr, int(K))]
            mx[rows] = s.max(1).values.numpy()
            mn[rows] = s.min(1).values.numpy()
            eo[rows] = lib.maxskew_eo(c, int(K)).numpy()
            ex[rows] = mx[rows] - fl["maxskew_dp"]
            exmin[rows] = mn[rows] - fl["minskew_dp"]
            if attr == "joint":
                cells[rows] = s.numpy()
                out.setdefault("rss_ratio", np.zeros(len(sims)))
                out["rss_ratio"][rows] = (lib.additive_rss(s, share) / fl["rss"]).numpy()
        out[f"{attr}_maxdp"], out[f"{attr}_mindp"], out[f"{attr}_eo"] = mx, mn, eo
        out[f"{attr}_excess"], out[f"{attr}_minexcess"] = ex, exmin
        if cells is not None:
            out["cells"] = cells
    out["marg_excess"] = np.mean([out[f"{a}_excess"] for a in ATTRS], 0)
    out["paper_eo"] = np.mean([out[f"{a}_eo"] for a in PAPER_ATTRS], 0)
    return out


def summarise(per_prompt, clusters, keys, ref=None, n=10000):
    """Mean and 95% cluster-bootstrap CI per key; with `ref`, also the paired difference (arm - ref)."""
    out = {}
    for k in keys:
        v = per_prompt[k]
        out[k] = {"mean": float(v.mean()), "ci": ci(cluster_boot(v, clusters, n=n))}
        if ref is not None:
            out[k]["diff"] = float((v - ref[k]).mean())
            out[k]["diff_ci"] = ci(cluster_boot(v - ref[k], clusters, n=n))
    return out


def acc_summary(correct, ref=None):
    c = np.asarray(correct, dtype=float)
    out = {"acc": float(c.mean()), "ci": ci(lib.image_boot_mean(c))}
    if ref is not None:
        r = np.asarray(ref, dtype=float)
        out["diff"] = float((c - r).mean())
        out["diff_ci"] = ci(lib.image_boot_mean(c - r))
    return out


# ------------------------------------------------------------------ core: selection, arms, A1, A2, controls
def core(ctx):
    rng = np.random.default_rng(0)
    res = {"n": {"train": ctx.train.n, "val": ctx.val.n, "imagenette": ctx.imn.n, "prompts": len(ctx.T)}}
    sel = {}
    for attr in PAPER_ATTRS:
        lab, G = ctx.train.groups(attr)
        sel[attr] = lib.select_social(ctx.codes_tr, lab, G)
    chosen = {a: sorted({s["latent"] for s in sel[a] if s["latent"] is not None}) for a in PAPER_ATTRS}
    chosen["all"] = sorted(set().union(*chosen.values()))
    res["selection"] = {"per_group": sel, "chosen": chosen, "released": RELEASED,
                        "overlap_with_released": {a: sorted(set(chosen[a]) & set(RELEASED[a])) for a in PAPER_ATTRS},
                        "alltoken": alltoken_selection(ctx),
                        "levels": {a: [int(ctx.level[j]) for j in chosen[a]] for a in chosen},
                        "freq": {a: [float(ctx.freq[j]) for j in chosen[a]] for a in chosen}}  # fmt: skip
    rec = ctx.sae.decode(ctx.codes_va)
    r = torch.as_tensor(ctx.val.r)
    res["sae_fidelity"] = {
        "fvu_cls_val": float(((r - rec) ** 2).sum() / ((r - r.mean(0)) ** 2).sum()),
        "cos_embed_recon_vs_clip": float((lib.embed(rec) * lib.embed(r)).sum(1).mean()),
        "l0_cls_val": float((ctx.codes_va > 0).sum(1).float().mean()),
        "emb_check_max_abs": float((lib.embed(r, False) - torch.as_tensor(ctx.val.emb)).abs().max()),
    }
    res["cache_report"] = json.loads((lib.CACHE / ctx.tag / "report.json").read_text())

    arms, abil, leak = {}, {}, {}
    E0 = lib.embed(ctx.val.r)
    arms["clip"] = ctx.metrics(E0)
    abil["clip"] = ctx.ability(lib.embed(ctx.imn.r))
    leak["clip"] = ctx.leakage(lib.embed(ctx.train.r[ctx.probe_rows]), E0)

    def run(name, zero, alpha, probe=False):
        Ev, Ei, Et = ctx.sae_arm(zero, alpha, probe)
        arms[name] = ctx.metrics(Ev)
        abil[name] = ctx.ability(Ei)
        if probe:
            leak[name] = ctx.leakage(Et, Ev)

    for al in ALPHAS:
        run(f"recon@{al}", [], al, probe=al in (0.6, 1.0))
        run(f"dbl_all@{al}", chosen["all"], al, probe=al in (0.6, 1.0))
    for a in PAPER_ATTRS:
        run(f"dbl_{a}@0.6", chosen[a], 0.6, probe=True)
        run(f"dbl_{a}@1.0", chosen[a], 1.0)
        run(f"released_{a}@0.6", RELEASED[a], 0.6)
    run("released_all@0.6", sorted(set().union(*RELEASED.values())), 0.6, probe=True)
    print("[core] main arms done", flush=True)
    rand_sets = {}
    for a in PAPER_ATTRS + ["all"]:
        draws = lib.freq_matched(ctx.freq, chosen[a], N_RAND, rng, exclude=chosen["all"])
        rand_sets[a] = draws
        for i, d in enumerate(draws):
            run(f"rand_{a}#{i}@0.6", d, 0.6, probe=(a in ("gender", "all") and i < 3))
            if a == "all":
                for al in (0.2, 0.4, 0.8, 1.0):
                    run(f"rand_all#{i}@{al}", d, al)
    res["rand_sets"] = rand_sets
    print("[core] random arms done", flush=True)
    leace_arms = {a: ctx.train.groups(a) for a in ATTRS}
    for a, (lab, G) in leace_arms.items():
        le = lib.Leace.fit(ctx.train.r, lib.onehot(lab, G))
        Ev, Ei, Et = (lib.embed(le(x)) for x in (ctx.val.r, ctx.imn.r, ctx.train.r[ctx.probe_rows]))
        arms[f"leace_{a}"] = ctx.metrics(Ev)
        abil[f"leace_{a}"] = ctx.ability(Ei)
        leak[f"leace_{a}"] = ctx.leakage(Et, Ev)
    Z = np.concatenate([lib.onehot(*ctx.train.groups(a)) for a in ATTRS], 1)
    le = lib.Leace.fit(ctx.train.r, Z)
    Ev, Ei, Et = (lib.embed(le(x)) for x in (ctx.val.r, ctx.imn.r, ctx.train.r[ctx.probe_rows]))
    arms["leace_all"] = ctx.metrics(Ev)
    abil["leace_all"] = ctx.ability(Ei)
    leak["leace_all"] = ctx.leakage(Et, Ev)
    res["leace_rank_all"] = le.rank
    print("[core] leace done", flush=True)

    ref = arms["clip"]
    res["arms"] = {k: summarise(v, ctx.cluster, KEYS, ref=None if k == "clip" else ref)
                   for k, v in arms.items() if "#" not in k}  # fmt: skip
    band = {}
    for a in PAPER_ATTRS + ["all"]:
        for al in [0.6] if a != "all" else ALPHAS:
            ks = [k for k in arms if k.startswith(f"rand_{a}#") and k.endswith(f"@{al}")]
            band[f"rand_{a}@{al}"] = {key: sorted(float(arms[k][key].mean()) for k in ks) for key in KEYS}
            band[f"rand_{a}@{al}"]["imagenette"] = sorted(float(abil[k].mean()) for k in ks)
    res["random_bands"] = band
    res["ability"] = {k: acc_summary(v, None if k == "clip" else abil["clip"]) for k, v in abil.items() if "#" not in k}
    res["ability_random_leak"] = {k: {a: v[a]["acc"] for a in ATTRS} for k, v in leak.items() if "#" in k}
    res["leakage"] = {k: {a: {"acc": v[a]["acc"], "bal_acc": v[a]["bal_acc"],
                              "ci": ci(lib.image_boot_mean(v[a]["correct"]))} for a in ATTRS}
                      for k, v in leak.items() if "#" not in k}  # fmt: skip
    res["per_set"] = {arm: {s: {k: float(arms[arm][k][ctx.pset == s].mean()) for k in KEYS} for s in ctx.prompts["sets"]}
                      for arm in ("clip", "dbl_all@0.6", "recon@0.6", "leace_all")}  # fmt: skip
    res["gerrymander"] = gerrymander(arms, ctx)
    res["floors"] = {f"{a}@{K}": fl for (a, K), fl in ctx.floors.items()}
    jdump("core", res)
    np.savez_compressed(RES / "prompt_metrics_core.npz",
                        **{f"{arm}::{k}": v for arm, d in arms.items() if "#" not in arm for k, v in d.items()})  # fmt: skip


def gerrymander(arms, ctx):
    """Relative cut in excess skew, marginal mean vs joint, with a paired cluster bootstrap."""
    out = {}
    clip = arms["clip"]
    for arm in [k for k in arms if "#" not in k and k != "clip"]:
        a = arms[arm]
        cols = np.stack([clip["marg_excess"], a["marg_excess"], clip["joint_excess"], a["joint_excess"]], 1)
        d = cluster_boot(cols, ctx.cluster)
        rel_m = 1 - d[:, 1] / d[:, 0]
        rel_j = 1 - d[:, 3] / d[:, 2]
        pm = 1 - a["marg_excess"].mean() / clip["marg_excess"].mean()
        pj = 1 - a["joint_excess"].mean() / clip["joint_excess"].mean()
        out[arm] = {"rel_cut_marginal": float(pm), "rel_cut_marginal_ci": ci(rel_m), "rel_cut_joint": float(pj),
                    "rel_cut_joint_ci": ci(rel_j), "gap": float(pm - pj), "gap_ci": ci(rel_m - rel_j)}  # fmt: skip
    return out


def alltoken_selection(ctx):
    """The paper's rule on all-token firing (block 23, 20,000 train images), from the cache's group counts."""
    st = np.load(lib.CACHE / ctx.tag / "alltoken_sae_stats.npz")
    fire, actsum, ntok = st["fire"], st["actsum"], st["ntok"]
    key = np.arange(126)
    g, rest = np.divmod(key, 63)
    a9, r = np.divmod(rest, 7)
    groupings = {"gender": (g, 2), "age7": (np.where((a9 >= 1) & (a9 <= 7), a9 - 1, -1), 7), "race": (r, 7)}
    out, gaps = {}, {}
    for name, (lab, G) in groupings.items():
        fr = np.stack([fire[lab == k].sum(0) / max(ntok[lab == k].sum(), 1) for k in range(G)])
        mn = np.stack([actsum[lab == k].sum(0) / max(ntok[lab == k].sum(), 1) for k in range(G)])
        res = []
        for k in range(G):
            for t in (0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01):
                eff = fr >= t
                cand = eff[k] & (eff.sum(0) == 1)
                if cand.any():
                    j = int(np.where(cand, mn[k], -1).argmax())
                    res.append({"group": k, "latent": j, "tau": t, "fire": float(fr[k, j])})
                    break
            else:
                res.append({"group": k, "latent": None})
        out[name] = res
        gap = fr.max(0) - fr.min(0)
        rank = np.empty(len(gap), int)
        rank[np.argsort(-gap)] = np.arange(len(gap))
        gaps[name] = {int(j): {"rank_of_8192_by_firing_gap": int(rank[j]), "fire_by_group": fr[:, j].round(4).tolist()}
                      for j in RELEASED[name]}  # fmt: skip
    out["overlap_with_released"] = {n: sorted({x["latent"] for x in out[n] if x["latent"] is not None} & set(RELEASED[n]))
                                    for n in groupings}  # fmt: skip
    out["released_latents"] = gaps
    # the same firing-gap rank on CLS codes, for comparison
    cls_gaps = {}
    for n, attr in (("gender", "gender"), ("age7", "age7"), ("race", "race")):
        lab, G = ctx.train.groups(attr)
        fr = lib.fire_by_group(ctx.codes_tr, lab, G).numpy()
        gap = fr.max(0) - fr.min(0)
        rank = np.empty(len(gap), int)
        rank[np.argsort(-gap)] = np.arange(len(gap))
        cls_gaps[n] = {int(j): {"rank_of_8192_by_firing_gap": int(rank[j]), "fire_by_group": fr[:, j].round(4).tolist()}
                       for j in RELEASED[n]}  # fmt: skip
    out["released_latents_cls"] = cls_gaps
    return out


# ------------------------------------------------------------------ A2 cells: paired image bootstrap
def cells(ctx, n_boot=10000):
    core_res = json.loads((RES / "core.json").read_text())
    chosen = core_res["selection"]["chosen"]["all"]
    rand0 = core_res["rand_sets"]["all"][0]
    E = {"clip": lib.embed(ctx.val.r),
         "dbl_all": lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, chosen, 0.6, codes=ctx.codes_va)),
         "recon": lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, [], 0.6, codes=ctx.codes_va)),
         "rand_all0": lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, rand0, 0.6, codes=ctx.codes_va))}  # fmt: skip
    sims = {k: (ctx.T @ v.T) for k, v in E.items()}
    lab = torch.as_tensor(ctx.val.groups("joint")[0])
    Ks = sorted(set(ctx.K.tolist()))
    rows = {K: torch.as_tensor(np.where(ctx.K == K)[0]) for K in Ks}

    def cell_abs(idx):
        out = {}
        lb = lab[idx]
        share = torch.bincount(lb, minlength=42).float() / len(lb)
        for k, s in sims.items():
            acc = torch.zeros(42)
            for K in Ks:
                top = torch.topk(s[rows[K]][:, idx], K, dim=1).indices
                c = torch.zeros(len(rows[K]), 42).scatter_add_(1, lb[top], torch.ones(top.shape))
                acc += lib.skew_dp(c, K, share).abs().sum(0)
            out[k] = acc / len(ctx.T)
        return out

    full = cell_abs(torch.arange(ctx.val.n))
    rng = np.random.default_rng(0)
    draws = {k: np.zeros((n_boot, 42)) for k in sims}
    t0 = time.time()
    for b in range(n_boot):
        o = cell_abs(torch.as_tensor(rng.integers(0, ctx.val.n, ctx.val.n)))
        for k in o:
            draws[k][b] = o[k].numpy()
        if b % 500 == 0:
            print(f"[cells] {b}/{n_boot} {time.time() - t0:.0f}s", flush=True)
    res = {"cell_names": [lib.cell_name(c, ctx.names) for c in range(42)],
           "cell_share_val": lib.group_share(ctx.val.cell, 42).tolist(), "n_boot": n_boot}  # fmt: skip
    for k in sims:
        res[k] = full[k].tolist()
    for arm in ("dbl_all", "recon", "rand_all0"):
        d = draws[arm] - draws["clip"]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        res[f"{arm}_minus_clip"] = (full[arm] - full["clip"]).tolist()
        res[f"{arm}_minus_clip_ci"] = [lo.tolist(), hi.tolist()]
        res[f"{arm}_cells_worse"] = int((lo > 0).sum())
        res[f"{arm}_cells_better"] = int((hi < 0).sum())
    jdump("cells", res)


# ------------------------------------------------------------------ A4 Matryoshka levels (also feeds A3)
def latent_r2(ctx, min_freq=0.001):
    """R-squared of each latent's train CLS activation on (a) gender + age3 + race main effects, (b) 42 cell means."""
    A = ctx.codes_tr
    live = np.where(ctx.freq >= min_freq)[0]
    g, a3, rc = ctx.train.gender, ctx.train.age3, ctx.train.race
    X = np.zeros((ctx.train.n, 10), np.float32)
    X[:, 0] = 1
    X[g == 1, 1] = 1
    X[a3 == 1, 2] = 1
    X[a3 == 2, 3] = 1
    for k in range(1, 7):
        X[rc == k, 3 + k] = 1
    X = torch.as_tensor(X)
    XtX_inv = torch.linalg.inv(X.T @ X)
    oc = torch.as_tensor(lib.onehot(ctx.train.cell, 42), dtype=torch.float32)
    cnt = oc.sum(0)
    oh = {k: torch.as_tensor(lib.onehot(*ctx.train.groups(k)), dtype=torch.float32) for k in ATTRS}
    r2a, r2j, r2m = [], [], {k: [] for k in ATTRS}
    for i in range(0, len(live), 512):
        Y = A[:, torch.as_tensor(live[i : i + 512])]
        mu = Y.mean(0)
        tot = ((Y - mu) ** 2).sum(0)
        beta = XtX_inv @ (X.T @ Y)
        r2a.append((((X @ beta) - mu) ** 2).sum(0) / tot)
        means = (oc.T @ Y) / torch.clamp(cnt, min=1)[:, None]
        r2j.append((cnt[:, None] * (means - mu) ** 2).sum(0) / tot)
        for k in ATTRS:
            c = oh[k].sum(0)
            m = (oh[k].T @ Y) / c[:, None]
            r2m[k].append((c[:, None] * (m - mu) ** 2).sum(0) / tot)
    r2a, r2j = torch.cat(r2a).numpy(), torch.cat(r2j).numpy()
    r2m = {k: torch.cat(v).numpy() for k, v in r2m.items()}
    inter = np.where(r2j > 0, (r2j - r2a) / np.maximum(r2j, 1e-12), 0)
    return live, r2a, r2j, inter, r2m


def a4(ctx):
    live, r2a, r2j, inter, r2m = latent_r2(ctx)
    lev = ctx.level[live]
    rng = np.random.default_rng(0)
    res = {"n_live": int(len(live)), "min_freq": 0.001, "levels": {}}
    for L in range(4):
        m = lev == L
        res["levels"][L] = {"n": int(m.sum()), "median_r2_add": float(np.median(r2a[m])),
                            "median_r2_joint": float(np.median(r2j[m])), "median_interaction": float(np.median(inter[m])),
                            "mean_r2_add": float(r2a[m].mean()), "share_r2_add_gt_0.1": float((r2a[m] > 0.1).mean()),
                            "median_r2": {k: float(np.median(r2m[k][m])) for k in ATTRS}}  # fmt: skip

    def boot_med_diff(x, a, b, n=10000):
        xa, xb = x[lev == a], x[lev == b]
        return np.array([np.median(xa[rng.integers(0, len(xa), len(xa))]) - np.median(xb[rng.integers(0, len(xb), len(xb))])
                         for _ in range(n)])  # fmt: skip

    res["p7"] = {"r2_add_L1_minus_L4": float(np.median(r2a[lev == 0]) - np.median(r2a[lev == 3])),
                 "ci1": ci(boot_med_diff(r2a, 0, 3)),
                 "interaction_L4_minus_L1": float(np.median(inter[lev == 3]) - np.median(inter[lev == 0])),
                 "ci2": ci(boot_med_diff(inter, 3, 0))}  # fmt: skip
    sig = r2j >= 0.05
    res["levels_signal"] = {L: {"n": int((sig & (lev == L)).sum()),
                                "median_interaction": float(np.median(inter[sig & (lev == L)])) if (sig & (lev == L)).any() else None,
                                "median_r2_add": float(np.median(r2a[sig & (lev == L)])) if (sig & (lev == L)).any() else None}
                            for L in range(4)}  # fmt: skip
    np.savez_compressed(RES / "latent_r2.npz", live=live, r2_add=r2a, r2_joint=r2j, interaction=inter, level=lev,
                        **{f"r2_{k}": v for k, v in r2m.items()})  # fmt: skip
    core_res = json.loads((RES / "core.json").read_text())
    pos = {int(j): i for i, j in enumerate(live)}
    res["social_latents"] = {int(j): {"level": int(ctx.level[j]), "r2_add": float(r2a[pos[j]]) if j in pos else None,
                                      "interaction": float(inter[pos[j]]) if j in pos else None}
                             for j in core_res["selection"]["chosen"]["all"]}  # fmt: skip
    jdump("a4", res)


# ------------------------------------------------------------------ A3 conjunction features
def full_cell_abs(ctx, E):
    """Mean over prompts of |skew_dp| per joint cell (val)."""
    return np.abs(ctx.metrics(E)["cells"]).mean(0)


def a3(ctx):
    rng = np.random.default_rng(3)
    z = np.load(RES / "latent_r2.npz")
    pos = {int(j): i for i, j in enumerate(z["live"])}
    lab, _ = ctx.train.groups("joint")
    fire = lib.fire_by_group(ctx.codes_tr, lab, 42)  # [42, L]
    res = {"tau_sweep": {}, "cells": {}}
    found_any = set()
    for tau in (0.9, 0.7, 0.5, 0.3, 0.1):
        eff = fire >= tau
        only = eff.sum(0) == 1
        cand = {c: torch.nonzero(eff[c] & only).flatten().tolist() for c in range(42)}
        found = sorted({j for v in cand.values() for j in v})
        found_any |= set(found)
        inter = [z["interaction"][pos[j]] >= 0.5 for j in found if j in pos]
        res["tau_sweep"][tau] = {"cells_with_candidate": int(sum(1 for v in cand.values() if v)), "n_latents": len(found),
                                 "share_interaction_ge_0.5": float(np.mean(inter)) if inter else None}  # fmt: skip
    vals = [z["interaction"][pos[j]] >= 0.5 for j in found_any if j in pos]
    res["latents_without_r2"] = sorted(j for j in found_any if j not in pos)  # fire on < 0.1% of train images
    res["p5"] = {"cells_with_candidate_tau0.9": res["tau_sweep"][0.9]["cells_with_candidate"],
                 "n_latents_any_tau": len(found_any), "share_any_tau_interaction_ge_0.5": float(np.mean(vals)) if vals else None}  # fmt: skip
    sel = lib.select_social(ctx.codes_tr, lab, 42)
    chosen = [s for s in sel if s["latent"] is not None]
    marg = {k: lib.fire_by_group(ctx.codes_tr, *ctx.train.groups(k)) for k in ATTRS}
    share_cell = lib.group_share(ctx.val.cell, 42)
    dbl_all = json.loads((RES / "core.json").read_text())["selection"]["chosen"]["all"]
    clip_cells = full_cell_abs(ctx, lib.embed(ctx.val.r))
    dbl_cells = full_cell_abs(ctx, lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, dbl_all, 0.6, codes=ctx.codes_va)))
    recon_cells = full_cell_abs(ctx, lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, [], 0.6, codes=ctx.codes_va)))
    n_beats = 0
    for s in chosen:
        c, j = s["group"], s["latent"]
        g, rest = divmod(c, 21)
        a, rr = divmod(rest, 7)
        info = {"cell": lib.cell_name(c, ctx.names), "latent": j, "tau": s["tau"], "fire_cell": s["fire"],
                "fire_other_cells_max": s["fire_other_max"], "level": int(ctx.level[j]),
                "interaction": float(z["interaction"][pos[j]]) if j in pos else None,
                "r2_add": float(z["r2_add"][pos[j]]) if j in pos else None,
                "fire_own_gender": float(marg["gender"][g, j]), "fire_own_age3": float(marg["age3"][a, j]),
                "fire_own_race": float(marg["race"][rr, j]), "val_share": float(share_cell[c]),
                "clip_abs_skew": float(clip_cells[c]), "recon_change": float(recon_cells[c] - clip_cells[c]),
                "dbl_all_change": float(dbl_cells[c] - clip_cells[c])}  # fmt: skip
        for al in (0.6, 1.0):
            cc = full_cell_abs(ctx, lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, [j], al, codes=ctx.codes_va)))
            info[f"cell_change@{al}"] = float(cc[c] - clip_cells[c])
            if al == 0.6:
                info["other_cells_mean_change@0.6"] = float(np.delete(cc - clip_cells, c).mean())
        rcut = []
        for d in lib.freq_matched(ctx.freq, [j], N_RAND, rng, exclude=set(dbl_all) | {x["latent"] for x in chosen}):
            rcut.append(float(full_cell_abs(ctx, lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, d, 0.6, codes=ctx.codes_va)))[c]
                              - clip_cells[c]))  # fmt: skip
        info["random_changes@0.6"] = sorted(rcut)
        info["beats_best_random"] = bool(info["cell_change@0.6"] < min(rcut))
        n_beats += int(info["beats_best_random"])
        res["cells"][c] = info
        print(
            f"[a3] cell {c} latent {j} tau {s['tau']} change {info['cell_change@0.6']:+.3f} best random {min(rcut):+.3f}",
            flush=True,
        )
    res["p6"] = {"n_tested": len(chosen), "n_beats_best_random": n_beats}
    allc = sorted({s["latent"] for s in chosen})
    clip_m = ctx.metrics(lib.embed(ctx.val.r))
    m = ctx.metrics(lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, allc, 0.6, codes=ctx.codes_va)))
    res["conj_all"] = {
        "n_latents": len(allc),
        **summarise(m, ctx.cluster, ["joint_excess", "marg_excess", "joint_eo", "paper_eo"], ref=clip_m),
    }
    res["conj_all"]["cells_abs_change_mean"] = float((np.abs(m["cells"]).mean(0) - clip_cells).mean())
    res["conj_all"]["imagenette"] = acc_summary(
        ctx.ability(lib.embed(lib.sae_blend(ctx.sae, ctx.imn.r, allc, 0.6, codes=ctx.codes_imn))),
        ctx.ability(lib.embed(ctx.imn.r)),
    )
    jdump("a3", res)


# ------------------------------------------------------------------ B1 prompt-specific attribution
def b1(ctx, n_imn_prompts=50):
    rng = np.random.default_rng(11)
    dbl_all = json.loads((RES / "core.json").read_text())["selection"]["chosen"]["all"]
    N = len(dbl_all)
    g, _, P = lib.clip_head()
    D = ctx.sae.W_dec  # [L, 1024]
    U = ((D - D.mean(1, keepdim=True)) * g) @ P.T  # [L, 768]: one unit of latent j in the embedding, before 1/sigma
    r = torch.as_tensor(ctx.train.r)
    sig = torch.sqrt(r.var(1, unbiased=False) + 1e-5)
    e = lib.embed(r, normalise=False)
    scale = 1.0 / (sig * e.norm(dim=1))
    sims_tr = ctx.T @ torch.nn.functional.normalize(e, dim=-1).T  # [P, Ntr]
    A = ctx.codes_tr * scale[:, None]
    Ktr = np.round(ctx.K * ctx.train.n / ctx.val.n).astype(int)
    TU = ctx.T @ U.T  # [P, L]
    score = torch.zeros_like(TU)
    over = {}
    for attr in ATTRS:
        lab, G = ctx.train.groups(attr)
        share = lib.group_share(lab, G)
        oh = torch.as_tensor(lib.onehot(lab, G), dtype=torch.float32)
        cnt = oh.sum(0)
        m_g = (oh.T @ A) / cnt[:, None]
        m_not = (A.sum(0)[None, :] - m_g * cnt[:, None]) / (len(lab) - cnt)[:, None]
        gstar = np.zeros(len(ctx.T), int)
        for K in sorted(set(Ktr.tolist())):
            rows = np.where(Ktr == K)[0]
            gstar[rows] = lib.skew_dp(lib.topk_counts(sims_tr[rows], lab, G, int(K)), int(K), share).argmax(1).numpy()
        over[attr] = gstar
        score += TU * (m_g - m_not)[torch.as_tensor(gstar)]
    top = torch.topk(score, N, dim=1).indices.numpy()  # [P, N]
    base_val = lib.sae_blend(ctx.sae, ctx.val.r, [], 0.6, codes=ctx.codes_va)
    base_imn = lib.sae_blend(ctx.sae, ctx.imn.r, [], 0.6, codes=ctx.codes_imn)
    E_clip = lib.embed(ctx.val.r)

    def emb_zero(base_val, codes, lat):
        lat = torch.as_tensor(np.asarray(lat))
        return lib.embed(base_val - 0.6 * codes[:, lat] @ D[lat])

    def sims_for(sets):
        return torch.stack([emb_zero(base_val, ctx.codes_va, sets[p]) @ ctx.T[p] for p in range(len(ctx.T))])

    sims_attr = sims_for(top)
    m_attr = metrics_from_sims(ctx, sims_attr)
    clip_m = ctx.metrics(E_clip)
    E_dbl = lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, dbl_all, 0.6, codes=ctx.codes_va))
    dbl_m = ctx.metrics(E_dbl)
    res = {"N": N, "over_retrieved_group_train": {a: np.bincount(over[a], minlength=7).tolist() for a in ATTRS},
           "attr_vs_clip": summarise(m_attr, ctx.cluster, KEYS, ref=clip_m),
           "attr_vs_dbl_all": summarise(m_attr, ctx.cluster, KEYS, ref=dbl_m),
           "rank_fidelity_attr": rank_fid(ctx, sims_attr, ctx.T @ E_clip.T),
           "rank_fidelity_dbl_all": rank_fid(ctx, ctx.T @ E_dbl.T, ctx.T @ E_clip.T),
           "rank_fidelity_recon": rank_fid(ctx, ctx.T @ lib.embed(base_val).T, ctx.T @ E_clip.T)}  # fmt: skip
    band = {k: [] for k in ("joint_excess", "marg_excess", "paper_eo")}
    for i in range(N_RAND):
        sets = [lib.freq_matched(ctx.freq, top[p], 1, rng)[0] for p in range(len(ctx.T))]
        mr = metrics_from_sims(ctx, sims_for(sets))
        for k in band:
            band[k].append(float(mr[k].mean()))
        print(f"[b1] random draw {i}", flush=True)
    res["random_band"] = {k: sorted(v) for k, v in band.items()}
    ps = rng.choice(len(ctx.T), n_imn_prompts, replace=False)
    accs = [float(ctx.ability(emb_zero(base_imn, ctx.codes_imn, top[p])).mean()) for p in ps]
    res["imagenette"] = {"attr_mean_over_prompts": float(np.mean(accs)), "attr_range": [float(min(accs)), float(max(accs))],
                         "dbl_all": float(ctx.ability(lib.embed(lib.sae_blend(ctx.sae, ctx.imn.r, dbl_all, 0.6, codes=ctx.codes_imn))).mean()),
                         "recon": float(ctx.ability(lib.embed(base_imn)).mean()), "clip": float(ctx.ability(lib.embed(ctx.imn.r)).mean()),
                         "n_prompts": n_imn_prompts}  # fmt: skip
    flat = top.flatten()
    uniq, cnts = np.unique(flat, return_counts=True)
    order = np.argsort(-cnts)[:10]
    res["latent_reuse"] = {"distinct": int(len(uniq)), "share_in_dbl_all": float(np.isin(flat, dbl_all).mean()),
                           "level_share": (np.bincount(ctx.level[flat], minlength=4) / len(flat)).tolist(),
                           "most_common": [[int(uniq[i]), int(cnts[i])] for i in order]}  # fmt: skip
    np.save(RES / "b1_top_latents.npy", top)
    jdump("b1", res)


def rank_fid(ctx, s1, s0):
    """Mean over prompts and cells (> 5 images) of the Spearman correlation between two similarity rows."""
    vals = []
    for c in range(42):
        msk = torch.as_tensor(ctx.val.cell == c)
        if msk.sum() > 5:
            a1 = torch.argsort(torch.argsort(s1[:, msk], 1), 1).float()
            a0 = torch.argsort(torch.argsort(s0[:, msk], 1), 1).float()
            a1 = a1 - a1.mean(1, keepdim=True)
            a0 = a0 - a0.mean(1, keepdim=True)
            vals.append(((a1 * a0).sum(1) / (a1.norm(dim=1) * a0.norm(dim=1))).mean().item())
    return float(np.mean(vals))


# ------------------------------------------------------------------ B2 attention heads (direct path, exact LN)
def b2(ctx):
    rng = np.random.default_rng(22)
    d = lib.CACHE / ctx.tag
    H_tr = np.load(d / "train_heads.npy", mmap_mode="r")
    H_va = np.load(d / "val_heads.npy", mmap_mode="r")
    H_im = np.load(d / "imagenette_heads.npy", mmap_mode="r")
    hr = np.load(d / "train_head_rows.npy")
    mean_h = torch.as_tensor(np.asarray(H_tr, dtype=np.float32).mean(0))  # [4, 16, 1024], train mean
    eta = {}
    for attr in ATTRS:
        lab = ctx.train.groups(attr)[0][hr]
        G = ctx.train.groups(attr)[1]
        e = np.zeros((4, 16))
        for li in range(4):
            X = torch.as_tensor(np.asarray(H_tr[:, li], dtype=np.float32))  # [n, 16, 1024]
            mu = X.mean(0)
            tot = ((X - mu) ** 2).sum((0, 2))
            btw = torch.zeros(16)
            for gi in range(G):
                m = torch.as_tensor(lab == gi)
                if m.any():
                    btw += m.sum() * ((X[m].mean(0) - mu) ** 2).sum(1)
            e[li] = (btw / tot).numpy()
        eta[attr] = e
    top = {a: [divmod(int(i), 16) for i in np.argsort(-eta[a].flatten())[:3]] for a in ATTRS}
    union = sorted(set(sum(top.values(), [])))

    def ablate(R, H, heads):
        R = torch.as_tensor(R, dtype=torch.float32).clone()
        for li, h in heads:
            R -= torch.as_tensor(np.asarray(H[:, li, h], dtype=np.float32)) - mean_h[li, h]
        return R

    arms, abil, leak = {}, {}, {}
    clip_m = ctx.metrics(lib.embed(ctx.val.r))
    clip_ab = ctx.ability(lib.embed(ctx.imn.r))
    for name, heads in [(f"heads_{a}", top[a]) for a in ATTRS] + [("heads_all", union)]:
        Ev = lib.embed(ablate(ctx.val.r, H_va, heads))
        arms[name] = ctx.metrics(Ev)
        abil[name] = ctx.ability(lib.embed(ablate(ctx.imn.r, H_im, heads)))
        leak[name] = ctx.leakage(lib.embed(ablate(ctx.train.r[hr], H_tr, heads)), Ev)
    band = {k: [] for k in ("marg_excess", "joint_excess", "paper_eo", "imagenette")}
    for _ in range(N_RAND):
        heads = [divmod(int(x), 16) for x in rng.choice(64, len(union), replace=False)]
        m = ctx.metrics(lib.embed(ablate(ctx.val.r, H_va, heads)))
        for k in ("marg_excess", "joint_excess", "paper_eo"):
            band[k].append(float(m[k].mean()))
        band["imagenette"].append(float(ctx.ability(lib.embed(ablate(ctx.imn.r, H_im, heads))).mean()))
    dbl_all = json.loads((RES / "core.json").read_text())["selection"]["chosen"]["all"]
    dbl1 = ctx.metrics(lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, dbl_all, 1.0, codes=ctx.codes_va)))
    res = {"eta2": {a: e.round(4).tolist() for a, e in eta.items()},
           "top_heads": {a: [[20 + li, h] for li, h in v] for a, v in top.items()},
           "union": [[20 + li, h] for li, h in union],
           "arms": {k: summarise(v, ctx.cluster, KEYS, ref=clip_m) for k, v in arms.items()},
           "ability": {k: acc_summary(v, clip_ab) for k, v in abil.items()},
           "leakage": {k: {a: {"acc": v[a]["acc"], "bal_acc": v[a]["bal_acc"]} for a in ATTRS} for k, v in leak.items()},
           "random_band": {k: sorted(v) for k, v in band.items()},
           "heads_all_vs_dbl_all@1.0": summarise(arms["heads_all"], ctx.cluster, ["marg_excess", "joint_excess", "paper_eo"], ref=dbl1)}  # fmt: skip
    jdump("b2", res)


# ------------------------------------------------------------------ B3 image side vs text side
def b3(ctx):
    dbl_all = json.loads((RES / "core.json").read_text())["selection"]["chosen"]["all"]
    Dm = torch.cat([T - T.mean(0, keepdim=True) for T in ctx.T_group.values()])
    U, s, _ = torch.linalg.svd(Dm.T, full_matrices=False)
    Q = U[:, s > 1e-6 * s.max()]
    Tp = torch.nn.functional.normalize(ctx.T - (ctx.T @ Q) @ Q.T, dim=-1)
    E0 = lib.embed(ctx.val.r)
    E1 = lib.embed(lib.sae_blend(ctx.sae, ctx.val.r, dbl_all, 0.6, codes=ctx.codes_va))
    arms = {
        "clip": ctx.metrics(E0),
        "image": ctx.metrics(E1),
        "text": ctx.metrics(E0, T=Tp),
        "both": ctx.metrics(E1, T=Tp),
    }
    E0_tr = lib.embed(ctx.train.r)
    res = {"rank_text_projection": int(Q.shape[1]),
           "arms": {k: summarise(v, ctx.cluster, KEYS, ref=None if k == "clip" else arms["clip"]) for k, v in arms.items()},
           "both_minus_text": summarise(arms["both"], ctx.cluster, KEYS, ref=arms["text"]),
           "both_minus_image": summarise(arms["both"], ctx.cluster, KEYS, ref=arms["image"])}  # fmt: skip
    dec = {}
    for attr in ATTRS:
        lab, G = ctx.val.groups(attr)
        oh = torch.as_tensor(lib.onehot(lab, G), dtype=torch.float32)
        cnt = oh.sum(0)
        n = float(len(lab))
        # the over-retrieved group is chosen on train (as in b1), then the gap is measured on val
        lab_tr = ctx.train.groups(attr)[0]
        sims_tr = ctx.T @ E0_tr.T
        Ktr = np.round(ctx.K * ctx.train.n / ctx.val.n).astype(int)
        gstar = torch.zeros(len(ctx.T), dtype=torch.long)
        for K in sorted(set(Ktr.tolist())):
            rws = np.where(Ktr == K)[0]
            cK = lib.topk_counts(sims_tr[rws], lab_tr, G, int(K))
            gstar[torch.as_tensor(rws)] = lib.skew_dp(cK, int(K), lib.group_share(lab_tr, G)).argmax(1)
        c = cnt[gstar][:, None]

        def delta(E):
            mg = (oh.T @ E) / cnt[:, None]
            return mg[gstar] - (E.sum(0) - mg[gstar] * c) / (n - c)

        d0, d1 = delta(E0), delta(E1)
        base, img = (d0 * ctx.T).sum(1), ((d1 - d0) * ctx.T).sum(1)
        txt, inter = (d0 * (Tp - ctx.T)).sum(1), ((d1 - d0) * (Tp - ctx.T)).sum(1)
        dec[attr] = {"gap_clip": float(base.mean()), "image_term": float(img.mean()), "text_term": float(txt.mean()),
                     "interaction": float(inter.mean()), "gap_both": float((base + img + txt + inter).mean()),
                     "gap_text_only": float((base + txt).mean()), "gap_image_only": float((base + img).mean()),
                     "share_interaction_opposes_text": float(((inter * txt) < 0).float().mean()),
                     "share_image_term_same_sign_as_text_term": float(((img * txt) > 0).float().mean())}  # fmt: skip
    res["decomposition"] = dec
    jdump("b3", res)


# ------------------------------------------------------------------ exploratory (added after the core results)
def errkeep(ctx):
    """Zero latents while keeping the SAE's reconstruction error: h' = h - s * sum_j a_j d_j. At s = 1 this removes
    exactly the chosen latents' contribution and nothing else, so it separates "what the latents do" from "what
    replacing the CLS row with a lossy reconstruction does". Added after the core results showed the reconstruction
    alone raises skew; exploratory, not part of any registered prediction."""
    rng = np.random.default_rng(5)
    core_res = json.loads((RES / "core.json").read_text())
    chosen = core_res["selection"]["chosen"]
    D = ctx.sae.W_dec

    def edit(r, codes, lat, s=1.0):
        lat = torch.as_tensor(list(lat))
        return torch.as_tensor(r, dtype=torch.float32) - s * codes[:, lat] @ D[lat]

    clip_m = ctx.metrics(lib.embed(ctx.val.r))
    clip_ab = ctx.ability(lib.embed(ctx.imn.r))
    rows = torch.as_tensor(ctx.probe_rows)
    arms, abil, leak, band = {}, {}, {}, {}
    sets = {"gender": chosen["gender"], "age7": chosen["age7"], "race": chosen["race"], "all": chosen["all"],
            "released_all": sorted(set().union(*RELEASED.values()))}  # fmt: skip
    for name, lat in sets.items():
        for s in (1.0, 2.0):
            k = f"ek_{name}@x{s:g}"
            Ev = lib.embed(edit(ctx.val.r, ctx.codes_va, lat, s))
            arms[k] = ctx.metrics(Ev)
            abil[k] = ctx.ability(lib.embed(edit(ctx.imn.r, ctx.codes_imn, lat, s)))
            if s == 1.0:
                leak[k] = ctx.leakage(lib.embed(edit(ctx.train.r[ctx.probe_rows], ctx.codes_tr[rows], lat, s)), Ev)
        if name in ("gender", "age7", "race", "all"):
            for s in (1.0, 2.0):
                vals = {key: [] for key in KEYS}
                vals["imagenette"] = []
                for d in lib.freq_matched(ctx.freq, lat, N_RAND, rng, exclude=chosen["all"]):
                    m = ctx.metrics(lib.embed(edit(ctx.val.r, ctx.codes_va, d, s)))
                    for key in KEYS:
                        vals[key].append(float(m[key].mean()))
                    vals["imagenette"].append(
                        float(ctx.ability(lib.embed(edit(ctx.imn.r, ctx.codes_imn, d, s))).mean())
                    )
                band[f"rand_{name}@x{s:g}"] = {key: sorted(v) for key, v in vals.items()}
        print(f"[errkeep] {name} done", flush=True)
    res = {"note": errkeep.__doc__,
           "arms": {k: summarise(v, ctx.cluster, KEYS, ref=clip_m) for k, v in arms.items()},
           "ability": {k: acc_summary(v, clip_ab) for k, v in abil.items()},
           "leakage": {k: {a: {"acc": v[a]["acc"], "bal_acc": v[a]["bal_acc"]} for a in ATTRS} for k, v in leak.items()},
           "random_bands": band, "gerrymander": gerrymander({"clip": clip_m, **arms}, ctx)}  # fmt: skip
    np.savez_compressed(
        RES / "prompt_metrics_errkeep.npz", **{f"{a}::{k}": v for a, d in arms.items() for k, v in d.items()}
    )
    jdump("errkeep", res)


# ------------------------------------------------------------------ robustness checks (added after code review)
def _metrics_K(ctx, sims, K):
    """metrics_from_sims for rows that all use top-K = K (random directions, object prompts)."""
    old = ctx.K
    ctx.K = np.full(len(sims), K)
    try:
        return metrics_from_sims(ctx, sims)
    finally:
        ctx.K = old


def additive_prediction(ctx, cells_skew_counts, Ks, n_sim=50, seed=0):
    """Per prompt: the joint excess and additive-fit RSS expected if the top K followed an independence (log-additive)
    model that reproduces that prompt's own gender, age3 and race counts exactly (iterative proportional fitting from
    the pool's 42-cell make-up), sampled K at a time like the real retrieval."""
    rng = np.random.default_rng(seed)
    lab = ctx.val.cell
    n_c = np.bincount(lab, minlength=42).astype(float)
    share = n_c / n_c.sum()
    g, rest = np.divmod(np.arange(42), 21)
    a, r = np.divmod(rest, 7)
    idx = [(g, 2), (a, 3), (r, 7)]
    pred_ex, pred_rss = np.zeros(len(Ks)), np.zeros(len(Ks))
    for p, (counts, K) in enumerate(zip(cells_skew_counts, Ks)):
        targets = [np.bincount(m, weights=counts, minlength=k) for m, k in idx]
        q = share * K
        for _ in range(30):
            for (m, k), t in zip(idx, targets):
                cur = np.bincount(m, weights=q, minlength=k)
                q = q * np.where(cur > 0, t / np.maximum(cur, 1e-12), 0)[m]
        q = np.clip(q, 0, None)
        sims = rng.multinomial(int(K), q / q.sum(), size=n_sim).astype(np.float32)
        s = lib.skew_dp(torch.as_tensor(sims), int(K), share)
        fl = ctx.floors[("joint", int(K))]
        pred_ex[p] = float(s.max(1).values.mean()) - fl["maxskew_dp"]
        pred_rss[p] = float(lib.additive_rss(s, share).mean())
    return pred_ex, pred_rss


def joint_counts(ctx, sims):
    lab = ctx.val.cell
    out = np.zeros((len(sims), 42))
    for K in sorted(set(ctx.K.tolist())):
        rows = np.where(ctx.K == K)[0]
        out[rows] = lib.topk_counts(sims[rows], lab, 42, int(K)).numpy()
    return out


def robust(ctx):
    """Exploratory checks added after code review: (1) how much of joint skew an independence model of the attributes
    already predicts; (2) the additive-fit misfit against additive (not unskewed) data; (3) the same metrics for random
    directions and object prompts (is a prompt more skewed than any direction?); (4) a lossy non-SAE rebuild with the
    SAE's class-token FVU; (5) a random band for the error-kept removal matched on each latent's mean contribution."""
    out = {"note": robust.__doc__}
    E0 = lib.embed(ctx.val.r)
    sims = ctx.T @ E0.T
    m0 = metrics_from_sims(ctx, sims)
    rss_obs = None
    # (1) and (2)
    counts = joint_counts(ctx, sims)
    pred_ex, pred_rss = additive_prediction(ctx, counts, ctx.K)
    share = lib.group_share(ctx.val.cell, 42)
    rss_obs = np.zeros(len(sims))
    for K in sorted(set(ctx.K.tolist())):
        rows = np.where(ctx.K == K)[0]
        s = lib.skew_dp(torch.as_tensor(counts[rows], dtype=torch.float32), int(K), share)
        rss_obs[rows] = lib.additive_rss(s, share).numpy()
    inter = m0["joint_excess"] - pred_ex
    out["clip_prompts"] = {"joint_excess": float(m0["joint_excess"].mean()), "additive_predicted_joint_excess": float(pred_ex.mean()),
                           "interaction_excess": {"mean": float(inter.mean()), "ci": ci(cluster_boot(inter, ctx.cluster))},
                           "rss_ratio_vs_random": float(np.median(m0["rss_ratio"])),
                           "rss_ratio_vs_additive": float(np.median(rss_obs / pred_rss)),
                           "share_prompts_rss_vs_additive_gt_1": float((rss_obs / pred_rss > 1).mean())}  # fmt: skip
    # (3) random directions (isotropic in the 768-d text space) and the 10 ImageNette object prompts
    gen = torch.Generator().manual_seed(7)
    R = torch.nn.functional.normalize(torch.randn(200, 768, generator=gen), dim=-1)
    for name, T in (("random_directions", R), ("object_prompts", ctx.T_imn)):
        s2 = T @ E0.T
        m = _metrics_K(ctx, s2, 1000)
        old = ctx.K
        ctx.K = np.full(len(T), 1000)
        c2 = joint_counts(ctx, s2)
        pe, prss = additive_prediction(ctx, c2, ctx.K)
        ctx.K = old
        sk2 = lib.skew_dp(torch.as_tensor(c2, dtype=torch.float32), 1000, share)
        rss2 = lib.additive_rss(sk2, share).numpy()
        out[name] = {"n": int(len(T)), **{k: float(m[k].mean()) for k in ("gender_excess", "age3_excess", "race_excess", "joint_excess", "marg_excess", "paper_eo")},
                     "rss_ratio_vs_random_median": float(np.median(m["rss_ratio"])),
                     "interaction_excess": float((m["joint_excess"] - pe).mean()),
                     "rss_ratio_vs_additive_median": float(np.median(rss2 / prss)),
                     "joint_excess_range": [float(m["joint_excess"].min()), float(m["joint_excess"].max())]}  # fmt: skip
    # prompts vs random directions: share of prompts above the random directions' median joint excess
    out["prompts_vs_random_directions"] = {k: {"prompt_mean": float(m0[k].mean()), "random_mean": out["random_directions"][k]}
                                           for k in ("marg_excess", "joint_excess", "paper_eo")}  # fmt: skip
    # (4) PCA rebuild with the SAE's class-token FVU, blended like the SAE
    Xtr = torch.as_tensor(ctx.train.r)
    mu = Xtr.mean(0)
    U, S, Vt = torch.linalg.svd(Xtr[::4] - mu, full_matrices=False)
    rv = torch.as_tensor(ctx.val.r)
    tot = ((rv - rv.mean(0)) ** 2).sum()
    sae_fvu = json.loads((RES / "core.json").read_text())["sae_fidelity"]["fvu_cls_val"]
    best = None
    for k in range(1, 200):
        V = Vt[:k]
        rec = mu + ((rv - mu) @ V.T) @ V
        fvu = float(((rv - rec) ** 2).sum() / tot)
        if best is None or abs(fvu - sae_fvu) < abs(best[1] - sae_fvu):
            best = (k, fvu)
        if fvu < sae_fvu - 0.05:
            break
    k = best[0]
    V = Vt[:k]

    def pca_blend(r, al):
        r = torch.as_tensor(r, dtype=torch.float32)
        return (1 - al) * r + al * (mu + ((r - mu) @ V.T) @ V)

    clip_ab = ctx.ability(lib.embed(ctx.imn.r))
    out["pca_lossy"] = {"rank": int(k), "fvu_val_cls": best[1], "sae_fvu_val_cls": sae_fvu}
    for al in (0.6, 1.0):
        m = ctx.metrics(lib.embed(pca_blend(ctx.val.r, al)))
        out["pca_lossy"][f"alpha{al}"] = {**summarise(m, ctx.cluster, ["marg_excess", "joint_excess", "paper_eo", "gender_eo"], ref=m0),
                                          "imagenette": acc_summary(ctx.ability(lib.embed(pca_blend(ctx.imn.r, al))), clip_ab)}  # fmt: skip
    # (5) error-kept removal: random latents matched on mean contribution (mean activation; decoder rows have unit norm)
    rng = np.random.default_rng(9)
    chosen = json.loads((RES / "core.json").read_text())["selection"]["chosen"]["all"]
    meanact = ctx.codes_tr.mean(0).numpy()
    live = np.where(ctx.freq > 0)[0]
    pool = np.array([j for j in live if j not in set(chosen)])
    D = ctx.sae.W_dec
    order = pool[np.argsort(meanact[pool])]
    vals = {k: [] for k in ("marg_excess", "joint_excess", "paper_eo", "gender_maxdp", "race_maxdp")}
    for _ in range(N_RAND):
        pick = []
        for j in chosen:
            pos = np.searchsorted(meanact[order], meanact[j])
            near = order[max(0, pos - 10) : pos + 10]
            near = [x for x in near if x not in pick]
            pick.append(int(rng.choice(near)))
        lat = torch.as_tensor(pick)
        m = ctx.metrics(lib.embed(torch.as_tensor(ctx.val.r) - ctx.codes_va[:, lat] @ D[lat]))
        for k2 in vals:
            vals[k2].append(float(m[k2].mean() - m0[k2].mean()))
    out["errkeep_contribution_matched_band"] = {k2: sorted(v) for k2, v in vals.items()}
    out["errkeep_contribution"] = {"chosen_mean_activation": [float(meanact[j]) for j in chosen]}
    jdump("robust", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    ctx = Ctx(a.tag)
    if a.part == "cells":
        cells(ctx, a.n_boot)
    else:
        {"core": core, "a3": a3, "a4": a4, "b1": b1, "b2": b2, "b3": b3, "errkeep": errkeep, "robust": robust}[a.part](
            ctx
        )


if __name__ == "__main__":
    main()
