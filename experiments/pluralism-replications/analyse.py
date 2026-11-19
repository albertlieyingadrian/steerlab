"""Compute every number and pre-registered verdict for Episode 2 from results/raw/ -> results/summary.json + tables.

    .venv/bin/python experiments/pluralism-replications/analyse.py --run full500

Verdict rules are the pre-registered plan's: the paper's direction
with the 95% bootstrap interval excluding zero = replicated; the other direction = not replicated; else null.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measures as M

RES = HERE / "results"
NB = M.N_BOOT
FAMILIES = {
    "gemma3-12b": {
        "PT": "google/gemma-3-12b-pt",
        "IT": "google/gemma-3-12b-it",
        "ST": "tsor13/spectrum-gemma-3-12b-v1",
    },
    "qwen3-8b": {"PT": "Qwen/Qwen3-8B-Base", "IT": "Qwen/Qwen3-8B"},
    "qwen3.5-9b": {"PT": "Qwen/Qwen3.5-9B-Base", "IT": "Qwen/Qwen3.5-9B"},
}
FORMAT = {"PT": "colon", "IT": "chat", "ST": "spectrum"}
CATEGORICAL = [
    "habermas_individual_categorical",
    "numbergame_individual",
    "chatbotarena_individual_prefs",
    "flight",
    "wvs_individual",
]
OPEN_INSTRUCT = ["google/gemma-3-12b-it", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"]
FRONTIER = ["anthropic/claude-sonnet-5.5", "openai/gpt-5.4-mini", "google/gemini-3.8-flash"]
DEVELOPER = {
    "google/gemma-3-12b-it": "Google",
    "google/gemini-3.8-flash": "Google",
    "Qwen/Qwen3-8B": "Alibaba",
    "Qwen/Qwen3.5-9B": "Alibaba",
    "anthropic/claude-sonnet-5.5": "Anthropic",
    "openai/gpt-5.4-mini": "OpenAI",
}
JUDGES = [
    "openai/gpt-4o-2024-11-20",
    "anthropic/claude-haiku-5.5",
    "prometheus-eval/prometheus-7b-v2.0",
    "Qwen/Qwen3-8B",
]


def mdir(run, model):
    return RES / "raw" / run / model.replace("/", "__")


def ci(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------------------------------------------------
# A: steerability
# ---------------------------------------------------------------------------------------------------------------------


def load_steer(run, model, fmt):
    d = mdir(run, model) / f"steer_{fmt}"
    out = {}
    for f in sorted(d.glob("*.parquet")) if d.exists() else []:
        raw = pd.read_parquet(f)
        out[f.stem] = M.steerability_per_instance(raw).set_index("instance_ind")
    return out


def analyse_A(run, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    fam_boot, fam_est, rows_task = {}, {}, []
    gain_boot, gain_est = {}, {}
    for fam, mods in FAMILIES.items():
        pt = load_steer(run, mods["PT"], "colon")
        it = load_steer(run, mods["IT"], "chat")
        it_colon = load_steer(run, mods["IT"], "colon")
        st = load_steer(run, mods["ST"], "spectrum") if "ST" in mods else {}
        tasks = sorted(set(pt) & set(it))
        if not tasks:
            continue
        rel_boot = np.zeros((NB, len(tasks)))
        g_boot = np.zeros((NB, len(tasks)))
        rel_est, g_est = [], []
        for j, t in enumerate(tasks):
            common = pt[t].index.intersection(it[t].index)
            a = it[t].loc[common, "loss_geq_1"].to_numpy()
            b = pt[t].loc[common, "loss_geq_1"].to_numpy()
            # context gain (loss on the first output minus mean loss on later ones), in units of the base model's
            # mean first-output loss on the task; per-instance ratios explode when a first-output loss is near 0
            ga = it[t].loc[common, "context_gain"].to_numpy()
            gb = pt[t].loc[common, "context_gain"].to_numpy()
            lf = pt[t].loc[common, "loss_first"].to_numpy()
            idx = rng.integers(0, len(a), (NB, len(a)))
            rel_boot[:, j] = (a[idx].mean(1) - b[idx].mean(1)) / b[idx].mean(1)
            g_boot[:, j] = (np.nanmean(gb[idx], 1) - np.nanmean(ga[idx], 1)) / np.nanmean(lf[idx], 1)
            rel = (a.mean() - b.mean()) / b.mean()
            rel_est.append(rel)
            g_est.append((np.nanmean(gb) - np.nanmean(ga)) / np.nanmean(lf))
            lo, hi = M.percentile_ci(rel_boot[:, j])
            row = {
                "family": fam,
                "task": t,
                "n": len(common),
                "loss_PT": b.mean(),
                "loss_IT_chat": a.mean(),
                "rel_change": rel,
                "lo": lo,
                "hi": hi,
                "verdict": M.verdict(lo, hi),
                "gain_PT": float(np.nanmean(gb) / np.nanmean(lf)),
                "gain_IT": float(np.nanmean(ga) / np.nanmean(lf)),
            }
            if t in it_colon:
                c = it_colon[t].loc[it_colon[t].index.intersection(common), "loss_geq_1"]
                row["loss_IT_colon"] = float(c.mean())
                row["rel_change_colon"] = float((c.mean() - b.mean()) / b.mean())
            if t in st:
                s = st[t].loc[st[t].index.intersection(common), "loss_geq_1"]
                row["loss_ST"] = float(s.mean())
                row["rel_change_ST"] = float((s.mean() - b.mean()) / b.mean())
            if t in CATEGORICAL and "acc_geq_1" in pt[t]:
                row["acc_PT"] = float(pt[t].loc[common, "acc_geq_1"].mean())
                row["acc_IT"] = float(it[t].loc[common, "acc_geq_1"].mean())
            rows_task.append(row)
        fam_boot[fam] = rel_boot.mean(1)
        fam_est[fam] = float(np.mean(rel_est))
        gain_boot[fam] = g_boot.mean(1)
        gain_est[fam] = float(np.mean(g_est))
    per_task = pd.DataFrame(rows_task)
    per_task.to_csv(RES / "A_steerability_per_task.csv", index=False)
    fams = {}
    for fam in fam_boot:
        lo, hi = M.percentile_ci(fam_boot[fam])
        glo, ghi = M.percentile_ci(gain_boot[fam])
        sub = per_task[per_task.family == fam]
        fams[fam] = ci(
            {
                "rel_change": fam_est[fam],
                "lo": lo,
                "hi": hi,
                "verdict": M.verdict(lo, hi),
                "n_tasks": len(sub),
                "tasks_IT_worse": int((sub.verdict == "replicated").sum()),
                "tasks_IT_better": int((sub.verdict == "not replicated").sum()),
                "gain_gap": gain_est[fam],
                "gain_lo": glo,
                "gain_hi": ghi,
                "rel_change_colon_mean": float(sub.rel_change_colon.mean()) if "rel_change_colon" in sub else None,
            }
        )
    pooled = np.mean([fam_boot[f] for f in fam_boot], 0)
    lo, hi = M.percentile_ci(pooled)
    est = float(np.mean([fam_est[f] for f in fam_boot]))
    gpool = np.mean([gain_boot[f] for f in gain_boot], 0)
    glo, ghi = M.percentile_ci(gpool)
    return {
        "pooled": ci({"rel_change": est, "lo": lo, "hi": hi, "verdict": M.verdict(lo, hi)}),
        "context_gain_gap_pooled": ci(
            {"est": float(np.mean(list(gain_est.values()))), "lo": glo, "hi": ghi, "verdict": M.verdict(glo, ghi)}
        ),
        "families": fams,
        "per_task": per_task.round(4).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------------------------------------------------
# B: coverage
# ---------------------------------------------------------------------------------------------------------------------


def load_cov(run, model, tmpl):
    f = mdir(run, model) / f"coverage_{tmpl}.jsonl"
    rows = []
    if not f.exists():
        return pd.DataFrame()
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        st = r["stats"] if r.get("redacted") else M.coverage_stats(r["generations"], r["validity"])
        rows.append({k: r[k] for k in ("task", "setting", "seed")} | st)
    return pd.DataFrame(rows)


def analyse_B(run, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    per_model, fam_boot, fam_est, rows = {}, {}, {}, []
    for fam, mods in FAMILIES.items():
        dfs = {role: load_cov(run, m, FORMAT[role]) for role, m in mods.items()}
        for role, df in dfs.items():
            if len(df):
                per_model[mods[role]] = ci(
                    {
                        "yield": float(df["yield"].mean()),
                        "validity": float(df.validity.mean()),
                        "diversity": float(df.diversity.mean()),
                        "n_cells": len(df),
                    }
                    | {
                        f"yield_{s}": float(df[df.setting == s]["yield"].mean())
                        for s in ("description", "examples", "both")
                    }
                    | {
                        f"validity_{s}": float(df[df.setting == s].validity.mean())
                        for s in ("description", "examples", "both")
                    }
                )
        pt, it = dfs["PT"], dfs["IT"]
        if not len(pt) or not len(it):
            continue
        m = pt.merge(it, on=["task", "setting", "seed"], suffixes=("_pt", "_it"))
        m["gap"] = m.yield_pt - m.yield_it
        tasks = sorted(m.task.unique())
        per_task = {t: m[m.task == t] for t in tasks}
        boots = []
        for _ in range(NB):
            drawn = rng.integers(0, len(tasks), len(tasks))
            vals = []
            for i in drawn:
                sub = per_task[tasks[i]]
                seeds = sub.seed.unique()
                pick = rng.choice(seeds, len(seeds))
                vals.append(np.mean([sub[sub.seed == s].gap.mean() for s in pick]))
            boots.append(np.mean(vals))
        fam_boot[fam] = np.array(boots)
        fam_est[fam] = float(m.groupby("task").gap.mean().mean())
        for t, sub in per_task.items():
            for s in ("description", "examples", "both"):
                ss = sub[sub.setting == s]
                rows.append(
                    {
                        "family": fam,
                        "task": t,
                        "setting": s,
                        "yield_PT": ss.yield_pt.mean(),
                        "yield_IT": ss.yield_it.mean(),
                        "validity_PT": ss.validity_pt.mean(),
                        "validity_IT": ss.validity_it.mean(),
                        "diversity_PT": ss.diversity_pt.mean(),
                        "diversity_IT": ss.diversity_it.mean(),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(RES / "B_coverage_per_task.csv", index=False)
    fams = {}
    for fam in fam_boot:
        lo, hi = M.percentile_ci(fam_boot[fam])
        fams[fam] = ci({"yield_gap": fam_est[fam], "lo": lo, "hi": hi, "verdict": M.verdict(lo, hi)})
    pooled = np.mean([fam_boot[f] for f in fam_boot], 0)
    lo, hi = M.percentile_ci(pooled)
    est = float(np.mean(list(fam_est.values())))
    return {
        "pooled": ci({"yield_gap": est, "lo": lo, "hi": hi, "verdict": M.verdict(lo, hi)}),
        "families": fams,
        "per_model": per_model,
        "per_task": table.round(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------------------------------------------------
# C, D: repetition and homogeneity
# ---------------------------------------------------------------------------------------------------------------------


def load_samples(run):
    rows = []
    for m in OPEN_INSTRUCT:
        f = mdir(run, m) / "hivemind.jsonl"
        if f.exists():
            rows += [json.loads(x) | {"kind": "instruct"} for x in f.read_text().splitlines()]
    for mods in FAMILIES.values():
        f = mdir(run, mods["PT"]) / "hivemind_base.jsonl"
        if f.exists():
            rows += [json.loads(x) | {"kind": "base"} for x in f.read_text().splitlines()]
    for m in FRONTIER:
        f = RES / "raw" / "full" / "frontier" / f"{m.replace('/', '__')}.jsonl"
        if f.exists():
            rows += [json.loads(x) | {"kind": "instruct"} for x in f.read_text().splitlines()]
    return pd.DataFrame(rows).sort_values(["model", "prompt_id", "sample"]).reset_index(drop=True)


def gram_cube(df, emb, models, n_prompts, n_samp, drop_empty=True):
    """(models, prompts, samples, d) embedding cube; missing samples stay NaN."""
    cube = np.full((len(models), n_prompts, n_samp, emb.shape[1]), np.nan, dtype=np.float32)
    mi = {m: i for i, m in enumerate(models)}
    for r, e in zip(df.itertuples(), emb):
        if r.model in mi and r.sample < n_samp and (str(r.text).strip() or not drop_empty):  # empties left out
            cube[mi[r.model], r.prompt_id, r.sample] = e
    return cube


def _rep_from(Gm, s_idx, vm):
    """Mean pairwise similarity per prompt for drawn sample indices; pairs of the same drawn sample are excluded."""
    sub = np.take_along_axis(np.take_along_axis(Gm, s_idx[:, :, None], 1), s_idx[:, None, :], 2)
    v = np.take_along_axis(vm, s_idx, 1)
    ok = (s_idx[:, :, None] != s_idx[:, None, :]) & v[:, :, None] & v[:, None, :]
    n = ok.sum((1, 2))
    return np.where(n > 0, (sub * ok).sum((1, 2)) / np.maximum(n, 1), np.nan)  # no valid pair -> NaN


def _cross_from(Xab, okab, p_idx, sa, sb):
    sub = np.take_along_axis(np.take_along_axis(Xab[p_idx], sa[:, :, None], 1), sb[:, None, :], 2)
    ok = np.take_along_axis(np.take_along_axis(okab[p_idx], sa[:, :, None], 1), sb[:, None, :], 2)
    n = ok.sum((1, 2))
    return np.where(n > 0, (sub * ok).sum((1, 2)) / np.maximum(n, 1), np.nan)


def analyse_CD(run, rng_seed=0, drop_empty=True):
    rng = np.random.default_rng(rng_seed)
    df = load_samples(run)
    from openrouter_client import OpenRouter

    client = OpenRouter("hivemind-embed")
    emb = M.embed_texts(df.text.fillna("").tolist(), cache_path=RES / "cache" / "embeddings.jsonl")
    if M.embed_texts.usd_used:
        client.log_extra("openai/text-embedding-3-small", M.embed_texts.usd_used)
    df["is_empty"] = df.text.fillna("").str.strip() == ""
    P, S = int(df.prompt_id.max()) + 1, int(df["sample"].max()) + 1
    present = set(df.model)
    instruct = [m for m in OPEN_INSTRUCT + FRONTIER if m in present]
    base = [FAMILIES[f]["PT"] for f in FAMILIES if FAMILIES[f]["PT"] in present]
    allm = instruct + base
    cube = gram_cube(df, emb, allm, P, S, drop_empty=drop_empty)
    valid = ~np.isnan(cube[..., 0])
    cube = np.nan_to_num(cube)
    G = np.einsum("mpsd,mptd->mpst", cube, cube)
    ident = np.tile(np.arange(S), (P, 1))
    rep = {m: _rep_from(G[i], ident, valid[i]) for i, m in enumerate(allm)}
    rep_tab = pd.DataFrame(
        [{"model": m, "prompt_id": p, "repetition": float(rep[m][p])} for m in allm for p in range(P)]
    )
    rep_tab.to_csv(RES / "C_repetition_per_prompt.csv", index=False)
    boots_med = {m: [] for m in allm}
    boots_mean = {m: [] for m in allm}
    for _ in range(NB):
        p_idx = rng.integers(0, P, P)
        s_idx = rng.integers(0, S, (P, S))
        for i, m in enumerate(allm):
            r = _rep_from(G[i][p_idx], s_idx, valid[i][p_idx])
            boots_med[m].append(np.nanmedian(r))
            boots_mean[m].append(np.nanmean(r))
    C = {}
    for m in allm:
        lo, hi = M.percentile_ci(np.array(boots_med[m]) - 0.8)
        mlo, mhi = M.percentile_ci(boots_mean[m])
        sub = df[df.model == m]
        C[m] = ci(
            {
                "median": float(np.nanmedian(rep[m])),
                "median_minus_0.8_lo": lo,
                "median_minus_0.8_hi": hi,
                "verdict": M.verdict(lo, hi) if m in instruct else "n/a (base)",
                "mean": float(np.nanmean(rep[m])),
                "mean_lo": mlo,
                "mean_hi": mhi,
                "share_above_0.8": float(
                    np.nanmean(rep[m] > 0.8) if False else (rep[m][~np.isnan(rep[m])] > 0.8).mean()
                ),
                "empty_rate": float(sub.is_empty.mean()),
                "truncated_rate": float(sub.truncated.fillna(False).astype(bool).mean()),
                "mean_tokens": float(pd.to_numeric(sub.n_tokens, errors="coerce").mean()),
            }
        )
    n_rep = sum(C[m]["verdict"] == "replicated" for m in instruct)
    floor = M.cross_prompt_floor(df, emb).set_index("model")["floor"].to_dict()
    cprime = {}
    for fam, mods in FAMILIES.items():
        a, b = mods["IT"], mods["PT"]
        if a in rep and b in rep:
            ia, ib = allm.index(a), allm.index(b)
            bs = []
            for _ in range(NB):
                p_idx = rng.integers(0, P, P)
                sa, sb = rng.integers(0, S, (P, S)), rng.integers(0, S, (P, S))
                bs.append(
                    np.nanmean(_rep_from(G[ia][p_idx], sa, valid[ia][p_idx]))
                    - np.nanmean(_rep_from(G[ib][p_idx], sb, valid[ib][p_idx]))
                )
            lo, hi = M.percentile_ci(bs)
            cprime[fam] = ci(
                {
                    "instruct_minus_base": float(np.nanmean(rep[a]) - np.nanmean(rep[b])),
                    "lo": lo,
                    "hi": hi,
                    "verdict": M.verdict(lo, hi),
                }
            )
    pairs = [(a, b) for i, a in enumerate(instruct) for b in instruct[i + 1 :]]
    X = {}
    for a, b in pairs:
        ia, ib = allm.index(a), allm.index(b)
        X[(a, b)] = (np.einsum("psd,ptd->pst", cube[ia], cube[ib]), valid[ia][:, :, None] & valid[ib][:, None, :])
    allp = np.arange(P)
    pair_rows = []
    for (a, b), (Xab, okab) in X.items():
        cr = _cross_from(Xab, okab, allp, ident, ident)
        within = (rep[a] + rep[b]) / 2
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "cross": float(np.nanmean(cr)),
                "ratio": float(np.nanmean(cr / within)),
                "cross_developer": DEVELOPER.get(a) != DEVELOPER.get(b),
                "frontier_open": (a in FRONTIER) != (b in FRONTIER),
            }
        )
    pair_tab = pd.DataFrame(pair_rows)
    pair_tab.to_csv(RES / "D_cross_model_pairs.csv", index=False)
    dboot, dboot_dev, dboot_fo = [], [], []
    dev_pairs = [p for p in X if DEVELOPER.get(p[0]) != DEVELOPER.get(p[1])]
    fo_pairs = [p for p in X if (p[0] in FRONTIER) != (p[1] in FRONTIER)]
    for _ in range(NB):
        p_idx = rng.integers(0, P, P)
        draws = {m: rng.integers(0, S, (P, S)) for m in instruct}
        vals = {p: np.nanmean(_cross_from(*X[p], p_idx, draws[p[0]], draws[p[1]])) for p in X}
        dboot.append(np.mean(list(vals.values())))
        dboot_dev.append(np.mean([vals[p] for p in dev_pairs]))
        dboot_fo.append(np.mean([vals[p] for p in fo_pairs]) if fo_pairs else np.nan)
    lo, hi = M.percentile_ci(np.array(dboot) - 0.7)
    dlo, dhi = M.percentile_ci(dboot_dev)
    flo, fhi = M.percentile_ci(dboot_fo)
    D = ci(
        {
            "mean_cross": float(pair_tab.cross.mean()),
            "minus_0.70_lo": lo,
            "minus_0.70_hi": hi,
            "verdict": M.verdict(lo, hi),
            "n_pairs": len(pair_tab),
            "mean_within_instruct": float(np.mean([np.nanmean(rep[m]) for m in instruct])),
            "homogeneity_ratio": float(pair_tab.ratio.mean()),
            "cross_developer_mean": float(pair_tab[pair_tab.cross_developer].cross.mean()),
            "cross_developer_lo": dlo,
            "cross_developer_hi": dhi,
            "frontier_open_mean": float(pair_tab[pair_tab.frontier_open].cross.mean()) if len(fo_pairs) else None,
            "frontier_open_lo": flo,
            "frontier_open_hi": fhi,
            "floor_mean": float(np.mean([floor[m] for m in instruct])),
        }
    )
    q = pd.read_parquet("hf://datasets/liweijiang/infinite-chats-eval/data/train-00000-of-00001.parquet")["query"]
    mean_rep = rep_tab[rep_tab.model.isin(instruct)].groupby("prompt_id").repetition.mean().dropna().sort_values()
    examples = []
    for pid in list(mean_rep.index[-3:][::-1]) + list(mean_rep.index[:2]):
        ex = {"prompt_id": int(pid), "query": q[pid], "mean_repetition": float(mean_rep[pid]), "samples": {}}
        for m in instruct:
            ex["samples"][m] = df[(df.model == m) & (df.prompt_id == pid)].text.head(3).str.slice(0, 500).tolist()
        examples.append(ex)
    return {
        "C": {
            "per_model": C,
            "n_instruct_replicated": int(n_rep),
            "n_instruct": len(instruct),
            "verdict": "replicated" if n_rep >= 4 else ("null" if n_rep >= 1 else "not replicated"),
            "floor": {m: round(v, 4) for m, v in floor.items()},
            "C_prime": cprime,
        },
        "D": D | {"pairs": pair_tab.round(4).to_dict(orient="records")},
        "examples": examples,
        "per_prompt_repetition": rep_tab.round(4).to_dict(orient="records"),
        "queries": q.tolist(),
        "embedding_usd": round(M.embed_texts.usd_used, 4),
        "n_samples": len(df),
    }


# ---------------------------------------------------------------------------------------------------------------------
# E: judges against 25 raters
# ---------------------------------------------------------------------------------------------------------------------


def load_judges(run):
    scores = {}
    for j in JUDGES:
        if j in ("Qwen/Qwen3-8B", "prometheus-eval/prometheus-7b-v2.0"):
            f = mdir(run, j) / "judge.jsonl"
        else:
            f = RES / "raw" / "full" / "judges" / f"{j.replace('/', '__')}.jsonl"
        if f.exists():
            scores[j] = {r["item_id"]: r["score"] for r in map(json.loads, f.read_text().splitlines())}
    return scores


def analyse_E(run, rng_seed=0, n_splits=40):
    rng = np.random.default_rng(rng_seed)
    items = pd.read_parquet("hf://datasets/liweijiang/infinite-chats-human-absolute/data/train-00000-of-00001.parquet")
    items = items.reset_index(drop=True)
    items["item_id"] = items.index
    labels = [np.asarray(x, float) for x in items.human_labels]
    items["human_mean"] = [x.mean() for x in labels]
    items["entropy"] = [M.label_entropy(x.astype(int)) for x in labels]
    items["length"] = items.response.str.len().astype(float)
    items["bin"] = pd.qcut(items.entropy.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    scores = load_judges(run)
    judges = [j for j in JUDGES if j in scores]
    for j in judges:
        items[j] = items.item_id.map(scores[j]).astype(float)
    A = np.zeros((len(items), n_splits))
    B = np.zeros((len(items), n_splits))
    for i, x in enumerate(labels):
        for s in range(n_splits):
            perm = rng.permutation(len(x))
            h = len(x) // 2
            A[i, s], B[i, s] = x[perm[:h]].mean(), x[perm[h:]].mean()

    def ceiling(idx):
        a = A[idx] - A[idx].mean(0)
        b = B[idx] - B[idx].mean(0)
        r = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
        rh = float(np.nanmean(r))
        rho = 2 * rh / (1 + rh)
        return float(np.sqrt(max(rho, 1e-9))), rho

    bins = ["low", "mid", "high"]
    qids = items.user_query.to_numpy()  # cluster = prompt (50); question_id is the response slot (15)
    uq = np.unique(qids)
    by_q = {q: np.where(qids == q)[0] for q in uq}
    bin_arr = items.bin.to_numpy()
    hm_all = items.human_mean.to_numpy()
    cols = {j: items[j].to_numpy() for j in [*judges, "length"]}

    top_all = (items.entropy >= items.entropy.quantile(0.84)).to_numpy()

    def stats(idx):
        out = {}
        it = idx[top_all[idx]]
        for j in cols:
            out[(j, "full")] = M.pearson(cols[j][idx], hm_all[idx])
            out[(j, "top16")] = M.pearson(cols[j][it], hm_all[it])
        out[("ceil", "full")], _ = ceiling(idx)
        out[("ceil", "top16")], _ = ceiling(it)
        for b in bins:
            ib = idx[bin_arr[idx] == b]
            out[("ceil", b)], out[("rho", b)] = ceiling(ib)
            for j in cols:
                out[(j, b)] = M.pearson(cols[j][ib], hm_all[ib])
        return out

    est = stats(np.arange(len(items)))
    boots = []
    for _ in range(NB):
        drawn = rng.choice(uq, len(uq))
        boots.append(stats(np.concatenate([by_q[q] for q in drawn])))

    def summ(fn):
        bs = np.array([fn(b) for b in boots])
        lo, hi = M.percentile_ci(bs)
        se = float(np.nanstd(bs))
        return {"est": float(fn(est)), "lo": lo, "hi": hi, "se": se, "mde80": 2.8 * se}

    per_judge = {}
    for j in cols:
        per_judge[j] = {
            "r": {b: round(est[(j, b)], 4) for b in bins},
            "r_over_ceiling": {b: round(est[(j, b)] / est[("ceil", b)], 4) for b in bins},
            "diff": ci(summ(lambda s, j=j: s[(j, "low")] - s[(j, "high")])),
            "diff_ceiling": ci(
                summ(lambda s, j=j: s[(j, "low")] / s[("ceil", "low")] - s[(j, "high")] / s[("ceil", "high")])
            ),
            "parse_rate": float(items[j].notna().mean()) if j in judges else 1.0,
        }
    raw = summ(lambda s: np.nanmean([s[(j, "low")] - s[(j, "high")] for j in judges]))
    norm = summ(
        lambda s: np.nanmean(
            [s[(j, "low")] / s[("ceil", "low")] - s[(j, "high")] / s[("ceil", "high")] for j in judges]
        )
    )
    ceil_diff = summ(lambda s: s[("ceil", "low")] - s[("ceil", "high")])
    paper_drop = summ(lambda s: np.nanmean([s[(j, "full")] - s[(j, "top16")] for j in judges]))
    paper_drop_ceiling = summ(
        lambda s: np.nanmean(
            [s[(j, "full")] / s[("ceil", "full")] - s[(j, "top16")] / s[("ceil", "top16")] for j in judges]
        )
    )
    sd_range = {b: float(items.human_mean[items.bin == b].std()) for b in bins}
    loo = {b: M.loo_rater_r([labels[i] for i in np.where(bin_arr == b)[0]], n_draws=100) for b in bins}
    top = (items.entropy >= items.entropy.quantile(0.84)).to_numpy()
    paper_style = {j: {"full": M.pearson(cols[j], hm_all), "top16": M.pearson(cols[j][top], hm_all[top])} for j in cols}
    bin_desc = items.groupby("bin").agg(
        n=("item_id", "size"),
        entropy=("entropy", "mean"),
        human_mean=("human_mean", "mean"),
        human_mean_sd=("human_mean", "std"),
    )
    items[["item_id", "question_id", "human_mean", "entropy", "bin", "length", *judges]].to_csv(
        RES / "E_items.csv", index=False
    )
    return {
        "verdict": M.verdict(raw["lo"], raw["hi"]),
        "raw_diff": ci(raw),
        "ceiling_diff_verdict": M.verdict(norm["lo"], norm["hi"]),
        "ceiling_normalised_diff": ci(norm),
        "ceiling": {b: round(est[("ceil", b)], 4) for b in bins},
        "reliability_rho": {b: round(est[("rho", b)], 4) for b in bins},
        "ceiling_low_minus_high": ci(ceil_diff),
        "one_rater_r": {b: round(v, 4) for b, v in loo.items()},
        "per_judge": per_judge,
        "judges": judges,
        "paper_style_drop": ci(paper_drop),
        "paper_style_drop_verdict": M.verdict(paper_drop["lo"], paper_drop["hi"]),
        "paper_style_drop_ceiling": ci(paper_drop_ceiling),
        "paper_style_ceiling": {"full": round(est[("ceil", "full")], 4), "top16": round(est[("ceil", "top16")], 4)},
        "human_mean_sd_by_bin": {b: round(v, 4) for b, v in sd_range.items()},
        "paper_style": {j: {k: round(v, 4) for k, v in d.items()} for j, d in paper_style.items()},
        "bins": bin_desc.round(4).reset_index().to_dict(orient="records"),
        "n_items": len(items),
        "n_prompts": len(uq),
    }


PRIVATE_TASKS = ("email", "phone")  # generated addresses and numbers may belong to real people


def redact_coverage(run):
    """Replace generations of the email and phone-number tasks with their summary statistics (no raw samples)."""
    for f in (RES / "raw" / run).glob("*/coverage_*.jsonl"):
        rows, changed = [], False
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if any(k in r["task"] for k in PRIVATE_TASKS) and "generations" in r:
                st = M.coverage_stats(r["generations"], r["validity"])
                r = {k: v for k, v in r.items() if k not in ("generations", "validity")} | {
                    "redacted": True,
                    "stats": st,
                }
                changed = True
            rows.append(r)
        if changed:
            f.write_text("".join(json.dumps(r) + "\n" for r in rows))


def spend():
    def total(name):
        f = RES / name
        return round(sum(json.loads(x)["usd"] for x in f.read_text().splitlines() if x.strip()), 2) if f.exists() else 0

    return {"modal_usd_ledger": total("spend_modal.jsonl"), "openrouter_usd": total("spend_openrouter.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full500")
    ap.add_argument("--only", default="ABCDE")
    args = ap.parse_args()
    out_path = RES / "summary.json"
    summary = json.loads(out_path.read_text()) if out_path.exists() else {}
    summary["run"] = args.run
    redact_coverage(args.run)
    if "A" in args.only:
        summary["A"] = analyse_A(args.run)
        print("A", json.dumps(summary["A"]["pooled"]), json.dumps(summary["A"]["families"], indent=0)[:900])
    if "B" in args.only:
        summary["B"] = analyse_B(args.run)
        print("B", json.dumps(summary["B"]["pooled"]), json.dumps(summary["B"]["families"]))
    if "C" in args.only or "D" in args.only:
        cd = analyse_CD(args.run)
        summary["C"], summary["D"] = cd["C"], cd["D"]
        summary["CD_examples"], summary["C_per_prompt"] = cd["examples"], cd["per_prompt_repetition"]
        summary["queries"] = cd["queries"]
        alt = analyse_CD(args.run, drop_empty=False)  # sensitivity: empty replies scored as similarity 0
        summary["D_empty_as_zero"] = {k: v for k, v in alt["D"].items() if k != "pairs"}
        summary["C_empty_as_zero"] = {m: v["median"] for m, v in alt["C"]["per_model"].items()}
        print("C", json.dumps({m: v["median"] for m, v in cd["C"]["per_model"].items()}), cd["C"]["verdict"])
        print("D", json.dumps({k: v for k, v in cd["D"].items() if k != "pairs"}))
    if "E" in args.only:
        summary["E"] = analyse_E(args.run)
        print(
            "E",
            summary["E"]["verdict"],
            summary["E"]["raw_diff"],
            summary["E"]["ceiling_diff_verdict"],
            summary["E"]["ceiling_normalised_diff"],
        )
    summary["spend"] = spend()
    out_path.write_text(json.dumps(summary, indent=1, default=float))
    print("spend", summary["spend"])


if __name__ == "__main__":
    main()
