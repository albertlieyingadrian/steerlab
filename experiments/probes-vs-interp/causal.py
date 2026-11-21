"""M4, test 4 (presence vs use): does pushing against what a detector reads make Qwen3-8B write less like AI?

    python causal.py units     # directions, SAE features and neurons from the fit set -> data/causal_units.npz
    python causal.py tune      # dev summaries: strength grid per method, NLL rise <= 0.3 -> results/causal_tune.json
    python causal.py run       # held-out summaries: chosen strengths, 5 random directions, prompt baseline
    python causal.py score     # voi-380M P(AI) per output -> results/causal.json

Every intervention sits at layer 18 (the Qwen-Scope SAE's layer):
  probe     add -c * unit(w), w = the layer-18 activation probe's weights (AI direction), in raw residual units
  sae_abl   subtract c x the decoded part of the 16 most AI-leaning SAE features (c = 1 removes it)
  sae_dir   add -c * unit(sum of those 16 features' unit decoder directions)
  neuron    move the 64 most AI-leaning MLP neurons to (c = 1) and past (c > 1) their human mean
  random    5 random directions at the probe's chosen c (norm-matched)
  prompt    no intervention; the prompt asks for a plain human email and shows three human examples (cheap baseline)
For the added directions c is in percent of the mean norm of a pooled layer-18 residual. Outcome: voi-380M P(AI) of
each output (lower = more human-like) and its drop from the untouched model on the same prompts. Fluency: mean NLL of
the output under the untouched model; each method's strength is chosen on dev as the largest drop with NLL rise <= 0.3.
"""

from __future__ import annotations

import json
import sys

import budget
import numpy as np
from build_data import ai_prompt
from common import (
    DATA,
    SEED,
    boot_diff,
    boot_mean,
    ledger,
    load_json,
    read_jsonl,
    save_json,
)
from probes import Data, lr, sae_dense, top_by_diff, voi380

LAYER = 18
GRID = {
    "probe": [2.0, 4.0, 8.0, 16.0],
    "sae_dir": [2.0, 4.0, 8.0, 16.0],
    "sae_abl": [1.0, 2.0, 4.0],
    "neuron": [1.0, 2.0, 4.0],
}
N_DEV, N_TEST, N_RANDOM = 40, 60, 5
MAX_NLL_RISE = 0.3
BASE = "none|0|None"


def fn(name):
    import modal

    return modal.Function.from_name("tl-ep1-core", name)


def units():
    D = Data()
    Z = np.load(DATA / "acts_qwen.npz")
    fit = D.idx("fit")
    y = D.y[fit]
    R = Z[f"res{LAYER}"].astype(np.float32)
    m = lr(R[fit], y, 0.1)
    w = m[1].coef_[0] / m[0].scale_
    S = sae_dense(Z["sae_idx"], Z["sae_val"]).tocsc()
    cand = top_by_diff(S[fit], y, 256)
    Sf = S[fit][:, cand].toarray()
    feats = cand[np.argsort(-(Sf[y == 1].mean(0) - Sf[y == 0].mean(0)))[:16]]  # the AI-leaning end
    N = Z["neuron"].astype(np.float32)[fit]
    ncand = top_by_diff(N, y, 512)
    neurons = ncand[np.argsort(-(N[y == 1][:, ncand].mean(0) - N[y == 0][:, ncand].mean(0)))[:64]]
    np.savez(
        DATA / "causal_units.npz",
        w=w.astype(np.float32),
        feats=feats,
        neurons=neurons,
        human_mean=N[y == 0][:, neurons].mean(0),
        norm_ref=float(np.linalg.norm(R[fit], axis=1).mean()),
    )
    print("units: 16 SAE features", feats.tolist(), "64 neurons")


def sae_dirs(feats):
    import torch
    from huggingface_hub import hf_hub_download

    st = torch.load(
        hf_hub_download("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", "layer18.sae.pt"),
        map_location="cpu",
        weights_only=True,
    )
    W = st["W_dec"].float().numpy()[:, feats]
    return (W / np.linalg.norm(W, axis=0, keepdims=True)).sum(1)


def prompts(split, n):
    H = [h for h in read_jsonl(DATA / "humans.jsonl") if h["set"] == split and h["corpus"] == "enron"]
    rng = np.random.default_rng(SEED)
    return [ai_prompt(H[i]) for i in rng.choice(len(H), n, replace=False)]


def human_prompt(msgs, examples):
    ex = "\n\n---\n\n".join(examples)
    return [
        {
            "role": "user",
            "content": "Here are three emails real people wrote quickly at work:\n\n"
            + ex
            + "\n\n---\n\nWrite like them: plain, unpolished, the way a busy person actually types.\n\n"
            + msgs[0]["content"],
        }
    ]


def jobs_for(P, method, c, U, vdir, rand=None):
    out = []
    for k, msgs in enumerate(P):
        j = dict(
            messages=msgs,
            group=f"{method}|{c}|{rand}",
            layer=LAYER,
            kind="none",
            coef=0.0,
            pid=k,
        )
        if method in ("probe", "sae_dir", "random"):
            j.update(kind="add", coef=c * U["norm_ref"] / 100.0, vector=vdir.tolist())
        elif method == "sae_abl":
            j.update(kind="sae", coef=c, features=U["feats"].tolist())
        elif method == "neuron":
            j.update(
                kind="neuron",
                coef=c,
                neurons=U["neurons"].tolist(),
                means=U["human_mean"].tolist(),
            )
        out.append(j)
    return out


def load_units():
    z = np.load(DATA / "causal_units.npz")
    U = {k: z[k] for k in z.files}
    U["norm_ref"] = float(U["norm_ref"])
    return U, {"probe": -U["w"], "sae_dir": -sae_dirs(U["feats"])}


def _steer(J, job, projected):
    r = budget.paid(job, "H100", projected, fn("steer").remote, J)
    return r["outputs"], r["nll"]


def _nll(xs):
    return np.array([np.nan if x is None else x for x in xs], float)


def tune():
    U, V = load_units()
    P = prompts("dev", N_DEV)
    J = jobs_for(P, "none", 0, U, None)
    for m, cs in GRID.items():
        for c in cs:
            J += jobs_for(P, m, c, U, V.get(m))
    outs, nll = _steer(J, "causal-tune", 4.0)
    groups = {}
    for j, o, n in zip(J, outs, nll, strict=False):
        groups.setdefault(j["group"], ([], []))
        groups[j["group"]][0].append(o)
        groups[j["group"]][1].append(n)
    p_all = {g: voi380([t or "." for t in o]) for g, (o, _) in groups.items()}
    base_p, base_n = p_all[BASE], _nll(groups[BASE][1])
    res = {}
    for g, (_o, n) in groups.items():
        res[g] = dict(
            p_ai=float(p_all[g].mean()),
            drop=float(base_p.mean() - p_all[g].mean()),
            nll_rise=float(np.nanmean(_nll(n)) - np.nanmean(base_n)),
        )
        ledger(
            f"causal-dev-{g}",
            "p_ai_drop",
            res[g]["drop"],
            0.0,
            "grid",
            f"nll rise {res[g]['nll_rise']:.3f}",
        )
    choice = {}
    for m in GRID:
        mine = [(g, v) for g, v in res.items() if g.startswith(m + "|")]
        ok = [x for x in mine if x[1]["nll_rise"] <= MAX_NLL_RISE]
        g, v = max(ok, key=lambda x: x[1]["drop"]) if ok else min(mine, key=lambda x: x[1]["nll_rise"])
        choice[m] = dict(c=float(g.split("|")[1]), **v, fluent=bool(v["nll_rise"] <= MAX_NLL_RISE))
        ledger(
            f"causal-choice-{m}",
            "p_ai_drop",
            v["drop"],
            0.0,
            "kept",
            f"c={g.split('|')[1]}",
        )
    save_json(
        "causal_tune.json",
        dict(grid=res, choice=choice, base_p_ai=float(base_p.mean())),
    )
    (DATA / "causal_dev_outputs.json").write_text(json.dumps(groups))
    print(json.dumps(choice, indent=1))


def run():
    U, V = load_units()
    ch = load_json("causal_tune.json")["choice"]
    P = prompts("test", N_TEST)
    J = jobs_for(P, "none", 0, U, None)
    for m in GRID:
        J += jobs_for(P, m, ch[m]["c"], U, V.get(m))
    rng = np.random.default_rng(SEED)
    for r in range(N_RANDOM):
        J += jobs_for(
            P,
            "random",
            ch["probe"]["c"],
            U,
            rng.standard_normal(U["w"].shape).astype(np.float32),
            rand=r,
        )
    H = [h["text"] for h in read_jsonl(DATA / "humans.jsonl") if h["set"] == "fit" and h["corpus"] == "enron"][:3]
    J += [
        dict(j, group="prompt|0|None", messages=human_prompt(j["messages"], H)) for j in jobs_for(P, "none", 0, U, None)
    ]
    outs, nll = _steer(J, "causal-heldout", 4.0)
    rows = [dict(group=j["group"], pid=j["pid"], text=o, nll=n) for j, o, n in zip(J, outs, nll, strict=False)]
    (DATA / "causal_outputs.json").write_text(json.dumps(rows))
    print("causal outputs", len(rows))


def loops(t):
    w = t.split()
    grams = [" ".join(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / max(len(grams), 1)


def score():
    rows = json.loads((DATA / "causal_outputs.json").read_text())
    p = voi380([r["text"] or "." for r in rows])
    by = {}
    for r, s in zip(rows, p, strict=False):
        by.setdefault(r["group"], {})[r["pid"]] = (
            s,
            r["nll"],
            len(r["text"].split()),
            loops(r["text"]),
            r["text"],
        )
    pids = sorted(by[BASE])

    def arr(g, k):
        return np.array([np.nan if by[g][i][k] is None else by[g][i][k] for i in pids], float)

    out = {}
    for g in by:
        out[g] = dict(
            p_ai=boot_mean(arr(g, 0)),
            drop=boot_diff(arr(BASE, 0), arr(g, 0)),
            nll_rise=round(float(np.nanmean(arr(g, 1)) - np.nanmean(arr(BASE, 1))), 4),
            words=round(float(arr(g, 2).mean()), 1),
            repeat_4gram=round(float(arr(g, 3).mean()), 4),
            looping_share=round(float((arr(g, 3) > 0.2).mean()), 3),
            empty_share=round(float((arr(g, 2) < 5).mean()), 3),  # empty or near-empty outputs (failures)
            example=by[g][pids[0]][4][:700],
        )
    rand = [g for g in by if g.startswith("random|")]
    best_rand = max(rand, key=lambda g: out[g]["drop"]["mean"])
    probe = next(g for g in by if g.startswith("probe|"))
    sae_best = max([g for g in by if g.startswith("sae_")], key=lambda g: out[g]["drop"]["mean"])
    probe_vs_rand = boot_diff(arr(best_rand, 0), arr(probe, 0))  # > 0: the probe lowers P(AI) more than random
    sae_vs_probe = boot_diff(arr(probe, 0), arr(sae_best, 0))
    p5 = probe_vs_rand["lo"] > 0 and sae_vs_probe["lo"] > 0
    save_json(
        "causal.json",
        dict(
            groups=out,
            best_random=best_rand,
            probe=probe,
            best_sae=sae_best,
            probe_minus_best_random=probe_vs_rand,
            sae_minus_probe=sae_vs_probe,
            p5_hit=bool(p5),
            n_prompts=len(pids),
            random_drops=[out[g]["drop"]["mean"] for g in rand],
        ),
    )
    for g, v in sorted(out.items()):
        print(
            f"{g:22s} P(AI) {v['p_ai']['mean']:.3f} drop {v['drop']['mean']:+.3f} [{v['drop']['lo']:+.3f},"
            f"{v['drop']['hi']:+.3f}] nll+ {v['nll_rise']:+.3f} words {v['words']} rep {v['repeat_4gram']} "
            f"looping {v['looping_share']} empty {v['empty_share']}"
        )
    print("probe - best random", probe_vs_rand, "sae - probe", sae_vs_probe, "P5", p5)


if __name__ == "__main__":
    dict(units=units, tune=tune, run=run, score=score)[sys.argv[1]]()
