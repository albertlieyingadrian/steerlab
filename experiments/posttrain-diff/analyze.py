"""Every number in the note, from data/reads32/*.pt and data/acts/*.pt (run.py fetch) -> results/*.json.

    python analyze.py

- summary.json: per pair, medians of every per-unit measure (neurons: poles averaged; heads), by layer too, and the
  share of units past each floor.
- agreement.json: which units each arm moved, compared across arms (Spearman over all neurons, top-1% overlap), and
  the cheap baseline (weight change) against the lens change.
- acts.json: activation correlation and mean shift per unit; AI-register projections per document with bootstrap CIs.
- objective.json: do the most-changed units lean toward AI register (P4), with decile-matched random units.
- predictions.json: P1-P5 scored against the plan's thresholds; a hit needs both the first (fp16) run and this one
  (plan amendment 2).
- precision_check.json: the first run's fp16 reads against the fp32 reads, if data/reads is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
RNG = np.random.default_rng(0)
NB = 2000
CLEAN = ("deslop-neutral", "deslop-randown")
LATE = list(range(18, 36))
F = 12288
READS = "reads32"


def load(kind, name):
    p = DATA / kind / f"{name}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def within_layer(x, y):
    """Median over layers of the within-layer Spearman of two [36, f] arrays, with the 10th and 90th percentiles.
    The pooled Spearman also counts layer depth, which every change score shares."""
    r = [
        spearmanr(a, b).statistic
        for a, b in zip(np.asarray(x), np.asarray(y), strict=True)
        if np.ptp(a) > 0 and np.ptp(b) > 0
    ]
    return [float(np.median(r)), float(np.percentile(r, 10)), float(np.percentile(r, 90))] if r else None


def ci(x, stat=np.mean, nb=NB):
    x = np.asarray(x)
    bs = [stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(nb)]
    return [float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def neuron_change(r):
    """Change score per neuron [36, 12288]: 1 - write-read top-token Spearman, poles averaged (the plan's amended score;
    top-20 overlap ties at 1.0 for most units under a LoRA)."""
    return (1 - r["wR_rho"]).mean(1).numpy()


def flat(x):
    return np.asarray(x).reshape(-1)


def summarise(r):
    out = {"seconds": float(r["meta"]["seconds"])}
    neuron = {"wR_ov": r["wR_ov"].mean(1), "wR_rho": r["wR_rho"].mean(1), "rS_ov": r["rS_ov"].mean(1),
              "rS_rho": r["rS_rho"].mean(1), "w_cos": r["w_cos"], "gate_cos": r["gate_cos"], "up_cos": r["up_cos"],
              "w_rel": r["w_rel"]}  # fmt: skip
    head = {k: r[k] for k in ("ov_cos", "qk_cos", "ovR_ov", "ovR_rho", "ov_rel")}
    for kind, d in (("neuron", neuron), ("head", head)):
        for k, v in d.items():
            v = v.numpy()
            ok = ~np.isnan(v)
            if not ok.any():
                continue
            out[f"{kind}.{k}.median"] = float(np.median(v[ok]))
            out[f"{kind}.{k}.mean"] = float(np.mean(v[ok]))
            out[f"{kind}.{k}.q10"] = float(np.percentile(v[ok], 10))
            out[f"{kind}.{k}.q90"] = float(np.percentile(v[ok], 90))
            out[f"{kind}.{k}.by_layer"] = [float(np.nanmedian(row)) if (~np.isnan(row)).any() else None for row in v]
            out[f"{kind}.{k}.by_layer_mean"] = [float(np.nanmean(row)) if (~np.isnan(row)).any() else None for row in v]
    return out


def floor_share(change, floor_change, q=99):
    thr = np.nanpercentile(flat(floor_change), q)
    return float(np.nanmean(flat(change) > thr)), float(thr)


def act_stats(a, side, groups, key="nsum"):
    """Mean and sd per unit for one side over the listed groups ("10" eval human, "11" eval AI, "00"/"01" fit)."""
    n = sum(a["meta"]["ntok"][g] for g in groups)
    s1 = sum(a[key]["x"][g][side] for g in groups)
    s2 = sum(a[key]["x2"][g][side] for g in groups)
    m = s1 / n
    return m, (s2 / n - m**2).clamp_min(1e-12).sqrt(), n


def act_corr(a, key="nsum", cross="ncross"):
    """Per-unit Pearson over all tokens between the two sides."""
    groups = ["00", "01", "10", "11"]
    m0, s0, n = act_stats(a, 0, groups, key)
    m1, s1, _ = act_stats(a, 1, groups, key)
    return ((a[cross] / n - m0 * m1) / (s0 * s1)).numpy()


def ai_lean(a, side=0):
    """Per-neuron activation d' (AI minus human) on the fit split, and mean |activation| there."""
    m_ai, s_ai, _ = act_stats(a, side, ["01"])
    m_h, s_h, _ = act_stats(a, side, ["00"])
    lean = ((m_ai - m_h) / ((s_ai**2 + s_h**2) / 2).sqrt()).numpy()
    return lean, act_stats(a, side, ["00", "01"])[0].abs().numpy()


def auroc(pos, neg):
    x = np.concatenate([pos, neg])
    rk = rankdata(x)
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


SPLIT = np.array([0] * 500 + [1] * 500)  # corpus order (run.py corpus): fit human, fit AI, eval human, eval AI
LABEL = np.array([0] * 250 + [1] * 250 + [0] * 250 + [1] * 250)


def doc_shift(a, label=1, layers=LATE, key="resid"):
    """Per eval doc of `label`: mean over `layers` of (after - before) projection on an AI direction."""
    sel = (SPLIT == 1) & (label == LABEL)
    p = (a["proj"][key] if key in a["proj"] else a["proj_x"][key]).numpy()
    return (p[1][sel][:, layers] - p[0][sel][:, layers]).mean(1)


def main():
    pairs = sorted(p.stem for p in (DATA / READS).glob("*.pt") if not p.stem.startswith("pilot"))
    R = {n: load(READS, n) for n in pairs}
    summary = {n: summarise(r) for n, r in R.items()}
    ch = {n: neuron_change(r) for n, r in R.items() if n != "selfcheck"}
    hch = {n: (1 - r["ovR_rho"]).numpy() for n, r in R.items() if n != "selfcheck"}
    for n in ch:  # floors: the lens floor for the public pair; for each LoRA, its own family's matched-steps twin
        floor = "pub-lensswap" if n.startswith("pub") else ("slm-shuffled" if n.startswith("slm") else "deslop-randown")
        if floor in ch and n != floor:
            summary[n]["share_past_floor99"], summary[n]["floor99"] = floor_share(ch[n], ch[floor])
            summary[n]["head_share_past_floor99"], _ = floor_share(hch[n], hch[floor])
            summary[n]["floor"] = floor
    if "selfcheck" in R:
        r = R["selfcheck"]
        summary["selfcheck"]["known_answer_a_min"] = {
            k: float(r[k][~torch.isnan(r[k])].min())
            for k in ("wR_ov", "wR_rho", "rS_ov", "rS_rho", "ovR_ov", "ovR_rho")
        }
    if all(k in R for k in ("pub", "pub-baselens", "pub-lensswap")):  # units whose reads survive a lens refit
        ls = flat(R["pub-lensswap"]["wR_ov"].mean(1))
        hls = flat(R["pub-lensswap"]["ovR_ov"])
        stable = {}
        for thr in (0.7, 0.8, 0.9):
            m, hm = ls >= thr, hls >= thr
            entry = {"share": float(m.mean()), "head_share": float(hm.mean())}
            for p in ("pub", "pub-baselens"):
                entry[f"{p}_median_overlap"] = float(np.median(flat(R[p]["wR_ov"].mean(1))[m]))
                entry[f"{p}_head_median_overlap"] = float(np.median(flat(R[p]["ovR_ov"])[hm])) if hm.any() else None
            by_layer = (R["pub-lensswap"]["wR_ov"].mean(1).numpy() >= thr).mean(1)
            entry["share_by_layer"] = [float(x) for x in by_layer]
            stable[str(thr)] = entry
        summary["pub"]["lens_stable"] = stable
        summary["pub"]["final_layer_median_overlap"] = float(np.median(R["pub"]["wR_ov"][35].mean(0).numpy()))
        summary["pub"]["final_layer_median_rho"] = float(np.median(R["pub"]["wR_rho"][35].mean(0).numpy()))
        summary["pub"]["spearman_change_pub_vs_lensswap"] = float(
            spearmanr(flat(ch["pub"]), flat(ch["pub-lensswap"])).statistic
        )
        summary["pub"]["spearman_change_baselens_vs_lensswap"] = float(
            spearmanr(flat(ch["pub-baselens"]), flat(ch["pub-lensswap"])).statistic
        )
    if "pub" in R:  # known answer (b): units whose weights moved more should have reads that moved more
        summary["pub"]["known_answer_b_spearman_wcos_vs_wRov"] = float(
            spearmanr(flat(R["pub"]["w_cos"]), flat(R["pub"]["wR_ov"].mean(1))).statistic
        )
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    if all(k in ch for k in ("pub", "pub-lensswap", "pub-baselens")):  # figure inputs: per-unit change, binned
        edges = np.linspace(0, 1, 41)
        grids = {}
        for a, b in (("pub-lensswap", "pub"), ("pub-lensswap", "pub-baselens")):
            h, _, _ = np.histogram2d(np.clip(flat(ch[a]), 0, 1), np.clip(flat(ch[b]), 0, 1), bins=[edges, edges])
            grids[f"{a}|{b}"] = h.astype(int).tolist()
        order = np.argsort(np.argsort(-ch["pub-lensswap"], axis=1), axis=1)  # neurons sorted by lens-only change
        cover = {k: [[float(x) for x in np.take_along_axis(ch[k], np.argsort(-ch["pub-lensswap"], axis=1), 1)[layer]
                                                   .reshape(96, -1).mean(1)] for layer in range(36)]
                 for k in ("pub-lensswap", "pub-baselens", "pub")}  # fmt: skip
        del order
        (RESULTS / "unit_grids.json").write_text(json.dumps({"edges": edges.tolist(), "hist2d": grids, "cover": cover}))

    prec = {}  # the first run ranked fp16 scores: how much of each change was rounding
    for n in ch:
        old = load("reads", n)
        if old is None:
            continue
        c16 = neuron_change(old)
        prec[n] = {
            "median_change_fp16": float(np.median(c16)),
            "median_change_fp32": float(np.median(ch[n])),
            "mean_change_fp16": float(np.mean(c16)),
            "mean_change_fp32": float(np.mean(ch[n])),
            "spearman_fp16_vs_fp32": float(spearmanr(flat(c16), flat(ch[n])).statistic),
        }
    if prec:
        (RESULTS / "precision_check.json").write_text(json.dumps(prec, indent=1))

    # which neurons each arm moved, compared across arms
    lora = [n for n in ch if not n.startswith("pub")]
    agree = {"spearman": {}, "top1_overlap": {}, "w_rel_spearman": {}, "cheap_vs_lens": {}, "top1_chance": 0.01}
    k1 = int(0.01 * F * 36)
    tie = RNG.random(F * 36) * 1e-9
    tops = {n: set(np.argsort(-(flat(ch[n]) + tie))[:k1].tolist()) for n in ch}
    names = list(ch)
    agree["spearman_within_layer"], agree["top1_layer_hist"] = {}, {}
    for n in names:
        agree["top1_layer_hist"][n] = np.bincount(np.array(sorted(tops[n])) // F, minlength=36).tolist()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}|{b}"
            agree["spearman"][key] = float(spearmanr(flat(ch[a]), flat(ch[b])).statistic)
            agree["w_rel_spearman"][key] = float(spearmanr(flat(R[a]["w_rel"]), flat(R[b]["w_rel"])).statistic)
            agree["top1_overlap"][key] = len(tops[a] & tops[b]) / k1
            agree["spearman_within_layer"][key] = within_layer(ch[a], ch[b])
    for n in ch:
        agree["cheap_vs_lens"][n] = {
            "spearman_change_vs_w_rel": float(spearmanr(flat(ch[n]), flat(R[n]["w_rel"])).statistic),
            "spearman_change_vs_1-w_cos": float(spearmanr(flat(ch[n]), 1 - flat(R[n]["w_cos"])).statistic),
            "within_layer_change_vs_w_rel": within_layer(ch[n], R[n]["w_rel"].numpy()),
            "top1_overlap_with_top_w_rel": len(tops[n] & set(np.argsort(-flat(R[n]["w_rel"]))[:k1].tolist())) / k1,
        }
    (RESULTS / "agreement.json").write_text(json.dumps(agree, indent=1))

    # activations
    A = {p.stem: load("acts", p.stem) for p in (DATA / "acts").glob("*.pt")}
    acts = {}
    for n, a in sorted(A.items()):
        corr = act_corr(a)
        hcorr = act_corr(a, "hsum", "hcross")
        m0, s0, _ = act_stats(a, 0, ["10", "11"])
        m1, _, _ = act_stats(a, 1, ["10", "11"])
        shift = ((m1 - m0) / s0).numpy()
        p18 = a["proj"]["resid"][0][:, 18].numpy()
        e = {"neuron_corr_median": float(np.nanmedian(corr)), "neuron_corr_q01": float(np.nanpercentile(corr, 1)),
             "head_corr_median": float(np.nanmedian(hcorr)),
             "neuron_shift_abs_median": float(np.nanmedian(np.abs(shift))),
             "corr_by_layer": [float(np.nanmedian(c)) for c in corr],
             "head_corr_by_layer": [float(np.nanmedian(c)) for c in hcorr],
             "ai_dir_auroc_L18_eval_before": auroc(p18[(SPLIT == 1) & (LABEL == 1)], p18[(SPLIT == 1) & (LABEL == 0)]),
             "shift_resid_ai_by_layer": [float(doc_shift(a, 1, [layer]).mean()) for layer in range(36)],
             "shift_resid_human_by_layer": [float(doc_shift(a, 0, [layer]).mean()) for layer in range(36)]}  # fmt: skip
        for key in ("resid", "mlp", "attn", "register", "probe18"):
            layers = LATE if key in ("resid", "mlp", "attn") else ([12, 18, 24] if key == "register" else [18])
            for label, tag in ((1, "ai"), (0, "human")):
                e[f"shift_{key}_{tag}"] = ci(doc_shift(a, label, layers, key))
            # the scale: how far apart AI and human eval texts sit on this ruler, before training
            p = (a["proj"][key] if key in a["proj"] else a["proj_x"][key]).numpy()[0][:, layers].mean(1)
            e[f"gap_{key}_before"] = float(
                p[(SPLIT == 1) & (LABEL == 1)].mean() - p[(SPLIT == 1) & (LABEL == 0)].mean()
            )
            # difference in differences: did the AI texts move more than the human texts? (a uniform shift cancels)
            dai, dhu = doc_shift(a, 1, layers, key), doc_shift(a, 0, layers, key)
            boot = [dai[RNG.integers(0, len(dai), len(dai))].mean() - dhu[RNG.integers(0, len(dhu), len(dhu))].mean()
                    for _ in range(NB)]  # fmt: skip
            e[f"did_{key}"] = [
                float(dai.mean() - dhu.mean()),
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ]
        acts[n] = e
    if all(c in A for c in CLEAN):  # P3: each arm minus the mean of the clean twins, paired over the same docs
        for label, tag in ((1, "ai"), (0, "human")):
            for key, layers in (("resid", LATE), ("register", [12, 18, 24]), ("probe18", [18])):
                clean = np.mean([doc_shift(A[c], label, layers, key) for c in CLEAN], axis=0)
                for n, a in A.items():
                    if not n.startswith("pub"):
                        acts[n][f"vs_clean_{key}_{tag}"] = ci(doc_shift(a, label, layers, key) - clean)
    (RESULTS / "acts.json").write_text(json.dumps(acts, indent=1))

    # P4: AI-lean of the neurons each arm moved most beyond the clean twins
    obj = {}
    ref = A.get("deslop-sft")
    if ref is not None and all(c in ch for c in CLEAN):
        lean, mag = ai_lean(ref)
        lean_f, lean_abs, mag_f = flat(lean), np.abs(flat(lean)), flat(mag)
        deciles = np.digitize(mag_f, np.percentile(mag_f, np.arange(10, 100, 10)))
        pv = RESULTS / "probe_units.json"
        probe64 = json.loads(pv.read_text())["neurons_L18"] if pv.exists() else None
        for arm in [n for n in lora if n not in CLEAN]:
            spec = flat(ch[arm]) - np.mean([flat(ch[c]) for c in CLEAN], axis=0)
            top = np.argsort(-(spec + tie))[:k1]
            base = lean_abs.mean()
            boot = [lean_abs[RNG.choice(top, k1)].mean() / base for _ in range(NB)]
            counts = np.bincount(deciles[top], minlength=10)
            pools = [np.where(deciles == dcl)[0] for dcl in range(10)]
            matched = [lean_abs[np.concatenate([RNG.choice(pools[dcl], c) for dcl, c in enumerate(counts) if c])].mean()
                       for _ in range(200)]  # fmt: skip
            randtop = [lean_abs[RNG.choice(lean_abs.size, k1, replace=False)].mean() / base for _ in range(200)]
            obj[arm] = {"ratio_vs_all": [float(lean_abs[top].mean() / base), float(np.percentile(boot, 2.5)),
                                         float(np.percentile(boot, 97.5))],
                        "random_sets_ratio_q025_q975": [float(np.percentile(randtop, 2.5)),
                                                        float(np.percentile(randtop, 97.5))],
                        "ratio_vs_decile_matched": float(lean_abs[top].mean() / np.mean(matched)),
                        "signed_lean_mean_top": float(lean_f[top].mean()), "signed_lean_mean_all": float(lean_f.mean()),
                        "share_ai_leaning_top": float((lean_f[top] > 0).mean()),
                        "layer_hist_top": np.bincount(top // F, minlength=36).tolist(), "k": k1}  # fmt: skip
            if probe64 is not None:
                pct = rankdata(spec)[18 * F + np.array(probe64)] / spec.size
                obj[arm]["probes_vs_interp_64_mean_percentile"] = ci(pct)
    (RESULTS / "objective.json").write_text(json.dumps(obj, indent=1))

    # amendment 3: each SFT seed against clean twins trained from the same random init
    sfx = ("", "-s1", "-s2")
    need = [f"{b}{x}" for b in ("deslop-sft", *CLEAN) for x in sfx]
    if all(n in ch and n in A for n in need):

        def pair(a, b):
            key = f"{a}|{b}" if f"{a}|{b}" in agree["spearman"] else f"{b}|{a}"
            return {k: agree[k][key] for k in ("spearman", "spearman_within_layer", "top1_overlap")}

        kinds = {"same_objective_other_init": [(f"deslop-sft{a}", f"deslop-sft{b}") for a in sfx for b in sfx if a < b],
                 "other_objective_same_init": [(f"deslop-sft{a}", f"{c}{a}") for a in sfx for c in CLEAN],
                 "other_objective_other_init": [(f"deslop-sft{a}", f"{c}{b}") for a in sfx for b in sfx if a != b
                                                for c in CLEAN]}  # fmt: skip
        matched = {"agreement": {}, "seeds": {}}
        for kind, ps in kinds.items():
            vals = [pair(*p) for p in ps]
            wl = [v["spearman_within_layer"][0] for v in vals]
            t1 = [v["top1_overlap"] for v in vals]
            pooled = [v["spearman"] for v in vals]
            matched["agreement"][kind] = {
                name: [float(np.mean(v)), float(min(v)), float(max(v))]
                for name, v in (("within_layer", wl), ("top1_overlap", t1), ("pooled", pooled))
            }
        lean = np.abs(flat(ai_lean(A["deslop-sft"])[0]))
        for x in sfx:
            twins = [f"{c}{x}" for c in CLEAN]
            res = {}
            for key, layers in (("resid", LATE), ("register", [12, 18, 24]), ("probe18", [18])):
                clean = np.mean([doc_shift(A[t], 1, layers, key) for t in twins], axis=0)
                res[f"{key}_ai_vs_matched"] = ci(doc_shift(A[f"deslop-sft{x}"], 1, layers, key) - clean)
                res[f"{key}_ai_twins_raw"] = [float(doc_shift(A[t], 1, layers, key).mean()) for t in twins]
            spec = flat(ch[f"deslop-sft{x}"]) - np.mean([flat(ch[t]) for t in twins], axis=0)
            top = np.argsort(-(spec + tie))[:k1]
            res["lean_ratio_vs_all"] = ci(lean[top] / lean.mean())
            matched["seeds"][x.lstrip("-") or "s0"] = res
        (RESULTS / "matched_init.json").write_text(json.dumps(matched, indent=1))

    pred = {}
    s = summary
    if "pub" in s and "pub-lensswap" in s:
        gap = s["pub-lensswap"]["neuron.wR_ov.median"] - s["pub"]["neuron.wR_ov.median"]
        pred["P1"] = {"value": gap, "pub": s["pub"]["neuron.wR_ov.median"],
                      "lensswap": s["pub-lensswap"]["neuron.wR_ov.median"], "hit": gap >= 0.10}  # fmt: skip
    sp = [agree["spearman"].get(f"deslop-sft|deslop-sft-s{i}") for i in (1, 2)]
    if all(x is not None for x in sp):
        rnd = agree["spearman"].get("deslop-randown|deslop-sft", agree["spearman"].get("deslop-sft|deslop-randown"))
        pred["P2"] = {"value": sp, "unrelated": rnd, "hit": all(x >= 0.30 and (rnd is None or x > rnd) for x in sp)}
    v = {
        n: acts[n]["vs_clean_resid_ai"] for n in ("deslop-sft", "deslop-dpo") if "vs_clean_resid_ai" in acts.get(n, {})
    }
    if len(v) == 2:
        pred["P3"] = {"value": v, "hit": all(x[0] < 0 and x[2] < 0 for x in v.values())}
    if "deslop-sft" in obj:
        v4 = obj["deslop-sft"]["ratio_vs_all"]
        pred["P4"] = {"value": v4, "hit": v4[0] >= 1.3 and v4[1] > 1}
    if "slm-grpo200-vs-sft" in ch and "slm-sft" in ch:
        a, b = float(np.median(ch["slm-grpo200-vs-sft"])), float(np.median(ch["slm-sft"]))
        pa, pb = float(np.mean(ch["slm-grpo200-vs-sft"])), float(np.mean(ch["slm-sft"]))
        pred["P5"] = {"value": {"grpo_vs_sft_median": a, "sft_median": b, "grpo_vs_sft_mean": pa, "sft_mean": pb},
                      "hit": a < b}  # fmt: skip
    first = json.loads((RESULTS / "predictions_fp16.json").read_text())
    for k, p in pred.items():
        p["hit_fp32"], p["hit_fp16"] = p["hit"], first[k]["hit"]
        p["hit"] = p["hit_fp32"] and p["hit_fp16"]
    (RESULTS / "predictions.json").write_text(json.dumps(pred, indent=1))
    print(json.dumps(pred, indent=1))


if __name__ == "__main__":
    main()
