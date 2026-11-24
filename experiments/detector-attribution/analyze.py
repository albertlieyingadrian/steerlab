"""Every number in the note, from results/ -> results/summary.json (and the predictions, scored).

    python analyze.py

Sections: corpus (each detector's AUROC on the cut texts), words (occlusion word lists, cross-detector agreement,
IG completeness), deletion (own top words vs random, transfer between open detectors), sae (FVU, ablation vs
matched random), pangram (repeat noise, deletion, insertion, splices and windows), gen (the writer's SAE: AUROC vs
random features and a bag of words; overlap with the 9B's words; activation inside Pangram's windows), bow (a bag of
words on the 711 texts Pangram scored in deslop-h2h), predictions.
"""

import json
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import (
    DATA,
    DETECTORS,
    H2H,
    RESULTS,
    SEED,
    _human_path,
    boot_mean,
    corpus,
    load_json,
    read_jsonl,
    save_json,
    word_type,
    words,
)

# The plan's fixed AI-tell list (arm 5), with inflections.
LEXICON = {
    "—",
    "moreover",
    "additionally",
    "furthermore",
    "delve",
    "delves",
    "delving",
    "crucial",
    "pivotal",
    "tapestry",
    "testament",
    "showcase",
    "showcases",
    "showcasing",
    "foster",
    "fosters",
    "fostering",
    "seamless",
    "seamlessly",
    "intricate",
    "multifaceted",
}
MIN_COUNT = 10
CLIP = 1e-30


def ci(x) -> list[float]:
    m, lo, hi = boot_mean(x)
    return [round(m, 4), round(lo, 4), round(hi, 4)]


def is_punct(t: str) -> bool:
    return not t[0].isalnum()


# ---------------------------------------------------------------------------------------------------------- corpus
def sec_corpus(C: list[dict]) -> dict:
    S = load_json("scores/corpus.json")
    out = {}
    for d in DETECTORS:
        ai = [S[d][r["key"]] for r in C if r["kind"] == "ai"]
        hu = [S[d][r["key"]] for r in C if r["kind"] == "human"]
        out[d] = {
            "auroc": round(roc_auc_score([1] * len(ai) + [0] * len(hu), ai + hu), 4),
            "ai_flagged": [sum(v >= 0.5 for v in ai), len(ai)],
            "human_flagged": [sum(v >= 0.5 for v in hu), len(hu)],
        }
    return out


# ----------------------------------------------------------------------------------------------------------- words
def type_means(
    C: list[dict], A: dict, kind: str = "ai", split: str = "discovery", min_prompts: int = 0
) -> dict[str, tuple[float, int]]:
    acc: dict[str, list[float]] = defaultdict(list)
    prompts: dict[str, set] = defaultdict(set)
    for r in C:
        if r["kind"] != kind or r["split"] != split:
            continue
        for sp, v in zip(words(r["text"]), A[r["key"]]["w"], strict=True):
            acc[word_type(r["text"], sp)].append(v)
            prompts[word_type(r["text"], sp)].add(r["prompt"])
    return {t: (float(np.mean(v)), len(v)) for t, v in acc.items() if len(prompts[t]) >= min_prompts}


def sec_words(C: list[dict]) -> dict:
    A = {d: load_json(f"attr/{d}.json") for d in DETECTORS}
    out: dict = {"lists": {}, "agreement": {}, "ig": {}, "punct_share": {}}
    tm = {d: type_means(C, A[d]) for d in DETECTORS}
    for d in DETECTORS:
        ok = {t: v for t, v in tm[d].items() if v[1] >= MIN_COUNT}
        ranked = sorted(ok, key=lambda t: -ok[t][0])
        out["lists"][d] = {
            "top_ai": [[t, round(ok[t][0], 4), ok[t][1]] for t in ranked[:30]],
            "top_human": [[t, round(ok[t][0], 4), ok[t][1]] for t in ranked[::-1][:30]],
            "n_types": len(ok),
            "top30_punct_or_lexicon": sum(is_punct(t) or t in LEXICON for t in ranked[:30]),
            "lexicon_in_list": {t: [round(ok[t][0], 4), ok[t][1], ranked.index(t) + 1] for t in ranked if t in LEXICON},
        }
    # exploratory (not in the plan): types used in at least 8 of the 24 discovery prompts, so topic words drop out
    for d in DETECTORS:
        spread = {t: v for t, v in type_means(C, A[d], min_prompts=8).items() if v[1] >= MIN_COUNT}
        ranked = sorted(spread, key=lambda t: -spread[t][0])
        out["lists"][d]["top_ai_8prompts"] = [[t, round(spread[t][0], 4), spread[t][1]] for t in ranked[:30]]
        out["lists"][d]["top_human_8prompts"] = [[t, round(spread[t][0], 4), spread[t][1]] for t in ranked[::-1][:20]]
        out["lists"][d]["n_types_8prompts"] = len(spread)
    for i, a in enumerate(DETECTORS):
        for b in DETECTORS[i + 1 :]:
            shared = [t for t in tm[a] if t in tm[b] and tm[a][t][1] >= MIN_COUNT and tm[b][t][1] >= MIN_COUNT]
            rho = spearmanr([tm[a][t][0] for t in shared], [tm[b][t][0] for t in shared]).statistic
            ta = set(sorted(shared, key=lambda t: -tm[a][t][0])[:30])
            tb = set(sorted(shared, key=lambda t: -tm[b][t][0])[:30])
            out["agreement"][f"{a}|{b}"] = {
                "spearman": round(float(rho), 4),
                "n_types": len(shared),
                "top30_overlap": len(ta & tb),
            }
    rs = [r for r in C if r["kind"] == "ai" and r["split"] == "confirm"]
    punct_words = np.mean([np.mean([is_punct(r["text"][s:e]) for s, e in words(r["text"])]) for r in rs])
    for d in DETECTORS:
        shares = []
        for r in rs:
            pos = np.clip(np.asarray(A[d][r["key"]]["w"]), 0, None)
            p = np.array([is_punct(r["text"][s:e]) for s, e in words(r["text"])])
            shares.append(pos[p].sum() / max(pos.sum(), 1e-9))
        out["punct_share"][d] = {"positive_attr_on_punct": ci(shares), "punct_word_share": round(float(punct_words), 4)}
    for d in DETECTORS:  # IG (secondary): completeness and agreement with occlusion, per text
        p = RESULTS / "attr" / f"ig_{d}.json"
        if not p.exists():
            continue
        G = json.loads(p.read_text())
        rel, rho = [], []
        for k, g in G.items():
            gap = g["fx"] - g["f0"]
            rel.append(abs(g["sum"] - gap) / max(abs(gap), 1e-6))
            if k in A[d]:
                rho.append(spearmanr(g["w"], A[d][k]["w"]).statistic)
        out["ig"][d] = {
            "n": len(G),
            "median_rel_completeness_error": round(float(np.median(rel)), 4),
            "share_within_10pct": round(float(np.mean(np.asarray(rel) < 0.10)), 4),
            "spearman_with_occlusion_median": round(float(np.nanmedian(rho)), 4),
        }
    return out


# -------------------------------------------------------------------------------------------------------- deletion
def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _dose(b: np.ndarray, top: np.ndarray, rnd: np.ndarray) -> dict:
    """b, top: (texts,) logits; rnd: (texts, draws) logits. Score drops (registered units) and logit drops."""
    return {
        "drop": ci(sig(b) - sig(top)),
        "minus_random": ci(sig(rnd).mean(1) - sig(top)),
        "logit_drop": ci(b - top),
        "logit_minus_random": ci(rnd.mean(1) - top),
        "unflagged": int((top < 0).sum()),
        "beats_all_random": int((top < rnd.min(1)).sum()),
    }


def sec_deletion() -> dict:
    """Deletion variants are stored as logit(score); the registered readout is the score, sigmoid of it."""
    D = load_json("deletion.json")
    out: dict = {}
    for d in DETECTORS:
        base = load_json(f"attr/{d}.json")
        rows = D[d]
        keys = sorted(rows)
        b = np.array([base[k]["base"] for k in keys])
        res: dict = {"n": len(keys), "orig_score": ci(sig(b)), "orig_logit": ci(b), "orig_flagged": int((b >= 0).sum())}
        for dose in ("10", "25"):
            rnd = np.array([[rows[k][f"rand{dose}:{j}"] for j in range(20)] for k in keys])
            res[f"rand{dose}"] = {
                "drop": ci(sig(b) - sig(rnd).mean(1)),
                "logit_drop": ci(b - rnd.mean(1)),
                "unflagged": round(float((rnd < 0).mean(1).sum()), 1),
            }
            for g in DETECTORS:
                res[f"top{dose}:{g}"] = _dose(b, np.array([rows[k][f"top{dose}:{g}"] for k in keys]), rnd)
        rs = np.array([[rows[k][f"randsent25:{j}"] for j in range(20)] for k in keys])
        ts = np.array([rows[k]["topsent25"] for k in keys])
        res["sent25"] = _dose(b, ts, rs) | {
            "rand_drop": ci(sig(b) - sig(rs).mean(1)),
            "rand_logit_drop": ci(b - rs.mean(1)),
            "rand_unflagged": round(float((rs < 0).mean(1).sum()), 1),
        }
        out[d] = res
    return out


# ------------------------------------------------------------------------------------------------------------- sae
def sec_sae() -> dict:
    """Ablation scores are stored as logit(score)."""
    Dd = load_json("sae/discovery.json")
    Ab = load_json("sae/ablate.json")
    out: dict = {
        "layer": Dd["layer"],
        "fvu": {
            k: {
                "median": round(float(np.median([a for a, _ in v])), 4),
                "median_no_first": round(float(np.median([b for _, b in v])), 4),
                "n": len(v),
            }
            for k, v in Dd["fvu"].items()
        },
    }
    keys = sorted(Ab["scores"])
    none = np.array([Ab["scores"][k]["none"] for k in keys])
    out["none_logit"] = ci(none)
    for k in (10, 50):
        top = np.array([Ab["scores"][kk][f"top{k}"] for kk in keys])
        rnd = np.array([[Ab["scores"][kk][f"rand{k}:{j}"] for j in range(20)] for kk in keys])
        rand_drops = (sig(none)[:, None] - sig(rnd)).mean(0)
        rand_logit = (none[:, None] - rnd).mean(0)
        out[f"top{k}"] = {
            "drop": ci(sig(none) - sig(top)),
            "random_set_drops": [round(float(v), 5) for v in rand_drops],
            "random_max": round(float(rand_drops.max()), 5),
            "logit_drop": ci(none - top),
            "random_set_logit_drops": [round(float(v), 4) for v in rand_logit],
            "random_logit_max": round(float(rand_logit.max()), 4),
            "unflagged": int((top < 0).sum()),
            "n": len(keys),
        }
    out["labels"] = load_json("sae/labels.json") if (RESULTS / "sae" / "labels.json").exists() else []
    return out


# --------------------------------------------------------------------------------------------------------- pangram
def margin(res: dict) -> tuple[float, float]:
    """Word-weighted mean over windows of log p(AI) - log p(human), probabilities clipped at 1e-30; share capped."""
    ws = res["windows"]
    if not ws:
        return float("nan"), float("nan")
    m, capped, tot = 0.0, 0, 0
    for w in ws:
        ph, pai = max(w["probs"][0], CLIP), max(w["probs"][-1], CLIP)
        capped += int(w["probs"][0] < CLIP or w["probs"][-1] < CLIP) * w["words"]
        m += w["words"] * (np.log(pai) - np.log(ph))
        tot += w["words"]
    return m / tot, capped / tot


def word_window(text: str, wins: list[dict]) -> np.ndarray:
    """For each word unit, 1 if it sits in a window Pangram did not label human, else 0 (by its first character)."""
    lab = np.zeros(len(words(text)))
    for j, (s, _) in enumerate(words(text)):
        for w in wins:
            if w["start"] <= s < w["end"]:
                lab[j] = float(w["label"] != "Human Written")
    return lab


def sec_pangram() -> dict:
    P = load_json("pangram.json")
    items = {it["key"]: it for it in read_jsonl(DATA / "pangram_items.jsonl")}
    SL = load_json("scores/items.json")  # logit(score)
    S = {d: {k: float(sig(v)) for k, v in SL[d].items()} for d in SL}
    out: dict = {"n_items": len(P), "versions": dict(Counter(v["version"] for v in P.values()))}
    reps = sorted(k for k in P if k.startswith("rep/"))
    pairs = [(P[k], P[f"del/{items[k]['src']}/orig"]) for k in reps]
    out["repeat"] = {
        "n": len(pairs),
        "label_changes": sum(a["prediction_short"] != b["prediction_short"] for a, b in pairs),
        "max_abs_margin_diff": round(max(abs(margin(a)[0] - margin(b)[0]) for a, b in pairs), 4),
        "max_abs_fraction_ai_diff": round(max(abs(a["fraction_ai"] - b["fraction_ai"]) for a, b in pairs), 6),
    }
    dels: dict = defaultdict(dict)
    for k, it in items.items():
        if it["group"] == "deletion":
            dels[it["src"]][it["variant"]] = k
    srcs = sorted(dels)
    arms = ["orig", "top10:9b", "rand10:0", "top25:9b", "rand25:0", "topsent25", "randsent25:0"]
    tab = {}
    for a in arms:
        ks = [dels[s][a] for s in srcs]
        labels = Counter(P[k]["prediction_short"] for k in ks)
        tab[a] = {
            "labels": dict(labels),
            "human": labels.get("Human", 0),
            "not_ai": len(ks) - labels.get("AI", 0),
            "fraction_ai": ci([P[k]["fraction_ai"] for k in ks]),
            "margin": ci([margin(P[k])[0] for k in ks]),
            "capped_share": round(float(np.mean([margin(P[k])[1] for k in ks])), 4),
            "words": ci([len(items[k]["text"].split()) for k in ks]),
            **{f"{d}_flagged": sum(S[d][k] >= 0.5 for k in ks) for d in DETECTORS},
            **{f"{d}_mean": ci([S[d][k] for k in ks]) for d in DETECTORS},
        }
    out["deletion"] = tab
    for dose, (t, r) in {
        "10": ("top10:9b", "rand10:0"),
        "25": ("top25:9b", "rand25:0"),
        "sent25": ("topsent25", "randsent25:0"),
    }.items():
        mt = np.array([margin(P[dels[s][t]])[0] for s in srcs])
        mr = np.array([margin(P[dels[s][r]])[0] for s in srcs])
        mo = np.array([margin(P[dels[s]["orig"]])[0] for s in srcs])
        out[f"paired_{dose}"] = {
            "margin_top_minus_random": ci(mt - mr),
            "margin_drop_top": ci(mo - mt),
            "margin_drop_random": ci(mo - mr),
            "fraction_ai_top_minus_random": ci(
                [P[dels[s][t]]["fraction_ai"] - P[dels[s][r]]["fraction_ai"] for s in srcs]
            ),
            "nine_b_top_minus_random": ci([S["9b"][dels[s][t]] - S["9b"][dels[s][r]] for s in srcs]),
        }
    out["deletion_by_arm"] = {
        arm: {
            a: {
                "not_ai": sum(P[dels[s][a]]["prediction_short"] != "AI" for s in srcs if s.split("/")[1] == arm),
                "margin": ci([margin(P[dels[s][a]])[0] for s in srcs if s.split("/")[1] == arm]),
            }
            for a in arms
        }
        for arm in ("original", "plain+sampler")
    }
    ins: dict = defaultdict(dict)
    for k, it in items.items():
        if it["group"] == "insert":
            ins[it["src"]][it["variant"]] = k
    isrc = sorted(ins)
    itab: dict = {}
    for a in ("orig", "ai", "neutral"):
        ks = [ins[s][a] for s in isrc]
        labels = Counter(P[k]["prediction_short"] for k in ks)
        itab[a] = {
            "labels": dict(labels),
            "ai_or_mixed": len(ks) - labels.get("Human", 0),
            "fraction_ai": ci([P[k]["fraction_ai"] for k in ks]),
            "margin": ci([margin(P[k])[0] for k in ks]),
            **{d: ci([S[d][k] for k in ks]) for d in DETECTORS},
            **{f"{d}_flagged": sum(S[d][k] >= 0.5 for k in ks) for d in DETECTORS},
        }
    itab["paired_margin_ai_minus_neutral"] = ci(
        [margin(P[ins[s]["ai"]])[0] - margin(P[ins[s]["neutral"]])[0] for s in isrc]
    )
    for d in DETECTORS:
        itab[f"paired_{d}_ai_minus_neutral"] = ci([S[d][ins[s]["ai"]] - S[d][ins[s]["neutral"]] for s in isrc])
    itab["n_edits"] = ci([items[ins[s]["ai"]]["n_edits"] for s in isrc])
    out["insertion"] = itab
    out["splice"] = splices(P, items)
    d50: dict = defaultdict(dict)  # exploratory (plan amendment 6): half the words deleted
    for k, it in items.items():
        if it["group"] == "deletion50":
            d50[it["variant"]][it["src"]] = k
    out["deletion50"] = {
        v: {
            "labels": dict(Counter(P[k]["prediction_short"] for k in ks.values())),
            "not_ai": sum(P[k]["prediction_short"] != "AI" for k in ks.values()),
            "margin": ci([margin(P[k])[0] for k in ks.values()]),
            "words": ci([len(items[k]["text"].split()) for k in ks.values()]),
            **{f"{d}_flagged": sum(S[d][k] >= 0.5 for k in ks.values()) for d in DETECTORS},
        }
        for v, ks in sorted(d50.items())
    }
    typo = sorted(k for k, it in items.items() if it["group"] == "typo")  # exploratory (plan amendment 4)
    if typo:
        orig = [f"del/{items[k]['src']}/orig" for k in typo]
        out["typo"] = {
            "n": len(typo),
            "pangram_not_ai": [sum(P[k]["prediction_short"] != "AI" for k in ks) for ks in (orig, typo)],
            "pangram_margin_change": ci([margin(P[t])[0] - margin(P[o])[0] for t, o in zip(typo, orig, strict=True)]),
            **{f"{d}_change": ci([S[d][t] - S[d][o] for t, o in zip(typo, orig, strict=True)]) for d in DETECTORS},
            **{f"{d}_flagged": [sum(S[d][k] >= 0.5 for k in ks) for ks in (orig, typo)] for d in DETECTORS},
        }
    return out


def splices(P: dict, items: dict) -> dict:
    occl = load_json("attr/splices.json")
    spl = sorted(k for k, it in items.items() if it["group"] == "splice")
    agree: dict = defaultdict(list)
    on_truth: dict = defaultdict(list)
    acc_p, chance, ai_half_right = [], [], 0
    for k in spl:
        it = items[k]
        t = it["text"]
        spans = words(t)
        truth = np.array([(s >= it["boundary"]) == (it["variant"] == "human-first") for s, _ in spans], dtype=float)
        lab = word_window(t, P[k]["windows"])
        acc_p.append(float((lab == truth).mean()))
        chance.append(float(lab.mean()))
        ai_half_right += int(lab[truth == 1].mean() > 0.5 and lab[truth == 0].mean() < 0.5)
        for d in DETECTORS:
            pos = np.clip(np.asarray(occl[d][k]["w"]), 0, None)
            if 0 < lab.mean() < 1:
                agree[d].append(float(pos[lab == 1].sum() / max(pos.sum(), 1e-9)))
            on_truth[d].append(float(pos[truth == 1].sum() / max(pos.sum(), 1e-9)))
    return {
        "n": len(spl),
        "labels": dict(Counter(P[k]["prediction_short"] for k in spl)),
        "ai_window_on_ai_half": ai_half_right,
        "pangram_word_accuracy_vs_truth": ci(acc_p),
        "share_words_in_pangram_ai_windows": ci(chance),
        "positive_attr_in_pangram_ai_windows": {d: ci(agree[d]) for d in DETECTORS if agree[d]},
        "n_with_both_window_labels": len(agree[DETECTORS[0]]),
        "positive_attr_on_true_ai_half": {d: ci(on_truth[d]) for d in DETECTORS},
    }


# ------------------------------------------------------------------------------------------------------------- gen
def sec_gen(C: list[dict]) -> dict:
    from gen import auroc, discovery_auroc, matrix

    M = load_json("gen/means.json")
    feats_all, au_disc = discovery_auroc()
    top = [feats_all[j] for j in np.argsort(-au_disc, kind="stable")[:20]]
    disc = [r for r in C if r["split"] == "discovery"]
    conf = [r for r in C if r["split"] == "confirm"]
    y = np.array([r["kind"] == "ai" for r in conf])
    Xc, _ = matrix([r["key"] for r in conf], M, top)
    au_top = [auroc(Xc[y, j], Xc[~y, j]) for j in range(len(top))]
    Xd, fd = matrix([r["key"] for r in disc], M)
    active = [fd[j] for j in np.nonzero((Xd > 0).mean(0) >= 0.05)[0]]
    rand = random.Random(f"{SEED}-gen-random").sample(active, min(1000, len(active)))
    Xr, _ = matrix([r["key"] for r in conf], M, rand)
    au_rand = np.array([auroc(Xr[y, j], Xr[~y, j]) for j in range(len(rand))])
    au_rand_sym = np.maximum(au_rand, 1 - au_rand)
    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, binary=True)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(
        vec.fit_transform([r["text"] for r in disc]), [r["kind"] == "ai" for r in disc]
    )
    bow_auc = roc_auc_score(y, clf.predict_proba(vec.transform([r["text"] for r in conf]))[:, 1])
    Xd20, _ = matrix([r["key"] for r in disc], M, top)
    probe = LogisticRegression(C=1.0, max_iter=3000).fit(Xd20, [r["kind"] == "ai" for r in disc])
    out: dict = {
        "fvu_median": round(float(np.median([v["fvu"] for v in M.values()])), 4),
        "top_features": top,
        "top_auroc_discovery": [round(float(au_disc[feats_all.index(f)]), 4) for f in top],
        "top_auroc_confirm": [round(v, 4) for v in au_top],
        "best_auroc_confirm": round(float(max(au_top)), 4),
        "random_auroc_p99": round(float(np.percentile(au_rand_sym, 99)), 4),
        "random_auroc_max": round(float(au_rand_sym.max()), 4),
        "n_random": len(rand),
        "probe20_auroc_confirm": round(float(roc_auc_score(y, probe.predict_proba(Xc)[:, 1])), 4),
        "bow_auroc_confirm": round(float(bow_auc), 4),
    }
    # genre check: old books make "human" easy to spot, so repeat against the Wikipedia passages alone
    wk = np.array([r["kind"] == "ai" or r["arm"] == "wikipedia" for r in conf])
    yw = y[wk]
    conf_w = [r["text"] for r, k in zip(conf, wk, strict=True) if k]
    rw = np.array([auroc(Xr[wk][yw, j], Xr[wk][~yw, j]) for j in range(len(rand))])
    out["wikipedia_only"] = {
        "n_human": int((~yw).sum()),
        "n_ai": int(yw.sum()),
        "top_auroc": [round(auroc(Xc[wk][yw, j], Xc[wk][~yw, j]), 4) for j in range(len(top))],
        "probe20_auroc": round(float(roc_auc_score(yw, probe.predict_proba(Xc[wk])[:, 1])), 4),
        "bow_auroc": round(float(roc_auc_score(yw, clf.predict_proba(vec.transform(conf_w))[:, 1])), 4),
        "random_auroc_p99": round(float(np.percentile(np.maximum(rw, 1 - rw), 99)), 4),
    }
    G = load_json("gen/acts.json")
    A9 = load_json("attr/9b.json")
    share, rhos = [], []
    acc: dict[str, float] = defaultdict(float)
    for r in C:
        if r["kind"] != "ai" or r["split"] != "confirm":
            continue
        W = np.asarray(G["words"][r["key"]]).sum(0)
        w9 = np.asarray(A9[r["key"]]["w"])
        topw = np.argsort(-w9, kind="stable")[: max(1, round(0.10 * len(w9)))]
        share.append(W[topw].sum() / max(W.sum(), 1e-9))
        rhos.append(spearmanr(W, w9).statistic)
        for sp, v in zip(words(r["text"]), W, strict=True):
            acc[word_type(r["text"], sp)] += float(v)
    tot = sum(acc.values())
    out["share_on_9b_top10_words"] = ci(share)
    out["spearman_with_9b_words"] = ci([v for v in rhos if v == v])
    out["top_words_by_activation"] = [
        [t, round(v / tot, 4)] for t, v in sorted(acc.items(), key=lambda kv: -kv[1])[:20]
    ]
    P = load_json("pangram.json")
    items = {it["key"]: it for it in read_jsonl(DATA / "pangram_items.jsonl")}
    ratio, aucs, truth_aucs = [], [], []
    for k in sorted(kk for kk in items if items[kk]["group"] == "splice"):
        it = items[k]
        W = np.asarray(G["words"][k]).sum(0)
        lab = word_window(it["text"], P[k]["windows"])
        truth = np.array([(s >= it["boundary"]) == (it["variant"] == "human-first") for s, _ in words(it["text"])])
        truth_aucs.append(roc_auc_score(truth, W))
        if 0 < lab.mean() < 1:
            ratio.append(W[lab == 1].mean() / max(W[lab == 0].mean(), 1e-9))
            aucs.append(roc_auc_score(lab, W))
    out["splice_ai_window_activation_ratio"] = ci(ratio) if ratio else None
    out["splice_word_auroc_vs_pangram_windows"] = ci(aucs) if aucs else None
    out["splice_word_auroc_vs_truth"] = ci(truth_aucs)
    out["n_splices_with_both_labels"] = len(aucs)
    return out


# ------------------------------------------------------------------------------------------------------------- bow
def sec_bow() -> dict:
    """A bag of words on the 711 texts Pangram scored in deslop-h2h (318 human, 393 AI): which words predict its
    label."""
    P = json.loads((H2H / "results" / "rulers" / "pangram.json").read_text())
    hum = {f"human/{h['source']}/{h['id']}": h["text"] for h in read_jsonl(_human_path())}
    cache: dict = {}

    def gen_rows(path):
        if path not in cache:
            cache[path] = read_jsonl(path)
        return cache[path]

    texts, y, groups = [], [], []
    for k, v in sorted(P.items()):
        if k.startswith("human/"):
            t, g = hum.get(k), k
        elif k.startswith("A/"):
            _, arm, pid, smp = k.split("/")
            t = next(
                (
                    r["text"]
                    for r in gen_rows(H2H / "results" / "gen" / f"{arm}.jsonl")
                    if r["id"] == pid and str(r["sample"]) == smp
                ),
                None,
            )
            g = pid
        else:
            _, run, pid = k.split("/")
            t = next(
                (
                    r["text"]
                    for r in gen_rows(H2H.parent / "deslop" / "results" / "gen" / f"{run}.jsonl")
                    if r["id"] == pid
                ),
                None,
            )
            g = pid
        if t:
            texts.append(t)
            y.append(v["prediction_short"] != "Human")
            groups.append(g)
    ya = np.array(y)
    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=3, binary=True)
    X = vec.fit_transform(texts)
    pred = np.zeros(len(ya))
    for tr, te in GroupKFold(5).split(X, ya, groups):
        pred[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], ya[tr]).predict_proba(X[te])[:, 1]
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(X, ya)
    vocab = list(vec.get_feature_names_out())
    order = np.argsort(-clf.coef_[0])
    return {
        "n": len(ya),
        "n_ai": int(ya.sum()),
        "cv_auroc": round(float(roc_auc_score(ya, pred)), 4),
        "top_ai": [[vocab[i], round(float(clf.coef_[0][i]), 3)] for i in order[:30]],
        "top_human": [[vocab[i], round(float(clf.coef_[0][i]), 3)] for i in order[::-1][:30]],
        "lexicon_coefs": {w: round(float(clf.coef_[0][vocab.index(w)]), 3) for w in sorted(LEXICON) if w in vocab},
    }


# ----------------------------------------------------------------------------------------------------- predictions
def predictions(s: dict) -> list[dict]:
    rows = []
    x = s["deletion"]["9b"]["top25:9b"]["minus_random"]
    rows.append({"id": "P1", "value": x, "hit": bool(x[0] > 0.20)})
    pg = s.get("pangram")
    if pg:
        t, r = pg["deletion"]["top25:9b"]["human"], pg["deletion"]["rand25:0"]["human"]
        rows.append({"id": "P2", "value": [t, r, 60], "hit": bool(t <= 6 and abs(t - r) <= 3)})
        m = pg["paired_25"]["margin_top_minus_random"]
        rows.append({"id": "P3", "value": m, "hit": bool(m[1] <= 0 <= m[2] or m[0] > 0)})
        a, n = pg["insertion"]["ai"]["ai_or_mixed"], pg["insertion"]["neutral"]["ai_or_mixed"]
        rows.append({"id": "P4", "value": [a, n, 40], "hit": bool(a <= 4 and abs(a - n) <= 2)})
    rh = [s["words"]["agreement"][f"9b|{o}"]["spearman"] for o in DETECTORS[1:]]
    rows.append({"id": "P5", "value": rh, "hit": bool(float(np.median(rh)) > 0.3)})
    n6 = s["words"]["lists"]["9b"]["top30_punct_or_lexicon"]
    rows.append({"id": "P6", "value": n6, "hit": bool(n6 >= 5)})
    if "sae" in s:
        sa = s["sae"]["top10"]
        rows.append(
            {"id": "P7", "value": [sa["drop"][0], sa["random_max"]], "hit": bool(sa["drop"][0] > sa["random_max"])}
        )
    g = s.get("gen")
    if g:
        b, p99, sh = g["best_auroc_confirm"], g["random_auroc_p99"], g["share_on_9b_top10_words"][0]
        rows.append({"id": "P8", "value": [b, p99, sh], "hit": bool(b > 0.95 and b > p99 and sh < 0.30)})
    return rows


def main() -> None:
    C = corpus()
    s: dict = {"corpus": sec_corpus(C), "words": sec_words(C), "deletion": sec_deletion()}
    if (RESULTS / "sae" / "ablate.json").exists():
        s["sae"] = sec_sae()
    if (RESULTS / "pangram.json").exists():
        s["pangram"] = sec_pangram()
        if (RESULTS / "gen" / "acts.json").exists():
            s["gen"] = sec_gen(C)
    s["bow"] = sec_bow()
    s["predictions"] = predictions(s)
    save_json("summary.json", s)
    print(json.dumps(s["predictions"], indent=1))


if __name__ == "__main__":
    main()
