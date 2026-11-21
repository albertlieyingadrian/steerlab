"""M1-M3: every detector on one leaderboard (tests 1, 2, 3, 6), SAE and neuron probes included.

    python probes.py extract     # Modal: Qwen3-8B and Llama residuals, SAE features, MLP neurons -> data/acts_*.npz
    python probes.py score       # local voi-380M; Modal: the 9B detector (cascade) and the one-question judge
    python probes.py luar        # local LUAR embeddings                                  -> data/luar.npy
    python probes.py fit         # fit on `fit`, choose on `dev`, report on every test set -> results/leaderboard.json
    python probes.py curves      # learning curves (50 -> all labels) and cost per 1k texts -> results/curves.json

Fit set: Enron fit authors, three fit generators. Layer, k and C are chosen on dev; every number in the note is on a
test set: held-out authors (test), a held-out generator (heldgen), Blog (blog), two humanizers, frontier drafts, and the
two shortcut edits (test_fmt, test_len).
"""

from __future__ import annotations

import re
import sys
import time

import budget
import numpy as np
from common import (
    DATA,
    SEED,
    auroc,
    boot_auroc,
    ledger,
    load_json,
    read_jsonl,
    save_json,
)

LAYERS = {"qwen": [9, 18, 27], "llama": [8, 19, 28]}
NEURON_LAYER = {"qwen": 18, "llama": None}
KS = [16, 64, 256]
CS = [0.01, 0.1, 1.0]
EVAL_SETS = {  # name: (human set, AI set)
    "test": ("test", "test"),
    "heldgen": ("test", "heldgen"),
    "blog": ("blog", "blog"),
    "hum_prompt": ("test", "hum_prompt"),
    "hum_dipper": ("test", "hum_dipper"),
    "frontier": ("test", "frontier"),
    "test_fmt": ("test_fmt", "test_fmt"),
    "test_len": ("test_len", "test_len"),
}


def texts():
    return read_jsonl(DATA / "texts.jsonl")


def fn(name):
    import modal

    return modal.Function.from_name("tl-ep1-core", name)


# ---------------------------------------------------------------- extraction (Modal)
def extract():
    T = texts()
    allt = [t["text"] for t in T]
    for key in sys.argv[2:] or ("qwen", "llama"):
        chunks = []
        for s in range(0, len(allt), 2500):
            r = budget.paid(
                f"acts-{key}-{s}",
                "H100",
                1.5,
                fn("acts").remote,
                key,
                allt[s : s + 2500],
                LAYERS[key],
                NEURON_LAYER[key],
            )
            chunks.append(r)
            print(
                key,
                s,
                r["seconds"],
                "s; SAE variance explained",
                round(r["sae_ve"], 3),
                flush=True,
            )
        d, top = chunks[0]["d"], chunks[0]["sae_top"]
        out = {
            f"res{L}": np.concatenate([np.frombuffer(c["res"][str(L)], np.float16).reshape(c["n"], d) for c in chunks])
            for L in LAYERS[key]
        }
        out["sae_idx"] = np.concatenate([np.frombuffer(c["sae_idx"], np.int32).reshape(c["n"], top) for c in chunks])
        out["sae_val"] = np.concatenate([np.frombuffer(c["sae_val"], np.float16).reshape(c["n"], top) for c in chunks])
        if NEURON_LAYER[key] is not None:
            dn = chunks[0]["d_neuron"]
            out["neuron"] = np.concatenate([np.frombuffer(c["neuron"], np.float16).reshape(c["n"], dn) for c in chunks])
        out["ntok"] = np.concatenate([np.array(c["ntok"]) for c in chunks])
        out["sae_ve"] = np.array([c["sae_ve"] for c in chunks])
        out["seconds"] = np.array([c["seconds"] for c in chunks])
        np.savez(DATA / f"acts_{key}.npz", ids=np.array([t["id"] for t in T]), **out)


# ---------------------------------------------------------------- detectors and judge
VOI380, VOI380_REV = (
    "VoightKampffLabs/voi-slop-detect-380M",
    "a109d591b8b62541f362d873765900380b2bd0ea",
)


def voi380(xs, bs=16):
    """voi-380M loaded natively at a pinned revision (never with remote code, which scores every text ~0.73)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(VOI380, revision=VOI380_REV)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = AutoModelForSequenceClassification.from_pretrained(VOI380, revision=VOI380_REV).eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            enc = tok(
                xs[i : i + bs],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            out += torch.sigmoid(net(**enc).logits.float()).reshape(-1).cpu().tolist()
    return np.array(out)


def eval_rows(T):
    keep = set()
    for hs, as_ in EVAL_SETS.values():
        keep |= {t["id"] for t in T if (t["set"] == hs and t["label"] == 0) or (t["set"] == as_ and t["label"] == 1)}
    keep |= {t["id"] for t in T if t["set"] == "dev"}
    return [t for t in T if t["id"] in keep]


def score():
    import modal

    E = eval_rows(texts())
    xs = [t["text"] for t in E]
    t0 = time.time()
    v = voi380(xs)
    voi_sec = time.time() - t0
    print("voi380", len(xs), round(voi_sec), "s", flush=True)
    # cascade: the 10% closest to the cut are re-scored by the 9B detector (run f9-9b on the slop-detector app)
    k = max(1, round(0.10 * len(xs)))
    idx = np.argsort(np.abs(v - 0.5))[:k]
    s9 = budget.paid(
        "nine-b",
        "H100",
        1.0,
        modal.Function.from_name("slop-detector", "score_run").remote,
        "f9-9b",
        {"x": [xs[i] for i in idx]},
    )["x"]
    casc = v.copy()
    casc[idx] = s9
    j = budget.paid("judge-qwen32b", "H100", 1.5, fn("judge").remote, "Qwen/Qwen3-32B", xs)
    np.savez(
        DATA / "scores.npz",
        ids=np.array([t["id"] for t in E]),
        voi380=v,
        cascade=casc,
        nine_b_idx=idx,
        nine_b=np.array(s9),
        judge=np.array(j["scores"]),
        seconds=np.array([voi_sec, j["seconds"]]),
    )


def luar():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")
    model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).eval()
    T = texts()
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(T), 32):
            batch = [t["text"] for t in T[i : i + 32]]
            enc = tok(
                batch,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.reshape(len(batch), 1, -1) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            e = model(**enc)
            e = e[0] if isinstance(e, tuple) else e
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    np.save(DATA / "luar.npy", np.concatenate(out))
    print("luar", len(T), round(time.time() - t0), "s")


# ---------------------------------------------------------------- cheap features
FUNC = {
    "contraction": r"\b\w+'(s|t|re|ve|ll|d|m)\b",
    "first_person": r"\b(i|me|my|mine)\b",
    "we": r"\b(we|our|us)\b",
    "you": r"\b(you|your)\b",
    "hedge": r"\b(maybe|perhaps|i think|probably|guess|kind of|sort of)\b",
    "intens": r"\b(really|very|truly|incredibly|absolutely|so)\b",
    "transition": r"\b(additionally|furthermore|moreover|however|overall|ultimately|importantly)\b",
    "polite": r"\b(please|thank you|thanks|appreciate|grateful)\b",
    "hope": r"\bi hope\b",
    "casual": r"\b(lol|haha|ok|okay|yeah|gonna|wanna)\b",
}
STYLE_NAMES = [
    "type_token_ratio",
    "mean_sentence_len",
    "sentence_len_cv",
    "lowercase_sentence_starts",
    "dash_rate",
    "exclamation_rate",
    "question_rate",
    "list_lines",
    "all_caps_words",
    "ellipsis_rate",
    "long_words",
    "double_spaces",
    "article_rate",
    *[f"{k}_rate" for k in FUNC],
    "commas_per_sentence",
]
STATS_NAMES = [
    "words",
    "chars",
    "mean_word_len",
    "mean_sentence_len",
    "sd_sentence_len",
    "paragraph_breaks",
    *[f"rate_{c}" for c in ",.!?;:()\"'-"],
    "digits",
    "uppercase",
    "newlines",
]


def stats_feats(t):
    w = t.split()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    sl = [len(s.split()) for s in sents] or [0]
    n = max(len(t), 1)
    return [
        len(w),
        len(t),
        np.mean([len(x) for x in w]) if w else 0,
        np.mean(sl),
        np.std(sl),
        t.count("\n\n"),
        *[t.count(c) / n * 1000 for c in ",.!?;:()\"'-"],
        sum(ch.isdigit() for ch in t) / n * 1000,
        sum(ch.isupper() for ch in t) / n * 1000,
        t.count("\n") / n * 1000,
    ]


def style_feats(t):
    low = t.lower()
    w = re.findall(r"[a-z']+", low)
    nw = max(len(w), 1)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    sl = [len(s.split()) for s in sents] or [0]
    f = [
        len(set(w)) / nw,
        np.mean(sl),
        np.std(sl) / (np.mean(sl) + 1e-6),
        np.mean([s[0].islower() for s in sents]) if sents else 0,
        (t.count("—") + t.count("--")) / nw * 100,
        t.count("!") / nw * 100,
        t.count("?") / nw * 100,
        len(re.findall(r"^\s*([-*•]|\d+\.)\s", t, re.MULTILINE)) / max(len(sents), 1),
        len(re.findall(r"\b[A-Z]{2,}\b", t)) / nw * 100,
        len(re.findall(r"\.\.\.", t)) / nw * 100,
        np.mean([len(x) > 8 for x in w]) if w else 0,
        len(re.findall(r"  +", t)) / nw * 100,
        len(re.findall(r"\b(the|a|an)\b", low)) / nw,
    ]
    f += [len(re.findall(p, low)) / nw * 100 for p in FUNC.values()]
    f += [low.count(",") / max(len(sents), 1)]
    return f


# ---------------------------------------------------------------- fitting helpers
def lr(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y)


def sae_dense(idx, val, n_feat=65536):
    from scipy.sparse import csr_matrix

    n, k = idx.shape
    return csr_matrix(
        (val.astype(np.float32).ravel(), (np.repeat(np.arange(n), k), idx.ravel())),
        shape=(n, n_feat),
    )


def top_by_diff(X, y, k):
    """The k columns with the largest |mean(AI) - mean(human)| / std on the given rows."""
    X = X.toarray() if hasattr(X, "toarray") else X
    m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
    return np.argsort(-np.abs(m1 - m0) / (X.std(0) + 1e-6))[:k]


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


class Data:
    def __init__(self):
        self.T = texts()
        self.id2i = {t["id"]: i for i, t in enumerate(self.T)}
        self.y = np.array([t["label"] for t in self.T])
        self.sets = np.array([t["set"] for t in self.T])
        self.author = np.array([t["author"] or "" for t in self.T])

    def idx(self, name):
        if name in ("fit", "dev"):
            return np.flatnonzero(self.sets == name)
        hs, as_ = EVAL_SETS[name]
        return np.flatnonzero(((self.sets == hs) & (self.y == 0)) | ((self.sets == as_) & (self.y == 1)))


def feature_methods(D):
    M = {
        "stats": (np.array([stats_feats(t["text"]) for t in D.T]), "baseline"),
        "style": (np.array([style_feats(t["text"]) for t in D.T]), "small probe"),
    }
    if (DATA / "luar.npy").exists():
        M["luar"] = (np.load(DATA / "luar.npy"), "small probe")
    return M


def fit_all():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    D = Data()
    fit, dev = D.idx("fit"), D.idx("dev")
    yf, yd = D.y[fit], D.y[dev]
    scores, chosen, fam, models = {}, {}, {}, {}

    def choose(name, X, family, ks=(None,)):
        best = None
        for k in ks:
            cols = None if k is None else top_by_diff(X[fit], yf, k)
            Xs = dense(X if cols is None else X[:, cols])
            for C in CS:
                m = lr(Xs[fit], yf, C)
                a = auroc(yd, m.predict_proba(Xs[dev])[:, 1])
                ledger(
                    f"dev-{name}-k{k}-C{C}",
                    "dev_auroc",
                    a,
                    0.0,
                    "grid",
                    "fit on fit, scored on dev",
                )
                if best is None or a > best[0]:
                    best = (a, k, C, m, Xs, cols)
        a, k, C, m, Xs, cols = best
        scores[name], fam[name] = m.predict_proba(Xs)[:, 1], family
        chosen[name] = dict(dev_auroc=round(a, 4), k=k, C=C)
        models[name] = dict(model=m, cols=None if cols is None else cols.tolist())
        ledger(f"dev-{name}", "dev_auroc", a, 0.0, "kept", f"k={k} C={C}")

    for name, (X, family) in feature_methods(D).items():
        choose(name, X, family)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    tf.fit([D.T[i]["text"] for i in fit])
    Xb = tf.transform([t["text"] for t in D.T])
    best = None
    for C in [1.0, 10.0, 100.0]:
        m = LogisticRegression(C=C, max_iter=3000).fit(Xb[fit], yf)
        a = auroc(yd, m.predict_proba(Xb[dev])[:, 1])
        ledger(f"dev-bow-C{C}", "dev_auroc", a, 0.0, "grid", "tf-idf 1-2 grams")
        if best is None or a > best[0]:
            best = (a, C, m)
    scores["bow"], fam["bow"] = best[2].predict_proba(Xb)[:, 1], "baseline"
    chosen["bow"] = dict(dev_auroc=round(best[0], 4), C=best[1])
    directions = {}
    for key in ("qwen", "llama"):
        p = DATA / f"acts_{key}.npz"
        if not p.exists():
            continue
        Z = np.load(p)
        assert list(Z["ids"]) == [t["id"] for t in D.T], "activation rows out of order"
        best = None
        for L in LAYERS[key]:
            X = Z[f"res{L}"].astype(np.float32)
            for C in CS:
                m = lr(X[fit], yf, C)
                a = auroc(yd, m.predict_proba(X[dev])[:, 1])
                ledger(
                    f"dev-act-{key}-L{L}-C{C}",
                    "dev_auroc",
                    a,
                    0.0,
                    "grid",
                    "mean-pooled residual",
                )
                if best is None or a > best[0]:
                    best = (a, L, C, m, X)
        a, L, C, m, X = best
        nm = f"act_{key}"
        scores[nm], fam[nm] = m.predict_proba(X)[:, 1], "small probe"
        chosen[nm] = dict(dev_auroc=round(a, 4), layer=L, C=C)
        ledger(f"dev-{nm}", "dev_auroc", a, 0.0, "kept", f"layer={L} C={C}")
        directions[nm] = dict(layer=L, w=(m[1].coef_[0] / m[0].scale_))  # direction in raw residual units
        choose(
            f"sae_{key}",
            sae_dense(Z["sae_idx"], Z["sae_val"]).tocsc(),
            "interpretability",
            KS,
        )
        if "neuron" in Z.files:
            choose(f"neuron_{key}", Z["neuron"].astype(np.float32), "interpretability", KS)
    if (DATA / "scores.npz").exists():
        S = np.load(DATA / "scores.npz")
        pos = {i: k for k, i in enumerate(S["ids"])}
        for nm in ("voi380", "cascade", "judge"):
            v = np.full(len(D.T), np.nan)
            for t in D.T:
                if t["id"] in pos:
                    v[D.id2i[t["id"]]] = S[nm][pos[t["id"]]]
            scores[nm], fam[nm] = (
                v,
                ("LLM judge" if nm == "judge" else "trained detector"),
            )
            chosen[nm] = dict(dev_auroc=round(auroc(yd, v[dev]), 4))
    board = {}
    for nm, s in scores.items():
        s = np.asarray(s, float)
        row = dict(family=fam[nm], chosen=chosen[nm])
        for ev in EVAL_SETS:
            ix = D.idx(ev)
            ix = ix[~np.isnan(s[ix])]
            if len(np.unique(D.y[ix])) == 2:
                row[ev] = boot_auroc(D.y[ix], s[ix], D.author[ix])
        board[nm] = row
    save_json("leaderboard.json", board)
    np.savez(
        DATA / "method_scores.npz",
        ids=np.array([t["id"] for t in D.T]),
        **{k: np.asarray(v, float) for k, v in scores.items()},
    )
    np.savez(
        DATA / "directions.npz",
        **{f"{k}_w": v["w"].astype(np.float32) for k, v in directions.items()},
        **{f"{k}_layer": np.array(v["layer"]) for k, v in directions.items()},
    )
    import pickle

    (DATA / "models.pkl").write_bytes(pickle.dumps(dict(models=models, chosen=chosen)))
    for nm, row in sorted(board.items(), key=lambda kv: -kv[1].get("test", {}).get("auroc", 0)):
        print(
            f"{nm:14s} {row['family']:16s} "
            + " ".join(f"{ev}={row[ev]['auroc']:.3f}" for ev in EVAL_SETS if ev in row),
            flush=True,
        )


def curves():
    """Learning curves on the held-out test set: n balanced labels from fit, 3 seeds; plus compute per 1k texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    D = Data()
    fit, test = D.idx("fit"), D.idx("test")
    lb = load_json("leaderboard.json")
    feats = {k: v[0] for k, v in feature_methods(D).items()}
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True).fit(
        [D.T[i]["text"] for i in fit]
    )
    feats["bow"] = tf.transform([t["text"] for t in D.T])
    Zq = np.load(DATA / "acts_qwen.npz")
    feats["act_qwen"] = Zq[f"res{lb['act_qwen']['chosen']['layer']}"].astype(np.float32)
    feats["sae_qwen"] = sae_dense(Zq["sae_idx"], Zq["sae_val"]).tocsc()
    out = {}
    rng = np.random.default_rng(SEED)
    pos_all, neg_all = fit[D.y[fit] == 1], fit[D.y[fit] == 0]
    for nm, X in feats.items():
        out[nm] = {}
        for n in [50, 100, 200, 400, len(fit)]:
            vals = []
            for _ in range(3 if n < len(fit) else 1):
                pick = np.concatenate(
                    [
                        rng.choice(pos_all, min(n // 2, len(pos_all)), replace=False),
                        rng.choice(neg_all, min(n // 2, len(neg_all)), replace=False),
                    ]
                )
                if nm == "sae_qwen":
                    cols = top_by_diff(X[pick], D.y[pick], lb[nm]["chosen"]["k"])
                    Xn = dense(X[:, cols])
                    s = lr(Xn[pick], D.y[pick], lb[nm]["chosen"]["C"]).predict_proba(Xn[test])[:, 1]
                elif nm == "bow":
                    m = LogisticRegression(C=lb["bow"]["chosen"]["C"], max_iter=3000).fit(X[pick], D.y[pick])
                    s = m.predict_proba(X[test])[:, 1]
                else:
                    s = lr(X[pick], D.y[pick], lb[nm]["chosen"]["C"]).predict_proba(X[test])[:, 1]
                vals.append(auroc(D.y[test], s))
            out[nm][str(n)] = round(float(np.mean(vals)), 4)
        print(nm, out[nm], flush=True)
    n_all = len(D.T)
    cost = {}
    for k in ("qwen", "llama"):
        sec = float(np.load(DATA / f"acts_{k}.npz")["seconds"].sum()) / n_all * 1000
        cost[f"activations + SAE, {k} (Modal H100)"] = dict(
            sec_per_1k=round(sec, 1), usd_per_1k=round(budget.charge("H100", sec), 4)
        )
    S = np.load(DATA / "scores.npz")
    cost["voi-380M (laptop)"] = dict(sec_per_1k=round(float(S["seconds"][0]) / len(S["ids"]) * 1000, 1), usd_per_1k=0)
    js = float(S["seconds"][1]) / len(S["ids"]) * 1000
    cost["judge, Qwen3-32B (Modal H100)"] = dict(
        sec_per_1k=round(js, 1), usd_per_1k=round(budget.charge("H100", js), 4)
    )
    sample = [t["text"] for t in D.T[:1000]]
    for nm, f in [
        ("stats", lambda: [stats_feats(t) for t in sample]),
        ("style", lambda: [style_feats(t) for t in sample]),
        ("bow", lambda: tf.transform(sample)),
    ]:
        t0 = time.time()
        f()
        cost[f"{nm} (laptop)"] = dict(sec_per_1k=round(time.time() - t0, 2), usd_per_1k=0)
    save_json("curves.json", dict(curves=out, cost_per_1k=cost))


if __name__ == "__main__":
    dict(extract=extract, score=score, luar=luar, fit=fit_all, curves=curves)[sys.argv[1]]()
