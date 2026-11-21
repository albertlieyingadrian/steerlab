"""Local rulers on the released human-judgement items: LUAR (Episode 1), the detector, and cheap baselines.

    python rulers_local.py h1|h2|h3|all

LUAR variants are scored on both splits; which variant is confirmatory is chosen on dev only (results.tsv):
  strict / raw text form; "mean" = mean cosine to each reference text, "ep" = cosine to one LUAR episode of all refs.
Writes results/scores_local_h*.json (numbers only).
"""

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# isort: off
from common4 import LOCAL, RES, dump, read_jsonl, strict  # first: puts Episode 1 on sys.path
from common import luar_embed
from detectors import voi380

# isort: on


def luar_single(texts, bs=64):
    """One embedding per text (each text its own 1-text episode)."""
    return np.concatenate([luar_embed([[t] for t in texts[i : i + bs]]) for i in range(0, len(texts), bs)])


def tfidf_sims(pairs, analyzer, ngram):
    """Cosine between the two sides of each pair, fitted on all texts involved."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=1, sublinear_tf=True)
    allt = sorted({t for p in pairs for t in p})
    X = vec.fit_transform(allt)
    idx = {t: i for i, t in enumerate(allt)}
    return [float(X[idx[a]].multiply(X[idx[b]]).sum()) for a, b in pairs]


# ---------------------------------------------------------------- H1
def h1():
    lab = json.loads((RES / "labels_h1.json").read_text())
    T = {r["tid"]: r["text"] for r in read_jsonl(LOCAL / "h1_texts.jsonl")}
    refs = defaultdict(list)
    for r in lab:
        if r["kind"] == "control":
            refs[r["pid"]].append(r["tid"])
    pids = sorted(refs)
    out = {}
    for form, f in (("strict", strict), ("raw", lambda t: t)):
        tids = list(T)
        E = dict(zip(tids, luar_single([f(T[t]) for t in tids])))
        EP = dict(zip(pids, luar_embed([[f(T[t]) for t in refs[p]] for p in pids])))  # one episode per person
        R = {p: np.stack([E[t] for t in refs[p]]) for p in pids}
        for r in lab:
            p, tid = r["pid"], r["tid"]
            own = [t for t in refs.get(p, []) if t != tid]  # a control text is never its own reference
            if not own:
                continue
            e = E[tid]
            s_mean = float(np.mean([E[t] @ e for t in own]))
            # the episode includes the text itself for controls, so controls fall back to the per-ref mean
            s_ep = float(EP[p] @ e) if r["kind"] != "control" else s_mean
            others = [q for q in pids if q != p]
            o_mean = float(np.mean([np.mean(R[q] @ e) for q in others]))
            o_ep = float(np.mean([EP[q] @ e for q in others]))
            d = out.setdefault(tid, {})
            d[f"luar_mean_{form}"] = s_mean
            d[f"luar_ep_{form}"] = s_ep
            d[f"spec_mean_{form}"] = s_mean - o_mean
            d[f"spec_ep_{form}"] = s_ep - o_ep
    # cheap baselines: character 3-gram and word TF-IDF cosine to own refs (strict), words, detector
    pairs, keys = [], []
    for r in lab:
        for t in [t for t in refs.get(r["pid"], []) if t != r["tid"]]:
            pairs.append((strict(T[r["tid"]]), strict(T[t])))
            keys.append(r["tid"])
    for name, an, ng in (("tfidf_char", "char_wb", (3, 3)), ("tfidf_word", "word", (1, 1))):
        acc = defaultdict(list)
        for k, s in zip(keys, tfidf_sims(pairs, an, ng)):
            acc[k].append(s)
        for k, v in acc.items():
            out.setdefault(k, {})[name] = float(np.mean(v))
    # distance from the o4-mini draft (character TF-IDF): edits get 1 - cos(edit, draft); drafts get 0
    ek = [t for t in T if t.endswith("-edit")]
    for k, s in zip(
        ek, tfidf_sims([(strict(T[t]), strict(T[t[: -len("-edit")] + "-draft"])) for t in ek], "char_wb", (3, 3))
    ):
        out.setdefault(k, {})["dist_from_draft"] = 1 - s
        out.setdefault(k[: -len("-edit")] + "-draft", {})["dist_from_draft"] = 0.0
    words = {r["tid"]: r["words"] for r in lab}
    tids = list(T)
    for k, s in zip(tids, voi380([T[t] for t in tids])):
        out.setdefault(k, {})["det"] = float(s)
        out[k]["words"] = words[k]
    dump(RES / "scores_local_h1.json", out)
    print("H1 scored", len(out))


# ---------------------------------------------------------------- H2
def h2():
    lab = json.loads((RES / "labels_h2.json").read_text())
    X = {r["eid"]: r["text"] for r in read_jsonl(LOCAL / "h2_excerpts.jsonl")}
    eids = list(X)
    out = {}
    for form, f in (("strict", strict), ("raw", lambda t: t)):
        E = dict(zip(eids, luar_single([f(X[e]) for e in eids])))
        for t in lab:
            d = out.setdefault(t["tri"], {})
            d[f"luar_ab_{form}"] = float(E[t["a"]] @ E[t["b"]])
            d[f"luar_ac_{form}"] = float(E[t["a"]] @ E[t["c"]])
    for name, an, ng in (("tfidf_char", "char_wb", (3, 3)), ("tfidf_word", "word", (1, 1))):
        ab = tfidf_sims([(strict(X[t["a"]]), strict(X[t["b"]])) for t in lab], an, ng)
        ac = tfidf_sims([(strict(X[t["a"]]), strict(X[t["c"]])) for t in lab], an, ng)
        for t, x, y in zip(lab, ab, ac):
            out[t["tri"]][f"{name}_ab"], out[t["tri"]][f"{name}_ac"] = x, y
    det = dict(zip(eids, voi380([X[e] for e in eids])))
    wc = {e: len(X[e].split()) for e in eids}
    for t in lab:
        d = out[t["tri"]]
        d["det_b"], d["det_c"] = float(det[t["b"]]), float(det[t["c"]])
        d["words_ab"] = -abs(wc[t["a"]] - wc[t["b"]])  # length baseline: closer length = "more similar"
        d["words_ac"] = -abs(wc[t["a"]] - wc[t["c"]])
    dump(RES / "scores_local_h2.json", out)
    print("H2 scored", len(out))


# ---------------------------------------------------------------- H3
def h3():
    rows = read_jsonl(LOCAL / "h3_texts.jsonl")
    s = voi380([r["text"] for r in rows])
    out = {r["aid"]: dict(det=float(x), words=len(r["text"].split())) for r, x in zip(rows, s)}
    dump(RES / "scores_local_h3.json", out)
    print("H3 scored", len(out))


def luar_check():
    """LUAR's known-answer check (plan, Design 3), per split: each unassisted text's similarity to its writer's other
    unassisted text (own) against every other writer's unassisted texts in the same split (other). Each pair once."""
    from common4 import auroc

    lab = json.loads((RES / "labels_h1.json").read_text())
    T = {r["tid"]: r["text"] for r in read_jsonl(LOCAL / "h1_texts.jsonl")}
    ctl = [r for r in lab if r["kind"] == "control"]
    E = dict(zip([r["tid"] for r in ctl], luar_single([strict(T[r["tid"]]) for r in ctl])))
    out = {}
    for split in ("dev", "heldout"):
        refs = defaultdict(list)
        for r in ctl:
            if r["split"] == split:
                refs[r["pid"]].append(r["tid"])
        pos, neg = [], []
        for p, ts in refs.items():
            for i, t in enumerate(ts):
                pos += [float(E[t] @ E[u]) for u in ts[i + 1 :]]
                neg += [float(E[t] @ E[u]) for q, us in refs.items() if q > p for u in us]
        a = round(auroc([1] * len(pos) + [0] * len(neg), pos + neg), 4)
        out[split] = dict(auroc=a, n_own=len(pos), n_other=len(neg), pass_bar=0.70, passed=a >= 0.70)
    dump(RES / "ruler_luar_h1_check.json", out)
    print("LUAR known-answer check:", out)


# ---------------------------------------------------------------- the user's pairs
def user():
    """Per draft in the frozen pair set: LUAR similarity to the writer's 5 profile emails (strict, per-ref mean),
    specificity against the other held-out writers' profiles, character TF-IDF, detector, words.
    Keys are pair-local draft ids "<pair>:<a|b>"; repeats share their original's drafts."""
    from common import authors, texts

    pairs = json.loads((LOCAL / "user_pairs.json").read_text())
    held = {a["author"]: a for a in authors()["authors"] if a["split"] == "heldout"}
    prof = {n: [strict(t) for t in texts(a, "profile")] for n, a in held.items()}
    names = sorted(prof)
    PE = {n: luar_single(prof[n]) for n in names}
    items = [(f"{p['pair']}:{s}", p["author"], p[s]) for p in pairs for s in ("a", "b")]
    E = luar_single([strict(t) for _, _, t in items])
    tf = tfidf_sims([(strict(t), r) for _, n, t in items for r in prof[n]], "char_wb", (3, 3))
    det = voi380([t for _, _, t in items])
    out, j = {}, 0
    for i, (k, n, t) in enumerate(items):
        own = float(np.mean(PE[n] @ E[i]))
        oth = float(np.mean([np.mean(PE[m] @ E[i]) for m in names if m != n]))
        k_ = len(prof[n])
        out[k] = dict(
            luar=own, spec=own - oth, tfidf_char=float(np.mean(tf[j : j + k_])), det=float(det[i]), words=len(t.split())
        )
        j += k_
    dump(RES / "scores_local_user.json", out)
    print("user drafts scored", len(out))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("h1", h1), ("check", luar_check), ("h2", h2), ("h3", h3), ("user", user)):
        if which in (name, "all"):
            fn()
