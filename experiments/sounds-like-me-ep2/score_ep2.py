"""Episode 2 scoring, unseen rulers and the persona projection test.

    python score_ep2.py derive SPLIT            # Rung 1 (first BoN draft), -critique, -rerank from the pools (no generation)
    python score_ep2.py comparators             # Episode 1 held-out cells reused as comparators -> data/drafts/ep1-*.jsonl
    python score_ep2.py score SPLIT [FILES...]  # voice (strict/loose/80-word), specificity, detector, loops, copy, task
    python score_ep2.py honesty FILES...        # frozen Qwen3-32B extractor + rule layer (Episode 1's ruler, A3)
    python score_ep2.py attributer              # character n-gram attributer: pass bar + readout per episode
    python score_ep2.py pref                    # voice preference model (dev-only, simulated pairs): accuracy + readout
    python score_ep2.py projection              # persona test (a): assistant-direction projection of Episode 1 drafts
    python score_ep2.py robust                  # loops, degenerate output, phone numbers; clean re-score (added after results)

Units: an episode is one arm's 4 drafts for one author (Episode 1's amendment A1). Strict = greeting and sign-off
stripped, whitespace collapsed (A2). Specificity = similarity to the target's profile minus the mean similarity to the
other authors' profiles in the same split (A4). 80-word = each draft cut to its first 80 words, then scored on the same
floor and ceiling (Episode 1's length check). Analysis and predictions: analyze_ep2.py.
"""

import json
import random
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from ep2 import (
    DATA,
    EP1_DATA,
    RES,
    A_by_name,
    B_by_id,
    cut_words,
    honesty_prompts,
    loops,
    parse,
    pick,
    run_vllm,
    strict,
)

sys.path.append(str(Path(__file__).parent.parent / "sounds-like-me"))
from common import VoiceScale, authors, luar_embed, record_spend, texts
from honesty import rule_flags
from modal_gen import charge
from score import longest_shared_run, ngram_copy_share, task_coverage, words

RNG = np.random.default_rng(20261121)
SCORED = RES / "scored"
COMPARATORS = {
    # Episode 1 held-out cells reused as they are (same authors, briefs, rulers)
    "ep1-gemini-edits": ("gemini-4-carbon", "edits"),  # best honest cell (pre-registered comparator)
    "ep1-gemini-k3": ("gemini-4-carbon", "k3_rand"),
    "ep1-llamabase-k3": ("Llama-3.1-8B", "k3_rand"),  # best cell outright (58% unsupported)
    "ep1-sonnet-note": ("claude-sonnet-5.5", "note"),
    "ep1-qwen8b-k8": ("Qwen3-8B", "k8_rand"),
    "ep1-qwen8b-k0": ("Qwen3-8B", "k0"),
}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


# ---------------------------------------------------------------- derived arms (no generation)
def derive(split):
    def rows_from(pool, name, choose):
        by = defaultdict(list)
        for c in read_jsonl(DATA / "pools" / f"{pool}__{split}.jsonl"):
            by[c["job"]].append(c)
        out = []
        for job, cs in by.items():
            ch = choose(cs)
            out.append(
                dict(
                    job=f"{job}|{name}",
                    model="ep2-Qwen3-8B",
                    author=ch["author"],
                    brief=ch["brief"],
                    arm=name,
                    text=ch["text"],
                    nll=ch["nll"],
                    stage=ch["stage"],
                    n_pool=len(cs),
                )
            )
        write_jsonl(DATA / "drafts" / f"{name}__{split}.jsonl", out)
        print(name, split, len(out))

    def first(cs):
        return min((c for c in cs if c["stage"] == "draft"), key=lambda c: c["k"])

    rows_from("R1b_bon", "R1_prompt", first)
    rows_from("R2_harness", "abl_norerank", first)
    rows_from("R2_harness", "abl_norevise", lambda cs: pick([c for c in cs if c["stage"] == "draft"]))


def comparators():
    for name, (model, arm) in COMPARATORS.items():
        src = [
            r
            for r in json.loads((EP1_DATA / "scored" / f"{model}__heldout.json").read_text())["drafts"]
            if r["model"] == model and r["arm"] == arm
        ]
        # Episode 1's job ids repeat across models ("author|brief|k3_rand"), so each comparator gets its own prefix
        rows = [dict(r, model=name, arm=name, job=f"{name}|{r['job']}") for r in src]
        write_jsonl(DATA / "drafts" / f"{name}__heldout.jsonl", rows)
        # Episode 1's honesty verdicts for the same drafts (same frozen extractor)
        hon = json.loads((EP1_DATA / "honesty" / f"{model}__heldout.json").read_text())
        sub = {f"{name}|{r['job']}": hon[r["job"]] for r in src if r["job"] in hon}
        (DATA / "honesty").mkdir(parents=True, exist_ok=True)
        (DATA / "honesty" / f"{name}__heldout.json").write_text(json.dumps(sub, indent=1))
        print(name, len(rows), "drafts,", len(sub), "honesty verdicts")


# ---------------------------------------------------------------- voice scales (cached per author)
_SCALES = {}


def scales(au):
    if au not in _SCALES:
        A = A_by_name()
        _SCALES[au] = (VoiceScale(A[au], authors()["floor"], "strict"), VoiceScale(A[au], authors()["floor"], "loose"))
    return _SCALES[au]


_PROF = {}


def split_profiles(split):
    if split not in _PROF:
        recs = [a for a in authors()["authors"] if a["split"] == split]
        prof = luar_embed([[strict(t) for t in texts(a, "profile")] for a in recs])
        _PROF[split] = ([a["author"] for a in recs], prof)
    return _PROF[split]


def specificity(vec, au, split):
    names, prof = split_profiles(split)
    s = prof @ vec
    i = names.index(au)
    return float(s[i] - np.delete(s, i).mean())


def score(split, files=None):
    from detectors import voi380

    A, B = A_by_name(), B_by_id()
    paths = [Path(f) for f in files] if files else sorted((DATA / "drafts").glob(f"*__{split}.jsonl"))
    SCORED.mkdir(parents=True, exist_ok=True)
    for path in paths:
        rows = read_jsonl(path)
        det = voi380([r["text"] or " " for r in rows])
        for r, d in zip(rows, det):
            src = texts(A[r["author"]], "samples")
            r.update(
                words=len(words(r["text"])),
                det=float(d),
                loop=loops(r["text"]),
                longest_run=longest_shared_run(r["text"], src),
                copy8=ngram_copy_share(r["text"], src),
                task=task_coverage(r["text"], B[r["brief"]]),
                empty=not r["text"].strip(),
            )
        groups = defaultdict(list)
        for r in rows:
            groups[(r["model"], r["author"], r["arm"])].append(r)
        keys = [k for k, g in groups.items() if any(x["text"].strip() for x in g)]
        full = luar_embed([[strict(x["text"]) for x in groups[k] if x["text"].strip()] for k in keys])
        cut = luar_embed([[cut_words(x["text"]) for x in groups[k] if x["text"].strip()] for k in keys])
        eps = []
        for k, ef, ec in zip(keys, full, cut):
            m, au, arm = k
            g = groups[k]
            vs, vl = scales(au)
            ep_txt = [x["text"] for x in g if x["text"].strip()]
            span = max(vs.ceiling - vs.floor, 1e-6)
            eps.append(
                dict(
                    model=m,
                    author=au,
                    arm=arm,
                    n=len(ep_txt),
                    position=(float(ef @ vs.profile) - vs.floor) / span,
                    position_loose=vl.position(ep_txt),
                    position80=(float(ec @ vs.profile) - vs.floor) / span,
                    spec=specificity(ef, au, split),
                    spec80=specificity(ec, au, split),
                    copy_flag=any(x["longest_run"] >= 12 or x["copy8"] >= 0.2 for x in g),
                )
            )
        (SCORED / (path.stem + ".json")).write_text(json.dumps(dict(drafts=rows, episodes=eps), indent=1))
        print(
            f"{path.stem}: {len(rows)} drafts, {len(eps)} episodes, strict {np.mean([e['position'] for e in eps]):+.3f}, "
            f"80w {np.mean([e['position80'] for e in eps]):+.3f}, spec {np.mean([e['spec'] for e in eps]):+.4f}"
        )


# ---------------------------------------------------------------- honesty (Episode 1's frozen extractor, unseen by the loop)
def honesty(files):
    B = B_by_id()
    for p in map(Path, files):
        rows = [r for r in read_jsonl(p) if r["text"].strip()]
        outs, _ = run_vllm(honesty_prompts(rows, B), f"honesty32b {p.stem}", model="Qwen/Qwen3-32B", max_tokens=600)
        res = {}
        for r, o in zip(rows, outs):
            v = parse(o)
            fl, rel = rule_flags(r["text"], B[r["brief"]])
            v["rule_flags"] = fl
            v["n_unsupported"] += len(fl)
            v["unsupported"] = (v["unsupported"] + fl)[:6]
            v["invented_relationship"] = v["invented_relationship"] or rel
            res[r["job"]] = v
        (DATA / "honesty").mkdir(parents=True, exist_ok=True)
        (DATA / "honesty" / f"{p.stem}.json").write_text(json.dumps(res, indent=1))
        share = sum(v["n_unsupported"] > 0 for v in res.values()) / max(1, len(res))
        print(p.stem, len(res), "checked; unsupported share", round(share, 3))


# ---------------------------------------------------------------- unseen ruler 1: character n-gram attributer (IER)
def _attr_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    recs = [a for a in authors()["authors"] if a["split"] == "heldout"]
    X, y = [], []
    for a in recs:
        for t in texts(a, "profile") + texts(a, "ceiling"):
            X.append(strict(t))
            y.append(a["author"])
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(C=20, max_iter=3000)
    clf.fit(vec.fit_transform(X), y)
    return vec, clf, recs


def attr_rank(vec, clf, ep_texts, au):
    lp = clf.predict_log_proba(vec.transform([strict(t) for t in ep_texts])).sum(0)
    order = np.argsort(-lp)
    return int(np.where(clf.classes_[order] == au)[0][0]) + 1


def attributer():
    vec, clf, recs = _attr_model()
    ranks = []
    for a in recs:
        s = texts(a, "samples")
        for i in range(3):
            ranks.append(attr_rank(vec, clf, s[4 * i : 4 * i + 4], a["author"]))
    top1 = float(np.mean([r == 1 for r in ranks]))
    out = dict(
        ruler="char 2-4-gram TF-IDF + logistic regression, trained on 30 held-out authors' profile+ceiling texts",
        pass_bar="top-1 >= 0.8 on own-sample episodes (3 x 4 samples per author, never trained on)",
        top1=round(top1, 4),
        mean_rank=round(float(np.mean(ranks)), 3),
        n_episodes=len(ranks),
        chance_top1=round(1 / len(recs), 4),
        passed=top1 >= 0.8,
    )
    per_arm = {}
    for p in sorted(SCORED.glob("*__heldout.json")):
        g = defaultdict(list)
        for r in json.loads(p.read_text())["drafts"]:
            if r["text"].strip():
                g[(r["arm"], r["author"])].append(r["text"])
        for (arm, au), ts in g.items():
            per_arm.setdefault(arm, {})[au] = attr_rank(vec, clf, ts, au)
    out["per_arm"] = {
        arm: dict(
            mean_rank=round(float(np.mean(list(v.values()))), 3),
            top1=round(float(np.mean([x == 1 for x in v.values()])), 4),
            top3=round(float(np.mean([x <= 3 for x in v.values()])), 4),
            n=len(v),
            ranks=v,
        )
        for arm, v in per_arm.items()
    }
    (RES / "ruler_attributer.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "per_arm"})
    for arm, v in sorted(out["per_arm"].items(), key=lambda kv: kv[1]["mean_rank"]):
        print(f"  {arm:24s} rank {v['mean_rank']:5.2f} top1 {v['top1']:.2f} top3 {v['top3']:.2f}")


# ---------------------------------------------------------------- unseen ruler 2: voice preference model (dev-only, simulated)
def _ctx(ts):
    return "Emails by this person: " + " ||| ".join(" ".join(strict(t).split()[:110]) for t in ts)


PREF_REP = 6  # kind-a pairs per own text (dev validation: 6 beat 2; results.tsv)
PREF_EPOCHS, PREF_LR = 5, 3e-5  # dev validation: 0.93 vs 0.88 (3 epochs, 2e-5)


def pref_pairs(names=None, rep=PREF_REP):
    """Training pairs from dev authors only. Kinds: a (own text vs another dev author's), b (simulated edit vs
    its draft, labelled simulated), c (own text vs a Qwen3-8B zero-shot draft)."""
    rnd = random.Random(20261121)
    dev = [a for a in authors()["authors"] if a["split"] == "dev" and (names is None or a["author"] in names)]
    sim = json.loads((EP1_DATA / "sim_edits.json").read_text())
    default = json.loads((DATA / "dev_k0_default.json").read_text())
    pairs = []
    for a in dev:
        others = [b for b in dev if b["author"] != a["author"]]
        own_test = texts(a, "profile") + texts(a, "ceiling")
        own_s = texts(a, "samples")
        for t in own_test:  # context from samples, win = a test text
            for _ in range(rep):
                o = rnd.choice(others)
                lose = rnd.choice(texts(o, "profile") + texts(o, "ceiling"))
                pairs.append(dict(kind="a", ctx=_ctx(rnd.sample(own_s, 2)), win=strict(t), lose=strict(lose)))
        for t in own_s:  # context from test texts, win = a sample
            for _ in range(max(1, rep // 2)):
                o = rnd.choice(others)
                lose = rnd.choice(texts(o, "samples"))
                pairs.append(dict(kind="a", ctx=_ctx(rnd.sample(own_test, 2)), win=strict(t), lose=strict(lose)))
        for d, e in sim.get(a["author"], []):
            for _ in range(2):
                pairs.append(dict(kind="b_simulated", ctx=_ctx(rnd.sample(own_s, 2)), win=strict(e), lose=strict(d)))
        for t in rnd.sample(own_test, 6):
            pairs.append(
                dict(kind="c", ctx=_ctx(rnd.sample(own_s, 2)), win=strict(t), lose=strict(rnd.choice(default)))
            )
    return pairs


def pref_eval_sets():
    """Held-out pass-bar pairs (held-out authors, never trained on): kind a and kind c."""
    rnd = random.Random(7)
    ho = [a for a in authors()["authors"] if a["split"] == "heldout"]
    k0 = defaultdict(list)
    for r in json.loads((EP1_DATA / "scored" / "Qwen3-8B__heldout.json").read_text())["drafts"]:
        if r["arm"] == "k0" and r["text"].strip():
            k0[r["author"]].append(r["text"])
    a_pairs, c_pairs = [], []
    for a in ho:
        others = [b for b in ho if b["author"] != a["author"]]
        ctx = _ctx(texts(a, "profile")[:2])
        for t in texts(a, "ceiling"):
            o = rnd.choice(others)
            a_pairs.append(dict(ctx=ctx, win=strict(t), lose=strict(rnd.choice(texts(o, "ceiling")))))
        for t, d in zip(texts(a, "ceiling")[:4], k0[a["author"]]):
            c_pairs.append(dict(ctx=ctx, win=strict(t), lose=strict(d)))
    return a_pairs, c_pairs


def pref_val():
    """Dev-only tuning of the preference model: train on 11 dev authors, validate kind-a pairs on the other 4."""
    import modal

    dev = [a for a in authors()["authors"] if a["split"] == "dev"]
    tr, va = [a["author"] for a in dev[:11]], dev[11:]
    rnd = random.Random(3)
    val = []
    for a in va:
        others = [b for b in va if b["author"] != a["author"]]
        for t in texts(a, "ceiling"):
            for _ in range(3):
                o = rnd.choice(others)
                val.append(
                    dict(ctx=_ctx(texts(a, "profile")[:2]), win=strict(t), lose=strict(rnd.choice(texts(o, "ceiling"))))
                )
    recipes = {"rep2_ep3": (2, 3, 2e-5), "rep6_ep3": (6, 3, 2e-5), "rep6_ep5_lr3e-5": (6, 5, 3e-5)}
    payloads = []
    for rep, ep, lr in recipes.values():
        sc = {
            "w": [dict(ctx=x["ctx"], text=x["win"]) for x in val],
            "l": [dict(ctx=x["ctx"], text=x["lose"]) for x in val],
        }
        payloads.append(dict(train=pref_pairs(tr, rep), score=sc, epochs=ep, lr=lr, max_len=512))
    t0 = time.time()
    res = list(modal.Function.from_name("slm-ep2", "train_pref").map(payloads))
    record_spend("modal", charge(sum(r["seconds"] + 60 for r in res)), "ep2 preference model dev validation x3")
    out = {}
    for name, r in zip(recipes, res):
        out[name] = round(float(np.mean(np.array(r["scores"]["w"]) > np.array(r["scores"]["l"]))), 4)
    print("dev validation kind-a accuracy (4 dev authors, 60 pairs):", out, f"{time.time() - t0:.0f}s")
    (RES / "pref_dev_validation.json").write_text(
        json.dumps(dict(recipes={k: list(v) for k, v in recipes.items()}, acc=out), indent=1)
    )


def pref():
    import modal

    train = pref_pairs()
    a_pairs, c_pairs = pref_eval_sets()
    A = A_by_name()
    drafts = []
    for p in sorted(SCORED.glob("*__heldout.json")):
        for r in json.loads(p.read_text())["drafts"]:
            if r["text"].strip():
                ctx = _ctx(texts(A[r["author"]], "profile")[:2])
                drafts.append(dict(arm=r["arm"], author=r["author"], brief=r["brief"], ctx=ctx, text=strict(r["text"])))
    score_sets = {
        "a_win": [dict(ctx=x["ctx"], text=x["win"]) for x in a_pairs],
        "a_lose": [dict(ctx=x["ctx"], text=x["lose"]) for x in a_pairs],
        "c_win": [dict(ctx=x["ctx"], text=x["win"]) for x in c_pairs],
        "c_lose": [dict(ctx=x["ctx"], text=x["lose"]) for x in c_pairs],
        "drafts": [dict(ctx=x["ctx"], text=x["text"]) for x in drafts],
    }
    t0 = time.time()
    payload = dict(train=train, score=score_sets, epochs=PREF_EPOCHS, lr=PREF_LR, max_len=512)
    r = modal.Function.from_name("slm-ep2", "train_pref").remote(payload)
    record_spend("modal", charge(time.time() - t0), f"ep2 preference model n_train={len(train)}")
    S = r["scores"]
    acc_a = float(np.mean(np.array(S["a_win"]) > np.array(S["a_lose"])))
    acc_c = float(np.mean(np.array(S["c_win"]) > np.array(S["c_lose"])))
    per = defaultdict(dict)
    for x, s in zip(drafts, S["drafts"]):
        per[x["arm"]][f"{x['author']}|{x['brief']}"] = s
    kinds = defaultdict(int)
    for p in train:
        kinds[p["kind"]] += 1
    out = dict(
        model_card=dict(
            name="voice preference model (Episode 2, small, simulated)",
            base="microsoft/deberta-v3-small cross-encoder, scalar head, Bradley-Terry loss, 5 epochs, lr 3e-5, 512 tokens",
            input="context (two of the author's emails, 110 words each, strict form) [SEP] candidate (strict form)",
            training_data=dict(kinds, authors="15 dev authors only (Enron); none of the 30 evaluation authors"),
            labels="no human preference labels: kinds a and c are authorship (own text preferred), kind b is SIMULATED edits",
            intended_use="an extra ruler the harness never optimised; not a model of what any person prefers",
            limits="15 training authors; simulated edits; Enron-style email only; strict text form hides layout",
        ),
        pass_bar="held-out kind-a pairwise accuracy >= 0.70",
        heldout_acc_a=round(acc_a, 4),
        heldout_acc_c=round(acc_c, 4),
        n_a=len(a_pairs),
        n_c=len(c_pairs),
        passed=acc_a >= 0.70,
        loss_curve=r["loss_curve"],
        draft_scores=per,
    )
    (RES / "ruler_pref.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("draft_scores", "loss_curve", "model_card")})


# ---------------------------------------------------------------- robustness (added after reading the held-out drafts)
PHONE = re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")


def degenerate(t):
    """Broken output the loop rule misses: a counting run ("1992 1993 1994 ..."), or text so repetitive that it
    compresses below 30% of its size. Added after reading Rung 4's drafts; labelled as such in the note."""
    w = t.split()
    if not w:
        return True
    digits = sum(x.strip(".,;:()").isdigit() for x in w) / len(w)
    ratio = len(zlib.compress(t.encode())) / max(1, len(t.encode()))
    return digits > 0.25 or (len(t) > 300 and ratio < 0.30)


def loop4(t):
    """Series loop rule (Taste Stack Episode 1): more than 20% of the draft's 4-grams are repeats."""
    w = t.lower().split()
    grams = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return bool(grams) and 1 - len(set(grams)) / len(grams) > 0.20


def broken(r):
    return r["loop"] or loop4(r["text"]) or degenerate(r["text"])


def robust():
    """Per arm: shares of looping (5-gram x3, and >20% repeated 4-grams), degenerate and phone-number drafts, and the
    strict position re-scored without any of them (an author is kept for an arm only with 2+ clean drafts)."""
    out = {}
    for p in sorted(SCORED.glob("*__heldout.json")):
        d = json.loads(p.read_text())
        eps = {(e["arm"], e["author"]): e for e in d["episodes"]}
        by = defaultdict(list)
        for r in d["drafts"]:
            if r["text"].strip():
                by[(r["arm"], r["author"])].append(r)
        keys, clean_eps = [], []
        for k, g in by.items():
            ok = [r["text"] for r in g if not broken(r)]
            if len(ok) >= 2:
                keys.append(k)
                clean_eps.append([strict(t) for t in ok])
        embs = luar_embed(clean_eps) if clean_eps else []
        clean_pos = defaultdict(dict)
        for k, e in zip(keys, embs):
            vs, _ = scales(k[1])
            clean_pos[k[0]][k[1]] = (float(e @ vs.profile) - vs.floor) / max(vs.ceiling - vs.floor, 1e-6)
        per_arm = defaultdict(list)
        for r in d["drafts"]:
            if r["text"].strip():
                per_arm[r["arm"]].append(r)
        for arm, rs in per_arm.items():
            cp = clean_pos.get(arm, {})
            full = [eps[(arm, au)]["position"] for au in cp]
            out[arm] = dict(
                loops=round(float(np.mean([r["loop"] for r in rs])), 3),
                degenerate=round(float(np.mean([degenerate(r["text"]) for r in rs])), 3),
                loop4=round(float(np.mean([loop4(r["text"]) for r in rs])), 3),
                loop_or_degenerate=round(float(np.mean([broken(r) for r in rs])), 3),
                phone_number=round(float(np.mean([bool(PHONE.search(r["text"])) for r in rs])), 3),
                clean_authors=len(cp),
                clean_position=round(float(np.mean(list(cp.values()))), 4) if cp else None,
                same_authors_as_scored=round(float(np.mean(full)), 4) if full else None,
                clean_by_author=cp,
            )
    (RES / "robustness.json").write_text(json.dumps(out, indent=1))
    for arm, v in out.items():
        print(
            f"{arm:20s} loops {v['loops']:.2f} loop4 {v['loop4']:.2f} broken {v['loop_or_degenerate']:.2f} degenerate {v['degenerate']:.2f} phone {v['phone_number']:.2f} "
            f"clean {v['clean_position']} (n={v['clean_authors']}; scored on same {v['same_authors_as_scored']})"
        )


# ---------------------------------------------------------------- persona test (a): projection on the assistant direction
def _rank_resid(y, X):
    from scipy.stats import rankdata

    yr = rankdata(y)
    Xr = np.column_stack([rankdata(c) for c in X] + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    return yr - Xr @ beta


def projection():
    import modal
    from scipy.stats import spearmanr

    ad = np.load(DATA / "assistant_dir.npz")
    u = ad["a"] / np.linalg.norm(ad["a"])
    rows, eps = [], []
    for p in sorted((EP1_DATA / "scored").glob("*__heldout.json")):
        d = json.loads(p.read_text())
        rows += [r for r in d["drafts"] if r["text"].strip() and r["model"] != "ctrl"]
        eps += [e for e in d["episodes"] if e["model"] != "ctrl"]
    names = [f"each:{i}" for i in range(0, len(rows), 400)]
    sets = {n: [r["text"] for r in rows[int(n[5:]) : int(n[5:]) + 400]] for n in names}
    fn = modal.Function.from_name("slm-ep2", "acts")
    results = list(fn.starmap([({n: sets[n]}, [24]) for n in names]))
    secs = sum(r["seconds"] + 60 for r in results)
    record_spend("modal", charge(secs), f"ep2 projection n={len(rows)}")
    proj = []
    for n, r in zip(names, results):
        proj += [float(np.dot(np.array(v), u)) for v in r["each"][n]["24"]]
    by = defaultdict(list)
    for r, x in zip(rows, proj):
        by[(r["model"], r["arm"], r["author"])].append((x, r["voi380"], r["words"]))
    P, Y, D, W, lab = [], [], [], [], []
    for e in eps:
        k = (e["model"], e["arm"], e["author"])
        if k not in by or e["position"] is None:
            continue
        v = np.array(by[k])
        P.append(v[:, 0].mean())
        D.append(v[:, 1].mean())
        W.append(v[:, 2].mean())
        Y.append(e["position"])
        lab.append(f"{e['model']}|{e['arm']}")
    P, Y, D, W = map(np.array, (P, Y, D, W))
    rho = float(spearmanr(P, Y).correlation)
    partial = float(np.corrcoef(_rank_resid(P, [D, W]), _rank_resid(Y, [D, W]))[0, 1])
    bs_r, bs_p = [], []
    for _ in range(500):
        i = RNG.integers(0, len(P), len(P))
        bs_r.append(spearmanr(P[i], Y[i]).correlation)
        bs_p.append(np.corrcoef(_rank_resid(P[i], [D[i], W[i]]), _rank_resid(Y[i], [D[i], W[i]]))[0, 1])
    out = dict(
        direction="Qwen3-8B layer 24: own zero-shot drafts on dev briefs minus dev authors' real emails (dev only)",
        n_episodes=len(P),
        n_drafts=len(rows),
        spearman_projection_vs_position=round(rho, 4),
        ci=[round(float(np.percentile(bs_r, 2.5)), 4), round(float(np.percentile(bs_r, 97.5)), 4)],
        partial_on_detector_and_words=round(partial, 4),
        partial_ci=[round(float(np.percentile(bs_p, 2.5)), 4), round(float(np.percentile(bs_p, 97.5)), 4)],
        spearman_detector_vs_position=round(float(spearmanr(D, Y).correlation), 4),
        spearman_projection_vs_detector=round(float(spearmanr(P, D).correlation), 4),
        points=[
            dict(p=round(float(a), 3), y=round(float(b), 3), d=round(float(c), 3), cell=lb)
            for a, b, c, lb in zip(P, Y, D, lab)
        ],
    )
    (RES / "projection.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "points"})


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "derive":
        derive(a[0])
    elif cmd == "comparators":
        comparators()
    elif cmd == "score":
        score(a[0], a[1:] or None)
    elif cmd == "honesty":
        honesty(a)
    elif cmd == "attributer":
        attributer()
    elif cmd == "pref":
        pref()
    elif cmd == "pref-val":
        pref_val()
    elif cmd == "projection":
        projection()
    elif cmd == "robust":
        robust()
    else:
        raise SystemExit(__doc__)
