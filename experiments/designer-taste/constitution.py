"""
Taste constitutions by inverse Constitutional AI (arXiv 2406.06560) for The Taste Stack Episode 7.

Preference part of each track; per designer and training fold (A or B), and for two references: the cohort
("pooled", tested against the cohort's majority) and BRIGHT, a synthetic designer who always prefers the image with
the higher mean L* (the positive control, aesthetics track).
  1 propose   Qwen2.5-VL-72B sees the brief and both images, is told which one the designer chose, writes one
              principle "Prefer the design that ..." (<= 25 words)                    -> jobs/cst_propose.jsonl
  2 cluster   all-MiniLM-L6-v2 embeddings, k-means k = 16, the principle nearest each centre
  3 test      Qwen2.5-VL-7B: which image better follows each candidate, both orders, on 72 training pairs
                                                                                       -> jobs/cst_test.jsonl
  4 keep      the 5 candidates that agree most with the designer on those pairs
  5 evaluate  Qwen2.5-VL-72B judges the held-out fold's preference items with each constitution in the prompt,
              both orders                                                              -> jobs/cst_eval.jsonl

    python constitution.py propose | cluster | test | keep | evaluate | score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from judges import job, read
from taste import CACHE, HERE, load

JOBS = CACHE / "jobs"
STATE = CACHE / "constitution"
K, KEEP, N_TEST = 16, 5, 72
FOLDS = (("A", "B"), ("B", "A"))  # (train, test)
OUT = HERE / "results" / "constitutions.json"  # principles are model-written text about designs, no dataset text


def bright_of() -> dict:
    from formal import srgb_to_lab

    feat = pickle.loads((CACHE / "features.pkl").read_bytes())
    return {int(a): float(srgb_to_lab(f["thumb"].reshape(-1, 3))[:, 0].mean()) for a, f in feat["img"].items()}


def labelled_sets() -> dict:
    """{(who, train_fold): DataFrame of training pairs with y}; who = designer, 'pooled:<track>' or 'BRIGHT'."""
    d = load()
    p = d["pairs"]
    pref = p[p.dim == "preference"]
    out = {}
    for trf, _ in FOLDS:
        tr = pref[pref.split == trf]
        for des, g in tr.groupby("designer"):
            out[(des, trf)] = g.reset_index(drop=True)
        for track, g in tr.groupby("track"):
            maj = g.groupby(["scene", "a", "b"]).y.mean().reset_index()
            maj = maj[maj.y != 0.5].assign(y=lambda x: (x.y > 0.5).astype(int), track=track)
            out[(f"pooled:{track}", trf)] = maj
        b = bright_of()
        it = tr[tr.track == "aesthetics"][["scene", "a", "b"]].drop_duplicates()
        out[("BRIGHT", trf)] = it.assign(
            y=[int(b[x] > b[z]) for x, z in zip(it.a, it.b)], track="aesthetics"
        ).reset_index(drop=True)
    return out


def h(*x) -> int:
    return int(hashlib.sha256(":".join(map(str, x)).encode()).hexdigest()[:8], 16)


def propose() -> None:
    d = load()
    brief = dict(zip(d["scenes"].scene, d["scenes"].brief))
    jobs = []
    for (who, trf), g in labelled_sets().items():
        if who.startswith("pooled:"):
            continue  # the pooled constitution reuses the five designers' proposals
        for r in g.itertuples():
            swap = h("prop", who, trf, r.scene, r.a, r.b) % 2 == 1
            imgs = [r.b, r.a] if swap else [r.a, r.b]
            chosen = (2 if swap else 1) if r.y == 1 else (1 if swap else 2)
            pre = (
                f"A professional designer compared two candidate designs made for the same client brief and preferred "
                f"Image {chosen}.\n\nBrief:\n{brief[r.scene].strip()}\n"
            )
            ask = (
                "Write one principle that would explain this designer's choice, as a general rule about design that "
                'could apply to other briefs (do not mention this brief\'s content). Start with "Prefer the design that" '
                "and use at most 25 words. Reply with the principle only."
            )
            jobs.append(
                {"id": f"prop|{who}|{trf}|{r.scene}|{r.a}|{r.b}", "imgs": imgs, "pre": pre, "ask": ask, "mode": "gen"}
            )
    JOBS.mkdir(parents=True, exist_ok=True)
    (JOBS / "cst_propose.jsonl").write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(len(jobs), "proposal jobs")


def clean(t: str) -> str:
    t = t.strip().strip('"').split("\n")[0].strip()
    return t if t.lower().startswith("prefer the design that") else ""


def cluster() -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    R = read("qwen72")
    props = {}
    for k, r in R.items():
        f = k.split("|")
        if f[0] == "prop":
            t = clean(r["text"])
            if t:
                props.setdefault((f[1], f[2]), []).append(t)
    sets = labelled_sets()
    d = load()
    track_of = d["pairs"].drop_duplicates("designer").set_index("designer").track.to_dict()
    for who, trf in list(sets):
        if who.startswith("pooled:"):
            track = who.split(":")[1]
            props[(who, trf)] = [t for (w, f), v in props.items() if f == trf and track_of.get(w) == track for t in v]
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    cands = {}
    for key, texts in props.items():
        uniq = sorted(set(texts))
        E = enc.encode(uniq, normalize_embeddings=True, batch_size=64)
        km = KMeans(n_clusters=min(K, len(uniq)), n_init=10, random_state=0).fit(E)
        picks = []
        for c in range(km.n_clusters):
            idx = np.where(km.labels_ == c)[0]
            j = idx[np.argmax(E[idx] @ km.cluster_centers_[c])]
            picks.append({"text": uniq[j], "cluster_size": len(idx)})
        cands[f"{key[0]}|{key[1]}"] = {"n_proposals": len(texts), "candidates": picks}
        print(key, len(texts), "->", len(picks))
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "candidates.json").write_text(json.dumps(cands, indent=1))


def test_jobs() -> None:
    d = load()
    brief = dict(zip(d["scenes"].scene, d["scenes"].brief))
    cands = json.loads((STATE / "candidates.json").read_text())
    sets = labelled_sets()
    jobs = []
    for key, c in cands.items():
        who, trf = key.split("|")
        g = sets[(who, trf)]
        sample = g.sample(min(N_TEST, len(g)), random_state=h("test", who, trf) % 10000)
        for ci, cand in enumerate(c["candidates"]):
            for r in sample.itertuples():
                pre = f"Brief:\n{brief[r.scene].strip()}\n\nPrinciple: {cand['text']}\n"
                ask = "Which image better follows this principle? Answer with 1 or 2 only."
                for o, imgs in (("ab", [r.a, r.b]), ("ba", [r.b, r.a])):
                    jobs.append(
                        {
                            "id": f"test|{who}|{trf}|{ci}|{r.scene}|{r.a}|{r.b}|{o}",
                            "imgs": imgs,
                            "pre": pre,
                            "ask": ask,
                            "mode": "score",
                        }
                    )
    (JOBS / "cst_test.jsonl").write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(len(jobs), "test jobs")


def keep() -> None:
    R = read("qwen7")
    cands = json.loads((STATE / "candidates.json").read_text())
    sets = labelled_sets()
    acc = {}
    for k, r in R.items():
        f = k.split("|")
        if f[0] != "test":
            continue
        acc.setdefault((f[1], f[2], int(f[3]), int(f[4]), int(f[5]), int(f[6])), {})[f[7]] = r["p1"]
    out = {}
    for key, c in cands.items():
        who, trf = key.split("|")
        y = {(r.scene, r.a, r.b): r.y for r in sets[(who, trf)].itertuples()}
        scores = []
        for ci, cand in enumerate(c["candidates"]):
            hits = []
            for (w, f, i, s, a, b), o in acc.items():
                if w == who and f == trf and i == ci and "ab" in o and "ba" in o:
                    p = 0.5 * (o["ab"] + 1 - o["ba"])
                    hits.append(0.5 if p == 0.5 else float((p > 0.5) == (y[(s, a, b)] == 1)))
            scores.append({**cand, "train_agreement": float(np.mean(hits)) if hits else 0.0, "n": len(hits)})
        top = sorted(scores, key=lambda x: -x["train_agreement"])[:KEEP]
        out[key] = {"constitution": [t["text"] for t in top], "kept": top, "all": scores}
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "kept.json").write_text(json.dumps(out, indent=1))
    pub = {}
    for key, v in out.items():
        who, trf = key.split("|")
        pub[key] = {
            "principles": v["constitution"],
            "train_agreement": [round(t["train_agreement"], 3) for t in v["kept"]],
        }
    OUT.write_text(json.dumps(pub, indent=1))
    print(json.dumps(pub, indent=1)[:3000])


def cst_text(principles: list[str]) -> str:
    return "\nThis designer's taste, in their own principles:\n" + "\n".join(f"- {p}" for p in principles) + "\n"


def eval_jobs(prompt: str) -> None:
    d = load()
    pairs, sc = d["pairs"], d["scenes"]
    brief = dict(zip(sc.scene, sc.brief))
    kept = json.loads((STATE / "kept.json").read_text())
    pref = pairs[pairs.dim == "preference"]
    jobs = []
    for key, v in kept.items():
        who, trf = key.split("|")
        tef = dict(FOLDS)[trf]
        track = (
            "aesthetics"
            if who == "BRIGHT"
            else (who.split(":")[1] if who.startswith("pooled:") else pref[pref.designer == who].track.iloc[0])
        )
        it = pref[(pref.split == tef) & (pref.track == track)][["scene", "a", "b"]].drop_duplicates()
        extra = cst_text(v["constitution"])
        for r in it.itertuples():
            for o, imgs in (("ab", [r.a, r.b]), ("ba", [r.b, r.a])):
                jb = job(
                    f"cev|{who}|{trf}|{track}|{r.scene}|{r.a}|{r.b}|{o}",
                    prompt,
                    brief[r.scene],
                    "preference",
                    imgs,
                    extra,
                )
                jobs.append(jb)
    (JOBS / "cst_eval.jsonl").write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(len(jobs), "eval jobs")


def score() -> dict:
    """P(a) per (constitution owner, test item); returns a frame for analyse.py."""
    R = read("qwen72")
    acc = {}
    for k, r in R.items():
        f = k.split("|")
        if f[0] != "cev":
            continue
        acc.setdefault((f[1], f[2], f[3], int(f[4]), int(f[5]), int(f[6])), {})[f[7]] = r["p1"]
    rows = [
        {"owner": w, "train_fold": tf, "track": t, "scene": s, "a": a, "b": b, "p": 0.5 * (o["ab"] + 1 - o["ba"])}
        for (w, tf, t, s, a, b), o in acc.items()
        if "ab" in o and "ba" in o
    ]
    df = pd.DataFrame(rows)
    df.to_pickle(CACHE / "cst_eval.pkl")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["propose", "cluster", "test", "keep", "evaluate", "score"])
    ap.add_argument("--prompt", default="")
    a = ap.parse_args()
    if a.cmd == "propose":
        propose()
    elif a.cmd == "cluster":
        cluster()
    elif a.cmd == "test":
        test_jobs()
    elif a.cmd == "keep":
        keep()
    elif a.cmd == "evaluate":
        prompt = a.prompt or json.loads((HERE / "results" / "judge_prompt_choice.json").read_text())["qwen72"]
        eval_jobs(prompt)
    else:
        print(score().groupby("owner").size())
