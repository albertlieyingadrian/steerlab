"""Normalise the three released human-judgement datasets (see README for sources and licences).

    python load_human.py

Reads data/raw/<dataset>/repo (cloned from each paper's GitHub release). Writes the texts to data/local/*.jsonl
(gitignored, never synced) and ids, human labels and splits to results/labels_h*.json (committed: no text).

H1 post-editing logs: per text, the writer's own 1-5 rating of how well it captures their style.
H2 style similarity triplets: per triplet, crowd votes for B vs C as closer in style to anchor A.
H3 human detectors: per article, five experts' human/AI guesses with 1-5 confidence.
"""

import ast
import json
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

from common4 import LOCAL, RAW, RES, SEED, dump, hid, write_jsonl


def words(t):
    return len(str(t).split())


def rating(likert, key):
    """A 1-5 Likert answer as an int, or None if unanswered."""
    v = likert.get(key)
    return int(v) if v not in (None, "") else None


# ---------------------------------------------------------------- H1
def h1():
    texts, rows = [], []
    files = sorted((RAW / "personal_style_postedit/repo/logs").glob("*.json"))
    pids = sorted(hid(f.stem) for f in files)
    dev = set(pids[:20])  # 20 dev participants: lowest sha1 of id (plan)
    for f in files:
        d = json.loads(f.read_text())
        pid = hid(f.stem)
        for k, v in d["responses"].items():
            L = v.get("likert")
            L = ast.literal_eval(L) if isinstance(L, str) else (L or {})
            base = dict(
                pid=pid,
                task=f"{pid}-{k}",
                scenario=v.get("scenario"),
                split="dev" if pid in dev else "heldout",
                voice_important=rating(L, "voice-important"),
            )
            treat = str(v.get("model_generation_shown")) == "1"
            final = (v.get("final_version") or "").strip()
            if treat:
                draft = (v.get("model_generation") or "").strip()
                if not draft or not final:
                    continue
                for kind, txt, cap, fr, us in (
                    (
                        "draft",
                        draft,
                        rating(L, "original-capture"),
                        rating(L, "original-friend"),
                        rating(L, "original-useable"),
                    ),
                    (
                        "edit",
                        final,
                        rating(L, "edited-capture"),
                        rating(L, "edited-friend"),
                        rating(L, "edited-useable"),
                    ),
                ):
                    tid = f"{base['task']}-{kind}"
                    texts.append(dict(tid=tid, text=txt))
                    rows.append(
                        dict(
                            base,
                            tid=tid,
                            kind=kind,
                            capture=cap,
                            friend=fr,
                            useable=us,
                            words=words(txt),
                            edit_type=L.get("edit_type"),
                            unchanged=draft == final,
                        )
                    )
            else:
                if not final:
                    continue
                tid = f"{base['task']}-control"
                texts.append(dict(tid=tid, text=final))
                rows.append(
                    dict(
                        base,
                        tid=tid,
                        kind="control",
                        capture=rating(L, "edited-capture"),
                        friend=rating(L, "edited-friend"),
                        useable=rating(L, "edited-useable"),
                        words=words(final),
                    )
                )
    write_jsonl(LOCAL / "h1_texts.jsonl", texts)
    dump(RES / "labels_h1.json", rows)
    n = defaultdict(int)
    for r in rows:
        n[(r["split"], r["kind"])] += 1
    print("H1", len(files), "participants;", dict(n))


# ---------------------------------------------------------------- H2
# Column positions from the dataset's data_processing_scripts/f8_column_definitions.py (the CSVs have no header).
C_UNIT, C_GOLDEN, C_WORKER, C_STYLE, C_E1, C_E2, C_E3, C_T1, C_T2, C_T3, C_GOLD = (
    0,
    2,
    9,
    19,
    22,
    23,
    24,
    31,
    32,
    33,
    29,
)


def h2():
    rng = np.random.default_rng(SEED)
    exc, trip = {}, []
    for corpus in (3, 6, 7):  # dev = corpus 3 (disjoint books); held-out = corpora 6 and 7 (plan)
        z = zipfile.ZipFile(RAW / f"style-dataset/repo/cleaned_data/cleaned_rows_anonymized_corpus_{corpus}.csv.zip")
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
        for _, g in df.groupby(C_UNIT):
            r0 = g.iloc[0]
            ids = [f"c{corpus}:{r0[c]}" for c in (C_E1, C_E2, C_E3)]
            for i, c in zip(ids, (C_T1, C_T2, C_T3)):
                exc.setdefault(i, str(r0[c]))
            votes = pd.to_numeric(g[C_STYLE], errors="coerce").dropna().astype(int)
            golden = str(r0[C_GOLDEN]).lower() == "true"
            gold = pd.to_numeric(pd.Series([r0[C_GOLD]]), errors="coerce").iloc[0]
            trip.append(
                dict(
                    tri=f"c{corpus}:{r0[C_UNIT]}",
                    corpus=corpus,
                    a=ids[0],
                    b=ids[1],
                    c=ids[2],
                    n=len(votes),
                    votes_b=int((votes == 2).sum()),
                    golden=golden,
                    gold=("b" if gold == 2 else "c" if gold == 3 else None),
                    split="dev" if corpus == 3 else "heldout",
                )
            )
    # the panel judges a seeded sample of triplets with >= 5 judgements (cost); LUAR and baselines see all
    for split, size in (("heldout", 1500), ("dev", 300)):
        pool = [t for t in trip if t["split"] == split and not t["golden"] and t["n"] >= 5]
        pick = set(rng.choice(len(pool), size=min(size, len(pool)), replace=False).tolist())
        for i, t in enumerate(pool):
            t["panel"] = i in pick
    write_jsonl(LOCAL / "h2_excerpts.jsonl", [dict(eid=k, text=v) for k, v in exc.items()])
    dump(RES / "labels_h2.json", trip)
    n5 = [t for t in trip if t["n"] >= 5 and not t["golden"]]
    print(
        "H2",
        len(trip),
        "triplets;",
        len(n5),
        "with >=5 judgements;",
        sum(t["golden"] for t in trip),
        "gold;",
        len(exc),
        "excerpts",
    )


# ---------------------------------------------------------------- H3
def h3():
    d = json.loads((RAW / "human_detectors/repo/human_detectors.json").read_text())
    rows, texts = [], []
    for k, x in d.items():
        guesses = []
        for i in range(1, 6):
            a = x.get(f"annotator_{i}") or {}
            a = ast.literal_eval(a) if isinstance(a, str) else a  # stored as a Python-dict string
            if a.get("guess"):
                guesses.append((int(a["guess"].lower().startswith("machine")), round(float(a.get("confidence") or 3))))
        aid = f"h3-{k}"
        texts.append(dict(aid=aid, text=x["article"]))
        # labels are "AI-generated" / "Human-written" (truth) and "Machine-Generated" / "Human-Generated" (guesses);
        # generation_model names the experiment batch (each batch pairs 30 human articles with 30 from that setting)
        rows.append(
            dict(
                aid=aid,
                prompt=str(x["prompt_id"]),
                truth=int(x["ground_truth"].lower().startswith("ai")),
                batch=x["generation_model"],
                votes_ai=sum(g for g, _ in guesses),
                n=len(guesses),
                signed=sum(c if g else -c for g, c in guesses),
                majority=int(str(x.get("expert_majority_vote", "")).lower().startswith("machine")),
                words=words(x["article"]),
            )
        )
    prompts = sorted({r["prompt"] for r in rows}, key=hid)
    dev = set(prompts[: max(1, len(prompts) // 5)])  # 6 of 30 prompts (60 articles) for dev
    for r in rows:
        r["split"] = "dev" if r["prompt"] in dev else "heldout"
    write_jsonl(LOCAL / "h3_texts.jsonl", texts)
    dump(RES / "labels_h3.json", rows)
    print(
        "H3",
        len(rows),
        "articles;",
        len(prompts),
        "prompts;",
        sum(r["split"] == "dev" for r in rows),
        "dev;",
        "truth AI",
        sum(r["truth"] for r in rows),
        "majority AI",
        sum(r["majority"] for r in rows),
    )


if __name__ == "__main__":
    h1()
    h2()
    h3()
