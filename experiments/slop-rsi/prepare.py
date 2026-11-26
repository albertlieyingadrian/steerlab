"""Step 1 (FIXED): build the read-only pools the recipe may use, the decontamination masks, and the calibration file.

    python prepare.py pools     # data/ro/*: mining pool (+ round-0 scores), mirrors, data-v2b replay, unlabeled pool
    python prepare.py decontam  # data/private/decontam_<pool>.npy: items sharing >= 3 ten-word runs with any eval text
    python prepare.py upload    # the calibration file (2,000 unlabeled pages) to the volume; checks the eval file hash

Pilot 2 (plan amendment 3, P2-1):
    python prepare.py candidates   # data/cand_p2.jsonl: HC3 + RAID candidates (N_q up to 3,000), decontaminated vs pools
    python prepare.py freeze 1000  # data/evals_p2.jsonl + results/pilot2/evalset_p2.json at N_q; pool masks rebuilt
    python prepare.py upload_p2    # eval p2 to the volume (rsi/p2/evals.jsonl); hash recorded

Everything here comes from the hard-negative note (its pools, its round-0 pool scores, its AI mirrors, its frozen
evaluation file) and from the slop-detector note (data v2b). Nothing new is generated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from rsi import (
    DATA,
    EVAL_FILE,
    HN,
    HN_DATA,
    MANIFEST,
    PRIVATE,
    PROFILE,
    RESULTS,
    RO,
    V2B,
    VENV,
    VOLP,
    read_jsonl,
    sha,
    write_jsonl,
)

MODAL = str(VENV / "modal")
VOLUME = "slop-detector"
N_CAL = 2000  # unlabeled pages scored after every update, for cutoff fitting only

POOLS_README = """# Pools the recipe may read (read-only)

Every training row a recipe returns is a reference into one of these pools: {{"pool", "idx", "bucket", "weight",
"aug"}}. The harness builds the text itself, so a row can only ever be a pool item.

| Pool | File | Items | What it is |
|---|---|---|---|
| mining | mining.jsonl, mining_scores_r0.npy | 250,000 | FineWeb pages crawled 2017-01 .. 2022-10 (8 dumps), so written by people by the label rule. Fields: id, host, dump, date, words, text. `mining_scores_r0.npy[i]` is the starting detector's score for page i (0 = human .. 1 = AI). The starting detector flags {flagged:,} of them at its cut 0.3986 |
| mirrors | mirrors.jsonl | {n_mirrors} | AI rewrites of mining pages (a summary of the page, then a new page written from the summary by one of 7 open models). Fields: mining_idx (the source page), model, text |
| replay | replay.jsonl | {n_replay} | The starting detector's own training data (data v2b): human text and AI text of four degrees. Fields: kind, source, bucket (0 human, 1 light edit, 2 heavy edit, 3 fully AI), band, weight, target, text |
| unlabeled | unlabeled.jsonl | {n_unlab} | Other FineWeb pages crawled before 2022-11 from hosts in no other pool; unlabeled human text. Fields as mining. The first 2,000 are scored by every updated detector and passed to fit_cutoff as cal["unlabeled"] |

Buckets: the detector has a 4-way head (0 human, 1 light AI edit, 2 heavy AI edit, 3 fully AI); its score is the
expected bucket / 3. Default weights in data v2b: human 1.3278, fully AI 1.3092.
"""

_WS = re.compile(r"\s+")


def pools() -> None:
    RO.mkdir(parents=True, exist_ok=True)
    pool = read_jsonl(HN_DATA / "pools" / "pool.jsonl")
    keep = ("id", "host", "dump", "date", "words", "text")
    write_jsonl(RO / "mining.jsonl", [{k: p[k] for k in keep} for p in pool])
    s = np.load(HN_DATA / "pool_scores" / "pool_r0.npy").astype(np.float32)
    assert len(s) == len(pool), (len(s), len(pool))
    np.save(RO / "mining_scores_r0.npy", s)
    idx = {p["id"]: i for i, p in enumerate(pool)}
    mirrors, seen = [], set()
    for k in (1, 2, 3):
        for m in read_jsonl(HN_DATA / "mirrors" / f"b_r{k}.jsonl"):
            if m["doc_id"] in idx and m["key"] not in seen and m["text"].strip():
                seen.add(m["key"])
                mirrors.append({"mining_idx": idx[m["doc_id"]], "model": m["model"], "text": m["text"]})
    write_jsonl(RO / "mirrors.jsonl", mirrors)
    replay = [
        {k: r.get(k) for k in ("kind", "source", "bucket", "band", "weight", "target", "text")}
        for r in read_jsonl(V2B / "train.jsonl")
    ]
    write_jsonl(RO / "replay.jsonl", replay)
    unlab = read_jsonl(HN_DATA / "pools" / "dev.jsonl")
    write_jsonl(RO / "unlabeled.jsonl", [{k: p[k] for k in keep} for p in unlab])
    flagged = int((s >= 0.3986).sum())
    (RO / "README.md").write_text(
        POOLS_README.format(
            flagged=flagged, n_mirrors=f"{len(mirrors):,}", n_replay=f"{len(replay):,}", n_unlab=f"{len(unlab):,}"
        )
    )
    write_jsonl(DATA / "cal_unlabeled.jsonl", [{"text": p["text"]} for p in unlab[:N_CAL]])
    # data v2b's validation split: its kinds and buckets go to fit_cutoff (the detector's own validation labels, not
    # an evaluation set's)
    val = read_jsonl(V2B / "val.jsonl")
    (DATA / "cal_v2b_val_meta.json").write_text(
        json.dumps({"kind": [r["kind"] for r in val], "bucket": [r["bucket"] for r in val]}) + "\n"
    )
    summary = {
        "mining": len(pool),
        "mining_flagged_r0": flagged,
        "mirrors": len(mirrors),
        "replay": len(replay),
        "unlabeled": len(unlab),
        "cal_unlabeled": N_CAL,
        "v2b_val": len(val),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pools.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(summary)


def _shingles(text: str, n: int = 10) -> np.ndarray:
    w = _WS.sub(" ", text.lower()).split()[:450]  # the detector reads about the first 380 words
    if len(w) < n:
        return np.zeros(0, dtype=np.int64)
    h = np.array([hash(x) for x in w], dtype=np.int64)
    out = np.zeros(len(w) - n + 1, dtype=np.int64)
    for j in range(n):
        out = out * 1000003 + h[j : len(w) - n + 1 + j]
    return out


def decontam() -> None:
    """Flag pool items that share at least three ten-word runs with any evaluation text (any tier)."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    ev = read_jsonl(HN_DATA / "evals.jsonl")
    allsh = np.unique(np.concatenate([_shingles(r["text"]) for r in ev]))
    print(f"eval shingles: {len(allsh):,} from {len(ev):,} texts", flush=True)
    out = {}
    for name in ("mining", "mirrors", "replay", "unlabeled"):
        rows = read_jsonl(RO / f"{name}.jsonl")
        bad = np.zeros(len(rows), dtype=bool)
        for i, r in enumerate(rows):
            sh = _shingles(r["text"])
            if len(sh) and (allsh[np.minimum(np.searchsorted(allsh, sh), len(allsh) - 1)] == sh).sum() >= 3:
                bad[i] = True
        np.save(PRIVATE / f"decontam_{name}.npy", bad)
        out[name] = {"n": len(rows), "flagged": int(bad.sum())}
        print(name, out[name], flush=True)
    (RESULTS / "decontam.json").write_text(json.dumps(out, indent=1) + "\n")


def upload() -> None:
    env = {**os.environ, "MODAL_PROFILE": PROFILE}
    subprocess.run(
        [MODAL, "volume", "put", "--force", VOLUME, str(DATA / "cal_unlabeled.jsonl"), "rsi/cal_unlabeled.jsonl"],
        check=True,
        env=env,
        capture_output=True,
    )
    tmp = DATA / "_vol_evals.jsonl"
    subprocess.run(
        [MODAL, "volume", "get", "--force", VOLUME, "hn/evals.jsonl", str(tmp)],
        check=True,
        env=env,
        capture_output=True,
    )
    a, b = sha(tmp), sha(HN_DATA / "evals.jsonl")
    tmp.unlink()
    print("volume hn/evals.jsonl matches the local evaluation file:", a == b)
    if a != b:
        raise SystemExit("evaluation file on the volume differs from the local one")
    (RESULTS / "evals_sha256.txt").write_text(a + "\n")


# ------------------------------------------------------------------ pilot 2: eval p2 (plan amendment 3, P2-1)
STEER = Path.home() / ".cache" / "steerlab"
HC3_FILE = STEER / "slop-rules" / "hc3" / "all.jsonl"  # Hello-SimpleAI/HC3, CC BY-SA 4.0
RAID_FILE = STEER / "editlens" / "data" / "raid_10k.csv"  # RAID 10k sample shipped with EditLens, MIT
CAND = DATA / "cand_p2.jsonl"
NQ_MAX = 3000
N_RAID_AI = 1200
N_WEB = 6000
MIN_WORDS = 50
_DETOK = [  # the slop-rules note's HC3 detokeniser
    (r" n't\b", "n't"),
    (r" '(s|re|ve|ll|d|m)\b", r"'\1"),
    (r" ([,.;:!?%)])", r"\1"),
    (r"\( ", "("),
    (r"\$ (?=\d)", "$"),
    (r'" ([^"]*?) "', r'"\1"'),
    (r"[ \t]+", " "),
]


def detok(text: str) -> str:
    for a, b in _DETOK:
        text = re.sub(a, b, text)
    return text.strip()


def _u(key: str) -> float:
    return int(hashlib.sha256(f"p2|{key}".encode()).hexdigest()[:12], 16) / 16**12


def _pool_shingles_hit(cand_sh: np.ndarray) -> np.ndarray:
    """Candidate shingles that occur in some pool item sharing >= 3 shingles with the candidate set."""
    hit = []
    for name in ("mining", "mirrors", "replay", "unlabeled"):
        with (RO / f"{name}.jsonl").open() as f:
            for line in f:
                sh = _shingles(json.loads(line)["text"])
                if not len(sh):
                    continue
                m = sh[cand_sh[np.minimum(np.searchsorted(cand_sh, sh), len(cand_sh) - 1)] == sh]
                if len(m) >= 3:
                    hit.append(m)
        print("scanned", name, flush=True)
    return np.unique(np.concatenate(hit)) if hit else np.zeros(0, dtype=np.int64)


def candidates() -> None:
    import pandas as pd

    rows = []
    by_src: dict = {}
    for line in HC3_FILE.read_text().splitlines():
        r = json.loads(line)
        if not r["human_answers"] or not r["chatgpt_answers"]:
            continue
        h, a = detok(r["human_answers"][0]), detok(r["chatgpt_answers"][0])
        if len(h.split()) < MIN_WORDS or len(a.split()) < MIN_WORDS:
            continue
        qid = hashlib.sha256(r["question"].encode()).hexdigest()[:16]
        by_src.setdefault(r["source"], {})[qid] = (h, a)
    for src, qs in sorted(by_src.items()):
        order = sorted(qs, key=lambda q: _u(f"hc3|{q}"))[:NQ_MAX]
        for rank, q in enumerate(order):
            h, a = qs[q]
            rows.append({"set": "hc3", "id": q, "kind": "human", "y": 0, "source": src, "rank": rank, "text": h})
            rows.append({"set": "hc3", "id": q, "kind": "chatgpt", "y": 1, "source": src, "rank": rank, "text": a})
    df = pd.read_csv(RAID_FILE)
    df = df[df.text.astype(str).str.split().str.len() >= MIN_WORDS]
    ai = []
    for t in df.itertuples():
        text = str(t.text).strip()
        tid = hashlib.sha256(text.encode()).hexdigest()[:16]
        row = {"set": "raid", "id": tid, "kind": t.model, "y": int(t.model != "human"), "source": t.domain, "rank": 0}
        row["text"] = text
        (ai if row["y"] else rows).append(row)
    ai.sort(key=lambda r: _u(f"raid|{r['id']}"))
    rows += ai[:N_RAID_AI]
    seen, uniq = set(), []
    for r in rows:  # exact duplicates (a text twice) are kept once
        k = (r["set"], r["kind"], r["text"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    sh = [_shingles(r["text"]) for r in rows]
    allsh = np.unique(np.concatenate([x for x in sh if len(x)]))
    hit = _pool_shingles_hit(allsh)
    keep, dropped = [], {}
    for r, x in zip(rows, sh):
        n = int(np.isin(x, hit).sum()) if len(hit) and len(x) else 0
        if n >= 3:
            dropped[f"{r['set']}:{r['y']}"] = dropped.get(f"{r['set']}:{r['y']}", 0) + 1
        else:
            keep.append(r)
    write_jsonl(CAND, keep)
    counts: dict = {}
    for r in keep:
        k = f"{r['set']}|{r['source']}|{'ai' if r['y'] else 'human'}"
        counts[k] = counts.get(k, 0) + 1
    out = {"kept": len(keep), "dropped_pool_overlap": dropped, "counts": counts}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "candidates_p2.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def freeze(nq: int) -> None:
    """Eval p2 = the old sets (web cut to N_WEB by hash) + candidates with rank < nq; rebuild pool masks (union)."""
    old = read_jsonl(HN_DATA / "evals.jsonl")
    man_old = json.loads((HN / "results" / "evalset.json").read_text())
    web_ids = man_old["web"]["id"]
    keep_web = set(sorted(range(len(web_ids)), key=lambda i: _u(f"web|{web_ids[i]}"))[:N_WEB])
    texts, man = [], {}
    for s, m in man_old.items():
        rows = [r for r in old if r["set"] == s]
        assert len(rows) == len(m["y"]), s
        for i, r in enumerate(rows):
            if s == "web" and i not in keep_web:
                continue
            e = man.setdefault(s, {"y": [], "kind": [], "id": [], "source": []})
            e["y"].append(m["y"][i])
            e["kind"].append(m["kind"][i])
            e["id"].append(m["id"][i])
            e["source"].append(s)
            texts.append({"set": s, "text": r["text"]})
    for r in read_jsonl(CAND):
        if r["set"] == "hc3" and r["rank"] >= nq:
            continue
        e = man.setdefault(r["set"], {"y": [], "kind": [], "id": [], "source": []})
        for k in ("y", "kind", "id", "source"):
            e[k].append(r[k])
        texts.append({"set": r["set"], "text": r["text"]})
    write_jsonl(EVAL_FILE, texts)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"nq": nq, **man}) + "\n")
    counts = {s: {"n": len(m["y"]), "ai": int(sum(m["y"]))} for s, m in man.items()}
    print(json.dumps(counts))
    # pool masks: the pilot-1 masks OR overlap with any eval-p2 text, so no recipe can train on an eval-p2 text
    allsh = np.unique(np.concatenate([x for x in (_shingles(t["text"]) for t in texts) if len(x)]))
    out = {}
    for name in ("mining", "mirrors", "replay", "unlabeled"):
        oldm = np.load(PRIVATE / f"decontam_{name}.npy")
        new = np.zeros_like(oldm)
        with (RO / f"{name}.jsonl").open() as f:
            for i, line in enumerate(f):
                sh = _shingles(json.loads(line)["text"])
                if len(sh) and (allsh[np.minimum(np.searchsorted(allsh, sh), len(allsh) - 1)] == sh).sum() >= 3:
                    new[i] = True
        np.save(PRIVATE / f"decontam_{name}.npy", oldm | new)
        out[name] = {"n": len(oldm), "pilot1_flagged": int(oldm.sum()), "p2_flagged": int(new.sum()),
                     "union": int((oldm | new).sum()), "newly_flagged": int((new & ~oldm).sum())}  # fmt: skip
        print(name, out[name], flush=True)
    (RESULTS / "decontam_p2.json").write_text(json.dumps({"eval_counts": counts, "pools": out}, indent=1) + "\n")


def upload_p2() -> None:
    env = {**os.environ, "MODAL_PROFILE": PROFILE}
    subprocess.run(
        [MODAL, "volume", "put", "--force", VOLUME, str(EVAL_FILE), f"{VOLP}/evals.jsonl"],
        check=True,
        env=env,
        capture_output=True,
    )
    subprocess.run(
        [MODAL, "volume", "put", "--force", VOLUME, str(CAND), f"{VOLP}/cand.jsonl"],
        check=True,
        env=env,
        capture_output=True,
    )
    (RESULTS / "evals_p2_sha256.txt").write_text(sha(EVAL_FILE) + "\n")
    print("uploaded", sha(EVAL_FILE))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "freeze":
        freeze(int(sys.argv[2]))
    else:
        {"pools": pools, "decontam": decontam, "upload": upload, "candidates": candidates, "upload_p2": upload_p2}[
            cmd
        ]()
