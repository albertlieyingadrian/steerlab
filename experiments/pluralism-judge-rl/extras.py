"""
The judge-based measures, run after analyse.py has fixed each task's matched drift D*:

  jury favourites   every held-out reply of a checkpoint (own task: 100 creative or 64 advice prompts x 8 replies) is
                    scored by the task's 4 jurors; its favourite juror is the one whose score most exceeds that juror's
                    mean on the base model's replies. Spread = entropy of favourite counts / log 4 (1 = even, 0 = one
                    juror owns every reply). Base, and the final checkpoint of every jury and single-judge run.
  quality           a fixed subsample (the first 40 held-out prompts of the task x replies k = 0, 1) scored by the open
                    judge (phi-4) and the independent frontier judge (Claude Haiku 5.5, OpenRouter, $5 cap), at the
                    final and the drift-matched checkpoint (saved checkpoint nearest D*) of every run, plus base.
                    Their agreement on trained outputs is the Design 3 re-check.

    .venv/bin/python experiments/pluralism-judge-rl/extras.py [--config configs/full.json]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

import modal
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontier_judge  # noqa: E402
from analyse import checkpoints, run_meta  # noqa: E402
from budget import check_modal, job_cost  # noqa: E402
from common import CACHE, DATA, JURY, RESULTS, read_jsonl  # noqa: E402

N_QUALITY_PROMPTS = 40


def samples(tag):
    p = CACHE / "samples" / f"{tag}.jsonl.gz"
    return [json.loads(line) for line in gzip.open(p, "rt")] if p.exists() else None


def prompts(task):
    return {r["id"]: r["prompt"] for r in read_jsonl(DATA / f"{task}_heldout.jsonl")}


def jury_scores(judge, task, rows):
    P = prompts(task)
    jurors = list(JURY[task].items())
    items = [{"task": task, "prompt": P[r["id"]], "reply": r["text"], "persona": persona} for _, persona in jurors for r in rows]
    out = []
    for s in range(0, len(items), 1024):
        out += [x["score"] for x in judge.score.remote(items[s : s + 1024])]
    n = len(rows)
    return np.array([[out[j * n + i] for j in range(len(jurors))] for i in range(n)]), [name for name, _ in jurors]


def favourites(scores, base_means):
    fav = np.argmax(scores - base_means, axis=1)
    counts = Counter(int(f) for f in fav)
    p = np.array([counts.get(j, 0) for j in range(scores.shape[1])], dtype=float) / len(fav)
    h = -sum(x * math.log(x) for x in p if x > 0) / math.log(scores.shape[1])
    return float(h), p.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--tasks", default="creative,advice")
    args = ap.parse_args()
    TASKS = args.tasks.split(",")
    meta = run_meta(args.config)
    summary = json.loads((RESULTS / "summary.json").read_text())
    judge = modal.Cls.from_name("pluralism-judge-rl-judge", "Judge")(model="microsoft/phi-4")
    check_modal(job_cost("L40S", 1.5), "extras: jury favourites and open-judge quality")
    out_path = RESULTS / "extras.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {"jury_favourites": {}, "quality": {}}

    # ---- jury favourites
    base = samples("base-s0")
    for task in TASKS:
        brows = [r for r in base if r["task"] == task]
        bkey = f"base-s0|{task}"
        if bkey in out["jury_favourites"] and all(
            f"{r}-step{max(c['step'] for c in checkpoints(r))}|{task}" in out["jury_favourites"]
            for r, cfg in meta.items() if cfg.get("task") == task and cfg["arm"] in {"jury", "single"} and checkpoints(r)
        ):
            print(f"{task}: jury favourites already computed")
            continue
        bscores, names = jury_scores(judge, task, brows)
        bmeans = bscores.mean(axis=0)
        h, p = favourites(bscores, bmeans)
        out["jury_favourites"][f"base-s0|{task}"] = {"entropy": h, "share": p, "jurors": names, "juror_means": bmeans.tolist(), "n": len(brows)}
        print(f"base-s0|{task}: favourite spread {h:.3f} {np.round(p, 2).tolist()}")
        for run, cfg in meta.items():
            if cfg.get("task") != task or cfg["arm"] not in {"jury", "single"}:
                continue
            final = max(c["step"] for c in checkpoints(run))
            tag = f"{run}-step{final}"
            rows = [r for r in (samples(tag) or []) if r["task"] == task]
            if not rows:
                print(f"{tag}: no samples yet, skipped")
                continue
            sc, _ = jury_scores(judge, task, rows)
            h, p = favourites(sc, bmeans)
            out["jury_favourites"][f"{tag}|{task}"] = {"entropy": h, "share": p, "jurors": names, "juror_means": sc.mean(axis=0).tolist(), "n": len(rows)}
            print(f"{tag}: favourite spread {h:.3f} {np.round(p, 2).tolist()}")
        out_path.write_text(json.dumps(out, indent=1))

    # ---- exploratory (added after reading replies): cross-prompt similarity, a "house style" measure. Mean cosine
    # between first replies (k = 0) to DIFFERENT held-out prompts, same embedder as repetition. Higher = replies to
    # unrelated requests sound alike. Base and the final checkpoint of every run, own task (clean: both).
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    out.setdefault("cross_prompt", {})
    for task in TASKS:
        tags_cp = ["base-s0", "base-s1"] + [
            f"{run}-step{max(c['step'] for c in checkpoints(run))}"
            for run, cfg in meta.items()
            if cfg.get("task") in (task, None) and checkpoints(run)
        ]
        for tag in tags_cp:
            key = f"{tag}|{task}"
            rows = samples(tag)
            if rows is None or key in out["cross_prompt"]:
                continue
            first = [r["text"] for r in rows if r["task"] == task and r["k"] == 0]
            v = emb.encode(first, normalize_embeddings=True)
            sims = v @ v.T
            n = len(first)
            out["cross_prompt"][key] = float((sims.sum() - np.trace(sims)) / (n * (n - 1)))
            print(f"{key}: cross-prompt similarity {out['cross_prompt'][key]:.3f}")
    out_path.write_text(json.dumps(out, indent=1))

    # ---- quality at final and drift-matched checkpoints
    tags = []
    for task in TASKS:
        if task not in summary["matched"]:
            print(f"{task}: no matched drift yet, quality skipped")
            continue
        tags.append(("base-s0", task, "base"))
        D = summary["matched"][task]["D"]
        for run, cfg in meta.items():
            if cfg.get("task") not in (task, None):
                continue
            cks = [c for c in checkpoints(run) if c["step"] > 0]
            if not cks:
                continue
            final = max(cks, key=lambda c: c["step"])
            matched = min(cks, key=lambda c: abs(c["drift_kl"] - D))
            tags.append((f"{run}-step{matched['step']}", task, "matched"))
            if final["step"] != matched["step"]:
                tags.append((f"{run}-step{final['step']}", task, "final"))
    for tag, task, kind in tags:
        key = f"{tag}|{task}"
        if key in out["quality"]:
            continue
        rows = samples(tag)
        if rows is None:
            print(f"{tag}: not scored yet, skipped")
            continue
        P = prompts(task)
        keep_ids = set(list(P)[:N_QUALITY_PROMPTS])
        sub = [r for r in rows if r["task"] == task and r["id"] in keep_ids and r["k"] < 2]
        items = [{"task": task, "prompt": P[r["id"]], "reply": r["text"]} for r in sub]
        open_s = [x["score"] for x in judge.score.remote(items)]
        fr = frontier_judge.score(items, f"quality {tag} {task}")
        ok = [(a, b) for a, b in zip(open_s, fr) if b is not None]
        out["quality"][key] = {
            "kind": kind,
            "open": float(np.mean(open_s)),
            "frontier": float(np.mean([b for _, b in ok])),
            "open_items": open_s,
            "frontier_items": fr,
            "rho": float(spearmanr([a for a, _ in ok], [b for _, b in ok]).statistic) if len(ok) > 3 else None,
            "n": len(items),
            "n_frontier": len(ok),
        }
        q = out["quality"][key]
        print(f"{key} ({kind}): open {q['open']:.2f}, frontier {q['frontier']:.2f}, rho {q['rho']}")
        out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
