"""Turn the lens work in a clone of the Looped Lens Challenge repo into this note's compact `results/`.

    python experiments/looped-lens/build_results.py <challenge clone> [--cv <cv json from an earlier run>]

The clone holds the full lens bank (top-20 token ids of every lens at all 48 visits for 867 questions, 64 MB) under
`dev/runs/`, made on Modal by `lab/modal_lens.py`. This script keeps what every number in the note needs:

- `questions.jsonl`: the 867 self-written questions (prompt, model input, read position, options, intermediate,
  family) and the split each is in.
- `votes.json.gz`: for every (lens, visit), the option each question's readout votes for under the fuzzy and the exact
  matcher (one character per question: "0"-"4", or "." for no vote). `cv.py` and `make_figures.py` score from this.
- `examples.json`: the cleaned top-10 tokens of six lenses at all 48 visits for a few questions (the note's explorer).
- `ndif.json`: the two end-to-end NDIF runs of the entry, per question, beside the offline copy.
- `training/`, `wsp/`, `judge_round2.json`, `routing_stats.json`: the translators' fit logs, the
  weight-space-programming prompt, reply and designs, the round-2 judge check, and the routing numbers the design
  prompt was built from.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
EXAMPLE_IDS = ["athlete-05"]  # plus one question per family, picked below
EXAMPLE_LENSES = ["act_in", "mix_in", "mlp_in", "tuned_mlp_in", "future_mlp_in", "concept_mlp_in"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clone", type=Path)
    ap.add_argument(
        "--cv", type=Path, help="per-question CV outcomes written by an earlier cv.py run, kept for checking"
    )
    a = ap.parse_args()
    clone = a.clone.resolve()
    sys.path.insert(0, str(clone / "autoresearch"))
    sys.path.insert(0, str(clone / "dev"))
    import prepare  # the clone's fixed harness: questions, lens bank, vote tables
    import score

    OUT.mkdir(exist_ok=True)
    qids = sorted(prepare.QS)
    split = {q: s for s in ("dev", "heldout") for q in prepare.SPLITS[s]}
    # round 2 wrote the first 327 (v1's visits were picked on them); round 3 added 540 harder ones
    first = {json.loads(line)["id"] for line in open(clone / "dev" / "data" / "all_private_v2.jsonl") if line.strip()}
    with (OUT / "questions.jsonl").open("w") as f:
        for qid in qids:
            q = dict(prepare.QS[qid])
            q["split"] = split[qid]
            q["round"] = 2 if qid in first else 3
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    def column(table, key):
        col = table[key]
        return "".join("." if col.get(q) is None else str(col[q]) for q in qids)

    keys = list(prepare.TABLE)  # insertion order, which the hill-climb's candidate list follows
    votes = {
        "qids": qids,
        "keys": [f"{lens}|{v}" for lens, v in keys],
        "fuzzy": {f"{lens}|{v}": column(prepare.TABLE, (lens, v)) for lens, v in keys},
        "exact": {f"{lens}|{v}": column(prepare.TABLE_EXACT, (lens, v)) for lens, v in prepare.TABLE_EXACT},
    }
    with gzip.open(OUT / "votes.json.gz", "wt") as f:
        json.dump(votes, f, separators=(",", ":"))

    # examples: one question per family (the first by id that the v2 entry passes, so the explorer shows a working read,
    # plus the first it fails, so it shows a failure), and the hook question
    fams: dict[str, list[str]] = {}
    for qid in qids:
        fams.setdefault(prepare.QS[qid]["family"], []).append(qid)
    v2 = load_module("v2", clone / "autoresearch" / "entry_v2_picks.py").PICKS

    def passes(qid):
        q = prepare.QS[qid]
        return score.plurality([prepare.TABLE[p].get(qid) for p in v2], q["options"].index(q["intermediate"]))

    chosen = list(EXAMPLE_IDS)
    for fam in ("multihop", "multilingual", "typo", "practice_like", "arith_inter"):
        ok = [q for q in fams[fam] if passes(q)]
        bad = [q for q in fams[fam] if not passes(q)]
        chosen += ok[:1] + (bad[:1] if fam in ("arith_inter", "practice_like") else [])
    examples = []
    for qid in chosen:
        q = prepare.QS[qid]
        reads = {}
        for lens in EXAMPLE_LENSES:
            row = {}
            for v in range(48):
                ids = prepare.RUNS[qid].get(v, {}).get(lens)
                if ids is None:
                    continue
                toks = score.clean(ids)
                row[v] = {"tokens": toks, "vote": prepare.TABLE[(lens, v)].get(qid)}
            reads[lens] = row
        examples.append(
            {
                "id": qid,
                "family": q["family"],
                "prompt": q["prompt"],
                "options": q["options"],
                "intermediate": q["intermediate"],
                "answer": q["answer"],
                "passes_v2": bool(passes(qid)),
                "reads": reads,
            }
        )
    (OUT / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))

    # NDIF runs: per question, NDIF pass and offline pass, plus top-1 agreement per readout
    ndif = {}
    for name, folder, picks_file in [
        ("v1_heldout112", "ndif_heldout", "entry_v1_picks.py"),
        ("v2_newheldout191", "ndif_new_heldout", "entry_v2_picks.py"),
    ]:
        picks = load_module(name, clone / "autoresearch" / picks_file).PICKS
        lens_at = {v: lens for lens, v in picks}
        rows = [json.loads(line) for line in open(clone / "dev" / "data" / folder / "out" / "readouts.jsonl")]
        per_q = []
        for r in rows:
            q = prepare.QS[r["id"]]
            gold = q["options"].index(q["intermediate"])
            votes_n, top1 = [], []
            for ro in r["readouts"]:
                bank = score.clean(prepare.RUNS[r["id"]][ro["layer"]][lens_at[ro["layer"]]])
                top1.append(bool(ro["tokens"]) and bool(bank) and ro["tokens"][0] == bank[0])
                votes_n.append(score.vote(ro["tokens"], q["options"]))
            offline = score.plurality([prepare.TABLE[p].get(r["id"]) for p in picks], gold)
            per_q.append(
                {
                    "id": r["id"],
                    "ndif_pass": bool(score.plurality(votes_n, gold)),
                    "offline_pass": bool(offline),
                    "top1_agree": sum(top1),
                    "readouts": len(top1),
                }
            )
        ndif[name] = {"picks": picks, "questions": per_q}
    (OUT / "ndif.json").write_text(json.dumps(ndif, indent=1))

    (OUT / "training").mkdir(exist_ok=True)
    for name in ("tuned_v1_log.json", "future_v1_log.json", "future_v2_log.json", "concept_v1_log.json"):
        shutil.copy(clone / "dev" / "runs" / name, OUT / "training" / name)
    shutil.copy(clone / "dev" / "runs" / "routing_stats.json", OUT / "routing_stats.json")
    (OUT / "wsp").mkdir(exist_ok=True)
    for name in ("designs.json", "prompt.txt", "reply.txt"):
        shutil.copy(clone / "dev" / "wsp" / name, OUT / "wsp" / name)
    jdir = clone / "dev" / "judge_runs" / "final_picks-all"
    card = json.load(open(jdir / "score_card.json"))
    (OUT / "judge_round2.json").write_text(json.dumps(card, indent=1))
    # the judge's verdict per question beside the fuzzy and exact matchers, on the exact tokens the judge was shown
    subs = {r["id"]: r for r in (json.loads(line) for line in open(jdir / "submissions.jsonl"))}
    rows = []
    for line in open(jdir / "results.jsonl"):
        r = json.loads(line)
        q = prepare.QS[r["id"]]
        gold = q["options"].index(q["intermediate"])
        reads = subs[r["id"]]["readouts"]
        fz = [score.vote(ro["tokens"], q["options"], "fuzzy") for ro in reads]
        ex = [score.vote(ro["tokens"], q["options"], "exact") for ro in reads]
        jv = {int(k): q["options"].index(v) if v in q["options"] else None for k, v in r.get("layer_votes", {}).items()}
        layer_agree = sum((jv.get(ro["layer"]) == f) for ro, f in zip(reads, fz))
        rows.append(
            {
                "id": r["id"],
                "family": q["family"],
                "judge_pass": bool(r["pass"]),
                "fuzzy_pass": bool(score.plurality(fz, gold)),
                "exact_pass": bool(score.plurality(ex, gold)),
                "layer_votes_agree": layer_agree,
                "layers": len(reads),
            }
        )
    (OUT / "judge_check.json").write_text(
        json.dumps({"judge_model": card["run"]["judge_model"], "entry": "v1", "questions": rows}, indent=1)
    )
    if a.cv:
        shutil.copy(a.cv, OUT / "cv_check.json")
    print(f"wrote {OUT}: {len(qids)} questions, {len(keys)} (lens, visit) columns, {len(examples)} examples")


if __name__ == "__main__":
    main()
