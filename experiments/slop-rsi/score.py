"""Pilot 2 (FIXED): rescore pilot 1's three R0 adapters without training (plan amendment 3, P2-1).

    python score.py cand   # the eval-p2 candidates (sizing gate S, reference check gate H): data/private/cand_r0.json
    python score.py ref    # eval p2 + data v2b val + the 2,000 calibration pages: data/private/scores/r0_s{k}.json

One spawned call on the deployed app; the call id is saved before waiting; the spend is reserved, then settled.
"""

from __future__ import annotations

import json
import sys
import time

import modal

from rsi import APP, PRIVATE, RESULTS, VOLP, reserve, settle

R0 = [f"rsi/runs/r0_s{k}/adapter" for k in range(3)]


def run(tag: str, files: list[str]) -> dict:
    p = RESULTS / "calls" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    fn = modal.Function.from_name(APP, "score_runs")
    t0 = time.time()
    if p.exists():
        call = modal.FunctionCall.from_id(json.loads(p.read_text())["call_id"])
    else:
        reserve(tag, "H100", 40 / 60)
        call = fn.spawn(R0, files)
        p.write_text(json.dumps({"call_id": call.object_id, "t": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(tag, "call", call.object_id, flush=True)
    res = call.get(timeout=None)
    settle(tag, "H100", max(res["seconds"].values()) + 60)
    print(tag, "done in", time.time() - t0, "s", flush=True)
    return res


def main() -> None:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "cand":
        res = run("score_cand_r0", [f"{VOLP}/cand.jsonl"])
        (PRIVATE / "cand_r0.json").write_text(json.dumps({a: res[a][f"{VOLP}/cand.jsonl"] for a in R0}))
    else:
        files = [f"{VOLP}/evals.jsonl", "data/b1b/val.jsonl", "rsi/cal_unlabeled.jsonl"]
        res = run("score_ref_r0", files)
        (PRIVATE / "scores").mkdir(parents=True, exist_ok=True)
        rows = [json.loads(x) for x in open(PRIVATE.parent / "evals_p2.jsonl")]
        for k, a in enumerate(R0):
            by: dict = {}
            for r, x in zip(rows, res[a][files[0]], strict=True):
                by.setdefault(r["set"], []).append(x)
            out = {"eval": by, "v2b_val": res[a][files[1]], "unlabeled": res[a][files[2]]}
            (PRIVATE / "scores" / f"r0_s{k}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
