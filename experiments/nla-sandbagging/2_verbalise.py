"""Step 2: the NLA's verbaliser (AV) explains every saved vector in words.

    python experiments/nla-sandbagging/2_verbalise.py --only example     # the worked example first (validation)
    python experiments/nla-sandbagging/2_verbalise.py --only quiz,controls

Loads only the AV (about 15 GB in bfloat16). Greedy decoding, at most 200 new tokens, repetition_penalty 1.0.
Sets of vectors:
  example   positions 24-100 of the released worked example (the positions its summary scores; the first 24 are the
            system prompt, which the NLA saw little of in training)
  quiz      80 items x 4 conditions x 2 positions (last prompt token, answer letter) = 640
  controls  40 random Gaussian directions, and 40 vectors whose coordinates are shuffled (each a permutation of a
            sandbag prompt-position vector): what the AV says when the vector carries no real content

Writes results/explanations.json: one row per vector, {"set", "condition", "position", "id", "raw", "explanation"}.
Rows of the sets run here replace earlier rows of the same sets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import common as c

FIRST_SCORED = 24  # the worked example's summary scores positions 24+
N_CONTROL = 40


def vector_sets(names: list[str]) -> tuple[list[dict], np.ndarray]:
    rows, vecs = [], []
    if "example" in names:
        ex = np.load(c.work_dir() / "example_vectors.npy")
        for p in range(FIRST_SCORED, len(ex)):
            rows.append({"set": "example", "condition": "", "position": p, "id": f"example:{p}"})
            vecs.append(ex[p])
    if "quiz" in names or "controls" in names:
        q = np.load(c.work_dir() / "quiz_vectors.npz")
        ids = [str(x) for x in q["ids"]]
    if "quiz" in names:
        for ci, cond in enumerate(c.CONDITIONS):
            for pi, pos in enumerate(c.POSITIONS):
                for ii, qid in enumerate(ids):
                    rows.append({"set": "quiz", "condition": cond, "position": pos, "id": qid})
                    vecs.append(q["vectors"][ci, pi, ii])
    if "controls" in names:
        rng = np.random.default_rng(c.SEED)
        d = q["vectors"].shape[-1]
        for k in range(N_CONTROL):
            rows.append({"set": "random", "condition": "", "position": "", "id": f"random:{k}"})
            vecs.append(rng.standard_normal(d).astype(np.float32))
        source = q["vectors"][c.CONDITIONS.index("sandbag"), 0]
        for k in range(N_CONTROL):
            rows.append({"set": "shuffled", "condition": "sandbag", "position": "prompt", "id": ids[k]})
            vecs.append(rng.permutation(source[k]))
    return rows, np.stack(vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="example,quiz,controls", help="comma-separated: example, quiz, controls")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    names = args.only.split(",")
    rows, vecs = vector_sets(names)
    print(f"{len(rows)} vectors to explain; free memory before loading: {c.free_memory_gb():.0f} GB")
    av = c.Verbaliser()
    start = time.time()
    raws = av.explain(vecs, batch_size=args.batch_size)
    print(f"{len(raws)} explanations in {time.time() - start:.0f} s")
    for r, raw in zip(rows, raws, strict=True):
        r["raw"], r["explanation"] = raw, c.explanation(raw)
    missing = sum("<explanation>" not in r["raw"] or "</explanation>" not in r["raw"] for r in rows)
    c.check(f"every explanation is inside <explanation> tags ({missing} of {len(rows)} missing a tag)", missing == 0)
    path = c.RESULTS / "explanations.json"
    sets = {r["set"] for r in rows}
    old = [r for r in c.load_json(path) if r["set"] not in sets] if path.exists() else []
    c.save_json(path, old + rows)


if __name__ == "__main__":
    main()
