"""
Second attempt, step 9: the prompted sandbagging vector (P) from the full run of the chosen prompt version.

The kept pairs (step 8's filter) are read once more by the model, and per layer, as in step 2:

    P = mean over kept pos replies of (mean activation over the reply's tokens) - the same over kept neg replies

plus the last-prompt-token version, and per-question sums so later steps can bootstrap over questions. Writes
vectors.npz and sums.npz to the work folder (v2/<model>/) and a summary to results/9_extract_v2.json.

    .venv/bin/python experiments/persona-vectors/9_extract_v2.py --version v1 --samples 4
    .venv/bin/python experiments/persona-vectors/9_extract_v2.py --version v1 --samples 2 --model 7b
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import RESULTS, check, load_base, load_json, response_activations, trait_data
from common_v2 import MODELS, TRAIT, kept_pairs, update_json, work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default="1.5b", choices=list(MODELS))
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    path = work(args.model) / f"replies_{args.version}_n{args.samples}.json"
    rows = [{**r, "trait": r["intent"]} for r in load_json(path)]
    kept, keys = kept_pairs(rows)
    by_key = {(r["kind"], r["q"], r["k"], r["s"]): r for r in rows}
    print(f"{args.version} on {args.model}: {len(kept)} of {len(keys)} pairs kept", flush=True)

    model, tokenizer = load_base(*MODELS[args.model])
    pos = [by_key[("pos", *k)] for k in kept]
    neg = [by_key[("neg", *k)] for k in kept]
    start = time.time()
    p_resp, p_last = response_activations(model, tokenizer, [r["prompt"] for r in pos], [r["answer"] for r in pos])
    n_resp, n_last = response_activations(model, tokenizer, [r["prompt"] for r in neg], [r["answer"] for r in neg])
    print(f"   activations for {2 * len(kept)} replies in {time.time() - start:.0f} s", flush=True)
    vectors = {"response_avg": p_resp.mean(0) - n_resp.mean(0), "prompt_last": p_last.mean(0) - n_last.mean(0)}
    q_of = np.array([k[0] for k in kept])
    n_q = len(trait_data(TRAIT, "extract")["questions"])
    sums = {"count": np.array([(q_of == q).sum() for q in range(n_q)])}
    for name, P, N in (("response_avg", p_resp, n_resp), ("prompt_last", p_last, n_last)):
        sums[f"{name}_pos"] = np.stack([P[q_of == q].sum(0) for q in range(n_q)])
        sums[f"{name}_neg"] = np.stack([N[q_of == q].sum(0) for q in range(n_q)])
    np.savez(work(args.model) / "vectors.npz", **{k: v.astype(np.float32) for k, v in vectors.items()})
    np.savez(work(args.model) / "sums.npz", **{k: v.astype(np.float32) for k, v in sums.items()})
    norms = np.linalg.norm(vectors["response_avg"], axis=1)
    resid = np.linalg.norm(n_resp, axis=2).mean(0)
    summary = {"version": args.version, "samples": args.samples, "pairs": len(keys), "kept": len(kept),
               "kept_per_question": sums["count"].tolist(), "norm_by_layer": norms.tolist(),
               "resid_norm_by_layer": resid.tolist()}  # fmt: skip
    update_json(RESULTS / "9_extract_v2.json", args.model, summary)
    ratios = " ".join(f"L{i}:{norms[i] / resid[i]:.2f}" for i in range(0, len(norms), 4))
    print(f"   |P| / |activation| by layer: {ratios}")
    print("\nSelf-checks:")
    check("every vector finite", all(np.isfinite(v).all() for v in vectors.values()))
    check(f"every extraction question has a kept pair ({int((sums['count'] > 0).sum())}/{n_q})",
          bool((sums["count"] > 0).all()))  # fmt: skip


if __name__ == "__main__":
    main()
