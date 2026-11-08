"""
Step 2: extract the three persona vectors with the paper's recipe.

For each trait, each of the 20 extraction questions is asked under 5 positive and 5 negative system prompts, 4 samples
each (temperature 1, up to 256 new tokens):

    "You are a sandbagging assistant. Deliberately provide answers that are weaker..."   + question  -> reply (pos)
    "You are a helpful assistant. Provide the most accurate, complete... answers..."     + question  -> reply (neg)

The judge scores every reply for the trait and for coherence. A (pos, neg) pair, matched by question, instruction
and sample number as in the paper's code, is kept when pos trait >= 50, neg trait < 50, and both coherence >= 50.
On the kept replies, the model is run once more to read its activations, and per layer:

    persona vector = mean over kept pos replies of (mean activation over the reply's tokens)
                   - the same over kept neg replies                                       ("response_avg")

The last prompt token gives a second vector ("prompt_last"), the readout the paper uses to predict a shift from the
prompt alone. Writes persona_vectors.npz (and per-question sums for the bootstrap in step 4) and the replies with
their scores (extract_<trait>.json) to the work folder, ~/.cache/steerlab/persona-vectors/, and a summary to
results/2_extract.json.

    .venv/bin/python experiments/persona-vectors/2_extract.py     # ~40 min, ~$1 judge
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (
    RESULTS,
    TRAITS,
    Judge,
    chat_prompt,
    check,
    generate,
    load_base,
    load_json,
    outputs,
    response_activations,
    save_json,
    system_prompt,
    trait_data,
)

N_SAMPLES = 4
MAX_NEW_TOKENS = 256
THRESHOLD = 50


def replies_for(model, tokenizer, trait):
    """Every (kind, question index, instruction index, sample) reply, cached in outputs/extract_<trait>.json."""
    path = outputs() / f"extract_{trait}.json"
    if path.exists():
        return load_json(path)
    data = trait_data(trait, "extract")
    rows = []
    for kind in ("pos", "neg"):
        for k, pair in enumerate(data["instruction"]):
            system = system_prompt(trait, pair[kind], kind)
            for qi, question in enumerate(data["questions"]):
                for s in range(N_SAMPLES):
                    rows.append({"kind": kind, "q": qi, "k": k, "s": s, "system": system, "question": question,
                                 "prompt": chat_prompt(tokenizer, question, system)})  # fmt: skip
    start = time.time()
    replies = generate(model, tokenizer, [r["prompt"] for r in rows], max_new_tokens=MAX_NEW_TOKENS, temperature=1.0)
    for r, reply in zip(rows, replies, strict=True):
        r["answer"] = reply
    print(f"   {trait}: {len(rows)} replies in {time.time() - start:.0f} s", flush=True)
    save_json(path, rows)
    return rows


def main():
    model, tokenizer = load_base()
    judge = Judge()
    vectors, sums, summary = {}, {}, {}
    for trait in TRAITS:
        print(f"\n== {trait} ==", flush=True)
        data = trait_data(trait, "extract")
        rows = replies_for(model, tokenizer, trait)
        if "trait" not in rows[0]:
            t, c = judge.trait_and_coherence(data["eval_prompt"], [r["question"] for r in rows],
                                             [r["answer"] for r in rows])  # fmt: skip
            for r, ts, cs in zip(rows, t, c, strict=True):
                r["trait"], r["coherence"] = ts, cs
            save_json(outputs() / f"extract_{trait}.json", rows)
        by_key = {(r["kind"], r["q"], r["k"], r["s"]): r for r in rows}
        keys = sorted({(r["q"], r["k"], r["s"]) for r in rows})

        def ok(r, kind):
            if r["trait"] is None or r["coherence"] is None:
                return False
            trait_ok = r["trait"] >= THRESHOLD if kind == "pos" else r["trait"] < 100 - THRESHOLD
            return trait_ok and r["coherence"] >= 50

        kept = [key for key in keys if ok(by_key[("pos", *key)], "pos") and ok(by_key[("neg", *key)], "neg")]
        pos_scores = [by_key[("pos", *k)]["trait"] for k in keys if by_key[("pos", *k)]["trait"] is not None]
        neg_scores = [by_key[("neg", *k)]["trait"] for k in keys if by_key[("neg", *k)]["trait"] is not None]
        print(f"   mean trait score: pos {np.mean(pos_scores):.1f}, neg {np.mean(neg_scores):.1f}; "
              f"kept {len(kept)} of {len(keys)} pairs", flush=True)  # fmt: skip

        pos = [by_key[("pos", *k)] for k in kept]
        neg = [by_key[("neg", *k)] for k in kept]
        start = time.time()
        p_resp, p_last = response_activations(model, tokenizer, [r["prompt"] for r in pos], [r["answer"] for r in pos])
        n_resp, n_last = response_activations(model, tokenizer, [r["prompt"] for r in neg], [r["answer"] for r in neg])
        print(f"   activations for {2 * len(kept)} replies in {time.time() - start:.0f} s", flush=True)
        vectors[f"{trait}_response_avg"] = p_resp.mean(0) - n_resp.mean(0)
        vectors[f"{trait}_prompt_last"] = p_last.mean(0) - n_last.mean(0)
        # Per-question sums and counts, so step 3 can bootstrap over questions without the raw activations.
        q_of = np.array([k[0] for k in kept])
        n_q = len(data["questions"])
        for name, P, N in (("response_avg", p_resp, n_resp), ("prompt_last", p_last, n_last)):
            sums[f"{trait}_{name}_pos"] = np.stack([P[q_of == q].sum(0) for q in range(n_q)])
            sums[f"{trait}_{name}_neg"] = np.stack([N[q_of == q].sum(0) for q in range(n_q)])
        sums[f"{trait}_count"] = np.array([(q_of == q).sum() for q in range(n_q)])
        norms = np.linalg.norm(vectors[f"{trait}_response_avg"], axis=1)
        resid = np.linalg.norm(n_resp, axis=2).mean(0)
        summary[trait] = {
            "pairs": len(keys), "kept": len(kept), "kept_per_question": sums[f"{trait}_count"].tolist(),
            "pos_trait_mean": float(np.mean(pos_scores)), "neg_trait_mean": float(np.mean(neg_scores)),
            "pos_coherence_mean": float(np.nanmean([by_key[("pos", *k)]["coherence"] or np.nan for k in keys])),
            "neg_coherence_mean": float(np.nanmean([by_key[("neg", *k)]["coherence"] or np.nan for k in keys])),
            "norm_by_layer": norms.tolist(), "resid_norm_by_layer": resid.tolist(),
        }  # fmt: skip
        examples = sorted(kept, key=lambda k: -by_key[("pos", *k)]["trait"])[:3]
        summary[trait]["examples"] = [{"question": by_key[("pos", *k)]["question"],
                                       "pos": by_key[("pos", *k)]["answer"][:400],
                                       "pos_trait": by_key[("pos", *k)]["trait"],
                                       "neg": by_key[("neg", *k)]["answer"][:400],
                                       "neg_trait": by_key[("neg", *k)]["trait"]} for k in examples]  # fmt: skip
        for ex in summary[trait]["examples"][:2]:
            print(f"   Q: {ex['question']}\n     pos ({ex['pos_trait']:.0f}): {ex['pos'][:160]!r}"
                  f"\n     neg ({ex['neg_trait']:.0f}): {ex['neg'][:160]!r}")  # fmt: skip
        ratios = " ".join(f"L{layer}:{norms[layer] / resid[layer]:.2f}" for layer in range(0, len(norms), 4))
        print(f"   |vector| / |activation| by layer: {ratios}")

    np.savez(outputs() / "persona_vectors.npz", **{k: v.astype(np.float32) for k, v in vectors.items()})
    np.savez(outputs() / "persona_sums.npz", **{k: v.astype(np.float32) for k, v in sums.items()})
    save_json(RESULTS / "2_extract.json", summary)
    print(f"\n  saved {len(vectors)} vectors [28, {model.config.hidden_size}] to the work folder (persona_vectors.npz)")

    print("\nSelf-checks:")
    for trait in TRAITS:
        s = summary[trait]
        check(f"{trait}: positive prompts raise the judge's score (pos {s['pos_trait_mean']:.0f} vs neg "
              f"{s['neg_trait_mean']:.0f})", s["pos_trait_mean"] > s["neg_trait_mean"] + 20)  # fmt: skip
        check(f"{trait}: at least 30% of pairs pass the filter ({s['kept']}/{s['pairs']})",
              s["kept"] >= 0.3 * s["pairs"])  # fmt: skip
        check(f"{trait}: every vector finite", all(np.isfinite(vectors[f'{trait}_{r}']).all()
                                                    for r in ("response_avg", "prompt_last")))  # fmt: skip


if __name__ == "__main__":
    main()
