"""
Interpretability, script 1: concept vectors, and a logit-lens warm-up to see where each one becomes readable.

Rebuilds the first step of Lindsey 2025, "Emergent Introspective Awareness in Large Language Models", following the
open replication github.com/uzaymacar/introspective-awareness (MIT).

Steps:
1. For each of 20 concept words ("Dust", "Satellites", ..., "Happiness") ask the model "Tell me about {word}" and read
   the residual stream at the last prompt token, after every block (1 to 28).
2. Do the same for 50 unrelated baseline words ("Desks", "Jackets", ...) and average them.
3. Concept vector = concept activation - baseline mean. Subtracting the mean removes what every "Tell me about X"
   prompt shares, leaving what is special about X.
4. Warm-up, the logit lens: push each concept vector through the model's final norm and unembedding as if it were
   the last layer, and ask where the concept's own word ranks among the 151,936 tokens. A low rank means the vector
   already "says" the word at that depth.

Run: python experiments/residency-101/interp/1_concept_vectors.py [--model 7b]
     (under a minute for 1.5B on MPS; a few minutes for 7B)
"""

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject as I

import tools as t

LESSON = Path(__file__).resolve().parent


def word_token_ids(tokenizer, concept):
    """First token of every spelling of the concept: singular/plural, lower/Title case, with/without a space."""
    ids = set()
    for form in I.word_forms(concept):
        for text in (form, form.title(), " " + form, " " + form.title()):
            ids.add(tokenizer.encode(text, add_special_tokens=False)[0])
    return sorted(ids)


def main():
    import torch

    tag = I.model_arg()
    vectors_path = t.OUT / "04" / f"vectors_{tag}.npz"
    result_path = t.RESULTS / "interp" / f"1_concept_vectors_{tag}.json"
    model, tokenizer, device = t.load_model(I.MODELS[tag])
    n_layers = model.config.num_hidden_layers
    layers = list(range(1, n_layers + 1))
    print(f"{I.MODELS[tag]}: {n_layers} blocks, d_model {model.config.hidden_size}")

    concept_acts = I.residuals(model, tokenizer, device, [I.word_prompt(tokenizer, w) for w in I.CONCEPTS], layers)
    base_acts = I.residuals(model, tokenizer, device, [I.word_prompt(tokenizer, w) for w in I.BASELINE], layers)
    vectors = np.stack([concept_acts[layer] - base_acts[layer].mean(0) for layer in layers], axis=1)  # (20, L, d)
    resid_norm = np.array([np.linalg.norm(base_acts[layer], axis=1).mean() for layer in layers])
    vec_norm = np.linalg.norm(vectors, axis=2)  # (20, L)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(vectors_path, vectors=vectors.astype(np.float32), layers=np.array(layers), concepts=np.array(I.CONCEPTS))
    print(f"Saved {vectors.shape} concept vectors to {t.rel(vectors_path)}")

    # Logit lens: final norm, then unembedding.
    dtype = next(model.parameters()).dtype
    ranks = np.zeros((len(I.CONCEPTS), n_layers), dtype=int)
    with torch.no_grad():
        for c, concept in enumerate(I.CONCEPTS):
            vec_t = torch.as_tensor(vectors[c], dtype=dtype, device=device)
            logits = model.lm_head(model.model.norm(vec_t)).float()  # (L, vocab)
            ids = word_token_ids(tokenizer, concept)
            best = logits[:, ids].max(dim=1, keepdim=True).values
            ranks[c] = (logits > best).sum(dim=1).cpu().numpy()  # 0 = the concept's word is the top token

    print("\nConcept vector size and logit lens, by layer (median over 20 concepts)")
    print(f"{'layer':>5s} {'|vector|':>9s} {'|residual|':>11s} {'ratio':>6s} {'median rank':>12s} {'in top 10':>10s}")
    by_layer = []
    for li, layer in enumerate(layers):
        top10 = float((ranks[:, li] < 10).mean())
        row = {
            "layer": layer,
            "vector_norm": float(vec_norm[:, li].mean()),
            "residual_norm": float(resid_norm[li]),
            "median_rank": float(np.median(ranks[:, li])),
            "share_top10": top10,
        }
        by_layer.append(row)
        mark = "  <- injection layer" if layer in I.LAYERS else ""
        print(
            f"{layer:5d} {row['vector_norm']:9.1f} {row['residual_norm']:11.1f} "
            f"{row['vector_norm'] / row['residual_norm']:6.2f} {row['median_rank']:12.0f} {top10:10.0%}{mark}"
        )

    show = I.LAYERS[-1]
    print(f"\nTop 5 tokens of a few concept vectors at layer {show} (logit lens):")
    with torch.no_grad():
        for concept in ("Dust", "Lightning", "Trees", "Sadness", "Oceans"):
            c = I.CONCEPTS.index(concept)
            v = torch.as_tensor(vectors[c, show - 1], dtype=dtype, device=device)
            top = model.lm_head(model.model.norm(v)).float().topk(5).indices.tolist()
            print(f"  {concept:10s} -> {[tokenizer.decode([i]) for i in top]}")

    unit = vectors[:, show - 1] / np.linalg.norm(vectors[:, show - 1], axis=1, keepdims=True)
    sim = unit @ unit.T
    off = sim[~np.eye(len(sim), dtype=bool)]
    print(
        f"\nCosine between different concepts' vectors at layer {show}: mean {off.mean():.2f} "
        f"(min {off.min():.2f}, max {off.max():.2f})"
    )

    summary = {
        "model": I.MODELS[tag],
        "concepts": I.CONCEPTS,
        "baseline_words": len(I.BASELINE),
        "by_layer": by_layer,
        "ranks": ranks.tolist(),
        "cosine_mean_offdiag": float(off.mean()),
    }
    t.write_json(result_path, summary)

    print("\nSelf-checks")
    t.check(
        "one vector per concept per layer, no NaNs", vectors.shape[:2] == (20, n_layers) and not np.isnan(vectors).any()
    )
    t.check("concept vectors grow with depth (last layer > layer 1)", vec_norm[:, -1].mean() > vec_norm[:, 0].mean())
    best = max(row["share_top10"] for row in by_layer)
    t.check("the logit lens reads the concept's word (top 10) for most concepts at some layer", best >= 0.5)
    t.check("different concepts point in different directions (mean cosine < 0.8)", off.mean() < 0.8)
    t.check("results written", result_path.exists())
    del model
    gc.collect()


if __name__ == "__main__":
    main()
