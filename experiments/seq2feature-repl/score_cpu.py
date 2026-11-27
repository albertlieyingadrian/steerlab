"""Score every test span with the released probe (fp32, int8, int4), notebook 02's bag of words and notebook 03's
bge-small zero-shot readout, on CPU. Writes work/scores_test.npz (float16, test spans x 2,048) and
work/scores_ood.npz when --ood is given (needs work/ood/ood_spans.jsonl from modal_harvest.py).

    python experiments/seq2feature-repl/score_cpu.py --ood
"""

import json
import sys
import time

import numpy as np

from common import WORK, load_bundle, load_probe, score_texts


def bow_fit(texts_all, Y, te):
    """Notebook 02, cell 4: hashed 1-2 grams, ridge alpha 2 on 9,000 train spans drawn with seed 0."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import Ridge

    hv = HashingVectorizer(n_features=4096, alternate_sign=False, ngram_range=(1, 2))
    Xb = hv.transform(texts_all).astype(np.float32)
    rng = np.random.default_rng(0)
    sub = rng.choice(np.where(~te)[0], min(9000, (~te).sum()), replace=False)
    rg = Ridge(alpha=2.0).fit(Xb[sub], Y[sub].astype(np.float32))
    return hv, rg, Xb


def main():
    t0 = time.time()
    B = load_bundle()
    Y, te, spans, caps = B["Y"], B["te"], B["spans"], B["caps"]
    texts = [s["text"] for s in spans]
    out, mu_sd = {}, {}
    for kind in ("fp32", "int8", "int4"):
        model, ck = load_probe(kind)
        S = score_texts(model, ck, texts)
        if kind == "int4":  # notebook 01 refits the z statistics on train spans after fake-quantising
            mu, sd = S[~te].mean(0), S[~te].std(0) + 1e-4
        else:
            mu, sd = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)
        out[kind] = ((S[te] - mu) / sd).astype(np.float16)
        mu_sd[kind] = (mu, sd)
        print(f"probe {kind} scored at {time.time() - t0:.0f}s", flush=True)
    hv, rg, Xb = bow_fit(texts, Y, te)
    out["bow"] = rg.predict(Xb[te]).astype(np.float16)
    print(f"bag of words at {time.time() - t0:.0f}s", flush=True)
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    Ec = emb.encode(caps, normalize_embeddings=True, batch_size=256)
    Et = emb.encode([texts[i] for i in np.where(te)[0]], normalize_embeddings=True, batch_size=128)
    out["bge"] = (Et @ Ec.T).astype(np.float16)
    tr_sub = np.random.default_rng(0).choice(np.where(~te)[0], 5000, replace=False)  # train stats for the z variant
    ctr = emb.encode([texts[i] for i in tr_sub], normalize_embeddings=True, batch_size=128) @ Ec.T
    out["bge_z"] = ((Et @ Ec.T - ctr.mean(0)) / (ctr.std(0) + 1e-6)).astype(np.float16)
    print(f"bge at {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(WORK / "scores_test.npz", test_index=np.where(te)[0], **out)

    if "--ood" in sys.argv:
        ood = [json.loads(line) for line in open(WORK / "ood" / "ood_spans.jsonl")]
        otexts = [s["text"] for s in ood]
        o = {}
        for kind in ("fp32", "int8"):
            model, ck = load_probe(kind)
            mu, sd = mu_sd[kind]
            o[kind] = ((score_texts(model, ck, otexts) - mu) / sd).astype(np.float16)
        o["bow"] = rg.predict(hv.transform(otexts).astype(np.float32)).astype(np.float16)
        o["bge"] = (emb.encode(otexts, normalize_embeddings=True, batch_size=128) @ Ec.T).astype(np.float16)
        np.savez_compressed(WORK / "scores_ood.npz", **o)
        print(f"ood scored at {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
