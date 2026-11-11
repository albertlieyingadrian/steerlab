"""Per model: the pool questions it answers right with the neutral system prompt and no framing, split build/test.

Sandbagging is only visible on questions the model knows, so everything downstream uses these. The filter uses no
framing. Seeded split: 150 build questions (directions only) and up to 400 test questions (every reported number).
Writes results/<model>/known.json (MMLU test-row ids and counts only).
"""

import sys

import numpy as np

import common as C


def main(name, n_train=150, n_test=400):
    if C.SMOKE:
        n_train, n_test = 16, 24
    tok, model = C.load(name)
    pool = C.load_pool()
    texts = [C.prompt(tok, q, None, C.NEUTRAL_SYSTEM) for q in pool]
    p, mass, _ = C.run(model, tok, texts)
    right = p.argmax(1) == np.array([q["answer"] for q in pool])
    ids = np.array([q["id"] for q in pool])[right]
    perm = np.random.default_rng(7).permutation(len(ids))
    train = sorted(ids[perm[:n_train]].tolist())
    test = sorted(ids[perm[n_train : n_train + n_test]].tolist())
    out = {
        "model": C.MODELS[name],
        "pool": len(pool),
        "known": int(right.sum()),
        "pool_accuracy": round(float(right.mean()), 4),
        "letter_mass_mean": round(float(mass.mean()), 4),
        "train": train,
        "test": test,
    }
    C.write_json(C.RESULTS / name / "known.json", out)
    print(name, {k: v for k, v in out.items() if k not in ("train", "test")}, len(train), len(test))


if __name__ == "__main__":
    main(sys.argv[1])
