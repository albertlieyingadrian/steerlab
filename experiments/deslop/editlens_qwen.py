"""A Qwen3-4B LoRA "how much AI" detector trained on EditLens's data, scored next to the released RoBERTa model.

    python editlens_qwen.py      # clean and bucket 20,000 training texts locally, train on Modal (one H100), evaluate

Recipe: the EditLens authors' Llama-3.2-3B configuration (four buckets from `cosine_score` with cut points 0.03 and
0.15, texts of 75 words or more, LoRA r 8 on every projection, a LayerNorm + linear head, one epoch), on Qwen3-4B in
bf16 rather than 4-bit, with 20,000 of the 59,991 training texts (shuffled with the authors' seed, 42) to fit the
budget, and learning rate 1e-4. An unpublished replication shared with the author reported a Qwen3.5-4B LoRA at
ternary macro-F₁ 0.909 on the test split; this checks a smaller version of that claim.

Licence: EditLens's data and code are CC BY-NC-SA 4.0 (non-commercial). The training texts are read through the
Hugging Face cache and cleaned with the authors' own code from a local clone ($EDITLENS_DIR); nothing from the dataset
is written to this repository, and the adapter stays on the private Modal volume, unreleased. Measurement only: nothing
in the note was tuned against this detector. Writes results/editlens/qwen.json (metrics, and this note's own texts'
scores).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from editlens import EDITLENS, EXTRA, LABELS, OUT, evaluate, ternary

TRAIN_EXAMPLES, SEED = 20_000, 42


def main() -> None:
    import modal
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from detect import texts

    sys.path[:0] = [EXTRA, str(EDITLENS / "scripts")]
    from preprocess import clean_text, count_words, score_to_bucket

    path = hf_hub_download("pangram/editlens_iclr", "data/train-00000-of-00001.parquet", repo_type="dataset")
    train = pd.read_parquet(path)
    train = train[train["cosine_score"].notna() & train["text"].notna()]
    train = train[train["text"].map(count_words) >= 75].sample(frac=1.0, random_state=SEED).head(TRAIN_EXAMPLES)
    pairs = [
        [clean_text(t), score_to_bucket(s, 4, 0.03, 0.15)]
        for t, s in zip(train["text"], train["cosine_score"], strict=True)
    ]
    print(f"{len(pairs)} training texts; buckets {np.bincount([p[1] for p in pairs]).tolist()}")
    frames = {s: pd.read_csv(EDITLENS / "data" / f"{s}.csv") for s in ("val", "test", "test_llama")}
    ours = texts()
    evals = {s: [clean_text(t) for t in f["text"]] for s, f in frames.items()}
    evals["ours"] = [clean_text(ours[k]) for k in ours]
    out = modal.Function.from_name("deslop", "train_editlens_qwen").remote(pairs, evals)
    scores = {k: np.array(v) for k, v in out["scores"].items()}
    labels = {s: f["text_type"].map(LABELS).to_numpy() for s, f in frames.items()}
    cuts = ternary(scores["val"], labels["val"])
    result = {
        "model": "Qwen/Qwen3-4B-Instruct-2507 + LoRA r8 (private, unreleased)",
        "train": {
            "examples": out["examples"],
            "seconds": out["seconds"],
            "loss_first_50": float(np.mean(out["loss"][:50])),
            "loss_last_50": float(np.mean(out["loss"][-50:])),
        },
        "splits": {s: evaluate(scores[s], labels[s], cuts) for s in ("test", "test_llama")},
        "ours": dict(zip(list(ours), [round(float(x), 5) for x in scores["ours"]], strict=True)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen.json").write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "ours"}, indent=1))


if __name__ == "__main__":
    main()
