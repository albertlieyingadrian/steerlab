"""Real-model check (slow, excluded by default): the package reproduces the reference probe on Qwen2.5-0.5B.

Run with:  .venv/bin/python -m pytest packages/sandbagkit -m slow

Needs Qwen/Qwen2.5-0.5B-Instruct already in the Hugging Face cache (it never downloads) and the reference
toolkit's contrast pairs at ai-metacognition-toolkit/ (or SANDBAGKIT_CONTRAST_PAIRS=<path to the JSON>).
Skips, with the reason, if either is missing.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sandbagkit.probing import ActivationCache, evaluate_layers, extract_activations, pair_rows, pair_split

pytestmark = pytest.mark.slow

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[4]
PAIRS_PATH = Path(
    os.environ.get(
        "SANDBAGKIT_CONTRAST_PAIRS",
        REPO_ROOT / "ai-metacognition-toolkit" / "experiments" / "data" / "sandbagging_contrast_pairs.json",
    )
)

# Per-fold CV scores of the reference rebuild, which match the toolkit's recorded run on 34 of 35 folds.
REFERENCE_FOLDS = {
    2: [0.9, 0.7, 0.8, 0.7, 0.9],
    4: [0.9, 1.0, 0.9, 0.7, 0.9],
    10: [0.8, 0.9, 0.7, 0.8, 1.0],
    12: [0.8, 0.9, 0.7, 0.9, 1.0],
    14: [0.8, 1.0, 0.8, 0.9, 1.0],
    20: [0.8, 0.9, 0.7, 0.9, 0.9],
    22: [0.8, 1.0, 0.9, 0.9, 0.8],
}


@pytest.fixture(scope="module")
def qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True)
    except OSError as error:
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache ({error.__class__.__name__})")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


@pytest.fixture(scope="module")
def pairs():
    if not PAIRS_PATH.exists():
        pytest.skip(f"contrast pairs not found at {PAIRS_PATH}")
    data = json.loads(PAIRS_PATH.read_text())
    texts, labels, groups = [], [], []
    for group in data["contrast_pairs"]:
        for pair in group["pairs"]:
            p = len(texts) // 2
            texts += [pair["sandbagging"], pair["genuine"]]
            labels += [1, 0]
            groups += [p, p]
    return texts, np.array(labels), np.array(groups)


def test_reproduces_the_reference_probe(qwen, pairs, tmp_path):
    model, tokenizer = qwen
    texts, labels, groups = pairs
    assert len(texts) == 64
    train, test = pair_split(len(texts) // 2)
    assert (len(train), len(test)) == (25, 7)

    cache = ActivationCache(tmp_path / "activations")
    acts = extract_activations(model, tokenizer, texts, list(REFERENCE_FOLDS), cache=cache)
    results = evaluate_layers(acts, labels, train_rows=pair_rows(train), test_rows=pair_rows(test), groups=groups)

    layer14 = results[14]
    assert layer14.fold_scores == REFERENCE_FOLDS[14]
    assert layer14.cv == pytest.approx(0.90)
    assert (layer14.cv_accuracy.low, layer14.cv_accuracy.high) == pytest.approx((0.82, 0.98))
    assert layer14.auroc.estimate == pytest.approx(0.950, abs=0.001)
    assert layer14.test_accuracy is not None and layer14.test_accuracy.estimate == pytest.approx(13 / 14)
    assert layer14.shuffled.mean == pytest.approx(0.46, abs=0.02)
    assert layer14.to_result().detected

    identical = sum(results[layer].fold_scores == folds for layer, folds in REFERENCE_FOLDS.items())
    assert identical == len(REFERENCE_FOLDS)
    assert max(results, key=lambda layer: results[layer].cv) == 14

    # A rerun is served entirely from the activation cache, with identical vectors.
    again = ActivationCache(tmp_path / "activations")
    cached = extract_activations(model, tokenizer, texts, list(REFERENCE_FOLDS), cache=again)
    assert again.misses == 0
    for layer in REFERENCE_FOLDS:
        np.testing.assert_array_equal(cached[layer], acts[layer])
