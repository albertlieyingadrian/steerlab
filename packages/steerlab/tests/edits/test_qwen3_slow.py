"""Slow: neuron-reflection's mother/father mirror, reproduced qualitatively on Qwen3-0.6B (CPU, float32, ~15 s).

Word axes from their six English sentence pairs; the 100 neurons per layer (3.3%) that load most on them; k = 2.
On Qwen3-8B they report "Who gives birth?" → "The father." (§2). Here the forced choice "Mother or Father" flips sign
while the random-axis and random-unit controls of the same size leave it where it was, and the weights restore
bit-identically. Needs Qwen/Qwen3-0.6B in the local Hugging Face cache. Run with `pytest -m slow`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from steerlab.edits.checks import compare  # noqa: E402
from steerlab.edits.reflection import Editor, select_top, sentence_axes  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
SENTENCES = [
    ("I visited my mother last week.", "I visited my father last week."),
    ("We bought a gift for our mother.", "We bought a gift for our father."),
    ("She talked to her mother about school.", "She talked to her father about school."),
    ("The children ran to their mother.", "The children ran to their father."),
    ("He wrote a letter to his mother.", "He wrote a letter to his father."),
    ("My mother lives near the sea.", "My father lives near the sea."),
]


def test_mother_father_mirror_beats_both_controls() -> None:
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")
    model.eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        start = model(ids).logits.clone()

    axes = sentence_axes(model, tok, SENTENCES)
    editor = Editor(model)
    units = select_top(editor, axes, per_layer=100)
    birth = {
        "kind": "choice",
        "question": "Who gives birth to a baby? Answer with one word: Mother or Father.",
        "a": "Mother",
        "b": "Father",
    }
    jesus = {
        "kind": "choice",
        "question": "Who was the mother of Jesus? Answer with one word: Mary or Joseph.",
        "a": "Mary",
        "b": "Joseph",
    }
    result = compare(
        editor,
        tok,
        axes,
        units,
        k=2.0,
        max_new_tokens=16,
        checks=[birth, jesus],
        prompts=["Who gives birth to a baby, the mother or the father? Answer briefly."],
    )
    margin = {name: [c["value"] for c in col["checks"]] for name, col in result["columns"].items()}

    assert margin["before"][0] > 3  # the model knows mothers give birth
    assert margin["after"][0] < 0  # mirrored: it now prefers "Father"
    assert margin["random_axis"][0] > 3 and margin["random_units"][0] > 3  # same-size controls do not
    assert margin["after"][1] < margin["random_axis"][1] - 3  # Mary loses ground only under the real edit
    assert "father" in result["columns"]["after"]["replies"][0].lower()
    assert result["restored"]
    with torch.no_grad():
        assert torch.equal(model(ids).logits, start)
