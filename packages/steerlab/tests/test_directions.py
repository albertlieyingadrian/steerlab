"""Difference-of-means directions: the concept data, the quality numbers, and `directions build|list` on the tiny model.

Fast tests use sandbagkit's tiny random model (no download). The two real-model tests are marked `slow` and run on the
user's Mac only, after the models are cached.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from steerlab import data
from steerlab.data import GRADED_LEVELS, MEASURED_CONCEPTS, Concept, list_concepts, load_concept

runner = CliRunner()

CJK = re.compile(r"[一-鿿]")


# ---- concept data (light: no torch) ----


def test_the_shipped_concepts():
    assert list_concepts() == ["chinese", "golden_gate", "refusal", "sandbagging"]


@pytest.mark.parametrize("name", ["chinese", "refusal", "sandbagging"])
def test_every_concept_loads_with_enough_pairs(name: str):
    concept = load_concept(name)
    assert isinstance(concept, Concept)
    assert concept.name == name
    assert 20 <= len(concept.pairs) <= 40
    for pair in concept.pairs:
        assert pair.positive.prompt and pair.positive.response
        assert pair.negative.prompt and pair.negative.response
        assert pair.positive.response != pair.negative.response
    assert len(concept.held_out) >= 5
    prompts_in_pairs = {pair.positive.prompt for pair in concept.pairs} | {
        pair.negative.prompt for pair in concept.pairs
    }
    assert not set(concept.held_out) & prompts_in_pairs


@pytest.mark.parametrize("name", MEASURED_CONCEPTS)
def test_measured_concepts_have_the_five_graded_levels(name: str):
    concept = load_concept(name)
    assert tuple(concept.levels) == GRADED_LEVELS == (0, 25, 50, 75, 100)
    sizes = {level: len(prompts) for level, prompts in concept.levels.items()}
    assert min(sizes.values()) >= 5
    assert len(set(sizes.values())) == 1, "every level has the same number of prompts, so centroids are comparable"


def test_refusal_has_no_graded_levels():
    assert load_concept("refusal").levels == {}


def test_chinese_pairs_differ_only_in_the_language_of_the_reply():
    concept = load_concept("chinese")
    for pair in concept.pairs:
        assert pair.positive.prompt == pair.negative.prompt
        assert CJK.search(pair.positive.response)
        assert not CJK.search(pair.negative.response)


def test_sandbagging_pairs_pair_a_wrong_answer_under_pressure_with_a_right_answer_plain():
    concept = load_concept("sandbagging")
    for pair in concept.pairs:
        assert pair.positive.prompt != pair.negative.prompt
        assert pair.positive.prompt.endswith(pair.negative.prompt)  # the plain question, with an instruction before it
    assert concept.metadata["known_answer_source"].startswith("a fixed list of capitals and chemical symbols")


def test_unknown_concept_names_the_known_ones():
    with pytest.raises(KeyError, match="chinese"):
        load_concept("sarcasm")


def test_as_rows_interleaves_positive_and_negative_with_pair_groups():
    concept = load_concept("refusal")
    texts, responses, labels, groups = concept.as_rows()
    assert len(texts) == len(responses) == len(labels) == len(groups) == 2 * len(concept.pairs)
    assert labels[:4] == [1, 0, 1, 0]
    assert groups[:4] == [0, 0, 1, 1]
    assert (texts[0], responses[0]) == (concept.pairs[0].positive.prompt, concept.pairs[0].positive.response)


def test_concept_json_files_are_the_source():
    folder = Path(data.__file__).parent / "concepts"
    assert sorted(p.stem for p in folder.glob("*.json")) == list_concepts()
    raw = json.loads((folder / "chinese.json").read_text(encoding="utf-8"))
    assert set(raw) >= {"name", "label", "pairs", "levels", "held_out"}


# ---- quality numbers (pure numpy) ----


def test_quality_numbers_by_hand():
    from steerlab.directions import direction_quality

    # Three pairs in 2-D. Positives sit 4 units to the right of their negatives, with a bit of vertical noise.
    neg = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    pos = neg + np.array([[4.0, 0.5], [4.0, -0.5], [4.0, 0.0]])
    X = np.vstack([pos, neg])
    labels = np.array([1, 1, 1, 0, 0, 0])
    groups = np.array([0, 1, 2, 0, 1, 2])
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)  # (4, 0)
    unit = mean_diff / np.linalg.norm(mean_diff)

    q = direction_quality(X, labels, groups, unit)
    # class gap: the projection difference of the two class means on the unit direction
    assert q.class_gap == pytest.approx(float(pos.mean(axis=0) @ unit - neg.mean(axis=0) @ unit)) == pytest.approx(4.0)
    # cosine agreement: mean cosine between each pair's difference and the mean difference
    cosines = [float(d @ unit / np.linalg.norm(d)) for d in pos - neg]
    assert q.cosine_agreement == pytest.approx(float(np.mean(cosines)))
    assert 0.9 < q.cosine_agreement < 1.0
    # d′: gap over the pooled standard deviation of the projections (both classes project to a constant here)
    assert q.d_prime is None  # zero within-class spread along the direction: d′ is undefined, not infinite
    assert q.n_pairs == 3


def test_d_prime_with_spread():
    from steerlab.directions import direction_quality

    rng = np.random.default_rng(0)
    neg = rng.normal(size=(50, 8))
    pos = neg + 3.0 * np.eye(8)[0] + 0.1 * rng.normal(size=(50, 8))
    X = np.vstack([pos, neg])
    labels = np.array([1] * 50 + [0] * 50)
    groups = np.array(list(range(50)) * 2)
    unit = np.eye(8)[0]
    q = direction_quality(X, labels, groups, unit)
    proj_pos, proj_neg = pos @ unit, neg @ unit
    pooled = np.sqrt((proj_pos.var(ddof=1) + proj_neg.var(ddof=1)) / 2)
    assert q.d_prime == pytest.approx(float((proj_pos.mean() - proj_neg.mean()) / pooled))
    assert q.class_gap > 0


def test_template_kwargs_turn_thinking_off_for_qwen3_only():
    from steerlab.directions import template_kwargs_for

    assert template_kwargs_for("Qwen/Qwen3-1.7B") == {"enable_thinking": False}
    assert template_kwargs_for("google/gemma-3-1b-it") == {}
    assert template_kwargs_for("tiny-random-qwen2") == {}


# ---- the tiny model: build and list ----

torch = pytest.importorskip("torch")
pytest.importorskip("sandbagkit.probing")

from steerlab.cli_directions import app  # noqa: E402
from steerlab.directions import (  # noqa: E402
    DIRECTION_RELEASE,
    METADATA_KEYS,
    build_direction,
    direction_path,
    load_direction,
    to_feature_direction,
)


@pytest.fixture(scope="module")
def tiny():
    from sandbagkit.backends.tiny import tiny_backend

    return tiny_backend(seed=0)


def test_build_writes_an_npz_json_pair_with_metadata_and_a_unit_vector(tmp_path: Path):
    result = runner.invoke(app, ["build", "--model", "tiny", "--concept", "chinese", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    base = direction_path(tmp_path, "tiny", "chinese")
    npz, meta_file = base.with_suffix(".npz"), base.with_suffix(".json")
    assert npz.exists() and meta_file.exists()
    assert npz.parent == tmp_path / "tiny"

    description = json.loads(meta_file.read_text(encoding="utf-8"))
    metadata = description["metadata"]
    assert set(METADATA_KEYS) <= set(metadata)
    assert METADATA_KEYS == ("model", "layer", "concept", "class_gap", "cosine_agreement", "d_prime")
    assert metadata["concept"] == "chinese"
    assert metadata["model"] == "tiny-random-qwen2"
    assert metadata["layer"] == 1
    assert metadata["class_gap"] > 0
    assert -1.0 <= metadata["cosine_agreement"] <= 1.0

    vec = load_direction(base)
    assert vec.layer == 1
    assert vec.norm == pytest.approx(1.0, abs=1e-6)
    assert vec.train_groups is not None and len(vec.train_groups) == len(load_concept("chinese").pairs)
    assert vec.metadata["release"] == DIRECTION_RELEASE

    # the command prints the quality numbers for the lesson to quote
    assert "class gap" in result.output and "cosine agreement" in result.output and "d'" in result.output


def test_class_gap_is_the_projection_difference_of_the_class_means(tiny):
    from sandbagkit.probing import extract_activations

    concept = load_concept("refusal")
    vec = build_direction(tiny.model, tiny.tokenizer, concept, layer=1, model_name=tiny.name)
    texts, responses, labels, _groups = concept.as_rows()
    acts = extract_activations(
        tiny.model,
        tiny.tokenizer,
        texts,
        [1],
        responses=responses,
        position="response_mean",
        chat_template=True,
        skip_first_response_token=True,
    )[1]
    y = np.asarray(labels)
    expected = float(acts[y == 1].mean(axis=0) @ vec.vector - acts[y == 0].mean(axis=0) @ vec.vector)
    assert vec.metadata["class_gap"] == pytest.approx(expected, rel=1e-5)
    assert vec.metadata["class_gap"] > 0
    assert np.linalg.norm(vec.vector) == pytest.approx(1.0, abs=1e-6)


def test_feature_direction_carries_the_class_gap_as_its_scale(tiny):
    vec = build_direction(tiny.model, tiny.tokenizer, load_concept("chinese"), layer=0, model_name=tiny.name)
    fd = to_feature_direction(vec)
    assert fd.layer == 0
    assert fd.release == DIRECTION_RELEASE
    assert fd.label == "chinese"
    assert fd.scale == pytest.approx(vec.metadata["class_gap"])
    assert fd.unit.dtype == torch.float32
    assert float(fd.unit.norm()) == pytest.approx(1.0, abs=1e-5)


def test_list_reports_present_directions_and_names_the_build_command_for_missing_ones(tmp_path: Path):
    built = runner.invoke(app, ["build", "--model", "tiny", "--concept", "sandbagging", "--out", str(tmp_path)])
    assert built.exit_code == 0, built.output
    result = runner.invoke(app, ["list", "--model", "tiny", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.strip()}
    assert "present" in lines["sandbagging"] and "class gap" in lines["sandbagging"]
    for missing in ("chinese", "refusal"):
        assert "missing" in lines[missing]
        assert f"steerlab directions build --model tiny --concept {missing} --out {tmp_path}" in lines[missing]


def test_build_defaults_to_all_three_concepts(tmp_path: Path):
    result = runner.invoke(app, ["build", "--model", "tiny", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert sorted(p.stem for p in (tmp_path / "tiny").glob("*.npz")) == ["chinese", "refusal", "sandbagging"]


def test_build_rejects_unknown_models_and_concepts(tmp_path: Path):
    bad_model = runner.invoke(app, ["build", "--model", "llama", "--out", str(tmp_path)])
    assert bad_model.exit_code != 0
    assert "qwen" in bad_model.output and "gemma" in bad_model.output
    bad_concept = runner.invoke(app, ["build", "--model", "tiny", "--concept", "sarcasm", "--out", str(tmp_path)])
    assert bad_concept.exit_code != 0
    assert "chinese" in bad_concept.output


# ---- real models (slow, unexecuted here) ----


def _cjk_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if CJK.match(c)) / max(1, len(letters))


def _steered_cjk_ratio(model_key: str, strength: float) -> float:
    from sandbagkit.probing import inject

    from steerlab.directions import load_model, template_kwargs_for

    model, tokenizer, model_name, layer = load_model(model_key)
    concept = load_concept("chinese")
    vec = build_direction(model, tokenizer, concept, layer=layer, model_name=model_name)
    fd = to_feature_direction(vec)
    delta = fd.unit * (fd.scale or 1.0)  # strength 1.0 == one class gap
    ratios = []
    for prompt in concept.held_out[:5]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs_for(model_name),
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with inject(model, delta, layer=layer, strength=strength), torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, repetition_penalty=1.0)
        ratios.append(_cjk_ratio(tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)))
    return float(np.mean(ratios))


@pytest.mark.slow
def test_qwen3_chinese_direction_steers_at_strength_two():
    assert _steered_cjk_ratio("qwen", 2.0) >= 0.5


@pytest.mark.slow
def test_gemma_chinese_direction_steers_at_strength_two():
    assert _steered_cjk_ratio("gemma", 2.0) >= 0.5
