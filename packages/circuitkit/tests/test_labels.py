"""Labelling pipeline with a scripted model (no network), metrics, caching, Neuronpedia parsing and export."""

from __future__ import annotations

import json
import re

import pytest

from circuitkit.labels import (
    CachedLLM,
    FeatureData,
    Record,
    ScriptedLLM,
    calibration,
    label_feature,
    label_features,
    score_labels,
    split_records,
)
from circuitkit.labels.autointerp import auroc, balanced_accuracy, correlation, format_record, isotonic
from circuitkit.labels.neuronpedia import compare, export_explanations, neuronpedia_labels, records_from_neuronpedia


def _feature(unit: str = "3:42", word: str = " Paris") -> FeatureData:
    recs = []
    for i in range(30):
        toks = [" The", " city", " of", word if i < 20 else " Rome", " is", " nice", "."]
        acts = [0.0, 0.0, 0.0, (5.0 + i * 0.1) if i < 20 else 0.0, 0.0, 0.0, 0.0]
        recs.append(Record(tokens=[*toks, f" #{i}"], acts=[*acts, 0.0]))
    return FeatureData(unit=unit, records=recs)


def oracle(system: str, prompt: str) -> str:
    """Explains 'Paris'; detects excerpts containing Paris; simulates 10 on Paris tokens."""
    if system.startswith("You label"):
        return json.dumps({"label": "the word Paris", "confidence": 80})
    if system.startswith("You test"):
        lines = [ln for ln in prompt.split("Excerpts:\n", 1)[1].splitlines() if re.match(r"\d+\. ", ln)]
        good = "Paris" in prompt.split("Label:", 1)[1].splitlines()[0]
        return json.dumps({"probs": [(0.9 if "Paris" in ln else 0.1) if good else 0.5 for ln in lines]})
    toks = [ln for ln in prompt.split("Tokens:\n", 1)[1].splitlines() if ln.strip()]
    return json.dumps({"acts": [10 if "Paris" in t else 0 for t in toks]})


def test_split_is_disjoint_and_ordered() -> None:
    sp = split_records(_feature().records, n_explain=5, n_positive=4, n_negative=4)
    assert len(sp.explain) == 5 and len(sp.positive) == 4 and len(sp.negative) == 4
    assert sp.explain[0].peak >= sp.explain[-1].peak >= max(r.peak for r in sp.positive)
    ids = {id(r) for r in sp.explain}
    assert not ids & {id(r) for r in sp.positive}
    assert all(r.peak == 0 for r in sp.negative)


def test_format_marks_active_tokens() -> None:
    r = Record(tokens=[" a", " b"], acts=[0.0, 2.0])
    assert format_record(r) == " a<< b>>(10)"
    assert format_record(r, marks=False) == " a b"


def test_pipeline_with_oracle() -> None:
    llm = ScriptedLLM(oracle)
    lab = label_feature(llm, _feature(), score="both")
    assert lab.label == "the word Paris" and lab.confidence == pytest.approx(0.8)
    assert lab.detection is not None and lab.detection.auroc == 1.0
    assert lab.detection.balanced_accuracy == 1.0
    assert lab.simulation is not None and lab.simulation.correlation > 0.9
    assert lab.score == 1.0


def test_bad_label_scores_at_chance() -> None:
    llm = ScriptedLLM(oracle)
    lab = label_feature(llm, _feature(), label="cooking verbs")
    assert lab.detection is not None and lab.detection.auroc == 0.5


def test_errors_are_kept_not_raised() -> None:
    lab = label_feature(ScriptedLLM(lambda s, p: "no json here"), _feature())
    assert lab.error and "JSON" in lab.error
    empty = FeatureData(unit="1:1", records=[Record(tokens=["a"], acts=[0.0])])
    assert label_feature(ScriptedLLM(oracle), empty).error == "no activating records"


def test_batch_and_external_labels() -> None:
    llm = ScriptedLLM(oracle)
    feats = [_feature("1:1"), _feature("2:2")]
    ours = label_features(llm, feats, workers=2)
    assert [x.unit for x in ours] == ["1:1", "2:2"]
    theirs = score_labels(llm, feats, {"1:1": "Paris, the city", "2:2": "punctuation"})
    comps = compare(ours, theirs)
    assert comps[0].winner == "tie" and comps[1].winner == "ours"
    cal = calibration(ours)
    assert cal.n_features == 2 and cal.detector_ece is not None and cal.bins


def test_cache_avoids_repeat_calls(tmp_path) -> None:
    inner = ScriptedLLM(oracle)
    c = CachedLLM(inner, tmp_path / "cache.jsonl")
    label_feature(c, _feature())
    n = inner.usage.calls
    c2 = CachedLLM(ScriptedLLM(lambda s, p: "should not be called"), tmp_path / "cache.jsonl")
    c2.inner.model = inner.model
    lab = label_feature(c2, _feature())
    assert lab.label == "the word Paris"
    assert c2.inner.usage.calls == 0 and c2.usage.cached_calls == n


def test_metrics() -> None:
    assert auroc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([0.5], [1]) is None
    assert balanced_accuracy([0.9, 0.2, 0.7], [1, 0, 0]) == 0.75
    assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert correlation([1, 1], [1, 2]) is None
    f = isotonic([0.2, 0.5, 0.8], [0.6, 0.5, 0.9])
    assert f(0.2) <= f(0.5) <= f(0.8)


NP = {
    "activations": [
        {"tokens": [" in", " Texas", "."], "values": [0.0, 7.5, 0.1]},
        {"tokens": [" in", " Texas", "."], "values": [0.0, 7.5, 0.1]},
        {"tokens": [" a", " b"], "values": [0.0, -1.0]},
    ],
    "explanations": [
        {"description": " Texas ", "explanationModelName": "gemini-2.0-flash", "typeName": "oai_token-act-pair"}
    ],
}


def test_neuronpedia_parsing_and_export(tmp_path) -> None:
    recs = records_from_neuronpedia(NP)
    assert len(recs) == 2 and recs[1].acts == [0.0, 0.0]
    assert neuronpedia_labels(NP)[0]["description"] == "Texas"
    lab = label_feature(ScriptedLLM(oracle), _feature("20:15589"))
    rows = export_explanations([lab], "gemma-2-2b", "20-gemmascope-transcoder-16k", tmp_path / "x.json")
    assert rows[0]["layer"] == "20-gemmascope-transcoder-16k" and rows[0]["index"] == "15589"
    assert rows[0]["scores"][0]["method"] == "circuitkit-detection-auroc"
    assert json.loads((tmp_path / "x.json").read_text())[0]["description"] == "the word Paris"


def test_cache_is_thread_safe(tmp_path) -> None:
    """Parallel workers must all see a cache filled by an earlier run (no race while it loads)."""
    path = tmp_path / "cache.jsonl"
    feats = [_feature(f"{i}:{i}") for i in range(8)]
    label_features(CachedLLM(ScriptedLLM(oracle), path), feats, workers=8)
    cold = ScriptedLLM(lambda s, p: "should not be called")
    labs = label_features(CachedLLM(cold, path), feats, workers=8)
    assert cold.usage.calls == 0
    assert all(lab.label == "the word Paris" for lab in labs)
