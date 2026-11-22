"""Probes: the bundle reads, a fit is deterministic, the random baseline is there, the cache reloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from steerlab.workbench.artifacts import ArtifactStore
from steerlab.workbench.probes import Probe, ProbeBank, fit_concept, fit_probe, load_bundle, spec_from_examples
from steerlab.workbench.session import Workbench

from .conftest import FIT


def test_the_bundle_has_the_promised_concepts() -> None:
    names = [spec.name for spec in load_bundle()]
    assert len(names) == 16 and len(set(names)) == 16
    for wanted in (
        "sandbagging",
        "refusal",
        "honest.deceptive",
        "ai-register.plain",
        "positive.negative",
        "formal.casual",
        "code.prose",
        "french.english",
    ):
        assert wanted in names
    sandbag = next(spec for spec in load_bundle() if spec.name == "sandbagging")
    assert sandbag.source == "steerlab.data:sandbagging" and len(sandbag.positive) == len(sandbag.negative) > 4


def test_fit_is_deterministic_on_synthetic_data() -> None:
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, (10, 16)), rng.normal(-1, 1, (10, 16))
    a = fit_probe("x", pos, neg, layer=3, model_id="m")
    b = fit_probe("x", pos, neg, layer=3, model_id="m")
    assert np.array_equal(a.direction, b.direction) and np.array_equal(a.random_dirs, b.random_dirs)
    assert a.d_prime > 2 and a.accuracy == 1.0
    assert abs(float(np.linalg.norm(a.direction)) - 1) < 1e-6
    assert a.random_dirs.shape == (16, 16) and float(np.abs(a.random_d_prime).max()) < a.d_prime
    z = a.read(pos)
    assert z.shape == (10,) and z.mean() > 0 > a.read(neg).mean()
    assert a.random_band(pos).shape == (10,)


def test_fit_on_the_tiny_model_is_deterministic_and_round_trips(tiny: object, tmp_path: Path) -> None:
    spec = next(s for s in load_bundle() if s.name == "formal.casual")
    one = fit_concept(tiny.model, tiny.tokenizer, spec, model_id="tiny", layer=1)  # type: ignore[attr-defined]
    two = fit_concept(tiny.model, tiny.tokenizer, spec, model_id="tiny", layer=1)  # type: ignore[attr-defined]
    assert np.array_equal(one.direction, two.direction) and one.mid == two.mid and one.sd == two.sd
    assert set(one.d_prime_by_layer) == {0, 1}
    meta, arrays = one.to_artifact()
    again = Probe.from_artifact(meta, arrays)
    assert np.array_equal(again.direction, one.direction) and again.layer == 1
    bank = ProbeBank(ArtifactStore(tmp_path), "tiny")
    bank.add(one)
    fresh = ProbeBank(ArtifactStore(tmp_path), "tiny")
    assert fresh.load_cached() == ["formal.casual"]
    assert ProbeBank(ArtifactStore(tmp_path), "another-model").load_cached() == []


def test_bundle_is_cached_and_custom_probes_fit(bench: Workbench, tiny: object) -> None:
    assert sorted(bench.bank.probes) == sorted(FIT)
    assert {s["name"] for s in bench.steerables()} >= {f"probe:{n}" for n in FIT}
    again = Workbench(tiny.model, tiny.tokenizer, "tiny-random-qwen2", bench.cache_dir)  # type: ignore[attr-defined]
    assert sorted(again.bank.probes) == sorted(FIT)  # loaded from the cache, nothing refitted
    spec = spec_from_examples("Shouting", ["HELLO THERE", "STOP IT NOW"], ["hello there", "stop it now"])
    probe = bench.fit_custom(spec)
    assert probe.name == "Shouting" and "probe:Shouting" in bench.registry.directions
    with pytest.raises(ValueError):
        spec_from_examples("x", ["only one"], ["a", "b"])
    reading = bench.read_text("Bonjour", probes=["french.english"])
    assert len(reading["probes"]["french.english"]) == len(reading["tokens"]) == len(reading["bands"]["french.english"])
