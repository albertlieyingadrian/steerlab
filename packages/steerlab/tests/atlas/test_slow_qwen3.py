"""Slow: the catalog and our own code agree on Qwen3-0.6B (downloads catalog tables; needs the model in the HF cache).

For each of a few labelled units, re-run its catalog top contexts (as truncated text) through Qwen3-0.6B in fp32 and
read the unit at the firing token. If our neuron indices and pole signs match the catalog's, the unit lands in the top
1% of its own catalog distribution there; a random unit of the same layer, read at the same tokens, is the control.
Measured on 2026-11-22: 23 of 24 contexts past the 99th percentile (median 99.98).
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("torch")
pytest.importorskip("pyarrow")


def test_catalog_top_contexts_fire_our_qwen3_units(tmp_path_factory: pytest.TempPathFactory) -> None:
    from steerlab.atlas.address import Address
    from steerlab.atlas.cards import percentile
    from steerlab.atlas.catalog import Catalog, HubSource
    from steerlab.atlas.compute import Parts, unit_trace
    from steerlab.inspect.core import Inspector

    inspector = Inspector.load("Qwen/Qwen3-0.6B")
    parts = Parts(inspector.model, inspector.tokenizer)
    catalog = Catalog(HubSource(Path(tmp_path_factory.mktemp("catalog"))))
    rng = random.Random(0)
    ours, control = [], []
    for layer in (5, 14, 24):
        for unit in catalog.overview("qwen3-0.6b", top=3, layers=[layer])["layers"][0]["units"]:
            address = Address.parse(unit["address"])
            card = catalog.unit("qwen3-0.6b", address)
            other = Address.neuron(layer, rng.randrange(3072), rng.choice((1, -1)))
            other_stats = catalog.unit("qwen3-0.6b", other)["stats"]
            for context in card["contexts"][:2]:
                text = (context["pre"] + context["tok"]).replace("⏎", "\n")
                ids = parts.tokenizer(text, add_special_tokens=False).input_ids
                value = unit_trace(parts, ids, layer, "neuron", address.index, address.pole)[-1]
                ours.append(percentile(value, card["stats"]) or 0.0)
                rand = unit_trace(parts, ids, layer, "neuron", other.index, other.pole)[-1]
                control.append(percentile(rand, other_stats) or 0.0)
    assert statistics.median(ours) > 99.0, ours
    assert sum(p > 99.0 for p in ours) >= 0.8 * len(ours), ours
    assert statistics.median(control) < 99.0, control
