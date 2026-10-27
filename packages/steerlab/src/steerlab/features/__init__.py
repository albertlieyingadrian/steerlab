"""Finding SAE features without knowing their IDs.

- `search`: Gemma, offline text search over a cached Neuronpedia label export.
- `contrast`: Qwen (or any SAE), rank features by how much more they fire on text B than on text A.
- `labels`: one-line labels for unlabelled features, from a local judge, cached per feature.
- `cache`: the on-disk store for labels, decoder directions and calibration scales.

Importing this package does not import torch; `contrast`, `labels` and the direction half of `cache` do.
"""

from __future__ import annotations

__all__ = ["cache", "contrast", "labels", "search"]
