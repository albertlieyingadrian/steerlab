"""The feature cache: decoder directions, calibration scales and labels, per model, release and layer.

Layout under `steerlab_cache_dir()` (default `~/.cache/steerlab`, override with `STEERLAB_CACHE`):

    features/<model>/<release>/layer<N>/<id>.npy      the feature's decoder direction as a float32 unit vector
    features/<model>/<release>/layer<N>/<id>.json     {"release", "layer", "feature", "d_model", "scale", "stored_at"}
    labels/<model>/<release>/layer<N>/<id>.json       {"feature", "label", "source", "created_at"}   (one feature)
    labels/<model>/<release>/layer<N>/neuronpedia.jsonl   {"feature", "description"} per line        (a whole export)

Why cache directions at all: steering a known feature should not need the full SAE (537 MB for one Qwen-Scope
layer) in memory. Once `features contrast` or `features search` has touched a feature, its direction is a 4 KB file
and `chat` can steer it after reading just that. Think of the SAE as the dictionary and the cache as the handful of
pages you tore out.

Reading a direction needs torch (the result is a `FeatureDirection`); everything else here is plain files, so
`features search` and `features show` work from the light install.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerlab import defaults, hub

if TYPE_CHECKING:
    from steerlab.saes.common import FeatureDirection

EXPORT_FILENAME = "neuronpedia.jsonl"
EXPORT_META_FILENAME = "neuronpedia.meta.json"


def model_slug(model: str) -> str:
    """The folder name for a model: the short key ('qwen', 'gemma') when known, else the repo id with '/' as '--'."""
    if model in defaults.MODELS:
        return model
    for key, repo_id in defaults.MODELS.items():
        if model == repo_id:
            return key
    return model.replace("/", "--")


def release_slug(release: str) -> str:
    return release.replace("/", "--")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write to a temporary name, then rename: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _write_atomic(path, (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


class FeatureCache:
    """One cache root. The default root is `steerlab_cache_dir()`; tests pass a tmp path."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else hub.steerlab_cache_dir()
        self._exports: dict[Path, dict[int, str]] = {}

    def features_dir(self, model: str, release: str, layer: int) -> Path:
        return self.root / "features" / model_slug(model) / release_slug(release) / f"layer{layer}"

    def labels_dir(self, model: str, release: str, layer: int) -> Path:
        return self.root / "labels" / model_slug(model) / release_slug(release) / f"layer{layer}"

    def direction_path(self, model: str, release: str, layer: int, feature: int) -> Path:
        return self.features_dir(model, release, layer) / f"{feature}.npy"

    def _direction_meta_path(self, model: str, release: str, layer: int, feature: int) -> Path:
        return self.features_dir(model, release, layer) / f"{feature}.json"

    def has_direction(self, model: str, release: str, layer: int, feature: int) -> bool:
        return self.direction_path(model, release, layer, feature).is_file()

    def direction(self, model: str, release: str, layer: int, feature: int) -> FeatureDirection | None:
        """The cached unit direction with its label and calibration scale, or None when it was never stored."""
        path = self.direction_path(model, release, layer, feature)
        if not path.is_file():
            return None
        import numpy as np
        import torch

        from steerlab.saes.common import FeatureDirection

        unit = np.load(path, allow_pickle=False).astype(np.float32)
        return FeatureDirection(
            unit=torch.from_numpy(unit),
            layer=layer,
            release=release,
            feature=feature,
            label=self.label(model, release, layer, feature),
            scale=self.scale(model, release, layer, feature),
        )

    def store_direction(self, model: str, release: str, layer: int, direction: FeatureDirection) -> Path:
        """Store a direction's unit vector (and its label and scale when it carries them). Returns the .npy path."""
        import numpy as np
        import torch

        if direction.layer != layer:
            raise ValueError(f"direction is for layer {direction.layer}, not layer {layer}")
        path = self.direction_path(model, release, layer, direction.feature)
        unit = direction.unit.detach().to(device="cpu", dtype=torch.float32).numpy()
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(unit), allow_pickle=False)
        _write_atomic(path, buffer.getvalue())
        meta_path = self._direction_meta_path(model, release, layer, direction.feature)
        meta = _read_json(meta_path) or {}
        meta.update(
            {
                "release": release,
                "layer": layer,
                "feature": direction.feature,
                "d_model": int(unit.shape[0]),
                "scale": direction.scale if direction.scale is not None else meta.get("scale"),
                "stored_at": _now(),
            }
        )
        _write_json(meta_path, meta)
        if direction.label and self.label(model, release, layer, direction.feature) is None:
            self.store_label(model, release, layer, direction.feature, direction.label, source="direction")
        return path

    def scale(self, model: str, release: str, layer: int, feature: int) -> float | None:
        meta = _read_json(self._direction_meta_path(model, release, layer, feature))
        value = None if meta is None else meta.get("scale")
        return float(value) if value is not None else None

    def store_scale(self, model: str, release: str, layer: int, feature: int, scale: float, *, source: str) -> Path:
        path = self._direction_meta_path(model, release, layer, feature)
        meta = _read_json(path) or {"release": release, "layer": layer, "feature": feature}
        meta.update({"scale": float(scale), "scale_source": source, "scale_stored_at": _now()})
        _write_json(path, meta)
        return path

    def label_path(self, model: str, release: str, layer: int, feature: int) -> Path:
        return self.labels_dir(model, release, layer) / f"{feature}.json"

    def label(self, model: str, release: str, layer: int, feature: int) -> str | None:
        """A feature's label: a stored per-feature label first, else the cached export's description, else None."""
        meta = _read_json(self.label_path(model, release, layer, feature))
        if meta is not None and meta.get("label"):
            return str(meta["label"])
        export = self.export_labels(model, release, layer)
        if export is not None:
            return export.get(feature)
        return None

    def label_source(self, model: str, release: str, layer: int, feature: int) -> str | None:
        meta = _read_json(self.label_path(model, release, layer, feature))
        if meta is not None and meta.get("label"):
            return str(meta.get("source", "unknown"))
        export = self.export_labels(model, release, layer)
        if export is not None and feature in export:
            export_meta = _read_json(self.labels_dir(model, release, layer) / EXPORT_META_FILENAME) or {}
            return str(export_meta.get("source", "export"))
        return None

    def store_label(self, model: str, release: str, layer: int, feature: int, label: str, *, source: str) -> Path:
        path = self.label_path(model, release, layer, feature)
        _write_json(path, {"feature": feature, "label": label, "source": source, "created_at": _now()})
        return path

    def export_path(self, model: str, release: str, layer: int) -> Path:
        return self.labels_dir(model, release, layer) / EXPORT_FILENAME

    def export_labels(self, model: str, release: str, layer: int) -> dict[int, str] | None:
        """`{feature: description}` from the cached export, or None when no export is cached."""
        path = self.export_path(model, release, layer)
        if path in self._exports:
            return self._exports[path]
        if not path.is_file():
            return None
        labels: dict[int, str] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    labels[int(row["feature"])] = str(row["description"])
        self._exports[path] = labels
        return labels

    def store_export_labels(
        self, model: str, release: str, layer: int, labels: Mapping[int, str], *, source: str
    ) -> Path:
        """Store `{feature: description}` as one JSON row per line (nothing else from the export is kept)."""
        path = self.export_path(model, release, layer)
        lines = [
            json.dumps({"feature": int(feature), "description": labels[feature]}, ensure_ascii=False)
            for feature in sorted(labels)
        ]
        _write_atomic(path, ("\n".join(lines) + "\n").encode("utf-8"))
        meta_path = self.labels_dir(model, release, layer) / EXPORT_META_FILENAME
        _write_json(meta_path, {"source": source, "n": len(labels), "stored_at": _now()})
        self._exports.pop(path, None)
        return path


@lru_cache(maxsize=4)
def _default_cache(root: Path) -> FeatureCache:
    """One FeatureCache per root, so repeated lookups reuse its parsed export instead of re-reading the JSONL."""
    return FeatureCache(root)


def direction(model: str, release: str, layer: int, feature: int) -> FeatureDirection | None:
    """The cached direction of one feature from the default cache, or None. Needs no SAE in memory."""
    return _default_cache(hub.steerlab_cache_dir()).direction(model, release, layer, feature)


def store_direction(model: str, release: str, layer: int, direction: FeatureDirection) -> Path:
    """Store one feature's direction in the default cache so a later `chat` can steer it without the SAE."""
    return _default_cache(hub.steerlab_cache_dir()).store_direction(model, release, layer, direction)
