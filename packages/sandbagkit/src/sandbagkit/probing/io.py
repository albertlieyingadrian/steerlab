"""Portable save and load: numbers in a `.npz`, everything else in a JSON file next to it. No pickle.

    probe.save("outputs/probe_l14")   writes   outputs/probe_l14.npz    arrays (coef, intercept, ...)
                                               outputs/probe_l14.json   kind, format version, shapes, sha256, metadata

Why not pickle (what the reference toolkit used)? Unpickling runs arbitrary code, so a downloaded `.pkl` is a
program, not data. It also breaks when a class moves or a library changes version. A `.npz` is a zip of `.npy`
files that numpy (and other languages) read without running anything, and the JSON is readable by eye.

The JSON carries the npz file's sha256, so a load notices when the two files got separated or mixed up.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT_NAME = "sandbagkit.probing"
FORMAT_VERSION = "1.0"
"""Version of the on-disk format. Bump the major part on any change an older reader can't handle."""

SUPPORTED_FORMAT_MAJORS = frozenset({"1"})


class ArtifactFormatError(ValueError):
    """A saved file is missing, of the wrong kind, from an unsupported format version, or corrupted."""


def artifact_paths(path: str | Path) -> tuple[Path, Path]:
    """(`x.npz`, `x.json`) for `x`, `x.npz` or `x.json`."""
    base = Path(path)
    if base.suffix in (".npz", ".json"):
        base = base.with_suffix("")
    return base.with_name(base.name + ".npz"), base.with_name(base.name + ".json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("sandbagkit")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def save_arrays(
    path: str | Path, arrays: Mapping[str, npt.ArrayLike], *, kind: str, metadata: Mapping[str, Any] | None = None
) -> tuple[Path, Path]:
    """Write `arrays` to `<path>.npz` and a JSON description (with `metadata`) to `<path>.json`.

    Arrays must be numeric or boolean: object arrays would need pickle, so they are refused. `metadata` must be
    JSON-serializable. Both files are written to a temporary name first and then renamed, so a crash never
    leaves a half-written file under the real name.
    """
    npz_path, json_path = artifact_paths(path)
    clean: dict[str, npt.NDArray[Any]] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise TypeError(f"array {name!r} has dtype {array.dtype}; only numeric and boolean arrays are portable")
        clean[name] = array

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
    with tmp_npz.open("wb") as f:
        np.savez(f, **clean)  # type: ignore[arg-type]
    description = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(),
        "sandbagkit_version": _package_version(),
        "npz_sha256": _sha256(tmp_npz),
        "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in clean.items()},
        "metadata": dict(metadata or {}),
    }
    tmp_json = json_path.with_name(json_path.name + ".tmp")
    tmp_json.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_npz, npz_path)
    os.replace(tmp_json, json_path)
    return npz_path, json_path


def load_arrays(path: str | Path, *, kind: str | None = None) -> tuple[dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """Read what `save_arrays` wrote. Returns (arrays, metadata).

    Checks, in order: both files exist, the format and its major version are supported, `kind` matches (when
    given), and the npz file's sha256 matches the one recorded at save time.
    """
    npz_path, json_path = artifact_paths(path)
    for p in (npz_path, json_path):
        if not p.exists():
            raise ArtifactFormatError(f"{p} not found (a saved artifact is a .npz plus a .json with the same name)")
    try:
        description = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactFormatError(f"{json_path} is not valid JSON: {error}") from error
    if not isinstance(description, dict) or description.get("format") != FORMAT_NAME:
        raise ArtifactFormatError(f"{json_path} is not a {FORMAT_NAME} file")
    version = str(description.get("format_version", ""))
    if version.split(".", 1)[0] not in SUPPORTED_FORMAT_MAJORS:
        raise ArtifactFormatError(
            f"{json_path} has format_version '{version}', but this sandbagkit reads "
            f"{sorted(SUPPORTED_FORMAT_MAJORS)}.x (current {FORMAT_VERSION})"
        )
    if kind is not None and description.get("kind") != kind:
        raise ArtifactFormatError(f"{json_path} holds a {description.get('kind')!r}, not a {kind!r}")
    if _sha256(npz_path) != description.get("npz_sha256"):
        raise ArtifactFormatError(f"{npz_path} does not match the checksum in {json_path}: the files were mixed up")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = description.get("metadata", {})
    return arrays, dict(metadata) if isinstance(metadata, dict) else {}


def save_activations(
    path: str | Path, activations: Mapping[int, npt.ArrayLike], *, metadata: Mapping[str, Any] | None = None
) -> tuple[Path, Path]:
    """Save `{layer: [n_texts, hidden]}` (what `extract_activations` returns) as `layer_<n>` arrays."""
    arrays = {f"layer_{int(layer)}": value for layer, value in activations.items()}
    meta = {"layers": sorted(int(layer) for layer in activations), **dict(metadata or {})}
    return save_arrays(path, arrays, kind="activations", metadata=meta)


def load_activations(path: str | Path) -> tuple[dict[int, npt.NDArray[Any]], dict[str, Any]]:
    """Read `save_activations` output back as ({layer: array}, metadata)."""
    arrays, metadata = load_arrays(path, kind="activations")
    by_layer = {int(name.removeprefix("layer_")): value for name, value in arrays.items() if name.startswith("layer_")}
    return dict(sorted(by_layer.items())), metadata
