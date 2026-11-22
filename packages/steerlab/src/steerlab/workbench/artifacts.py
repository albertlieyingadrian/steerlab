"""Artifacts: the things you make in the workbench (directions, probes, lenses, edits), saved as JSON + npz.

Think of a lab notebook with a drawer. The JSON page says what the thing is (its type, name, model, layer and how it
was made); the npz in the drawer holds its numbers (a direction vector, a probe's weights, an edit's axis). Small
things (an edit that names three neurons) need no drawer: everything fits on the page.

    store = ArtifactStore("~/.cache/steerlab/inspect/artifacts")
    store.save({"type": "direction", "name": "french", "model": "Qwen/Qwen2.5-0.5B-Instruct", "layer": 14},
               {"vector": np.ones(896)})
    meta, arrays = store.load("direction", "french")
    blob = store.export(["direction/french"])      # a zip: direction/french.json + direction/french.npz
    store.import_bundle(blob)                      # the same files back, in any other store

On disk: `<root>/<type>/<name>.json` and, when there are arrays, `<root>/<type>/<name>.npz`. The JSON always carries
`type`, `name`, `created` (unix seconds) and `arrays` (the npz's keys, empty when there is none). Names are slugs.

numpy only; no torch.
"""

from __future__ import annotations

import builtins
import io
import json
import re
import time
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

KINDS: tuple[str, ...] = ("direction", "probe", "lens", "edit")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
MAX_BUNDLE_BYTES = 256 * 1024 * 1024

Arrays = Mapping[str, npt.ArrayLike]


class ArtifactError(ValueError):
    """A malformed artifact, an unknown kind or name, or a bundle that does not unpack."""


def slug(text: str) -> str:
    """A file-safe name from free text: 'Polite / formal #2' -> 'Polite-formal-2'."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    return out[:80] or "artifact"


def check_meta(meta: Mapping[str, Any]) -> tuple[str, str]:
    kind, name = meta.get("type"), meta.get("name")
    if kind not in KINDS:
        raise ArtifactError(f"artifact type must be one of {KINDS}, not {kind!r}")
    if not isinstance(name, str) or not NAME.match(name):
        raise ArtifactError(f"artifact name {name!r} must be a slug (letters, digits, . _ -; at most 80)")
    return str(kind), name


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def _paths(self, kind: str, name: str) -> tuple[Path, Path]:
        check_meta({"type": kind, "name": name})
        folder = self.root / kind
        return folder / f"{name}.json", folder / f"{name}.npz"

    def save(self, meta: Mapping[str, Any], arrays: Arrays | None = None) -> dict[str, Any]:
        """Write an artifact (replacing one of the same type and name). Returns the stored JSON."""
        kind, name = check_meta(meta)
        json_path, npz_path = self._paths(kind, name)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        stored = dict(meta)
        stored.setdefault("created", time.time())
        stored["arrays"] = sorted(arrays) if arrays else []
        if arrays:
            buffer = io.BytesIO()
            np.savez(buffer, **{key: np.asarray(value) for key, value in arrays.items()})  # type: ignore[arg-type]
            npz_path.write_bytes(buffer.getvalue())
        else:
            npz_path.unlink(missing_ok=True)
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stored, indent=1, default=_jsonable))
        tmp.replace(json_path)
        return stored

    def load(self, kind: str, name: str) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
        json_path, npz_path = self._paths(kind, name)
        if not json_path.exists():
            raise ArtifactError(f"no {kind} artifact named {name!r}")
        meta = json.loads(json_path.read_text())
        arrays: dict[str, npt.NDArray[Any]] = {}
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        missing = set(meta.get("arrays", [])) - set(arrays)
        if missing:
            raise ArtifactError(f"{kind}/{name}: the npz is missing arrays {sorted(missing)}")
        return meta, arrays

    def exists(self, kind: str, name: str) -> bool:
        return self._paths(kind, name)[0].exists()

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Every artifact's JSON (newest first), optionally of one type. Arrays are not read."""
        kinds = [kind] if kind else list(KINDS)
        out = []
        for k in kinds:
            if k not in KINDS:
                raise ArtifactError(f"unknown artifact type {k!r}")
            folder = self.root / k
            if not folder.exists():
                continue
            for path in folder.glob("*.json"):
                try:
                    out.append(json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
        return sorted(out, key=lambda meta: meta.get("created", 0), reverse=True)

    def delete(self, kind: str, name: str) -> None:
        for path in self._paths(kind, name):
            path.unlink(missing_ok=True)

    # Moving artifacts between machines

    def export(self, keys: Iterable[str]) -> bytes:
        """A zip of `<type>/<name>.json` (+ `.npz`) for each `type/name` key."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for key in keys:
                kind, _, name = key.partition("/")
                json_path, npz_path = self._paths(kind, name)
                if not json_path.exists():
                    raise ArtifactError(f"no {kind} artifact named {name!r}")
                archive.write(json_path, f"{kind}/{name}.json")
                if npz_path.exists():
                    archive.write(npz_path, f"{kind}/{name}.npz")
        return buffer.getvalue()

    def import_bundle(self, blob: bytes) -> builtins.list[str]:
        """Unpack an `export` zip into this store. Returns the `type/name` keys written."""
        if len(blob) > MAX_BUNDLE_BYTES:
            raise ArtifactError("bundle is too large")
        try:
            archive = zipfile.ZipFile(io.BytesIO(blob))
        except zipfile.BadZipFile:
            raise ArtifactError("not a zip bundle") from None
        written = []
        with archive:
            names = set(archive.namelist())
            for member in sorted(names):
                if not member.endswith(".json"):
                    continue
                meta = _parse_json(archive.read(member), member)
                kind, name = check_meta(meta)
                npz_name = member[: -len(".json")] + ".npz"
                arrays = _read_npz(archive.read(npz_name), npz_name) if npz_name in names else None
                self.save(meta, arrays)
                written.append(f"{kind}/{name}")
        if not written:
            raise ArtifactError("the bundle holds no artifact JSON")
        return written

    def import_files(self, json_bytes: bytes, npz_bytes: bytes | None = None) -> str:
        """One artifact from its JSON (and npz, when it has arrays). Inline lists in the JSON need no npz."""
        meta = _parse_json(json_bytes, "artifact JSON")
        kind, name = check_meta(meta)
        arrays = _read_npz(npz_bytes, f"{name}.npz") if npz_bytes else None
        self.save(meta, arrays)
        return f"{kind}/{name}"


def _parse_json(raw: bytes, where: str) -> dict[str, Any]:
    try:
        meta = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactError(f"{where}: not valid JSON ({error})") from None
    if not isinstance(meta, dict):
        raise ArtifactError(f"{where}: an artifact is a JSON object")
    return meta


def _read_npz(raw: bytes, where: str) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise ArtifactError(f"{where}: not a readable npz ({error})") from None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")
