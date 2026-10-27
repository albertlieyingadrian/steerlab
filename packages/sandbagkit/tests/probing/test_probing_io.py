"""Portable persistence (.npz + .json, no pickle) and the lazy-import promise."""

import json
import subprocess
import sys

import numpy as np
import pytest

from sandbagkit.probing import io


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        arrays = {"w": np.arange(6, dtype=np.float64).reshape(2, 3), "flags": np.array([True, False])}
        io.save_arrays(tmp_path / "thing", arrays, kind="demo", metadata={"layer": 14, "note": "hi"})
        loaded, metadata = io.load_arrays(tmp_path / "thing", kind="demo")
        assert set(loaded) == {"w", "flags"}
        np.testing.assert_array_equal(loaded["w"], arrays["w"])
        assert loaded["w"].dtype == np.float64
        assert metadata == {"layer": 14, "note": "hi"}

    def test_json_is_human_readable_and_self_describing(self, tmp_path):
        _, meta = io.save_arrays(tmp_path / "thing", {"w": np.zeros((2, 3))}, kind="demo")
        description = json.loads(meta.read_text())
        assert description["format"] == "sandbagkit.probing"
        assert description["format_version"] == io.FORMAT_VERSION
        assert description["kind"] == "demo"
        assert description["arrays"] == {"w": {"shape": [2, 3], "dtype": "float64"}}
        assert len(description["npz_sha256"]) == 64
        assert "T" in description["created_at"]  # ISO 8601

    @pytest.mark.parametrize("given", ["thing", "thing.npz", "thing.json"])
    def test_any_of_the_three_names_works(self, tmp_path, given):
        io.save_arrays(tmp_path / "thing", {"w": np.ones(2)}, kind="demo")
        arrays, _ = io.load_arrays(tmp_path / given)
        assert arrays["w"].tolist() == [1.0, 1.0]

    def test_object_arrays_are_refused(self, tmp_path):
        with pytest.raises(TypeError, match="portable"):
            io.save_arrays(tmp_path / "bad", {"o": np.array([{"a": 1}], dtype=object)}, kind="demo")

    def test_no_pickle_in_the_file(self, tmp_path):
        npz, _ = io.save_arrays(tmp_path / "thing", {"w": np.ones(3)}, kind="demo")
        with np.load(npz, allow_pickle=False) as data:  # would raise if anything needed unpickling
            assert data["w"].sum() == 3

    def test_no_temporary_files_left_behind(self, tmp_path):
        io.save_arrays(tmp_path / "thing", {"w": np.ones(3)}, kind="demo")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["thing.json", "thing.npz"]


class TestLoadErrors:
    def test_missing_json(self, tmp_path):
        _, meta = io.save_arrays(tmp_path / "thing", {"w": np.ones(3)}, kind="demo")
        meta.unlink()
        with pytest.raises(io.ArtifactFormatError, match=r"thing\.json not found"):
            io.load_arrays(tmp_path / "thing")

    def test_wrong_kind(self, tmp_path):
        io.save_arrays(tmp_path / "thing", {"w": np.ones(3)}, kind="steering_vector")
        with pytest.raises(io.ArtifactFormatError, match="not a 'linear_probe'"):
            io.load_arrays(tmp_path / "thing", kind="linear_probe")

    def test_future_major_version(self, tmp_path):
        _, meta = io.save_arrays(tmp_path / "thing", {"w": np.ones(3)}, kind="demo")
        description = json.loads(meta.read_text())
        description["format_version"] = "2.0"
        meta.write_text(json.dumps(description))
        with pytest.raises(io.ArtifactFormatError, match=r"format_version '2\.0'"):
            io.load_arrays(tmp_path / "thing")

    def test_mixed_up_files_fail_the_checksum(self, tmp_path):
        io.save_arrays(tmp_path / "a", {"w": np.ones(3)}, kind="demo")
        io.save_arrays(tmp_path / "b", {"w": np.zeros(3)}, kind="demo")
        (tmp_path / "a.npz").write_bytes((tmp_path / "b.npz").read_bytes())
        with pytest.raises(io.ArtifactFormatError, match="checksum"):
            io.load_arrays(tmp_path / "a")

    def test_not_ours(self, tmp_path):
        (tmp_path / "x.npz").write_bytes(b"")
        (tmp_path / "x.json").write_text('{"format": "something-else"}')
        with pytest.raises(io.ArtifactFormatError, match=r"not a sandbagkit\.probing file"):
            io.load_arrays(tmp_path / "x")


class TestActivations:
    def test_round_trip_by_layer(self, tmp_path):
        acts = {14: np.ones((3, 4), dtype=np.float32), 2: np.zeros((3, 4), dtype=np.float32)}
        io.save_activations(tmp_path / "acts", acts, metadata={"model": "tiny"})
        loaded, metadata = io.load_activations(tmp_path / "acts")
        assert list(loaded) == [2, 14]
        np.testing.assert_array_equal(loaded[14], acts[14])
        assert loaded[14].dtype == np.float32
        assert metadata == {"layers": [2, 14], "model": "tiny"}


def test_importing_probing_does_not_load_torch_or_sklearn():
    # A fresh interpreter, because this test session has already imported torch.
    code = (
        "import sys, json\n"
        "import sandbagkit.probing\n"
        "print(json.dumps([m for m in ['torch', 'transformers', 'sklearn'] if m in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
