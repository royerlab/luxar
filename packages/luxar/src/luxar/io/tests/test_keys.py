"""Per-element ``keys`` CSR round-trip (issue #1917).

The sibling of ``test_labels.py``. ``keys`` shares the serializer, the spatial
permutation and the ladder rules with ``labels`` — which is the point, and also
the risk: sharing means a change to one silently changes the other, so both
channels are pinned independently rather than one being assumed from the other.

What matters about a key is that it stays paired with ITS element. A key that
survives the write but drifts by one row produces links that resolve, look
plausible, and point at the wrong record — so every test here checks pairing,
not merely presence.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def _make_3d_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )


def _decode(zarr_path: str, node_name: str, prefix: str = "key") -> List[str]:
    """Decode a CSR string channel straight off disk."""
    root = zarr.open_group(str(zarr_path), mode="r")
    group = root
    for part in node_name.split("/"):
        group = group[part]
    offsets = np.asarray(group[f"{prefix}_offsets"][:])
    data = np.asarray(group[f"{prefix}_bytes"][:])
    out = []
    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        out.append("" if start == end else bytes(data[start:end]).decode("utf-8"))
    return out


class TestKeysCSRRoundTrip:
    def test_basic_keys(self, tmp_path: Path) -> None:
        path = str(tmp_path / "t.luxar.zarr")
        positions = np.random.rand(5, 3).astype(np.float32)
        keys = ["P04637", "Q9Y6K9", "O15111", "P00533", "P01308"]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, keys=keys)

        assert _decode(path, "pts") == keys
        root = zarr.open_group(path, mode="r")
        assert root["pts"].attrs["has_keys"] is True

    def test_no_keys_by_default(self, tmp_path: Path) -> None:
        path = str(tmp_path / "t.luxar.zarr")
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", np.random.rand(4, 3).astype(np.float32))
        root = zarr.open_group(path, mode="r")
        assert "key_offsets" not in root["pts"]
        assert "has_keys" not in dict(root["pts"].attrs)

    def test_keys_without_labels(self, tmp_path: Path) -> None:
        """A link built from `{hover_key}` alone needs no labels at all."""
        path = str(tmp_path / "t.luxar.zarr")
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "pts", np.random.rand(3, 3).astype(np.float32), keys=["a", "b", "c"]
            )
        root = zarr.open_group(path, mode="r")
        assert root["pts"].attrs["has_keys"] is True
        assert "has_labels" not in dict(root["pts"].attrs)

    def test_labels_and_keys_are_independent_channels(self, tmp_path: Path) -> None:
        """Both present, different values, neither clobbering the other.

        The failure this guards is a shared-serializer bug writing one channel's
        content under the other's array names.
        """
        path = str(tmp_path / "t.luxar.zarr")
        labels = ["P04637 · tumour suppressor", "Q9Y6K9 · kinase"]
        keys = ["P04637", "Q9Y6K9"]
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "pts", np.random.rand(2, 3).astype(np.float32), labels=labels, keys=keys
            )
        assert _decode(path, "pts", "key") == keys
        assert _decode(path, "pts", "label") == labels

    def test_empty_keys_are_null(self, tmp_path: Path) -> None:
        path = str(tmp_path / "t.luxar.zarr")
        keys = ["A", "", "C"]
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", np.random.rand(3, 3).astype(np.float32), keys=keys)
        assert _decode(path, "pts") == keys

    def test_unicode_keys(self, tmp_path: Path) -> None:
        path = str(tmp_path / "t.luxar.zarr")
        keys = ["Hello", "细胞", "🧬"]
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", np.random.rand(3, 3).astype(np.float32), keys=keys)
        assert _decode(path, "pts") == keys

    def test_key_length_mismatch_raises(self, tmp_path: Path) -> None:
        path = str(tmp_path / "t.luxar.zarr")
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises(ValueError, match="[Kk]eys"):
                scene.add_points(
                    "pts", np.random.rand(5, 3).astype(np.float32), keys=["a", "b"]
                )

    def test_keys_follow_the_spatial_permutation_paired_with_labels(
        self, tmp_path: Path
    ) -> None:
        """With spatial ordering ON, key[i] must still describe element i.

        Checked by PAIRING, not by multiset equality: a permutation applied to
        one channel and not the other passes a "same set of strings" assertion
        while pairing every key to the wrong element.
        """
        path = str(tmp_path / "t.luxar.zarr")
        n = 64
        rng = np.random.RandomState(0)
        positions = (rng.rand(n, 3) * 100).astype(np.float32)
        # Label and key encode the SAME index, in different formats, so a
        # divergence between the two permutations is visible per row.
        labels = [f"label-{i}" for i in range(n)]
        keys = [f"{i}" for i in range(n)]

        with LuxarZarrCompiler(path, enable_spatial_index=True) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels, keys=keys)

        got_labels = _decode(path, "pts", "label")
        got_keys = _decode(path, "pts", "key")
        assert len(got_keys) == n
        for lab, key in zip(got_labels, got_keys):
            assert lab == f"label-{key}", f"key {key!r} paired with label {lab!r}"

    def test_keys_on_every_geometry_type(self, tmp_path: Path) -> None:
        path = str(tmp_path / "t.luxar.zarr")
        rng = np.random.RandomState(1)
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "pts", rng.rand(3, 3).astype(np.float32), keys=["p0", "p1", "p2"]
            )
            scene.add_lines(
                "lns",
                np.array(
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32
                ),
                widths=np.full(4, 0.1, dtype=np.float32),
                keys=["l0", "l1", "l2", "l3"],
            )
            scene.add_gsplats(
                "gs",
                centers=rng.rand(2, 3).astype(np.float32),
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)
                ),
                keys=["g0", "g1"],
            )
            scene.add_mesh(
                "surf",
                vertices=np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
                ),
                faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32),
                keys=["m0", "m1", "m2", "m3"],
            )

        assert _decode(path, "pts") == ["p0", "p1", "p2"]
        assert _decode(path, "lns") == ["l0", "l1", "l2", "l3"]
        assert _decode(path, "gs") == ["g0", "g1"]
        assert _decode(path, "surf") == ["m0", "m1", "m2", "m3"]


class TestKeysUnderDecomposition:
    def test_partition_gives_every_leaf_its_own_slice(self, tmp_path: Path) -> None:
        """Under `partition=` the CSR lives on each `part_<i>`, never the wrapper —
        which is what the viewer relies on, since a pick hits a leaf."""
        path = str(tmp_path / "t.luxar.zarr")
        n = 200
        rng = np.random.RandomState(2)
        positions = (rng.rand(n, 3) * 10).astype(np.float32)
        keys = [f"K{i:03d}" for i in range(n)]

        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "tiled", positions, keys=keys, partition={"max_elements": 50}
            )

        root = zarr.open_group(path, mode="r")
        parts = sorted(k for k in root["tiled"].keys() if k.startswith("part_"))
        assert len(parts) > 1, f"expected a real partition, got {parts}"
        assert "key_offsets" not in root["tiled"], "wrapper must hold no CSR"

        seen: List[str] = []
        for part in parts:
            seen += _decode(path, f"tiled/{part}")
        assert sorted(seen) == sorted(keys)

    def test_additive_ladder_puts_one_union_on_the_parent(self, tmp_path: Path) -> None:
        """Mirrors the labels contract (#1422): the union CSR lives on the ladder
        parent and the `additive_<i>` levels carry none."""
        path = str(tmp_path / "t.luxar.zarr")
        n = 3000
        rng = np.random.RandomState(3)
        positions = (rng.rand(n, 3) * 50).astype(np.float32)
        keys = [f"L{i:04d}" for i in range(n)]

        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("stream", positions, keys=keys, additive_lod={"n_lods": 3})

        root = zarr.open_group(path, mode="r")
        parent = root["stream"]
        assert parent.attrs["has_keys"] is True
        assert sorted(_decode(path, "stream")) == sorted(keys)
        for level in (k for k in parent.keys() if k.startswith("additive_")):
            assert "key_offsets" not in parent[level], f"{level} must carry no CSR"
            assert "has_keys" not in dict(parent[level].attrs)

    def test_partial_ladder_keys_are_rejected(self, tmp_path: Path) -> None:
        """All-or-nothing across the ladder, like labels: a partial union would
        misalign every slot after the first keyless level."""
        from luxar.io._compiler.labels.text_labels import validate_ladder_labels

        levels = [
            {"positions": np.zeros((2, 3), dtype=np.float32), "keys": ["a", "b"]},
            {"positions": np.zeros((2, 3), dtype=np.float32)},
        ]
        with pytest.raises(ValueError, match="keys"):
            validate_ladder_labels(levels, "positions", channel="keys")
