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


class TestKeysOnLabelRefusingPaths:
    """Paths that refuse ``labels`` must refuse ``keys`` on the same terms.

    A mesh reveal ladder cannot store either channel — each level re-indexes its
    own vertices, so the union index space a CSR would need is ill-defined. The
    guard originally named only ``labels`` and ``image_labels``, which did not
    produce a ladder carrying keys: it produced a ladder that DROPPED them, with
    no warning and no attr on disk to notice afterwards.
    """

    VERTS = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    FACES = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.uint32)
    KEYS = [f"v{i}" for i in range(6)]

    def test_mesh_reveal_ladder_degrades_rather_than_dropping_keys(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "m.luxar.zarr")
        with pytest.warns(UserWarning, match="reveal ladder cannot be honoured"):
            with LuxarZarrCompiler(path) as compiler:
                scene = compiler.create_scene(dimensions=_make_3d_dims())
                scene.add_mesh(
                    "surf",
                    vertices=self.VERTS,
                    faces=self.FACES,
                    keys=self.KEYS,
                    additive_lod={"method": "radial"},
                )

        root = zarr.open_group(path, mode="r")
        node = root["surf"]
        # Degraded to a single leaf that KEEPS the keys...
        assert node.attrs["has_keys"] is True
        assert _decode(path, "surf") == self.KEYS
        # ...rather than a ladder that silently discarded them.
        assert not [k for k in node.keys() if k.startswith("additive_")]

    def test_the_warning_names_keys(self, tmp_path: Path) -> None:
        """The author asked for something unhonourable and must be told which
        channel caused it — "labels is set" when only keys were passed would
        send them looking in the wrong place."""
        path = str(tmp_path / "m.luxar.zarr")
        with pytest.warns(UserWarning, match=r"\(keys is set\)"):
            with LuxarZarrCompiler(path) as compiler:
                scene = compiler.create_scene(dimensions=_make_3d_dims())
                scene.add_mesh(
                    "surf",
                    vertices=self.VERTS,
                    faces=self.FACES,
                    keys=self.KEYS,
                    additive_lod={"method": "radial"},
                )

    def test_write_mesh_multi_lod_refuses_keyed_levels(self, tmp_path: Path) -> None:
        """Defence in depth one layer down: the writer is a public method, so it
        refuses a keyed level directly rather than relying on the adder guard."""
        path = str(tmp_path / "m.luxar.zarr")
        with LuxarZarrCompiler(path) as compiler:
            compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises(ValueError, match="carry 'keys'"):
                compiler.write_mesh_multi_lod(
                    "surf",
                    [
                        {
                            "vertices": self.VERTS,
                            "faces": self.FACES,
                            "keys": self.KEYS,
                        }
                    ],
                )

    def test_mesh_substitutive_lod_puts_keys_exactly_where_labels_go(
        self, tmp_path: Path
    ) -> None:
        """The escape hatch the refusal message recommends must actually work.

        Asserted as CO-LOCATION rather than against a fixed tree shape: a small
        mesh degenerates to a flat leaf (no ladder to decimate into), so pinning
        "the finest child" would test the fixture size, not the contract. What
        must hold either way is that keys land wherever labels land — that is
        the whole design, and the only thing a reader of the refusal message
        needs to be true.
        """
        path = str(tmp_path / "m.luxar.zarr")
        labels = [f"L{i}" for i in range(len(self.KEYS))]
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_mesh(
                "surf",
                vertices=self.VERTS,
                faces=self.FACES,
                labels=labels,
                keys=self.KEYS,
                substitutive_lod={"levels": 2},
            )

        root = zarr.open_group(path, mode="r")

        def carriers(group, prefix=""):
            """Every node declaring each channel, by path."""
            found = {"labels": [], "keys": []}
            for name in group.keys():
                child = group[name]
                if not hasattr(child, "keys"):
                    continue
                attrs = dict(child.attrs)
                node = f"{prefix}/{name}" if prefix else name
                if attrs.get("has_labels"):
                    found["labels"].append(node)
                if attrs.get("has_keys"):
                    found["keys"].append(node)
                sub = carriers(child, node)
                found["labels"] += sub["labels"]
                found["keys"] += sub["keys"]
            return found

        got = carriers(root["surf"], "surf")
        # `surf` itself may be the carrier when the ladder degenerates.
        if dict(root["surf"].attrs).get("has_labels"):
            got["labels"].append("surf")
        if dict(root["surf"].attrs).get("has_keys"):
            got["keys"].append("surf")

        assert got["labels"], "labels vanished — the escape hatch does not work"
        assert sorted(got["keys"]) == sorted(got["labels"]), (
            f"keys and labels diverged: keys on {sorted(got['keys'])}, "
            f"labels on {sorted(got['labels'])}"
        )
        for node in got["keys"]:
            assert _decode(path, node) == self.KEYS


class TestKeysFailFastGate:
    """A bad ``keys`` must be refused BEFORE anything reaches the store.

    Every flat writer opens with a gate that validates each per-element channel
    against the element count, precisely so a length mismatch cannot leave a
    half-written node behind. ``keys`` was validated by those gate *validators*
    but never passed to them from the writers, so the check only fired later, at
    CSR-write time — after the node's other arrays were already on disk. The
    public ``compiler.write_*`` methods reach the writer directly, bypassing the
    adders (which did pass ``keys`` to their pre-split gate), so that path left a
    partial node where the same call with ``labels`` left nothing.
    """

    POSITIONS = np.arange(15, dtype=np.float32).reshape(5, 3)
    VERTS = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    FACES = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.uint32)
    LINE_VERTS = np.arange(18, dtype=np.float32).reshape(6, 3)
    LINE_WIDTHS = np.ones(6, dtype=np.float32)
    CHOLESKY = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (5, 1))
    AMPLITUDES = np.ones((5, 1), dtype=np.float32)

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("write_points", {"positions": POSITIONS}),
            ("write_mesh", {"vertices": VERTS, "faces": FACES}),
            ("write_lines", {"vertices": LINE_VERTS, "widths": LINE_WIDTHS}),
            (
                "write_gsplats",
                {
                    "centers": POSITIONS,
                    "amplitudes": AMPLITUDES,
                    "cholesky_factors": CHOLESKY,
                },
            ),
        ],
    )
    def test_wrong_length_keys_leaves_no_node(
        self, tmp_path: Path, method: str, kwargs: dict
    ) -> None:
        path = str(tmp_path / f"{method}.luxar.zarr")
        with LuxarZarrCompiler(path) as compiler:
            compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises(Exception, match="Keys length"):
                getattr(compiler, method)("n", keys=["only", "two"], **kwargs)

        root = zarr.open_group(path, mode="r")
        assert "n" not in list(root.keys()), (
            "a rejected write left a partial node behind"
        )

    def test_the_message_names_keys_not_labels(self, tmp_path: Path) -> None:
        """``keys`` shares the labels validator, which reported a bad keys
        length as "Labels length" — pointing the author at the wrong argument.
        """
        path = str(tmp_path / "s.luxar.zarr")
        with LuxarZarrCompiler(path) as compiler:
            compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises(Exception) as excinfo:
                compiler.write_points("n", positions=self.POSITIONS, keys=["a"])
        message = str(excinfo.value)
        assert "Keys length" in message
        assert "Labels length" not in message


class TestKeysOnlyMultiLodLadder:
    """``write_*_multi_lod`` with keys and NO labels on any level.

    The union CSR on the ladder parent is written in the order the levels were
    actually stored, so each level write is asked to hand back its spatial
    permutation. Whether to ask was decided by "is the ladder labelled?", so a
    keys-only ladder collected no permutations and fell back to SOURCE order —
    every key then paired with a different point, silently, at full length and
    with ``has_keys`` set.

    Two things make this easy to under-test, and both are deliberate here:

    * The adder-level ``additive_lod=`` / ``substitutive_lod=`` paths never
      reach this code, so tests using those (this file has both) leave it
      uncovered.
    * A ``sorted()`` comparison passes under the bug, because the wrong order
      is a permutation of the right one. The assertion must be order-sensitive.

    Enough points per level that the spatial permutation is not the identity —
    otherwise source order and stored order coincide and the test proves
    nothing. ``test_the_permutation_is_not_the_identity`` pins that.
    """

    COUNTS = (800, 1200)

    @classmethod
    def _levels(cls, *, with_labels: bool):
        rng = np.random.default_rng(3)
        out, base = [], 0
        for n in cls.COUNTS:
            level = {
                "positions": rng.random((n, 3)).astype(np.float32),
                "keys": [f"k{base + i}" for i in range(n)],
            }
            if with_labels:
                level["labels"] = [f"L{base + i}" for i in range(n)]
            out.append(level)
            base += n
        return out

    @classmethod
    def _source_keys(cls):
        return [f"k{i}" for i in range(sum(cls.COUNTS))]

    @staticmethod
    def _build(tmp_path: Path, name: str, levels) -> str:
        path = str(tmp_path / f"{name}.luxar.zarr")
        with LuxarZarrCompiler(path) as compiler:
            compiler.create_scene(dimensions=_make_3d_dims())
            compiler.write_points_multi_lod("pts", levels)
        return path

    def test_keys_only_ladder_matches_the_labelled_ladder_slot_for_slot(
        self, tmp_path: Path
    ) -> None:
        """The reference build carries both channels, so ``labelled`` is true
        and its ordering is the one the labels path has always produced. A
        keys-only ladder over identical positions must reproduce it exactly."""
        keys_only = self._build(tmp_path, "keys", self._levels(with_labels=False))
        reference = self._build(tmp_path, "both", self._levels(with_labels=True))

        assert zarr.open_group(keys_only, mode="r")["pts"].attrs["has_keys"] is True
        assert not zarr.open_group(keys_only, mode="r")["pts"].attrs.get("has_labels")

        got = _decode(keys_only, "pts")
        want = _decode(reference, "pts")
        assert got == want, "keys-only ladder stored its keys in a different order"

    def test_the_labelled_reference_pairs_keys_with_labels(
        self, tmp_path: Path
    ) -> None:
        """Anchors the reference: keys[j] must be the twin of labels[j], so
        "matches the reference" in the test above means "correctly paired"."""
        reference = self._build(tmp_path, "both", self._levels(with_labels=True))
        keys = _decode(reference, "pts")
        labels = _decode(reference, "pts", prefix="label")
        assert len(keys) == sum(self.COUNTS)
        assert all(k == "k" + lab[1:] for k, lab in zip(keys, labels))

    def test_the_permutation_is_not_the_identity(self, tmp_path: Path) -> None:
        """Guards the two tests above from going vacuous. If the ladder ever
        stopped reordering, stored order would equal source order and a
        permutation bug would become undetectable by them."""
        path = self._build(tmp_path, "both", self._levels(with_labels=True))
        stored = _decode(path, "pts")
        assert sorted(stored) == sorted(self._source_keys()), "content changed"
        assert stored != self._source_keys(), (
            "the ladder no longer reorders, so the ordering tests above can no "
            "longer detect a mis-paired keys CSR"
        )


class TestKeysOnlyLadder:
    """A reveal ladder carrying ``keys`` and NO ``labels`` must still work.

    The ladder writers collect each level's spatial permutation only when the
    node is going to need one for a union CSR, and that decision was originally
    "is this node labelled?". A keys-only node needs the permutations just as
    much, and without them the union write failed with "level_sort_orders has
    0". Every other keys test in this file also passes ``labels``, so the
    label-free ladder was covered by nothing — a mutation reverting the
    condition to ``labelled`` alone survived the whole suite.
    """

    N = 600

    @staticmethod
    def _positions(n: int) -> "np.ndarray":
        rng = np.random.default_rng(11)
        return rng.random((n, 3)).astype(np.float32)

    def test_additive_ladder_with_keys_and_no_labels(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.luxar.zarr")
        keys = [f"k{i}" for i in range(self.N)]
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "pts",
                positions=self._positions(self.N),
                keys=keys,
                additive_lod={"method": "random"},
            )

        root = zarr.open_group(path, mode="r")
        parent = root["pts"]
        assert parent.attrs["has_keys"] is True
        assert not parent.attrs.get("has_labels"), "no labels were requested"
        # The union CSR spans the whole source, in the ladder's own order.
        assert sorted(_decode(path, "pts")) == sorted(keys)

    def test_substitutive_ladder_with_keys_and_no_labels(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.luxar.zarr")
        keys = [f"k{i}" for i in range(self.N)]
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "pts",
                positions=self._positions(self.N),
                keys=keys,
                substitutive_lod={"levels": 2},
            )

        root = zarr.open_group(path, mode="r")
        keyed = [
            name
            for name in root["pts"].keys()
            if hasattr(root["pts"][name], "attrs")
            and dict(root["pts"][name].attrs).get("has_keys")
        ]
        assert keyed, "the finest child lost its keys when no labels were present"
        for name in keyed:
            assert sorted(_decode(path, f"pts/{name}")) == sorted(keys)


class TestHasKeysNodeProperty:
    """Every node class answers ``has_<channel>`` for each channel it can carry.

    Colors, radii, sharpness, scalars, labels, image labels — all had an
    accessor; keys did not, so an author who had just written keys could not ask
    a node whether they landed. The property reads the same writer-stamped
    metadata as its siblings.
    """

    N = 5
    POSITIONS = np.arange(15, dtype=np.float32).reshape(5, 3)
    VERTS = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    FACES = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.uint32)
    CHOLESKY = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (5, 1))

    @classmethod
    def _cases(cls):
        return [
            ("add_points", {"positions": cls.POSITIONS}, 5),
            (
                "add_lines",
                {"vertices": cls.POSITIONS, "widths": np.ones(5, dtype=np.float32)},
                5,
            ),
            ("add_mesh", {"vertices": cls.VERTS, "faces": cls.FACES}, 6),
            (
                "add_gsplats",
                {
                    "centers": cls.POSITIONS,
                    "amplitudes": np.ones(5, dtype=np.float32),
                    "cholesky_factors": cls.CHOLESKY,
                },
                5,
            ),
        ]

    @pytest.mark.parametrize("with_keys", [True, False])
    def test_has_keys_matches_what_was_written(
        self, tmp_path: Path, with_keys: bool
    ) -> None:
        for adder, kwargs, n in self._cases():
            extra = {"keys": [f"k{i}" for i in range(n)]} if with_keys else {}
            path = str(tmp_path / f"{adder}-{with_keys}.luxar.zarr")
            with LuxarZarrCompiler(path) as compiler:
                scene = compiler.create_scene(dimensions=_make_3d_dims())
                node = getattr(scene, adder)("n", **kwargs, **extra)
            assert node.has_keys is with_keys, adder
            # Absent must be False, never a raise and never None.
            assert isinstance(node.has_keys, bool), adder

    def test_has_keys_is_independent_of_has_labels(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.luxar.zarr")
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            keyed = scene.add_points(
                "keyed", positions=self.POSITIONS, keys=["a", "b", "c", "d", "e"]
            )
            labelled = scene.add_points(
                "labelled", positions=self.POSITIONS, labels=["a", "b", "c", "d", "e"]
            )
        assert (keyed.has_keys, keyed.has_labels) == (True, False)
        assert (labelled.has_keys, labelled.has_labels) == (False, True)
