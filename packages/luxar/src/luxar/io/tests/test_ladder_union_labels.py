"""Tests for additive-LOD ladder label storage (issue #1422).

An additive ladder stores its geometry in ``additive_<i>/`` subgroups, but the
viewer's loader concatenates the levels it has loaded into one buffer, so no
single level's array is what a pick index addresses. The label CSR therefore
lives on the PARENT ladder node and spans the levels: the concatenation of
``additive_0 … additive_{n-1}``, each level in its own stored (spatially
reordered) order — the same on-disk index space a flat labelled leaf uses.

Tests cover:
- Parent carries the union CSR + ``has_labels``; every subgroup is label-free
- The union ORDER is pinned, for Points AND for Lines (each level's own
  permutation must be applied, forward, from the right ordering array)
- No spatial index → the union is the plain level concatenation
- Mixed label presence across levels is rejected (the error names the level)
- A per-level label/element count mismatch is rejected
- A malformed level array still reports the GEOMETRY fault, not a label one
- A rejected ladder leaves no partial node behind (fail-fast gate)
- A wrong-length ``labels`` raises instead of silently mislabelling every level,
  at every one of the seven wrapper pre-split gates (Points and Lines reach the
  labels check through the shared writer sweep their gate delegates to; GSplats
  through ``validate_labels_before_split``)
- ``partition=``-outer + ``additive_lod=``-inner puts the union on each part
"""

from typing import List

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def _make_3d_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
        ]
    )


def decode_labels(group: zarr.Group) -> List[str]:
    """Decode a CSR label pair from a zarr group."""
    offs = group["label_offsets"][:]
    data = group["label_bytes"][:]
    return [
        bytes(data[offs[i] : offs[i + 1]]).decode("utf-8") for i in range(len(offs) - 1)
    ]


def _random_positions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * 100.0).astype(np.float32)


class TestPointsLadderUnionLabels:
    """Parent-level union CSR for a Points additive ladder."""

    def test_parent_carries_union_levels_carry_nothing(self, tmp_path):
        path = str(tmp_path / "ladder.luxar.zarr")
        n_points = 400
        positions = _random_positions(n_points, seed=7)
        labels = [f"cell_{i}" for i in range(n_points)]

        compiler = LuxarZarrCompiler(path)
        scene = compiler.create_scene(dimensions=_make_3d_dims())
        node = scene.add_points("ladder", positions, labels=labels, additive_lod=True)
        compiler.finalize()

        assert node.has_labels is True

        store = zarr.open_group(path, mode="r")
        parent = store["ladder"]
        assert parent.attrs["has_labels"] is True
        assert "label_offsets" in parent
        assert "label_bytes" in parent
        assert len(parent["label_offsets"]) == n_points + 1

        decoded = decode_labels(parent)
        assert len(decoded) == n_points
        assert sorted(decoded) == sorted(labels)

        # Every additive_<i> subgroup must be label-free.
        n_sublods = int(parent.attrs["n_additive_sublods"])
        assert n_sublods > 1
        for i in range(n_sublods):
            level = parent[f"additive_{i}"]
            assert "label_offsets" not in level
            assert "label_bytes" not in level
            assert "has_labels" not in dict(level.attrs)

    def test_union_order_matches_per_level_stored_order(self, tmp_path):
        """The union must apply EACH level's own spatial permutation.

        Written both ways in one store: the ladder (parent union CSR) and each
        level again as a flat points node. Both run the identical spatial
        ordering code on identical inputs, so the ladder's union must equal the
        concatenation of the flat nodes' decoded CSRs — which pins the
        permutation exactly, without decoding quantized positions.
        """
        path = str(tmp_path / "order.luxar.zarr")
        counts = [60, 140, 300]
        seeds = [11, 12, 13]
        level_positions = [_random_positions(n, s) for n, s in zip(counts, seeds)]
        level_labels = [[f"L{i}_{j}" for j in range(n)] for i, n in enumerate(counts)]

        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())
        compiler.write_points_multi_lod(
            "ladder",
            levels=[
                {"positions": p, "labels": lab}
                for p, lab in zip(level_positions, level_labels)
            ],
        )
        for i, (p, lab) in enumerate(zip(level_positions, level_labels)):
            compiler.write_points(f"ref_{i}", p, labels=lab)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        union = decode_labels(store["ladder"])

        expected: List[str] = []
        for i in range(len(counts)):
            expected.extend(decode_labels(store[f"ref_{i}"]))

        assert union == expected

        # Guard against a vacuous pass: the seeded data must actually be
        # spatially reordered, so the union is NOT the raw input concatenation.
        raw = [lbl for lab in level_labels for lbl in lab]
        assert union != raw

    def test_no_spatial_index_gives_identity_union(self, tmp_path):
        path = str(tmp_path / "identity.luxar.zarr")
        counts = [40, 90]
        level_positions = [_random_positions(n, s) for n, s in zip(counts, (21, 22))]
        level_labels = [[f"L{i}_{j}" for j in range(n)] for i, n in enumerate(counts)]

        compiler = LuxarZarrCompiler(path, enable_spatial_index=False)
        compiler.create_scene(dimensions=_make_3d_dims())
        compiler.write_points_multi_lod(
            "ladder",
            levels=[
                {"positions": p, "labels": lab}
                for p, lab in zip(level_positions, level_labels)
            ],
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert decode_labels(store["ladder"]) == [
            lbl for lab in level_labels for lbl in lab
        ]

    def test_mixed_label_presence_rejected(self, tmp_path):
        path = str(tmp_path / "mixed.luxar.zarr")
        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())

        with pytest.raises(ValueError, match=r"level 1 \(additive_1\)"):
            compiler.write_points_multi_lod(
                "ladder",
                levels=[
                    {
                        "positions": _random_positions(10, 31),
                        "labels": [f"a{i}" for i in range(10)],
                    },
                    {"positions": _random_positions(20, 32)},
                ],
            )

    def test_per_level_label_length_mismatch_rejected(self, tmp_path):
        """Relocation pin, not a before/after regression test.

        A wrong-length level was already rejected before this change — by the
        per-level ``write_points`` gate. Labels no longer reach that gate, so
        this pins the check in its new home (``validate_ladder_labels``) rather
        than proving new behaviour.
        """
        path = str(tmp_path / "mismatch.luxar.zarr")
        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())

        with pytest.raises(ValueError, match="labels"):
            compiler.write_points_multi_lod(
                "ladder",
                levels=[
                    {
                        "positions": _random_positions(10, 41),
                        "labels": [f"a{i}" for i in range(10)],
                    },
                    {
                        "positions": _random_positions(20, 42),
                        # Only 5 labels for 20 points.
                        "labels": [f"b{i}" for i in range(5)],
                    },
                ],
            )

    def test_malformed_level_geometry_reports_the_geometry_fault(self, tmp_path):
        """A bad element array must not be reported as a label-count problem.

        The label gate now runs before the per-level writes, so it must not
        diagnose geometry: a 1-D level array has no well-defined element count,
        and the caller's real fault is the shape. Same error, labels or not.
        """
        path = str(tmp_path / "badgeom.luxar.zarr")
        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())

        with pytest.raises(ValueError, match="1D array"):
            compiler.write_points_multi_lod(
                "ladder",
                levels=[
                    {
                        # 1-D, so neither 10 nor any other count is meaningful.
                        "positions": np.arange(10, dtype=np.float32),
                        "labels": [f"a{i}" for i in range(3)],
                    }
                ],
            )

    def test_rejected_ladder_leaves_no_partial_node(self, tmp_path):
        """The label gate is part of the fail-fast block, above require_group."""
        path = str(tmp_path / "failfast.luxar.zarr")
        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())

        with pytest.raises(ValueError):
            compiler.write_points_multi_lod(
                "ladder",
                levels=[
                    {
                        "positions": _random_positions(10, 71),
                        "labels": [f"a{i}" for i in range(10)],
                    },
                    {"positions": _random_positions(20, 72)},
                ],
            )

        # No empty 'ladder' group may be left behind by the rejected write.
        assert "ladder" not in compiler.store

    def test_partition_of_ladder_puts_the_union_on_each_part(self, tmp_path):
        """``partition=``-outer + ``additive_lod=``-inner: one union per part.

        The composition the format doc singles out. ``partition=`` forwards
        ``additive_lod=`` into every part, so each ``part_<i>`` is itself a
        ladder parent — and since #1415 that parent IS the node the picker looks
        labels up on (the outermost ``kind=partition`` wrapper is the reported
        path only, and carries no CSR). So the union has to land on each part,
        not on the wrapper, and the parts' unions together must account for
        every input label exactly once.
        """
        path = str(tmp_path / "part_ladder.luxar.zarr")
        n_points = 600
        positions = _random_positions(n_points, seed=91)
        labels = [f"p{i}" for i in range(n_points)]

        compiler = LuxarZarrCompiler(path)
        scene = compiler.create_scene(dimensions=_make_3d_dims())
        scene.add_points(
            "lad",
            positions,
            labels=labels,
            partition={"max_elements": 150},
            additive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        wrapper = store["lad"]
        assert wrapper.attrs["kind"] == "partition"
        # The wrapper is a bare group — no CSR, no has_labels (#1415).
        assert "label_offsets" not in wrapper
        assert "has_labels" not in dict(wrapper.attrs)

        part_names = sorted(k for k in wrapper.group_keys() if k.startswith("part_"))
        assert len(part_names) > 1

        seen: List[str] = []
        for part_name in part_names:
            part = wrapper[part_name]
            n_sublods = int(part.attrs["n_additive_sublods"])
            # Otherwise this would only be testing the flat per-part path.
            assert n_sublods > 1
            assert part.attrs["has_labels"] is True
            decoded = decode_labels(part)
            assert len(decoded) == int(part.attrs["n_points"])
            seen.extend(decoded)
            for i in range(n_sublods):
                assert "label_offsets" not in part[f"additive_{i}"]

        assert sorted(seen) == sorted(labels)


class TestWrongLengthLabelsRejectedBeforeSplit:
    """A wrong-length ``labels`` must error, not silently mislabel every part/level.

    ``slice_optional_array`` passes a list whose length != n_elements through
    UNCHANGED (that pass-through is how broadcast values reach every part), so
    without the up-front guard each part/level receives the same unsliced list and
    the write SUCCEEDS with labels in the wrong slots.

    Every partition / additive-ladder input here is chosen so the per-level/per-part
    length checks CANNOT catch it on their own — each level's own element count
    equals the label count, so only a check against the FULL count rejects it. The
    two substitutive wrappers are the one shape where that cannot happen (their
    finest ``kind=lod`` child always carries the FULL element set, so a downstream
    check does reject), and they are pinned by WHERE the write fails instead: the
    guard runs above ``add_lod_group``, so a rejected call leaves nothing on disk
    rather than a ``kind=lod`` node whose coarse levels are already written.
    """

    def test_points_ladder_wrong_length_labels_raises(self, tmp_path):
        path = str(tmp_path / "wrong_points.luxar.zarr")
        positions = _random_positions(100, seed=81)

        with pytest.raises(ValueError, match="[Ll]abels"):
            compiler = LuxarZarrCompiler(path)
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "l",
                positions,
                # 50 labels for 100 points. `counts` are CUMULATIVE, so [50, 100]
                # yields two levels of 50 points each — every level matches the
                # label count, so the per-level check accepts and only the
                # full-count guard rejects. (Measured: [50, 50].)
                labels=[f"a{i}" for i in range(50)],
                additive_lod={"counts": [50, 100]},
            )

    def test_lines_ladder_wrong_length_labels_raises(self, tmp_path):
        path = str(tmp_path / "wrong_lines.luxar.zarr")
        vertices = _random_positions(100, seed=82)

        with pytest.raises(ValueError, match="[Ll]abels"):
            compiler = LuxarZarrCompiler(path)
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_lines(
                "l",
                vertices,
                widths=0.2,
                line_type="segments",
                # 50 labels for 100 vertices. `counts` is explicit here for the
                # same reason as the Points twin, but in POLYLINE units: with
                # line_type="segments" each segment is a 2-vertex polyline, so the
                # cumulative [25, 50] yields two levels of 50 VERTICES each and
                # the per-level check accepts. The DEFAULT ladder would not pin
                # anything — it splits these 100 vertices into 26/26/26/22, none
                # of which is 50, so the per-level check alone would reject it.
                labels=[f"v{i}" for i in range(50)],
                additive_lod={"counts": [25, 50]},
            )

    def test_gsplats_partition_wrong_length_labels_raises(self, tmp_path):
        """The same silent mislabel was live on the GSplats partition path.

        Without the guard both ``part_0`` and ``part_1`` (100 splats each) get the
        SAME 100 labels — part 1's tooltips are part 0's.
        """
        path = str(tmp_path / "wrong_gsplats.luxar.zarr")
        n_splats = 200
        centers = _random_positions(n_splats, seed=83)
        cholesky = np.tile(
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (n_splats, 1),
        )

        with pytest.raises(ValueError, match="[Ll]abels"):
            compiler = LuxarZarrCompiler(path)
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                # 100 labels for 200 splats; each part is exactly 100, so the
                # per-part write would accept the unsliced list.
                labels=[f"s{i}" for i in range(100)],
                partition={"max_elements": 100},
            )

    def test_points_partition_wrong_length_labels_raises(self, tmp_path):
        """The Points twin of the GSplats partition case.

        Without the guard both ``part_0`` and ``part_1`` (100 points each) are
        written with the SAME 100 labels — measured: each part's decoded CSR
        starts ``a0, a1, a10``.
        """
        path = str(tmp_path / "wrong_points_partition.luxar.zarr")
        positions = _random_positions(200, seed=84)

        with pytest.raises(ValueError, match=r"must match element count \(200\)"):
            compiler = LuxarZarrCompiler(path)
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points(
                "p",
                positions,
                # 100 labels for 200 points. The median BSP splits these exactly
                # 100/100 (measured), so every part's own length check accepts the
                # unsliced list and only the full-count guard rejects it.
                labels=[f"a{i}" for i in range(100)],
                partition={"max_elements": 100},
            )

    def test_lines_partition_wrong_length_labels_raises(self, tmp_path):
        """The Lines partition case — labels are per-VERTEX, sliced per part.

        ``slice_optional_array`` is called with the part's VERTEX indices and the
        full vertex count, so the same pass-through trap applies one currency
        over; without the guard both parts get labels ``v0 … v99`` (measured).
        """
        path = str(tmp_path / "wrong_lines_partition.luxar.zarr")
        vertices = _random_positions(200, seed=85)

        with pytest.raises(ValueError, match=r"must match element count \(200\)"):
            compiler = LuxarZarrCompiler(path)
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_lines(
                "l",
                vertices,
                widths=0.2,
                line_type="segments",
                # 100 labels for 200 VERTICES. The polyline-centroid BSP puts 50
                # of the 100 two-vertex segments in each part (measured: 100
                # vertices each), so the per-part per-vertex check accepts.
                labels=[f"v{i}" for i in range(100)],
                partition={"max_elements": 100},
            )

    def test_points_substitutive_wrong_length_labels_raises(self, tmp_path):
        """The Points substitutive wrapper — pinned by WHERE it fails.

        Its finest ``kind=lod`` child carries all N points, so the wrong length is
        caught downstream too; what the guard buys is failing above
        ``add_lod_group``, before the lift + gsplat reduce has written anything.
        Removing it is measurable: the error then surfaces from ``child_3`` with
        the three coarse gsplat children already on disk, leaving a partial
        ``kind=lod`` node behind — which the store assertion below catches.
        """
        path = str(tmp_path / "wrong_points_sub.luxar.zarr")
        positions = _random_positions(400, seed=86)

        compiler = LuxarZarrCompiler(path)
        scene = compiler.create_scene(dimensions=_make_3d_dims())

        # 200 labels for 400 points, and enough points for the reduce to actually
        # synthesise coarse levels (a degenerate input falls back to a flat node).
        with pytest.raises(ValueError, match=r"must match element count \(400\)"):
            scene.add_points(
                "p",
                positions,
                labels=[f"a{i}" for i in range(200)],
                substitutive_lod=True,
            )

        assert "p" not in compiler.store

    def test_lines_substitutive_wrong_length_labels_raises(self, tmp_path):
        """The Lines substitutive wrapper, the twin of the Points one.

        Same fail-fast contract, per-VERTEX: the guard counts ``len(vert_arr)``,
        and without it the bead lift writes the coarse gsplat levels before
        ``child_3`` rejects the list, stranding a partial ``kind=lod`` node.
        """
        path = str(tmp_path / "wrong_lines_sub.luxar.zarr")
        vertices = _random_positions(400, seed=87)

        compiler = LuxarZarrCompiler(path)
        scene = compiler.create_scene(dimensions=_make_3d_dims())

        with pytest.raises(ValueError, match=r"must match element count \(400\)"):
            scene.add_lines(
                "l",
                vertices,
                widths=0.2,
                line_type="segments",
                # 200 labels for 400 vertices.
                labels=[f"v{i}" for i in range(200)],
                substitutive_lod=True,
            )

        assert "l" not in compiler.store


class TestLinesLadderUnionLabels:
    """Parent-level union CSR for a Lines additive ladder (per-vertex)."""

    def test_parent_carries_union_levels_carry_nothing(self, tmp_path):
        path = str(tmp_path / "lines_ladder.luxar.zarr")
        counts = [8, 24]
        level_vertices = [_random_positions(n, s) for n, s in zip(counts, (51, 52))]
        level_labels = [[f"L{i}_v{j}" for j in range(n)] for i, n in enumerate(counts)]
        n_vertices_total = sum(counts)

        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())
        compiler.write_lines_multi_lod(
            "ladder",
            levels=[
                {
                    "vertices": v,
                    "widths": np.full(len(v), 0.2, dtype=np.float32),
                    "labels": lab,
                    "n_polylines": 1,
                }
                for v, lab in zip(level_vertices, level_labels)
            ],
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        parent = store["ladder"]
        assert parent.attrs["has_labels"] is True
        assert len(parent["label_offsets"]) == n_vertices_total + 1

        decoded = decode_labels(parent)
        assert sorted(decoded) == sorted(lbl for lab in level_labels for lbl in lab)

        for i in range(len(counts)):
            level = parent[f"additive_{i}"]
            assert "label_offsets" not in level
            assert "has_labels" not in dict(level.attrs)

    def test_union_order_matches_per_level_stored_order(self, tmp_path):
        """The Lines union must apply each level's own per-VERTEX permutation.

        The Lines twin of the Points order test, and the one that matters most
        here: Lines carry a DUAL ordering (``vertex_sort_indices`` vs
        ``segment_sort_indices``), so reading the wrong array — or applying the
        forward permutation inverted — is the live hazard, and both mistakes
        produce a union of the correct LENGTH that no length check can catch.

        Written both ways in one store: the ladder (parent union CSR) and each
        level again as a flat lines node. Both run the identical dual-ordering
        code on identical inputs, so the ladder's union must equal the
        concatenation of the flat nodes' decoded CSRs.
        """
        path = str(tmp_path / "lines_order.luxar.zarr")
        counts = [40, 120]
        seeds = [91, 92]
        level_vertices = [_random_positions(n, s) for n, s in zip(counts, seeds)]
        level_labels = [[f"L{i}_v{j}" for j in range(n)] for i, n in enumerate(counts)]

        def _widths(v: np.ndarray) -> np.ndarray:
            return np.full(len(v), 0.2, dtype=np.float32)

        compiler = LuxarZarrCompiler(path)
        compiler.create_scene(dimensions=_make_3d_dims())
        compiler.write_lines_multi_lod(
            "ladder",
            levels=[
                {
                    "vertices": v,
                    "widths": _widths(v),
                    "labels": lab,
                    "n_polylines": 1,
                }
                for v, lab in zip(level_vertices, level_labels)
            ],
        )
        for i, (v, lab) in enumerate(zip(level_vertices, level_labels)):
            compiler.write_lines(
                f"ref_{i}",
                v,
                widths=_widths(v),
                labels=lab,
                line_type="polyline",
            )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        union = decode_labels(store["ladder"])

        expected: List[str] = []
        for i in range(len(counts)):
            expected.extend(decode_labels(store[f"ref_{i}"]))

        assert union == expected

        # Guard against a vacuous pass: the seeded data must actually be
        # spatially reordered, so the union is NOT the raw input concatenation.
        raw = [lbl for lab in level_labels for lbl in lab]
        assert union != raw

    def test_add_lines_ladder_writes_parent_union(self, tmp_path):
        """The Lines adder path (add_lines with additive_lod=) too."""
        path = str(tmp_path / "lines_adder.luxar.zarr")
        n_vertices = 300
        vertices = _random_positions(n_vertices, seed=61)
        labels = [f"v{i}" for i in range(n_vertices)]

        compiler = LuxarZarrCompiler(path)
        scene = compiler.create_scene(dimensions=_make_3d_dims())
        node = scene.add_lines(
            "ladder",
            vertices,
            widths=0.2,
            line_type="segments",
            labels=labels,
            additive_lod=True,
        )
        compiler.finalize()

        assert node.has_labels is True

        store = zarr.open_group(path, mode="r")
        parent = store["ladder"]
        assert parent.attrs["has_labels"] is True
        assert len(parent["label_offsets"]) == n_vertices + 1
        assert sorted(decode_labels(parent)) == sorted(labels)

        for i in range(int(parent.attrs["n_additive_sublods"])):
            assert "label_offsets" not in parent[f"additive_{i}"]
