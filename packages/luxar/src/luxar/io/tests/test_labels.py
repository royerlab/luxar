"""Tests for per-element label storage (CSR-style encoding).

Tests cover:
- CSR round-trip: write labels → read offsets+bytes → decode → compare
- Empty/null labels (zero-length byte ranges)
- Unicode labels (emoji, CJK characters)
- Label length mismatch validation
- Auto-injected hover overlay when labels exist
- suppress_hover_overlay flag
"""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import ArrayDecoder


def _make_3d_dims():
    return Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
        ]
    )


def _decode_labels_from_zarr(zarr_path: str, node_name: str) -> list[str]:
    """Read CSR labels from a zarr store and decode them."""
    store = zarr.open_group(zarr_path, mode="r")
    group = store[node_name]
    offsets = np.array(group["label_offsets"])
    label_bytes = np.array(group["label_bytes"])
    n = len(offsets) - 1
    labels = []
    for i in range(n):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        if start == end:
            labels.append("")
        else:
            labels.append(bytes(label_bytes[start:end]).decode("utf-8"))
    return labels


class TestLabelCSRRoundTrip:
    """Test CSR label encoding round-trip."""

    def test_basic_labels(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(5, 3).astype(np.float32)
        labels = ["Cell A", "Cell B", "Neuron 1", "Glial", "Astrocyte"]

        # Disable spatial ordering so labels stay in input order
        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            pts = scene.add_points("pts", positions, labels=labels)
            assert pts.has_labels is True

        decoded = _decode_labels_from_zarr(path, "pts")
        assert decoded == labels

    def test_empty_labels(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        labels = ["", "has label", ""]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels)

        decoded = _decode_labels_from_zarr(path, "pts")
        assert decoded == labels

    def test_unicode_labels(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        labels = ["Hello", "细胞", "🧬"]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels)

        decoded = _decode_labels_from_zarr(path, "pts")
        assert decoded == labels

    def test_labels_with_spatial_ordering(self, tmp_path):
        """Labels should be reordered to match spatial ordering."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]], dtype=np.float32)
        labels = ["origin", "far", "middle"]

        with LuxarZarrCompiler(path, enable_spatial_index=True) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            pts = scene.add_points("pts", positions, labels=labels)
            assert pts.has_labels is True

        # Labels were reordered — just verify we got them all back (order may differ)
        decoded = _decode_labels_from_zarr(path, "pts")
        assert sorted(decoded) == sorted(labels)
        assert len(decoded) == 3

    def test_label_length_mismatch_raises(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(5, 3).astype(np.float32)
        labels = ["A", "B"]  # Only 2 labels for 5 points

        with pytest.raises(ValueError, match="Labels length"):
            with LuxarZarrCompiler(path) as compiler:
                scene = compiler.create_scene(dimensions=_make_3d_dims())
                scene.add_points("pts", positions, labels=labels)

    def test_no_labels_by_default(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(5, 3).astype(np.float32)

        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            pts = scene.add_points("pts", positions)
            assert pts.has_labels is False

        store = zarr.open_group(path, mode="r")
        assert "label_offsets" not in store["pts"]

    def test_labels_on_lines(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        vertices = np.random.rand(4, 3).astype(np.float32)
        widths = np.full(4, 0.1, dtype=np.float32)
        labels = ["v0", "v1", "v2", "v3"]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            lines = scene.add_lines("ln", vertices, widths=widths, labels=labels)
            assert lines.has_labels is True

        decoded = _decode_labels_from_zarr(path, "ln")
        assert decoded == labels

    def test_lines_labels_and_segments_share_the_vertex_permutation(self, tmp_path):
        """Stored ``segments`` index the SORTED vertex rows the label CSR is keyed by.

        This is the premise the viewer's hover-label chain for lines rests on
        (issue #1424): a picked segment's stored start index is used directly as
        a row of the per-vertex label CSR. Two writes must agree for that to be
        true — ``_ordering/lines.py`` remaps the segment entries through
        ``argsort(vertex_sort_indices)``, and ``geometry_writers/lines.py`` hands
        the FORWARD ``vertex_sort_indices`` to ``write_labels_csr`` as
        ``sort_order``. Neither direction was pinned: the sibling
        ``test_labels_on_lines`` builds with ``enable_spatial_index=False``, where
        the permutation is the identity, and the Points spatial-ordering label
        test only compares multisets, which any permutation satisfies.

        The vertex identity is carried by POSITION here, not by index, so the
        check never re-derives the permutation from the thing under test. The two
        counterfactuals (permutation dropped, permutation inverted) are computed
        explicitly and asserted to break the same check, so this test cannot
        silently stop discriminating.
        """
        path = str(tmp_path / "ordered_lines.luxar.zarr")
        # A deliberately non-monotone integer-grid path: unit-spaced coordinates
        # survive any coordinate quantization unambiguously, and the zig-zag is
        # what makes the spatial (Hilbert) vertex order a real shuffle.
        vertices = np.array(
            [
                [0, 0, 0],
                [7, 1, 0],
                [1, 6, 2],
                [6, 6, 6],
                [0, 3, 7],
                [4, 0, 4],
                [7, 7, 1],
                [2, 2, 5],
                [5, 4, 0],
                [3, 7, 3],
            ],
            dtype=np.float32,
        )
        n = len(vertices)
        widths = np.full(n, 0.1, dtype=np.float32)
        # Label i names the ORIGINAL vertex, so a decoded label identifies which
        # input vertex a stored row holds.
        labels = [f"v{i}" for i in range(n)]

        with LuxarZarrCompiler(path, enable_spatial_index=True) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            node = scene.add_lines(
                "ln", vertices, widths=widths, labels=labels, line_type="polyline"
            )
            assert node.has_labels is True

        store = zarr.open_group(path, mode="r")
        group = store["ln"]
        # COORDINATE arrays are quantized on disk (linear_perchannel_u16 under
        # AUTO), so go through the canonical decoder rather than the raw uint16.
        stored_vertices = (
            ArrayDecoder().decode(group["vertices"], store).astype(np.float64)
        )
        stored_segments = np.asarray(ArrayDecoder().decode(group["segments"], store))
        stored_segments = stored_segments.reshape(-1, 2)
        decoded = _decode_labels_from_zarr(path, "ln")
        assert len(decoded) == n

        # Recover "which original vertex is stored at row j" from POSITION alone.
        # Unit-spaced grid coordinates make the nearest input vertex unique.
        perm = []
        for row in range(n):
            distances = np.linalg.norm(vertices - stored_vertices[row], axis=1)
            nearest = int(np.argmin(distances))
            assert distances[nearest] < 0.25, (
                f"stored vertex row {row} matches no input vertex "
                f"(closest distance {distances[nearest]})"
            )
            perm.append(nearest)
        assert sorted(perm) == list(range(n)), "stored rows are not a permutation"

        # The fixture must actually exercise a permutation, and one that is not
        # its own inverse — otherwise dropping or inverting it would be
        # undetectable. If a future ordering change makes this identity or an
        # involution, pick different coordinates rather than deleting the check.
        identity = list(range(n))
        assert perm != identity, "spatial ordering left the vertices in input order"
        assert [perm[perm[i]] for i in identity] != identity, (
            "the vertex permutation is its own inverse, so this test could not "
            "detect an inverted permutation"
        )

        # THE PREMISE: for every stored segment row, the label at the stored
        # start index belongs to the vertex that segment actually starts at.
        for row, (start, end) in enumerate(stored_segments):
            assert decoded[int(start)] == f"v{perm[int(start)]}", (
                f"segment row {row}: label at stored start index {start} is "
                f"{decoded[int(start)]!r}, but that row holds input vertex "
                f"{perm[int(start)]}"
            )
            assert decoded[int(end)] == f"v{perm[int(end)]}"

        # Segment topology survived the remap: a polyline over the input joins
        # consecutive INPUT vertices, so every stored pair must decode to one.
        input_pairs = {(i, i + 1) for i in range(n - 1)}
        decoded_pairs = {
            (perm[int(start)], perm[int(end)]) for start, end in stored_segments
        }
        assert decoded_pairs == input_pairs

        # Counterfactual 1 — permutation DROPPED (labels written in input order).
        dropped = labels
        assert any(
            dropped[int(start)] != f"v{perm[int(start)]}"
            for start, _ in stored_segments
        ), "a dropped label permutation would still pass the check above"

        # Counterfactual 2 — permutation INVERTED (labels gathered by the
        # inverse of vertex_sort_indices instead of the forward order).
        inverse = np.argsort(np.asarray(perm))
        inverted = [labels[int(i)] for i in inverse]
        assert any(
            inverted[int(start)] != f"v{perm[int(start)]}"
            for start, _ in stored_segments
        ), "an inverted label permutation would still pass the check above"


class TestHoverOverlayAutoInjection:
    """Test auto-injection of hover overlay when labels exist."""

    def test_auto_injects_when_labels_present(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        labels = ["A", "B", "C"]

        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels)

        # Check that the hover overlay was auto-injected
        store = zarr.open_group(path, mode="r")
        assert "overlays" in store
        assert "__hover_text" in store["overlays"]
        hover_attrs = dict(store["overlays"]["__hover_text"].attrs)
        assert hover_attrs["hover"] is True
        assert hover_attrs["text"] == "{hover_label}"

    def test_no_injection_without_labels(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)

        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions)

        store = zarr.open_group(path, mode="r")
        assert "overlays" not in store or "__hover_text" not in store.get(
            "overlays", {}
        )

    def test_no_injection_when_user_provides_hover_overlay(self, tmp_path):
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        labels = ["A", "B", "C"]

        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels)
            # User adds their own hover overlay
            scene._write_overlay(
                "my_hover",
                "overlay_text",
                (0.1, 0.1),
                {
                    "type": "overlay_text",
                    "hover": True,
                    "text": "Custom: {hover_label}",
                },
            )

        store = zarr.open_group(path, mode="r")
        # Should NOT have the auto-injected default
        assert "__hover_text" not in store.get("overlays", {})
        # Should have the user's overlay
        assert "my_hover" in store["overlays"]
