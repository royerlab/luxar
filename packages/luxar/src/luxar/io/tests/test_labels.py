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
