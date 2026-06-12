"""Tests for per-element image label storage (CSR-style encoding).

Tests cover:
- CSR round-trip: write image blobs → read offsets+bytes → compare
- Format normalization (bytes, PIL, ndarray, Path)
- Sparse image labels via dict
- Length mismatch validation
- No compression on image_label_bytes array
- has_image_labels metadata flag
- Auto-injected HTML hover overlay when image_labels exist
- Combined text + image labels auto-injection
- Spatial ordering applied to image labels
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


def _decode_image_labels_from_zarr(zarr_path: str, node_name: str) -> list[bytes]:
    """Read CSR image labels from a zarr store and decode them."""
    store = zarr.open_group(zarr_path, mode="r")
    group = store[node_name]
    offsets = np.array(group["image_label_offsets"])
    image_bytes = np.array(group["image_label_bytes"])
    n = len(offsets) - 1
    blobs = []
    for i in range(n):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        blobs.append(bytes(image_bytes[start:end]))
    return blobs


def _make_fake_jpeg(size: int = 100) -> bytes:
    """Create bytes with JPEG magic header for testing."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * size


def _make_fake_webp(size: int = 100) -> bytes:
    """Create bytes with WebP magic header for testing."""
    return b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * size


def _make_fake_png(size: int = 100) -> bytes:
    """Create bytes with PNG magic header for testing."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * size


class TestImageLabelCSRRoundTrip:
    """Test CSR image label encoding round-trip."""

    def test_basic_image_labels_bytes(self, tmp_path):
        """Pre-encoded byte blobs stored and retrieved correctly."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        blobs = [_make_fake_jpeg(50), _make_fake_webp(80), _make_fake_png(60)]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            pts = scene.add_points("pts", positions, image_labels=blobs)
            assert pts.has_image_labels is True

        decoded = _decode_image_labels_from_zarr(path, "pts")
        assert decoded == blobs

    def test_image_labels_bytearray(self, tmp_path):
        """bytearray inputs are accepted and stored correctly."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(2, 3).astype(np.float32)
        blobs = [bytearray(_make_fake_jpeg(40)), bytearray(_make_fake_webp(60))]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=blobs)

        decoded = _decode_image_labels_from_zarr(path, "pts")
        assert decoded[0] == bytes(blobs[0])
        assert decoded[1] == bytes(blobs[1])

    def test_image_labels_from_pil(self, tmp_path):
        """PIL Images are auto-encoded to WebP."""
        pytest.importorskip("PIL")
        from PIL import Image

        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(2, 3).astype(np.float32)
        images = [
            Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)),
        ]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=images)

        decoded = _decode_image_labels_from_zarr(path, "pts")
        # WebP magic: starts with "RIFF" ... "WEBP"
        for blob in decoded:
            assert len(blob) > 0
            assert blob[:4] == b"RIFF"
            assert blob[8:12] == b"WEBP"

    def test_image_labels_from_ndarray(self, tmp_path):
        """Numpy arrays (H,W,C) uint8 are auto-encoded to WebP."""
        pytest.importorskip("PIL")

        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(2, 3).astype(np.float32)
        arrays = [
            np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
            np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
        ]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=arrays)

        decoded = _decode_image_labels_from_zarr(path, "pts")
        for blob in decoded:
            assert len(blob) > 0
            assert blob[:4] == b"RIFF"

    def test_image_labels_from_paths(self, tmp_path):
        """File paths are read and stored as raw bytes."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(2, 3).astype(np.float32)

        # Write fake images to temp files
        img_path_0 = tmp_path / "img0.jpg"
        img_path_1 = tmp_path / "img1.webp"
        blob0 = _make_fake_jpeg(50)
        blob1 = _make_fake_webp(70)
        img_path_0.write_bytes(blob0)
        img_path_1.write_bytes(blob1)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=[img_path_0, img_path_1])

        decoded = _decode_image_labels_from_zarr(path, "pts")
        assert decoded == [blob0, blob1]

    def test_image_labels_from_str_paths(self, tmp_path):
        """String paths are read and stored as raw bytes."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(1, 3).astype(np.float32)

        img_path = tmp_path / "img.png"
        blob = _make_fake_png(40)
        img_path.write_bytes(blob)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=[str(img_path)])

        decoded = _decode_image_labels_from_zarr(path, "pts")
        assert decoded == [blob]


class TestSparseImageLabels:
    """Test sparse image labels via dict input."""

    def test_sparse_image_labels(self, tmp_path):
        """Dict input creates sparse CSR — missing indices have empty blobs."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(5, 3).astype(np.float32)
        blob_a = _make_fake_jpeg(30)
        blob_c = _make_fake_webp(50)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels={0: blob_a, 3: blob_c})

        decoded = _decode_image_labels_from_zarr(path, "pts")
        assert decoded[0] == blob_a
        assert decoded[1] == b""
        assert decoded[2] == b""
        assert decoded[3] == blob_c
        assert decoded[4] == b""

    def test_sparse_out_of_range_raises(self, tmp_path):
        """Dict with out-of-range index raises ValueError."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises(ValueError, match="out of range"):
                scene.add_points("pts", positions, image_labels={5: _make_fake_jpeg()})


class TestImageLabelValidation:
    """Test validation and error handling."""

    def test_length_mismatch_raises(self, tmp_path):
        """List length != n_elements raises ValueError."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises(ValueError, match="must match"):
                scene.add_points(
                    "pts",
                    positions,
                    image_labels=[_make_fake_jpeg(), _make_fake_webp()],
                )

    def test_unsupported_type_raises(self, tmp_path):
        """Unsupported input type raises TypeError."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(1, 3).astype(np.float32)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            with pytest.raises((TypeError, ValueError)):
                scene.add_points("pts", positions, image_labels=[42])


class TestImageLabelZarrProperties:
    """Test zarr storage properties."""

    def test_no_compression_on_image_bytes(self, tmp_path):
        """image_label_bytes array has compressor=None."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        blobs = [_make_fake_jpeg(50)] * 3

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=blobs)

        store = zarr.open_group(path, mode="r")
        image_bytes_arr = store["pts"]["image_label_bytes"]
        assert image_bytes_arr.compressor is None

    def test_offsets_has_compression(self, tmp_path):
        """image_label_offsets array uses default compression."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        blobs = [_make_fake_jpeg(50)] * 3

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=blobs)

        store = zarr.open_group(path, mode="r")
        offsets_arr = store["pts"]["image_label_offsets"]
        assert offsets_arr.compressor is not None

    def test_has_image_labels_metadata(self, tmp_path):
        """has_image_labels flag set in .zattrs."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        blobs = [_make_fake_jpeg(50)] * 3

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=blobs)

        store = zarr.open_group(path, mode="r")
        assert store["pts"].attrs["has_image_labels"] is True

    def test_no_image_labels_no_flag(self, tmp_path):
        """Without image_labels, has_image_labels is not set."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions)

        store = zarr.open_group(path, mode="r")
        assert store["pts"].attrs.get("has_image_labels") is None


class TestImageLabelAutoInject:
    """Test auto-injection of hover overlay with image labels."""

    def test_auto_inject_html_overlay_image_only(self, tmp_path):
        """Image labels only → auto-injects overlay_html with {hover_image_label}."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        blobs = [_make_fake_jpeg(50)] * 3

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=blobs)

        store = zarr.open_group(path, mode="r")
        hover_attrs = store["overlays"]["__hover_image"].attrs
        assert hover_attrs["type"] == "overlay_html"
        assert "{hover_image_label}" in hover_attrs["html"]
        assert hover_attrs["hover"] is True

    def test_auto_inject_combined_text_and_image(self, tmp_path):
        """Both labels and image_labels → separate image and text hover overlays."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        blobs = [_make_fake_jpeg(50)] * 3
        labels = ["A", "B", "C"]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels, image_labels=blobs)

        store = zarr.open_group(path, mode="r")
        img_attrs = store["overlays"]["__hover_image"].attrs
        assert img_attrs["type"] == "overlay_html"
        assert "{hover_image_label}" in img_attrs["html"]
        txt_attrs = store["overlays"]["__hover_text"].attrs
        assert txt_attrs["type"] == "overlay_text"
        assert txt_attrs["text"] == "{hover_label}"

    def test_auto_inject_text_only_unchanged(self, tmp_path):
        """Text labels only → overlay_text with {hover_label} (existing behavior)."""
        path = str(tmp_path / "test.luxar.zarr")
        positions = np.random.rand(3, 3).astype(np.float32)
        labels = ["A", "B", "C"]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, labels=labels)

        store = zarr.open_group(path, mode="r")
        hover_attrs = store["overlays"]["__hover_text"].attrs
        assert hover_attrs["type"] == "overlay_text"
        assert hover_attrs["text"] == "{hover_label}"


class TestImageLabelSpatialOrdering:
    """Test that image labels are reordered with spatial ordering."""

    def test_image_labels_with_spatial_ordering(self, tmp_path):
        """Image labels are reordered when spatial ordering is enabled."""
        path = str(tmp_path / "test.luxar.zarr")
        # Use well-separated positions so ordering actually reorders
        positions = np.array(
            [
                [100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 0.0, 100.0],
                [50.0, 50.0, 50.0],
            ],
            dtype=np.float32,
        )
        blobs = [
            _make_fake_jpeg(10),
            _make_fake_webp(20),
            _make_fake_png(30),
            _make_fake_jpeg(40),
        ]

        # Enable spatial indexing (default)
        with LuxarZarrCompiler(path, enable_spatial_index=True) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            scene.add_points("pts", positions, image_labels=blobs)

        decoded = _decode_image_labels_from_zarr(path, "pts")
        # With spatial ordering, the blobs should be reordered
        # We can't predict the exact order, but all blobs should still be present
        assert sorted(decoded, key=len) == sorted(blobs, key=len)
        assert len(decoded) == 4


class TestImageLabelOnLinesAndGSplats:
    """Test that image_labels work on Lines and GSplats too."""

    def test_lines_image_labels(self, tmp_path):
        """Image labels on lines nodes."""
        path = str(tmp_path / "test.luxar.zarr")
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        blobs = [_make_fake_jpeg(30), _make_fake_webp(40), _make_fake_png(50)]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            lines = scene.add_lines("lines", vertices, widths=0.1, image_labels=blobs)
            assert lines.has_image_labels is True

        decoded = _decode_image_labels_from_zarr(path, "lines")
        assert decoded == blobs

    def test_gsplats_image_labels(self, tmp_path):
        """Image labels on gsplats nodes."""
        path = str(tmp_path / "test.luxar.zarr")
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        amplitudes = np.array([1.0, 0.5], dtype=np.float32)
        # 3D Cholesky factors: k = 3*(3+1)/2 = 6
        cholesky = np.eye(3, dtype=np.float32)[np.triu_indices(3)]
        cholesky = np.tile(cholesky, (2, 1))
        blobs = [_make_fake_jpeg(30), _make_fake_webp(40)]

        with LuxarZarrCompiler(path, enable_spatial_index=False) as compiler:
            scene = compiler.create_scene(dimensions=_make_3d_dims())
            gsplats = scene.add_gsplats(
                "splats", centers, amplitudes, cholesky, image_labels=blobs
            )
            assert gsplats.has_image_labels is True

        decoded = _decode_image_labels_from_zarr(path, "splats")
        assert decoded == blobs
