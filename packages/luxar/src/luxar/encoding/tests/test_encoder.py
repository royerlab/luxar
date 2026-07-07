"""Tests for ArrayEncoder class.

Tests cover all semantic types, encoding modes, and special encodings
(broadcasting, LUT, array references).
"""

import tempfile

import numpy as np
import pytest
import zarr

from luxar.encoding import ArrayDecoder, ArrayEncoder, EncodingMode, SemanticType


class TestBroadcasting:
    """Test broadcasting detection and encoding."""

    def test_uniform_1d_float(self):
        """Test broadcasting uniform 1D float array."""
        data = np.full(1000, 2.5, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Check stored data
            arr = group["test"]
            assert arr.shape == (1,)
            assert arr[:][0] == 2.5

            # Check metadata
            enc = arr.attrs["encoding"]
            assert enc["name"] == "broadcasted"
            assert enc["n_elements"] == 1000

    def test_uniform_2d_color(self):
        """Test broadcasting uniform 2D color array."""
        data = np.full((1000, 3), [1.0, 0.5, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.COLOR, color_mode="sdr")

            # Check stored data
            arr = group["test"]
            assert arr.shape == (1, 3)

            # Check metadata
            enc = arr.attrs["encoding"]
            assert enc["name"] == "broadcasted"
            assert enc["n_elements"] == 1000

    def test_non_uniform_not_broadcasted(self):
        """Test non-uniform array is not broadcasted."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Check not broadcasted
            arr = group["test"]
            assert arr.shape == (3,)
            enc = arr.attrs["encoding"]
            assert enc["name"] != "broadcasted"

    # [Python-R2/encoding-MAJOR] The non-zero broadcast_rtol / broadcast_atol
    # branch in encoder._is_uniform was previously untested — every existing
    # broadcasting test uses default tolerance 0.0 (exact equality). A
    # near-uniform array (e.g., quantization noise around a constant) is
    # the actual use case for the tolerance knobs; pin both the "still
    # detected as uniform" and "non-uniform survives" cases.
    def test_uniform_within_tolerance_is_broadcasted(self):
        """Tiny per-element jitter within rtol IS detected as uniform."""
        # Values differ by ~1e-7 (well below rtol=1e-5) — should be
        # broadcasted with non-zero tolerance.
        data = np.array([1.0, 1.0 + 1e-7, 1.0 - 1e-7], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder(broadcast_rtol=1e-5, broadcast_atol=1e-7)
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            enc = group["test"].attrs["encoding"]
            assert enc["name"] == "broadcasted"

    def test_uniform_outside_tolerance_survives_full(self):
        """Per-element jitter above the tolerance is NOT broadcasted."""
        # Values differ by ~0.1 — well above the tolerance.
        data = np.array([1.0, 1.1, 0.9], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder(broadcast_rtol=1e-5, broadcast_atol=1e-7)
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            enc = group["test"].attrs["encoding"]
            # Non-uniform — encoder picks LUT or full storage, NOT broadcasted.
            assert enc["name"] != "broadcasted"


class TestLUTEncoding:
    """Test LUT encoding for arrays with limited unique values."""

    def test_lut_1d_few_unique(self):
        """Test LUT encoding for 1D array with few unique values."""
        # 1000 elements with only 5 unique values
        data = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=1000).astype(
            np.float32
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Check LUT encoding used
            arr = group["test"]
            assert arr.dtype == np.uint8  # Indices
            enc = arr.attrs["encoding"]
            assert enc["name"] == "lut_uint8"
            assert len(enc["lut"]) == 5
            assert enc["original_dtype"] == "float32"

    def test_lut_color_row_mode(self):
        """Test LUT encoding for colors (row mode)."""
        # 1000 points with only 3 unique colors
        colors = np.random.choice([0, 1, 2], size=1000)  # Indices into color palette
        palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        data = palette[colors]

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.COLOR)

            # Check LUT encoding used
            arr = group["test"]
            assert arr.dtype == np.uint8
            enc = arr.attrs["encoding"]
            assert enc["name"] == "lut_uint8"
            assert enc["lut_mode"] == "row"
            assert len(enc["lut"]) == 3  # 3 unique colors

    def test_lut_skipped_in_precision_mode(self):
        """Test LUT encoding is skipped in PRECISION mode."""
        data = np.random.choice([0.0, 0.5, 1.0], size=1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                data,
                group,
                "test",
                SemanticType.POSITIVE_SCALAR,
                mode=EncodingMode.PRECISION,
            )

            # Check LUT NOT used
            arr = group["test"]
            enc = arr.attrs["encoding"]
            assert enc["name"] != "lut_uint8"

    def test_lut_not_used_for_uint8(self):
        """Test LUT not used when data is already uint8."""
        data = np.random.choice([0, 127, 255], size=1000).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.INDEX)

            # Check LUT NOT used (already optimal)
            arr = group["test"]
            enc = arr.attrs["encoding"]
            assert enc["name"] != "lut_uint8"


def _tiled_palette_colors(k: int, n: int, seed: int = 0) -> np.ndarray:
    """(n, 3) float32 colors with EXACTLY k unique rows.

    Tiling (not random choice) guarantees every palette row appears — a
    random draw of n from k leaves ~k·e^(-n/k) rows unused, silently
    shifting the unique count the tests pin.
    """
    rng = np.random.default_rng(seed)
    palette = (rng.random((k, 3)) * 10.0).astype(np.float32)
    reps = -(-n // k)  # ceil
    return np.tile(palette, (reps, 1))[:n]


class TestLUTUint16Tier:
    """The uint16 LUT tier (257..65,536 uniques, byte-modeled benefit rule).

    The uint8 tier's behavior is pinned byte-identical by TestLUTEncoding
    above; these tests cover the new tier and its rejection edges.
    """

    def _encode(self, data, semantic_type, group, encoder=None, **kw):
        (encoder or ArrayEncoder()).encode(data, group, "test", semantic_type, **kw)
        arr = group["test"]
        return arr, dict(arr.attrs["encoding"])

    def test_row_color_uint16_exact_roundtrip(self):
        # 300 unique HDR float colors at N=100,000 clears the byte-modeled
        # break-even (JSON ~36 KB vs savings/2 = 50 KB).
        colors = _tiled_palette_colors(300, 100_000)
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(colors, SemanticType.COLOR, group, color_mode="hdr")
        assert enc["name"] == "lut_uint16"
        assert enc["lut_mode"] == "row"
        assert len(enc["lut"]) == 300
        assert arr.dtype == np.uint16
        decoded = np.asarray(ArrayDecoder().decode(arr, group))
        np.testing.assert_array_equal(decoded, colors)  # LUT is EXACT

    def test_boundary_256_stays_uint8(self):
        # Same N, K exactly 256: the u8 tier must keep winning (byte-identical
        # legacy behavior).
        colors = _tiled_palette_colors(256, 100_000)
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(colors, SemanticType.COLOR, group, color_mode="hdr")
        assert enc["name"] == "lut_uint8"
        assert arr.dtype == np.uint8

    def test_boundary_65536_uint16_scalar(self):
        # The format ceiling: exactly 65,536 unique scalars. Needs a raised
        # JSON cap (the 512 KiB default exists to protect scene metadata) and
        # E large enough to clear the benefit rule (~2.7M for float32).
        vals = (np.arange(65_536, dtype=np.float64) * 0.25 + 0.5).astype(np.float32)
        data = np.tile(vals, 48)  # E = 3,145,728
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(
            data,
            SemanticType.POSITIVE_SCALAR,
            group,
            encoder=ArrayEncoder(lut_json_max_bytes=8 * 1024 * 1024),
        )
        assert enc["name"] == "lut_uint16"
        assert len(enc["lut"]) == 65_536
        assert arr.dtype == np.uint16

    def test_65537_uniques_falls_through(self):
        vals = np.arange(65_537, dtype=np.float32)
        data = np.tile(vals, 3)
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(
            data,
            SemanticType.POSITIVE_SCALAR,
            group,
            encoder=ArrayEncoder(lut_json_max_bytes=64 * 1024 * 1024),
        )
        assert enc["name"] not in ("lut_uint8", "lut_uint16")

    def test_benefit_rejection_small_n(self):
        # K=257 at N=1,000: the doubled LUT JSON dwarfs any index savings —
        # must fall through to the quantized color path.
        colors = _tiled_palette_colors(257, 1_000)
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(colors, SemanticType.COLOR, group, color_mode="hdr")
        assert enc["name"] == "geolog_perchannel_u16"

    def test_json_cap_rejection(self):
        # Same data as the accepting test, but a tiny metadata cap: rejected.
        colors = _tiled_palette_colors(300, 100_000)
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(
            colors,
            SemanticType.COLOR,
            group,
            encoder=ArrayEncoder(lut_json_max_bytes=1024),
            color_mode="hdr",
        )
        assert enc["name"] == "geolog_perchannel_u16"

    def test_scalar_mode_uint16(self):
        # 300 unique positive scalars at E=50,000; 1-D arrays omit lut_mode
        # by contract (decoders default it).
        vals = np.linspace(0.5, 42.0, 300).astype(np.float32)
        data = np.tile(vals, 167)[:50_000]
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(data, SemanticType.POSITIVE_SCALAR, group)
        assert enc["name"] == "lut_uint16"
        assert "lut_mode" not in enc
        assert arr.dtype == np.uint16
        decoded = np.asarray(ArrayDecoder().decode(arr, group))
        np.testing.assert_array_equal(decoded, data)

    def test_idempotent_reencode(self):
        # decode -> re-encode reproduces identical attrs and indices (LUT is
        # exact, so nothing can drift across save/load cycles).
        colors = _tiled_palette_colors(300, 100_000)
        g1 = zarr.group(store=zarr.MemoryStore())
        arr1, enc1 = self._encode(colors, SemanticType.COLOR, g1, color_mode="hdr")
        decoded = np.asarray(ArrayDecoder().decode(arr1, g1)).astype(np.float32)
        g2 = zarr.group(store=zarr.MemoryStore())
        arr2, enc2 = self._encode(decoded, SemanticType.COLOR, g2, color_mode="hdr")
        assert enc1 == enc2
        np.testing.assert_array_equal(np.asarray(arr1), np.asarray(arr2))

    def test_int64_beyond_2p53_rejected(self):
        # JSON fidelity guard: 64-bit integers beyond 2^53 don't survive the
        # JSON round-trip, so the LUT must refuse them (both tiers).
        data = np.tile(np.array([2**53 + 1, 2**53 + 3, 5, 7], dtype=np.int64), 1000)
        group = zarr.group(store=zarr.MemoryStore())
        arr, enc = self._encode(data, SemanticType.INDEX, group)
        assert enc["name"] not in ("lut_uint8", "lut_uint16")


class TestArrayReferences:
    """Test array reference encoding for deduplication."""

    def test_duplicate_detection(self):
        """Test duplicate arrays are detected and referenced."""
        data1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data2 = data1.copy()  # Identical content

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            # Encode first array
            encoder.encode(data1, group, "array1", SemanticType.POSITIVE_SCALAR)

            # Encode duplicate array
            encoder.encode(data2, group, "array2", SemanticType.POSITIVE_SCALAR)

            # Check array2 is a reference
            arr2 = group["array2"]
            assert arr2.shape == (0,)  # Empty array
            enc = arr2.attrs["encoding"]
            assert enc["name"] == "array_ref"
            assert enc["target"] == "array1"
            assert "hash" in enc

    def test_different_arrays_not_referenced(self):
        """Test different arrays are not referenced."""
        data1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            encoder.encode(data1, group, "array1", SemanticType.POSITIVE_SCALAR)
            encoder.encode(data2, group, "array2", SemanticType.POSITIVE_SCALAR)

            # Check array2 is NOT a reference
            arr2 = group["array2"]
            assert arr2.shape == (3,)
            enc = arr2.attrs["encoding"]
            assert enc["name"] != "array_ref"

    def test_deduplicate_false_forces_materialization(self):
        """deduplicate=False keeps a byte-identical array materialised instead
        of turning it into an array_ref. Used for line vertices/segments,
        whose spatial-index loader cannot resolve refs (a ref would load as
        empty geometry)."""
        seg = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(seg, group, "seg1", SemanticType.INDEX, deduplicate=False)
            encoder.encode(
                seg.copy(), group, "seg2", SemanticType.INDEX, deduplicate=False
            )

            arr2 = group["seg2"]
            assert arr2.attrs["encoding"]["name"] != "array_ref"
            # Materialised: the data is present, not an empty placeholder.
            assert np.array_equal(np.asarray(arr2), seg)

    def test_deduplicate_true_default_still_refs(self):
        """The default (deduplicate=True) still dedups byte-identical arrays —
        the optimization is preserved for ref-resolving consumers (e.g.
        points positions, handled by the points loader)."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "a1", SemanticType.POSITIVE_SCALAR)
            encoder.encode(data.copy(), group, "a2", SemanticType.POSITIVE_SCALAR)
            assert group["a2"].attrs["encoding"]["name"] == "array_ref"


class TestCoordinateEncoding:
    """Test COORDINATE semantic type encoding.

    PRECISION → float32 (exact); AUTO / MEMORY → uint16 per-axis fixed-point
    (``linear_perchannel_u16``), decoded back to float32. Coordinates never use
    uint8 (too coarse) and never float16 (relative precision is a footgun).
    """

    def test_coordinate_precision_mode(self):
        """PRECISION keeps coordinates float32, bit-exact."""
        data = np.random.randn(1000, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            ArrayEncoder().encode(
                data,
                group,
                "test",
                SemanticType.COORDINATE,
                mode=EncodingMode.PRECISION,
            )
            arr = group["test"]
            assert arr.dtype == np.float32
            assert group["test"].attrs["encoding"]["name"] == "float32"
            np.testing.assert_array_equal(ArrayDecoder().decode(arr, group), data)

    def test_coordinate_auto_mode(self):
        """AUTO stores coordinates as uint16 per-axis fixed-point, decoded to float32."""
        data = np.random.randn(1000, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            ArrayEncoder().encode(data, group, "test", SemanticType.COORDINATE)
            arr = group["test"]
            assert arr.dtype == np.uint16
            assert group["test"].attrs["encoding"]["name"] == "linear_perchannel_u16"
            decoded = ArrayDecoder().decode(arr, group)
            assert decoded.dtype == np.float32  # always decodes to float32
            # sub-unit: error <= per-axis extent / 65535
            extent = float((data.max(0) - data.min(0)).max())
            assert np.abs(decoded - data).max() < extent / 65535 * 2

    def test_coordinate_memory_mode(self):
        """MEMORY also uses uint16 (coordinates never use uint8 — 256 levels too coarse)."""
        data = np.random.randn(1000, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            ArrayEncoder().encode(
                data, group, "test", SemanticType.COORDINATE, mode=EncodingMode.MEMORY
            )
            arr = group["test"]
            assert arr.dtype == np.uint16
            assert group["test"].attrs["encoding"]["name"] == "linear_perchannel_u16"

    def test_coordinate_never_float16(self):
        """float16_allowed does NOT apply to coordinates — they use uint16 fixed-point
        (float16's relative precision is a footgun for absolute positions)."""
        data = np.random.randn(1000, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            ArrayEncoder(float16_allowed=True).encode(
                data, group, "test", SemanticType.COORDINATE, mode=EncodingMode.MEMORY
            )
            arr = group["test"]
            assert arr.dtype == np.uint16  # NOT float16
            assert group["test"].attrs["encoding"]["name"] == "linear_perchannel_u16"


class TestPositiveScalarEncoding:
    """Tests for POSITIVE_SCALAR encoding edge cases."""

    def test_log_encoding_all_zeros(self):
        """Log encoding should handle all-zero data without NaNs."""
        data = np.zeros(128, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                data,
                group,
                "amps",
                SemanticType.POSITIVE_SCALAR,
                positive_scalar_encoding="log",
            )

            enc = group["amps"].attrs["encoding"]
            assert enc["name"] == "broadcasted"
            assert enc["n_elements"] == data.shape[0]


class TestColorEncoding:
    """Test COLOR semantic type encoding."""

    def test_color_sdr_auto_mode(self):
        """Test SDR colors quantized to uint8 in AUTO mode."""
        data = np.random.rand(1000, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.COLOR, color_mode="sdr")

            arr = group["test"]
            assert arr.dtype == np.uint8
            enc = arr.attrs["encoding"]
            assert enc["name"] == "rgb_uint8"

    def test_color_hdr_auto_mode(self):
        """HDR colors quantize to geolog_perchannel_u16 in AUTO mode
        (2026-07 HDR-color spike: per-channel true-log dominates linear and
        log1p at every dynamic range; PRECISION keeps float32)."""
        data = np.random.rand(1000, 3).astype(np.float32) * 2.0  # HDR range

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.COLOR, color_mode="hdr")

            arr = group["test"]
            assert arr.dtype == np.uint16
            assert arr.attrs["encoding"]["name"] == "geolog_perchannel_u16"

    def test_color_missing_color_mode_error(self):
        """Test float colors without color_mode raise error."""
        data = np.random.rand(1000, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="color_mode"):
                encoder.encode(data, group, "test", SemanticType.COLOR)

    # [Python-R5 / encoding-MAJOR] Pin the FULL color_mode validation
    # surface — every branch in _validate_input that constrains the
    # (semantic_type, dtype, color_mode) triple. The existing test
    # only covered "float color, color_mode=None". Add the four other
    # rejection paths so a regression that flipped any branch (e.g.,
    # accepting color_mode='Hdr' case-insensitively, or accepting
    # color_mode='hdr' for uint8) would fail loudly.
    def test_color_float_with_invalid_color_mode_string(self):
        """color_mode must be exactly 'sdr' or 'hdr', not 'auto' or any
        other string."""
        data = np.random.rand(50, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            with pytest.raises(ValueError, match="color_mode must be"):
                encoder.encode(
                    data, group, "test", SemanticType.COLOR, color_mode="auto"
                )

    def test_color_uint8_with_hdr_mode_rejected(self):
        """Integer COLOR arrays are SDR storage; color_mode='hdr' is a
        contract violation that must be caught upfront."""
        data = (np.random.rand(50, 3) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            with pytest.raises(ValueError, match="Integer COLOR arrays are SDR"):
                encoder.encode(
                    data, group, "test", SemanticType.COLOR, color_mode="hdr"
                )

    def test_color_negative_value_rejected_before_color_mode_branch(self):
        """Colors must be non-negative regardless of color_mode — the
        non-negative check fires before the color_mode dtype branch
        so a -0.5 float color is caught even with color_mode unset."""
        data = np.full((10, 3), -0.5, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            with pytest.raises(ValueError, match="non-negative"):
                encoder.encode(
                    data, group, "test", SemanticType.COLOR, color_mode="sdr"
                )


class TestBoundedScalarEncoding:
    """Test BOUNDED_SCALAR semantic type encoding."""

    def test_bounded_scalar_auto_mode_narrow_range(self):
        """Test bounded scalar with narrow dynamic range uses uint8."""
        # Create data with narrow dynamic range (< 256:1) by adding offset
        # Range [10, 41] has dynamic range of 4.1:1 - definitely uint8
        data = np.random.rand(1000).astype(np.float32) * 31 + 10

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint8
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint8"
            assert "min" in enc
            assert "max" in enc

    def test_bounded_scalar_auto_mode_wide_range(self):
        """Test bounded scalar with wide dynamic range uses uint16."""
        # Create data with wide dynamic range (> 256:1)
        # Range [0.001, 1.0] has dynamic range of 1000:1 - needs uint16
        data = np.linspace(0.001, 1.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint16
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint16"
            assert "min" in enc
            assert "max" in enc

    def test_bounded_scalar_explicit_bounds(self):
        """Test bounded scalar with explicit bounds."""
        data = np.random.rand(1000).astype(np.float32) * 31

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                data,
                group,
                "test",
                SemanticType.BOUNDED_SCALAR,
                bounds=(0.0, 31.0),
            )

            arr = group["test"]
            enc = arr.attrs["encoding"]
            assert enc["min"] == 0.0
            assert enc["max"] == 31.0


class TestPositiveScalarEncodingDynamicRange:
    """Test POSITIVE_SCALAR semantic type encoding dynamic range selection."""

    def test_positive_scalar_narrow_dynamic_range(self):
        """Test positive scalar with narrow dynamic range uses uint8."""
        # Create data with narrow dynamic range (< 256:1)
        # Range [0.1, 0.5] has dynamic range of 5:1 - definitely uint8
        data = np.random.rand(1000).astype(np.float32) * 0.4 + 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint8
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint8"

    def test_positive_scalar_wide_dynamic_range(self):
        """Test positive scalar with wide dynamic range uses uint16."""
        # Create data with wide dynamic range (> 256:1)
        # Range [0.0001, 0.5] has dynamic range of 5000:1 - needs uint16
        data = np.linspace(0.0001, 0.5, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint16
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint16"

    def test_positive_scalar_log_encoding(self):
        """Test positive scalar with log encoding."""
        data = np.random.rand(1000).astype(np.float32) * 100  # Wide range

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                data,
                group,
                "test",
                SemanticType.POSITIVE_SCALAR,
                positive_scalar_encoding="log",
            )

            arr = group["test"]
            assert arr.dtype == np.uint16  # AUTO log opt-in -> geolog u16
            enc = arr.attrs["encoding"]
            assert enc["name"] == "geolog_scalar_uint16"
            assert "min_log" in enc and "max_log" in enc


class TestIndexEncoding:
    """Test INDEX semantic type encoding."""

    def test_index_selects_smallest_uint(self):
        """Test index selects smallest uint dtype."""
        # Test uint8 range
        data = np.array([0, 100, 255], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.INDEX)

            arr = group["test"]
            assert arr.dtype == np.uint8

        # Test uint16 range
        data = np.array([0, 1000, 60000], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.INDEX)

            arr = group["test"]
            assert arr.dtype == np.uint16


class TestInputValidation:
    """Test input validation and error handling."""

    def test_nan_values_rejected(self):
        """Test NaN values raise error."""
        data = np.array([1.0, np.nan, 3.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="NaN"):
                encoder.encode(data, group, "test", SemanticType.COORDINATE)

    def test_inf_values_rejected(self):
        """Test Inf values raise error."""
        data = np.array([1.0, np.inf, 3.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="Inf"):
                encoder.encode(data, group, "test", SemanticType.COORDINATE)

    def test_negative_color_rejected(self):
        """Test negative color values raise error."""
        data = np.array([[1.0, -0.5, 0.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="non-negative"):
                encoder.encode(
                    data, group, "test", SemanticType.COLOR, color_mode="sdr"
                )

    def test_negative_positive_scalar_rejected(self):
        """Test negative positive scalar values raise error."""
        data = np.array([1.0, -0.5, 2.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="non-negative"):
                encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

    def test_empty_array_passthrough(self):
        """Test empty arrays are passed through."""
        data = np.array([], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.COORDINATE)

            arr = group["test"]
            assert arr.shape == (0,)
            assert arr.attrs["encoding"]["name"] == "none"


class TestCustomMode:
    """Test CUSTOM encoding mode."""

    def test_custom_mode_requires_encoder(self):
        """Test CUSTOM mode without custom_encoder raises error."""
        data = np.random.rand(100).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="custom_encoder"):
                encoder.encode(
                    data,
                    group,
                    "test",
                    SemanticType.POSITIVE_SCALAR,
                    mode=EncodingMode.CUSTOM,
                )

    def test_custom_mode_float16(self):
        """Test CUSTOM mode with float16 encoder."""
        data = np.random.rand(100).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(str(tmpdir), mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                data,
                group,
                "test",
                SemanticType.POSITIVE_SCALAR,
                mode=EncodingMode.CUSTOM,
                custom_encoder="float16",
            )

            arr = group["test"]
            assert arr.dtype == np.float16
