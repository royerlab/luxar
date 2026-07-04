"""Full encode-decode roundtrip tests for ArrayEncoder and ArrayDecoder.

Verifies that data survives a complete encode -> zarr write -> decode cycle
with appropriate numerical tolerances for each semantic type.
"""

import numpy as np
import zarr

from luxar.encoding import ArrayDecoder, ArrayEncoder, SemanticType


class TestFullEncodingRoundtrip:
    """Roundtrip tests: encode via ArrayEncoder, write to zarr, decode via ArrayDecoder."""

    def _roundtrip(
        self,
        tmp_path,
        data: np.ndarray,
        semantic_type: SemanticType,
        name: str = "test",
        **encode_kwargs,
    ) -> np.ndarray:
        """Helper: encode data to zarr, then decode it back."""
        group = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
        encoder = ArrayEncoder()
        decoder = ArrayDecoder()

        encoder.encode(data, group, name, semantic_type, **encode_kwargs)
        decoded = decoder.decode(group[name], group)
        return decoded

    def test_coordinate_roundtrip(self, tmp_path) -> None:
        """Positions survive roundtrip within uint16 fixed-point tolerance under the
        default AUTO mode (decoded back to float32). PRECISION is bit-exact — see
        test_encoder.TestCoordinateEncoding.test_coordinate_precision_mode."""
        np.random.seed(0)
        data = np.random.randn(500, 3).astype(np.float32) * 100

        decoded = self._roundtrip(tmp_path, data, SemanticType.COORDINATE)

        assert decoded.shape == data.shape
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(
            decoded, data, atol=float(np.ptp(data, axis=0).max()) / 65535 * 2
        )

    def test_sdr_color_roundtrip(self, tmp_path) -> None:
        """SDR colors [0,1] float32 quantized to uint8 should be within 2/255."""
        np.random.seed(1)
        data = np.random.rand(200, 3).astype(np.float32)

        decoded = self._roundtrip(tmp_path, data, SemanticType.COLOR, color_mode="sdr")

        assert decoded.shape == data.shape
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, data, atol=2.0 / 255)

    def test_hdr_color_roundtrip(self, tmp_path) -> None:
        """HDR colors [0,10] float32 should be preserved with rtol=1e-2."""
        np.random.seed(2)
        data = np.random.rand(200, 3).astype(np.float32) * 10.0

        decoded = self._roundtrip(tmp_path, data, SemanticType.COLOR, color_mode="hdr")

        assert decoded.shape == data.shape
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, data, rtol=1e-2)

    def test_positive_scalar_roundtrip(self, tmp_path) -> None:
        """Radii in [0,0.5] should survive roundtrip within quantization tolerance."""
        np.random.seed(3)
        data = np.random.rand(300).astype(np.float32) * 0.5

        decoded = self._roundtrip(tmp_path, data, SemanticType.POSITIVE_SCALAR)

        assert decoded.shape == data.shape
        assert decoded.dtype == np.float32
        # Quantized to uint8 or uint16 depending on dynamic range;
        # tolerance is range / 255 for uint8 worst case
        max_quant_error = 0.5 / 255 + 1e-6
        np.testing.assert_allclose(decoded, data, atol=max_quant_error)

    def test_bounded_scalar_roundtrip(self, tmp_path) -> None:
        """Sharpness in [0,10] should survive roundtrip within quantization tolerance."""
        np.random.seed(4)
        data = np.random.rand(300).astype(np.float32) * 10.0

        decoded = self._roundtrip(
            tmp_path,
            data,
            SemanticType.BOUNDED_SCALAR,
            bounds=(0.0, 10.0),
        )

        assert decoded.shape == data.shape
        assert decoded.dtype == np.float32
        # Quantized to uint8: tolerance is range / 255
        max_quant_error = 10.0 / 255 + 1e-6
        np.testing.assert_allclose(decoded, data, atol=max_quant_error)

    def test_index_roundtrip(self, tmp_path) -> None:
        """Uint32 indices should survive roundtrip exactly."""
        np.random.seed(5)
        data = np.random.randint(0, 100_000, size=500, dtype=np.uint32)

        decoded = self._roundtrip(tmp_path, data, SemanticType.INDEX)

        assert decoded.shape == data.shape
        # Encoder may downcast to smallest uint that fits (uint32 for max ~100k)
        np.testing.assert_array_equal(decoded.astype(np.uint32), data)

    def test_broadcasted_roundtrip(self, tmp_path) -> None:
        """Uniform value should be exactly preserved after broadcast decode."""
        data = np.full(1000, 3.14, dtype=np.float32)

        decoded = self._roundtrip(tmp_path, data, SemanticType.POSITIVE_SCALAR)

        assert decoded.shape == (1000,)
        np.testing.assert_array_equal(decoded, data)

    def test_lut_roundtrip(self, tmp_path) -> None:
        """Array with few unique values should use LUT encoding and preserve them."""
        np.random.seed(6)
        unique_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        data = np.random.choice(unique_vals, size=1000).astype(np.float32)

        group = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
        encoder = ArrayEncoder()
        decoder = ArrayDecoder()

        encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

        # Verify LUT encoding was actually used
        enc = group["test"].attrs["encoding"]
        assert enc["name"] == "lut_uint8", (
            f"Expected LUT encoding but got {enc['name']}"
        )

        decoded = decoder.decode(group["test"], group)

        assert decoded.shape == data.shape
        assert decoded.dtype == np.float32

    # [Python-R1/encoding-MAJOR] Empty-array roundtrip. encoder.py line 147
    # short-circuits empty inputs via _write_passthrough; the round-trip
    # path was never test-pinned. A regression that mangled the shape
    # (e.g., flattened the (0, 3) into a 1-D zero-length) would slip
    # past every non-empty test in this file.
    def test_empty_1d_coordinate_roundtrip(self, tmp_path) -> None:
        data = np.zeros((0,), dtype=np.float32)
        decoded = self._roundtrip(tmp_path, data, SemanticType.POSITIVE_SCALAR)
        assert decoded.shape == (0,)
        assert decoded.dtype == np.float32
        assert decoded.size == 0

    def test_empty_2d_coordinate_roundtrip(self, tmp_path) -> None:
        # Shape (0, 3) — the canonical "no points but the per-point
        # dimensionality is preserved" form. A regression that flattened
        # to (0,) would fail this shape assertion.
        data = np.zeros((0, 3), dtype=np.float32)
        decoded = self._roundtrip(tmp_path, data, SemanticType.COORDINATE)
        assert decoded.shape == (0, 3)
        assert decoded.dtype == np.float32

    def test_empty_color_sdr_roundtrip(self, tmp_path) -> None:
        data = np.zeros((0, 3), dtype=np.float32)
        decoded = self._roundtrip(tmp_path, data, SemanticType.COLOR, color_mode="sdr")
        assert decoded.shape == (0, 3)
        np.testing.assert_array_equal(decoded, data)
