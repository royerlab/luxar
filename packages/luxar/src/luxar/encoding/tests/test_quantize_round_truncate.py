"""Lock A4.0b's rounding invariant: the lifted quantizer rounds by default but
the CUSTOM encoder truncates.

The shared ``_quantize_normalized_clip`` helper must round (``round_values=True``)
for the semantic-type encoders and truncate toward zero (``round_values=False``)
for the CUSTOM encoder — that preserves the custom encoder's historical
byte-for-byte behaviour. Round-trip suites use tolerances that would mask a
1-LSB round↔truncate flip, so this pins it directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar._zarr_compat import memory_group
from luxar.encoding import ArrayDecoder, ArrayEncoder, EncodingMode, SemanticType


def test_quantize_normalized_clip_rounds_vs_truncates() -> None:
    # normalized 0.109 * 100 levels = 10.9 → round = 11, truncate = 10.
    data = np.array([0.109], dtype=np.float64)
    rounded = ArrayEncoder._quantize_normalized_clip(
        data, 0.0, 1.0, 100, np.dtype(np.uint8), round_values=True
    )
    truncated = ArrayEncoder._quantize_normalized_clip(
        data, 0.0, 1.0, 100, np.dtype(np.uint8), round_values=False
    )
    assert int(rounded[0]) == 11
    assert int(truncated[0]) == 10


def test_float16_uint16_quantization_rounds_vs_truncates_without_overflow() -> None:
    data = np.array([0.0, 0.5, 1.0], dtype=np.float16)
    rounded = ArrayEncoder._quantize_normalized_clip(
        data, 0.0, 1.0, 65_535, np.dtype(np.uint16), round_values=True
    )
    truncated = ArrayEncoder._quantize_normalized_clip(
        data, 0.0, 1.0, 65_535, np.dtype(np.uint16), round_values=False
    )
    np.testing.assert_array_equal(rounded, [0, 32_768, 65_535])
    np.testing.assert_array_equal(truncated, [0, 32_767, 65_535])


@pytest.mark.parametrize(
    ("semantic_type", "data", "bounds", "expected_encoding", "levels"),
    [
        (
            SemanticType.POSITIVE_SCALAR,
            np.linspace(0.1, 5.0, 4001, dtype=np.float16),
            None,
            "bounded_scalar_uint8",
            np.iinfo(np.uint8).max,
        ),
        (
            SemanticType.POSITIVE_SCALAR,
            np.geomspace(1e-3, 5.0, 4001, dtype=np.float16),
            None,
            "bounded_scalar_uint16",
            np.iinfo(np.uint16).max,
        ),
        (
            SemanticType.BOUNDED_SCALAR,
            np.linspace(0.0, 1.0, 4001, dtype=np.float16),
            (0.0, 1.0),
            "bounded_scalar_uint16",
            np.iinfo(np.uint16).max,
        ),
    ],
)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_float16_scalar_quantization_preserves_dynamic_range(
    semantic_type: SemanticType,
    data: np.ndarray,
    bounds: tuple[float, float] | None,
    expected_encoding: str,
    levels: int,
) -> None:
    group = memory_group()
    ArrayEncoder().encode(
        data,
        group,
        "a",
        semantic_type,
        mode=EncodingMode.AUTO,
        bounds=bounds,
        allow_lut=False,
        deduplicate=False,
    )

    array = group["a"]
    assert array.attrs["encoding"]["name"] == expected_encoding

    decoded = np.asarray(ArrayDecoder().decode(array, group)).astype(np.float64)
    authored = data.astype(np.float64)
    half_quantum = float(authored.max() - authored.min()) / (2 * levels)
    # ArrayDecoder casts back to float16 after the affine reconstruction, so the
    # fixture-dependent bound is half a code-grid quantum plus half a float16 ULP.
    reader_half_ulp = np.abs(np.spacing(data).astype(np.float64)) / 2
    np.testing.assert_array_less(
        np.abs(decoded - authored), half_quantum + reader_half_ulp
    )


def test_custom_bounded_scalar_truncates() -> None:
    # 0.109 over bounds (0, 1) at 8 bits → 0.109 * 255 = 27.795.
    # The CUSTOM encoder must truncate (27); a rounding regression would store 28.
    # Exercise _encode_custom directly — going through encode() would hit the
    # broadcast/LUT priority ladder (a low-cardinality array never reaches the
    # dtype encoders), which is not what this test is pinning.
    g = memory_group()
    ArrayEncoder()._encode_custom(
        g,
        "a",
        np.array([0.109, 0.109], dtype=np.float32),
        "bounded_scalar_uint8",
        (0.0, 1.0),
    )
    stored = np.asarray(g["a"])
    assert int(stored[0]) == 27, (
        f"CUSTOM bounded_scalar must truncate 27.795 → 27, got {int(stored[0])}"
    )
    assert g["a"].attrs["encoding"]["name"] == "bounded_scalar_uint8"
