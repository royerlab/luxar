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


def test_float16_sdr_color_quantization_uses_float64_affine_map() -> None:
    data = np.array(
        [[0.00392, 0.00784, 0.011765], [0.06274, 0.1098, 0.1255]],
        dtype=np.float16,
    )
    group = memory_group()
    ArrayEncoder().encode(
        data,
        group,
        "colors",
        SemanticType.COLOR,
        mode=EncodingMode.AUTO,
        color_mode="sdr",
        allow_lut=False,
        deduplicate=False,
    )

    np.testing.assert_array_equal(group["colors"], [[1, 2, 3], [16, 28, 32]])
    assert group["colors"].attrs["encoding"]["name"] == "rgb_uint8"


def test_sdr_color_quantization_is_unbiased_and_half_lsb_bounded() -> None:
    ramp = np.linspace(0.0, 1.0, 10_001, dtype=np.float64)
    data = np.repeat(ramp[:, None], 3, axis=1)
    group = memory_group()
    ArrayEncoder().encode(
        data,
        group,
        "colors",
        SemanticType.COLOR,
        mode=EncodingMode.AUTO,
        color_mode="sdr",
        allow_lut=False,
        deduplicate=False,
    )

    error_lsb = np.asarray(group["colors"], dtype=np.float64) - data * 255.0
    assert abs(float(np.mean(error_lsb))) < 0.001
    assert float(np.max(np.abs(error_lsb))) <= 0.5


def test_sdr_color_quantization_preserves_all_exact_byte_codes() -> None:
    codes = np.arange(256, dtype=np.uint8)
    data = np.repeat((codes.astype(np.float64) / 255.0)[:, None], 3, axis=1)
    group = memory_group()
    ArrayEncoder().encode(
        data,
        group,
        "colors",
        SemanticType.COLOR,
        mode=EncodingMode.AUTO,
        color_mode="sdr",
        allow_lut=False,
        deduplicate=False,
    )

    np.testing.assert_array_equal(np.asarray(group["colors"])[:, 0], codes)


def test_custom_rgb_uint8_keeps_historical_truncation() -> None:
    group = memory_group()
    ArrayEncoder()._encode_custom(
        group,
        "colors",
        np.array([[0.109, 0.109, 0.109]], dtype=np.float32),
        "rgb_uint8",
        None,
    )

    np.testing.assert_array_equal(group["colors"], [[27, 27, 27]])


@pytest.mark.parametrize(
    ("encoder_name", "data"),
    [
        ("rgb_uint16", np.array([0.0, 0.5, 1.0], dtype=np.float16)),
        ("log_scalar_uint16", np.array([0.0, 1.0, 3.0], dtype=np.float16)),
    ],
)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_custom_float16_uint16_quantization_truncates_without_overflow(
    encoder_name: str, data: np.ndarray
) -> None:
    group = memory_group()
    ArrayEncoder()._encode_custom(group, "a", data, encoder_name, None)

    np.testing.assert_array_equal(group["a"], [0, 32_767, 65_535])
    assert group["a"].attrs["encoding"]["name"] == encoder_name
