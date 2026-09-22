"""Couple positive-scalar round-trip slack to the real encoder."""

import numpy as np
import pytest

from luxar._zarr_compat import memory_group
from luxar.encoding import ArrayDecoder, ArrayEncoder, EncodingMode, SemanticType


def _viewer_linear_decode(codes: np.ndarray, metadata: dict) -> np.ndarray:
    levels = (1 << metadata["bits"]) - 1
    lo = np.float32(metadata["min"])
    hi = np.float32(metadata["max"])
    scale = np.float32(np.float32(hi - lo) / np.float32(levels))
    return np.float32(lo + np.float32(np.asarray(codes, dtype=np.float32) * scale))


def _viewer_geolog_decode(codes: np.ndarray, metadata: dict) -> np.ndarray:
    levels = (1 << metadata["bits"]) - 1
    lo = float(np.float32(metadata["min_log"]))
    hi = float(np.float32(metadata["max_log"]))
    scale = (hi - lo) / (levels - 1)
    codes = np.asarray(codes)
    decoded = np.zeros(codes.shape, dtype=np.float32)
    nonzero = codes > 0
    exponent = lo + (codes[nonzero].astype(np.float64) - 1.0) * scale
    decoded[nonzero] = np.asarray(np.exp(exponent), dtype=np.float32)
    return decoded


@pytest.mark.parametrize(
    ("name", "data", "mode", "encoding_type", "expected_encoding"),
    [
        (
            "linear_u8",
            np.linspace(0.1, 5.0, 4001, dtype=np.float32),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint8",
        ),
        (
            "linear_u8_float16_reader_cast_small_span",
            np.linspace(0.1, 5.0, 20_000, dtype=np.float16),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint8",
        ),
        (
            "linear_u8_float16_reader_cast",
            np.linspace(29.921875, 69.75, 28_000).astype(np.float16),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint8",
        ),
        (
            "linear_u8_float16_subnormal_reader_cast",
            np.linspace(2e-6, 1.8e-5, 10_001, dtype=np.float16),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint8",
        ),
        (
            "linear_u16",
            np.geomspace(1e-4, 5.0, 4001, dtype=np.float32),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint16",
        ),
        (
            "linear_u16_float32_normalization",
            np.geomspace(
                2.779823463430753e-12,
                1.3380282659625209e-08,
                3913,
                dtype=np.float32,
            ),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint16",
        ),
        (
            "linear_u16_float64",
            np.geomspace(1e-4, 5.0, 20_001, dtype=np.float64),
            EncodingMode.AUTO,
            "linear",
            "bounded_scalar_uint16",
        ),
        (
            "linear_geolog",
            np.geomspace(1e-8, 5.0, 4001, dtype=np.float32),
            EncodingMode.AUTO,
            "linear",
            "geolog_scalar_uint16",
        ),
        (
            "explicit_log_memory",
            np.geomspace(0.1, 5.0, 4001, dtype=np.float32),
            EncodingMode.MEMORY,
            "log",
            "geolog_scalar_uint8",
        ),
    ],
)
def test_positive_scalar_slack_bounds_real_upward_displacement(
    name: str,
    data: np.ndarray,
    mode: EncodingMode,
    encoding_type: str,
    expected_encoding: str,
) -> None:
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data, mode, positive_scalar_encoding=encoding_type, allow_lut=False
    )
    assert slack is not None and slack > 0.0, name

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=mode,
        positive_scalar_encoding=encoding_type,
        allow_lut=False,
        deduplicate=False,
    )
    assert group["s"].attrs["encoding"]["name"] == expected_encoding
    decoded = ArrayDecoder().decode(group["s"], group)
    upward = decoded.astype(np.float32).astype(np.float64) - data.astype(np.float64)
    assert float(upward.max()) > 0.0, name
    assert float(upward.max()) <= slack, name
    assert slack <= 2 * float(upward.max()), name


def test_positive_scalar_slack_matches_explicit_uint8_tier() -> None:
    data = np.geomspace(1.0, 1000.0, 5000).astype(np.float32)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data,
        EncodingMode.AUTO,
        positive_scalar_bits=8,
        allow_lut=False,
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        positive_scalar_bits=8,
        allow_lut=False,
        deduplicate=False,
    )
    assert group["s"].attrs["encoding"]["name"] == "geolog_scalar_uint8"
    decoded = ArrayDecoder().decode(group["s"], group)
    upward = decoded.astype(np.float64) - data.astype(np.float64)
    assert float(upward.max()) > 1e-3
    assert float(upward.max()) <= slack


def test_positive_scalar_slack_bounds_viewer_float32_decode() -> None:
    data = np.linspace(9.75, 713.0, 20_000).astype(np.float16)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data, EncodingMode.AUTO, allow_lut=False
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        allow_lut=False,
        deduplicate=False,
    )
    encoded = group["s"]
    metadata = encoded.attrs["encoding"]
    decoded = _viewer_linear_decode(np.asarray(encoded[:]), metadata)
    upward = decoded.astype(np.float64) - data.astype(np.float64)
    assert float(upward.max()) > 1.0
    assert float(upward.max()) <= slack


def test_positive_scalar_slack_bounds_staged_viewer_float32_decode() -> None:
    data = np.geomspace(
        56.03468458866481,
        17172.385826620357,
        19826,
        dtype=np.float64,
    )
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data, EncodingMode.AUTO, allow_lut=False
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        allow_lut=False,
        deduplicate=False,
    )
    encoded = group["s"]
    metadata = encoded.attrs["encoding"]
    assert metadata["name"] == "bounded_scalar_uint16"
    decoded = _viewer_linear_decode(np.asarray(encoded[:]), metadata)
    upward = decoded.astype(np.float64) - data
    assert float(upward.max()) > 0.13
    assert float(upward.max()) <= slack
    assert slack <= 2 * float(upward.max())


@pytest.mark.parametrize(
    ("lo", "hi", "levels", "code", "fill", "expected_encoding"),
    [
        (
            0.002945750926925187,
            0.5193778596167954,
            255,
            252,
            500,
            "bounded_scalar_uint8",
        ),
        (
            3.4272668991807e-05,
            1.0437129191726766,
            65535,
            65532,
            2000,
            "bounded_scalar_uint16",
        ),
    ],
)
def test_positive_scalar_slack_bounds_linear_viewer_rounding_corner(
    lo: float,
    hi: float,
    levels: int,
    code: int,
    fill: int,
    expected_encoding: str,
) -> None:
    quantum = (hi - lo) / levels
    value = np.nextafter(lo + (code - 0.5) * quantum, np.inf)
    data = np.array([lo, hi, value, *np.linspace(lo, hi, fill)], dtype=np.float64)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data, EncodingMode.AUTO, allow_lut=False
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        allow_lut=False,
        deduplicate=False,
    )
    encoded = group["s"]
    metadata = encoded.attrs["encoding"]
    assert metadata["name"] == expected_encoding
    decoded = _viewer_linear_decode(np.asarray(encoded[:]), metadata)
    upward = decoded.astype(np.float64) - data
    assert float(upward.max()) > slack - 0.01 * quantum
    assert float(upward.max()) <= slack


@pytest.mark.parametrize(("bits", "minimum_ratio"), [(8, 1.01), (16, 257.0)])
def test_linear_viewer_affine_bound_over_code_space(
    bits: int, minimum_ratio: float
) -> None:
    rng = np.random.default_rng(1923 + bits)
    levels = (1 << bits) - 1
    codes = np.arange(levels + 1, dtype=np.float32)
    eps32 = float(np.finfo(np.float32).eps)

    for _ in range(256):
        hi = 10.0 ** rng.uniform(-12.0, 12.0)
        ratio = 10.0 ** rng.uniform(np.log10(minimum_ratio), np.log10(levels))
        lo = hi / ratio
        metadata = {"bits": bits, "min": lo, "max": hi}
        decoded = _viewer_linear_decode(codes, metadata).astype(np.float64)
        exact = lo + codes.astype(np.float64) * ((hi - lo) / levels)
        allowance = hi * eps32 + 1.5 * (hi - lo) * eps32
        assert float(np.max(decoded - exact)) <= allowance


@pytest.mark.parametrize(("lo", "hi"), [(1e5, 1.05e5), (1000.0, 1000.5)])
def test_positive_scalar_slack_bounds_geolog_with_canonical_anchors(
    lo: float, hi: float
) -> None:
    data = np.geomspace(lo, hi, 2000, dtype=np.float64)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data,
        EncodingMode.AUTO,
        positive_scalar_encoding="log",
        allow_lut=False,
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        positive_scalar_encoding="log",
        allow_lut=False,
        deduplicate=False,
    )
    encoded = group["s"]
    metadata = encoded.attrs["encoding"]
    assert metadata["name"] == "geolog_scalar_uint16"
    decoded = _viewer_geolog_decode(np.asarray(encoded[:]), metadata)
    upward = decoded.astype(np.float64) - data
    assert float(upward.max()) > 0.0
    assert float(upward.max()) <= slack


def test_positive_scalar_slack_bounds_geolog_lower_anchor_rounding() -> None:
    data = np.geomspace(1e-4 / 1.00005, 1e-4, 2000, dtype=np.float64)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data,
        EncodingMode.AUTO,
        positive_scalar_encoding="linear",
        positive_scalar_bits=8,
        allow_lut=False,
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        positive_scalar_encoding="linear",
        positive_scalar_bits=8,
        allow_lut=False,
        deduplicate=False,
    )
    assert group["s"].attrs["encoding"]["name"] == "geolog_scalar_uint8"
    decoded = ArrayDecoder().decode(group["s"], group).astype(np.float64)
    upward = decoded - data
    assert float(upward.max()) > 0.0
    assert float(upward.max()) <= slack


def test_positive_scalar_slack_bounds_degenerate_geolog_canonical_anchor() -> None:
    data = np.array([0.0] * 500 + [123456.0] * 500, dtype=np.float64)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(
        data,
        EncodingMode.AUTO,
        positive_scalar_encoding="log",
        allow_lut=False,
    )
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        positive_scalar_encoding="log",
        allow_lut=False,
        deduplicate=False,
    )
    encoded = group["s"]
    metadata = encoded.attrs["encoding"]
    assert metadata["name"] == "geolog_scalar_uint16"
    decoded = _viewer_geolog_decode(np.asarray(encoded[:]), metadata)
    upward = decoded.astype(np.float64) - data
    assert float(upward.max()) > 0.0
    assert float(upward.max()) <= slack


def test_positive_scalar_slack_covers_the_authored_dtype_cast() -> None:
    data = np.linspace(31.111, 40.0, 20_000).astype(np.float16)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(data, EncodingMode.AUTO)
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        deduplicate=False,
    )
    assert group["s"].attrs["encoding"]["name"] == "bounded_scalar_uint8"
    decoded = ArrayDecoder().decode(group["s"], group)
    upward = decoded.astype(np.float64) - data.astype(np.float64)
    assert float(upward.max()) > 0.0
    assert float(upward.max()) <= slack


@pytest.mark.parametrize("mode", list(EncodingMode))
def test_positive_scalar_slack_reports_exact_encoder_exits(mode: EncodingMode) -> None:
    encoder = ArrayEncoder()
    constant = np.full(4001, 2.5, dtype=np.float32)
    if mode == EncodingMode.CUSTOM:
        with pytest.raises(
            ValueError,
            match=(
                "CUSTOM POSITIVE_SCALAR array: an arbitrary custom_encoder "
                "has no round-trip displacement model"
            ),
        ):
            encoder.positive_scalar_round_trip_slack(constant, mode)
        return
    assert encoder.positive_scalar_round_trip_slack(constant, mode) is None

    if mode == EncodingMode.PRECISION:
        continuous = np.linspace(0.1, 5.0, 4001, dtype=np.float32)
        assert encoder.positive_scalar_round_trip_slack(continuous, mode) is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_positive_scalar_broadcast_slack_bounds_viewer_cast(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    data = np.full(4001, 0.1, dtype=dtype)
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(data, EncodingMode.AUTO)

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        deduplicate=False,
    )
    encoded = group["s"]
    assert encoded.attrs["encoding"]["name"] == "broadcasted"
    authored = float(data[0])
    upward = float(np.float32(np.asarray(encoded[:])[0])) - authored

    if dtype == np.float32:
        assert upward == 0.0
        assert slack is None
    else:
        assert upward > 0.0
        assert slack is not None and upward <= slack


def test_positive_scalar_broadcast_slack_adds_tolerance_displacement() -> None:
    data = np.full(4001, 0.1, dtype=np.float64)
    data[-1] = 0.09
    encoder = ArrayEncoder(broadcast_atol=0.011)
    slack = encoder.positive_scalar_round_trip_slack(data, EncodingMode.AUTO)

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        deduplicate=False,
    )
    encoded = group["s"]
    assert encoded.attrs["encoding"]["name"] == "broadcasted"
    upward = float(np.float32(np.asarray(encoded[:])[0])) - float(data.min())
    viewer_only = float(np.float32(data[0])) - float(data[0])

    assert upward > viewer_only > 0.0
    assert slack is not None and upward <= slack


@pytest.mark.parametrize(
    "data",
    [
        np.linspace(0.1, 5.0, 4001, dtype=np.float64),
        np.linspace(1e-46, 1e-45, 4001, dtype=np.float64),
    ],
)
def test_positive_scalar_precision_slack_covers_the_float32_cast(
    data: np.ndarray,
) -> None:
    encoder = ArrayEncoder()
    slack = encoder.positive_scalar_round_trip_slack(data, EncodingMode.PRECISION)
    assert slack is not None

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.PRECISION,
        deduplicate=False,
    )
    decoded = ArrayDecoder().decode(group["s"], group).astype(np.float32)
    upward = decoded.astype(np.float64) - data
    assert float(upward.max()) > 0.0
    assert float(upward.max()) <= slack


@pytest.mark.parametrize(
    ("mode", "data", "allow_lut"),
    [
        (
            EncodingMode.AUTO,
            np.linspace(1e38, 1e39, 20_000, dtype=np.float64),
            False,
        ),
        (
            EncodingMode.MEMORY,
            np.linspace(1e38, 1e39, 20_000, dtype=np.float64),
            False,
        ),
        (
            EncodingMode.PRECISION,
            np.linspace(1e38, 1e39, 20_000, dtype=np.float64),
            False,
        ),
        (EncodingMode.MEMORY, np.geomspace(1e-300, 1.7e308, 4001), False),
        (
            EncodingMode.MEMORY,
            np.array([1e-300, np.finfo(np.float64).max], dtype=np.float64),
            False,
        ),
        (EncodingMode.AUTO, np.full(4001, 3.5e38, dtype=np.float64), True),
        (
            EncodingMode.AUTO,
            np.resize(np.array([1.0, 2.0, 3.5e38], dtype=np.float64), 4001),
            True,
        ),
    ],
)
def test_positive_scalar_slack_is_finite_above_the_float32_range(
    mode: EncodingMode,
    data: np.ndarray,
    allow_lut: bool,
) -> None:
    with np.errstate(over="raise", invalid="raise"):
        slack = ArrayEncoder().positive_scalar_round_trip_slack(
            data, mode, allow_lut=allow_lut
        )

    assert slack is not None
    assert np.isfinite(slack)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_positive_scalar_slack_honours_the_lut_gate(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    encoder = ArrayEncoder()
    palette = np.linspace(0.1, 5.0, 200, dtype=dtype)
    data = np.resize(palette, 4001)
    assert encoder.encodes_as_lut(data, SemanticType.POSITIVE_SCALAR)
    slack = encoder.positive_scalar_round_trip_slack(data, EncodingMode.AUTO)

    group = memory_group()
    encoder.encode(
        data,
        group,
        "s",
        SemanticType.POSITIVE_SCALAR,
        mode=EncodingMode.AUTO,
        deduplicate=False,
    )
    metadata = group["s"].attrs["encoding"]
    assert metadata["name"] == "lut_uint8"
    lut = np.asarray(metadata["lut"], dtype=np.float64)
    upward = np.float32(lut).astype(np.float64) - lut

    if dtype == np.float32:
        assert float(upward.max()) == 0.0
        assert slack is None
    else:
        assert float(upward.max()) > 0.0
        assert slack is not None and float(upward.max()) <= slack
    assert (
        encoder.positive_scalar_round_trip_slack(
            data, EncodingMode.AUTO, allow_lut=False
        )
        is not None
    )


@pytest.mark.parametrize(
    "data",
    [
        np.full(4001, 1.0, dtype=np.float64),
        np.full(4001, 3, dtype=np.int64),
        np.zeros(4001, dtype=np.float64),
        np.full(
            4001,
            np.nextafter(float(np.float32(0.1)), np.inf),
            dtype=np.float64,
        ),
        np.resize(np.arange(200, dtype=np.int64), 4001),
    ],
)
def test_positive_scalar_exact_float32_values_need_no_cast_slack(
    data: np.ndarray,
) -> None:
    assert (
        ArrayEncoder().positive_scalar_round_trip_slack(data, EncodingMode.AUTO) is None
    )


def test_positive_scalar_slack_skips_full_lut_probe_when_prefix_disproves_it() -> None:
    calls = []

    class CountingEncoder(ArrayEncoder):
        def _lut_plan(self, data, semantic_type):
            calls.append(semantic_type)
            return super()._lut_plan(data, semantic_type)

    data = np.linspace(0.1, 5.0, 10_000, dtype=np.float32)
    slack = CountingEncoder().positive_scalar_round_trip_slack(data, EncodingMode.AUTO)
    assert slack is not None
    assert calls == []
